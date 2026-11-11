"""Limit order book with price-time priority.

Design notes
------------
* Prices are integer ticks and quantities are integers. No floats touch the
  matching path, so there is no rounding drift and results are reproducible.
* Each side keeps a dict of price -> Level (a FIFO deque plus a running total)
  and a sorted list of price keys. Keys are stored so that the *best* price is
  always the last element, which makes best-price lookup and removal O(1).
  Asks are stored negated to get that property from the same code path.
* Cancels are lazy at the queue level: the order is flagged dead and the level
  total is decremented immediately, but the deque entry is only dropped when
  matching reaches it. Cancel is O(1) amortized instead of O(queue length).
* Every command is appended to a journal. Replaying the journal into a fresh
  book yields an identical book (see ``replay``), which is how the tests check
  determinism and how you would implement crash recovery.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple


class Side(Enum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class TIF(Enum):
    GTC = "GTC"  # rest on the book until filled or cancelled
    IOC = "IOC"  # fill what you can immediately, drop the rest
    FOK = "FOK"  # fill completely or not at all


class Status(Enum):
    RESTING = "RESTING"          # nothing filled, fully on the book
    PARTIAL = "PARTIAL"          # partly filled, remainder on the book
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"      # IOC/market/FOK remainder dropped
    REJECTED = "REJECTED"        # post-only would cross, FOK not satisfiable, bad input


@dataclass(frozen=True)
class Trade:
    seq: int
    price: int
    qty: int
    taker_side: Side
    maker_id: int
    taker_id: int
    maker_owner: int
    taker_owner: int


@dataclass
class Result:
    order_id: int
    status: Status
    filled: int
    remaining: int
    trades: List[Trade] = field(default_factory=list)
    self_trade_cancels: List[int] = field(default_factory=list)
    reason: str = ""


class _Order:
    __slots__ = ("id", "owner", "side", "price", "qty", "seq", "live")

    def __init__(self, oid: int, owner: int, side: Side, price: int, qty: int, seq: int):
        self.id = oid
        self.owner = owner
        self.side = side
        self.price = price
        self.qty = qty
        self.seq = seq
        self.live = True


class _Level:
    __slots__ = ("price", "queue", "total")

    def __init__(self, price: int):
        self.price = price
        self.queue: Deque[_Order] = deque()
        self.total = 0


def _key(side: Side, price: int) -> int:
    return price if side is Side.BUY else -price


class OrderBook:
    def __init__(self) -> None:
        self._levels: Dict[Side, Dict[int, _Level]] = {Side.BUY: {}, Side.SELL: {}}
        self._keys: Dict[Side, List[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: Dict[int, _Order] = {}
        self._seq = 0
        self._next_id = 1
        self._trade_seq = 0
        self.listeners: List[Callable[[Trade], None]] = []
        self.journal: List[tuple] = []

    # ------------------------------------------------------------------ queries
    def best_bid(self) -> Optional[int]:
        k = self._keys[Side.BUY]
        return k[-1] if k else None

    def best_ask(self) -> Optional[int]:
        k = self._keys[Side.SELL]
        return -k[-1] if k else None

    def mid(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else (b + a) / 2

    def spread(self) -> Optional[int]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else a - b

    def depth(self, side: Side, n: int = 5) -> List[Tuple[int, int]]:
        keys = self._keys[side]
        out = []
        for k in reversed(keys[-n:]):
            lvl = self._levels[side][k if side is Side.BUY else -k]
            out.append((lvl.price, lvl.total))
        return out

    def get_order(self, order_id: int) -> Optional[Tuple[Side, int, int, int]]:
        o = self._orders.get(order_id)
        return None if o is None else (o.side, o.price, o.qty, o.owner)

    def __len__(self) -> int:
        return len(self._orders)

    def snapshot(self) -> tuple:
        """Canonical, comparable view of the full resting state."""
        out = []
        for side in (Side.BUY, Side.SELL):
            for k in reversed(self._keys[side]):
                lvl = self._levels[side][k if side is Side.BUY else -k]
                live = tuple((o.id, o.qty) for o in lvl.queue if o.live)
                out.append((side.value, lvl.price, live))
        return tuple(out)

    # ----------------------------------------------------------------- commands
    def submit(
        self,
        side: Side,
        qty: int,
        price: Optional[int] = None,
        tif: TIF = TIF.GTC,
        post_only: bool = False,
        owner: int = 0,
        order_id: Optional[int] = None,
    ) -> Result:
        """Submit an order. ``price=None`` means a market order.

        ``owner=0`` means anonymous. A non-zero owner enables self-trade
        prevention: if the incoming order would trade against a resting order
        of the same owner, the *resting* order is cancelled instead.
        """
        if order_id is None:
            order_id = self._next_id
            self._next_id += 1
        self.journal.append(("S", side.value, qty, price, tif.value, post_only, owner, order_id))
        return self._submit(side, qty, price, tif, post_only, owner, order_id)

    def _submit(self, side, qty, price, tif, post_only, owner, order_id) -> Result:
        if qty <= 0 or (price is not None and price <= 0):
            return Result(order_id, Status.REJECTED, 0, 0, reason="bad qty/price")
        if price is None and post_only:
            return Result(order_id, Status.REJECTED, 0, qty, reason="market order cannot be post-only")
        if price is None:
            tif = TIF.IOC  # market orders never rest

        opp = side.opposite
        if post_only and self._would_cross(side, price):
            return Result(order_id, Status.REJECTED, 0, qty, reason="post-only would cross")
        if tif is TIF.FOK and self._available(side, price, owner, qty) < qty:
            return Result(order_id, Status.REJECTED, 0, qty, reason="insufficient liquidity for FOK")

        res = Result(order_id, Status.RESTING, 0, qty)
        keys, levels = self._keys[opp], self._levels[opp]
        remaining = qty

        while remaining > 0 and keys:
            best_price = -keys[-1] if opp is Side.SELL else keys[-1]
            if not self._crosses(side, price, best_price):
                break
            level = levels[best_price]
            while remaining > 0 and level.total > 0:
                maker = level.queue[0]
                if not maker.live:
                    level.queue.popleft()
                    continue
                if owner != 0 and maker.owner == owner:
                    # self-trade prevention: cancel the resting order
                    maker.live = False
                    level.total -= maker.qty
                    del self._orders[maker.id]
                    level.queue.popleft()
                    res.self_trade_cancels.append(maker.id)
                    continue
                fill = min(remaining, maker.qty)
                maker.qty -= fill
                level.total -= fill
                remaining -= fill
                self._trade_seq += 1
                t = Trade(self._trade_seq, best_price, fill, side, maker.id, order_id,
                          maker.owner, owner)
                res.trades.append(t)
                for fn in self.listeners:
                    fn(t)
                if maker.qty == 0:
                    maker.live = False
                    del self._orders[maker.id]
                    level.queue.popleft()
            if level.total == 0:
                del levels[best_price]
                keys.pop()

        res.filled = qty - remaining
        res.remaining = remaining
        if remaining == 0:
            res.status = Status.FILLED
        elif tif is TIF.GTC:
            self._rest(side, price, remaining, owner, order_id)
            res.status = Status.PARTIAL if res.filled else Status.RESTING
        else:
            res.status = Status.CANCELLED  # IOC / market remainder is dropped
        return res

    def cancel(self, order_id: int) -> bool:
        self.journal.append(("C", order_id))
        return self._cancel(order_id)

    def _cancel(self, order_id: int) -> bool:
        o = self._orders.pop(order_id, None)
        if o is None:
            return False
        o.live = False
        levels = self._levels[o.side]
        level = levels[o.price]
        level.total -= o.qty
        if level.total == 0:
            del levels[o.price]
            keys = self._keys[o.side]
            del keys[bisect_left(keys, _key(o.side, o.price))]
        return True

    def amend(self, order_id: int, qty: Optional[int] = None,
              price: Optional[int] = None) -> Optional[Result]:
        """Reduce-in-place keeps queue priority. Anything else (price change or
        size increase) is cancel/replace and goes to the back of the queue.
        Returns the Result of the replacement, or None if it was in place."""
        self.journal.append(("A", order_id, qty, price))
        return self._amend(order_id, qty, price)

    def _amend(self, order_id, qty, price) -> Optional[Result]:
        o = self._orders.get(order_id)
        if o is None:
            return Result(order_id, Status.REJECTED, 0, 0, reason="unknown order")
        new_qty = o.qty if qty is None else qty
        new_price = o.price if price is None else price
        if new_qty <= 0:
            self._cancel(order_id)
            return Result(order_id, Status.CANCELLED, 0, 0)
        if new_price == o.price and new_qty <= o.qty:
            self._levels[o.side][o.price].total -= o.qty - new_qty
            o.qty = new_qty
            return None
        side, owner = o.side, o.owner
        self._cancel(order_id)
        return self._submit(side, new_qty, new_price, TIF.GTC, False, owner, order_id)

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _crosses(side: Side, limit: Optional[int], level_price: int) -> bool:
        if limit is None:
            return True
        return level_price <= limit if side is Side.BUY else level_price >= limit

    def _would_cross(self, side: Side, price: int) -> bool:
        best = self.best_ask() if side is Side.BUY else self.best_bid()
        return best is not None and self._crosses(side, price, best)

    def _available(self, side: Side, price: Optional[int], owner: int, need: int) -> int:
        """Liquidity an incoming order could actually take (excludes own orders)."""
        opp = side.opposite
        total = 0
        for k in reversed(self._keys[opp]):
            lp = -k if opp is Side.SELL else k
            if not self._crosses(side, price, lp):
                break
            for o in self._levels[opp][lp].queue:
                if o.live and not (owner != 0 and o.owner == owner):
                    total += o.qty
                    if total >= need:
                        return total
        return total

    def _rest(self, side: Side, price: int, qty: int, owner: int, order_id: int) -> None:
        self._seq += 1
        o = _Order(order_id, owner, side, price, qty, self._seq)
        levels = self._levels[side]
        level = levels.get(price)
        if level is None:
            level = levels[price] = _Level(price)
            insort(self._keys[side], _key(side, price))
        level.queue.append(o)
        level.total += qty
        self._orders[order_id] = o

    # --------------------------------------------------------------- invariants
    def check_invariants(self) -> None:
        for side in (Side.BUY, Side.SELL):
            keys = self._keys[side]
            assert keys == sorted(keys), "keys not sorted"
            assert len(keys) == len(set(keys)), "duplicate keys"
            assert len(keys) == len(self._levels[side]), "keys/levels mismatch"
            for k in keys:
                p = k if side is Side.BUY else -k
                lvl = self._levels[side][p]
                live = [o for o in lvl.queue if o.live]
                assert lvl.total == sum(o.qty for o in live) > 0, "level total wrong"
                assert all(o.price == p and o.side is side and o.qty > 0 for o in live)
        b, a = self.best_bid(), self.best_ask()
        assert b is None or a is None or b < a, "book is crossed"
        live_ids = {o.id for s in (Side.BUY, Side.SELL)
                    for lvl in self._levels[s].values() for o in lvl.queue if o.live}
        assert live_ids == set(self._orders), "order index out of sync"


def replay(journal: List[tuple]) -> OrderBook:
    """Rebuild a book from a journal. Deterministic by construction."""
    book = OrderBook()
    for cmd in journal:
        if cmd[0] == "S":
            _, s, qty, price, tif, post_only, owner, oid = cmd
            book._next_id = max(book._next_id, oid + 1)
            book.submit(Side(s), qty, price, TIF(tif), post_only, owner, oid)
        elif cmd[0] == "C":
            book.cancel(cmd[1])
        else:
            book.amend(cmd[1], cmd[2], cmd[3])
    return book

import random

import pytest

from orderbook import OrderBook, Side, Status, TIF, replay

B, S = Side.BUY, Side.SELL


# ------------------------------------------------------------ unit behaviour
def test_price_time_priority():
    ob = OrderBook()
    a = ob.submit(S, 5, 101).order_id
    b = ob.submit(S, 5, 101).order_id
    c = ob.submit(S, 5, 100).order_id
    r = ob.submit(B, 12, 101)
    # best price first (c at 100), then FIFO at 101 (a before b)
    assert [(t.maker_id, t.price, t.qty) for t in r.trades] == [(c, 100, 5), (a, 101, 5), (b, 101, 2)]
    assert r.status is Status.FILLED
    assert ob.get_order(b)[2] == 3
    ob.check_invariants()


def test_trade_executes_at_maker_price():
    ob = OrderBook()
    ob.submit(S, 10, 100)
    r = ob.submit(B, 10, 105)
    assert r.trades[0].price == 100


def test_partial_fill_rests_remainder():
    ob = OrderBook()
    ob.submit(S, 4, 100)
    r = ob.submit(B, 10, 100)
    assert r.status is Status.PARTIAL and r.filled == 4 and r.remaining == 6
    assert ob.best_bid() == 100 and ob.depth(B)[0] == (100, 6)


def test_ioc_drops_remainder():
    ob = OrderBook()
    ob.submit(S, 4, 100)
    r = ob.submit(B, 10, 100, tif=TIF.IOC)
    assert r.status is Status.CANCELLED and r.filled == 4
    assert ob.best_bid() is None


def test_market_order_sweeps_and_never_rests():
    ob = OrderBook()
    ob.submit(S, 3, 100)
    ob.submit(S, 3, 102)
    r = ob.submit(B, 10, None)
    assert r.filled == 6 and ob.best_bid() is None and ob.best_ask() is None


def test_fok_all_or_nothing_and_leaves_book_untouched():
    ob = OrderBook()
    ob.submit(S, 5, 100)
    before = ob.snapshot()
    r = ob.submit(B, 6, 100, tif=TIF.FOK)
    assert r.status is Status.REJECTED and ob.snapshot() == before
    r = ob.submit(B, 5, 100, tif=TIF.FOK)
    assert r.status is Status.FILLED


def test_post_only_rejected_when_crossing():
    ob = OrderBook()
    ob.submit(S, 5, 100)
    assert ob.submit(B, 5, 100, post_only=True).status is Status.REJECTED
    assert ob.submit(B, 5, 99, post_only=True).status is Status.RESTING


def test_self_trade_prevention_cancels_resting_order():
    ob = OrderBook()
    mine = ob.submit(S, 5, 100, owner=7).order_id
    other = ob.submit(S, 5, 100, owner=8).order_id
    r = ob.submit(B, 5, 100, owner=7)
    assert r.self_trade_cancels == [mine]
    assert [t.maker_id for t in r.trades] == [other]
    ob.check_invariants()


def test_fok_ignores_own_liquidity():
    ob = OrderBook()
    ob.submit(S, 5, 100, owner=7)
    assert ob.submit(B, 5, 100, tif=TIF.FOK, owner=7).status is Status.REJECTED


def test_cancel_removes_level_and_is_idempotent():
    ob = OrderBook()
    oid = ob.submit(B, 5, 99).order_id
    assert ob.cancel(oid) is True
    assert ob.cancel(oid) is False
    assert ob.best_bid() is None
    ob.check_invariants()


def test_cancelled_order_in_middle_of_queue_is_skipped():
    ob = OrderBook()
    a = ob.submit(S, 5, 100).order_id
    b = ob.submit(S, 5, 100).order_id
    c = ob.submit(S, 5, 100).order_id
    ob.cancel(b)
    r = ob.submit(B, 10, 100)
    assert [t.maker_id for t in r.trades] == [a, c]


def test_amend_down_keeps_priority_up_loses_it():
    ob = OrderBook()
    a = ob.submit(S, 10, 100).order_id
    b = ob.submit(S, 10, 100).order_id
    ob.amend(a, qty=4)  # in place
    assert ob.submit(B, 1, 100).trades[0].maker_id == a
    ob.amend(a, qty=9)  # size up -> back of queue
    assert ob.submit(B, 1, 100).trades[0].maker_id == b


def test_amend_price_into_cross_trades():
    ob = OrderBook()
    ob.submit(S, 5, 100)
    bid = ob.submit(B, 5, 98).order_id
    r = ob.amend(bid, price=100)
    assert r.status is Status.FILLED


def test_bad_input_rejected():
    ob = OrderBook()
    assert ob.submit(B, 0, 100).status is Status.REJECTED
    assert ob.submit(B, 5, -1).status is Status.REJECTED


# -------------------------------------------- differential test vs. reference
class RefBook:
    """Deliberately naive: a flat list, linear scans, no cleverness."""

    def __init__(self):
        self.orders = []  # dicts: id, side, price, qty, seq
        self.seq = 0
        self.next_id = 1

    def _crosses(self, side, limit, p):
        return True if limit is None else (p <= limit if side is B else p >= limit)

    def _best(self, side, limit):
        opp = [o for o in self.orders if o["side"] is side.opposite and self._crosses(side, limit, o["price"])]
        if not opp:
            return None
        return min(opp, key=lambda o: (o["price"] if side is B else -o["price"], o["seq"]))

    def submit(self, side, qty, price, tif, post_only, oid=None):
        if oid is None:
            oid, self.next_id = self.next_id, self.next_id + 1
        if qty <= 0 or (price is not None and price <= 0):
            return []
        if price is None:
            tif = TIF.IOC
        if post_only and self._best(side, price) is not None:
            return []
        if tif is TIF.FOK:
            avail = sum(o["qty"] for o in self.orders
                        if o["side"] is side.opposite and self._crosses(side, price, o["price"]))
            if avail < qty:
                return []
        trades, rem = [], qty
        while rem > 0:
            m = self._best(side, price)
            if m is None:
                break
            f = min(rem, m["qty"])
            m["qty"] -= f
            rem -= f
            trades.append((m["price"], f, m["id"], oid))
            if m["qty"] == 0:
                self.orders.remove(m)
        if rem > 0 and tif is TIF.GTC:
            self.seq += 1
            self.orders.append(dict(id=oid, side=side, price=price, qty=rem, seq=self.seq))
        return trades

    def cancel(self, oid):
        self.orders = [o for o in self.orders if o["id"] != oid]

    def amend(self, oid, qty, price):
        o = next((x for x in self.orders if x["id"] == oid), None)
        if o is None:
            return []
        nq = o["qty"] if qty is None else qty
        np_ = o["price"] if price is None else price
        if nq <= 0:
            self.cancel(oid)
            return []
        if np_ == o["price"] and nq <= o["qty"]:
            o["qty"] = nq
            return []
        side = o["side"]
        self.cancel(oid)
        return self.submit(side, nq, np_, TIF.GTC, False, oid)

    def snapshot(self):
        out = []
        for side in (B, S):
            prices = sorted({o["price"] for o in self.orders if o["side"] is side}, reverse=(side is B))
            for p in prices:
                q = sorted((o for o in self.orders if o["side"] is side and o["price"] == p), key=lambda o: o["seq"])
                out.append((side.value, p, tuple((o["id"], o["qty"]) for o in q)))
        return tuple(out)


def _tuples(res):
    return [(t.price, t.qty, t.maker_id, t.taker_id) for t in res.trades]


@pytest.mark.parametrize("seed", range(25))
def test_matches_reference_model(seed):
    rng = random.Random(seed)
    fast, ref = OrderBook(), RefBook()
    ids = []
    for _ in range(1500):
        r = rng.random()
        if r < 0.65:
            side = rng.choice([B, S])
            qty = rng.randint(-1, 20)
            price = None if rng.random() < 0.1 else rng.randint(95, 105)
            tif = rng.choice([TIF.GTC, TIF.GTC, TIF.IOC, TIF.FOK])
            po = rng.random() < 0.1 and price is not None
            res = fast.submit(side, qty, price, tif, po)
            ids.append(res.order_id)
            assert _tuples(res) == ref.submit(side, qty, price, tif, po, res.order_id)
        elif r < 0.85 and ids:
            oid = rng.choice(ids)
            fast.cancel(oid)
            ref.cancel(oid)
        elif ids:
            oid = rng.choice(ids)
            qty = rng.choice([None, rng.randint(0, 25)])
            price = rng.choice([None, rng.randint(95, 105)])
            res = fast.amend(oid, qty, price)
            expect = ref.amend(oid, qty, price)
            assert (_tuples(res) if res else []) == expect
        fast.check_invariants()
        assert fast.snapshot() == ref.snapshot()


def test_replay_reproduces_identical_book():
    rng = random.Random(99)
    ob = OrderBook()
    ids = []
    for _ in range(3000):
        if rng.random() < 0.7 or not ids:
            ids.append(ob.submit(rng.choice([B, S]), rng.randint(1, 15), rng.randint(95, 105),
                                 rng.choice(list(TIF)), owner=rng.randint(0, 3)).order_id)
        elif rng.random() < 0.5:
            ob.cancel(rng.choice(ids))
        else:
            ob.amend(rng.choice(ids), rng.randint(1, 15), rng.randint(95, 105))
    clone = replay(ob.journal)
    assert clone.snapshot() == ob.snapshot()
    clone.check_invariants()

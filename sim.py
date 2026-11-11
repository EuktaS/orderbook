"""Market-making simulation on top of the matching engine.

A latent fair value does a random walk. Noise traders supply resting liquidity
and occasional market orders. A stale-quote picker ("informed" flow) trades
against any quote that is more than ``edge`` ticks away from fair value, which
creates adverse selection. A market maker quotes both sides and has to manage
inventory risk against it.

Two quoting policies are compared:
  * FixedSpread: symmetric quotes around the mid, ignores inventory.
  * InventorySkew: Avellaneda-Stoikov style. The reservation price is shifted
    against inventory and the half-spread grows with volatility and risk aversion.

All randomness comes from one seeded ``random.Random`` so runs are reproducible.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional

from .book import OrderBook, Side, TIF, Trade

MM_OWNER = 1


class FixedSpread:
    def __init__(self, half_spread: int = 2, size: int = 5):
        self.half, self.size = half_spread, size

    def quotes(self, mid: float, inventory: int, t_left: float):
        m = round(mid)
        return m - self.half, m + self.half, self.size, self.size


class InventorySkew:
    """r = mid - q * gamma * sigma^2 * (T - t);  delta = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)"""

    def __init__(self, gamma: float = 0.1, sigma: float = 1.0, k: float = 1.5,
                 size: int = 5, max_inv: int = 60):
        self.gamma, self.sigma, self.k, self.size, self.max_inv = gamma, sigma, k, size, max_inv

    def quotes(self, mid: float, inventory: int, t_left: float):
        var = self.sigma ** 2
        reservation = mid - inventory * self.gamma * var * t_left
        half = 0.5 * (self.gamma * var * t_left + (2 / self.gamma) * math.log(1 + self.gamma / self.k))
        bid = math.floor(reservation - half)
        ask = math.ceil(reservation + half)
        bid_sz = 0 if inventory >= self.max_inv else self.size
        ask_sz = 0 if inventory <= -self.max_inv else self.size
        return bid, ask, bid_sz, ask_sz


@dataclass
class SimResult:
    pnl: float
    final_inventory: int
    max_abs_inventory: int
    fills: int
    pnl_path: List[float]


def run(policy, seed: int, steps: int = 4000, sigma: float = 1.0, start: int = 10_000,
        noise_rate: float = 3.0, picker_edge: float = 1.5) -> SimResult:
    rng = random.Random(seed)
    book = OrderBook()
    cash = 0
    inv = 0
    fills = 0
    max_inv = 0

    def on_trade(t: Trade) -> None:
        nonlocal cash, inv, fills
        if t.maker_owner == MM_OWNER:       # MM was resting; taker_side is the opposite of MM's side
            fills += 1
            if t.taker_side is Side.SELL:   # someone sold to us
                inv += t.qty
                cash -= t.price * t.qty
            else:
                inv -= t.qty
                cash += t.price * t.qty

    book.listeners.append(on_trade)

    fv = float(start)
    noise_ids: List[int] = []
    # seed the book with some noise liquidity so there is always a mid
    for d in range(1, 9):
        book.submit(Side.BUY, rng.randint(3, 10), start - d)
        book.submit(Side.SELL, rng.randint(3, 10), start + d)

    mm_ids: List[int] = []
    pnl_path: List[float] = []
    for step in range(steps):
        fv += rng.gauss(0, sigma)
        t_left = 1.0 - step / steps

        # 1. stale-quote picker trades against quotes that are off fair value
        ba, bb = book.best_ask(), book.best_bid()
        if ba is not None and ba < fv - picker_edge:
            book.submit(Side.BUY, rng.randint(1, 6), ba + 1, TIF.IOC)
        if bb is not None and bb > fv + picker_edge:
            book.submit(Side.SELL, rng.randint(1, 6), bb - 1, TIF.IOC)

        # 2. noise traders
        for _ in range(_poisson(rng, noise_rate)):
            side = rng.choice((Side.BUY, Side.SELL))
            if rng.random() < 0.3:
                book.submit(side, rng.randint(1, 4), None)
            else:
                off = 1 + int(rng.expovariate(0.5))
                px = round(fv) - off if side is Side.BUY else round(fv) + off
                noise_ids.append(book.submit(side, rng.randint(2, 8), px, post_only=True).order_id)
        for _ in range(3):  # random cancels keep the book from growing without bound
            if noise_ids:
                book.cancel(noise_ids.pop(rng.randrange(len(noise_ids))))

        # 3. market maker re-quotes: cancel old, place new
        for oid in mm_ids:
            book.cancel(oid)
        mm_ids.clear()
        mid = book.mid() or fv
        bid, ask, bsz, asz = policy.quotes(mid, inv, t_left)
        if bid >= ask:
            ask = bid + 1
        if bsz:
            r = book.submit(Side.BUY, bsz, bid, post_only=True, owner=MM_OWNER)
            mm_ids.append(r.order_id)
        if asz:
            r = book.submit(Side.SELL, asz, ask, post_only=True, owner=MM_OWNER)
            mm_ids.append(r.order_id)

        max_inv = max(max_inv, abs(inv))
        pnl_path.append(cash + inv * fv)

    return SimResult(pnl_path[-1], inv, max_inv, fills, pnl_path)


def _poisson(rng: random.Random, lam: float) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def compare(seeds: int = 30, steps: int = 4000) -> None:
    policies = {
        "FixedSpread(half=2)": lambda: FixedSpread(2),
        "FixedSpread(half=4)": lambda: FixedSpread(4),
        "InventorySkew(gamma=0.1)": lambda: InventorySkew(gamma=0.1),
        "InventorySkew(gamma=0.3)": lambda: InventorySkew(gamma=0.3),
    }
    print(f"{'policy':28s} {'mean PnL':>10s} {'stdev':>8s} {'mean fills':>11s} {'mean max|inv|':>14s} {'PnL/stdev':>10s}")
    for name, make in policies.items():
        rs = [run(make(), s, steps) for s in range(seeds)]
        pnls = [r.pnl for r in rs]
        sd = statistics.pstdev(pnls)
        mean = statistics.fmean(pnls)
        print(f"{name:28s} {mean:10.1f} {sd:8.1f} {statistics.fmean(r.fills for r in rs):11.1f} "
              f"{statistics.fmean(r.max_abs_inventory for r in rs):14.1f} {mean / sd if sd else float('nan'):10.2f}")


if __name__ == "__main__":
    compare()

import random
import time

from orderbook import OrderBook, Side, TIF


def main(n: int = 200_000, seed: int = 1) -> None:
    rng = random.Random(seed)
    ob = OrderBook()
    ids, lat = [], []
    t0 = time.perf_counter()
    for _ in range(n):
        s = time.perf_counter_ns()
        r = rng.random()
        if r < 0.55:
            side = rng.choice((Side.BUY, Side.SELL))
            px = 10_000 + rng.randint(-30, 30)
            ids.append(ob.submit(side, rng.randint(1, 20), px).order_id)
        elif r < 0.85 and ids:
            ob.cancel(ids.pop(rng.randrange(len(ids))))
        else:
            ob.submit(rng.choice((Side.BUY, Side.SELL)), rng.randint(1, 30), None, TIF.IOC)
        lat.append(time.perf_counter_ns() - s)
    dt = time.perf_counter() - t0
    lat.sort()
    pct = lambda p: lat[int(len(lat) * p)] / 1000
    print(f"{n:,} commands in {dt:.2f}s -> {n / dt:,.0f} cmds/sec")
    print(f"latency us: p50={pct(0.50):.1f} p99={pct(0.99):.1f} p99.9={pct(0.999):.1f}")
    print(f"resting orders at end: {len(ob):,}")
    ob.check_invariants()


if __name__ == "__main__":
    main()

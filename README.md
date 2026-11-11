# orderbook

A limit order book and matching engine in pure Python, plus a small
market-making simulator built on top of it.

I wanted to understand what actually happens inside an exchange when orders
arrive, and then see what a market maker has to deal with once adverse
selection and inventory risk enter the picture.

## What's in it

- `orderbook/book.py`: the matching engine
  - price-time priority, trades execute at the resting order's price
  - limit, market, IOC, FOK, post-only
  - self-trade prevention (cancels the resting order)
  - cancel and amend (reducing size keeps queue position; anything else loses it)
  - integer ticks and quantities only, no floats in the matching path
  - command journal and `replay()` for deterministic reconstruction
- `orderbook/sim.py`: seeded market simulation with a fair-value random walk,
  noise traders, a stale-quote picker, and two market-making policies
- `tests/`: 40 tests, including a differential test against a deliberately
  naive reference implementation over 25 random seeds
- `bench.py`: throughput and latency

## Run it

    pip install pytest
    python -m pytest -q
    python bench.py
    python -m orderbook.sim

## Design decisions

**Sorted price list plus per-level FIFO deque.** Each side keeps a dict of
price to level and a sorted list of price keys with the best price at the end,
so best-price lookup and removal are O(1). Inserting a new price level is
O(log n) to find the spot and O(n) for the list shift, which is a memmove and
fast for realistic depth. Asks are stored negated so both sides share one code
path. A heap would make inserts cheaper but makes FOK checks and depth queries
awkward; a tree would be the next step if level counts got very large.

**Lazy removal inside a level.** Cancel flags the order dead and fixes the
level total immediately; the deque entry is dropped when matching reaches it.
That makes cancel O(1) amortized instead of O(queue length).

**Differential testing.** The reference book is a flat list with linear scans.
The tests feed both implementations the same random command stream (including
bad quantities, FOK, post-only, amends) and require identical trades and
identical book state after every step. This caught real bugs in my own
first draft of amend.

**Determinism.** Every command goes in a journal and ids are assigned by a
counter, so `replay(journal)` rebuilds an identical book. Same idea as event
sourcing.

## Measured results (this machine, CPython 3.12)

Engine benchmark, 200k mixed commands (limit, cancel, market-IOC):

    ~170,000 commands/sec, latency p50 4.5us, p99 22us, p99.9 74us

Market-making comparison, 30 seeds x 4000 steps, fair value moves 1 tick per
step and a picker hits any quote more than 1.5 ticks off fair value:

    policy                       mean PnL   stdev   mean max|inventory|
    FixedSpread(half=2)             8388    12813        365
    FixedSpread(half=4)            18257    16028        287
    InventorySkew(gamma=0.1)        5173     1946         53
    InventorySkew(gamma=0.3)        5143     1282         45

The inventory-aware quoter earns less on average than a wide fixed spread but
holds roughly 6x less inventory and has far lower PnL variance. It did not
"win" on raw PnL, and the simulator is a toy (no queue-position modelling, no
latency, no fees), so read it as a demonstration of the tradeoff rather than
a trading result.

## Limitations

- Single-threaded. A real engine would be single-threaded per instrument too,
  but would be written in a language without a GIL and a garbage collector.
- No persistence beyond the in-memory journal.
- The simulator has no latency or queue-position modelling.

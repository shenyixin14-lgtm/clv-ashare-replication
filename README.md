# CLV Reversal Factor — A-Share Replication and Cost Analysis

Replicating a US-equity intraday-reversal factor (close location value) on
A-shares (CSI 300), with strict look-ahead control and an honest transaction-cost
analysis.

**One-line result:** the factor's alpha is real and survives out-of-sample, but
it is a high-turnover daily signal — out-of-sample break-even cost is only
**~1–2 bps**, far below the ~10 bps of real A-share trading friction. The factor
has genuine predictive power but is **not tradable** at daily frequency. This is
a negative result, documented as one.

This project is a direct sequel to
[ashare-multifactor-model](https://github.com/shenyixin14-lgtm/ashare-multifactor-model):
same 8-layer architecture, same look-ahead isolation, extended with a full
execution model (open-to-open fills, direction-aware limit locks) and a
segment-safe evaluator.

---

## The factor

Close location value measures where the close sits inside the day's high–low range:

```
clv_raw = ((close - low) - (high - close)) / (high - low)
```

A weak close (close near the low) is a bet on next-day rebound — an intraday
reversal signal. In US equities the original expression is

```
clv = rank(-((C - L) - (H - C)) / (H - L))
trade_when(volume > adv20, clv, -1)
```

The `trade_when(volume > adv20, ...)` gate is not decoration: on a volume spike
the close location reflects real supply–demand pressure, whereas on a thin day it
is microstructure noise. Both parts are replicated here.

**Hypothesis (stated before testing):** A-shares show short-horizon reversal at
every tested horizon (confirmed in the prior project), so a reversal signal should
carry over with the *same* sign as the US — positive cross-sectional Rank IC.

---

## Results (out-of-sample, 2019–2025)

Direction is fixed on the train split (2015–2018); all reported numbers below are
strictly out-of-sample.

| Metric | Value |
|---|---|
| Train Rank IC (direction only) | +0.044 |
| **OOS Rank IC** | **+0.022** |
| Daily one-side turnover | ~2.95 of 4.0 (~74% full turnover) |
| **OOS break-even cost** | **~1–2 bps** |
| OOS Sharpe @ 0 bps | +1.28 |
| OOS Sharpe @ 10 bps | −2.66 |

OOS IC roughly halves relative to train — expected, and a sign the factor is
*not* overfit (a collapse to zero or a sign flip would be the warning).

### Why it dies

The binding constraint is **turnover, not IC**. Three configurations were tested,
all evaluated out-of-sample so the whole table speaks one language:

| Variant | OOS IC | Turnover | Effect |
|---|---|---|---|
| Base + tradability filter | 0.018 | 3.08 | — |
| + `trade_when(volume>adv20)` | 0.022 | 2.95 | IC up, turnover slightly down |
| + 3-day signal smoothing | 0.017 | 2.45 | turnover −17%, but OOS IC −25% — **net worse** |

Two findings:

- **`trade_when` helps by purifying the signal, not by cutting turnover.** Gating
  out thin-volume days lifts OOS IC (0.018 → 0.022) while turnover barely moves —
  confirming that close location on a low-volume day is noise.
- **Smoothing fails, and fails worse out-of-sample than in-sample.** Averaging the
  signal over three days cuts turnover 17% but costs 25% of OOS IC. CLV's alpha is
  intrinsically intraday; averaging over days dilutes exactly the part that carries
  the edge, and that dilution is sharper out-of-sample. **The alpha cannot be moved
  to a lower frequency — it lives in the most recent day.**

(All numbers reproduced by `verify_variants.py`, which runs each configuration
through the pipeline independently of `main`.)

---

## Architecture

Eight layers, each doing one thing:

```
1. DATA LAYER          load_data          fetch, cache, build open-to-open next_ret
2. FACTOR LAYER        clv_factor         CLV + smoothing + trade_when gate
3. SIGNAL LAYER        measure_ic         [CONTROLLED ZONE] train-only IC, direction
                       build_signals      [CLEAN ZONE] aligned signals, no next_ret
4. PORTFOLIO           build_weights      dollar-neutral top/bottom-20% weights
5. EXECUTION           apply_tradability  direction-aware T+1 limit-lock filter
6. EVALUATOR           evaluate           segment-safe IC + Sharpe + drawdown
7. TEST SUITE          run_all_tests      5 checks
8. MAIN                                   wired end-to-end
```

### Look-ahead isolation

Two zones enforce that no future information reaches a decision:

- **CONTROLLED ZONE** (`measure_ic`) is the *only* function that touches
  `next_ret`. It merges the return in temporarily, computes signed IC on the train
  split to decide factor direction, and never lets `next_ret` leave.
- **CLEAN ZONE** (`build_signals`) receives only features and the train IC. A test
  asserts `next_ret` never appears as a column here.

### Timing convention

Signal at T close → **buy at T+1 open → exit at T+2 open** (open-to-open). This
drives every alignment:

- Return: `next_ret[t] = open_{t+2} / open_{t+1} - 1`, aligned to the signal row T.
- Suspension gate: the return spans T+1 → T+2, so the calendar-gap check is placed
  on *that* window, not on T → T+1.
- Limit-lock check reads T+1 OHLC — this is **execution modeling** (can the
  already-decided position be filled?), not look-ahead, since the signal is fixed
  at T close.

### Direction-aware tradability

A-shares cannot be shorted freely, and the factor most wants to buy weak-close
names — which are the ones most likely to be limit-locked. The filter is therefore
direction-aware:

- long leg + T+1 one-word limit-**up** → can't buy → weight 0
- short leg + T+1 one-word limit-**down** → can't sell → weight 0

Untradable names stay in the book with zero return (an unfilled order), rather than
being dropped and replaced — dropping would let future tradability reshape today's
portfolio (look-ahead). The long/short spread is an **evaluation construct**;
real deployment would be long-only + index-futures hedge.

---

## Test suite

```
[1/5] weight invariants   dollar-neutral, legs sum to ±1, no NaN
[2/5] look-ahead guards   no next_ret leak, aligned IC == |raw IC|
[3/5] data quality        no duplicate keys, one-to-one merge, 6-digit codes
[4/5] determinism         same input → identical IC across runs
[5/5] edge cases          degenerate cross-section flattens, does not crash
```

Two of these guard against mistakes made during development: the **look-ahead
guard** (the original bug — a price-limit filter using next-day data inflated
Sharpe ~2×) and the **determinism check** (reused cached objects producing results
that appear to change but are byte-identical to a prior run).

---

## Known limitations

Documented rather than hidden:

1. **Short-suspension jumps.** The calendar-gap gate (> 9 days) catches long
   suspensions but not 2–5 day ones, whose re-open jump slips through. Effect is
   partly two-sided and small, but present.
2. **Universe survivorship.** The panel uses the *current* CSI 300 constituent
   list backfilled ten years — a look-ahead in universe construction. Point-in-time
   membership would remove it.
3. **Suspended-name return.** A name suspended over its holding window keeps its
   weight but contributes zero return, mildly understating turnover-adjusted PnL.

---

## Running

```bash
pip install numpy pandas akshare
python clv_ashare.py        # full pipeline: train IC, OOS cost grid, test suite
python verify_variants.py   # reproduce the three-variant table
```

First run fetches ~288 CSI 300 stocks via akshare (Tencent backend) and caches to
`data_multifactor_raw.csv`; later runs read the cache.

---

## What this project is really about

Not a high Sharpe — the honest number is negative once costs are real. The point is
the **method**: a hypothesis stated before each test, direction fixed out-of-sample,
each improvement path tried and falsified with numbers, and the *actual* binding
constraint identified (turnover, not IC; and an alpha that cannot be moved to a
lower frequency because it is intraday by nature). A factor with real predictive
power that is nonetheless correctly judged untradable.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLV reversal factor — A-share (CSI 300) replication and cost analysis.

Replicates a US-equity intraday-reversal factor (close location value) on
A-shares, with strict look-ahead control and honest transaction-cost analysis.

Signal:  clv = -((C-L) - (H-C)) / (H-L)   -> long weak closes (bet on rebound)
Trading: signal at T close -> buy at T+1 open -> exit at T+2 open (open-to-open)
Result:  IC positive (reversal, same sign as US), but ~75% daily turnover;
         OOS break-even ~1-2bps << A-share ~10bps cost. Real alpha, untradable.
"""

import os
import time
import numpy as np
import pandas as pd
import akshare as ak

# ---- Config ----------------------------------------------------------------
SYMBOL      = "000300"                # CSI 300 index code
CACHE       = "data_multifactor_raw.csv"
START_DATE  = "20150101"
END_DATE    = "20251231"
Q           = 0.2                     # long/short quantile (top & bottom 20%)
COST_RATE   = 0.001                   # one-way cost per unit turnover
SPLIT       = "2019-01-01"            # train / test split date
MIN_STOCKS  = 10                      # min names per leg before a day is flat
TRADING_DAYS = 252                    # annualization factor
RETRIES     = 2                       # retries per stock on data fetch
SLEEP       = 0.5                     # delay between fetches (rate limiting)
COST_GRID   = [0, 0.0001, 0.0002, 0.0005, 0.0008, 0.001]


# ============================================================
# 1. DATA LAYER
# ============================================================

def get_one(code, start_date=START_DATE, end_date=END_DATE, retries=RETRIES):
    """Fetch one stock's qfq-adjusted daily bars; retry on failure, None if all fail."""
    for attempt in range(retries):
        try:
            name = ("sh" + code) if code.startswith("6") else ("sz" + code)
            raw = ak.stock_zh_a_daily(name, start_date, end_date, adjust="qfq")
            raw["code"] = code
            return raw
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {type(e).__name__}, retrying...")
            time.sleep(3)
    return None


def get_more(codes):
    """Fetch a list of stocks and concatenate into one long panel."""
    frame, total, success = [], len(codes), 0
    for i, code in enumerate(codes, 1):
        one = get_one(code)
        if one is not None:
            frame.append(one)
            success += 1
        print(f"progress {i}/{total} | success {success} | current {code}")
        time.sleep(SLEEP)
    return pd.concat(frame, ignore_index=True)


def load_data(cache=CACHE, threshold=9):
    """Load (or fetch+cache) the panel and build the open-to-open forward return.

    next_ret[t] = open_{t+2}/open_{t+1} - 1  (signal at T close, trade T+1 open).
    Rows whose holding window (T+1 -> T+2) spans a gap > `threshold` calendar days
    are set to NaN (suspension contamination). Stocks with < 250 rows are dropped.
    Returns: (features, next_ret).
    """
    if os.path.exists(cache):
        panel = pd.read_csv(cache, dtype={"code": str})
    else:
        cons = ak.index_stock_cons(symbol=SYMBOL)
        codes = cons["品种代码"].astype(str).str.zfill(6).tolist()
        panel = get_more(codes)
        panel.to_csv(cache, index=False)

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.drop_duplicates(subset=["code", "date"])
    panel = panel.sort_values(["code", "date"])

    # open-to-open forward return, aligned to the signal row T
    panel["today_chg"] = panel.groupby("code")["open"].pct_change()
    panel["next_ret"]  = panel.groupby("code")["today_chg"].shift(-2)

    # gate suspension contamination: measure the T+1 -> T+2 holding gap
    panel["days_diff"] = (panel.groupby("code")["date"].shift(-2)
                          - panel.groupby("code")["date"].shift(-1)).dt.days
    panel.loc[panel["days_diff"] > threshold, "next_ret"] = np.nan

    features = panel[["date", "code", "open", "close", "high", "low",
                      "volume", "today_chg"]].copy()
    next_ret = panel[["date", "code", "next_ret"]].copy()

    # drop stocks with too little history (recently-listed / late index entrants)
    valid = features.groupby("code").size()
    valid = valid[valid >= 250].index
    features = features[features["code"].isin(valid)]
    next_ret = next_ret[next_ret["code"].isin(valid)]
    return features, next_ret


# ============================================================
# 2. FACTOR LAYER
# ============================================================

def clv_factor(panel, adv_window=20, use_trade_when=True, smooth_window=1):
    """Close-location-value reversal signal, cross-sectionally ranked.

    clv_raw = ((C-L) - (H-C)) / (H-L); one-word-limit rows -> NaN (0/0).
    smooth_window > 1 : rolling mean of clv_raw per stock (smooth first).
    use_trade_when    : mask low-volume days (volume <= adv20) as noise (gate after).
    Output signal in [-0.5, 0.5]: weak close -> positive (long, bet on rebound).
    """
    panel = panel.copy()
    clv_raw = ((panel["close"] - panel["low"]) - (panel["high"] - panel["close"])) \
              / (panel["high"] - panel["low"])
    clv_raw = clv_raw.replace([np.inf, -np.inf], np.nan)
    panel["clv_raw"] = clv_raw

    # 1) smooth first on the continuous series (per stock)
    if smooth_window > 1:
        panel["clv_raw"] = panel.groupby("code")["clv_raw"].transform(
            lambda s: s.rolling(smooth_window, min_periods=1).mean())

    # 2) then trade_when: gate out thin-volume days (adv20 excludes today)
    if use_trade_when:
        adv20 = panel.groupby("code")["volume"].transform(
            lambda v: v.shift(1).rolling(adv_window).mean())
        panel["clv_raw"] = panel["clv_raw"].where(panel["volume"] > adv20)

    panel["clv"] = 0.5 - panel.groupby("date")["clv_raw"].rank(pct=True)
    return panel["clv"]


FACTORS = {"clv": clv_factor}


def compute_factors(panel, factors=FACTORS):
    """Compute all registered factors and add each as a column."""
    panel = panel.copy()
    for name, func in factors.items():
        panel[name] = func(panel)
    return panel


# ============================================================
# 3. SIGNAL LAYER (IC, direction, standardization)
# ============================================================

def split_segment(df, segment="oos", split_date=SPLIT):
    """Slice the panel into is / oos / full. Default 'oos' — safe even if forgotten.

    Raises on an unknown segment or an empty slice.
    """
    seg_map = {"oos":  df["date"] >= split_date,
               "is":   df["date"] <  split_date,
               "full": pd.Series(True, index=df.index)}
    if segment not in seg_map:
        raise ValueError(f"segment must be one of {list(seg_map)}, got {segment!r}")
    out = df[seg_map[segment]].copy()
    if len(out) == 0:
        raise ValueError(f"segment={segment!r} produced an empty slice")
    return out


def compute_ic(panel, cols, segment="full", split_date=SPLIT):
    """Per-factor cross-sectional Rank IC (Spearman) on the chosen segment.

    Skips days with too few names or a constant cross-section.
    Returns: {col: {'ic_mean', 'icir', 'ic_win'}}.
    """
    panel = split_segment(panel, segment, split_date)
    panel = panel.copy()

    def cross_ic(x, col, min_stocks=MIN_STOCKS):
        x = x.dropna(subset=[col, "next_ret"])
        if len(x) < min_stocks or x[col].nunique() <= 1 or x["next_ret"].nunique() <= 1:
            return np.nan
        return x[col].corr(x["next_ret"], method="spearman")

    ic_sum = {}
    for col in cols:
        daily_ic = panel.groupby("date").apply(cross_ic, col=col).dropna()
        ic_sum[col] = {
            "ic_mean": daily_ic.mean(),
            "icir":    daily_ic.mean() / daily_ic.std(),
            "ic_win":  (daily_ic > 0).mean(),
        }
    return ic_sum


def align_direction(panel, cols, train_ic):
    """Flip factors with negative TRAIN IC so all are positively oriented.

    Direction is decided on train IC only; test reuses it (no look-ahead).
    """
    panel = panel.copy()
    for col in cols:
        if train_ic[col] < 0:
            panel[col] = -panel[col]
    return panel


def standardize(panel, cols):
    """Per-day cross-sectional z-score (needed only for multi-factor blending)."""
    panel = panel.copy()
    panel[cols] = panel.groupby("date")[cols].transform(
        lambda x: (x - x.mean()) / x.std())
    return panel


def measure_ic(features, next_ret, cols, factors=FACTORS, split_date=SPLIT):
    """[CONTROLLED ZONE] The only place that touches next_ret.

    Computes signed Rank IC on the TRAIN split, used to decide factor direction.
    next_ret is merged in temporarily and never leaves this function.
    Returns: (raw_ic {col: signed ic_mean}, ic_stats full dict).
    """
    panel = compute_factors(features, factors)
    train = panel[panel["date"] < split_date].copy()
    panel = pd.merge(train, next_ret, on=["date", "code"])
    ic_stats = compute_ic(panel, cols)
    raw_ic = {col: ic_stats[col]["ic_mean"] for col in cols}
    return raw_ic, ic_stats


def build_signals(features, raw_ic, factors=FACTORS, do_standardize=False):
    """[CLEAN ZONE] Build direction-aligned signals; next_ret never enters here.

    do_standardize=True is required for multi-factor blending (unit alignment).
    Returns: (signals, aligned_ic) where aligned_ic = |train IC|.
    """
    features = features.drop(columns=["next_ret"], errors="ignore").copy()
    cols = list(factors.keys())
    signals = compute_factors(features, factors)
    signals = align_direction(signals, cols, raw_ic)
    aligned_ic = {col: abs(raw_ic[col]) for col in raw_ic}
    if do_standardize:
        signals = standardize(signals, cols)
    return signals, aligned_ic


# ============================================================
# 4. PORTFOLIO CONSTRUCTOR — scores -> dollar-neutral weights
# ============================================================

def assign_weights(x, col, q=Q, min_stocks=MIN_STOCKS):
    """One day's cross-section -> dollar-neutral weights.

    Long top (1-q), short bottom q, each leg equal-weighted to +1 / -1.
    Flat day if too few valid names or either leg is empty.
    """
    x = x.copy()
    x["weight"] = 0.0
    valid = x[col].notna()
    if valid.sum() < min_stocks:
        return x["weight"]

    lo = x.loc[valid, col].quantile(q)
    hi = x.loc[valid, col].quantile(1 - q)
    is_short = valid & (x[col] < lo)
    is_long  = valid & (x[col] > hi)
    n_long, n_short = is_long.sum(), is_short.sum()
    if n_long == 0 or n_short == 0:            # empty leg -> flat day
        return x["weight"]

    x.loc[is_long,  "weight"] =  1.0 / n_long
    x.loc[is_short, "weight"] = -1.0 / n_short
    return x["weight"]


def build_weights(panel, col, q=Q, min_stocks=MIN_STOCKS):
    """Add a per-day 'weight' column to the panel."""
    panel = panel.copy()
    panel["weight"] = panel.groupby("date", group_keys=False).apply(
        assign_weights, col=col, q=q, min_stocks=min_stocks)
    return panel


# ============================================================
# 5. EXECUTION ENGINE — tradability filter + renormalize
# ============================================================

def apply_tradability(panel, factor, min_stocks=MIN_STOCKS):
    """Zero out untradable names, then renormalize each leg to dollar-neutral.

    Untradable = direction-aware T+1 one-word limit lock:
      long leg  & T+1 one-word limit-up   -> can't buy   -> weight 0
      short leg & T+1 one-word limit-down -> can't sell  -> weight 0
    Uses T+1 OHLC = execution modeling (signal fixed at T close), not look-ahead.
    """
    panel = panel.copy()
    # pull T+1 OHLC onto the T row (per stock)
    t1_open = panel.groupby("code")["open"].shift(-1)
    t1_high = panel.groupby("code")["high"].shift(-1)
    t1_low  = panel.groupby("code")["low"].shift(-1)
    close_t = panel["close"]

    # one-word limit = T+1 open == high == low (float tolerance)
    is_lock = np.isclose(t1_high, t1_low) & np.isclose(t1_open, t1_high)
    limit_up_lock   = is_lock & (t1_high > close_t)   # limit-up   -> can't buy
    limit_down_lock = is_lock & (t1_high < close_t)   # limit-down -> can't sell

    panel.loc[limit_up_lock   & (panel[factor] > 0), "weight"] = 0.0
    panel.loc[limit_down_lock & (panel[factor] < 0), "weight"] = 0.0

    def normalize_one_day(g):
        long_m, short_m = g["weight"] > 0, g["weight"] < 0
        if long_m.sum() < min_stocks or short_m.sum() < min_stocks:
            g["weight"] = 0.0
            return g
        g.loc[long_m,  "weight"] /= g.loc[long_m,  "weight"].sum()
        g.loc[short_m, "weight"] /= abs(g.loc[short_m, "weight"].sum())
        return g

    return panel.groupby("date", group_keys=False).apply(normalize_one_day)


# ============================================================
# 6. EVALUATOR — backtest metrics
# ============================================================

def calc_sharpe_max_dd(backtest, cost_rate=COST_RATE, trading_days=TRADING_DAYS,
                       segment="full", split_date=SPLIT):
    """Annualized net Sharpe and max drawdown on a settled panel.

    Gross daily return = sum(weight * next_ret). Cost = cost_rate * turnover,
    turnover[t] = sum_i |w[t] - w[t-1]|. Requires 'weight' and 'next_ret' columns.
    """
    backtest = split_segment(backtest, segment, split_date)
    backtest = backtest.copy().sort_values(["code", "date"])

    daily_ret = backtest.groupby("date").apply(
        lambda x: (x["weight"] * x["next_ret"]).sum())
    backtest["prev_weight"] = backtest.groupby("code")["weight"].shift(1).fillna(0)
    backtest["weight_chg"]  = (backtest["weight"] - backtest["prev_weight"]).abs()
    turnover = backtest.groupby("date")["weight_chg"].sum()

    daily_net_ret = daily_ret - turnover * cost_rate
    sharpe = daily_net_ret.mean() / daily_net_ret.std() * np.sqrt(trading_days)
    acc = (1 + daily_net_ret).cumprod()
    max_dd = (acc / acc.cummax() - 1).min()
    return {"sharpe": sharpe, "max_dd": max_dd}


def evaluate(backtest, cols, cost_rate=COST_RATE, segment="oos", split_date=SPLIT):
    """All metrics (IC + Sharpe/drawdown) for one segment. Slicing happens only here.

    Default 'oos': call evaluate() and you get honest out-of-sample numbers.
    """
    seg  = split_segment(backtest, segment, split_date)
    ic   = compute_ic(seg, cols, segment="full")          # seg already sliced
    perf = calc_sharpe_max_dd(seg, cost_rate=cost_rate, segment="full")
    return {"segment": segment, "ic": ic, **perf}


# ============================================================
# 7. TEST SUITE
# ============================================================

def test_weights(weights):
    """Weight invariants: dollar-neutral, legs sum to +/-1, no NaN."""
    net = weights.groupby("date")["weight"].sum().abs().max()
    assert net < 1e-10, f"net exposure too large: {net}"
    long_sum  = weights[weights["weight"] > 0].groupby("date")["weight"].sum()
    short_sum = weights[weights["weight"] < 0].groupby("date")["weight"].sum()
    assert np.allclose(long_sum, 1),   "long leg should sum to +1"
    assert np.allclose(short_sum, -1), "short leg should sum to -1"
    assert weights["weight"].isna().sum() == 0, "weight has NaN"
    print("[1/5] weight invariants passed")


def test_no_lookahead(signals, raw_ic, aligned_ic):
    """Look-ahead guards: no next_ret leak, alignment correct, aligned == |raw|."""
    assert "next_ret" not in signals.columns, "next_ret leaked into signal pipeline"
    assert all(v >= 0 for v in aligned_ic.values()), "aligned IC must be non-negative"
    assert all(np.isclose(aligned_ic[c], abs(raw_ic[c])) for c in aligned_ic), \
        "aligned IC must equal |raw IC|"
    print("[2/5] look-ahead guards passed")


def test_data_quality(features, next_ret, weights, backtest):
    """Data quality: no duplicate keys, merge is one-to-one, code is 6-digit."""
    assert features.duplicated(subset=["date", "code"]).sum() == 0, "duplicate (date, code)"
    assert len(weights) == len(backtest), "merge inflated row count (one-to-many)"
    assert features["code"].str.len().eq(6).all(), "code is not 6-digit"
    print("[3/5] data quality passed")


def test_consistency(features, next_ret, cols, factors=FACTORS, split_date=SPLIT):
    """Determinism: same input yields identical IC across runs."""
    ic1 = measure_ic(features, next_ret, cols, factors=factors, split_date=split_date)
    ic2 = measure_ic(features, next_ret, cols, factors=factors, split_date=split_date)
    assert ic1 == ic2, "measure_ic is not deterministic"
    print("[4/5] determinism passed")


def test_edge_cases():
    """Degenerate cross-section: too few names should flatten the day, not crash."""
    tiny = pd.DataFrame({
        "date":  pd.to_datetime(["2020-01-01"] * 3),
        "code":  ["000001", "000002", "000003"],
        "weight": [0.5, -0.3, -0.2],
        "clv":    [0.4, -0.2, -0.1],
        "open":   [10.0, 20.0, 30.0],
        "high":   [10.5, 20.5, 30.5],
        "low":    [ 9.5, 19.5, 29.5],
        "close":  [10.0, 20.0, 30.0],
    })
    out = apply_tradability(tiny, factor="clv", min_stocks=10)
    assert (out["weight"] == 0).all(), "too-few-names day should be flat"
    print("[5/5] edge cases passed")


def run_all_tests(features, next_ret, cols, signals, raw_ic, aligned_ic, weights, backtest):
    """Run the full test suite."""
    test_weights(weights)
    test_no_lookahead(signals, raw_ic, aligned_ic)
    test_data_quality(features, next_ret, weights, backtest)
    test_consistency(features, next_ret, cols)
    test_edge_cases()
    print("All tests passed \u2713")


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    from functools import partial

    features, next_ret = load_data()
    FACTORS = {"clv": partial(clv_factor, use_trade_when=True, smooth_window=1)}
    cols = list(FACTORS.keys())

    raw_ic, ic_stats = measure_ic(features, next_ret, cols, factors=FACTORS)
    print("Train IC (direction):", raw_ic)

    signals, aligned_ic = build_signals(features, raw_ic, factors=FACTORS)
    col = "clv"
    weights = build_weights(signals, col)
    weights = apply_tradability(weights, col, min_stocks=MIN_STOCKS)
    backtest = weights.merge(next_ret, on=["date", "code"])

    for c in COST_GRID:
        r = evaluate(backtest, cols, cost_rate=c)
        print(f"[{r['segment']}] cost={c*1e4:>4.1f}bps  "
              f"IC={r['ic']['clv']['ic_mean']:+.4f}  "
              f"sharpe={r['sharpe']:+.3f}  maxdd={r['max_dd']:.3f}")

    run_all_tests(features, next_ret, cols, signals, raw_ic, aligned_ic, weights, backtest)
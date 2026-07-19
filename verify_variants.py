#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

from clv_ashare import (
    load_data, clv_factor, measure_ic, build_signals,
    build_weights, apply_tradability, compute_ic,
)


def turnover_of(backtest):
    """Daily one-side turnover = mean over days of sum_i |w[t] - w[t-1]|."""
    bt = backtest.sort_values(["code", "date"]).copy()
    bt["prev_w"] = bt.groupby("code")["weight"].shift(1).fillna(0)
    daily = bt.groupby("date").apply(lambda x: (x["weight"] - x["prev_w"]).abs().sum())
    return daily.mean()


def run_variant(features, next_ret, factor_func, label):
    """Full pipeline for one CLV config; print OOS IC and turnover."""
    factors = {"clv": factor_func}
    raw_ic, _ = measure_ic(features, next_ret, ["clv"], factors=factors)
    signals, _ = build_signals(features, raw_ic, factors=factors)
    weights = build_weights(signals, "clv")
    weights = apply_tradability(weights, "clv")
    backtest = weights.merge(next_ret, on=["date", "code"])

    ic = compute_ic(backtest, ["clv"], segment="oos")["clv"]["ic_mean"]
    turn = turnover_of(backtest)
    print(f"{label:<32} OOS IC={ic:+.4f}   turnover={turn:.4f}")


if __name__ == "__main__":
    features, next_ret = load_data()

    variants = [
        # Base: CLV, no volume gate, no smoothing
        (partial(clv_factor, use_trade_when=False, smooth_window=1),
         "Base + tradability filter"),
        # + trade_when(volume > adv20)
        (partial(clv_factor, use_trade_when=True, smooth_window=1),
         "+ trade_when(volume>adv20)"),
        # + 3-day smoothing (on top of trade_when)
        (partial(clv_factor, use_trade_when=True, smooth_window=3),
         "+ 3-day signal smoothing"),
    ]

    print("=" * 62)
    for factor_func, label in variants:
        run_variant(features, next_ret, factor_func, label)
    print("=" * 62)
    print("README table: 0.018/3.08, 0.022/2.95, 0.017/2.45")
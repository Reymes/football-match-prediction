#!/usr/bin/env python
"""Run the walk-forward backtest and print a scorecard.

Usage:
    python scripts/run_backtest.py [--test-start 2025-08-01] [--val-start 2024-08-01]

Trains on all history before the validation window, fits the ensemble +
calibrator on the validation season, and evaluates every model on the
out-of-time test season against the de-vigged bookmaker baseline.
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from match_predict.data import load_all, validate_matches
from match_predict.features import build_feature_frame
from match_predict.evaluation import WalkForwardBacktest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["football-data", "testing"])
    ap.add_argument("--test-start", default="2025-08-01")
    ap.add_argument("--val-start", default="2024-08-01")
    ap.add_argument("--leagues", nargs="*", default=None)
    ap.add_argument("--cache", default=None, help="pickle path to cache features")
    args = ap.parse_args()

    if args.cache:
        import os
        if os.path.exists(args.cache):
            feat = pd.read_pickle(args.cache)
        else:
            feat = build_feature_frame(load_all(*args.data))
            feat.to_pickle(args.cache)
    else:
        df = load_all(*args.data)
        print(validate_matches(df))
        feat = build_feature_frame(df)

    bt = WalkForwardBacktest()
    res = bt.run(feat, test_start=args.test_start, val_start=args.val_start,
                 leagues=args.leagues)

    print("\n================ WALK-FORWARD TEST SCORECARD ================")
    print(res.summary().to_string())
    print("\nLower is better for log_loss / brier / rps / ece.")
    print("\nEnsemble relative influence (normalised |coef|, NOT mixture weights):",
          res.extra["ensemble_influence"])
    print("Calibration temperature:  T =", round(res.extra["temperature"], 3))
    print("Dixon-Coles Poisson deviance  home=%.3f  away=%.3f"
          % (res.extra["poisson_deviance_home"], res.extra["poisson_deviance_away"]))

    # Per-league breakdown for the best model
    tf = res.test_frame.copy()
    from match_predict.evaluation import evaluate_proba
    best = res.probas["ensemble_cal"]
    print("\nPer-league (ensemble_cal) log-loss vs market:")
    for lg, idx in tf.groupby("league").groups.items():
        ii = [tf.index.get_loc(i) for i in idx]
        y = tf["y"].to_numpy()[ii]
        m = evaluate_proba(res.probas["market"][ii], y)
        e = evaluate_proba(best[ii], y)
        print(f"  {lg:12s} n={m['n']:4d}  market_ll={m['log_loss']:.4f}  "
              f"ensemble_ll={e['log_loss']:.4f}  ece={e['ece']:.4f}")


if __name__ == "__main__":
    main()

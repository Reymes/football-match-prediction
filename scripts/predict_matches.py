#!/usr/bin/env python
"""Produce full, explained predictions for real test-season fixtures.

Runs the walk-forward backtest to obtain frozen models, then formats
per-match predictions (1X2, xG, top scorelines, BTTS, O/U, Asian handicap,
confidence, SHAP-based reasons) for a chosen date window — with the ACTUAL
result shown for reference.

Usage:
    python scripts/predict_matches.py --cache <feat.pkl> \
        --league England-PL --from 2025-08-15 --to 2025-08-18 --limit 6
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from match_predict.data import load_all
from match_predict.features import build_feature_frame
from match_predict.evaluation import WalkForwardBacktest
from match_predict.pipeline import format_prediction

_LABEL_INV = {0: "H", 1: "D", 2: "A"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["football-data", "testing"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--test-start", default="2025-08-01")
    ap.add_argument("--val-start", default="2024-08-01")
    ap.add_argument("--league", default="England-PL")
    ap.add_argument("--from", dest="dfrom", default="2025-08-15")
    ap.add_argument("--to", dest="dto", default="2025-08-19")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    import os
    if args.cache and os.path.exists(args.cache):
        feat = pd.read_pickle(args.cache)
    else:
        feat = build_feature_frame(load_all(*args.data))
        if args.cache:
            feat.to_pickle(args.cache)

    bt = WalkForwardBacktest()
    res = bt.run(feat, test_start=args.test_start, val_start=args.val_start,
                 verbose=False)

    tf = res.test_frame.reset_index(drop=True)
    hda = res.probas["ensemble_cal"]
    lam = res.extra["dc_lambda"]
    mu = res.extra["dc_mu"]
    importance = res.gbm.feature_importance()

    mask = ((tf.league == args.league)
            & (tf.date >= pd.Timestamp(args.dfrom))
            & (tf.date <= pd.Timestamp(args.dto)))
    idx = np.where(mask.to_numpy())[0][: args.limit]
    if len(idx) == 0:
        print("No fixtures in that window; try a wider --from/--to.")
        return

    print(f"\n=== Predictions: {args.league}  {args.dfrom} .. {args.dto} ===\n")
    correct = 0
    for k in idx:
        row = tf.iloc[k]
        pred = format_prediction(
            row, hda[k], lam[k], mu[k], rho=-0.045,
            gbm=res.gbm, gbm_row_df=tf.iloc[[k]], importance=importance)
        pred.actual = _LABEL_INV[int(row["y"])]
        if np.argmax(hda[k]) == int(row["y"]):
            correct += 1
        print(pred.pretty())
        print()
    print(f"argmax hit-rate on shown fixtures: {correct}/{len(idx)}")
    print("\nGlobal GBM feature importance (top 8, gain):")
    print(importance.head(8).round(0).to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Chronological decision backtest (bet.md §16, §24).

    python -m scripts.backtest_decisions \
        --from-date 2022-08-01 --to-date 2026-05-31 \
        --val-start 2024-08-01 --test-start 2025-08-01 \
        --config config/decision_engine.yml

Loads canonical history from --data-root (real Football-Data if synced, else a
clearly-labelled SYNTHETIC dataset), runs the walk-forward decision backtest,
and writes a JSON report. Thresholds are selected on validation and only
measured on the untouched test period. Probability quality is reported
separately from realized return.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_predict.features.build import build_feature_frame       # noqa: E402
from match_predict.decisions.backtest import run_decision_backtest  # noqa: E402
from match_predict.decisions import load_config                     # noqa: E402


def _load_history(data_root: str):
    """Real canonical history if present; otherwise a labelled synthetic set."""
    try:
        from match_predict.data import load_all
        df = load_all(data_root)
        if df is not None and len(df):
            return df, False
    except Exception:            # noqa: BLE001
        pass
    from match_predict.decisions.synthetic import make_synthetic_matches
    return make_synthetic_matches(), True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-date", default=None)
    ap.add_argument("--to-date", default=None)
    ap.add_argument("--val-start", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--view", default="hybrid", choices=["pure", "market", "hybrid"])
    ap.add_argument("--config", default="config/decision_engine.yml")
    ap.add_argument("--data-root", default="football-data")
    ap.add_argument("--out", default="outputs/decisions/backtest.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    history, synthetic = _load_history(args.data_root)
    if args.from_date:
        history = history[history.date >= pd.Timestamp(args.from_date)]
    if args.to_date:
        history = history[history.date <= pd.Timestamp(args.to_date)]

    feat = build_feature_frame(history)
    res = run_decision_backtest(feat, test_start=args.test_start,
                                val_start=args.val_start, config=cfg,
                                view=args.view, verbose=True)

    report = {
        "data_source": "SYNTHETIC (not real performance)" if synthetic else "real",
        "view": args.view,
        "val_start": args.val_start, "test_start": args.test_start,
        "forecast_quality": res.forecast_quality,
        "decision_performance": res.decision_performance,
        "calibration_of_selected": res.calibration_selected,
        "profiles": res.profiles,
        "per_league": res.per_league.to_dict(orient="records"),
        "per_odds_band": res.per_odds_band.astype(str).to_dict(orient="records"),
        "per_edge_band": res.per_edge_band.astype(str).to_dict(orient="records"),
        "notes": res.notes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps({"data_source": report["data_source"],
                      "forecast_quality": res.forecast_quality,
                      "decision_performance": res.decision_performance,
                      "calibration_of_selected": res.calibration_selected,
                      "notes": res.notes}, indent=2, default=str))
    print(f"\nwrote {args.out}")
    if synthetic:
        print("WARNING: results are on SYNTHETIC data and are NOT real model "
              "performance.")


if __name__ == "__main__":
    main()

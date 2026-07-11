#!/usr/bin/env python
"""Build out-of-time market-validation profiles for the decision layer (bet.md §10, §18).

    python -m scripts.build_market_profiles \
        --val-start 2024-08-01 --test-start 2025-08-01

Runs the leakage-safe walk-forward decision backtest, extracts the per-market
validation profile (fitted on the VALIDATION period only, never the test set),
and writes it to `artifacts/market_profiles.json`. The serving path
(`/api/decisions`, `scripts/research_decisions.py`) loads this file so a market
is only ever evaluated once its probabilities have demonstrated acceptable
out-of-time quality; without it every selection is rejected as unvalidated.

This never places a bet and never claims certainty. A profile that FAILS the
quality bar is written too (with `passed_quality=false`) so the serving layer
keeps rejecting that market — the honest default.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_predict.decisions import load_config, save_profiles           # noqa: E402
from match_predict.decisions.backtest import run_decision_backtest       # noqa: E402
from match_predict.decisions.validation import profile_from_dict         # noqa: E402
from match_predict.features.build import build_feature_frame             # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-start", default="2024-08-01")
    ap.add_argument("--test-start", default="2025-08-01")
    ap.add_argument("--config", default="config/decision_engine.yml")
    ap.add_argument("--data-root", default="football-data")
    ap.add_argument("--out", default="artifacts/market_profiles.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    from match_predict.data import load_all
    history = load_all(args.data_root)
    if history is None or not len(history):
        print("no real history found — cannot build honest profiles. Sync data first.")
        return

    feat = build_feature_frame(history)
    res = run_decision_backtest(feat, test_start=args.test_start,
                                val_start=args.val_start, config=cfg, verbose=True)

    profiles = {m: profile_from_dict(d) for m, d in res.profiles.items()}
    meta = {
        "val_start": args.val_start, "test_start": args.test_start,
        "data_root": args.data_root, "n_matches": int(len(history)),
        "forecast_quality_out_of_time": res.forecast_quality,
        "notes": res.notes,
    }
    save_profiles(profiles, args.out, meta=meta)

    print(f"\nwrote {args.out}")
    for m, p in profiles.items():
        status = "PASSED" if p.passed_quality else "FAILED (market stays rejected)"
        print(f"  {m}: {status} — n={p.n_samples}, "
              f"log_loss={p.log_loss}, ece={p.ece}")
    print("Out-of-time validation only; thresholds were not tuned on the test "
          "period. 'No bet' remains the default when quality is not demonstrated.")


if __name__ == "__main__":
    main()

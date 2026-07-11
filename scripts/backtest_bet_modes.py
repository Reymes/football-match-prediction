#!/usr/bin/env python
"""Chronological backtest for the Smart Bet / High Return modes (bet-funcuanlty §16, §19).

    python -m scripts.backtest_bet_modes --mode smart
    python -m scripts.backtest_bet_modes --mode high-return
    python -m scripts.backtest_bet_modes --mode both

Runs the leakage-safe walk-forward backtest, freezes the market validation
profile on the VALIDATION period, and scores each mode's decision rules on the
untouched TEST period. Each mode is reported INDEPENDENTLY — they are never
merged and never compared by win rate alone (bet-funcuanlty §16). High-return
results always include the return with the largest wins removed (§17) so a
strategy that only looks profitable because of one rare win is flagged.

Never places a bet, never fetches odds from a website, never claims certainty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_predict.data import load_all                                # noqa: E402
from match_predict.decisions import load_config                        # noqa: E402
from match_predict.decisions.mode_backtest import run_mode_backtest    # noqa: E402
from match_predict.features.build import build_feature_frame           # noqa: E402

_ALIASES = {"smart": "smart", "high-return": "high_return"}


def _report(res) -> dict:
    return {
        "mode": res.mode, "profile_passed": res.profile_passed,
        "forecast_quality": res.forecast_quality,
        "performance": res.performance, "stability": res.stability,
        "calibration_selected": res.calibration_selected,
        "per_league": res.per_league.to_dict("records") if res.per_league is not None else [],
        "per_odds_band": [{**r, "odds_band": str(r["odds_band"])}
                          for r in res.per_odds_band.to_dict("records")]
        if res.per_odds_band is not None else [],
        "notes": res.notes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smart", "high-return", "both"],
                    default="both")
    ap.add_argument("--val-start", default="2024-08-01")
    ap.add_argument("--test-start", default="2025-08-01")
    ap.add_argument("--config", default="config/decision_engine.yml")
    ap.add_argument("--data-root", default="football-data")
    ap.add_argument("--out", default="outputs/decisions/mode_backtest.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    history = load_all(args.data_root)
    if history is None or not len(history):
        print("no real history found — sync data first. No synthetic results.")
        return
    feat = build_feature_frame(history)

    modes = ["smart", "high_return"] if args.mode == "both" \
        else [_ALIASES[args.mode]]
    out: dict = {"val_start": args.val_start, "test_start": args.test_start,
                 "modes": {}}
    for m in modes:
        res = run_mode_backtest(feat, test_start=args.test_start,
                                val_start=args.val_start, mode=m, config=cfg,
                                verbose=True)
        out["modes"][m] = _report(res)
        print(f"\n=== {m} ===")
        print(json.dumps({"performance": res.performance,
                          "stability": res.stability}, indent=2, default=str))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")
    print("Out-of-time only; thresholds were frozen on validation, never the "
          "test set. 'No bet' is a normal result. High-return strategies have a "
          "LOW hit rate and high variance — check the top-win-removal stability "
          "report before trusting any positive return.")


if __name__ == "__main__":
    main()

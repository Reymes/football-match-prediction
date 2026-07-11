#!/usr/bin/env python
"""Score upcoming fixtures with the trained Predictor bundle.

Usage:
    python scripts/predict_upcoming.py                     # bundled demo fixtures
    python scripts/predict_upcoming.py --fixtures my.csv   # your own fixtures
    python scripts/predict_upcoming.py --json out.json     # also dump JSON

Fixtures CSV columns:
    league,date,home_team,away_team[,odds_h,odds_d,odds_a]
`league` must match the trained leagues (England-PL, France-L1, Germany-BL,
Italy-SA, Spain-LL, Portugal-PL). Team names must match Football-Data spelling
(e.g. "Man City", "Nott'm Forest"). Odds are optional — without them the
market-free fallback ensemble is used.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from match_predict.pipeline import Predictor

DEMO = pd.DataFrame([
    # league, date, home, away, (optional) odds
    ["England-PL", "2026-08-15", "Liverpool", "Everton", 1.55, 4.30, 6.00],
    ["England-PL", "2026-08-15", "Arsenal", "Tottenham", 1.80, 3.90, 4.30],
    ["Spain-LL", "2026-08-16", "Barcelona", "Real Madrid", 2.10, 3.70, 3.30],
    ["Germany-BL", "2026-08-16", "Bayern Munich", "Dortmund", 1.65, 4.20, 4.80],
    ["Italy-SA", "2026-08-16", "Inter", "Juventus", 2.00, 3.40, 3.90],
    ["France-L1", "2026-08-16", "Paris SG", "Marseille", None, None, None],  # no odds
], columns=["league", "date", "home_team", "away_team",
            "odds_h", "odds_d", "odds_a"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--fixtures", default=None, help="CSV of fixtures to score")
    ap.add_argument("--json", default=None, help="optional JSON dump path")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.artifacts, "models.joblib")):
        raise SystemExit(f"No trained bundle at {args.artifacts}/. "
                         f"Run: python scripts/train.py --out {args.artifacts}")

    predictor = Predictor.load(args.artifacts)
    print(f"Loaded Predictor (trained through {predictor.trained_through}).\n")

    if args.fixtures:
        fixtures = pd.read_csv(args.fixtures)
    else:
        fixtures = DEMO
        print("No --fixtures given; scoring the bundled demo fixtures.\n")

    preds = predictor.predict_fixtures(fixtures)
    for p in preds:
        print(p.pretty())
        print()

    if args.json:
        def _clean(p):
            d = p.to_dict()
            d["top_scores"] = [[list(s), round(pr, 4)] for s, pr in d["top_scores"]]
            for key in ("over_under", "asian_handicap"):
                d[key] = {str(k): {kk: round(vv, 4) for kk, vv in v.items()}
                          for k, v in d[key].items()}
            d["team_totals"] = {side: {str(k): {kk: round(vv, 4) for kk, vv in v.items()}
                                       for k, v in tt.items()}
                                for side, tt in d["team_totals"].items()}
            d["btts"] = {k: round(v, 4) for k, v in d["btts"].items()}
            return d
        with open(args.json, "w") as f:
            json.dump([_clean(p) for p in preds], f, indent=2, default=str)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

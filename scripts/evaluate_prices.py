#!/usr/bin/env python
"""Evaluate offered prices against model predictions (bet.md §24).

    python -m scripts.evaluate_prices \
        --predictions outputs/upcoming_predictions.csv \
        --odds data/upcoming/latest_odds.csv \
        --config config/decision_engine.yml

The predictions CSV must contain the calibrated model probabilities per fixture
(pure_h/d/a, hybrid_h/d/a) and the odds CSV the offered decimal prices. This
script does NOT fetch odds from any website; it reads a local, already-captured
file. Output is advisory only — it never places a bet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_predict.models.markets import score_matrix, derive_markets  # noqa: E402
from match_predict.pipeline.predict import reconcile_matrix_to_1x2      # noqa: E402
from match_predict.decisions import (                                   # noqa: E402
    build_views, evaluate_match, load_config, ExposureLedger, summarize_day)


def _views_from_row(r):
    pure = {"H": float(r["pure_h"]), "D": float(r["pure_d"]), "A": float(r["pure_a"])}
    hybrid = {"H": float(r["hybrid_h"]), "D": float(r["hybrid_d"]), "A": float(r["hybrid_a"])}
    odds = None
    if all(pd.notna(r.get(k)) for k in ("odds_h", "odds_d", "odds_a")):
        odds = [float(r["odds_h"]), float(r["odds_d"]), float(r["odds_a"])]
    return pure, hybrid, odds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--odds", default=None,
                    help="optional odds CSV joined on fixture_id/home/away")
    ap.add_argument("--config", default="config/decision_engine.yml")
    ap.add_argument("--out", default="outputs/decisions/price_eval.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = pd.read_csv(args.predictions)
    if args.odds:
        odds = pd.read_csv(args.odds)
        on = [c for c in ("fixture_id", "home_team", "away_team") if c in df and c in odds]
        df = df.merge(odds, on=on, how="left", suffixes=("", "_odds"))

    ledger = ExposureLedger(cfg["exposure"], cfg["staking"])
    decisions = []
    for _, r in df.iterrows():
        pure, hybrid, o = _views_from_row(r)
        lam = float(r.get("lam_home", hybrid["H"] * 2 + 0.5))
        mu = float(r.get("lam_away", hybrid["A"] * 2 + 0.5))
        M = reconcile_matrix_to_1x2(score_matrix(lam, mu, rho=-0.045),
                                    [hybrid["H"], hybrid["D"], hybrid["A"]])
        book = derive_markets(M, lam, mu)
        views = build_views(pure, hybrid, outcome_odds_1x2=o,
                            devig_method=cfg.get("devig_method", "shin"))
        prices = {}
        if o:
            prices["match_winner"] = {"H": o[0], "D": o[1], "A": o[2]}
        md = evaluate_match(
            fixture_id=str(r.get("fixture_id", f"{r['home_team']}|{r['away_team']}")),
            league=str(r.get("league", "unknown")),
            home_team=str(r["home_team"]), away_team=str(r["away_team"]),
            kickoff=str(r.get("kickoff", r.get("date", ""))),
            views=views, market_book=book, prices=prices, config=cfg,
            data_quality=float(r.get("data_quality", 1.0)), ledger=ledger)
        decisions.append(md)

    summary = summarize_day(decisions)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary,
                   "decisions": [m.to_dict() for m in decisions]}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    print("NOTE: advisory research output only — no bets are placed, and "
          "no scoreline or outcome is certain.")


if __name__ == "__main__":
    main()

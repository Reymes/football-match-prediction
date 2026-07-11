#!/usr/bin/env python
"""Research upcoming fixtures with the decision engine (bet.md §24).

    python -m scripts.research_decisions \
        --days 7 --config config/decision_engine.yml

Loads the trained Predictor bundle READ-ONLY (never retrains), scores the
upcoming fixtures already synced to disk, attaches the REAL pre-kickoff feed
odds (1X2 and, where present, over/under 2.5) already carried by
`football-data/fixtures.csv`, builds the three independent views
(pure / market / hybrid) via `decisions.decide_for_fixtures`, and runs the
advisory decision engine. It NEVER fetches odds from a website, NEVER accepts
client-supplied odds, and NEVER places a bet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_predict.data import parse_fixtures, parse_fixture_totals_odds  # noqa: E402
from match_predict.decisions import (                                  # noqa: E402
    decide_for_fixtures, load_config, load_profiles, modes_for_fixtures,
    summarize_day)
from match_predict.pipeline import Predictor                            # noqa: E402

_BET_MODE_ALIASES = {"smart": "smart", "high-return": "high_return",
                     "high_return": "high_return"}


def _feed_timestamp(path: str) -> str | None:
    """UTC mtime of the fixtures feed = its verifiable observed-at time."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return pd.Timestamp(mtime, unit="s", tz="UTC").isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--config", default="config/decision_engine.yml")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data-root", default="football-data")
    ap.add_argument("--out", default="outputs/decisions/research.json")
    ap.add_argument("--bet-mode", choices=["smart", "high-return", "compare"],
                    default=None,
                    help="run the Smart Bet / High Return decision mode(s) "
                         "instead of the full market scan (bet-funcuanlty §19)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if not os.path.exists(os.path.join(args.artifacts, "models.joblib")):
        print("no trained bundle found — train one first (read-only here).")
        return
    pred = Predictor.load(args.artifacts)          # READ-ONLY
    fixtures_path = os.path.join(args.data_root, "fixtures.csv")
    fx = parse_fixtures(fixtures_path)
    if len(fx):
        horizon = pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(days=args.days)
        fx = fx[pd.to_datetime(fx["date"]) <= horizon]
    if fx.empty:
        print("no upcoming fixtures within the window.")
        return

    ou25 = parse_fixture_totals_odds(fixtures_path)
    # The feed is a pre-kickoff snapshot with no per-row odds timestamp; its
    # file mtime is a verifiable (non-leaking) capture time, so staleness is
    # judged honestly instead of rejecting every price outright (bet.md §3, §5).
    odds_ts = _feed_timestamp(fixtures_path)
    now = pd.Timestamp.now("UTC").isoformat()
    profiles = load_profiles(os.path.join(args.artifacts, "market_profiles.json"))

    if args.bet_mode:
        _run_modes(pred, fx, args, cfg, ou25, profiles, odds_ts, now)
        return

    decisions = decide_for_fixtures(pred, fx, config=cfg, odds_totals=ou25,
                                    market_profiles=profiles,
                                    odds_timestamp=odds_ts, decision_time=now)

    summary = summarize_day(decisions)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary,
                   "decisions": [m.to_dict() for m in decisions]}, fh,
                  indent=2, default=str)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    print("NOTE: advisory research output only. No odds are fetched from any "
          "website, no bet is placed, and no outcome is certain. 'No bet' is a "
          "normal, successful output; here every selection is rejected — most "
          "for INSUFFICIENT_HISTORICAL_SAMPLE (no per-market validation profiles "
          "are loaded yet) and for failing the conservative EV / edge thresholds.")


def _run_modes(pred, fx, args, cfg, ou25, profiles, odds_ts, now):
    """Run the requested bet MODE(S) and write a mode-keyed research file."""
    modes = ["smart", "high_return"] if args.bet_mode == "compare" \
        else [_BET_MODE_ALIASES[args.bet_mode]]
    results = modes_for_fixtures(pred, fx, modes, config=cfg, odds_totals=ou25,
                                 market_profiles=profiles,
                                 odds_timestamp=odds_ts, decision_time=now)
    payload = {m: {"summary": r["summary"],
                   "selections": [s.to_dict() for s in r["selections"]]}
               for m, r in results.items()}
    out = args.out.replace(".json", f".{args.bet_mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    for m, r in results.items():
        print(f"\n=== {m} ===")
        print(json.dumps(r["summary"], indent=2, default=str))
    print(f"\nwrote {out}")
    print("NOTE: advisory research only — no bet is placed, no outcome is "
          "certain, and neither mode is forced to pick a selection. 'No bet' is "
          "a normal result. High-return selections usually LOSE more often than "
          "they win; the value is in the price, not the hit rate.")


if __name__ == "__main__":
    main()

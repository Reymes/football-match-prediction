#!/usr/bin/env python
"""Honest performance & calibration report for the paper-bet ledger (bet.md §16/§18).

    python -m scripts.bet_report
    python -m scripts.bet_report --db matchpredict.db --out outputs/bet_report.json

Reads the paper-betting Store (SQLite) and prints a report that keeps *realized
return* (P&L, ROI, drawdown, longest losing run, return by odds band) strictly
separate from *forecast quality* (calibration of the model probabilities that
were actually bet). Paper money only; no bet is placed and no edge is proven.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from match_predict.bet_analytics import performance_report          # noqa: E402
from match_predict.store import Store                               # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="path to the paper-bet SQLite DB")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--reliability-bins", type=int, default=5)
    args = ap.parse_args(argv)

    store = Store(args.db) if args.db else Store()
    report = performance_report(store.all_bets(limit=100000),
                                reliability_bins=args.reliability_bins)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.out}\n")

    print(json.dumps(report, indent=2))
    rr = report["realized_return"]
    if report["n_bets_total"] == 0:
        print("\nNo bets on record yet — place paper bets from the app first.")
    elif not report["meaningful_sample"]:
        print(f"\nNOTE: only {rr['n_settled']} settled bets — too few to read "
              "anything into. Realized return is not evidence of an edge.")
    print("Advisory research over paper money. No bet is placed; no outcome is "
          "certain.")


if __name__ == "__main__":
    main()

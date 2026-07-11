#!/usr/bin/env python
"""Render a human-readable decision report from a saved run (bet.md §23, §24).

    python -m scripts.decision_report --run outputs/decisions/backtest.json
    python -m scripts.decision_report --run outputs/decisions/research.json

Prints the three-view forecast, price comparison, decision status, reasons and
research summary in the format of bet.md §23. Never uses certainty language.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pct(x):
    return f"{x*100:5.1f}%" if isinstance(x, (int, float)) else "  n/a"


def _print_backtest(data):
    print("=" * 70)
    print("DECISION-RULE BACKTEST REPORT")
    print("=" * 70)
    print(f"Data source: {data.get('data_source')}")
    print(f"View: {data.get('view')}   "
          f"val_start={data.get('val_start')}  test_start={data.get('test_start')}")
    print("\n-- FORECAST QUALITY (separate from decision performance) --")
    for view, sc in (data.get("forecast_quality") or {}).items():
        print(f"  {view:8} log_loss={sc['log_loss']}  brier={sc['brier']}  "
              f"ece={sc['ece']}  acc={sc['accuracy']}  n={sc['n']}")
    print("\n-- DECISION-RULE PERFORMANCE --")
    for k, v in (data.get("decision_performance") or {}).items():
        print(f"  {k}: {v}")
    print("\n-- CALIBRATION OF SELECTED OPPORTUNITIES --")
    print(f"  {data.get('calibration_of_selected') or 'n/a'}")
    print("\n-- NOTES --")
    for n in data.get("notes", []):
        print(f"  * {n}")
    print("\nReminder: no scoreline or outcome is certain; positive expected "
          "value is a research estimate, not a guarantee.")


def _print_research(data):
    print("=" * 70)
    print("UPCOMING-FIXTURE DECISION REPORT")
    print("=" * 70)
    print("SUMMARY:", json.dumps(data.get("summary", {}), indent=2))
    for m in data.get("decisions", []):
        print("\n" + "-" * 60)
        print(f"{m['home_team']} vs {m['away_team']}  "
              f"({m['league']}, kickoff {m['kickoff']})")
        v = m["views"]
        print("  PURE   H/D/A:", {k: _pct(v['pure'].get(k)) for k in ('H', 'D', 'A')})
        if v.get("market"):
            print("  MARKET H/D/A:", {k: _pct(v['market'].get(k)) for k in ('H', 'D', 'A')})
        print("  HYBRID H/D/A:", {k: _pct(v['hybrid'].get(k)) for k in ('H', 'D', 'A')})
        print("  Top scores:", ", ".join(
            f"{s['score']} {_pct(s['prob'])}" for s in m.get("top_scores", [])[:5]))
        qual = [s for s in m["selections"]
                if s["decision_status"] in ("QUALIFIED", "STRONG EVIDENCE")]
        if qual:
            for s in qual:
                print(f"  DECISION: {s['decision_status']} — {s['market']} "
                      f"{s['selection']} @ {s['offered_odds']} "
                      f"(consEV {_pct(s['conservative_expected_value'])})")
        else:
            reasons = sorted({r for s in m["selections"] for r in s["rejection_reasons"]})
            print(f"  DECISION: NO BET — {', '.join(reasons) or 'no eligible priced market'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    if not os.path.exists(args.run):
        print(f"run file not found: {args.run}")
        return
    with open(args.run) as fh:
        data = json.load(fh)
    if "decision_performance" in data:
        _print_backtest(data)
    else:
        _print_research(data)


if __name__ == "__main__":
    main()

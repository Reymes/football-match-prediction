#!/usr/bin/env python
"""Sync fresh data from football-data.co.uk.

    python scripts/sync_data.py latest      # current season only (fast)
    python scripts/sync_data.py fixtures    # upcoming fixtures + their season
    python scripts/sync_data.py all         # full archive rebuild

Then retrain so the new data reaches the models:  python scripts/train.py
"""
from __future__ import annotations

import argparse
import sys

from match_predict.data import sync_all, sync_latest, sync_fixtures


def _progress(msg, pct=None):
    bar = "" if pct is None else f"[{pct:3d}%] "
    print(f"  {bar}{msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["latest", "fixtures", "all"])
    ap.add_argument("--root", default="football-data")
    ap.add_argument("--cache", default=".data-cache")
    args = ap.parse_args(argv)

    fn = {"latest": sync_latest, "fixtures": sync_fixtures, "all": sync_all}[args.mode]
    info = fn(args.root, args.cache, _progress)
    print("\nDone:", {k: v for k, v in info.items() if k != "season"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

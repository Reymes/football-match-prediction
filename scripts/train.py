#!/usr/bin/env python
"""Train ALL production models on the full history and persist them.

Thin CLI wrapper around ``match_predict.pipeline.training.train_and_save`` (the
same code path the web UI's "Retrain" button uses). Writes a ready-to-serve
Predictor bundle plus a ``model_card.json`` (data used + per-model metrics) to
``artifacts/``.

After this runs, `scripts/predict_upcoming.py` (or `app.py`) can score fixtures.
Optionally also run the honest walk-forward evaluation with ``--evaluate``.
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from match_predict.pipeline.training import train_and_save, evaluate_walk_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["football-data", "testing"])
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--val-start", default="2024-08-01")
    ap.add_argument("--val-end", default="2025-08-01")
    ap.add_argument("--dc-window-days", type=int, default=900)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--evaluate", action="store_true",
                    help="also run the walk-forward backtest and record the "
                         "honest test scorecard in the model card")
    args = ap.parse_args()

    def log(msg):
        print(f"[train] {msg}")

    card = train_and_save(
        data=args.data, out=args.out, val_start=args.val_start,
        val_end=args.val_end, dc_window_days=args.dc_window_days,
        cache=args.cache, progress=log)

    print("\nValidation-season metrics (base models out-of-sample; "
          "ensemble/cal are fit-set/optimistic):")
    for m in card["models"]:
        vm = m["val_metrics"]
        print(f"  {m['name']:<13} log_loss={vm['log_loss']:.4f}  "
              f"brier={vm['brier']:.4f}  ece={vm['ece']:.4f}  n={vm['n']}")

    if args.evaluate:
        block = evaluate_walk_forward(
            data=args.data, out=args.out, val_start=args.val_start,
            test_start=args.val_end, cache=args.cache, progress=log)
        print("\nHONEST walk-forward test scorecard "
              f"({block['n_test_matches']} matches):")
        for name, row in block["scorecard"].items():
            print(f"  {name:<13} log_loss={row['log_loss']:.4f}  "
                  f"brier={row['brier']:.4f}  ece={row['ece']:.4f}")

    print("\nReady. Run: python app.py   (or scripts/predict_upcoming.py)")


if __name__ == "__main__":
    main()

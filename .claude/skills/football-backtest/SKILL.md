---
name: football-backtest
description: >-
  Run the honest chronological walk-forward backtest for the match_predict
  project and read the scorecard. Use when asked to evaluate models, produce a
  scorecard, or check calibration vs the market baseline.
---

# football-backtest

Runs the leakage-safe, out-of-time evaluation and reports metrics honestly.

## Steps
1. Ensure the venv is active and the package is importable
   (`pip install -e ".[dev]"` or `export PYTHONPATH=.`).
2. Run the backtest:
   ```bash
   python scripts/run_backtest.py
   ```
   Optional in-app: POST `/api/evaluate` (walk-forward scorecard as a job).
3. Report per-model log-loss, Brier, RPS, ECE, accuracy from `reports/scorecard.csv`.

## Rules
- Evaluation is chronological only — fit on `date < boundary`, 2025/26 is a
  pure hold-out. Never a random split.
- Report metrics honestly: the de-vigged market baseline currently has the best
  log-loss and the ensemble does not beat it; there are no significance tests
  yet. Do not overstate.
- Ensemble "influence" numbers are normalized coefficient magnitudes, not
  mixture weights — never phrase them as "X% market + Y% GBM".
- Accuracy is reported but is never the optimisation target; lead with
  log-loss / ECE.

---
name: football-predict-check
description: >-
  Train the serving bundle and sanity-check a single-fixture prediction for the
  match_predict project. Use when asked to train, predict a fixture, or verify
  the Flask prediction path works end-to-end.
---

# football-predict-check

Trains the bundle and verifies a full explained prediction comes back valid.

## Steps
1. Train the serving bundle if `artifacts/models.joblib` is missing:
   ```bash
   python scripts/train.py --out artifacts
   ```
2. Predict a demo fixture (CLI) or run the app:
   ```bash
   python scripts/predict_upcoming.py                 # bundled demo fixtures
   python app.py                                      # http://127.0.0.1:5001
   ```
3. Verify the output invariants:
   - 1X2 probabilities are finite, non-negative, and sum to 1.
   - Top scorelines come from the reconciled joint matrix; displayed xG equals
     the final matrix (not raw model rates).
   - Every market (BTTS, O/U, AH, team totals) is derivable from the same matrix.
   - Wording is probabilistic — "most likely score X-Y at N%", never certain.

## Rules
- Odds are optional; without them the market-free fallback ensemble is used.
- Do not predict past 2025/26 games with the trained bundle and call it an
  accuracy test — that is in-sample. Use the backtest skill for evaluation.
- After retraining, `STORE.clear_cache()` (the app does this on `/api/train`).

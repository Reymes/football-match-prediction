# No-leakage & honest-evaluation rules

- **Pre-kickoff features only.** Rolling/EWMA form is `shift(1)` per team; Elo
  attaches the pre-match rating (before the update step); closing odds are
  leakage-sensitive — evaluate on pre-match odds.
- **Chronological evaluation only.** Fit on `date < boundary`; walk-forward;
  DC refits on a rolling window; ensemble/calibrator fit on validation then
  frozen for test. 2025/26 is a pure hold-out. No random train/test splits.
- **Calibration over accuracy.** Accuracy is reported, never the objective.
- **Honest metrics.** The de-vigged market baseline currently has the best
  log-loss; the ensemble does NOT beat it and there are no significance tests
  (no paired bootstrap CIs). Do not overstate results. Ensemble "influence"
  figures are normalized coefficient magnitudes, NOT mixture weights.
- **Never duplicate feature logic.** Reuse `features/build.py`
  (`build_feature_frame`, `FEATURE_COLUMNS`) for both training and inference.

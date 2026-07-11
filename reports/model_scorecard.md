# Model scorecard — honest walk-forward test

Evaluated 2026-07-11T23:04:01+02:00 · test from 2025-08-01 (val from 2024-08-01) · 7626 out-of-time matches · trained through 2026-05-31 · 39 features · served model ensemble_cal · market features used: yes

| model | log_loss | brier | rps | ece | accuracy | n |
|---|---|---|---|---|---|---|
| market | 1.0026 | 0.5998 | 0.2038 | 0.0135 | 0.5056 | 7626.0 |
| dixon_coles | 1.0539 | 0.6306 | 0.2182 | 0.0267 | 0.4679 | 7626.0 |
| gbm | 1.0055 | 0.6016 | 0.2044 | 0.0134 | 0.5035 | 7626.0 |
| ensemble | 1.0025 | 0.5998 | 0.2036 | 0.0124 | 0.5062 | 7626.0 |
| ensemble_cal | 1.0025 | 0.5998 | 0.2036 | 0.0125 | 0.5062 | 7626.0 |

Lower is better for log-loss, Brier, RPS and ECE; accuracy is reported for reference only, never optimised.

# Paired significance vs the de-vigged market baseline

Evaluated 2026-07-11T23:04:01+02:00 · test from 2025-08-01 (val from 2024-08-01) · 7626 out-of-time matches · trained through 2026-05-31 · 39 features · served model ensemble_cal · market features used: yes

Paired per-match log-loss differences (challenger − market), 95% confidence interval from a block bootstrap grouped by match-day. Δ < 0 means the challenger has lower loss than the market.

| model | mean Δ log-loss | 95% CI | p-value | distinguishable |
|---|---|---|---|---|
| ensemble_cal | -7e-05 | -0.00195 … 0.00185 | 0.957 | no |
| ensemble | -7e-05 | -0.00194 … 0.00184 | 0.95 | no |
| gbm | 0.0029 | 0.00049 … 0.00546 | 0.017 | yes |
| dixon_coles | 0.05132 | 0.04391 … 0.05876 | 0.0 | yes |

**Served model (ensemble_cal):** ensemble_cal is numerically better than market but the 95% interval includes zero — not statistically distinguishable (practically negligible effect).

We do not claim to beat the bookmaker when the paired interval includes zero.

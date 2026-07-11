# Core rules — match_predict

Mirror of the root `CLAUDE.md`/`claude.md`. These are the must-follow rules.

1. **Milestone build.** Deliver as working milestones: architecture → config →
   data schema → leakage-safe features → Elo → Poisson/Dixon–Coles →
   chronological backtest → calibration → correct-score probabilities.
2. **Real code, tests, docs** — never pseudocode. Every milestone ships an
   implementation, tests, and recorded commands/results.
3. **Per-milestone workflow:** implement → run its tests → fix errors → record
   commands & results → continue.
4. **Stop conditions.** Pause and ask the user only for actions needing
   **credentials, payment, external deployment, or irreversible access.**
5. **Never claim certainty.** No certain scorelines or outcomes. Always return
   full probability distributions and *calibrated* probabilities. No
   "guaranteed / lock / banker / safe" language.
6. **Synthetic data is isolated and labelled** — never present synthetic (or
   fabricated) numbers as real model performance.

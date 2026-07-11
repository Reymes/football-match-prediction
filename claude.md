Read `CLAUDE.md` and inspect the entire repository.

Build the football prediction platform as a sequence of working milestones. Begin with the repository architecture, configuration system, data schema, leakage-safe feature pipeline, Elo baseline, Poisson and Dixon–Coles models, chronological backtesting, calibration and correct-score probability generation.

Create real implementation files, tests and documentation rather than only explaining the design.

Use synthetic fixture data initially when licensed real data is unavailable, but isolate it clearly and never present synthetic results as real model performance.

For each milestone:

1. Implement it.
2. Run its tests.
3. Fix errors.
4. Record the commands and results.
5. Continue to the next milestone unless an action requires credentials, payment, external deployment or irreversible access.

Do not claim that a scoreline is certain. Return full probability distributions and calibrated probabilities.

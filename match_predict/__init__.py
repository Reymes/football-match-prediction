"""match_predict — production-grade pre-match football prediction system.

A modular pipeline that turns Football-Data.co.uk match & odds history into
well-calibrated probability distributions for every major betting market.

Design goals (see ARCHITECTURE.md):
  * Calibration over raw accuracy.
  * Strict pre-kickoff-only features (no leakage).
  * Chronological (walk-forward) evaluation only.
  * A de-vigged bookmaker probability as the strong baseline to beat.
"""

__version__ = "0.1.0"

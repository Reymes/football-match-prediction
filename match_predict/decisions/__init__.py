"""Advisory betting-decision layer (bet.md).

This package sits ON TOP of the existing prediction system. It reuses the
validated joint score matrix (models.markets) and the calibrated ensemble
(pipeline.predictor); it never retrains models, never places bets, and never
claims certainty. "No bet" is a normal, successful output.

The paper-betting wallet (match_predict.betting / store) is a separate feature
and is untouched here.
"""
from .schema import (
    RejectionCode, Grade, DecisionStatus, ThreeViews, PriceQuote,
    SelectionDecision, MatchDecision, load_config,
)
from .decision_engine import (
    evaluate_match, build_views, candidate_selections, summarize_day,
)
from .exposure import ExposureLedger, full_kelly_fraction, hypothetical_stake_fraction
from .serve import (
    decide_for_fixtures, decide_for_prediction, modes_for_fixtures,
)
from .bet_modes import (
    MODE_SMART, MODE_HIGH_RETURN, ModeSelection, evaluate_mode,
    evaluate_match_for_mode, evaluate_selection_for_mode, summarize_mode,
)
from .validation import (
    MarketValidationProfile, build_1x2_profile_from_probs,
    build_profiles_from_backtest, load_profiles, save_profiles,
)

__all__ = [
    "RejectionCode", "Grade", "DecisionStatus", "ThreeViews", "PriceQuote",
    "SelectionDecision", "MatchDecision", "load_config",
    "evaluate_match", "build_views", "candidate_selections", "summarize_day",
    "ExposureLedger", "full_kelly_fraction", "hypothetical_stake_fraction",
    "decide_for_prediction", "decide_for_fixtures", "modes_for_fixtures",
    "MODE_SMART", "MODE_HIGH_RETURN", "ModeSelection", "evaluate_mode",
    "evaluate_match_for_mode", "evaluate_selection_for_mode", "summarize_mode",
    "MarketValidationProfile", "build_1x2_profile_from_probs",
    "build_profiles_from_backtest", "load_profiles", "save_profiles",
]

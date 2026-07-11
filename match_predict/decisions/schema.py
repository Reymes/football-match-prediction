"""Data structures and machine-readable rejection codes for the decision layer.

This module deliberately holds no logic beyond dataclasses and the config
loader, so every other decisions module can import it without cycles.

bet.md references:
  * §20  — the full object a decision must return.
  * §21  — machine-readable rejection reasons.
  * §9   — grades (Reject / Watchlist / Qualified / Strong Evidence).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------- #
# Rejection codes (bet.md §21). Every rejected selection carries at least one.  #
# --------------------------------------------------------------------------- #
class RejectionCode(str, Enum):
    NO_POSITIVE_CONSERVATIVE_EV = "NO_POSITIVE_CONSERVATIVE_EV"
    EDGE_BELOW_THRESHOLD = "EDGE_BELOW_THRESHOLD"
    ODDS_STALE = "ODDS_STALE"
    ODDS_IN_FUTURE = "ODDS_IN_FUTURE"                 # captured after the cutoff
    FIXTURE_SOURCE_STALE = "FIXTURE_SOURCE_STALE"
    MODEL_UNCALIBRATED_IN_BAND = "MODEL_UNCALIBRATED_IN_BAND"
    INSUFFICIENT_HISTORICAL_SAMPLE = "INSUFFICIENT_HISTORICAL_SAMPLE"
    MODEL_DISAGREEMENT_TOO_HIGH = "MODEL_DISAGREEMENT_TOO_HIGH"
    DATA_QUALITY_TOO_LOW = "DATA_QUALITY_TOO_LOW"
    UNRESOLVED_TEAM = "UNRESOLVED_TEAM"
    UNSUPPORTED_LEAGUE = "UNSUPPORTED_LEAGUE"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    PRICE_BELOW_MINIMUM = "PRICE_BELOW_MINIMUM"
    PRICE_ABOVE_VALIDATED_RANGE = "PRICE_ABOVE_VALIDATED_RANGE"
    CORRELATED_SELECTION_ALREADY_CHOSEN = "CORRELATED_SELECTION_ALREADY_CHOSEN"
    DAILY_EXPOSURE_LIMIT = "DAILY_EXPOSURE_LIMIT"
    MARKET_DISABLED = "MARKET_DISABLED"
    INVALID_PROBABILITIES = "INVALID_PROBABILITIES"
    INVALID_ODDS = "INVALID_ODDS"
    MISSING_ODDS = "MISSING_ODDS"
    MODEL_PROBABILITY_TOO_LOW = "MODEL_PROBABILITY_TOO_LOW"
    UNSUPPORTED_MODEL_VERSION = "UNSUPPORTED_MODEL_VERSION"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    # bet-mode layer (bet-funcuanlty §2, §10, §12)
    MARKET_NOT_IN_MODE = "MARKET_NOT_IN_MODE"
    MODE_SUPPORT_TOO_LOW = "MODE_SUPPORT_TOO_LOW"
    NO_PURE_MODEL_SUPPORT = "NO_PURE_MODEL_SUPPORT"
    STRESS_TEST_FAILED = "STRESS_TEST_FAILED"


class Grade(str, Enum):
    """Conservative grades (bet.md §9). A high grade can still lose."""
    REJECT = "Reject"
    WATCHLIST = "Watchlist"
    QUALIFIED = "Qualified"
    STRONG_EVIDENCE = "Strong Evidence"


class DecisionStatus(str, Enum):
    NO_BET = "NO BET"
    WATCHLIST = "WATCHLIST"
    QUALIFIED = "QUALIFIED"
    STRONG_EVIDENCE = "STRONG EVIDENCE"


# --------------------------------------------------------------------------- #
# The three independent probability views (bet.md §2).                         #
# --------------------------------------------------------------------------- #
@dataclass
class ThreeViews:
    """Pure (no odds), market (de-vigged), hybrid (validated ensemble) HDA.

    The pure view must never contain bookmaker information.
    """
    pure: dict = field(default_factory=dict)      # {"H":, "D":, "A":}
    market: dict | None = None                    # de-vigged; None if no odds
    hybrid: dict = field(default_factory=dict)    # calibrated ensemble

    def to_dict(self) -> dict:
        return {"pure": self.pure, "market": self.market, "hybrid": self.hybrid}


@dataclass
class PriceQuote:
    """A single offered price for one selection, with provenance."""
    market: str
    selection: str                     # e.g. "H", "OVER_2.5", "BTTS_YES", "2-1"
    offered_odds: float
    bookmaker: str = "unknown"
    odds_timestamp: str | None = None  # ISO8601; when the price was captured
    available: bool = True             # was the price actually offered/takeable


@dataclass
class SelectionDecision:
    """The full decision object for ONE selection (bet.md §20)."""
    selection: str
    market: str
    side_or_line: str
    offered_odds: float | None
    model_probability: float | None
    market_fair_probability: float | None
    conservative_probability: float | None
    probability_edge: float | None
    raw_expected_value: float | None
    conservative_expected_value: float | None
    confidence_interval: tuple | None
    grade: str
    decision_status: str
    rejection_reasons: list = field(default_factory=list)     # RejectionCode values
    supporting_evidence: list = field(default_factory=list)
    risk_warnings: list = field(default_factory=list)
    model_version: str | None = None
    odds_timestamp: str | None = None
    # research-only (never a recommendation to stake real money)
    hypothetical_kelly_fraction: float | None = None
    correlation_group: str | None = None
    is_primary: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rejection_reasons"] = [
            r.value if isinstance(r, RejectionCode) else r
            for r in self.rejection_reasons
        ]
        return d


@dataclass
class MatchDecision:
    """All selections + three views + score distribution for one fixture."""
    fixture_id: str
    league: str
    home_team: str
    away_team: str
    kickoff: str
    decision_time: str | None
    horizon: str | None
    views: ThreeViews
    selections: list = field(default_factory=list)      # list[SelectionDecision]
    top_scores: list = field(default_factory=list)      # [("2-1", 0.09), ...]
    data_quality: float | None = None
    model_disagreement: float | None = None
    model_version: str | None = None
    notes: list = field(default_factory=list)

    def qualifying(self) -> list:
        return [s for s in self.selections
                if s.decision_status in (DecisionStatus.QUALIFIED.value,
                                          DecisionStatus.STRONG_EVIDENCE.value)]

    def to_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id, "league": self.league,
            "home_team": self.home_team, "away_team": self.away_team,
            "kickoff": self.kickoff, "decision_time": self.decision_time,
            "horizon": self.horizon, "views": self.views.to_dict(),
            "selections": [s.to_dict() for s in self.selections],
            "top_scores": [{"score": s, "prob": p} for s, p in self.top_scores],
            "data_quality": self.data_quality,
            "model_disagreement": self.model_disagreement,
            "model_version": self.model_version, "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Config loading.                                                              #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG_PATH = os.path.join("config", "decision_engine.yml")


def load_config(path: str | None = None) -> dict:
    """Load the decision-engine YAML config; returns the inner mapping.

    Falls back to a conservative built-in default (correct-score disabled) if
    the file is missing so tests never depend on the file being present.
    """
    path = path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        try:
            import yaml
        except ImportError:
            # PyYAML is optional; the built-in defaults below mirror the shipped
            # config (correct-score disabled), so a missing dep never changes
            # behaviour or crashes tests.
            return _fill_defaults({})
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = raw.get("decision_engine", raw)
        return _fill_defaults(cfg)
    return _fill_defaults({})


def _fill_defaults(cfg: dict) -> dict:
    d = dict(_DEFAULTS)
    d.update(cfg or {})
    # deep-merge the nested blocks so a partial config keeps sane defaults
    for k in ("uncertainty", "exposure", "staking"):
        merged = dict(_DEFAULTS[k])
        merged.update(cfg.get(k, {}) or {})
        d[k] = merged
    modes = {m: dict(prof) for m, prof in _DEFAULTS["bet_modes"].items()}
    for name, prof in (cfg.get("bet_modes", {}) or {}).items():
        base = dict(modes.get(name, {}))
        base.update(prof or {})
        modes[name] = base
    d["bet_modes"] = modes
    markets = dict(_DEFAULTS["markets"])
    for name, prof in (cfg.get("markets", {}) or {}).items():
        base = dict(markets.get(name, {}))
        base.update(prof or {})
        markets[name] = base
    d["markets"] = markets
    return d


_DEFAULTS: dict[str, Any] = {
    "require_fresh_fixtures": True,
    "require_timestamped_odds": True,
    "allow_no_bet": True,
    "minimum_data_quality": 0.90,
    "maximum_model_disagreement": 0.10,
    "minimum_historical_samples": 300,
    "max_fixture_staleness_hours": 72,
    "max_odds_staleness_hours": 24,
    "probability_sum_tolerance": 0.02,
    "devig_method": "shin",
    "uncertainty": {
        "base_calibration_error": 0.02,
        "disagreement_weight": 0.50,
        "data_quality_weight": 0.10,
        "sample_weight": 0.05,
        "max_uncertainty_buffer": 0.15,
    },
    "markets": {
        "match_winner": {
            "enabled": True, "minimum_probability_edge": 0.04,
            "minimum_conservative_ev": 0.03, "minimum_odds": 1.50,
            "maximum_odds": 6.00, "minimum_historical_samples": 300},
        "over_under_2_5": {
            "enabled": True, "minimum_probability_edge": 0.035,
            "minimum_conservative_ev": 0.025, "minimum_odds": 1.50,
            "maximum_odds": 3.50, "minimum_historical_samples": 300},
        "btts": {
            "enabled": True, "minimum_probability_edge": 0.035,
            "minimum_conservative_ev": 0.025, "minimum_odds": 1.50,
            "maximum_odds": 3.50, "minimum_historical_samples": 300},
        "correct_score": {
            "enabled": False, "minimum_probability_edge": 0.025,
            "minimum_conservative_ev": 0.08, "minimum_model_probability": 0.07,
            "minimum_odds": 4.00, "maximum_odds": 41.00,
            "minimum_historical_samples": 1000},
    },
    "exposure": {
        "maximum_qualified_selections_per_day": 3,
        "maximum_primary_selections_per_match": 1,
        "maximum_hypothetical_fraction_per_selection": 0.005,
        "maximum_hypothetical_daily_fraction": 0.015,
        "maximum_league_daily_fraction": 0.010,
        "maximum_correlated_group_fraction": 0.005,
    },
    "staking": {
        "enabled": False, "kelly_fraction": 0.25, "hard_max_fraction": 0.005,
    },
    # Bet-mode decision profiles (bet-funcuanlty §4, §5). These are EXAMPLE
    # defaults pending validation — never claimed optimal. They re-threshold the
    # SAME model outputs; they never create a new prediction model.
    "bet_modes": {
        "smart": {
            "enabled": True,
            "minimum_data_quality": 0.92,
            "maximum_model_disagreement": 0.08,
            "minimum_historical_samples": 400,
            "minimum_probability_edge": 0.035,
            "minimum_conservative_ev": 0.025,
            "minimum_model_probability": 0.40,
            "minimum_odds": 1.35,
            "maximum_odds": 4.00,
            "maximum_primary_selections_per_match": 1,
            "maximum_qualified_selections_per_day": 3,
            "require_positive_lower_confidence_bound": True,
            "allowed_markets": ["match_winner", "over_under_2_5", "btts"],
            "strong_ev_multiple": 1.5,
        },
        "high_return": {
            "enabled": True,
            "minimum_data_quality": 0.88,
            "maximum_model_disagreement": 0.14,
            "minimum_historical_samples": 500,
            "minimum_probability_edge": 0.05,
            "minimum_conservative_ev": 0.08,
            "minimum_model_probability": 0.08,
            "minimum_odds": 3.00,
            "maximum_odds": 20.00,
            "maximum_primary_selections_per_match": 1,
            "maximum_qualified_selections_per_day": 2,
            "require_positive_lower_confidence_bound": True,
            "require_model_support_count": 2,
            "require_pure_model_support": True,
            "allowed_markets": ["match_winner", "over_under_2_5", "btts",
                                "correct_score"],
            "stress_reject_ev": -0.05,
            "strong_ev_multiple": 1.5,
        },
    },
}

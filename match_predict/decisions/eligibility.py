"""Strict no-bet rules (bet.md §4, §8, §21).

`evaluate_selection` takes everything known about one candidate selection at the
decision cutoff and returns a `SelectionDecision`. "No bet" is the default and a
successful output: a selection only becomes QUALIFIED / STRONG EVIDENCE if it
survives every applicable check.

The checks (each maps to a machine-readable RejectionCode from §21):
  * probabilities valid & normalized;               INVALID_PROBABILITIES
  * odds valid & timestamp-safe;                     INVALID_ODDS / ODDS_* / MISSING_ODDS
  * market enabled;                                   MARKET_DISABLED
  * league supported / team resolved / in-dist;       UNSUPPORTED_LEAGUE / UNRESOLVED_TEAM / OUT_OF_DISTRIBUTION
  * data quality;                                     DATA_QUALITY_TOO_LOW
  * model disagreement;                               MODEL_DISAGREEMENT_TOO_HIGH
  * historical sample size;                           INSUFFICIENT_HISTORICAL_SAMPLE
  * band calibration;                                 MODEL_UNCALIBRATED_IN_BAND
  * odds range;                                       PRICE_BELOW_MINIMUM / PRICE_ABOVE_VALIDATED_RANGE
  * model probability floor (score markets);          MODEL_PROBABILITY_TOO_LOW
  * edge >= threshold;                                EDGE_BELOW_THRESHOLD
  * conservative EV > 0;                              NO_POSITIVE_CONSERVATIVE_EV
  * conservative EV >= market threshold;              EDGE_BELOW_THRESHOLD

Correlation and exposure limits are applied AFTER per-selection eligibility, by
the decision engine, and add CORRELATED_SELECTION_ALREADY_CHOSEN /
DAILY_EXPOSURE_LIMIT.
"""
from __future__ import annotations

from . import odds as odds_mod
from . import devig as devig_mod
from . import edge as edge_mod
from . import uncertainty as unc_mod
from .schema import RejectionCode, SelectionDecision, DecisionStatus, Grade


def probabilities_valid(p: dict, tol: float = 0.02) -> bool:
    """Finite, non-negative, and (for exhaustive sets) summing to ~1."""
    vals = list(p.values())
    if not vals:
        return False
    for v in vals:
        if v is None or not _finite(v) or v < -1e-9 or v > 1 + 1e-9:
            return False
    return abs(sum(vals) - 1.0) <= tol


def _finite(x) -> bool:
    try:
        return x == x and abs(float(x)) != float("inf")
    except (TypeError, ValueError):
        return False


def evaluate_selection(
    *,
    market: str,
    selection: str,
    side_or_line: str,
    model_probability: float,
    offered_odds,
    outcome_odds_set,               # decimal odds for the full outcome set (devig)
    decision_cutoff,
    odds_timestamp,
    config: dict,
    market_profile=None,            # MarketValidationProfile or None
    data_quality: float = 1.0,
    model_disagreement: float = 0.0,
    market_fair_probability: float | None = None,
    supported_league: bool = True,
    team_resolved: bool = True,
    in_distribution: bool = True,
    model_version: str | None = None,
    supported_model_version: bool = True,
    feature_schema_ok: bool = True,
    fixture_stale: bool = False,
) -> SelectionDecision:
    """Evaluate ONE candidate selection. Never raises on bad input — it records
    a rejection code instead, because a malformed price must yield "no bet",
    not a crash."""
    reasons: list = []
    evidence: list = []
    warnings: list = []

    mkt_cfg = (config.get("markets", {}) or {}).get(market, {})

    # ---- hard structural gates ------------------------------------------
    if not mkt_cfg or not mkt_cfg.get("enabled", False):
        reasons.append(RejectionCode.MARKET_DISABLED)
    if not supported_league:
        reasons.append(RejectionCode.UNSUPPORTED_LEAGUE)
    if not team_resolved:
        reasons.append(RejectionCode.UNRESOLVED_TEAM)
    if not in_distribution:
        reasons.append(RejectionCode.OUT_OF_DISTRIBUTION)
    if not supported_model_version:
        reasons.append(RejectionCode.UNSUPPORTED_MODEL_VERSION)
    if not feature_schema_ok:
        reasons.append(RejectionCode.FEATURE_SCHEMA_MISMATCH)
    if config.get("require_fresh_fixtures", True) and fixture_stale:
        reasons.append(RejectionCode.FIXTURE_SOURCE_STALE)

    # ---- probability validity -------------------------------------------
    if model_probability is None or not _finite(model_probability) \
            or model_probability < 0 or model_probability > 1:
        reasons.append(RejectionCode.INVALID_PROBABILITIES)

    # ---- odds & timestamp safety ----------------------------------------
    if offered_odds is None:
        reasons.append(RejectionCode.MISSING_ODDS)
    elif not odds_mod.valid_decimal_odds(offered_odds):
        reasons.append(RejectionCode.INVALID_ODDS)
    else:
        reasons.extend(odds_mod.check_odds_timestamp(
            odds_timestamp, decision_cutoff,
            config.get("max_odds_staleness_hours", 24),
            require_timestamp=config.get("require_timestamped_odds", True)))

    # If we cannot even price the bet, stop here with a clean rejection.
    fatal = {RejectionCode.MISSING_ODDS, RejectionCode.INVALID_ODDS,
             RejectionCode.INVALID_PROBABILITIES}
    if fatal & set(reasons):
        return _reject(market, selection, side_or_line, offered_odds,
                       model_probability, market_fair_probability,
                       reasons, evidence, warnings, model_version, odds_timestamp)

    # ---- market fair probability (de-vig) -------------------------------
    if market_fair_probability is None and outcome_odds_set is not None:
        try:
            fair = devig_mod.devig(outcome_odds_set,
                                   config.get("devig_method", "shin"))
            # by convention the candidate selection is the FIRST entry
            market_fair_probability = float(fair[0])
        except (ValueError, IndexError):
            market_fair_probability = None
    if market_fair_probability is None:
        # fall back to raw implied (still lets us report break-even)
        market_fair_probability = odds_mod.offered_implied_probability(offered_odds)
        warnings.append("market fair probability approximated (no outcome set)")

    # ---- uncertainty buffer & conservative probability ------------------
    cal_err = None
    n_samples = None
    min_samples = mkt_cfg.get("minimum_historical_samples",
                              config.get("minimum_historical_samples", 300))
    if market_profile is not None:
        cal_err = market_profile.calibration_error_at(model_probability)
        n_samples = market_profile.samples_at(model_probability)

    buf = unc_mod.uncertainty_buffer(
        config.get("uncertainty", {}), calibration_error=cal_err,
        disagreement=model_disagreement, data_quality=data_quality,
        n_samples=n_samples, min_samples=min_samples)
    cons_p = unc_mod.conservative_probability(model_probability, buf)
    ci = unc_mod.confidence_interval(model_probability, buf)

    # ---- edge / EV -------------------------------------------------------
    es = edge_mod.edge_summary(model_probability, market_fair_probability,
                               cons_p, offered_odds)
    p_edge = es["probability_edge"]
    raw_ev = es["raw_expected_value"]
    cons_ev = es["conservative_expected_value"]

    # ---- quality / validation gates -------------------------------------
    if data_quality < config.get("minimum_data_quality", 0.90):
        reasons.append(RejectionCode.DATA_QUALITY_TOO_LOW)
    if model_disagreement > config.get("maximum_model_disagreement", 0.10):
        reasons.append(RejectionCode.MODEL_DISAGREEMENT_TOO_HIGH)

    if market_profile is None or not market_profile.passed_quality:
        reasons.append(RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE)
    else:
        if n_samples is not None and n_samples < min_samples:
            reasons.append(RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE)
        band = market_profile.band_for(model_probability)
        if band is not None and band.calibration_error > 0.10:
            reasons.append(RejectionCode.MODEL_UNCALIBRATED_IN_BAND)

    # ---- odds-range gate -------------------------------------------------
    min_odds = mkt_cfg.get("minimum_odds", 1.01)
    max_odds = mkt_cfg.get("maximum_odds", 1000.0)
    if offered_odds < min_odds:
        reasons.append(RejectionCode.PRICE_BELOW_MINIMUM)
    if offered_odds > max_odds:
        reasons.append(RejectionCode.PRICE_ABOVE_VALIDATED_RANGE)

    # ---- model-probability floor (score markets) ------------------------
    min_model_p = mkt_cfg.get("minimum_model_probability")
    if min_model_p is not None and model_probability < min_model_p:
        reasons.append(RejectionCode.MODEL_PROBABILITY_TOO_LOW)

    # ---- edge & EV gates -------------------------------------------------
    if p_edge < mkt_cfg.get("minimum_probability_edge", 0.0):
        reasons.append(RejectionCode.EDGE_BELOW_THRESHOLD)
    if cons_ev <= 0:
        reasons.append(RejectionCode.NO_POSITIVE_CONSERVATIVE_EV)
    elif cons_ev < mkt_cfg.get("minimum_conservative_ev", 0.0):
        reasons.append(RejectionCode.EDGE_BELOW_THRESHOLD)

    # ---- supporting evidence (for the report) ---------------------------
    if p_edge > 0:
        evidence.append(f"model above market by {p_edge*100:+.1f} pts")
    if raw_ev > 0:
        evidence.append(f"raw EV {raw_ev*100:+.1f}%")
    if cons_ev > 0:
        evidence.append(f"conservative EV {cons_ev*100:+.1f}% "
                        f"(after {buf*100:.1f}pt buffer)")
    if market_profile is not None and market_profile.passed_quality:
        evidence.append(f"market passed out-of-time quality "
                        f"(n={market_profile.n_samples})")

    # ---- assemble --------------------------------------------------------
    dec = SelectionDecision(
        selection=selection, market=market, side_or_line=side_or_line,
        offered_odds=float(offered_odds),
        model_probability=round(float(model_probability), 4),
        market_fair_probability=round(float(market_fair_probability), 4),
        conservative_probability=round(float(cons_p), 4),
        probability_edge=round(float(p_edge), 4),
        raw_expected_value=round(float(raw_ev), 4),
        conservative_expected_value=round(float(cons_ev), 4),
        confidence_interval=ci, grade=Grade.REJECT.value,
        decision_status=DecisionStatus.NO_BET.value,
        rejection_reasons=list(dict.fromkeys(reasons)),  # dedupe, keep order
        supporting_evidence=evidence, risk_warnings=warnings,
        model_version=model_version, odds_timestamp=(
            str(odds_timestamp) if odds_timestamp is not None else None),
    )
    return dec


def _reject(market, selection, side_or_line, offered_odds, model_p, market_p,
            reasons, evidence, warnings, model_version, odds_timestamp
            ) -> SelectionDecision:
    return SelectionDecision(
        selection=selection, market=market, side_or_line=side_or_line,
        offered_odds=(float(offered_odds)
                      if odds_mod.valid_decimal_odds(offered_odds) else None),
        model_probability=(round(float(model_p), 4)
                           if model_p is not None and _finite(model_p) else None),
        market_fair_probability=market_p,
        conservative_probability=None, probability_edge=None,
        raw_expected_value=None, conservative_expected_value=None,
        confidence_interval=None, grade=Grade.REJECT.value,
        decision_status=DecisionStatus.NO_BET.value,
        rejection_reasons=list(dict.fromkeys(reasons)),
        supporting_evidence=evidence, risk_warnings=warnings,
        model_version=model_version,
        odds_timestamp=(str(odds_timestamp) if odds_timestamp is not None else None),
    )

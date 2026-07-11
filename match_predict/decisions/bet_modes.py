"""Smart Bet & High Return Bet decision modes (bet-funcuanlty).

Two user-selectable modes sit ON TOP of the advisory decision engine. They do
NOT build a new prediction model and they never invent probabilities: each mode
consumes the SAME validated `SelectionDecision` outputs produced by
`decision_engine.evaluate_match` (calibrated model probability, de-vigged market
fair probability, conservative probability, edge, EV, confidence interval) and
simply applies its own thresholds, transparent scoring and risk criteria.

  * Smart Bet     — favours strong calibrated probability, low uncertainty, good
                    model agreement and moderate odds. Rejects weak edges and
                    stale / poorly-calibrated candidates. Prefers probability
                    quality over payout.
  * High Return   — may take higher odds and lower win probability, but demands a
                    larger positive conservative EV, multiple independent model
                    supports, and survival of a probability stress test. It never
                    selects a price only because it is large. Correct score is
                    only ever eligible here, and only when its market validation
                    profile passes.

Every candidate reports a transparent 0-100 score with its components exposed
(never a probability), a decision status (NO BET / WATCHLIST / QUALIFIED /
STRONG EVIDENCE), and the full §18 output schema. "No bet" is a normal output;
neither mode is forced to pick a selection.

A high-return selection usually LOSES more often than it wins — the value is in
the price, not the hit rate. That is stated in every high-return report.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .schema import DecisionStatus, RejectionCode

MODE_SMART = "smart"
MODE_HIGH_RETURN = "high_return"

# Structural rejections from the base engine that no threshold change can undo;
# a bet mode inherits them verbatim (bet.md §8 hard gates).
_HARD_INHERITED = {
    RejectionCode.ODDS_STALE, RejectionCode.ODDS_IN_FUTURE,
    RejectionCode.FIXTURE_SOURCE_STALE, RejectionCode.UNRESOLVED_TEAM,
    RejectionCode.UNSUPPORTED_LEAGUE, RejectionCode.OUT_OF_DISTRIBUTION,
    RejectionCode.INVALID_PROBABILITIES, RejectionCode.INVALID_ODDS,
    RejectionCode.MISSING_ODDS, RejectionCode.UNSUPPORTED_MODEL_VERSION,
    RejectionCode.FEATURE_SCHEMA_MISMATCH, RejectionCode.MODEL_UNCALIBRATED_IN_BAND,
}

# Soft codes: a positive but sub-threshold edge/EV → WATCHLIST, not a hard reject.
_SOFT = {RejectionCode.EDGE_BELOW_THRESHOLD}


@dataclass
class ModeSelection:
    """The full §18 bet-mode output for ONE selection under ONE mode."""
    bet_mode: str
    fixture_id: str
    league: str
    home_team: str
    away_team: str
    kickoff: str
    selection: str
    market: str
    line: str | None
    offered_odds: float | None
    model_probability: float | None
    market_probability: float | None
    hybrid_probability: float | None
    pure_probability: float | None
    conservative_probability: float | None
    break_even_probability: float | None
    probability_edge: float | None
    raw_expected_value: float | None
    conservative_expected_value: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    smart_bet_score: float | None
    high_return_score: float | None
    decision_status: str
    rejection_reasons: list = field(default_factory=list)
    model_support_count: int = 0
    stress_test_results: list = field(default_factory=list)
    score_components: dict = field(default_factory=dict)
    data_quality: float | None = None
    model_disagreement: float | None = None
    historical_sample_size: int | None = None
    model_version: str | None = None
    odds_timestamp: str | None = None
    prediction_timestamp: str | None = None
    supporting_evidence: list = field(default_factory=list)
    risk_warnings: list = field(default_factory=list)
    is_primary: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rejection_reasons"] = [
            r.value if isinstance(r, RejectionCode) else r
            for r in self.rejection_reasons]
        return d


# --------------------------------------------------------------------------- #
# Transparent scoring (bet-funcuanlty §7, §8). Each component is in [0, 1] and  #
# the score is 100 × product of components. A score is NOT a win probability.   #
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def _label(v: float) -> str:
    if v < 0.34:
        return "Weak"
    if v < 0.56:
        return "Moderate"
    if v < 0.76:
        return "Acceptable"
    return "Strong"


def _components(mode: str, *, model_p, edge, cons_ev, disagreement,
                max_disagreement, data_quality, calibration_error,
                n_samples, min_samples, min_model_p) -> dict:
    """Named components in [0, 1] with a plain-language label each."""
    prob_q = _clamp01((model_p - min_model_p) / max(0.85 - min_model_p, 1e-6))
    cons_ev_c = _clamp01((cons_ev or 0.0) / (0.20 if mode == MODE_HIGH_RETURN else 0.10))
    edge_c = _clamp01((edge or 0.0) / 0.15)
    calib_c = _clamp01(1.0 - (calibration_error or 0.02) / 0.10)
    agree_c = _clamp01(1.0 - disagreement / max(max_disagreement, 1e-6))
    dq_c = _clamp01((data_quality - 0.85) / 0.15)
    stab_c = _clamp01((n_samples or 0) / float(2 * max(min_samples, 1)))

    if mode == MODE_HIGH_RETURN:
        comp = {"expected_value": cons_ev_c, "edge": edge_c,
                "calibration": calib_c, "model_agreement": agree_c,
                "data_quality": dq_c, "historical_stability": stab_c}
    else:
        comp = {"probability_quality": prob_q, "conservative_ev": cons_ev_c,
                "calibration": calib_c, "model_agreement": agree_c,
                "data_quality": dq_c, "historical_stability": stab_c}
    return {k: {"value": round(v, 3), "label": _label(v)} for k, v in comp.items()}


def _score(components: dict) -> float:
    """Geometric mean of the components, scaled to 0-100.

    The product form (bet-funcuanlty §7, §8) means a single weak component drags
    the whole score down — you cannot buy a high score with odds alone. Taking
    the n-th root keeps the result on a readable 0-100 scale while preserving
    that "weakest link" behaviour. It is a research score, never a probability.
    """
    vals = [c["value"] for c in components.values()]
    if not vals:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= max(v, 0.0)
    return round(100.0 * prod ** (1.0 / len(vals)), 1)


def _stress_tests(model_p: float, offered_odds: float,
                  calibration_error: float | None) -> list:
    """EV under modest probability reductions (bet-funcuanlty §10).

    Reports conservative EV if the model probability were 1pt lower, 2pt lower,
    and one calibration-error lower. A high-odds candidate that turns strongly
    negative under a modest cut must not qualify.
    """
    deltas = [("minus_1pt", 0.01), ("minus_2pt", 0.02),
              ("minus_calibration_error", calibration_error or 0.02)]
    out = []
    for name, d in deltas:
        p = max(0.0, model_p - d)
        out.append({"scenario": name, "delta": round(d, 4),
                    "stressed_probability": round(p, 4),
                    "expected_value": round(p * offered_odds - 1.0, 4)})
    return out


# --------------------------------------------------------------------------- #
# Per-selection mode evaluation.                                               #
# --------------------------------------------------------------------------- #
def evaluate_selection_for_mode(dec, mode: str, mode_cfg: dict, *,
                                fixture_id: str, league: str, home_team: str,
                                away_team: str, kickoff: str,
                                data_quality: float, model_disagreement: float,
                                pure_probability: float | None,
                                calibration_error: float | None,
                                n_samples: int | None,
                                profile_passed: bool,
                                prediction_timestamp: str | None = None
                                ) -> ModeSelection:
    """Re-threshold ONE base `SelectionDecision` under a mode profile."""
    hybrid_p = dec.model_probability
    market_p = dec.market_fair_probability
    ci = dec.confidence_interval
    ci_lo = ci[0] if ci else None
    ci_hi = ci[1] if ci else None
    be = (1.0 / dec.offered_odds) if dec.offered_odds else None

    ms = ModeSelection(
        bet_mode=mode, fixture_id=fixture_id, league=league,
        home_team=home_team, away_team=away_team, kickoff=kickoff,
        selection=dec.selection, market=dec.market, line=dec.side_or_line,
        offered_odds=dec.offered_odds, model_probability=hybrid_p,
        market_probability=market_p, hybrid_probability=hybrid_p,
        pure_probability=(round(pure_probability, 4)
                          if pure_probability is not None else None),
        conservative_probability=dec.conservative_probability,
        break_even_probability=(round(be, 4) if be is not None else None),
        probability_edge=dec.probability_edge,
        raw_expected_value=dec.raw_expected_value,
        conservative_expected_value=dec.conservative_expected_value,
        confidence_interval_lower=ci_lo, confidence_interval_upper=ci_hi,
        smart_bet_score=None, high_return_score=None,
        decision_status=DecisionStatus.NO_BET.value,
        data_quality=round(data_quality, 4),
        model_disagreement=round(model_disagreement, 4),
        historical_sample_size=n_samples, model_version=dec.model_version,
        odds_timestamp=dec.odds_timestamp,
        prediction_timestamp=prediction_timestamp)

    # Unpriceable base decision (missing/invalid odds) → inherit NO BET.
    if dec.conservative_expected_value is None or dec.offered_odds is None:
        ms.rejection_reasons = list(dec.rejection_reasons)
        return ms

    hard = [r for r in dec.rejection_reasons if r in _HARD_INHERITED]
    soft: list = []

    if dec.market not in mode_cfg.get("allowed_markets", []):
        hard.append(RejectionCode.MARKET_NOT_IN_MODE)
    if dec.offered_odds < mode_cfg["minimum_odds"]:
        hard.append(RejectionCode.PRICE_BELOW_MINIMUM)
    if dec.offered_odds > mode_cfg["maximum_odds"]:
        hard.append(RejectionCode.PRICE_ABOVE_VALIDATED_RANGE)
    if hybrid_p < mode_cfg["minimum_model_probability"]:
        hard.append(RejectionCode.MODEL_PROBABILITY_TOO_LOW)
    if data_quality < mode_cfg["minimum_data_quality"]:
        hard.append(RejectionCode.DATA_QUALITY_TOO_LOW)
    if model_disagreement > mode_cfg["maximum_model_disagreement"]:
        hard.append(RejectionCode.MODEL_DISAGREEMENT_TOO_HIGH)
    if not profile_passed or (n_samples or 0) < mode_cfg["minimum_historical_samples"]:
        hard.append(RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE)

    # model support: how many independent views beat the market (bet-funcuanlty §12).
    hybrid_supports = dec.probability_edge is not None and dec.probability_edge > 0
    pure_supports = (pure_probability is not None and market_p is not None
                     and (pure_probability - market_p) > 0)
    support = int(hybrid_supports) + int(pure_supports)
    ms.model_support_count = support
    if mode == MODE_HIGH_RETURN:
        if support < mode_cfg.get("require_model_support_count", 1):
            hard.append(RejectionCode.MODE_SUPPORT_TOO_LOW)
        if mode_cfg.get("require_pure_model_support", False) and not pure_supports:
            hard.append(RejectionCode.NO_PURE_MODEL_SUPPORT)

    # edge / EV thresholds ------------------------------------------------
    if dec.conservative_expected_value <= 0:
        hard.append(RejectionCode.NO_POSITIVE_CONSERVATIVE_EV)
    else:
        if (dec.probability_edge or 0) < mode_cfg["minimum_probability_edge"]:
            soft.append(RejectionCode.EDGE_BELOW_THRESHOLD)
        if dec.conservative_expected_value < mode_cfg["minimum_conservative_ev"]:
            soft.append(RejectionCode.EDGE_BELOW_THRESHOLD)

    # lower confidence bound EV must be positive (bet-funcuanlty §4/§5) ----
    lcb_ev = None
    if ci_lo is not None:
        lcb_ev = ci_lo * dec.offered_odds - 1.0
        if mode_cfg.get("require_positive_lower_confidence_bound") and lcb_ev <= 0:
            soft.append(RejectionCode.EDGE_BELOW_THRESHOLD)

    # stress tests (bet-funcuanlty §10) -----------------------------------
    ms.stress_test_results = _stress_tests(hybrid_p, dec.offered_odds,
                                           calibration_error)
    modest_ev = ms.stress_test_results[1]["expected_value"]  # -2pt
    if mode == MODE_HIGH_RETURN and modest_ev < mode_cfg.get("stress_reject_ev", -0.05):
        hard.append(RejectionCode.STRESS_TEST_FAILED)

    reasons = list(dict.fromkeys(hard + soft))
    ms.rejection_reasons = reasons

    # transparent score (computed even for rejected candidates, for the report)
    comps = _components(
        mode, model_p=hybrid_p, edge=dec.probability_edge,
        cons_ev=dec.conservative_expected_value, disagreement=model_disagreement,
        max_disagreement=mode_cfg["maximum_model_disagreement"],
        data_quality=data_quality, calibration_error=calibration_error,
        n_samples=n_samples, min_samples=mode_cfg["minimum_historical_samples"],
        min_model_p=mode_cfg["minimum_model_probability"])
    ms.score_components = comps
    score = _score(comps)
    if mode == MODE_HIGH_RETURN:
        ms.high_return_score = score
    else:
        ms.smart_bet_score = score

    # decision status -----------------------------------------------------
    hard_fail = [r for r in reasons if r not in _SOFT]
    if hard_fail:
        ms.decision_status = DecisionStatus.NO_BET.value
    elif any(r in _SOFT for r in reasons):
        # positive-but-uncertain → watchlist only if the raw signal is positive
        if (dec.raw_expected_value or 0) > 0 and (dec.probability_edge or 0) > 0:
            ms.decision_status = DecisionStatus.WATCHLIST.value
        else:
            ms.decision_status = DecisionStatus.NO_BET.value
    else:
        strong = _is_strong(dec, mode_cfg, support, modest_ev,
                            calibration_error, lcb_ev)
        ms.decision_status = (DecisionStatus.STRONG_EVIDENCE.value if strong
                              else DecisionStatus.QUALIFIED.value)

    ms.supporting_evidence = _evidence(dec, support, pure_supports)
    if mode == MODE_HIGH_RETURN and ms.decision_status in (
            DecisionStatus.QUALIFIED.value, DecisionStatus.STRONG_EVIDENCE.value):
        ms.risk_warnings.append(
            "High-return selection: more likely to LOSE than win — the value is "
            "in the price, not the hit rate.")
    return ms


def _is_strong(dec, mode_cfg, support, modest_stress_ev, calibration_error,
               lcb_ev) -> bool:
    """Stricter bar: comfortable EV, 2 supports, stress-positive, well calibrated."""
    min_ev = mode_cfg["minimum_conservative_ev"]
    if dec.conservative_expected_value < min_ev * mode_cfg.get("strong_ev_multiple", 1.5):
        return False
    if support < 2:
        return False
    if modest_stress_ev <= 0:            # edge must survive a 2pt cut
        return False
    if lcb_ev is not None and lcb_ev <= 0:
        return False
    if calibration_error is not None and calibration_error > 0.05:
        return False
    return True


def _evidence(dec, support, pure_supports) -> list:
    ev = []
    if dec.probability_edge and dec.probability_edge > 0:
        ev.append(f"model above market by {dec.probability_edge*100:+.1f} pts")
    if dec.conservative_expected_value and dec.conservative_expected_value > 0:
        ev.append(f"conservative EV {dec.conservative_expected_value*100:+.1f}%")
    ev.append(f"{support} independent model support(s)")
    if pure_supports:
        ev.append("pure model independently beats the market")
    return ev


# --------------------------------------------------------------------------- #
# Match- and day-level aggregation, ranking and exposure caps.                 #
# --------------------------------------------------------------------------- #
def _rank_key(ms: ModeSelection):
    """Ranking is mode-specific (bet-funcuanlty §14); score already blends the
    right priorities, EV breaks ties."""
    score = ms.high_return_score if ms.bet_mode == MODE_HIGH_RETURN else ms.smart_bet_score
    return (-(score or 0.0), -(ms.conservative_expected_value or 0.0))


def _pure_prob_for(dec, views, pure_probs):
    if pure_probs and dec.selection in pure_probs:
        return pure_probs[dec.selection]
    if dec.selection in ("H", "D", "A"):
        return (views.pure or {}).get(dec.selection)
    return None


def evaluate_match_for_mode(match_decision, mode: str, config: dict,
                            market_profiles: dict | None = None,
                            pure_probs: dict | None = None) -> list:
    """All `ModeSelection`s for one fixture under one mode."""
    mode_cfg = (config.get("bet_modes", {}) or {}).get(mode)
    if not mode_cfg or not mode_cfg.get("enabled", False):
        return []
    market_profiles = market_profiles or {}
    m = match_decision
    out = []
    for dec in m.selections:
        prof = market_profiles.get(dec.market)
        cal_err = (prof.calibration_error_at(dec.model_probability)
                   if prof is not None and dec.model_probability is not None else None)
        n_samples = (prof.samples_at(dec.model_probability)
                     if prof is not None and dec.model_probability is not None else None)
        out.append(evaluate_selection_for_mode(
            dec, mode, mode_cfg, fixture_id=m.fixture_id, league=m.league,
            home_team=m.home_team, away_team=m.away_team, kickoff=m.kickoff,
            data_quality=(m.data_quality if m.data_quality is not None else 1.0),
            model_disagreement=(m.model_disagreement or 0.0),
            pure_probability=_pure_prob_for(dec, m.views, pure_probs),
            calibration_error=cal_err, n_samples=n_samples,
            profile_passed=(prof is not None and prof.passed_quality),
            prediction_timestamp=m.decision_time))
    return out


_QUALIFIED = {DecisionStatus.QUALIFIED.value, DecisionStatus.STRONG_EVIDENCE.value}


def apply_exposure_and_rank(mode_selections: list, mode_cfg: dict) -> list:
    """Enforce one primary per match + a daily qualified cap, then rank.

    Correlated / over-cap qualified selections are demoted to WATCHLIST with a
    DAILY_EXPOSURE_LIMIT / CORRELATED_SELECTION_ALREADY_CHOSEN reason rather than
    silently dropped, so the report stays honest (bet-funcuanlty §15).
    """
    ranked = sorted(mode_selections, key=_rank_key)
    max_per_match = mode_cfg.get("maximum_primary_selections_per_match", 1)
    max_per_day = mode_cfg.get("maximum_qualified_selections_per_day", 3)

    per_match: dict = {}
    day_count = 0
    for ms in ranked:
        if ms.decision_status not in _QUALIFIED:
            continue
        used = per_match.get(ms.fixture_id, 0)
        if used >= max_per_match:
            ms.decision_status = DecisionStatus.WATCHLIST.value
            ms.rejection_reasons.append(
                RejectionCode.CORRELATED_SELECTION_ALREADY_CHOSEN)
            continue
        if day_count >= max_per_day:
            ms.decision_status = DecisionStatus.WATCHLIST.value
            ms.rejection_reasons.append(RejectionCode.DAILY_EXPOSURE_LIMIT)
            continue
        ms.is_primary = True
        per_match[ms.fixture_id] = used + 1
        day_count += 1
    return ranked


def evaluate_mode(match_decisions: list, mode: str, config: dict,
                  market_profiles: dict | None = None,
                  pure_probs_by_fixture: dict | None = None) -> dict:
    """Evaluate a set of matches under one mode; return ranked selections + summary."""
    pure_by_fx = pure_probs_by_fixture or {}
    sels: list = []
    for m in match_decisions:
        sels.extend(evaluate_match_for_mode(
            m, mode, config, market_profiles,
            pure_probs=pure_by_fx.get(m.fixture_id)))
    mode_cfg = (config.get("bet_modes", {}) or {}).get(mode, {})
    ranked = apply_exposure_and_rank(sels, mode_cfg)
    return {"mode": mode, "selections": ranked, "summary": summarize_mode(ranked)}


def summarize_mode(selections: list) -> dict:
    """Research summary for one mode (bet-funcuanlty §16 subset at decision time)."""
    def _count(status):
        return sum(1 for s in selections if s.decision_status == status)
    qualified = [s for s in selections if s.decision_status in _QUALIFIED]
    evaluated = len(selections)
    odds = [s.offered_odds for s in qualified if s.offered_odds]
    probs = [s.model_probability for s in qualified if s.model_probability is not None]
    edges = [s.probability_edge for s in qualified if s.probability_edge is not None]
    cevs = [s.conservative_expected_value for s in qualified
            if s.conservative_expected_value is not None]
    rej: dict = {}
    for s in selections:
        for r in s.rejection_reasons:
            code = r.value if hasattr(r, "value") else r
            rej[code] = rej.get(code, 0) + 1

    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    return {
        "selections_evaluated": evaluated,
        "qualified": len(qualified),
        "watchlist": _count(DecisionStatus.WATCHLIST.value),
        "no_bet": _count(DecisionStatus.NO_BET.value),
        "no_bet_rate": round(_count(DecisionStatus.NO_BET.value) / evaluated, 4)
        if evaluated else None,
        "average_odds": _avg(odds),
        "average_model_probability": _avg(probs),
        "average_probability_edge": _avg(edges),
        "average_conservative_ev": _avg(cevs),
        "rejection_reason_counts": dict(sorted(rej.items(), key=lambda kv: -kv[1])),
    }

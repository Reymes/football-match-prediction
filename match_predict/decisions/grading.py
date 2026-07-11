"""Conservative grading (bet.md §9).

A selection that survives eligibility is graded on the STRENGTH of the evidence,
never on marketing language. Grades:

  * Reject          — one or more critical checks failed (has rejection codes).
  * Watchlist       — model/market disagreement exists but uncertainty is too
                      high to qualify (e.g. positive raw EV but conservative EV
                      not comfortably above the market threshold, or a thin but
                      non-fatal sample).
  * Qualified       — passes all required validation & edge thresholds.
  * Strong Evidence — passes stricter thresholds AND is supported by good data
                      quality, low model disagreement, a comfortable sample, and
                      an edge that survives a doubled uncertainty buffer.

Banned words (guaranteed / certain / lock / banker / safe win / fixed / 100%)
never appear anywhere. A high grade still means the selection can lose.
"""
from __future__ import annotations

from .schema import Grade, DecisionStatus, RejectionCode


_STRONG_EV_MULT = 1.5          # stricter conservative-EV multiple for Strong
_WATCHLIST_MIN_RAW_EV = 0.0    # any positive raw EV is at least watchlist-worthy


def grade_selection(dec, config: dict, market_profile=None):
    """Mutate `dec` in place with a grade & decision status, returning it."""
    mkt_cfg = (config.get("markets", {}) or {}).get(dec.market, {})

    if dec.rejection_reasons:
        # If the ONLY problem is that conservative EV missed the stricter market
        # threshold (edge below threshold) but raw EV is positive and there is
        # real disagreement, surface it as WATCHLIST rather than a hard reject.
        reasons = set(dec.rejection_reasons)
        soft = {RejectionCode.EDGE_BELOW_THRESHOLD,
                RejectionCode.NO_POSITIVE_CONSERVATIVE_EV}
        only_soft = reasons.issubset(soft)
        if (only_soft and dec.raw_expected_value is not None
                and dec.raw_expected_value > _WATCHLIST_MIN_RAW_EV
                and dec.probability_edge is not None
                and dec.probability_edge > 0):
            dec.grade = Grade.WATCHLIST.value
            dec.decision_status = DecisionStatus.WATCHLIST.value
            return dec
        dec.grade = Grade.REJECT.value
        dec.decision_status = DecisionStatus.NO_BET.value
        return dec

    # No rejection codes: it passed the required gates -> at least Qualified.
    min_ev = mkt_cfg.get("minimum_conservative_ev", 0.0)
    strong = _is_strong(dec, config, mkt_cfg, market_profile, min_ev)
    if strong:
        dec.grade = Grade.STRONG_EVIDENCE.value
        dec.decision_status = DecisionStatus.STRONG_EVIDENCE.value
    else:
        dec.grade = Grade.QUALIFIED.value
        dec.decision_status = DecisionStatus.QUALIFIED.value
    return dec


def _is_strong(dec, config, mkt_cfg, market_profile, min_ev) -> bool:
    if dec.conservative_expected_value is None:
        return False
    # 1) conservative EV clears a stricter multiple of the market threshold
    if dec.conservative_expected_value < max(min_ev * _STRONG_EV_MULT, min_ev + 0.01):
        return False
    # 2) edge survives DOUBLING the uncertainty buffer implied by the CI
    if dec.confidence_interval is not None and dec.offered_odds:
        buffer = dec.model_probability - dec.confidence_interval[0]
        double_cons_p = max(0.0, dec.model_probability - 2 * buffer)
        if double_cons_p * dec.offered_odds - 1.0 <= 0:
            return False
    # 3) validation passed with a comfortable sample and good calibration
    if market_profile is None or not market_profile.passed_quality:
        return False
    min_samples = mkt_cfg.get("minimum_historical_samples", 300)
    if market_profile.samples_at(dec.model_probability) < 2 * min_samples:
        return False
    band = market_profile.band_for(dec.model_probability)
    if band is not None and band.calibration_error > 0.05:
        return False
    return True

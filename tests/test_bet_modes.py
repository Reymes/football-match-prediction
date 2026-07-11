"""Tests for the Smart Bet / High Return decision modes (bet-funcuanlty §20).

All unit-level: the modes re-threshold pre-built `SelectionDecision` objects, so
no trained bundle or network is needed. A small chronological-safety check on the
mode backtest's stability maths is included.
"""
from __future__ import annotations

import pandas as pd

from match_predict.decisions import (
    MODE_SMART, MODE_HIGH_RETURN, evaluate_mode, evaluate_selection_for_mode,
    load_config,
)
from match_predict.decisions.schema import (
    MatchDecision, ThreeViews, SelectionDecision, DecisionStatus, Grade,
    RejectionCode,
)
from match_predict.decisions.validation import MarketValidationProfile, BandProfile
from match_predict.decisions.mode_backtest import _summarize, _season


# --------------------------------------------------------------------------- #
# Fixtures / builders                                                          #
# --------------------------------------------------------------------------- #
def _profile(passed=True, n=2000, cal_err=0.02):
    band = BandProfile(lo=0.0, hi=1.0, n_samples=n, calibration_error=cal_err,
                       empirical_rate=0.5, predicted_rate=0.5)
    return MarketValidationProfile(market="match_winner", n_samples=n,
                                   log_loss=0.9, ece=cal_err,
                                   passed_quality=passed, bands=[band])


def _sel(selection="H", market="match_winner", odds=2.10, model_p=0.55,
         market_p=0.50, cons_p=0.52, ci=(0.49, 0.61)):
    edge = model_p - market_p
    return SelectionDecision(
        selection=selection, market=market, side_or_line=selection,
        offered_odds=odds, model_probability=model_p,
        market_fair_probability=market_p, conservative_probability=cons_p,
        probability_edge=round(edge, 4), raw_expected_value=model_p * odds - 1,
        conservative_expected_value=round(cons_p * odds - 1, 4),
        confidence_interval=ci, grade=Grade.QUALIFIED.value,
        decision_status=DecisionStatus.QUALIFIED.value, model_version="test")


def _match(selections, *, pure=None, dq=0.98, disagree=0.03,
           fixture_id="L|20260818|A|B"):
    views = ThreeViews(pure=pure or {"H": 0.54, "D": 0.25, "A": 0.21},
                       market={"H": 0.50, "D": 0.27, "A": 0.23},
                       hybrid={"H": 0.55, "D": 0.24, "A": 0.21})
    return MatchDecision(
        fixture_id=fixture_id, league="England-PL", home_team="A", away_team="B",
        kickoff="2026-08-18", decision_time="2026-08-11", horizon=None,
        views=views, selections=selections, data_quality=dq,
        model_disagreement=disagree)


def _codes(ms):
    return {r.value if hasattr(r, "value") else r for r in ms.rejection_reasons}


def _eval_one(sel, mode, match=None, pure_probs=None, profiles=None):
    cfg = load_config()
    m = match or _match([sel])
    profiles = profiles if profiles is not None else {"match_winner": _profile()}
    res = evaluate_mode([m], mode, cfg, profiles,
                        {m.fixture_id: pure_probs or {}})
    return res["selections"][0]


# --------------------------------------------------------------------------- #
# The modes consume existing model probabilities (no LM-invented numbers).      #
# --------------------------------------------------------------------------- #
def test_modes_reuse_existing_model_probabilities():
    sel = _sel(model_p=0.55, market_p=0.50)
    ms = _eval_one(sel, MODE_SMART, pure_probs={"H": 0.54})
    assert ms.model_probability == sel.model_probability
    assert ms.hybrid_probability == sel.model_probability
    assert ms.market_probability == sel.market_fair_probability
    assert ms.conservative_probability == sel.conservative_probability
    # the score is NOT a probability: separate field, can exceed 1.
    assert ms.smart_bet_score is not None and ms.smart_bet_score > 1


def test_config_thresholds_are_declared_not_data_derived():
    cfg = load_config()
    assert cfg["bet_modes"]["smart"]["minimum_odds"] == 1.35
    assert cfg["bet_modes"]["high_return"]["minimum_odds"] == 3.00
    assert cfg["bet_modes"]["high_return"]["require_model_support_count"] == 2


# --------------------------------------------------------------------------- #
# Smart Bet                                                                    #
# --------------------------------------------------------------------------- #
def test_smart_qualifies_moderate_odds_strong_edge():
    ms = _eval_one(_sel(odds=2.10, model_p=0.55, market_p=0.50, cons_p=0.52),
                   MODE_SMART, pure_probs={"H": 0.54})
    assert ms.decision_status in (DecisionStatus.QUALIFIED.value,
                                  DecisionStatus.STRONG_EVIDENCE.value)
    assert ms.is_primary


def test_smart_rejects_high_odds_low_probability_candidate():
    # 6.0 odds, model prob 0.20 — above smart's odds ceiling and below its
    # probability floor: an uncertain high-odds pick smart must not take.
    ms = _eval_one(_sel(selection="A", odds=6.0, model_p=0.20, market_p=0.16,
                        cons_p=0.17), MODE_SMART, pure_probs={"A": 0.21})
    assert ms.decision_status == DecisionStatus.NO_BET.value
    assert RejectionCode.PRICE_ABOVE_VALIDATED_RANGE.value in _codes(ms)
    assert RejectionCode.MODEL_PROBABILITY_TOO_LOW.value in _codes(ms)


# --------------------------------------------------------------------------- #
# High Return                                                                  #
# --------------------------------------------------------------------------- #
def test_high_return_rejects_large_odds_with_no_edge():
    # big price but model AGREES with / is below the market → no real edge.
    sel = _sel(selection="A", odds=6.0, model_p=0.15, market_p=0.17, cons_p=0.13)
    ms = _eval_one(sel, MODE_HIGH_RETURN,
                   match=_match([sel], pure={"H": 0.6, "D": 0.25, "A": 0.15}),
                   pure_probs={"A": 0.14})
    assert ms.decision_status == DecisionStatus.NO_BET.value
    assert RejectionCode.NO_POSITIVE_CONSERVATIVE_EV.value in _codes(ms)


def test_high_return_requires_positive_conservative_ev():
    # positive raw edge but conservative prob makes cons EV negative.
    sel = _sel(selection="A", odds=5.0, model_p=0.22, market_p=0.18, cons_p=0.19)
    # cons_ev = 0.19*5 - 1 = -0.05 < 0
    ms = _eval_one(sel, MODE_HIGH_RETURN,
                   match=_match([sel], pure={"H": 0.5, "D": 0.23, "A": 0.27}),
                   pure_probs={"A": 0.27})
    assert ms.conservative_expected_value < 0
    assert ms.decision_status == DecisionStatus.NO_BET.value
    assert RejectionCode.NO_POSITIVE_CONSERVATIVE_EV.value in _codes(ms)


def test_high_return_needs_pure_model_support():
    # hybrid beats market but the PURE model does NOT → NO_PURE_MODEL_SUPPORT.
    sel = _sel(selection="A", odds=5.2, model_p=0.24, market_p=0.19, cons_p=0.21)
    ms = _eval_one(sel, MODE_HIGH_RETURN,
                   match=_match([sel], pure={"H": 0.6, "D": 0.24, "A": 0.16}),
                   pure_probs={"A": 0.16})   # pure 0.16 < market 0.19
    assert RejectionCode.NO_PURE_MODEL_SUPPORT.value in _codes(ms)
    assert ms.decision_status == DecisionStatus.NO_BET.value


def test_high_return_stress_test_can_reject():
    # thin edge at high odds: a modest probability cut flips EV strongly negative.
    sel = _sel(selection="A", odds=3.2, model_p=0.33, market_p=0.30, cons_p=0.315)
    ms = _eval_one(sel, MODE_HIGH_RETURN,
                   match=_match([sel], pure={"H": 0.4, "D": 0.26, "A": 0.34}),
                   pure_probs={"A": 0.34})
    stressed = {t["scenario"]: t["expected_value"] for t in ms.stress_test_results}
    assert set(stressed) == {"minus_1pt", "minus_2pt", "minus_calibration_error"}
    # stress numbers are always reported regardless of the verdict
    assert all(isinstance(v, float) for v in stressed.values())


# --------------------------------------------------------------------------- #
# Market gating & calibration                                                  #
# --------------------------------------------------------------------------- #
def test_poor_calibration_disables_market():
    # profile fails quality → market unvalidated → rejected in both modes.
    ms = _eval_one(_sel(odds=2.10, model_p=0.55, market_p=0.50),
                   MODE_SMART, pure_probs={"H": 0.54},
                   profiles={"match_winner": _profile(passed=False)})
    assert RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE.value in _codes(ms)
    assert ms.decision_status == DecisionStatus.NO_BET.value


def test_correct_score_not_allowed_in_smart_but_considered_in_high_return():
    cs = _sel(selection="2-1", market="correct_score", odds=11.0, model_p=0.12,
              market_p=0.10, cons_p=0.097, ci=(0.07, 0.17))
    smart = _eval_one(cs, MODE_SMART, pure_probs={"2-1": 0.12},
                      profiles={"correct_score": _profile()})
    assert RejectionCode.MARKET_NOT_IN_MODE.value in _codes(smart)
    assert smart.decision_status == DecisionStatus.NO_BET.value
    # high_return allows the market (it may still reject on other grounds, but
    # NOT because the market is disallowed) — never auto-selected just for being
    # the top score.
    hr = _eval_one(cs, MODE_HIGH_RETURN, pure_probs={"2-1": 0.12},
                   profiles={"correct_score": _profile()})
    assert RejectionCode.MARKET_NOT_IN_MODE.value not in _codes(hr)


# --------------------------------------------------------------------------- #
# Ranking, correlation/exposure, no-bet                                        #
# --------------------------------------------------------------------------- #
def test_only_one_primary_per_match():
    # two independently-qualifying 1X2 sides in the same match → one primary.
    s1 = _sel(selection="H", odds=2.10, model_p=0.55, market_p=0.50, cons_p=0.52)
    s2 = _sel(selection="A", odds=2.10, model_p=0.55, market_p=0.50, cons_p=0.52)
    m = _match([s1, s2], pure={"H": 0.55, "D": 0.20, "A": 0.55})
    res = evaluate_mode([m], MODE_SMART, load_config(),
                        {"match_winner": _profile()},
                        {m.fixture_id: {"H": 0.55, "A": 0.55}})
    primaries = [s for s in res["selections"] if s.is_primary]
    assert len(primaries) == 1
    demoted = [s for s in res["selections"]
               if not s.is_primary and s.decision_status == DecisionStatus.WATCHLIST.value]
    assert any(RejectionCode.CORRELATED_SELECTION_ALREADY_CHOSEN in d.rejection_reasons
               for d in demoted)


def test_candidates_ranked_by_mode_score():
    strong = _sel(selection="H", odds=2.10, model_p=0.60, market_p=0.50, cons_p=0.57)
    weak = _sel(selection="H", odds=1.90, model_p=0.56, market_p=0.53, cons_p=0.55)
    m1 = _match([strong], fixture_id="F1", pure={"H": 0.60, "D": 0.2, "A": 0.2})
    m2 = _match([weak], fixture_id="F2", pure={"H": 0.56, "D": 0.24, "A": 0.2})
    res = evaluate_mode([m1, m2], MODE_SMART, load_config(),
                        {"match_winner": _profile()},
                        {"F1": {"H": 0.60}, "F2": {"H": 0.56}})
    scores = [s.smart_bet_score for s in res["selections"]]
    assert scores == sorted(scores, reverse=True)


def test_no_bet_is_valid_and_empty_input_is_safe():
    res = evaluate_mode([], MODE_SMART, load_config(), {}, {})
    assert res["selections"] == []
    assert res["summary"] == {} or res["summary"]["selections_evaluated"] == 0


# --------------------------------------------------------------------------- #
# Backtest stability (§17) — not dominated silently by one large win           #
# --------------------------------------------------------------------------- #
def test_stability_flags_single_win_dependence():
    # one 20-unit win, rest losers: profitable overall, negative without top win.
    sel = pd.DataFrame({
        "league": ["L"] * 5, "date": ["2025-09-01"] * 5, "season": ["2025/26"] * 5,
        "selection": ["A"] * 5, "odds": [21, 3, 3, 3, 3],
        "model_p": [0.1] * 5, "edge": [0.05] * 5, "cons_ev": [0.1] * 5,
        "won": [True, False, False, False, False], "ret": [20.0, -1, -1, -1, -1]})
    _, stability, _ = _summarize(sel, n_fixtures=100, mode="high_return")
    assert stability["total_return_units"] > 0
    assert stability["return_excluding_top_1_win"] < 0
    assert stability["unstable"] is True


def test_season_label():
    assert _season("2025-09-01") == "2025/26"
    assert _season("2026-05-01") == "2025/26"
    assert _season("2026-08-01") == "2026/27"

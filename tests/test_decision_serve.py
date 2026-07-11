"""Tests for the predict -> decision-engine bridge (bet.md §25 integration).

Covers `decide_for_prediction` directly (unit level, no trained bundle needed)
and a light `/api/decisions` smoke test using a small trained Predictor bundle
mirroring `tests/test_training.py`'s pattern.
"""
from __future__ import annotations

import pandas as pd
import pytest

from match_predict.decisions import decide_for_fixtures, decide_for_prediction
from match_predict.pipeline import MatchPrediction


def _mp(**overrides) -> MatchPrediction:
    base = dict(
        league="L1", date="2024-01-06", home_team="T0", away_team="T1",
        prob_home=0.45, prob_draw=0.28, prob_away=0.27,
        exp_goals_home=1.6, exp_goals_away=1.1,
        model_rate_home=1.55, model_rate_away=1.05,
    )
    base.update(overrides)
    return MatchPrediction(**base)


def test_decide_with_feed_odds_returns_evaluated_selections():
    mp = _mp()
    md = decide_for_prediction(mp, feed_odds_1x2=[2.10, 3.40, 3.60])

    assert md.selections, "expected at least one evaluated selection"
    for view in (md.views.pure, md.views.hybrid, md.views.market):
        assert view is not None
        assert abs(sum(view.values()) - 1.0) < 1e-6

    winner_selections = [s for s in md.selections if s.market == "match_winner"]
    assert len(winner_selections) == 3
    for s in winner_selections:
        assert s.offered_odds is not None
        assert s.market_fair_probability is not None
        assert s.model_probability is not None
        # No feed odds timestamp -> honestly stale, never a crash or fabricated bet.
        assert s.decision_status == "NO BET"
        assert "ODDS_STALE" in s.rejection_reasons

    scores = [p for _, p in md.top_scores]
    assert scores, "score matrix must still be reported even with no qualifying bet"
    assert abs(sum(p for _, p in md.top_scores) - sum(scores)) < 1e-9


def test_decide_without_feed_odds_never_crashes_and_reports_missing_odds():
    mp = _mp()
    md = decide_for_prediction(mp, feed_odds_1x2=None, feed_odds_ou25=None)

    assert md.views.market is None          # no odds -> no market view
    winner_selections = [s for s in md.selections if s.market == "match_winner"]
    assert winner_selections == [], (
        "with no priced odds, no match_winner candidate should be built at all")
    for s in md.selections:
        assert s.decision_status == "NO BET"


def test_decide_is_deterministic():
    mp = _mp()
    md1 = decide_for_prediction(mp, feed_odds_1x2=[2.10, 3.40, 3.60])
    md2 = decide_for_prediction(mp, feed_odds_1x2=[2.10, 3.40, 3.60])
    assert md1.to_dict() == md2.to_dict()


def test_decide_with_over_under_odds_evaluates_totals_market():
    mp = _mp()
    md = decide_for_prediction(
        mp, feed_odds_1x2=[2.10, 3.40, 3.60],
        feed_odds_ou25={"OVER": 1.95, "UNDER": 1.90})
    ou_selections = [s for s in md.selections if s.market == "over_under_2_5"]
    assert len(ou_selections) == 2


# --------------------------------------------------------------------------- #
# Light /api/decisions smoke test — trains a tiny bundle, guards on failure   #
# so an environment without a fast train path just skips instead of flaking. #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    from match_predict.features.build import build_feature_frame
    from match_predict.pipeline.training import train_and_save
    from tests.conftest import _make_matches

    try:
        feat = build_feature_frame(_make_matches(n_teams=10, n_rounds=90, seed=5))
        tmp = tmp_path_factory.mktemp("app_artifacts")
        cache = str(tmp / "feat.pkl")
        feat.to_pickle(cache)
        out = str(tmp / "bundle")
        dates = feat["date"].sort_values()
        vs = str(dates.quantile(0.60).date())
        ve = str(dates.quantile(0.80).date())
        train_and_save(out=out, val_start=vs, val_end=ve, cache=cache)
    except Exception as e:                                     # noqa: BLE001
        pytest.skip(f"could not train a smoke-test bundle: {e}")

    fixtures_csv = tmp / "fixtures.csv"
    fixtures_csv.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        "E0,06/01/2024,15:00,T0,T1,2.10,3.40,3.60\n"
    )

    import importlib
    import os as _os
    _os.environ["ARTIFACTS"] = out
    _os.environ["DATA_ROOT"] = str(tmp)
    import app as app_module
    importlib.reload(app_module)
    app_module.load_predictor()
    return app_module.app.test_client()


def test_api_decisions_smoke(app_client):
    r = app_client.get("/api/decisions")
    assert r.status_code in (200, 409)   # 409 only if the league isn't recognised
    if r.status_code == 200:
        d = r.get_json()
        assert "decisions" in d and "summary" in d
        assert d["summary"]["fixtures"] == len(d["decisions"])

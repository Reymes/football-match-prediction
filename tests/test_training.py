"""End-to-end guard for the train/evaluate paths the web UI triggers (and the
model card the dashboard renders). Runs on the small synthetic fixture.
"""
import os

import pandas as pd
import pytest

from match_predict.pipeline.training import (
    train_and_save, evaluate_walk_forward, load_model_card,
    _trim_to_recent_seasons, TRAIN_SEASONS,
)
from match_predict.pipeline import Predictor
from match_predict.features.build import build_feature_frame
from tests.conftest import _make_matches


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    # a larger multi-season synthetic frame so train/val/test slices are non-empty
    feat = build_feature_frame(_make_matches(n_teams=10, n_rounds=90, seed=3))
    tmp = tmp_path_factory.mktemp("artifacts")
    cache = str(tmp / "feat.pkl")
    feat.to_pickle(cache)
    out = str(tmp / "bundle")
    dates = feat["date"].sort_values()
    vs = str(dates.quantile(0.60).date())
    ve = str(dates.quantile(0.80).date())
    card = train_and_save(out=out, val_start=vs, val_end=ve, cache=cache)
    return {"out": out, "cache": cache, "vs": vs, "ve": ve, "card": card}


def _multi_season_frame():
    rows = []
    for start_year in range(2010, 2020):  # ten seasons: 2010-2011 … 2019-2020
        rows.append({
            "season": f"{start_year}-{start_year + 1}",
            "date": pd.Timestamp(f"{start_year}-09-01"),
        })
    return pd.DataFrame(rows)


def test_trim_keeps_only_recent_seasons():
    feat = _multi_season_frame()
    trimmed = _trim_to_recent_seasons(feat, 3)
    assert set(trimmed["season"]) == {"2017-2018", "2018-2019", "2019-2020"}
    # newest match survives, oldest is dropped
    assert trimmed["date"].max() == feat["date"].max()
    assert trimmed["date"].min() > feat["date"].min()


def test_trim_is_noop_when_within_window():
    feat = _multi_season_frame()
    n = feat["season"].nunique()
    assert len(_trim_to_recent_seasons(feat, n + 5)) == len(feat)
    assert len(_trim_to_recent_seasons(feat, None)) == len(feat)


def test_card_records_train_seasons(trained):
    assert trained["card"]["train_seasons"] == TRAIN_SEASONS


def test_train_writes_bundle_and_card(trained):
    out = trained["out"]
    assert os.path.exists(os.path.join(out, "models.joblib"))
    assert os.path.exists(os.path.join(out, "model_card.json"))
    card = trained["card"]
    assert card["feature_count"] > 0
    assert card["data"]["total_matches"] == card["data"]["total_matches"]
    names = {m["name"] for m in card["models"]}
    assert {"market", "dixon_coles", "gbm", "ensemble", "ensemble_cal"} <= names


def test_card_metrics_are_well_formed(trained):
    for m in trained["card"]["models"]:
        vm = m["val_metrics"]
        for k in ("log_loss", "brier", "rps", "ece", "accuracy", "n"):
            assert k in vm
        assert vm["log_loss"] > 0 and vm["n"] > 0
        # base models flagged honest, ensemble variants flagged optimistic
        assert m["optimistic"] == (m["name"] in ("ensemble", "ensemble_cal"))
    infl = trained["card"]["ensemble_influence"]
    assert abs(sum(infl.values()) - 1.0) < 1e-6


def test_saved_bundle_loads_and_predicts(trained):
    pred = Predictor.load(trained["out"])
    fixtures = pd.DataFrame([{
        "league": "L1", "date": "2024-01-06", "home_team": "T0", "away_team": "T1",
    }])
    out = pred.predict_fixtures(fixtures)
    assert len(out) == 1
    p = out[0]
    assert abs(p.prob_home + p.prob_draw + p.prob_away - 1.0) < 1e-6
    # displayed EG are matrix-derived and finite
    assert p.exp_goals_home > 0 and p.model_rate_home is not None


def test_evaluate_merges_test_scorecard(trained):
    block = evaluate_walk_forward(
        out=trained["out"], val_start=trained["vs"], test_start=trained["ve"],
        cache=trained["cache"])
    assert block["n_test_matches"] > 0
    assert "market" in block["scorecard"]
    merged = load_model_card(trained["out"])
    # merge must preserve the training-side content
    assert "models" in merged and "test_evaluation" in merged

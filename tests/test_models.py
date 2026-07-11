"""Model sanity tests (fix.md §23 model tests): valid probabilities, determinism,
trainable on a small fixture dataset, and coherent Dixon-Coles goal rates.
"""
import numpy as np
import pytest

from match_predict.models.dixon_coles import DixonColes
from match_predict.models.baselines import MarketBaseline
from match_predict.ensemble.stacker import StackedEnsemble


def _valid_proba(p):
    p = np.asarray(p, float)
    assert np.all(p >= 0), "no negative probabilities"
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9), "rows sum to 1"


def test_dixon_coles_trains_and_predicts(matches):
    dc = DixonColes().fit(matches)
    assert dc.fitted_
    lam, mu = dc.expected_goals("T0", "T1")
    assert lam > 0 and mu > 0 and np.isfinite(lam) and np.isfinite(mu)
    M = dc.score_matrix("T0", "T1")
    assert M.sum() == pytest.approx(1.0, abs=1e-12)


def test_dixon_coles_is_deterministic(matches):
    a = DixonColes().fit(matches).expected_goals("T0", "T1")
    b = DixonColes().fit(matches).expected_goals("T0", "T1")
    assert a == pytest.approx(b)


def test_dixon_coles_unknown_team_is_league_average(matches):
    dc = DixonColes().fit(matches)
    lam, mu = dc.expected_goals("NON_EXISTENT_HOME", "NON_EXISTENT_AWAY")
    assert lam > 0 and mu > 0  # falls back to neutral strengths, still valid


def test_market_baseline_valid_proba(feat):
    p = MarketBaseline().predict_proba(feat)
    ok = np.isfinite(p).all(axis=1)
    _valid_proba(p[ok])


def test_stacked_ensemble_valid_proba_and_influence():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 3, n)
    def noisy_truth(sharpness):
        p = np.full((n, 3), (1 - sharpness) / 3)
        p[np.arange(n), y] += sharpness
        return p / p.sum(1, keepdims=True)
    base = {"a": noisy_truth(0.5), "b": noisy_truth(0.3)}
    ens = StackedEnsemble(["a", "b"]).fit(base, y)
    out = ens.predict_proba(base)
    _valid_proba(out)
    infl = ens.influence
    assert set(infl) == {"a", "b"}
    assert sum(infl.values()) == pytest.approx(1.0, abs=1e-6)
    # backwards-compatible alias still works
    assert ens.weights == infl

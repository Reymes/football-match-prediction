"""Reconciliation + displayed-expected-goals consistency (fix.md §12, §13).

After the score matrix is rescaled to the ensemble 1X2 headline, it must:
  * remain a valid, normalised, non-negative distribution;
  * have 1X2 marginals equal to the target within tolerance;
  * and the DISPLAYED expected goals must equal the matrix marginals (not the
    raw model rates) — this is acceptance criterion §31.9.
"""
import numpy as np
import pytest

from match_predict.models.markets import score_matrix, derive_markets
from match_predict.pipeline.predict import (
    reconcile_matrix_to_1x2, matrix_expected_goals, format_prediction,
)

TARGETS = [(0.30, 0.30, 0.40), (0.55, 0.25, 0.20), (0.20, 0.26, 0.54),
           (0.34, 0.33, 0.33)]


@pytest.mark.parametrize("target", TARGETS)
def test_reconcile_preserves_distribution(target):
    M = score_matrix(1.6, 1.1, rho=-0.045)
    R = reconcile_matrix_to_1x2(M, target)
    assert np.all(R >= 0.0)
    assert np.all(np.isfinite(R))
    assert R.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("target", TARGETS)
def test_reconcile_matches_target_marginals(target):
    M = score_matrix(1.6, 1.1, rho=-0.045)
    R = reconcile_matrix_to_1x2(M, target)
    book = derive_markets(R, 1.6, 1.1)
    assert book.p_home == pytest.approx(target[0], abs=1e-9)
    assert book.p_draw == pytest.approx(target[1], abs=1e-9)
    assert book.p_away == pytest.approx(target[2], abs=1e-9)


def test_matrix_expected_goals_matches_manual():
    M = score_matrix(2.1, 0.9, rho=-0.03)
    eg_h, eg_a = matrix_expected_goals(M)
    n = M.shape[0]
    idx = np.arange(n)
    assert eg_h == pytest.approx((M.sum(1) * idx).sum())
    assert eg_a == pytest.approx((M.sum(0) * idx).sum())


@pytest.mark.parametrize("target", TARGETS)
def test_displayed_exp_goals_equal_final_matrix(target):
    """The regression guard for the §13 fix: displayed EG == reconciled matrix EG,
    and (in the reconciled case) NOT the raw model lambdas."""
    row = {"league": "England-PL", "date": "2025-08-16",
           "home_team": "A", "away_team": "B"}
    lam, mu = 1.6, 1.1
    pred = format_prediction(row, np.array(target), lam, mu, rho=-0.045)

    # rebuild the exact final matrix the prediction used
    M = reconcile_matrix_to_1x2(score_matrix(lam, mu, rho=-0.045), target)
    eg_h, eg_a = matrix_expected_goals(M)
    assert pred.exp_goals_home == pytest.approx(eg_h, abs=1e-9)
    assert pred.exp_goals_away == pytest.approx(eg_a, abs=1e-9)
    # raw rates are retained but reported separately
    assert pred.model_rate_home == pytest.approx(lam)
    assert pred.model_rate_away == pytest.approx(mu)


def test_displayed_exp_goals_differ_from_lambda_when_reconciled():
    """Sanity: a lopsided target must move the matrix EG away from raw lambdas,
    proving the old behaviour (displaying lambda) would have been inconsistent."""
    row = {"league": "X", "date": "2025-08-16", "home_team": "A", "away_team": "B"}
    lam, mu = 1.6, 1.1
    pred = format_prediction(row, np.array([0.20, 0.26, 0.54]), lam, mu, rho=-0.045)
    assert abs(pred.exp_goals_home - lam) > 0.1
    assert pred.exp_goals_away > pred.exp_goals_home  # away-favoured target

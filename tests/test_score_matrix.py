"""Score-matrix invariants (fix.md §11) and market-derivation consistency.

Every displayed market must be re-derivable from the single joint score matrix,
and the matrix itself must be a valid probability distribution.
"""
import numpy as np
import pytest

from match_predict.models.markets import score_matrix, derive_markets

CASES = [(1.6, 1.1, -0.045), (0.7, 2.3, 0.0), (2.5, 2.5, -0.1), (0.3, 0.4, 0.08)]


@pytest.mark.parametrize("lam,mu,rho", CASES)
def test_matrix_is_valid_distribution(lam, mu, rho):
    M = score_matrix(lam, mu, rho=rho, max_goals=10)
    assert np.all(M >= 0.0), "probabilities must be non-negative"
    assert np.all(np.isfinite(M)), "probabilities must be finite"
    assert M.sum() == pytest.approx(1.0, abs=1e-12), "full distribution sums to 1"


@pytest.mark.parametrize("lam,mu,rho", CASES)
def test_1x2_marginals_from_cells(lam, mu, rho):
    M = score_matrix(lam, mu, rho=rho)
    book = derive_markets(M, lam, mu)
    n = M.shape[0]
    home = sum(M[i, j] for i in range(n) for j in range(n) if i > j)
    draw = sum(M[i, i] for i in range(n))
    away = sum(M[i, j] for i in range(n) for j in range(n) if i < j)
    assert book.p_home == pytest.approx(home, abs=1e-12)
    assert book.p_draw == pytest.approx(draw, abs=1e-12)
    assert book.p_away == pytest.approx(away, abs=1e-12)
    assert book.p_home + book.p_draw + book.p_away == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("lam,mu,rho", CASES)
def test_btts_and_totals_from_cells(lam, mu, rho):
    M = score_matrix(lam, mu, rho=rho)
    book = derive_markets(M, lam, mu)
    n = M.shape[0]
    btts_yes = sum(M[i, j] for i in range(1, n) for j in range(1, n))
    assert book.btts["yes"] == pytest.approx(btts_yes, abs=1e-12)
    assert book.btts["yes"] + book.btts["no"] == pytest.approx(1.0, abs=1e-9)
    # Over/Under 2.5 read directly off the goal-sum cells.
    over25 = sum(M[i, j] for i in range(n) for j in range(n) if i + j > 2.5)
    assert book.over_under[2.5]["over"] == pytest.approx(over25, abs=1e-12)
    assert (book.over_under[2.5]["over"] + book.over_under[2.5]["under"]
            == pytest.approx(1.0, abs=1e-9))


@pytest.mark.parametrize("lam,mu,rho", CASES)
def test_top_scores_sorted_descending(lam, mu, rho):
    M = score_matrix(lam, mu, rho=rho)
    book = derive_markets(M, lam, mu)
    probs = [p for _, p in book.correct_score]
    assert probs == sorted(probs, reverse=True), "correct scores must be ranked"
    # top score must equal the true argmax cell of the matrix
    assert probs[0] == pytest.approx(M.max(), abs=1e-12)


@pytest.mark.parametrize("lam,mu,rho", CASES)
def test_asian_handicap_normalised(lam, mu, rho):
    M = score_matrix(lam, mu, rho=rho)
    book = derive_markets(M, lam, mu)
    for line, d in book.asian_handicap.items():
        s = d["home"] + d["push"] + d["away"]
        assert s == pytest.approx(1.0, abs=1e-9), f"AH {line} must sum to 1"
        assert min(d.values()) >= -1e-12

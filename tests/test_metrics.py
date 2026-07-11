"""Metric correctness (fix.md §23 evaluation tests).

Reference-value checks so a future refactor can't silently invert a metric, plus
the 'lower log-loss is better' ordering guard the brief calls out explicitly.
"""
import numpy as np
import pytest

from match_predict.evaluation.metrics import (
    log_loss, brier_score, ranked_probability_score, poisson_deviance, accuracy,
)


def test_log_loss_perfect_is_zero():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.array([0, 1])
    assert log_loss(p, y) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_uniform_reference():
    p = np.full((4, 3), 1 / 3)
    y = np.array([0, 1, 2, 0])
    assert log_loss(p, y) == pytest.approx(np.log(3), abs=1e-9)


def test_log_loss_confident_wrong_is_penalised():
    good = log_loss(np.array([[0.7, 0.2, 0.1]]), np.array([0]))
    bad = log_loss(np.array([[0.1, 0.2, 0.7]]), np.array([0]))
    assert bad > good, "confident wrong prediction must score worse"


def test_brier_reference():
    # single sample, prob mass off the truth
    p = np.array([[0.6, 0.3, 0.1]])
    y = np.array([0])
    expected = (0.6 - 1) ** 2 + 0.3 ** 2 + 0.1 ** 2
    assert brier_score(p, y) == pytest.approx(expected)


def test_rps_reference_ordered():
    # true = Home (0). Putting mass on Away (2) is worse than on Draw (1)
    p_near = np.array([[0.5, 0.5, 0.0]])
    p_far = np.array([[0.5, 0.0, 0.5]])
    y = np.array([0])
    assert ranked_probability_score(p_far, y) > ranked_probability_score(p_near, y)


def test_rps_perfect_is_zero():
    p = np.array([[1.0, 0.0, 0.0]])
    assert ranked_probability_score(p, np.array([0])) == pytest.approx(0.0)


def test_poisson_deviance_perfect_is_zero():
    y = np.array([2.0, 1.0, 0.0, 3.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)


def test_accuracy_basic():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert accuracy(p, np.array([0, 2])) == pytest.approx(1.0)
    assert accuracy(p, np.array([1, 1])) == pytest.approx(0.0)


def test_lower_log_loss_ranks_better_model():
    """The exact ordering guard fix.md §23 requires."""
    y = np.array([0, 1, 2, 0, 1])
    sharp = np.array([[0.6, 0.25, 0.15]] * 5)
    sharp[np.arange(5), y] = 0.7  # concentrate mass on the truth
    sharp = sharp / sharp.sum(1, keepdims=True)
    vague = np.full((5, 3), 1 / 3)
    assert log_loss(sharp, y) < log_loss(vague, y)

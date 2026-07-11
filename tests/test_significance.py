"""Paired significance tests — the guardrail against overclaiming a market beat.

These verify the bootstrap is deterministic, that the sign convention is right
(lower loss => negative diff => "better"), and that an interval straddling zero
is reported as *not* distinguishable.
"""
import numpy as np
import pytest

from match_predict.evaluation.significance import (
    per_match_log_loss, per_match_brier, per_match_rps,
    paired_bootstrap, compare, compare_to_reference,
)


def _label(n, seed=0):
    return np.random.RandomState(seed).randint(0, 3, n)


def test_per_match_log_loss_matches_manual():
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    y = np.array([0, 2])
    got = per_match_log_loss(p, y)
    assert got == pytest.approx([-np.log(0.6), -np.log(0.6)])


def test_per_match_brier_and_rps_nonnegative():
    rng = np.random.RandomState(1)
    p = rng.dirichlet([2, 2, 2], 50)
    y = _label(50)
    assert (per_match_brier(p, y) >= 0).all()
    assert (per_match_rps(p, y) >= 0).all()


def test_bootstrap_is_deterministic():
    diff = np.random.RandomState(3).normal(-0.01, 0.1, 400)
    a = paired_bootstrap(diff, n_boot=500, seed=7)
    b = paired_bootstrap(diff, n_boot=500, seed=7)
    assert a == b


def test_better_model_reports_negative_diff_and_distinguishable():
    """A model that always puts more mass on the truth must beat the reference
    with a CI strictly below zero."""
    rng = np.random.RandomState(11)
    n = 3000
    y = rng.randint(0, 3, n)
    ref = rng.dirichlet([3, 3, 3], n)
    better = ref.copy()
    better[np.arange(n), y] += 0.05
    better /= better.sum(1, keepdims=True)
    res = compare(better, ref, y, model="better", reference="ref", n_boot=800)
    ll = res["log_loss"]
    assert ll.mean_diff < 0
    assert ll.ci_high < 0
    assert ll.distinguishable
    assert "better" in ll.verdict


def test_equal_models_not_distinguishable():
    rng = np.random.RandomState(5)
    n = 1500
    y = rng.randint(0, 3, n)
    p = rng.dirichlet([3, 3, 3], n)
    res = compare(p, p.copy(), y, model="a", reference="b", n_boot=600)
    ll = res["log_loss"]
    assert ll.mean_diff == pytest.approx(0.0, abs=1e-12)
    assert not ll.distinguishable
    assert "not statistically distinguishable" in ll.verdict


def test_block_bootstrap_widens_interval_vs_iid():
    """Correlated same-day fixtures: block resampling should not report a
    narrower interval than the naive i.i.d. bootstrap."""
    rng = np.random.RandomState(9)
    n_days, per_day = 60, 8
    n = n_days * per_day
    blocks = np.repeat(np.arange(n_days), per_day)
    # a per-day shared offset creates within-block correlation
    day_off = rng.normal(0, 0.05, n_days)[blocks]
    diff = day_off + rng.normal(-0.005, 0.02, n)
    _, lo_iid, hi_iid, _ = paired_bootstrap(diff, n_boot=800, seed=0)
    _, lo_blk, hi_blk, _ = paired_bootstrap(diff, blocks=blocks, n_boot=800, seed=0)
    assert (hi_blk - lo_blk) >= (hi_iid - lo_iid) * 0.9


def test_compare_to_reference_aligned_and_keyed():
    rng = np.random.RandomState(2)
    n = 500
    y = rng.randint(0, 3, n)
    probas = {"market": rng.dirichlet([3, 3, 3], n),
              "gbm": rng.dirichlet([3, 3, 3], n)}
    out = compare_to_reference(probas, y, reference="market", n_boot=300)
    assert set(out) == {"gbm"}
    assert set(out["gbm"]) == {"log_loss", "brier", "rps"}
    assert out["gbm"]["log_loss"]["reference"] == "market"


def test_compare_rejects_misaligned_inputs():
    y = np.array([0, 1, 2])
    a = np.full((3, 3), 1 / 3)
    b = np.full((2, 3), 1 / 3)
    with pytest.raises(ValueError):
        compare(a, b, y, model="a", reference="b")

"""Paired statistical comparison of two probabilistic forecasters.

A lower log-loss for the ensemble than for the market is only meaningful if the
*paired* per-match difference is distinguishable from zero. This module answers
"is model A actually better than reference B, or is it noise?" without ever
claiming victory the data does not support.

Everything here is deterministic: the bootstrap uses a seeded ``RandomState`` so
the reported confidence intervals reproduce exactly. Two resampling schemes are
provided:

  * ``bootstrap`` — resample individual matches with replacement (i.i.d.);
  * ``block`` — resample whole match-days with replacement, which respects the
    within-day correlation of fixtures and gives wider, more honest intervals.

Convention: for every metric *lower is better*, so a per-match difference
``d_i = loss_A(i) - loss_B(i)`` that is **negative on average** means A beats the
reference B. A 95% interval that straddles zero means "not statistically
distinguishable".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-15


def _clip(p):
    p = np.clip(np.asarray(p, float), _EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def per_match_log_loss(proba, y):
    p = _clip(proba)
    y = np.asarray(y)
    return -np.log(p[np.arange(len(y)), y])


def per_match_brier(proba, y):
    p = np.asarray(proba, float)
    Y = np.zeros_like(p)
    Y[np.arange(len(y)), np.asarray(y)] = 1.0
    return np.sum((p - Y) ** 2, axis=1)


def per_match_rps(proba, y):
    p = np.asarray(proba, float)
    Y = np.zeros_like(p)
    Y[np.arange(len(y)), np.asarray(y)] = 1.0
    cum = np.cumsum(p, axis=1) - np.cumsum(Y, axis=1)
    return np.sum(cum ** 2, axis=1) / (p.shape[1] - 1)


_PER_MATCH = {
    "log_loss": per_match_log_loss,
    "brier": per_match_brier,
    "rps": per_match_rps,
}


@dataclass
class PairedResult:
    """One metric, model A vs reference B, on the same matches."""

    metric: str
    model: str
    reference: str
    n: int
    mean_diff: float          # mean(loss_A - loss_B); negative => A better
    ci_low: float
    ci_high: float
    p_value: float
    method: str               # "bootstrap" | "block"
    distinguishable: bool     # CI excludes zero
    verdict: str

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "model": self.model,
            "reference": self.reference,
            "n": self.n,
            "mean_diff": round(self.mean_diff, 5),
            "ci_low": round(self.ci_low, 5),
            "ci_high": round(self.ci_high, 5),
            "p_value": round(self.p_value, 4),
            "method": self.method,
            "distinguishable": self.distinguishable,
            "verdict": self.verdict,
        }


def _verdict(mean_diff, lo, hi, model, reference) -> str:
    """Honest one-line reading of a paired interval (lower loss = better)."""
    distinguishable = (lo > 0) or (hi < 0)
    better = mean_diff < 0
    direction = "better than" if better else "worse than"
    if not distinguishable:
        practical = "practically negligible" if abs(mean_diff) < 5e-4 else "small"
        return (f"{model} is numerically {direction} {reference} but the 95% "
                f"interval includes zero — not statistically distinguishable "
                f"({practical} effect).")
    return (f"{model} is statistically {direction} {reference} "
            f"(95% interval excludes zero).")


def paired_bootstrap(diff, blocks=None, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap mean, (1-alpha) percentile CI and a two-sided p-value for the
    paired per-match differences ``diff``.

    If ``blocks`` (one label per match, e.g. the match date) is given, whole
    blocks are resampled with replacement (block bootstrap) so correlated
    same-day fixtures are not treated as independent.
    """
    diff = np.asarray(diff, float)
    n = len(diff)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)

    if blocks is None:
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            means[b] = diff[idx].mean()
    else:
        blocks = np.asarray(blocks)
        groups = [np.where(blocks == g)[0] for g in np.unique(blocks)]
        n_groups = len(groups)
        for b in range(n_boot):
            pick = rng.randint(0, n_groups, size=n_groups)
            idx = np.concatenate([groups[i] for i in pick])
            means[b] = diff[idx].mean()

    mean_diff = float(diff.mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    # two-sided bootstrap p-value: how often the resampled mean flips sign.
    frac_ge = float(np.mean(means >= 0))
    frac_le = float(np.mean(means <= 0))
    p_value = min(1.0, 2 * min(frac_ge, frac_le))
    return mean_diff, lo, hi, p_value


def compare(proba_a, proba_b, y, *, model, reference, blocks=None,
            metrics=("log_loss", "brier", "rps"), n_boot=2000, seed=0):
    """Paired comparison of model A against reference B on the same matches.

    Returns ``{metric: PairedResult}``. Requires the two probability matrices to
    be aligned row-for-row (same match order) — the caller must guarantee it.
    """
    proba_a = np.asarray(proba_a, float)
    proba_b = np.asarray(proba_b, float)
    y = np.asarray(y)
    if proba_a.shape != proba_b.shape or len(y) != len(proba_a):
        raise ValueError("proba_a, proba_b and y must be aligned and same length")
    method = "block" if blocks is not None else "bootstrap"
    out = {}
    for m in metrics:
        fn = _PER_MATCH[m]
        diff = fn(proba_a, y) - fn(proba_b, y)
        mean_diff, lo, hi, p = paired_bootstrap(
            diff, blocks=blocks, n_boot=n_boot, seed=seed)
        out[m] = PairedResult(
            metric=m, model=model, reference=reference, n=len(y),
            mean_diff=mean_diff, ci_low=lo, ci_high=hi, p_value=p,
            method=method, distinguishable=(lo > 0) or (hi < 0),
            verdict=_verdict(mean_diff, lo, hi, model, reference))
    return out


def compare_to_reference(probas, y, reference="market", *, blocks=None,
                         challengers=None, n_boot=2000, seed=0) -> dict:
    """Compare every challenger model against a reference (default: market).

    ``probas`` maps model name -> (n,3) probability array, all row-aligned to
    ``y``. Returns a JSON-friendly nested dict:
    ``{model: {metric: PairedResult.as_dict()}}``.
    """
    if reference not in probas:
        raise KeyError(f"reference model {reference!r} not in probas")
    ref = probas[reference]
    names = challengers or [k for k in probas if k != reference]
    result = {}
    for name in names:
        cmp = compare(probas[name], ref, y, model=name, reference=reference,
                      blocks=blocks, n_boot=n_boot, seed=seed)
        result[name] = {metric: pr.as_dict() for metric, pr in cmp.items()}
    return result

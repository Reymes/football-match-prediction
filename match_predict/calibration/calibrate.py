"""Probability calibration for multiclass (H/D/A) outputs.

task.md asks specifically for isotonic regression, Platt scaling, reliability
diagrams and Expected Calibration Error. We provide:

  * TemperatureScaler  — the multiclass generalisation of Platt scaling. Fits a
    single scalar T on validation log-loss; T>1 softens over-confident models,
    T<1 sharpens under-confident ones. One parameter, so it barely overfits and
    preserves the ranking/argmax of the model.
  * IsotonicCalibrator — per-class one-vs-rest isotonic regression, renormalised.
    More flexible (fixes non-monotone miscalibration) but needs more data and
    can distort the joint. Good when a single class is systematically off.
  * expected_calibration_error / reliability_curve — the diagnostics.

Calibration is fit ONLY on out-of-time validation predictions, then frozen for
the test period — exactly like the models themselves.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-12


def _onehot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class TemperatureScaler:
    """Single-temperature scaling on the log-probabilities (Platt, multiclass)."""

    def __init__(self):
        self.T = 1.0

    def fit(self, proba: np.ndarray, y: np.ndarray):
        logits = np.log(np.clip(proba, _EPS, 1.0))

        def nll(T):
            z = logits / T
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(axis=1, keepdims=True)
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1.0)))

        res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
        self.T = float(res.x)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(proba, _EPS, 1.0)) / self.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def fit_transform(self, proba, y):
        return self.fit(proba, y).transform(proba)


class IsotonicCalibrator:
    """Per-class one-vs-rest isotonic regression, then renormalise rows."""

    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.models = [IsotonicRegression(out_of_bounds="clip")
                       for _ in range(n_classes)]

    def fit(self, proba: np.ndarray, y: np.ndarray):
        Y = _onehot(y, self.n_classes)
        for c in range(self.n_classes):
            self.models[c].fit(proba[:, c], Y[:, c])
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        cols = [self.models[c].transform(proba[:, c])
                for c in range(self.n_classes)]
        out = np.clip(np.vstack(cols).T, _EPS, None)
        return out / out.sum(axis=1, keepdims=True)

    def fit_transform(self, proba, y):
        return self.fit(proba, y).transform(proba)


def expected_calibration_error(proba: np.ndarray, y: np.ndarray,
                               n_bins: int = 15) -> float:
    """Top-label ECE: bin by predicted confidence, compare to accuracy."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(proba: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Return (bin_confidence, bin_accuracy, bin_count) for a reliability plot."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)

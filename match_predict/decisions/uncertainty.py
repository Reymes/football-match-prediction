"""Uncertainty buffer and conservative probability (bet.md §7).

The buffer is NOT chosen arbitrarily. It is assembled from measured components,
each of which we can point to a source for:

    buffer = base_calibration_error                       (validation ECE in band)
           + disagreement_weight * model_disagreement     (pure vs hybrid gap)
           + data_quality_weight * (1 - data_quality)     (missing lineups etc.)
           + sample_weight        * sample_shortfall       (thin history penalty)

then clamped to [0, max_uncertainty_buffer].

    conservative_probability = max(0, calibrated_probability - buffer)

A confidence interval is derived from the same buffer so the report can show a
plausible range rather than a single point estimate. This is a pragmatic proxy
for a bootstrap interval when a full bootstrap is unavailable at decision time;
the chronological backtest (backtest.py) uses true out-of-sample resampling.
"""
from __future__ import annotations


def model_disagreement(view_a: dict, view_b: dict) -> float:
    """Mean absolute difference between two HDA probability views (0..1)."""
    keys = set(view_a) & set(view_b)
    if not keys:
        return 0.0
    return sum(abs(view_a[k] - view_b[k]) for k in keys) / len(keys)


def uncertainty_buffer(cfg_uncertainty: dict,
                       calibration_error: float | None = None,
                       disagreement: float = 0.0,
                       data_quality: float = 1.0,
                       n_samples: int | None = None,
                       min_samples: int | None = None) -> float:
    """Assemble the buffer from its measured components (see module docstring)."""
    u = cfg_uncertainty
    base = calibration_error if calibration_error is not None \
        else u.get("base_calibration_error", 0.02)
    buf = base
    buf += u.get("disagreement_weight", 0.5) * max(0.0, disagreement)
    buf += u.get("data_quality_weight", 0.1) * max(0.0, 1.0 - float(data_quality))
    if n_samples is not None and min_samples:
        shortfall = max(0.0, (min_samples - n_samples) / float(min_samples))
        buf += u.get("sample_weight", 0.05) * shortfall
    return float(min(max(buf, 0.0), u.get("max_uncertainty_buffer", 0.15)))


def conservative_probability(calibrated_probability: float, buffer: float) -> float:
    """Calibrated probability reduced by the uncertainty buffer, floored at 0."""
    return float(max(0.0, calibrated_probability - buffer))


def confidence_interval(calibrated_probability: float, buffer: float) -> tuple:
    """Symmetric interval [p - buffer, p + buffer] clamped to [0, 1]."""
    lo = max(0.0, calibrated_probability - buffer)
    hi = min(1.0, calibrated_probability + buffer)
    return (round(lo, 4), round(hi, 4))

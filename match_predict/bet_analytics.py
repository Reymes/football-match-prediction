"""Honest performance & calibration analysis of the paper-bet ledger.

This sits on top of the paper-betting store (``store.py`` / ``betting.py``) and
answers the question ``bet.md`` §16/§18 insists must be answered honestly:
*given what was actually staked, how did the paper wallet do — and were the
model probabilities we bet on actually calibrated?*

Two things are reported and kept strictly separate (bet.md §18):

  * **Realized return** — P&L, ROI, drawdown, longest losing run, return by
    odds band. This is what the €1000 play wallet actually did.
  * **Forecast quality** — Brier score, log loss and a reliability table over
    the ``model_prob`` stored on each bet vs. whether the backed selection won.
    A good ROI on badly calibrated probabilities (or a tiny sample) is flagged,
    not celebrated.

Everything is advisory research output over paper money. Nothing here places a
bet, and a positive historical ROI is never evidence of a proven edge — sample
sizes are always shown and small samples are called out.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from .evaluation.metrics import log_loss

# Odds buckets for return-by-price reporting (bet.md §16 "return by odds band").
_ODDS_BANDS = [
    (1.0, 1.5), (1.5, 2.0), (2.0, 3.0),
    (3.0, 5.0), (5.0, 10.0), (10.0, float("inf")),
]

# Below this many settled bets the ledger is too small to read anything into.
MIN_MEANINGFUL_SAMPLE = 30


def _settled(bets: list[dict]) -> list[dict]:
    """Won/lost bets in chronological order; voids are excluded from returns.

    Chronological order (settled_at, then id) is required so the equity curve,
    drawdown and losing-run reflect the true sequence rather than display order.
    """
    done = [b for b in bets if b.get("status") in ("won", "lost")]
    return sorted(done, key=lambda b: (b.get("settled_at") or "", b.get("id", 0)))


def _won_flag(b: dict) -> int:
    return 1 if b.get("status") == "won" else 0


def _max_drawdown(pnl_steps: list[float]) -> float:
    """Largest peak-to-trough drop of the cumulative-P&L curve (currency)."""
    peak = 0.0
    cum = 0.0
    worst = 0.0
    for step in pnl_steps:
        cum += step
        peak = max(peak, cum)
        worst = max(worst, peak - cum)
    return worst


def _longest_losing_run(bets: list[dict]) -> int:
    """Longest streak of consecutive losing bets (a win resets the count)."""
    run = worst = 0
    for b in bets:
        if b.get("status") == "lost":
            run += 1
            worst = max(worst, run)
        else:
            run = 0
    return worst


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}+" if hi == float("inf") else f"{lo:.1f}-{hi:.1f}"


def _return_by_odds_band(bets: list[dict]) -> list[dict]:
    out = []
    for lo, hi in _ODDS_BANDS:
        group = [b for b in bets if lo <= float(b["odds"]) < hi]
        if not group:
            continue
        staked = sum(float(b["stake"]) for b in group)
        returned = sum(float(b["payout"]) for b in group)
        won = sum(_won_flag(b) for b in group)
        out.append({
            "band": _band_label(lo, hi),
            "n": len(group),
            "won": won,
            "win_rate": round(won / len(group), 4),
            "staked": round(staked, 2),
            "returned": round(returned, 2),
            "pnl": round(returned - staked, 2),
            "roi": round((returned - staked) / staked, 4) if staked else None,
        })
    return out


def _reliability_table(p: np.ndarray, won: np.ndarray, n_bins: int) -> list[dict]:
    """Bucket predicted win-probabilities and compare to the empirical rate."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # include the right edge in the final bin so p == 1.0 is counted
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        table.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": int(mask.sum()),
            "mean_pred": round(float(p[mask].mean()), 4),
            "emp_win_rate": round(float(won[mask].mean()), 4),
        })
    return table


def _forecast_quality(bets: list[dict], n_bins: int) -> dict | None:
    """Calibration of the model_prob we actually bet on (bet.md §18).

    Only bets that stored a finite ``model_prob`` are scored; the count is
    always reported so a thin sample cannot masquerade as a validated result.
    Log loss reuses ``evaluation.metrics.log_loss`` (fed the 2-class win/lose
    distribution); Brier is the standard binary mean-squared error.
    """
    scored = [b for b in bets
              if b.get("model_prob") is not None and isfinite(float(b["model_prob"]))]
    if not scored:
        return None
    p = np.array([float(b["model_prob"]) for b in scored], dtype=float)
    won = np.array([_won_flag(b) for b in scored], dtype=int)
    proba2 = np.column_stack([1.0 - p, p])      # [P(lose), P(win)] for reuse
    return {
        "n_scored": len(scored),
        "brier": round(float(np.mean((p - won) ** 2)), 4),
        "log_loss": round(log_loss(proba2, won), 4),
        "avg_pred_win_prob": round(float(p.mean()), 4),
        "empirical_win_rate": round(float(won.mean()), 4),
        "reliability": _reliability_table(p, won, n_bins),
    }


def performance_report(bets: list[dict], *, reliability_bins: int = 5) -> dict:
    """Full honest report over a list of bet rows (``store.all_bets()``).

    Realized return and forecast quality are reported separately (bet.md §18).
    Every figure carries its sample size; ``meaningful_sample`` warns when the
    settled ledger is too small to interpret.
    """
    settled = _settled(bets)
    n = len(settled)
    open_bets = [b for b in bets if b.get("status") == "open"]
    void_bets = [b for b in bets if b.get("status") == "void"]

    staked = sum(float(b["stake"]) for b in settled)
    returned = sum(float(b["payout"]) for b in settled)
    won = sum(_won_flag(b) for b in settled)
    pnl = returned - staked

    report = {
        "n_bets_total": len(bets),
        "n_open": len(open_bets),
        "n_void": len(void_bets),
        "realized_return": {
            "n_settled": n,
            "won": won,
            "lost": n - won,
            "win_rate": round(won / n, 4) if n else None,
            "staked": round(staked, 2),
            "returned": round(returned, 2),
            "pnl": round(pnl, 2),
            "roi": round(pnl / staked, 4) if staked else None,
            "avg_odds": round(
                float(np.mean([float(b["odds"]) for b in settled])), 3) if n else None,
            "max_drawdown": round(
                _max_drawdown([float(b["payout"]) - float(b["stake"])
                               for b in settled]), 2),
            "longest_losing_run": _longest_losing_run(settled),
        },
        "by_odds_band": _return_by_odds_band(settled),
        "forecast_quality": _forecast_quality(settled, reliability_bins),
        "meaningful_sample": n >= MIN_MEANINGFUL_SAMPLE,
        "note": (
            "Advisory research over paper money. Realized return and forecast "
            "quality are separate: a positive ROI is not proof of an edge, "
            "especially below "
            f"{MIN_MEANINGFUL_SAMPLE} settled bets. No outcome is certain."
        ),
    }
    return report


def report_from_store(store) -> dict:
    """Convenience wrapper: build the report from a live paper-bet Store."""
    return performance_report(store.all_bets(limit=100000))

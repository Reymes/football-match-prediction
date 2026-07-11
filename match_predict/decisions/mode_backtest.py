"""Chronological backtest for the Smart Bet / High Return modes (bet-funcuanlty §16, §17).

Layers the mode decision rules on top of the SAME leakage-safe walk-forward
probabilities used by `decisions.backtest`:

  * models are refit strictly on prior data (walk-forward);
  * the market fair probability comes from the pre-match odds already in the row;
  * the per-market validation profile is fitted on the VALIDATION period only and
    FROZEN before the untouched test period is scored (thresholds are never tuned
    on the test set, bet-funcuanlty §16 / bet.md §11);
  * the pure (odds-free) GBM probability supplies the independent model support the
    High Return mode requires.

Each mode is backtested INDEPENDENTLY (bet-funcuanlty §16): the two modes are never
merged. High-return realized return is reported WITH the largest wins removed
(§17) so a strategy that only looks profitable because of one rare high-odds win
is flagged as unstable rather than trusted. A high-return strategy is expected to
have a LOW hit rate and high variance — that is not a defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..evaluation.backtest import WalkForwardBacktest
from ..evaluation.metrics import evaluate_proba
from .backtest import _bootstrap_ci, _flat_stake_curve, _validation_probs
from .bet_modes import evaluate_selection_for_mode
from .schema import DecisionStatus, load_config
from .validation import build_1x2_profile_from_probs
from . import eligibility, uncertainty as unc_mod

_LABEL = {"H": 0, "D": 1, "A": 2}
_SEL = {0: "H", 1: "D", 2: "A"}
_QUALIFIED = {DecisionStatus.QUALIFIED.value, DecisionStatus.STRONG_EVIDENCE.value}


@dataclass
class ModeBacktestResult:
    mode: str
    forecast_quality: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    stability: dict = field(default_factory=dict)          # §17 top-win removal
    per_league: pd.DataFrame = None
    per_odds_band: pd.DataFrame = None
    calibration_selected: dict = field(default_factory=dict)
    profile_passed: bool = False
    notes: list = field(default_factory=list)


def _season(dt) -> str:
    d = pd.Timestamp(dt)
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}/{str(start + 1)[-2:]}"


def run_mode_backtest(feat: pd.DataFrame, test_start, val_start, mode: str,
                      config: dict | None = None, leagues=None,
                      verbose: bool = True) -> ModeBacktestResult:
    """Backtest ONE bet mode on the untouched test period (1X2 market)."""
    config = config or load_config()
    mode_cfg = (config.get("bet_modes", {}) or {}).get(mode)
    if not mode_cfg:
        raise ValueError(f"unknown bet mode {mode!r}")

    wf = WalkForwardBacktest()
    res = wf.run(feat, test_start=test_start, val_start=val_start,
                 leagues=leagues, verbose=verbose)
    test = res.test_frame.reset_index(drop=True)
    y = test["y"].to_numpy()

    pure_p = res.probas.get("gbm")                       # odds-free proxy
    market_p = res.probas["market"]
    hybrid_p = res.probas.get("ensemble_cal", res.probas["ensemble"])
    forecast_quality = {
        k: evaluate_proba(v, y) for k, v in
        {"pure": pure_p, "market": market_p, "hybrid": hybrid_p}.items()
        if v is not None}

    # validation profile, fitted on validation only then frozen (no test leak)
    val_probs, val_y = _validation_probs(wf, feat, test_start, val_start, leagues)
    profile = build_1x2_profile_from_probs(
        val_probs, val_y, market="match_winner",
        min_samples=mode_cfg["minimum_historical_samples"])

    rows = []
    for i in range(len(test)):
        r = test.iloc[i]
        odds = [r.get("odds_h"), r.get("odds_d"), r.get("odds_a")]
        if not np.all(np.isfinite(np.array(odds, float))):
            continue
        disagree = unc_mod.model_disagreement(
            {"H": pure_p[i, 0], "D": pure_p[i, 1], "A": pure_p[i, 2]},
            {"H": hybrid_p[i, 0], "D": hybrid_p[i, 1], "A": hybrid_p[i, 2]})
        best = None
        for k in range(3):
            rot = [odds[k]] + [o for j, o in enumerate(odds) if j != k]
            base = eligibility.evaluate_selection(
                market="match_winner", selection=_SEL[k], side_or_line=_SEL[k],
                model_probability=float(hybrid_p[i, k]), offered_odds=odds[k],
                outcome_odds_set=rot, decision_cutoff=None, odds_timestamp=None,
                config=config, market_profile=profile, data_quality=1.0,
                model_disagreement=disagree, model_version="walkforward")
            ms = evaluate_selection_for_mode(
                base, mode, mode_cfg, fixture_id=f"row{i}", league=r["league"],
                home_team="", away_team="", kickoff=str(r["date"]),
                data_quality=1.0, model_disagreement=disagree,
                pure_probability=float(pure_p[i, k]),
                calibration_error=profile.calibration_error_at(float(hybrid_p[i, k])),
                n_samples=profile.samples_at(float(hybrid_p[i, k])),
                profile_passed=profile.passed_quality)
            if ms.decision_status in _QUALIFIED:
                score = ms.high_return_score if mode == "high_return" else ms.smart_bet_score
                if best is None or (score or 0) > best[0]:
                    best = (score or 0, ms, k)
        if best is not None:
            _, ms, k = best
            won = y[i] == k
            ret = (ms.offered_odds - 1.0) if won else -1.0
            rows.append({
                "idx": i, "league": r["league"], "date": r["date"],
                "season": _season(r["date"]), "selection": ms.selection,
                "odds": ms.offered_odds, "model_p": ms.model_probability,
                "edge": ms.probability_edge, "cons_ev": ms.conservative_expected_value,
                "won": bool(won), "ret": ret})

    sel = pd.DataFrame(rows)
    perf, stability, calib = _summarize(sel, len(test), mode)
    note = ("match_winner passed out-of-time validation quality."
            if profile.passed_quality else
            "match_winner FAILED out-of-time quality — every selection rejected.")
    if mode == "high_return":
        note += (" High-return strategies are expected to have a LOW hit rate and "
                 "high variance; see the top-win-removal stability report (§17).")
    return ModeBacktestResult(
        mode=mode, forecast_quality=forecast_quality, performance=perf,
        stability=stability, per_league=_breakdown(sel, "league"),
        per_odds_band=_breakdown(sel, "odds_band", _band(sel, "odds",
                                 [1.5, 2.0, 3.0, 4.5, 6.0, 10.0])),
        calibration_selected=calib, profile_passed=profile.passed_quality,
        notes=[note])


def _summarize(sel: pd.DataFrame, n_fixtures: int, mode: str):
    perf = {"mode": mode, "n_fixtures": n_fixtures, "n_selections": len(sel),
            "n_no_bet": n_fixtures - len(sel),
            "selection_rate": round(len(sel) / n_fixtures, 4) if n_fixtures else 0.0}
    stability: dict = {}
    calib: dict = {}
    if sel.empty:
        return perf, stability, calib
    rets = sel["ret"].to_numpy()
    _, max_dd, longest = _flat_stake_curve(rets)
    perf.update({
        "avg_offered_odds": round(float(sel["odds"].mean()), 3),
        "median_offered_odds": round(float(sel["odds"].median()), 3),
        "avg_model_prob": round(float(sel["model_p"].mean()), 4),
        "avg_edge": round(float(sel["edge"].mean()), 4),
        "avg_conservative_ev": round(float(sel["cons_ev"].mean()), 4),
        "hit_rate": round(float(sel["won"].mean()), 4),
        "roi_per_unit": round(float(rets.mean()), 4),
        "total_return_units": round(float(rets.sum()), 3),
        "roi_ci95": _bootstrap_ci(rets),
        "max_drawdown_units": round(max_dd, 3),
        "longest_losing_run": longest})

    # §17: is the return driven by a few large wins?
    ordered = np.sort(rets)[::-1]
    total = float(rets.sum())

    def _drop_top(n):
        return round(float(ordered[n:].sum()), 3) if len(ordered) > n else 0.0
    k5 = max(1, int(np.ceil(0.05 * len(ordered))))
    stability = {
        "total_return_units": round(total, 3),
        "return_excluding_top_1_win": _drop_top(1),
        "return_excluding_top_3_wins": _drop_top(3),
        "return_excluding_top_5pct_payouts": _drop_top(k5),
        "unstable": bool(total > 0 and _drop_top(1) < -1e-9),
    }
    if stability["unstable"]:
        stability["flag"] = ("UNSTABLE: profit disappears once the single largest "
                             "win is removed — do not trust this strategy.")
    if len(sel) >= 20:
        calib = {"n": len(sel),
                 "avg_model_prob": round(float(sel["model_p"].mean()), 4),
                 "realized_hit_rate": round(float(sel["won"].mean()), 4),
                 "calibration_gap": round(float(
                     sel["model_p"].mean() - sel["won"].mean()), 4)}
    return perf, stability, calib


def _band(sel: pd.DataFrame, col: str, edges: list):
    if sel.empty:
        return None
    return pd.cut(sel[col], bins=[-np.inf] + list(edges) + [np.inf])


def _breakdown(sel: pd.DataFrame, name: str, key=None):
    if sel.empty:
        return pd.DataFrame(columns=[name, "n", "hit_rate", "roi_per_unit"])
    g = sel.groupby(key if key is not None else name, observed=True)
    out = g.agg(n=("ret", "size"), hit_rate=("won", "mean"),
                roi_per_unit=("ret", "mean")).reset_index()
    out = out.rename(columns={out.columns[0]: name})
    out["hit_rate"] = out["hit_rate"].round(4)
    out["roi_per_unit"] = out["roi_per_unit"].round(4)
    return out

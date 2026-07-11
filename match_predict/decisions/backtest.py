"""Chronological decision backtest (bet.md §16, §17, §18).

Principles, enforced structurally:
  * thresholds/profiles are selected on the VALIDATION period and then FROZEN;
    the untouched TEST period only measures them (bet.md §11, §16).
  * every decision uses only information available at that historical fixture:
    the walk-forward models are refit strictly on prior data, the market
    probability comes from the pre-match odds already in the row, and no future
    price, tuning, or result is consulted (bet.md §16).
  * probability QUALITY (log loss / Brier / calibration) is reported SEPARATELY
    from decision-rule performance and realized return (bet.md §18): a
    profitable-looking backtest with bad calibration is flagged, not trusted.

We reuse `WalkForwardBacktest` for the heavy lifting (it already builds pure,
market, and hybrid probabilities out-of-time on the same test fixtures). This
module layers the decision rules on top and simulates flat-stake realized
return, with bootstrap confidence intervals and max drawdown.

The market baseline is the de-vigged 1X2 already used everywhere else, so the
1X2 decision backtest is fully driven by real pre-match odds when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..evaluation.backtest import WalkForwardBacktest
from ..evaluation.metrics import evaluate_proba
from ..features.build import market_implied_probs
from .schema import load_config, DecisionStatus
from .validation import build_1x2_profile_from_probs
from . import edge as edge_mod, devig as devig_mod, uncertainty as unc_mod
from . import eligibility, grading

_LABEL = {"H": 0, "D": 1, "A": 2}
_SEL = {0: "H", 1: "D", 2: "A"}


@dataclass
class DecisionBacktestResult:
    forecast_quality: dict = field(default_factory=dict)   # per view scorecards
    decision_performance: dict = field(default_factory=dict)
    selections: pd.DataFrame = None
    per_league: pd.DataFrame = None
    per_odds_band: pd.DataFrame = None
    per_edge_band: pd.DataFrame = None
    calibration_selected: dict = field(default_factory=dict)
    config_used: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _flat_stake_curve(returns: np.ndarray):
    """Cumulative P/L (flat 1-unit stakes) and max drawdown."""
    if len(returns) == 0:
        return np.array([]), 0.0, 0
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(dd.max()) if len(dd) else 0.0
    # longest losing sequence
    longest = cur = 0
    for r in returns:
        if r < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return equity, max_dd, longest


def _bootstrap_ci(returns: np.ndarray, n_boot: int = 2000, seed: int = 7):
    if len(returns) < 5:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = [rng.choice(returns, size=len(returns), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def run_decision_backtest(feat: pd.DataFrame, test_start, val_start,
                          config: dict | None = None, leagues=None,
                          market: str = "match_winner",
                          view: str = "hybrid", verbose: bool = True
                          ) -> DecisionBacktestResult:
    """Backtest the 1X2 decision rules chronologically on the untouched test set.

    `view` selects which probability drives the model side: "hybrid"
    (calibrated ensemble), "pure" (gbm-only proxy for the no-odds model), or
    "market" (sanity check — should never beat the market it is priced against).
    """
    config = config or load_config()
    wf = WalkForwardBacktest()
    res = wf.run(feat, test_start=test_start, val_start=val_start,
                 leagues=leagues, verbose=verbose)

    test = res.test_frame.reset_index(drop=True)
    y = test["y"].to_numpy()

    # --- forecast quality per view (bet.md §18), all OUT-OF-TIME ----------
    view_probs = {
        "pure": res.probas.get("gbm"),          # gbm-only = odds-free proxy
        "market": res.probas["market"],
        "hybrid": res.probas.get("ensemble_cal", res.probas["ensemble"]),
    }
    forecast_quality = {k: evaluate_proba(v, y) for k, v in view_probs.items()
                        if v is not None}

    # --- validation profile: fit on VALIDATION predictions only -----------
    # Re-derive validation-period probabilities to build the calibration
    # profile WITHOUT touching the test period (bet.md §11).
    val_probs, val_y = _validation_probs(wf, feat, test_start, val_start, leagues)
    profile = build_1x2_profile_from_probs(
        val_probs, val_y, market="match_winner",
        min_samples=config["markets"]["match_winner"].get(
            "minimum_historical_samples", 300))
    profiles = {"match_winner": profile}

    if not profile.passed_quality:
        note = ("match_winner FAILED out-of-time quality on validation — "
                "the decision layer treats the market as unvalidated and will "
                "reject every selection (no bet).")
    else:
        note = "match_winner passed out-of-time validation quality."

    # --- decision simulation on the TEST period ---------------------------
    model_p = view_probs[view]
    market_p = view_probs["market"]
    rows = []
    for i in range(len(test)):
        r = test.iloc[i]
        odds = [r.get("odds_h"), r.get("odds_d"), r.get("odds_a")]
        if not np.all(np.isfinite(np.array(odds, float))):
            continue
        # de-vig with the configured method
        try:
            fair = devig_mod.devig(odds, config.get("devig_method", "shin"))
        except ValueError:
            continue
        # evaluate each of H/D/A as a candidate; keep the best qualifying
        best = None
        for k in range(3):
            rot = [odds[k]] + [o for j, o in enumerate(odds) if j != k]
            dec = eligibility.evaluate_selection(
                market="match_winner", selection=_SEL[k], side_or_line=_SEL[k],
                model_probability=float(model_p[i, k]), offered_odds=odds[k],
                outcome_odds_set=rot, decision_cutoff=None, odds_timestamp=None,
                config=config, market_profile=profile, data_quality=1.0,
                model_disagreement=unc_mod.model_disagreement(
                    {"H": model_p[i, 0], "D": model_p[i, 1], "A": model_p[i, 2]},
                    {"H": market_p[i, 0], "D": market_p[i, 1], "A": market_p[i, 2]}),
                # timestamps are not simulated per-row here (pre-match odds in the
                # historical row are known before kickoff by construction), so we
                # do not require them for the offline backtest.
                model_version="walkforward",
                )
            grading.grade_selection(dec, config, profile)
            if dec.decision_status in (DecisionStatus.QUALIFIED.value,
                                       DecisionStatus.STRONG_EVIDENCE.value):
                if best is None or (dec.conservative_expected_value or -9) > \
                        (best.conservative_expected_value or -9):
                    best = dec
        if best is not None:
            won = (y[i] == _LABEL[best.selection])
            ret = (best.offered_odds - 1.0) if won else -1.0
            rows.append({
                "idx": i, "league": r["league"], "date": r["date"],
                "selection": best.selection, "odds": best.offered_odds,
                "model_p": best.model_probability,
                "market_p": best.market_fair_probability,
                "edge": best.probability_edge,
                "cons_ev": best.conservative_expected_value,
                "grade": best.grade, "won": bool(won), "ret": ret,
            })

    sel = pd.DataFrame(rows)
    n_fixtures = int(len(test))
    n_sel = len(sel)
    perf = {
        "view": view, "market": market,
        "n_fixtures": n_fixtures, "n_selections": n_sel,
        "n_no_bet": n_fixtures - n_sel,
        "selection_rate": round(n_sel / n_fixtures, 4) if n_fixtures else 0.0,
    }
    if n_sel:
        rets = sel["ret"].to_numpy()
        _, max_dd, longest = _flat_stake_curve(rets)
        perf.update({
            "avg_model_prob": round(float(sel["model_p"].mean()), 4),
            "avg_offered_odds": round(float(sel["odds"].mean()), 3),
            "avg_edge": round(float(sel["edge"].mean()), 4),
            "avg_decision_ev": round(float(sel["cons_ev"].mean()), 4),
            "hit_rate": round(float(sel["won"].mean()), 4),
            "roi_per_unit": round(float(rets.mean()), 4),
            "total_return_units": round(float(rets.sum()), 3),
            "roi_ci95": _bootstrap_ci(rets),
            "max_drawdown_units": round(max_dd, 3),
            "longest_losing_run": longest,
        })

    # --- breakdowns (bet.md §16, §26): every report shows sample sizes ----
    per_league = _breakdown(sel, "league")
    per_odds_band = _breakdown(sel, "odds_band", _band(sel, "odds",
                               [1.5, 2.0, 3.0, 4.5, 6.0]))
    per_edge_band = _breakdown(sel, "edge_band", _band(sel, "edge",
                               [0.04, 0.06, 0.09, 0.15]))

    # --- calibration of SELECTED opportunities (bet.md §16) ---------------
    calib = {}
    if n_sel >= 20:
        calib = {"n": n_sel,
                 "avg_model_prob": round(float(sel["model_p"].mean()), 4),
                 "realized_hit_rate": round(float(sel["won"].mean()), 4),
                 "calibration_gap": round(float(
                     sel["model_p"].mean() - sel["won"].mean()), 4)}

    return DecisionBacktestResult(
        forecast_quality=forecast_quality, decision_performance=perf,
        selections=sel, per_league=per_league, per_odds_band=per_odds_band,
        per_edge_band=per_edge_band, calibration_selected=calib,
        config_used=config, profiles={k: v.to_dict() for k, v in profiles.items()},
        notes=[note])


def _validation_probs(wf, feat, test_start, val_start, leagues):
    """Reconstruct calibrated VALIDATION-period 1X2 probs (no test leakage)."""
    from ..models import DixonColes, MarketBaseline, GBMOutcomeModel  # noqa
    from ..ensemble import StackedEnsemble
    from ..calibration import TemperatureScaler
    feat = feat.sort_values(["date", "league"]).reset_index(drop=True)
    if leagues:
        feat = feat[feat.league.isin(leagues)]
    test_start = pd.Timestamp(test_start)
    val_start = pd.Timestamp(val_start)
    has_odds = feat[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
    val = feat[(feat.date >= val_start) & (feat.date < test_start) & has_odds].copy()

    market = MarketBaseline()
    gbm = GBMOutcomeModel(wf.features, num_rounds=wf.gbm_rounds)
    gbm.fit(feat[feat.date < val_start])
    dc_map = wf._dc_rolling(feat[feat.date < test_start], val)
    dc_p, _, _ = wf._dc_probs_for(val, dc_map)
    base = {"market": market.predict_proba(val), "gbm": gbm.predict_proba(val),
            "dixon_coles": dc_p}
    yv = val["ftr"].map(_LABEL).to_numpy()
    ok = np.isfinite(np.hstack([base[k] for k in base])).all(axis=1)
    base = {k: v[ok] for k, v in base.items()}
    yv = yv[ok]
    ens = StackedEnsemble(["market", "gbm", "dixon_coles"]).fit(base, yv)
    cal = TemperatureScaler().fit(ens.predict_proba(base), yv)
    return cal.transform(ens.predict_proba(base)), yv


def _band(sel: pd.DataFrame, col: str, edges: list):
    if sel.empty:
        return None
    e = [-np.inf] + list(edges) + [np.inf]
    return pd.cut(sel[col], bins=e)


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

"""Reusable training + evaluation entry points shared by the CLI and the web UI.

`train_and_save` fits the full serving bundle (GBM, per-league Dixon-Coles,
stacked ensemble + calibrator, market-free fallback), writes it to ``out/`` and
emits a **model card** (``model_card.json``) describing what each model is, how
much data it saw, and its validation-season metrics.

`evaluate_walk_forward` runs the honest out-of-time backtest and records the test
scorecard into the model card (kept separate from the fit-set numbers).

Both accept a ``progress`` callback (``progress(str)``) so a background web job
can stream human-readable status lines.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data import load_all, validate_matches
from ..data.schema import CANONICAL_COLUMNS
from ..features import build_feature_frame, FEATURE_COLUMNS
from ..models import DixonColes, MarketBaseline, GBMOutcomeModel
from ..ensemble import StackedEnsemble
from ..calibration import TemperatureScaler
from ..evaluation import WalkForwardBacktest
from ..evaluation.metrics import evaluate_proba
from ..evaluation.significance import compare_to_reference
from .predictor import Predictor

_LABEL = {"H": 0, "D": 1, "A": 2}
MODEL_CARD = "model_card.json"

# Fit + backtest only on the most recent N seasons (~N years). Older football
# is a poor guide to current scoring rates / market efficiency, so the serving
# bundle is trained on a rolling 12-season window rather than the full archive.
TRAIN_SEASONS = 12

_MODEL_DESCRIPTIONS = {
    "market": "De-vigged bookmaker 1X2 odds (basic normalization). A strong "
              "baseline, not an independent football model.",
    "dixon_coles": "Time-weighted Dixon-Coles goal model, one per league, refit "
                   "on a rolling trailing window of prior matches.",
    "gbm": "LightGBM multiclass 1X2 model on Elo/form/context (+ market) features.",
    "ensemble": "Stacked multinomial-logistic meta-learner over market + GBM + "
                "Dixon-Coles log-probabilities.",
    "ensemble_cal": "The stacked ensemble after temperature scaling (final served "
                    "probabilities).",
}


def _noop(_msg):  # default progress sink
    pass


def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _trim_to_recent_seasons(feat, n_seasons):
    """Keep only the ``n_seasons`` most recent seasons (by their latest match
    date, so it is independent of the season-label format). Returns ``feat``
    unchanged when it already holds that many or fewer seasons."""
    if not n_seasons or feat["season"].nunique() <= n_seasons:
        return feat
    order = feat.groupby("season")["date"].max().sort_values().index
    keep = set(order[-n_seasons:])
    return feat[feat["season"].isin(keep)].reset_index(drop=True)


def _load_features(data, cache, progress, train_seasons=TRAIN_SEASONS):
    if cache and os.path.exists(cache):
        feat = pd.read_pickle(cache)
        progress(f"loaded cached features {feat.shape} from {cache}")
    else:
        progress(f"loading match history from {list(data)} …")
        df = load_all(*data)
        progress(str(validate_matches(df)))
        progress("building leakage-safe features …")
        feat = build_feature_frame(df)
        if cache:
            feat.to_pickle(cache)
            progress(f"cached features to {cache}")
    # The cache holds the full archive; the training window is applied on read
    # so both train_and_save and the backtest see the same recent-seasons slice.
    feat = _trim_to_recent_seasons(feat, train_seasons)
    progress(f"features ready: {feat.shape} — {feat['season'].nunique()} "
             f"seasons [{feat['date'].min().date()} … "
             f"{feat['date'].max().date()}]")
    return feat


def _data_summary(feat, val_start, val_end, n_val):
    by_league = {str(k): int(v) for k, v in feat.groupby("league").size().items()}
    by_season = {str(k): int(v) for k, v in feat.groupby("season").size().items()}
    return {
        "total_matches": int(len(feat)),
        "by_league": by_league,
        "n_seasons": len(by_season),
        "date_min": str(feat["date"].min().date()),
        "date_max": str(feat["date"].max().date()),
        "validation_season": {"start": str(val_start.date()),
                              "end": str(val_end.date()), "n_matches": int(n_val)},
    }


def train_and_save(data=("football-data", "testing"), out="artifacts",
                   val_start="2024-08-01", val_end="2025-08-01",
                   dc_window_days=900, train_seasons=TRAIN_SEASONS,
                   cache=None, progress=_noop) -> dict:
    """Fit + persist the serving bundle; return (and write) the model card.

    Only the most recent ``train_seasons`` seasons are used for fitting.
    """
    t0 = time.time()
    feat = _load_features(data, cache, progress, train_seasons)
    feat = feat.sort_values(["date", "league"]).reset_index(drop=True)
    val_start, val_end = pd.Timestamp(val_start), pd.Timestamp(val_end)
    has_odds = feat[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)

    bt = WalkForwardBacktest(dc_window_days=dc_window_days, features=FEATURE_COLUMNS)
    val = feat[(feat.date >= val_start) & (feat.date < val_end) & has_odds].copy()
    n_gbm_val = int((feat.date < val_start).sum())
    progress(f"validation season: {len(val)} matches "
             f"[{val_start.date()} … {val_end.date()})")

    progress("fitting base models on the validation season …")
    market = MarketBaseline()
    gbm_v = GBMOutcomeModel(FEATURE_COLUMNS).fit(feat[feat.date < val_start])
    dc_val = bt._dc_rolling(feat[feat.date < val_end], val)
    p_dc_val, _, _ = bt._dc_probs_for(val, dc_val)
    p_gbm_val = gbm_v.predict_proba(val)
    p_mkt_val = market.predict_proba(val)
    yv = val["ftr"].map(_LABEL).to_numpy()

    base_full = {"market": p_mkt_val, "gbm": p_gbm_val, "dixon_coles": p_dc_val}
    ok = np.isfinite(np.hstack(list(base_full.values()))).all(axis=1)
    base_full = {k: v[ok] for k, v in base_full.items()}
    yv_ok = yv[ok]

    progress("fitting stacked ensemble + calibrator …")
    ensemble = StackedEnsemble(["market", "gbm", "dixon_coles"]).fit(base_full, yv_ok)
    ens_val = ensemble.predict_proba(base_full)
    calibrator = TemperatureScaler().fit(ens_val, yv_ok)
    cal_val = calibrator.transform(ens_val)

    base_fb = {"gbm": p_gbm_val[ok], "dixon_coles": p_dc_val[ok]}
    ensemble_fb = StackedEnsemble(["gbm", "dixon_coles"]).fit(base_fb, yv_ok)
    calibrator_fb = TemperatureScaler().fit(ensemble_fb.predict_proba(base_fb), yv_ok)
    progress(f"ensemble influence (NOT mixture weights): {ensemble.influence}  "
             f"T={calibrator.T:.3f}")

    # per-model validation metrics ------------------------------------------
    val_metrics = {
        "market": evaluate_proba(base_full["market"], yv_ok),
        "dixon_coles": evaluate_proba(base_full["dixon_coles"], yv_ok),
        "gbm": evaluate_proba(base_full["gbm"], yv_ok),
        "ensemble": evaluate_proba(ens_val, yv_ok),
        "ensemble_cal": evaluate_proba(cal_val, yv_ok),
    }

    progress(f"training final LightGBM on all {len(feat)} matches …")
    gbm_final = GBMOutcomeModel(FEATURE_COLUMNS).fit(feat)

    latest = feat["date"].max()
    ref = latest + pd.Timedelta(days=1)
    dc_models = {}
    for league, pool in feat.groupby("league"):
        train = pool[pool.date > ref - pd.Timedelta(days=dc_window_days)]
        if len(train) < 60:
            train = pool.tail(500)
        dc_models[league] = DixonColes().fit(train, ref_date=ref)
    progress(f"fit {len(dc_models)} per-league Dixon-Coles models "
             f"as-of {latest.date()}")

    predictor = Predictor(
        history=feat[CANONICAL_COLUMNS].copy(),
        gbm=gbm_final, ensemble=ensemble, calibrator=calibrator,
        ensemble_nomkt=ensemble_fb, calibrator_nomkt=calibrator_fb,
        dc_models=dc_models, features=list(FEATURE_COLUMNS),
        trained_through=str(latest.date()))
    predictor.save(out)

    # trained-on descriptions per model
    trained_on = {
        "market": "n/a — reads live odds, nothing to fit",
        "dixon_coles": f"per league, rolling {dc_window_days}d window of prior matches",
        "gbm": f"{len(feat)} matches (all history)",
        "ensemble": f"{len(yv_ok)} validation matches (out-of-time)",
        "ensemble_cal": f"{len(yv_ok)} validation matches (out-of-time)",
    }
    models = []
    for name in ("market", "dixon_coles", "gbm", "ensemble", "ensemble_cal"):
        models.append({
            "name": name,
            "description": _MODEL_DESCRIPTIONS[name],
            "trained_on": trained_on[name],
            "val_metrics": val_metrics[name],
            "optimistic": name in ("ensemble", "ensemble_cal"),
        })

    card = {
        "trained_at": _now_iso(),
        "trained_through": str(latest.date()),
        "duration_sec": round(time.time() - t0, 1),
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "train_seasons": train_seasons,
        "data": _data_summary(feat, val_start, val_end, len(val)),
        "models": models,
        "ensemble_influence": {k: float(v) for k, v in ensemble.influence.items()},
        "ensemble_influence_note": "normalised |coef| of the logistic stacker — "
                                   "relative influence, NOT mixture weights.",
        "temperature": float(calibrator.T),
        "val_metrics_note": "Base models (market/dixon_coles/gbm) are scored "
                            "out-of-sample on the validation season. ensemble & "
                            "ensemble_cal are scored on their own fit set, so they "
                            "are OPTIMISTIC — use the walk-forward test scorecard "
                            "for honest, comparable numbers.",
    }
    _merge_write_card(out, card)
    progress(f"saved bundle + model card to {out}/  "
             f"(trained through {latest.date()}, {card['duration_sec']}s)")
    return card


def _per_group_scorecard(test_frame, probas, group_col, min_n=50):
    """Honest per-subgroup (league / season) scorecards from the out-of-time
    test predictions. Groups with fewer than ``min_n`` matches are kept but
    flagged so the UI can grey out small, noisy samples."""
    y = test_frame["y"].to_numpy()
    out = {}
    for key, idx in test_frame.groupby(group_col).groups.items():
        pos = test_frame.index.get_indexer(idx)
        n = len(pos)
        row = {"n": int(n), "small_sample": bool(n < min_n),
               "models": {name: evaluate_proba(p[pos], y[pos])
                          for name, p in probas.items()}}
        out[str(key)] = row
    return out


def evaluate_walk_forward(data=("football-data", "testing"), out="artifacts",
                          test_start="2025-08-01", val_start="2024-08-01",
                          train_seasons=TRAIN_SEASONS,
                          cache=None, progress=_noop) -> dict:
    """Run the honest out-of-time backtest and fold the test scorecard into the
    model card. Returns the scorecard dict.

    The backtest is restricted to the most recent ``train_seasons`` seasons so
    it evaluates the same data window the served bundle is trained on.
    """
    t0 = time.time()
    feat = _load_features(data, cache, progress, train_seasons)
    progress(f"walk-forward backtest: test>={test_start}  val>={val_start} …")
    res = WalkForwardBacktest().run(feat, test_start=test_start,
                                    val_start=val_start, verbose=False)
    sc = res.summary()
    scorecard = {name: {k: float(v) for k, v in row.items()}
                 for name, row in sc.to_dict("index").items()}

    # Paired significance vs the market baseline — the only honest way to say
    # whether the ensemble's lower log-loss is real or noise. Day-blocks keep
    # correlated same-day fixtures from inflating confidence.
    tf = res.test_frame
    y_test = tf["y"].to_numpy()
    day_blocks = tf["date"].astype("int64").to_numpy()
    progress("paired bootstrap vs market (block by match-day) …")
    significance = compare_to_reference(
        res.probas, y_test, reference="market",
        challengers=["gbm", "dixon_coles", "ensemble", "ensemble_cal"],
        blocks=day_blocks, seed=0)

    per_league = _per_group_scorecard(tf, res.probas, "league")
    per_season = (_per_group_scorecard(tf, res.probas, "season")
                  if "season" in tf.columns else {})

    test_block = {
        "evaluated_at": _now_iso(),
        "duration_sec": round(time.time() - t0, 1),
        "test_start": str(test_start), "val_start": str(val_start),
        "n_test_matches": int(res.test_frame.shape[0]),
        "scorecard": scorecard,
        "significance_vs_market": significance,
        "served_model": "ensemble_cal",
        "per_league": per_league,
        "per_season": per_season,
        "ensemble_influence": {k: float(v)
                               for k, v in res.extra["ensemble_influence"].items()},
        "temperature": float(res.extra["temperature"]),
        "poisson_deviance_home": res.extra["poisson_deviance_home"],
        "poisson_deviance_away": res.extra["poisson_deviance_away"],
    }
    _merge_write_card(out, {"test_evaluation": test_block})
    progress(f"walk-forward done: {test_block['n_test_matches']} test matches, "
             f"{test_block['duration_sec']}s")
    return test_block


def load_model_card(out="artifacts") -> dict | None:
    path = os.path.join(out, MODEL_CARD)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _merge_write_card(out, patch):
    os.makedirs(out, exist_ok=True)
    card = load_model_card(out) or {}
    card.update(patch)
    with open(os.path.join(out, MODEL_CARD), "w") as f:
        json.dump(card, f, indent=2)

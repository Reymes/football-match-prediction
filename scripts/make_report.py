#!/usr/bin/env python
"""Generate the evaluation report artifacts under reports/:
  * scorecard.csv          — walk-forward metrics per model
  * reliability.png        — reliability diagram (market vs calibrated ensemble)
  * example_predictions.json — full per-match outputs for a sample gameweek
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from match_predict.data import load_all
from match_predict.features import build_feature_frame
from match_predict.evaluation import WalkForwardBacktest
from match_predict.calibration import reliability_curve
from match_predict.pipeline import format_prediction

_LABEL_INV = {0: "H", 1: "D", 2: "A"}
OUT = "reports"


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = os.environ.get("FEAT_CACHE")
    if cache and os.path.exists(cache):
        feat = pd.read_pickle(cache)
    else:
        feat = build_feature_frame(load_all("football-data", "testing"))
        if cache:
            feat.to_pickle(cache)

    res = WalkForwardBacktest().run(
        feat, test_start="2025-08-01", val_start="2024-08-01", verbose=True)

    # 1) scorecard
    sc = res.summary()
    sc.to_csv(os.path.join(OUT, "scorecard.csv"))
    print("\n", sc.to_string())

    # 2) reliability diagram
    y = res.test_frame["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, style in [("market", "o-"), ("ensemble_cal", "s-")]:
        xs, ys, ns = reliability_curve(res.probas[name], y, n_bins=10)
        ax.plot(xs, ys, style, label=f"{name} (ECE {res.scorecards[name]['ece']})")
    ax.set_xlabel("Predicted confidence (top class)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram — 2025/26 out-of-time test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reliability.png"), dpi=110)
    print("wrote reports/reliability.png")

    # 3) example predictions (one PL gameweek), full market book
    tf = res.test_frame.reset_index(drop=True)
    hda = res.probas["ensemble_cal"]
    lam, mu = res.extra["dc_lambda"], res.extra["dc_mu"]
    imp = res.gbm.feature_importance()
    mask = ((tf.league == "England-PL")
            & (tf.date >= pd.Timestamp("2025-08-15"))
            & (tf.date <= pd.Timestamp("2025-08-25")))
    out = []
    for k in np.where(mask.to_numpy())[0][:10]:
        row = tf.iloc[k]
        pred = format_prediction(row, hda[k], lam[k], mu[k], rho=-0.045,
                                 gbm=res.gbm, gbm_row_df=tf.iloc[[k]], importance=imp)
        pred.actual = _LABEL_INV[int(row["y"])]
        d = pred.to_dict()
        d["top_scores"] = [[list(s), round(p, 4)] for s, p in d["top_scores"]]
        d["asian_handicap"] = {str(k2): {kk: round(vv, 4) for kk, vv in v.items()}
                               for k2, v in d["asian_handicap"].items()}
        d["over_under"] = {str(k2): {kk: round(vv, 4) for kk, vv in v.items()}
                           for k2, v in d["over_under"].items()}
        d["team_totals"] = {side: {str(k2): {kk: round(vv, 4) for kk, vv in v.items()}
                                   for k2, v in tt.items()}
                            for side, tt in d["team_totals"].items()}
        d["btts"] = {k2: round(v, 4) for k2, v in d["btts"].items()}
        out.append(d)
    with open(os.path.join(OUT, "example_predictions.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote reports/example_predictions.json ({len(out)} matches)")

    meta = {"ensemble_influence": {k: float(v) for k, v in res.extra["ensemble_influence"].items()},
            "ensemble_influence_note": "normalised |coef| of the logistic stacker; "
                                       "relative influence only, NOT mixture weights",
            "temperature": float(res.extra["temperature"]),
            "poisson_deviance_home": res.extra["poisson_deviance_home"],
            "poisson_deviance_away": res.extra["poisson_deviance_away"]}
    with open(os.path.join(OUT, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote reports/run_meta.json")


if __name__ == "__main__":
    main()

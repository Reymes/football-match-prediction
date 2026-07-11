"""Emit honest, reproducible reports from the model card's walk-forward block.

Reads ``artifacts/model_card.json`` (populated by ``evaluate_walk_forward``) and
writes the subset of the imrpove.md §14 reports that our current pipeline can
back with real, out-of-time numbers:

    reports/model_scorecard.csv
    reports/model_scorecard.md
    reports/market_significance_test.md
    reports/per_league_metrics.csv
    reports/per_season_metrics.csv

Every report is stamped with evaluation dates, sample size and the served model
so the numbers are never read out of context. Run:

    python scripts/write_reports.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

CARD = os.path.join("artifacts", "model_card.json")
OUT = "reports"
_MODELS = ["market", "dixon_coles", "gbm", "ensemble", "ensemble_cal"]
_METRICS = ["log_loss", "brier", "rps", "ece", "accuracy", "n"]


def _load():
    with open(CARD) as f:
        card = json.load(f)
    te = card.get("test_evaluation")
    if not te or not te.get("scorecard"):
        sys.exit("no test_evaluation in model card — run evaluate_walk_forward first")
    return card, te


def _stamp(card, te):
    d = card.get("data", {})
    return (f"Evaluated {te.get('evaluated_at','?')} · test from {te['test_start']} "
            f"(val from {te['val_start']}) · {te['n_test_matches']} out-of-time "
            f"matches · trained through {card.get('trained_through','?')} · "
            f"{card.get('feature_count','?')} features · served model "
            f"{te.get('served_model','ensemble_cal')} · market features used: yes")


def _scorecard_csv(te):
    with open(os.path.join(OUT, "model_scorecard.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", *_METRICS])
        for name, row in te["scorecard"].items():
            w.writerow([name, *[row.get(k, "") for k in _METRICS]])


def _scorecard_md(card, te):
    lines = ["# Model scorecard — honest walk-forward test", "", _stamp(card, te), "",
             "| model | " + " | ".join(_METRICS) + " |",
             "|" + "---|" * (len(_METRICS) + 1)]
    for name, row in te["scorecard"].items():
        lines.append("| " + name + " | " +
                     " | ".join(str(row.get(k, "")) for k in _METRICS) + " |")
    lines += ["", "Lower is better for log-loss, Brier, RPS and ECE; accuracy is "
              "reported for reference only, never optimised."]
    with open(os.path.join(OUT, "model_scorecard.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _significance_md(card, te):
    sig = te.get("significance_vs_market", {})
    lines = ["# Paired significance vs the de-vigged market baseline", "",
             _stamp(card, te), "",
             "Paired per-match log-loss differences (challenger − market), 95% "
             "confidence interval from a block bootstrap grouped by match-day. "
             "Δ < 0 means the challenger has lower loss than the market.", "",
             "| model | mean Δ log-loss | 95% CI | p-value | distinguishable |",
             "|---|---|---|---|---|"]
    for name in ["ensemble_cal", "ensemble", "gbm", "dixon_coles"]:
        r = sig.get(name, {}).get("log_loss")
        if not r:
            continue
        lines.append(f"| {name} | {r['mean_diff']} | {r['ci_low']} … "
                     f"{r['ci_high']} | {r['p_value']} | "
                     f"{'yes' if r['distinguishable'] else 'no'} |")
    served = te.get("served_model", "ensemble_cal")
    verdict = sig.get(served, {}).get("log_loss", {}).get("verdict", "n/a")
    lines += ["", f"**Served model ({served}):** {verdict}", "",
              "We do not claim to beat the bookmaker when the paired interval "
              "includes zero."]
    with open(os.path.join(OUT, "market_significance_test.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _group_csv(te, key, fname):
    groups = te.get(key, {})
    if not groups:
        return
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key.replace("per_", ""), "n", "small_sample",
                    *[f"{m}_log_loss" for m in _MODELS]])
        for name, row in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            ll = [row["models"].get(m, {}).get("log_loss", "") for m in _MODELS]
            w.writerow([name, row["n"], row["small_sample"], *ll])


def main():
    os.makedirs(OUT, exist_ok=True)
    card, te = _load()
    _scorecard_csv(te)
    _scorecard_md(card, te)
    _significance_md(card, te)
    _group_csv(te, "per_league", "per_league_metrics.csv")
    _group_csv(te, "per_season", "per_season_metrics.csv")
    print(f"wrote reports to {OUT}/ — {te['n_test_matches']} test matches, "
          f"served {te.get('served_model','ensemble_cal')}")


if __name__ == "__main__":
    main()

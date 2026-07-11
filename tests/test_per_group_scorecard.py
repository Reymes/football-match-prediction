"""The per-subgroup (league/season) scorecard folded into the model card must
align each group's rows to the right probability slice and flag small samples."""
import numpy as np
import pandas as pd

from match_predict.pipeline.training import _per_group_scorecard


def _frame():
    # two leagues, A large, B tiny -> B must be flagged small_sample
    leagues = ["A"] * 60 + ["B"] * 5
    y = ([0] * 30 + [1] * 30) + [2] * 5
    return pd.DataFrame({"league": leagues, "y": y}).reset_index(drop=True)


def test_per_group_aligns_and_flags_small_samples():
    tf = _frame()
    n = len(tf)
    # a "perfect" model puts all mass on the true class per row
    perfect = np.zeros((n, 3))
    perfect[np.arange(n), tf["y"].to_numpy()] = 1.0
    probas = {"perfect": perfect, "uniform": np.full((n, 3), 1 / 3)}

    out = _per_group_scorecard(tf, probas, "league", min_n=50)
    assert set(out) == {"A", "B"}
    assert out["A"]["n"] == 60 and out["A"]["small_sample"] is False
    assert out["B"]["n"] == 5 and out["B"]["small_sample"] is True
    # perfect model: log-loss ~0 in every group; uniform ~ log 3
    assert out["A"]["models"]["perfect"]["log_loss"] < 1e-6
    assert out["A"]["models"]["uniform"]["log_loss"] > 1.0
    assert out["B"]["models"]["perfect"]["log_loss"] < 1e-6

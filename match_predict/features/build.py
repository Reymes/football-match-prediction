"""Assemble the leakage-safe feature frame and the market-implied baseline.

`build_feature_frame` runs Elo -> form -> context in order and returns a frame
with `FEATURE_COLUMNS` populated. Every column is available strictly before
kickoff.

`market_implied_probs` converts bookmaker 1X2 odds into de-vigged
probabilities. This is our *strong baseline*: beating (or matching) the closing
line on log-loss is the real bar for any football model, and de-vigged odds are
also an excellent feature/ensemble input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import EloEngine, add_elo_features
from .form import add_form_features
from .context import add_context_features

# Columns fed to the gradient-boosted / meta models. Deliberately excludes any
# post-match statistic of the CURRENT game (those only enter via lagged form).
FEATURE_COLUMNS = [
    # strength
    "home_elo", "away_elo", "elo_diff", "elo_exp_home",
    # form (levels)
    "home_form_pts", "away_form_pts",
    "home_roll_gf", "home_roll_ga", "away_roll_gf", "away_roll_ga",
    "home_ewma_gf", "home_ewma_ga", "away_ewma_gf", "away_ewma_ga",
    "home_roll_stf", "home_roll_sta", "away_roll_stf", "away_roll_sta",
    "home_ewma_stf", "away_ewma_stf",
    "home_roll_cf", "away_roll_cf",
    "home_momentum", "away_momentum",
    "home_matches_played", "away_matches_played",
    # form (differentials)
    "form_pts_diff", "roll_gf_diff", "roll_ga_diff", "ewma_stf_diff",
    # context
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_congestion_14d", "away_congestion_14d",
    "season_progress",
    # market signal (de-vigged pre-match odds)
    "mkt_prob_h", "mkt_prob_d", "mkt_prob_a",
]


def _implied(odds: pd.Series) -> pd.Series:
    # Coerce to float first: a fixture with all-blank odds would otherwise leave
    # an object-dtype column that downstream models (LightGBM) reject.
    odds = pd.to_numeric(odds, errors="coerce")
    with np.errstate(divide="ignore"):
        return 1.0 / odds


def market_implied_probs(df: pd.DataFrame, prefix: str = "odds_") -> pd.DataFrame:
    """De-vig 1X2 odds via basic normalization (proportional to raw implied).

    Returns a DataFrame with columns p_h, p_d, p_a (NaN where odds missing).
    Basic normalization is simple and unbiased enough for a baseline; the
    ARCHITECTURE notes Shin's method / power methods as production upgrades.
    """
    h = _implied(df[f"{prefix}h"])
    d = _implied(df[f"{prefix}d"])
    a = _implied(df[f"{prefix}a"])
    total = h + d + a
    return pd.DataFrame({
        "p_h": h / total, "p_d": d / total, "p_a": a / total
    }, index=df.index)


def build_feature_frame(df: pd.DataFrame,
                        elo_engine: EloEngine | None = None) -> pd.DataFrame:
    """Full feature pipeline. Input: canonical matches. Output: + features."""
    df = df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    df = add_elo_features(df, elo_engine)
    df = add_form_features(df)
    df = add_context_features(df)

    mkt = market_implied_probs(df)
    df["mkt_prob_h"] = mkt["p_h"]
    df["mkt_prob_d"] = mkt["p_d"]
    df["mkt_prob_a"] = mkt["p_a"]

    # Guarantee every declared feature exists.
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df

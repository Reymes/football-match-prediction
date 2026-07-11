"""Ingestion: raw Football-Data CSVs -> one clean, canonical, sorted DataFrame.

Handles the two things that break naive loaders on this dataset:
  1. Mixed file encodings (utf-8-sig on newer files, cp1252/latin-1 on older).
  2. Mixed date formats: DD/MM/YY (1990s) and DD/MM/YYYY (modern).

The output is a tidy long table, one row per match, sorted by (date, league).
Downstream feature code assumes this ordering.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import (
    CANONICAL_COLUMNS,
    LEAGUE_BY_DIV,
    ODDS_PREFERENCE,
    RAW_TO_CANONICAL,
)

_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _read_csv_any_encoding(path: str) -> pd.DataFrame:
    """Read a CSV trying several encodings; last resort replaces bad bytes.

    Football-Data files occasionally contain a malformed row with extra
    trailing commas / inconsistent field counts. ``on_bad_lines='skip'``
    drops just those rows instead of failing the whole file.
    """
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False,
                               on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", encoding_errors="replace",
                       low_memory=False, on_bad_lines="skip")


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse DD/MM/YY and DD/MM/YYYY into datetime64 without dateutil fallback.

    We branch on string length so pandas gets an explicit format for each era
    (avoids the slow, warning-noisy per-element inference). Two-digit years are
    century-pivoted into the 1993..2030 window this dataset lives in.
    """
    s = series.astype("string").str.strip()
    length = s.str.len()
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    four_digit = length == 10          # DD/MM/YYYY
    two_digit = length == 8            # DD/MM/YY
    dt.loc[four_digit] = pd.to_datetime(
        s[four_digit], format="%d/%m/%Y", errors="coerce")
    dt.loc[two_digit] = pd.to_datetime(
        s[two_digit], format="%d/%m/%y", errors="coerce")
    # Anything else: last-resort dayfirst parse
    rest = dt.isna() & s.notna()
    if rest.any():
        dt.loc[rest] = pd.to_datetime(s[rest], dayfirst=True, errors="coerce")

    future = dt > pd.Timestamp("2035-01-01")
    if future.any():
        dt.loc[future] = dt.loc[future] - pd.DateOffset(years=100)
    return dt


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Return the first candidate column that exists AND has any value."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            # fill only where still missing -> preference order respected
            out = out.where(out.notna(), vals)
    return out


def _season_from_filename(path: str) -> str:
    m = re.search(r"_(\d{4}-\d{4})\.csv$", os.path.basename(path))
    return m.group(1) if m else "unknown"


def load_file(path: str) -> pd.DataFrame:
    """Load a single league-season CSV into the canonical schema."""
    raw = _read_csv_any_encoding(path)
    raw = raw.dropna(how="all")
    # Drop rows with no teams (trailing blank lines happen in this dataset).
    if "HomeTeam" in raw.columns:
        raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()]
    raw = raw.reset_index(drop=True)

    out = pd.DataFrame(index=raw.index)

    # Direct 1:1 mappings
    for src, dst in RAW_TO_CANONICAL.items():
        if src in raw.columns:
            out[dst] = raw[src]

    # Odds / multi-source fields via preference lists
    for dst, candidates in ODDS_PREFERENCE.items():
        out[dst] = _pick_first_present(raw, candidates)

    # League + season
    div = raw["Div"].iloc[0] if "Div" in raw.columns and len(raw) else None
    out["league"] = LEAGUE_BY_DIV.get(div, div or _league_from_path(path))
    out["season"] = _season_from_filename(path)

    # Dates
    out["date"] = _parse_dates(out["date"]) if "date" in out else pd.NaT
    if "time" not in out:
        out["time"] = np.nan

    # Numeric coercions for goals & stats
    numeric_cols = ["fthg", "ftag", "hthg", "htag", "hs", "as_", "hst", "ast",
                    "hf", "af", "hc", "ac", "hy", "ay", "hr", "ar", "ah_line"]
    for c in numeric_cols:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan

    # Ensure every canonical column exists
    for c in CANONICAL_COLUMNS:
        if c not in out:
            out[c] = np.nan

    # Deterministic match id
    out["match_id"] = (
        out["league"].astype(str) + "|" + out["season"].astype(str) + "|"
        + out["date"].dt.strftime("%Y%m%d").fillna("NA") + "|"
        + out["home_team"].astype(str) + "|" + out["away_team"].astype(str)
    )

    return out[CANONICAL_COLUMNS]


def _league_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def load_directory(root: str, pattern: str = "*/*.csv") -> pd.DataFrame:
    """Load every CSV under ``root`` (default: league-subfolder layout)."""
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if not files:
        # Flat directory (e.g. the testing/ folder)
        files = sorted(glob.glob(os.path.join(root, "*.csv")))
    frames = [load_file(f) for f in files]
    if not frames:
        raise FileNotFoundError(f"No CSVs found under {root!r}")
    df = pd.concat(frames, ignore_index=True)
    return _finalize(df)


def load_all(*roots: str) -> pd.DataFrame:
    """Load and concatenate one or more roots (e.g. train dir + test dir)."""
    frames = []
    for root in roots:
        pattern = "*/*.csv" if _has_subdirs(root) else "*.csv"
        frames.append(load_directory(root, pattern))
    return _finalize(pd.concat(frames, ignore_index=True))


def _has_subdirs(root: str) -> bool:
    return any(os.path.isdir(os.path.join(root, e)) for e in os.listdir(root))


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows, de-duplicate, sort chronologically."""
    df = df[df["date"].notna() & df["home_team"].notna() & df["away_team"].notna()]
    df = df.drop_duplicates(subset="match_id", keep="first")
    # Result target must exist for a usable historical row
    df = df[df["fthg"].notna() & df["ftag"].notna()]
    df["fthg"] = df["fthg"].astype(int)
    df["ftag"] = df["ftag"].astype(int)
    # Derive FTR if missing
    ftr = np.where(df["fthg"] > df["ftag"], "H",
                   np.where(df["fthg"] < df["ftag"], "A", "D"))
    df["ftr"] = df["ftr"].fillna(pd.Series(ftr, index=df.index))
    df = df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    return df

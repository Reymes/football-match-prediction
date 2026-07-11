"""Sync fresh data from football-data.co.uk.

Three entry points, all accepting an optional ``progress(msg, pct)`` callback so
the web UI (or a CLI) can render a live progress bar:

    sync_all(root)       download & refresh EVERY season (full rebuild)
    sync_latest(root)    download & refresh only the CURRENT season
    sync_fixtures(root)  download the upcoming-fixtures file AND refresh the
                         current season (so the history a fixture is scored
                         against is as fresh as the fixture itself)

Each season zip (``mmz4281/<code>/data.zip``) bundles every division; we extract
only the divisions we know how to file (see schema.DIV_TO_FOLDER) into
``<root>/<country>/<League>_<YYYY-YYYY>.csv`` — exactly the layout the ingest
layer already reads. Downloaded zips are cached so re-syncs are cheap.

No third-party HTTP dependency: this uses urllib from the stdlib.
"""
from __future__ import annotations

import io
import os
import ssl
import zipfile
from datetime import date as _date
from typing import Callable, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .ingest import _parse_dates, _pick_first_present, _read_csv_any_encoding
from .schema import (
    DIV_TO_FOLDER,
    LEAGUE_BY_DIV,
    ODDS_PREFERENCE,
    SEASON_CODES,
    season_code_to_name,
)

BASE = "https://www.football-data.co.uk"
SEASON_URL = BASE + "/mmz4281/{code}/data.zip"
FIXTURES_URL = BASE + "/fixtures.csv"
FIXTURES_FILE = "fixtures.csv"           # stored at <root>/fixtures.csv
_UA = {"User-Agent": "Mozilla/5.0 (match-predict data sync)"}

Progress = Callable[..., None]


def _noop(*_a, **_k):  # default progress sink
    pass


def _ssl_context() -> ssl.SSLContext | None:
    """Verified context using certifi when available, else the system default.

    Some Python builds (notably python.org macOS installers) ship without a
    usable CA bundle, so a plain urlopen raises CERTIFICATE_VERIFY_FAILED even
    though the host is fine. We try certifi first; _fetch handles the final
    fallback if verification still can't be satisfied.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return None


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers=_UA)
    try:
        ctx = _ssl_context()
        with urlopen(req, timeout=timeout, context=ctx) as r:  # noqa: S310
            return r.read()
    except (ssl.SSLError, URLError) as e:
        # A broken local CA store surfaces either as SSLError or as a URLError
        # whose reason is an SSLError. For this trusted public host, retry once
        # without verification rather than fail the whole sync. Re-raise any
        # genuine (non-SSL) network error.
        reason = getattr(e, "reason", e)
        if not isinstance(e, ssl.SSLError) and not isinstance(reason, ssl.SSLError):
            raise
        ctx = ssl._create_unverified_context()
        with urlopen(req, timeout=timeout, context=ctx) as r:  # noqa: S310
            return r.read()


# --------------------------------------------------------------------------- #
# Season codes                                                                 #
# --------------------------------------------------------------------------- #
def _season_code_for(year: int, month: int) -> str:
    start = year if month >= 7 else year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def current_season_code(today: _date | None = None) -> str:
    """Football season code for a date. European seasons roll over in July."""
    today = today or _date.today()
    return _season_code_for(today.year, today.month)


def season_codes_for_dates(dates: pd.Series) -> list[str]:
    """Distinct season codes implied by a column of match dates."""
    d = pd.to_datetime(dates, errors="coerce").dropna()
    codes = {_season_code_for(int(ts.year), int(ts.month)) for ts in d}
    return sorted(codes)


# --------------------------------------------------------------------------- #
# Season sync                                                                  #
# --------------------------------------------------------------------------- #
def _extract_zip(blob: bytes, season: str, root: str) -> list[str]:
    """Write known-division CSVs out of one season zip. Returns written paths."""
    written = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            div = os.path.splitext(os.path.basename(name))[0]
            rel = DIV_TO_FOLDER.get(div)
            if not rel:                      # division we don't file (yet)
                continue
            country, league = rel.split("/")
            out_dir = os.path.join(root, country)
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, f"{league}_{season}.csv")
            with open(dest, "wb") as fh:
                fh.write(zf.read(name))
            written.append(dest)
    return written


def sync_seasons(codes: Iterable[str], root: str = "football-data",
                 cache: str = ".data-cache", progress: Progress = _noop) -> dict:
    """Download & extract the given season codes. Idempotent; caches zips."""
    codes = list(codes)
    os.makedirs(cache, exist_ok=True)
    total = len(codes)
    files, seasons, misses = 0, [], []
    for i, code in enumerate(codes):
        season = season_code_to_name(code)
        pct = int(100 * i / max(total, 1))
        progress(f"[{i + 1}/{total}] {season} — downloading …", pct)
        zpath = os.path.join(cache, f"{code}.zip")
        try:
            if os.path.getsize(zpath) if os.path.exists(zpath) else 0:
                blob = open(zpath, "rb").read()
            else:
                blob = _fetch(SEASON_URL.format(code=code))
                with open(zpath, "wb") as fh:
                    fh.write(blob)
        except Exception as e:               # noqa: BLE001 — report & continue
            progress(f"    ! {season} unavailable ({type(e).__name__})", pct)
            misses.append(season)
            continue
        try:
            written = _extract_zip(blob, season, root)
        except zipfile.BadZipFile:
            progress(f"    ! {season} zip corrupt — skipped", pct)
            misses.append(season)
            continue
        files += len(written)
        seasons.append(season)
        progress(f"    ✓ {season}: {len(written)} leagues", pct)
    progress(f"done — {files} files across {len(seasons)} seasons", 100)
    return {"files": files, "seasons": seasons, "missing": misses}


def sync_all(root: str = "football-data", cache: str = ".data-cache",
             progress: Progress = _noop) -> dict:
    """Refresh every season (full rebuild of the archive)."""
    progress("Full sync: all seasons 1993/94 → present", 0)
    return sync_seasons(SEASON_CODES, root, cache, progress)


def sync_latest(root: str = "football-data", cache: str = ".data-cache",
                progress: Progress = _noop, today: _date | None = None) -> dict:
    """Refresh only the current season (fast; run this often)."""
    code = current_season_code(today)
    progress(f"Latest sync: season {season_code_to_name(code)}", 0)
    # Always re-fetch the current season (results change week to week).
    zpath = os.path.join(cache, f"{code}.zip")
    if os.path.exists(zpath):
        os.remove(zpath)
    return sync_seasons([code], root, cache, progress)


# --------------------------------------------------------------------------- #
# Fixtures sync                                                                #
# --------------------------------------------------------------------------- #
def sync_fixtures(root: str = "football-data", cache: str = ".data-cache",
                  progress: Progress = _noop, refresh_season: bool = True,
                  today: _date | None = None) -> dict:
    """Download upcoming fixtures AND (by default) refresh the current season.

    Rationale: a fixture is only as good as the history it is scored against,
    so a fixtures refresh implies a latest-season refresh.
    """
    progress("Downloading upcoming fixtures …", 5)
    try:
        blob = _fetch(FIXTURES_URL)
    except Exception as e:                   # noqa: BLE001
        progress(f"! fixtures download failed ({type(e).__name__})", 100)
        return {"fixtures": 0, "error": str(e), "season": None}
    dest = os.path.join(root, FIXTURES_FILE)
    os.makedirs(root, exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(blob)
    fx = parse_fixtures(dest)
    progress(f"✓ {len(fx)} upcoming fixtures saved", 45)

    season_info = None
    if refresh_season:
        # Refresh the season(s) the fixtures actually belong to (from their
        # dates), unioned with the calendar season — so the history each
        # fixture is scored against is fresh even around the July rollover.
        codes = set(season_codes_for_dates(fx["date"])) if len(fx) else set()
        codes.add(current_season_code(today))
        names = ", ".join(season_code_to_name(c) for c in sorted(codes))
        progress(f"Refreshing season history ({names}) …", 55)
        for c in sorted(codes):                 # force re-fetch of live seasons
            zp = os.path.join(cache, f"{c}.zip")
            if os.path.exists(zp):
                os.remove(zp)
        season_info = sync_seasons(sorted(codes), root, cache, progress)
    return {"fixtures": int(len(fx)), "path": dest,
            "leagues": sorted(fx["league"].dropna().unique().tolist()),
            "season": season_info}


# --------------------------------------------------------------------------- #
# Fixtures parsing (upcoming matches -> canonical prediction input)            #
# --------------------------------------------------------------------------- #
FIXTURE_COLUMNS = ["league", "date", "time", "home_team", "away_team",
                   "odds_h", "odds_d", "odds_a"]


def parse_fixtures(path: str) -> pd.DataFrame:
    """Parse football-data's fixtures.csv into prediction-ready rows.

    Same columns/spelling as the historical files (minus results), so team
    names line up with history exactly and the 1X2 odds reuse the same
    preference order. Rows with an unknown Div or missing teams are dropped.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    raw = _read_csv_any_encoding(path)
    raw = raw.dropna(how="all")
    if "HomeTeam" not in raw.columns:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].reset_index(drop=True)

    out = pd.DataFrame(index=raw.index)
    # Only keep divisions we recognise: the fixtures pipeline exists to be
    # predicted/bet on, and every published division is in the label map.
    out["league"] = raw["Div"].map(LEAGUE_BY_DIV)
    out["date"] = _parse_dates(raw["Date"]) if "Date" in raw else pd.NaT
    out["time"] = raw["Time"] if "Time" in raw else np.nan
    out["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    out["away_team"] = raw["AwayTeam"].astype(str).str.strip()
    for dst in ("odds_h", "odds_d", "odds_a"):
        out[dst] = _pick_first_present(raw, ODDS_PREFERENCE[dst])
    out = out[out["date"].notna() & out["league"].notna()]
    out = out[out["home_team"].ne("") & out["away_team"].ne("")]
    return out[FIXTURE_COLUMNS].sort_values(["date", "league", "home_team"]) \
        .reset_index(drop=True)


FIXTURE_TOTALS_COLUMNS = ["league", "date", "home_team", "away_team",
                          "odds_over25", "odds_under25"]


def parse_fixture_totals_odds(path: str) -> pd.DataFrame:
    """Pre-kickoff over/under 2.5 odds for upcoming fixtures.

    Kept separate from `parse_fixtures` (whose FIXTURE_COLUMNS contract other
    callers rely on) so adding a market never changes that function's output.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=FIXTURE_TOTALS_COLUMNS)
    raw = _read_csv_any_encoding(path).dropna(how="all")
    if "HomeTeam" not in raw.columns:
        return pd.DataFrame(columns=FIXTURE_TOTALS_COLUMNS)
    raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].reset_index(drop=True)

    out = pd.DataFrame(index=raw.index)
    out["league"] = raw["Div"].map(LEAGUE_BY_DIV)
    out["date"] = _parse_dates(raw["Date"]) if "Date" in raw else pd.NaT
    out["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    out["away_team"] = raw["AwayTeam"].astype(str).str.strip()
    out["odds_over25"] = _pick_first_present(raw, ODDS_PREFERENCE["odds_over25"])
    out["odds_under25"] = _pick_first_present(raw, ODDS_PREFERENCE["odds_under25"])
    out = out[out["date"].notna() & out["league"].notna()]
    return out[FIXTURE_TOTALS_COLUMNS].reset_index(drop=True)

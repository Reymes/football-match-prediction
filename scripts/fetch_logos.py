#!/usr/bin/env python
"""Fetch real club & league crests into ``static/logos/`` — accurately.

The web UI works fully without this: it falls back to generated uniform SVG
badges (``match_predict/viz/logos.py``). This script *enriches* the UI with real
crests where a free source (TheSportsDB) has one, and — unlike a naive name
search — it verifies each hit against the club's expected country and matches
terse Football-Data spellings via aliases + fuzzy matching, so it does not grab
a same-named club from the wrong country.

    python scripts/fetch_logos.py --all                 # every team + league
    python scripts/fetch_logos.py --league England-PL   # one league (repeatable)
    python scripts/fetch_logos.py --all --recent 3      # only clubs seen lately
    python scripts/fetch_logos.py --leagues-only        # just competition crests
    python scripts/fetch_logos.py --report              # print coverage, no fetch

It is idempotent (skips files already present unless ``--force``), polite to the
free API (``--sleep``), offline-safe (a network miss is recorded, never fatal),
and writes ``static/logos/manifest.json`` recording exactly which crests are real
vs generated so coverage is honest and auditable.

Note: crests are trademarks of their clubs/leagues; use for personal/local
display only.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from urllib.request import Request, urlopen

from match_predict.data import load_all
from match_predict.viz import slugify
from match_predict.viz.logo_fetch import (
    LEAGUE_TSDB,
    league_dest,
    resolve_league_badge,
    resolve_team,
    team_dest,
)

ROOT = os.path.join("static", "logos")
MANIFEST = os.path.join(ROOT, "manifest.json")
_UA = {"User-Agent": "Mozilla/5.0 (match-predict logo fetch)"}


def _ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to unverified on odd setups
        return ssl._create_unverified_context()


def _bytes(url: str) -> bytes:
    with urlopen(  # noqa: S310 - fixed https host
        Request(url, headers=_UA), timeout=30, context=_ctx()
    ) as r:
        return r.read()


# TheSportsDB's free tier throttles by returning an *empty* 200 body (not a 429)
# once you burst. An empty result is therefore ambiguous — a real miss or a
# throttle — so we back off and retry: a throttled call recovers, while a
# genuine miss just costs the (bounded) wait before we accept it as a miss.
_BACKOFFS = (6.0, 15.0)


def _json(url: str) -> dict:
    data = json.loads(_bytes(url))
    if _has_payload(data):
        return data
    for wait in _BACKOFFS:
        time.sleep(wait)
        data = json.loads(_bytes(url))
        if _has_payload(data):
            return data
    return data


def _has_payload(data: dict) -> bool:
    return bool((data or {}).get("teams") or (data or {}).get("leagues"))


def _optimize(path: str, max_px: int = 128) -> None:
    """Downscale/normalise a saved crest to a small square PNG if PIL is here."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - optimisation is optional
        return
    with Image.open(path) as im:
        im = im.convert("RGBA")
        if max(im.size) > max_px:
            im.thumbnail((max_px, max_px), Image.LANCZOS)
        im.save(path, "PNG")


def _team_country(leagues: list[str]) -> dict[str, str]:
    """Map every in-scope team to the country of the league it played in most.

    A club can appear across tiers of one country; the modal league fixes a
    single expected country for verification (rarely ambiguous within a nation).
    """
    df = load_all("football-data")
    df = df[df["league"].isin(leagues)]
    country: dict[str, str] = {}
    counts: dict[str, int] = {}
    for league, sub in df.groupby("league"):
        ref = LEAGUE_TSDB.get(league)
        if ref is None:
            continue
        names = set(sub["home_team"]).union(sub["away_team"])
        for n in names:
            n = str(n)
            if not n.strip():
                continue
            c = int((sub["home_team"] == n).sum() + (sub["away_team"] == n).sum())
            if c > counts.get(n, 0):
                counts[n], country[n] = c, ref.country
    return country


def _recent_teams(leagues: list[str], seasons: int) -> set[str] | None:
    """Teams seen in the last ``seasons`` seasons, or None to keep everyone."""
    if not seasons:
        return None
    df = load_all("football-data")
    df = df[df["league"].isin(leagues)]
    keep_codes = sorted(df["season"].dropna().unique())[-seasons:]
    df = df[df["season"].isin(keep_codes)]
    return set(map(str, df["home_team"])).union(map(str, df["away_team"]))


def _download(url: str, dest: str, optimize: bool) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(_bytes(url))
    if optimize:
        _optimize(dest)


def fetch_leagues(leagues: list[str], args) -> list[dict]:
    records: list[dict] = []
    for league in leagues:
        dest = league_dest(league, ROOT)
        rec = {"league": league, "slug": slugify(league), "dest": dest}
        if os.path.exists(dest) and not args.force:
            rec["status"] = "cached"
            records.append(rec)
            print(f"  [league] · {league} (cached)", flush=True)
            continue
        url = resolve_league_badge(league, _json)
        if not url:
            rec["status"] = "missing"
            print(f"  [league] ✗ {league} (no crest)", flush=True)
        else:
            try:
                _download(url, dest, args.optimize)
                rec.update(status="ok", source=url)
                print(f"  [league] ✓ {league}", flush=True)
            except Exception as e:  # noqa: BLE001 - record, keep going
                rec.update(status="error", error=type(e).__name__)
                print(f"  [league] ! {league}: {type(e).__name__}", flush=True)
        records.append(rec)
        time.sleep(args.sleep)
    return records


def fetch_teams(names: list[str], country: dict[str, str], args) -> list[dict]:
    records: list[dict] = []
    total = len(names)
    for i, name in enumerate(names, 1):
        dest = team_dest(name, ROOT)
        rec = {"team": name, "slug": slugify(name), "dest": dest}
        if os.path.exists(dest) and not args.force:
            rec["status"] = "cached"
            records.append(rec)
            print(f"  [{i}/{total}] · {name} (cached)", flush=True)
            continue
        expected = country.get(name)
        match = (
            resolve_team(name, expected, _json, threshold=args.threshold)
            if expected
            else None
        )
        if match is None:
            rec["status"] = "missing"
            print(f"  [{i}/{total}] ✗ {name} (no confident match)", flush=True)
        else:
            try:
                _download(match.badge_url, dest, args.optimize)
                rec.update(
                    status="ok",
                    matched_name=match.matched_name,
                    score=match.score,
                    via_alias=match.via_alias,
                    source=match.badge_url,
                )
                tag = "alias" if match.via_alias else f"{match.score:.2f}"
                print(
                    f"  [{i}/{total}] ✓ {name} -> {match.matched_name} ({tag})",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                rec.update(status="error", error=type(e).__name__)
                print(f"  [{i}/{total}] ! {name}: {type(e).__name__}", flush=True)
        records.append(rec)
        time.sleep(args.sleep)
    return records


def _summary(records: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in records:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return {"kind": key, "total": len(records), "by_status": out}


def _write_manifest(team_recs, league_recs) -> dict:
    manifest = {
        "note": "Real crests fetched from TheSportsDB (free tier). Missing "
        "entries fall back to generated SVG badges; nothing here is fabricated.",
        "teams": {"summary": _summary(team_recs, "teams"), "items": team_recs},
        "leagues": {
            "summary": _summary(league_recs, "leagues"),
            "items": league_recs,
        },
        "misses": {
            "teams": sorted(
                r["team"] for r in team_recs if r["status"] == "missing"
            ),
            "leagues": sorted(
                r["league"] for r in league_recs if r["status"] == "missing"
            ),
        },
    }
    os.makedirs(ROOT, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def _print_report() -> int:
    if not os.path.exists(MANIFEST):
        print(f"No manifest at {MANIFEST}; run a fetch first.")
        return 1
    with open(MANIFEST) as fh:
        m = json.load(fh)
    for kind in ("leagues", "teams"):
        s = m[kind]["summary"]
        print(f"{kind:8}: {s['by_status']}  (of {s['total']})")
    if m["misses"]["teams"]:
        print(f"\nUnresolved teams ({len(m['misses']['teams'])}):")
        print("  " + ", ".join(m["misses"]["teams"]))
    if m["misses"]["leagues"]:
        print(f"\nUnresolved leagues: {', '.join(m['misses']['leagues'])}")
    return 0


def _target_leagues(args) -> list[str]:
    if args.all or (not args.league):
        return list(LEAGUE_TSDB)
    unknown = [l for l in args.league if l not in LEAGUE_TSDB]
    if unknown:
        raise SystemExit(f"Unknown league label(s): {unknown}. "
                         f"Valid: {', '.join(LEAGUE_TSDB)}")
    return args.league


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", action="append", help="league label(s), repeatable")
    ap.add_argument("--all", action="store_true", help="every league in the archive")
    ap.add_argument("--recent", type=int, default=0,
                    help="only clubs seen in the last N seasons (0 = all)")
    ap.add_argument("--teams-only", action="store_true")
    ap.add_argument("--leagues-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--optimize", action="store_true",
                    help="downscale crests to 128px PNG (needs Pillow)")
    ap.add_argument("--threshold", type=float, default=0.72,
                    help="fuzzy-match acceptance bar (0-1)")
    ap.add_argument("--sleep", type=float, default=2.5,
                    help="delay between calls (free tier throttles if too fast)")
    ap.add_argument("--report", action="store_true",
                    help="print coverage from an existing manifest and exit")
    args = ap.parse_args(argv)

    if args.report:
        return _print_report()

    leagues = _target_leagues(args)

    league_recs: list[dict] = []
    if not args.teams_only:
        print(f"Fetching {len(leagues)} league crests -> {ROOT}/league")
        league_recs = fetch_leagues(leagues, args)

    team_recs: list[dict] = []
    if not args.leagues_only:
        country = _team_country(leagues)
        names = sorted(country)
        recent = _recent_teams(leagues, args.recent)
        if recent is not None:
            names = [n for n in names if n in recent]
        print(f"Fetching crests for {len(names)} teams -> {ROOT}/team")
        team_recs = fetch_teams(names, country, args)

    m = _write_manifest(team_recs, league_recs)
    ts = m["teams"]["summary"]["by_status"]
    ls = m["leagues"]["summary"]["by_status"]
    ok_t = ts.get("ok", 0) + ts.get("cached", 0)
    ok_l = ls.get("ok", 0) + ls.get("cached", 0)
    print(f"\nDone. Teams: {ok_t}/{m['teams']['summary']['total']} crests; "
          f"Leagues: {ok_l}/{m['leagues']['summary']['total']} crests.")
    print(f"Manifest: {MANIFEST}  (run --report for the miss list)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

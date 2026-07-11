#!/usr/bin/env python3
"""Download all La Liga Primera Division (SP1) CSV files from football-data.co.uk."""

import urllib.request
from pathlib import Path

SEASONS = [
    "2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617",
    "1516", "1415", "1314", "1213", "1112", "1011", "0910", "0809", "0708", "0607",
    "0506", "0405", "0304", "0203", "0102", "0001", "9900", "9899", "9798", "9697",
    "9596", "9495", "9394",
]

BASE_URL = "https://www.football-data.co.uk/mmz4281"
OUT_DIR = Path(__file__).parent / "data" / "la_liga_primera"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.football-data.co.uk/spainm.php",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0

    for season in SEASONS:
        url = f"{BASE_URL}/{season}/SP1.csv"
        out = OUT_DIR / f"SP1_{season}.csv"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                out.write_bytes(resp.read())
            print(f"OK: {out.name} ({out.stat().st_size} bytes)")
            ok += 1
        except Exception as e:
            print(f"FAILED: {url} - {e}")
            if out.exists():
                out.unlink()
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed")


if __name__ == "__main__":
    main()

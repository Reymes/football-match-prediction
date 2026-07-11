#!/usr/bin/env bash
#
# Refresh the football-data.co.uk archive.
#
# Downloads every season's "main leagues" zip, unzips it, and drops each
# division's CSV into football-data/<country>/<League>_YYYY-YYYY.csv using the
# same naming the ingest layer expects (league label is taken from the Div
# column inside each file, so the folder name is cosmetic).
#
# Idempotent: re-running only re-downloads seasons whose zip is missing from
# the cache and overwrites CSVs in place. To pick up the latest fixtures of the
# current season, delete the current season's cached zip first, e.g.:
#     rm "$CACHE/2526.zip" && scripts/refresh_data.sh
#
# Usage:
#     scripts/refresh_data.sh                 # all seasons 1993/94 -> 2025/26
#     SEASONS="2526 2425" scripts/refresh_data.sh   # just these season codes
#
# After refreshing, retrain so new data reaches the models:
#     python scripts/train.py
set -euo pipefail

BASE="https://www.football-data.co.uk/mmz4281"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/football-data"
CACHE="${CACHE:-$ROOT/.data-cache}"
mkdir -p "$CACHE"

# All season codes football-data currently publishes, newest first.
SEASONS="${SEASONS:-2526 2425 2324 2223 2122 2021 1920 1819 1718 1617 1516 1415 1314 1213 1112 1011 0910 0809 0708 0607 0506 0405 0304 0203 0102 0001 9900 9899 9798 9697 9596 9495 9394}"

map_div() {
  case "$1" in
    E0)  echo "england/PremierLeague" ;;
    E1)  echo "england/Championship" ;;
    E2)  echo "england/LeagueOne" ;;
    E3)  echo "england/LeagueTwo" ;;
    EC)  echo "england/NationalLeague" ;;
    SC0) echo "scotland/Premiership" ;;
    SC1) echo "scotland/Championship" ;;
    SC2) echo "scotland/LeagueOne" ;;
    SC3) echo "scotland/LeagueTwo" ;;
    D1)  echo "germany/Bundesliga" ;;
    D2)  echo "germany/Bundesliga2" ;;
    I1)  echo "italy/SerieA" ;;
    I2)  echo "italy/SerieB" ;;
    SP1) echo "spain/LaLiga" ;;
    SP2) echo "spain/LaLiga2" ;;
    F1)  echo "france/Ligue1" ;;
    F2)  echo "france/Ligue2" ;;
    N1)  echo "netherlands/Eredivisie" ;;
    B1)  echo "belgium/ProLeague" ;;
    P1)  echo "portugal/PrimeiraLiga" ;;
    T1)  echo "turkey/SuperLig" ;;
    G1)  echo "greece/SuperLeague" ;;
    *)   echo "" ;;   # unknown div -> skipped (ingest still loads it via Div)
  esac
}

season_name() {
  local c="$1" a b y1 y2
  a="${c:0:2}"; b="${c:2:2}"
  if [ "$a" -ge 93 ]; then y1="19$a"; else y1="20$a"; fi
  if [ "$b" -ge 93 ]; then y2="19$b"; else y2="20$b"; fi
  echo "${y1}-${y2}"
}

n=0
for code in $SEASONS; do
  zip="$CACHE/$code.zip"
  if [ ! -s "$zip" ]; then
    if ! curl -fsS -A "Mozilla/5.0" -o "$zip" "$BASE/$code/data.zip"; then
      echo "MISS  $code (no zip published)"; rm -f "$zip"; continue
    fi
  fi
  season="$(season_name "$code")"
  ex="$CACHE/ex_$code"; rm -rf "$ex"; mkdir -p "$ex"
  unzip -oq "$zip" -d "$ex" || { echo "BADZIP $code"; continue; }
  for f in "$ex"/*.csv "$ex"/*.CSV; do
    [ -f "$f" ] || continue
    div="$(basename "$f" | sed -E 's/\.[cC][sS][vV]$//')"
    rel="$(map_div "$div")"; [ -n "$rel" ] || continue
    country="${rel%%/*}"; league="${rel##*/}"
    mkdir -p "$DEST/$country"
    cp "$f" "$DEST/$country/${league}_${season}.csv"
    n=$((n+1))
  done
  rm -rf "$ex"
  echo "OK    $code -> $season"
done
echo "Wrote $n CSV files into $DEST"

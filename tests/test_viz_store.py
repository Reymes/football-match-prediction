"""Logo generation + SQLite store (sync log + fixture-prediction cache)."""
from match_predict.viz import (
    slugify, initials, team_badge_svg, league_badge_svg, palette_for,
)
from match_predict.store import Store


# --------------------------------------------------------------- logos
def test_slugify():
    assert slugify("Paris SG") == "paris-sg"
    assert slugify("Nott'm Forest") == "nott-m-forest"
    assert slugify("") == "na"


def test_initials():
    assert initials("Manchester United") == "MU"
    assert initials("Arsenal") == "ARS"
    assert initials("Real Madrid") == "RM"


def test_team_badge_is_valid_svg():
    svg = team_badge_svg("Barcelona", size=48)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert 'viewBox="0 0 64 64"' in svg          # uniform box -> uniform size
    assert 'width="48"' in svg


def test_league_badge_is_valid_svg():
    svg = league_badge_svg("Spain-LL2", "Spain · La Liga 2")
    assert svg.startswith("<svg") and 'viewBox="0 0 64 64"' in svg


def test_palette_is_deterministic():
    assert palette_for("Chelsea") == palette_for("Chelsea")
    assert palette_for("Chelsea") != palette_for("Everton")


# --------------------------------------------------------------- store
def test_sync_log_roundtrip(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.record_sync("latest", files=22, seasons=1, detail={"seasons": ["2025-2026"]})
    last = s.last_sync()
    assert last["kind"] == "latest" and last["files"] == 22
    assert last["detail"]["seasons"] == ["2025-2026"]
    assert len(s.recent_syncs()) == 1


def test_prediction_cache_keyed_by_model(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    preds = [{"match_id": "England-PL|2025-2026|20250815|A|B|UPCOMING",
              "prob_home": 0.5}]
    s.put_predictions("2025-08-14", preds)
    hit = s.get_predictions("2025-08-14", [preds[0]["match_id"]])
    assert hit[preds[0]["match_id"]]["prob_home"] == 0.5
    # different model stamp -> cache miss (auto-invalidation on retrain)
    assert s.get_predictions("2025-09-01", [preds[0]["match_id"]]) == {}


def test_clear_cache(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.put_predictions("m1", [{"match_id": "x", "p": 1}])
    s.clear_cache()
    assert s.get_predictions("m1", ["x"]) == {}

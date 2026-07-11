"""Market-validation profiles: persistence + the eligibility gate they open
(bet.md §10, §18). A market is only evaluable once it carries a passed-quality
out-of-time profile; otherwise every selection is rejected as unvalidated."""
import numpy as np

from match_predict.decisions import eligibility, load_config
from match_predict.decisions.schema import RejectionCode
from match_predict.decisions.validation import (
    BandProfile, MarketValidationProfile, build_1x2_profile_from_probs,
    load_profiles, profile_from_dict, save_profiles)


def _passing_profile():
    return MarketValidationProfile(
        market="match_winner", n_samples=500, log_loss=0.98, brier=0.19,
        ece=0.03, passed_quality=True,
        bands=[BandProfile(lo=0.4, hi=0.6, n_samples=400,
                           calibration_error=0.02, empirical_rate=0.5,
                           predicted_rate=0.5)])


def _evaluate(profile):
    cfg = load_config()
    cfg["require_timestamped_odds"] = False        # timestamp tested elsewhere
    return eligibility.evaluate_selection(
        market="match_winner", selection="H", side_or_line="H",
        model_probability=0.5, offered_odds=2.2, outcome_odds_set=[2.2, 1.9],
        decision_cutoff=None, odds_timestamp=None, config=cfg,
        market_profile=profile, data_quality=1.0, model_disagreement=0.0)


def test_no_profile_is_rejected_as_unvalidated():
    dec = _evaluate(None)
    assert RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE in dec.rejection_reasons


def test_passing_profile_opens_the_sample_gate():
    dec = _evaluate(_passing_profile())
    assert RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE not in dec.rejection_reasons


def test_failed_profile_still_rejected():
    prof = _passing_profile()
    prof.passed_quality = False
    dec = _evaluate(prof)
    assert RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE in dec.rejection_reasons


def test_save_load_round_trip(tmp_path):
    prof = _passing_profile()
    path = str(tmp_path / "profiles.json")
    save_profiles({"match_winner": prof}, path, meta={"val_start": "2024-08-01"})
    loaded = load_profiles(path)
    assert "match_winner" in loaded
    got = loaded["match_winner"]
    assert got.passed_quality and got.n_samples == 500
    assert len(got.bands) == 1 and got.bands[0].calibration_error == 0.02
    # the loaded profile drives eligibility identically to the in-memory one
    assert RejectionCode.INSUFFICIENT_HISTORICAL_SAMPLE not in _evaluate(got).rejection_reasons


def test_load_missing_file_is_empty():
    assert load_profiles("does/not/exist.json") == {}


def test_profile_from_dict_matches_build(tmp_path):
    rng = np.random.default_rng(0)
    # well-calibrated synthetic 1X2: draw labels from the probabilities
    proba = rng.dirichlet([3, 2, 2], size=800)
    y = np.array([rng.choice(3, p=row) for row in proba])
    prof = build_1x2_profile_from_probs(proba, y, min_samples=300)
    round_tripped = profile_from_dict(prof.to_dict())
    assert round_tripped.passed_quality == prof.passed_quality
    assert round_tripped.n_samples == prof.n_samples
    assert len(round_tripped.bands) == len(prof.bands)

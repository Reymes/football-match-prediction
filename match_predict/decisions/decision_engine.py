"""Decision engine — the main advisory entry point (bet.md §20, §2, §11, §12).

`evaluate_match` accepts everything known at the decision cutoff and returns a
`MatchDecision` containing:
  * the three independent probability views (pure / market / hybrid);
  * a `SelectionDecision` per candidate selection (fully graded, with rejection
    codes, evidence and warnings);
  * the top-scoreline distribution (always reported, even when no bet qualifies).

It never places a bet, never claims certainty, and treats "No bet" as normal.

Score-dependent markets (over/under, BTTS, correct score) are all derived from
the SAME validated joint score matrix (bet.md §4) via markets.derive_markets,
so their probabilities can never contradict one another. The pure view uses NO
bookmaker information.
"""
from __future__ import annotations

from . import devig as devig_mod
from . import eligibility, grading, correlation, exposure
from . import uncertainty as unc_mod
from .schema import (MatchDecision, ThreeViews, DecisionStatus, load_config)


# --------------------------------------------------------------------------- #
# View construction.                                                          #
# --------------------------------------------------------------------------- #
def build_views(pure_hda: dict, hybrid_hda: dict,
                outcome_odds_1x2=None, devig_method: str = "shin") -> ThreeViews:
    """Assemble the three HDA views. `pure_hda` must contain NO odds info."""
    market = None
    if outcome_odds_1x2 is not None:
        try:
            fair = devig_mod.devig(outcome_odds_1x2, devig_method)
            market = {"H": float(fair[0]), "D": float(fair[1]), "A": float(fair[2])}
        except (ValueError, IndexError):
            market = None
    return ThreeViews(pure=pure_hda, market=market, hybrid=hybrid_hda)


# --------------------------------------------------------------------------- #
# Candidate selections from a validated score matrix + views.                 #
# --------------------------------------------------------------------------- #
def _two_way_set(this_odd, other_odd):
    """Odds set for a 2-outcome market with the candidate FIRST (devig[0]).

    Index-based (not value-based) so equal prices — e.g. 1.95/1.95 — are not
    collapsed into a single entry.
    """
    if other_odd is None:
        return None
    return [this_odd, other_odd]


def candidate_selections(views: ThreeViews, market_book, prices: dict) -> list:
    """Build (market, selection, side, model_prob, offered_odds, outcome_set)
    tuples for every priced, config-known market. All score-derived probs come
    from `market_book` (a MarketBook read off ONE joint matrix).

    `prices` maps market -> {selection: offered_odds}. The hybrid view supplies
    the model probability for 1X2; score markets use the reconciled matrix.
    """
    cands = []
    hy = views.hybrid

    # 1X2 -----------------------------------------------------------------
    if "match_winner" in prices:
        p = prices["match_winner"]
        odds_set = [p.get("H"), p.get("D"), p.get("A")]
        for sel, key in (("H", 0), ("D", 1), ("A", 2)):
            if p.get(sel) is not None:
                model_p = {"H": hy["H"], "D": hy["D"], "A": hy["A"]}[sel]
                # rotate the outcome set so the candidate is first (devig[0])
                rot = [odds_set[key]] + [o for i, o in enumerate(odds_set) if i != key]
                cands.append(dict(market="match_winner", selection=sel,
                                  side_or_line=sel, model_probability=model_p,
                                  offered_odds=p.get(sel), outcome_odds_set=rot))

    # over/under 2.5 ------------------------------------------------------
    if "over_under_2_5" in prices:
        p = prices["over_under_2_5"]
        ou = market_book.over_under.get(2.5, {})
        pairs = [("OVER_2.5", ou.get("over"), "OVER"),
                 ("UNDER_2.5", ou.get("under"), "UNDER")]
        for sel, mp, key in pairs:
            odd = p.get(key)
            if odd is not None and mp is not None:
                other = "UNDER" if key == "OVER" else "OVER"
                rot = _two_way_set(odd, p.get(other))
                cands.append(dict(market="over_under_2_5", selection=sel,
                                  side_or_line=sel, model_probability=mp,
                                  offered_odds=odd, outcome_odds_set=rot))

    # BTTS ----------------------------------------------------------------
    if "btts" in prices:
        p = prices["btts"]
        pairs = [("BTTS_YES", market_book.btts.get("yes"), "YES"),
                 ("BTTS_NO", market_book.btts.get("no"), "NO")]
        for sel, mp, key in pairs:
            odd = p.get(key)
            if odd is not None and mp is not None:
                other = "NO" if key == "YES" else "YES"
                rot = _two_way_set(odd, p.get(other))
                cands.append(dict(market="btts", selection=sel,
                                  side_or_line=sel, model_probability=mp,
                                  offered_odds=odd, outcome_odds_set=rot))

    # correct score (disabled by default; still evaluated so it can be reported
    # and correctly REJECTED — never auto-selected, bet.md §11) --------------
    if "correct_score" in prices:
        cs_probs = {f"{i}-{j}": pr for (i, j), pr in market_book.correct_score}
        for sel, odd in prices["correct_score"].items():
            if odd is None:
                continue
            mp = cs_probs.get(sel)
            if mp is None:
                continue
            # correct score has no clean 2-outcome de-vig set; use raw implied
            cands.append(dict(market="correct_score", selection=sel,
                              side_or_line=sel, model_probability=mp,
                              offered_odds=odd, outcome_odds_set=None))
    return cands


# --------------------------------------------------------------------------- #
# Main entry.                                                                 #
# --------------------------------------------------------------------------- #
def evaluate_match(
    *,
    fixture_id: str,
    league: str,
    home_team: str,
    away_team: str,
    kickoff: str,
    views: ThreeViews,
    market_book,                    # MarketBook read off the validated matrix
    prices: dict,                   # market -> {selection: offered_odds}
    config: dict | None = None,
    market_profiles: dict | None = None,
    data_quality: float = 1.0,
    decision_time: str | None = None,
    horizon: str | None = None,
    odds_timestamp=None,
    model_version: str | None = None,
    ledger: "exposure.ExposureLedger | None" = None,
    supported_league: bool = True,
    team_resolved: bool = True,
    in_distribution: bool = True,
    fixture_stale: bool = False,
) -> MatchDecision:
    """Evaluate every priced market for one fixture and return a MatchDecision."""
    config = config or load_config()
    market_profiles = market_profiles or {}

    # model disagreement: pure vs hybrid HDA (bet.md §7). If no market view,
    # disagreement is measured pure-vs-hybrid only.
    disagree = unc_mod.model_disagreement(views.pure, views.hybrid)

    cands = candidate_selections(views, market_book, prices)
    decisions = []
    for c in cands:
        prof = market_profiles.get(c["market"])
        dec = eligibility.evaluate_selection(
            market=c["market"], selection=c["selection"],
            side_or_line=c["side_or_line"],
            model_probability=c["model_probability"],
            offered_odds=c["offered_odds"],
            outcome_odds_set=c["outcome_odds_set"],
            decision_cutoff=decision_time, odds_timestamp=odds_timestamp,
            config=config, market_profile=prof, data_quality=data_quality,
            model_disagreement=disagree, supported_league=supported_league,
            team_resolved=team_resolved, in_distribution=in_distribution,
            model_version=model_version, fixture_stale=fixture_stale)
        grading.grade_selection(dec, config, prof)
        decisions.append(dec)

    # correlation grouping + one primary per group/match (bet.md §13)
    groups = correlation.group_selections(fixture_id, decisions)
    max_primary = config.get("exposure", {}).get(
        "maximum_primary_selections_per_match", 1)
    correlation.select_primaries(decisions, groups, max_primary)

    # exposure admission (bet.md §14) — only if a ledger is supplied
    if ledger is not None:
        for dec in decisions:
            ledger.admit(dec, league, fixture_id)

    top_scores = [(f"{i}-{j}", round(float(pr), 4))
                  for (i, j), pr in market_book.top_scores(5)]

    return MatchDecision(
        fixture_id=fixture_id, league=league, home_team=home_team,
        away_team=away_team, kickoff=kickoff, decision_time=decision_time,
        horizon=horizon, views=views, selections=decisions,
        top_scores=top_scores, data_quality=data_quality,
        model_disagreement=round(disagree, 4), model_version=model_version)


def summarize_day(match_decisions: list) -> dict:
    """Research summary across a set of match decisions (bet.md §22)."""
    qualified = sum(len(m.qualifying()) for m in match_decisions)
    evaluated = sum(len(m.selections) for m in match_decisions)
    no_bet = sum(1 for m in match_decisions for s in m.selections
                 if s.decision_status == DecisionStatus.NO_BET.value)
    watch = sum(1 for m in match_decisions for s in m.selections
                if s.decision_status == DecisionStatus.WATCHLIST.value)
    rej_counts: dict = {}
    for m in match_decisions:
        for s in m.selections:
            for r in s.rejection_reasons:
                code = r.value if hasattr(r, "value") else r
                rej_counts[code] = rej_counts.get(code, 0) + 1
    return {
        "fixtures": len(match_decisions),
        "selections_evaluated": evaluated,
        "qualified": qualified, "watchlist": watch, "no_bet": no_bet,
        "rejection_reason_counts": dict(sorted(
            rej_counts.items(), key=lambda kv: -kv[1])),
    }

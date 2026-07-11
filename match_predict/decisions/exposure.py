"""Conservative exposure controls & optional fractional-Kelly (bet.md §14, §15).

All controls are configurable research safeguards. The engine is ADVISORY: it
never places a bet, never increases a stake after a loss, and implements no
martingale or doubling system. "No bet" is always acceptable.

`ExposureLedger` tracks hypothetical exposure within a single decision day and
enforces:
  * maximum qualified selections per day;
  * maximum primary selections per match;
  * maximum hypothetical fraction per selection / per day / per league /
    per correlated group.

Fractional Kelly (bet.md §15) is OFF by default. When explicitly enabled it uses
the CONSERVATIVE probability, sets negative Kelly to zero, applies a fraction
(e.g. quarter-Kelly) and a hard cap, and is labelled a hypothetical research
output only.
"""
from __future__ import annotations

from collections import defaultdict

from .schema import RejectionCode, DecisionStatus


def full_kelly_fraction(conservative_probability: float, offered_odds: float) -> float:
    """Full-Kelly stake fraction for decimal odds. Negative -> 0 (no bet)."""
    b = float(offered_odds) - 1.0
    if b <= 0:
        return 0.0
    p = float(conservative_probability)
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def hypothetical_stake_fraction(dec, staking_cfg: dict) -> float | None:
    """Optional research-only stake fraction (bet.md §15). None when disabled.

    * uses the CONSERVATIVE probability, never the raw probability;
    * negative Kelly -> 0;
    * applies kelly_fraction (e.g. quarter-Kelly);
    * clamped by a hard maximum allocation.
    """
    if not staking_cfg.get("enabled", False):
        return None
    if dec.conservative_probability is None or dec.offered_odds is None:
        return 0.0
    full = full_kelly_fraction(dec.conservative_probability, dec.offered_odds)
    frac = full * float(staking_cfg.get("kelly_fraction", 0.25))
    return float(min(frac, staking_cfg.get("hard_max_fraction", 0.005)))


class ExposureLedger:
    """Enforces daily/match/league/group exposure limits for a decision day."""

    def __init__(self, exposure_cfg: dict, staking_cfg: dict | None = None):
        self.cfg = exposure_cfg
        self.staking = staking_cfg or {"enabled": False}
        self.qualified_today = 0
        self.match_primaries: dict = defaultdict(int)
        self.daily_fraction = 0.0
        self.league_fraction: dict = defaultdict(float)
        self.group_fraction: dict = defaultdict(float)

    def admit(self, dec, league: str, match_id: str) -> bool:
        """Try to admit a primary qualifying selection. On rejection, append the
        relevant code, demote to NO BET, and return False.

        Only QUALIFIED / STRONG-EVIDENCE primaries consume exposure; watchlist
        and rejected selections are informational and never staked.
        """
        if dec.decision_status not in (DecisionStatus.QUALIFIED.value,
                                        DecisionStatus.STRONG_EVIDENCE.value):
            return False
        if not dec.is_primary:
            return False

        max_day = self.cfg.get("maximum_qualified_selections_per_day", 3)
        max_match = self.cfg.get("maximum_primary_selections_per_match", 1)
        per_sel = self.cfg.get("maximum_hypothetical_fraction_per_selection", 0.005)
        max_daily = self.cfg.get("maximum_hypothetical_daily_fraction", 0.015)
        max_league = self.cfg.get("maximum_league_daily_fraction", 0.010)
        max_group = self.cfg.get("maximum_correlated_group_fraction", 0.005)

        # hypothetical stake for this selection (or the per-selection cap)
        stake = hypothetical_stake_fraction(dec, self.staking)
        stake = per_sel if stake is None else min(stake, per_sel)
        grp = dec.correlation_group or f"{match_id}::solo"

        over_limit = (
            self.qualified_today >= max_day
            or self.match_primaries[match_id] >= max_match
            or self.daily_fraction + stake > max_daily + 1e-12
            or self.league_fraction[league] + stake > max_league + 1e-12
            or self.group_fraction[grp] + stake > max_group + 1e-12
        )
        if over_limit:
            dec.rejection_reasons.append(RejectionCode.DAILY_EXPOSURE_LIMIT)
            dec.decision_status = DecisionStatus.NO_BET.value
            dec.is_primary = False
            return False

        # admit
        self.qualified_today += 1
        self.match_primaries[match_id] += 1
        self.daily_fraction += stake
        self.league_fraction[league] += stake
        self.group_fraction[grp] += stake
        if self.staking.get("enabled", False):
            dec.hypothetical_kelly_fraction = round(stake, 5)
        return True

    def summary(self) -> dict:
        return {
            "qualified_today": self.qualified_today,
            "daily_hypothetical_fraction": round(self.daily_fraction, 5),
            "league_fraction": {k: round(v, 5) for k, v in self.league_fraction.items()},
        }

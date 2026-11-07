"""
Regime-transition cooldown.

A detector with hard thresholds (ADX >= 20, BBWidth <= 0.06, ...) will flicker
when a market sits on a boundary: TREND today, RANGE tomorrow, TREND again.
Under strict regime discipline every flip is a forced sell plus a re-buy, so a
flickering regime is a fee machine.

`RegimeCooldown` decides whether the *current* regime has been stable long
enough to trust with new capital: the latest label must have been observed on
`min_periods` consecutive bars.  Until then the regime is "in cooldown" --
open positions are still protected (the orchestrator flattens an ineligible
owner immediately) but no new positions are opened.

The assessment is computed from the detector's per-bar label history, so it
needs no memory between cycles: it is restart-safe and a backtester can
reproduce it bar by bar.

Regime families
---------------
Not every label change is a change of *market character*.  TREND and
VOL_BREAKOUT are both directional: a breakout bar is typically labelled
VOL_BREAKOUT after a run of TREND bars, and treating that as a brand-new
regime would make every breakout entry wait out the cooldown.  Labels in the
same family therefore share one streak -- switching between them neither
resets the clock nor blocks allocation.  The cooldown bites only on shifts
*between* families (directional <-> RANGE, NORMAL, QUIET, unknown ...).
Labels not listed in any family are families of their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

from regimes.detector import normalise_regime

DEFAULT_MIN_STABLE_PERIODS = 3

# family name -> labels that count as the same regime for stability purposes
DEFAULT_REGIME_FAMILIES: Dict[str, tuple] = {
    "DIRECTIONAL": ("TREND", "VOL_BREAKOUT"),
}


@dataclass(frozen=True)
class RegimeStability:
    """
    How settled the latest regime is.

    regime   -- latest label (None if missing / undetectable)
    family   -- family the latest label belongs to (the label itself when it
               is not grouped; None if regime is None)
    streak   -- consecutive trailing bars whose label is in `family`
               (0 if regime is None)
    required -- bars needed before the regime is trusted with new capital
    previous -- label observed immediately before the streak began, if any
               bar in the window shows one (None = no change visible in window)
    """

    regime: Optional[str]
    streak: int
    required: int
    previous: Optional[str] = None
    family: Optional[str] = None

    def __post_init__(self):
        if self.family is None and self.regime is not None:
            object.__setattr__(self, "family", self.regime)

    @property
    def stable(self) -> bool:
        return self.regime is not None and self.streak >= self.required

    @property
    def in_cooldown(self) -> bool:
        return not self.stable

    @property
    def remaining(self) -> int:
        """Bars still needed (0 when stable)."""
        return 0 if self.stable else max(0, self.required - self.streak)

    def __str__(self) -> str:
        if self.regime is None:
            return "regime unknown (cooldown)"
        state = "stable" if self.stable else f"cooldown, {self.remaining} more bar(s) needed"
        changed = f", was {self.previous}" if self.previous else ""
        fam = f" [{self.family}]" if self.family != self.regime else ""
        return f"{self.regime}{fam} x{self.streak}/{self.required} ({state}{changed})"


class RegimeCooldown:
    """
    min_periods: consecutive bars the latest regime (family) must have held
        before new capital may be allocated under it.  1 disables the cooldown
        (every regime is immediately stable).
    families: ``{family_name: [label, ...]}`` -- labels that share a streak.
        Defaults to DEFAULT_REGIME_FAMILIES (TREND + VOL_BREAKOUT).  Pass ``{}``
        to make every label its own family (strict label-level cooldown).
    """

    def __init__(self, min_periods: int = DEFAULT_MIN_STABLE_PERIODS,
                 families: Optional[Mapping[str, Iterable[str]]] = None):
        min_periods = int(min_periods)
        if min_periods < 1:
            raise ValueError("min_periods must be >= 1")
        self.min_periods = min_periods

        source = DEFAULT_REGIME_FAMILIES if families is None else families
        self._family_of: Dict[str, str] = {}
        self.families: Dict[str, tuple] = {}
        for name, labels in source.items():
            fam = normalise_regime(name)
            if fam is None:
                raise ValueError(f"Invalid family name {name!r}")
            members = []
            for raw in labels:
                label = normalise_regime(raw)
                if label is None:
                    raise ValueError(f"Invalid label {raw!r} in family {name!r}")
                other = self._family_of.get(label)
                if other is not None and other != fam:
                    raise ValueError(f"Label {label!r} is in two families: {other!r} and {fam!r}")
                self._family_of[label] = fam
                members.append(label)
            self.families[fam] = tuple(members)

    def family_of(self, label) -> Optional[str]:
        """Family a label belongs to; ungrouped labels are their own family."""
        key = normalise_regime(label)
        if key is None:
            return None
        return self._family_of.get(key, key)

    def same_family(self, a, b) -> bool:
        fa, fb = self.family_of(a), self.family_of(b)
        return fa is not None and fa == fb

    def assess(self, history: Optional[Iterable] | pd.Series) -> RegimeStability:
        """
        `history`: regime labels in chronological order (the detector's
        per-bar series or any sequence).  Only the tail matters; the streak is
        capped at `min_periods` since longer runs change nothing.  Consecutive
        bars are counted while they stay inside the latest label's *family*.
        """
        labels = [normalise_regime(x) for x in (list(history) if history is not None else [])]
        if not labels:
            return RegimeStability(None, 0, self.min_periods)

        latest = labels[-1]
        if latest is None:
            return RegimeStability(None, 0, self.min_periods)
        family = self.family_of(latest)

        streak, previous = 0, None
        for label in reversed(labels):
            if self.family_of(label) == family and streak < self.min_periods:
                streak += 1
                continue
            if self.family_of(label) != family:
                previous = label
            break
        return RegimeStability(latest, streak, self.min_periods, previous, family)

    def __repr__(self) -> str:
        return f"RegimeCooldown(min_periods={self.min_periods}, families={self.families})"


__all__ = ["RegimeCooldown", "RegimeStability", "DEFAULT_MIN_STABLE_PERIODS", "DEFAULT_REGIME_FAMILIES"]
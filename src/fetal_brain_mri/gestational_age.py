"""Gestational age parsing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_GA_WEEKS = 18.0
MAX_GA_WEEKS = 40.0

_WEEKS_DAYS_PATTERNS = (
    re.compile(r"^\s*(?P<weeks>\d{1,2})\s*\+\s*(?P<days>\d(?:\.\d+)?)\s*$"),
    re.compile(
        r"^\s*(?P<weeks>\d{1,2})\s*w(?:eeks?)?\s*(?P<days>\d(?:\.\d+)?)\s*d(?:ays?)?\s*$",
        re.IGNORECASE,
    ),
)
_DECIMAL_WEEKS_PATTERN = re.compile(
    r"^\s*(?P<weeks>\d{1,2}(?:\.\d+)?)\s*w?(?:eeks?)?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GestationalAge:
    """Gestational age in completed weeks plus days."""

    weeks: int
    days: float = 0.0

    def __post_init__(self) -> None:
        if self.weeks < 0:
            raise ValueError("Gestational age weeks must be non-negative.")
        if self.days < 0 or self.days >= 7:
            raise ValueError("Gestational age days must be at least 0 and less than 7.")
        if self.decimal_weeks < MIN_GA_WEEKS or self.decimal_weeks > MAX_GA_WEEKS:
            raise ValueError("Gestational age must be between 18+0 and 40+0 weeks.")

    @property
    def decimal_weeks(self) -> float:
        return self.weeks + self.days / 7

    def __str__(self) -> str:
        days = int(self.days) if self.days.is_integer() else self.days
        return f"{self.weeks}+{days}"


def parse_gestational_age(raw: str) -> GestationalAge:
    """Parse SPEC-supported gestational age input formats."""

    if not raw.strip():
        raise ValueError("Gestational age is required.")

    for pattern in _WEEKS_DAYS_PATTERNS:
        match = pattern.match(raw)
        if match:
            return GestationalAge(
                weeks=int(match.group("weeks")),
                days=float(match.group("days")),
            )

    match = _DECIMAL_WEEKS_PATTERN.match(raw)
    if not match:
        raise ValueError(f"Unsupported gestational age format: {raw!r}.")

    decimal_weeks = float(match.group("weeks"))
    weeks = int(decimal_weeks)
    days = (decimal_weeks - weeks) * 7
    return GestationalAge(weeks=weeks, days=days)

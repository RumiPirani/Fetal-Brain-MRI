"""Band classification helpers."""

from __future__ import annotations

from typing import Literal

StandardBand = Literal["<5th", "normal", ">95th"]

FIFTH_PERCENTILE_Z = -1.645
NINETY_FIFTH_PERCENTILE_Z = 1.645


def classify_standard_band(z_score: float) -> StandardBand:
    """Classify a z-score into the TEST 1.3 standard parameter bands."""

    if z_score <= FIFTH_PERCENTILE_Z:
        return "<5th"
    if z_score >= NINETY_FIFTH_PERCENTILE_Z:
        return ">95th"
    return "normal"


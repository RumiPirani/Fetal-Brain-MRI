"""Band classification helpers."""

from __future__ import annotations

from typing import Literal

StandardBand = Literal["<3rd", "<5th", "normal", ">95th", ">97th", "absent"]

# z-score thresholds
THIRD_PERCENTILE_Z = -1.881
FIFTH_PERCENTILE_Z = -1.645
NINETY_FIFTH_PERCENTILE_Z = 1.645
NINETY_SEVENTH_PERCENTILE_Z = 1.881


def classify_standard_band(z_score: float) -> StandardBand:
    """Classify a z-score into standard parameter bands."""

    if z_score <= THIRD_PERCENTILE_Z:
        return "<3rd"
    if z_score <= FIFTH_PERCENTILE_Z:
        return "<5th"
    if z_score >= NINETY_SEVENTH_PERCENTILE_Z:
        return ">97th"
    if z_score >= NINETY_FIFTH_PERCENTILE_Z:
        return ">95th"
    return "normal"

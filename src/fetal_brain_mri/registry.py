"""Normative source registry from SPEC 4.2.2."""

from __future__ import annotations

from fetal_brain_mri.consensus import Source
from fetal_brain_mri.normative_models import PerPercentileLinear, QuadraticMeanLinearSd

_REGISTRY: dict[str, tuple[Source, ...]] = {
    "tcd": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=0.0051,
                mean_b=1.5165,
                mean_c=-14.584,
                sd_a=0.0343,
                sd_b=0.415,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
        Source(
            label="Dovjak 2021",
            model=PerPercentileLinear(
                p5_k=1.52,
                p5_d=-12.48,
                p95_k=1.85,
                p95_d=-15.23,
            ),
            min_ga_weeks=14.0,
            max_ga_weeks=40.0,
        ),
    ),
}


def get_sources(parameter_id: str) -> list[Source]:
    """Return registered sources for a parameter."""

    try:
        return list(_REGISTRY[parameter_id])
    except KeyError as exc:
        raise KeyError(f"No normative sources registered for parameter {parameter_id!r}.") from exc


"""Calculator-facing evaluation API."""

from __future__ import annotations

from dataclasses import dataclass

from fetal_brain_mri.bands import StandardBand, classify_standard_band
from fetal_brain_mri.consensus import ConsensusResult, evaluate_measurement
from fetal_brain_mri.registry import get_sources


@dataclass(frozen=True)
class ParameterResult:
    """Interpreted result for one entered measurement."""

    parameter_id: str
    measurement: float
    consensus: ConsensusResult
    band: StandardBand


def evaluate_parameter(
    *,
    parameter_id: str,
    ga_weeks: float,
    measurement: float,
) -> ParameterResult:
    """Evaluate one measurement through registry, consensus, and band layers."""

    consensus = evaluate_measurement(
        ga_weeks=ga_weeks,
        measurement=measurement,
        sources=get_sources(parameter_id),
    )
    return ParameterResult(
        parameter_id=parameter_id,
        measurement=measurement,
        consensus=consensus,
        band=classify_standard_band(consensus.consensus_z),
    )

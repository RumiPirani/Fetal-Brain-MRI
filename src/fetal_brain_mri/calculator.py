"""Calculator-facing evaluation API."""

from __future__ import annotations

from dataclasses import dataclass

from fetal_brain_mri.bands import StandardBand, classify_standard_band
from fetal_brain_mri.consensus import ConsensusResult, evaluate_measurement
from fetal_brain_mri.ddx import DdxCard, evaluate_ddx
from fetal_brain_mri.inputs import MeasurementInput
from fetal_brain_mri.registry import get_sources


@dataclass(frozen=True)
class ParameterResult:
    """Interpreted result for one entered measurement."""

    parameter_id: str
    measurement: float
    consensus: ConsensusResult
    band: StandardBand


@dataclass(frozen=True)
class CaseResult:
    """Full result for one calculator session."""

    ga_weeks: float
    parameters: dict[str, ParameterResult]
    ddx_cards: tuple[DdxCard, ...]
    qualitative_absent: tuple[str, ...]  # parameter IDs reported as absent


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


def evaluate_case(inputs: MeasurementInput) -> CaseResult:
    """Evaluate a full measurement set and return parameters, DDx, and report data."""

    parameters: dict[str, ParameterResult] = {}

    for param_id, measurement in inputs.numeric_parameters().items():
        # Skip CSP z-score if reported absent
        if param_id == "csp" and inputs.csp_absent:
            continue
        # Skip CC z-score if reported absent
        if param_id == "cc_length" and inputs.cc_absent:
            continue
        parameters[param_id] = evaluate_parameter(
            parameter_id=param_id,
            ga_weeks=inputs.ga_weeks,
            measurement=measurement,
        )

    absent: list[str] = []
    if inputs.csp_absent:
        absent.append("csp")
    if inputs.cc_absent:
        absent.append("cc_length")

    ddx_cards = evaluate_ddx(parameters, inputs)

    return CaseResult(
        ga_weeks=inputs.ga_weeks,
        parameters=parameters,
        ddx_cards=tuple(ddx_cards),
        qualitative_absent=tuple(absent),
    )

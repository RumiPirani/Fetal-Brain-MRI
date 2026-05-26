"""Multi-source consensus reconciliation from SPEC 4.2.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fetal_brain_mri.normative_models import NormativeModel, ZScoreMixin

AgreementState = Literal["single", "agree", "disagree"]


@dataclass(frozen=True)
class Source:
    """A normative source entry for one parameter."""

    label: str
    model: NormativeModel
    min_ga_weeks: float
    max_ga_weeks: float

    def is_in_range(self, ga_weeks: float) -> bool:
        return self.min_ga_weeks <= ga_weeks <= self.max_ga_weeks


@dataclass(frozen=True)
class SourceResult:
    """Per-source evaluation result surfaced to UI and report layers."""

    label: str
    mean: float
    sigma: float
    z_score: float
    percentile: float
    in_range: bool


@dataclass(frozen=True)
class ConsensusResult:
    """Consensus value and source agreement metadata."""

    consensus_z: float
    percentile: float
    agreement: AgreementState
    disagreement_width: float
    extrapolated: bool
    sources: tuple[SourceResult, ...]


def evaluate_measurement(
    *,
    ga_weeks: float,
    measurement: float,
    sources: list[Source],
) -> ConsensusResult:
    if not sources:
        raise ValueError("At least one normative source is required.")

    source_results = tuple(_evaluate_source(ga_weeks, measurement, source) for source in sources)
    in_range_results = tuple(result for result in source_results if result.in_range)
    consensus_inputs = in_range_results or source_results
    extrapolated = not in_range_results

    consensus_z = sum(result.z_score for result in consensus_inputs) / len(consensus_inputs)
    disagreement_width = _disagreement_width(consensus_inputs)
    agreement = _agreement_state(source_results, consensus_inputs, disagreement_width)

    return ConsensusResult(
        consensus_z=consensus_z,
        percentile=ZScoreMixin.percentile_from_z(consensus_z),
        agreement=agreement,
        disagreement_width=disagreement_width,
        extrapolated=extrapolated,
        sources=source_results,
    )


def _evaluate_source(ga_weeks: float, measurement: float, source: Source) -> SourceResult:
    mean = source.model.mean(ga_weeks)
    sigma = source.model.sigma(ga_weeks)
    if sigma <= 0:
        raise ValueError(f"Normative source {source.label!r} produced non-positive sigma.")
    z_score = (measurement - mean) / sigma
    return SourceResult(
        label=source.label,
        mean=mean,
        sigma=sigma,
        z_score=z_score,
        percentile=ZScoreMixin.percentile_from_z(z_score),
        in_range=source.is_in_range(ga_weeks),
    )


def _disagreement_width(results: tuple[SourceResult, ...]) -> float:
    if len(results) < 2:
        return 0.0
    z_scores = [result.z_score for result in results]
    return max(z_scores) - min(z_scores)


def _agreement_state(
    all_results: tuple[SourceResult, ...],
    consensus_inputs: tuple[SourceResult, ...],
    disagreement_width: float,
) -> AgreementState:
    if len(all_results) == 1 or len(consensus_inputs) == 1:
        return "single"
    if disagreement_width < 1.0:
        return "agree"
    return "disagree"


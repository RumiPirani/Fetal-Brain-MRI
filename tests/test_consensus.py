import math

from fetal_brain_mri.consensus import Source, evaluate_measurement
from fetal_brain_mri.normative_models import LinearMeanConstantSd


def source(
    label: str,
    mean: float,
    sd: float = 1.0,
    min_ga: float = 20.0,
    max_ga: float = 40.0,
) -> Source:
    return Source(
        label=label,
        model=LinearMeanConstantSd(mean_m=0.0, mean_b=mean, sd=sd),
        min_ga_weeks=min_ga,
        max_ga_weeks=max_ga,
    )


def test_single_source_passes_through_with_source_detail() -> None:
    result = evaluate_measurement(ga_weeks=28.0, measurement=11.0, sources=[source("Only", 10.0)])

    assert result.agreement == "single"
    assert result.disagreement_width == 0.0
    assert math.isclose(result.consensus_z, 1.0)
    assert not result.extrapolated
    assert result.sources[0].label == "Only"
    assert result.sources[0].in_range


def test_two_sources_agree_when_z_scores_differ_by_less_than_one_sd() -> None:
    result = evaluate_measurement(
        ga_weeks=28.0,
        measurement=10.0,
        sources=[source("A", 10.0), source("B", 10.5)],
    )

    assert result.agreement == "agree"
    assert math.isclose(result.disagreement_width, 0.5)
    assert math.isclose(result.consensus_z, -0.25)


def test_two_sources_disagree_when_z_scores_differ_by_at_least_one_sd() -> None:
    result = evaluate_measurement(
        ga_weeks=28.0,
        measurement=10.0,
        sources=[source("A", 10.0), source("B", 11.0)],
    )

    assert result.agreement == "disagree"
    assert math.isclose(result.disagreement_width, 1.0)
    assert math.isclose(result.consensus_z, -0.5)


def test_extrapolated_sources_remain_visible_but_do_not_drive_in_range_consensus() -> None:
    result = evaluate_measurement(
        ga_weeks=28.0,
        measurement=10.0,
        sources=[
            source("In range", 10.0, min_ga=20.0, max_ga=40.0),
            source("Extrapolated", 20.0, min_ga=30.0, max_ga=40.0),
        ],
    )

    assert result.agreement == "single"
    assert math.isclose(result.consensus_z, 0.0)
    assert not result.extrapolated
    assert [source_result.in_range for source_result in result.sources] == [True, False]


def test_consensus_falls_back_to_extrapolated_sources_when_none_are_in_range() -> None:
    result = evaluate_measurement(
        ga_weeks=18.0,
        measurement=10.0,
        sources=[
            source("A", 9.0, min_ga=20.0, max_ga=40.0),
            source("B", 11.0, min_ga=20.0, max_ga=40.0),
        ],
    )

    assert result.agreement == "disagree"
    assert result.extrapolated
    assert math.isclose(result.disagreement_width, 2.0)
    assert math.isclose(result.consensus_z, 0.0)

import math

from fetal_brain_mri.consensus import evaluate_measurement
from fetal_brain_mri.registry import get_sources


def test_tcd_registry_contains_luis_and_dovjak_sources() -> None:
    sources = get_sources("tcd")

    assert [source.label for source in sources] == ["Luis 2025", "Dovjak 2021"]
    assert [(source.min_ga_weeks, source.max_ga_weeks) for source in sources] == [
        (20.0, 40.0),
        (14.0, 40.0),
    ]


def test_tcd_registry_evaluates_spec_worked_example_coefficients() -> None:
    result = evaluate_measurement(ga_weeks=28.0, measurement=33.0, sources=get_sources("tcd"))

    luis, dovjak = result.sources

    assert math.isclose(luis.mean, 31.8764)
    assert math.isclose(luis.sigma, 1.3754)
    assert math.isclose(luis.z_score, (33.0 - 31.8764) / 1.3754)
    assert math.isclose(dovjak.mean, 33.325)
    assert math.isclose(dovjak.sigma, (36.57 - 30.08) / (2 * 1.6449))
    assert math.isclose(dovjak.z_score, (33.0 - 33.325) / dovjak.sigma)
    assert result.agreement == "agree"
    assert math.isclose(result.consensus_z, (luis.z_score + dovjak.z_score) / 2)

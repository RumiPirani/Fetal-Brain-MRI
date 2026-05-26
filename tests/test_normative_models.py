import math

import pytest

from fetal_brain_mri.normative_models import (
    LinearMeanConstantSd,
    PerPercentileLinear,
    QuadraticMeanLinearSd,
)


def test_quadratic_mean_linear_sd_model_matches_spec_formula() -> None:
    model = QuadraticMeanLinearSd(
        mean_a=0.0051,
        mean_b=1.5165,
        mean_c=-14.584,
        sd_a=0.0343,
        sd_b=0.415,
    )

    assert math.isclose(model.mean(28.0), 31.8764)
    assert math.isclose(model.sigma(28.0), 1.3754)
    assert math.isclose(model.z_score(28.0, 33.0), (33.0 - 31.8764) / 1.3754)


def test_per_percentile_linear_model_recovers_mean_and_sd() -> None:
    model = PerPercentileLinear(p5_k=1.52, p5_d=-12.48, p95_k=1.85, p95_d=-15.23)

    assert math.isclose(model.mean(28.0), (30.08 + 36.57) / 2)
    assert math.isclose(model.sigma(28.0), (36.57 - 30.08) / (2 * 1.6449))
    assert math.isclose(model.z_score(28.0, 33.0), (33.0 - model.mean(28.0)) / model.sigma(28.0))


def test_linear_mean_constant_sd_model_matches_spec_formula() -> None:
    model = LinearMeanConstantSd(mean_m=0.02, mean_b=1.2, sd=0.6)

    assert math.isclose(model.mean(30.0), 1.8)
    assert math.isclose(model.sigma(30.0), 0.6)
    assert math.isclose(model.z_score(30.0, 3.0), 2.0)


@pytest.mark.parametrize(
    ("z_score", "percentile"),
    [(0.0, 50.0), (1.0, 84.1344746), (-1.0, 15.8655254)],
)
def test_percentile_uses_standard_normal_cdf(z_score: float, percentile: float) -> None:
    model = LinearMeanConstantSd(mean_m=0.0, mean_b=0.0, sd=1.0)

    assert math.isclose(model.percentile_from_z(z_score), percentile, rel_tol=1e-7)


def test_models_reject_non_positive_sigma() -> None:
    model = LinearMeanConstantSd(mean_m=0.0, mean_b=0.0, sd=0.0)

    with pytest.raises(ValueError):
        model.z_score(28.0, 1.0)

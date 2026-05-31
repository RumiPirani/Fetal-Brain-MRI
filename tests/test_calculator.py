import math

import pytest

from fetal_brain_mri.calculator import evaluate_parameter


def test_evaluate_parameter_combines_registry_consensus_and_band() -> None:
    result = evaluate_parameter(parameter_id="tcd", ga_weeks=28.0, measurement=33.0)

    assert result.parameter_id == "tcd"
    assert result.measurement == 33.0
    assert result.band == "normal"
    assert result.consensus.agreement == "agree"
    assert math.isclose(result.consensus.consensus_z, 0.326091, rel_tol=1e-6)


def test_evaluate_parameter_surfaces_high_band() -> None:
    result = evaluate_parameter(parameter_id="tcd", ga_weeks=28.0, measurement=38.0)

    assert result.band in (">95th", ">97th")


def test_evaluate_parameter_rejects_unknown_parameter() -> None:
    with pytest.raises(KeyError):
        evaluate_parameter(parameter_id="not_registered", ga_weeks=28.0, measurement=1.0)

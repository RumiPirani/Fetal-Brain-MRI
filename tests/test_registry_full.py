"""Tests for the full 15-parameter registry."""

import math

import pytest

from fetal_brain_mri.registry import get_sources, list_parameter_ids, check_source_admission


ALL_PARAMS = [
    "skull_bpd", "skull_ofd", "brain_bpd", "brain_ofd_l", "brain_ofd_r",
    "atrium_l", "atrium_r", "csp", "tcd", "vermis_cc", "vermis_ap",
    "pons_ap", "cc_length", "third_ventricle", "tdpf", "csa",
]


def test_all_parameters_are_registered() -> None:
    registered = list_parameter_ids()
    for param in ALL_PARAMS:
        assert param in registered, f"{param} not registered"


def test_single_source_parameters_have_one_source() -> None:
    single_source_params = [
        "skull_bpd", "skull_ofd", "brain_bpd", "brain_ofd_l", "brain_ofd_r",
        "atrium_l", "atrium_r", "csp", "cc_length", "third_ventricle", "tdpf", "csa",
    ]
    for param in single_source_params:
        assert len(get_sources(param)) == 1, f"{param} should have 1 source"


def test_dual_source_parameters_have_two_sources() -> None:
    dual_source_params = ["tcd", "vermis_cc", "vermis_ap", "pons_ap"]
    for param in dual_source_params:
        sources = get_sources(param)
        assert len(sources) == 2, f"{param} should have 2 sources"
        labels = [s.label for s in sources]
        assert "Luis 2025" in labels
        assert "Dovjak 2021" in labels


@pytest.mark.parametrize("param_id", ALL_PARAMS)
def test_every_source_produces_positive_sigma_at_28_weeks(param_id: str) -> None:
    for source in get_sources(param_id):
        ga = max(28.0, source.min_ga_weeks)
        sigma = source.model.sigma(ga)
        assert sigma > 0, f"{param_id}/{source.label} sigma non-positive at GA {ga}"


@pytest.mark.parametrize("param_id", ALL_PARAMS)
def test_every_source_mean_is_plausible_at_28_weeks(param_id: str) -> None:
    for source in get_sources(param_id):
        ga = max(28.0, source.min_ga_weeks)
        mean = source.model.mean(ga)
        assert mean > 0, f"{param_id}/{source.label} mean non-positive at GA {ga}"


def test_tcd_luis_2025_matches_spec_worked_example() -> None:
    sources = get_sources("tcd")
    luis = next(s for s in sources if s.label == "Luis 2025")
    assert math.isclose(luis.model.mean(28.0), 31.8764, rel_tol=1e-4)
    assert math.isclose(luis.model.sigma(28.0), 1.3754, rel_tol=1e-4)


def test_skull_bpd_luis_2025_coefficients() -> None:
    source = get_sources("skull_bpd")[0]
    mean_28 = source.model.mean(28.0)
    assert 70 < mean_28 < 85, f"skull_bpd mean at 28w should be ~75mm, got {mean_28:.1f}"


def test_atrium_mean_is_biologically_flat() -> None:
    source = get_sources("atrium_r")[0]
    mean_20 = source.model.mean(20.0)
    mean_38 = source.model.mean(38.0)
    assert abs(mean_38 - mean_20) < 3.0, "atrial mean should be nearly flat across GA"


def test_source_admission_check_passes_for_close_source() -> None:
    from fetal_brain_mri.consensus import Source
    from fetal_brain_mri.normative_models import QuadraticMeanLinearSd

    # skull_bpd has one source; a nearly identical candidate should pass
    candidate = Source(
        label="Candidate",
        model=QuadraticMeanLinearSd(
            mean_a=-0.0527, mean_b=5.7605, mean_c=-46.436,
            sd_a=0.0895, sd_b=0.1414,
        ),
        min_ga_weeks=20.0,
        max_ga_weeks=40.0,
    )
    passes, delta = check_source_admission("skull_bpd", candidate)
    assert passes
    assert delta < 0.01  # identical source should have near-zero divergence


def test_source_admission_check_fails_for_divergent_source() -> None:
    from fetal_brain_mri.consensus import Source
    from fetal_brain_mri.normative_models import QuadraticMeanLinearSd

    # A source with a mean offset of ~10mm — should fail
    candidate = Source(
        label="Divergent",
        model=QuadraticMeanLinearSd(
            mean_a=0.0051, mean_b=1.5165, mean_c=-4.584,  # +10mm offset
            sd_a=0.0343, sd_b=0.415,
        ),
        min_ga_weeks=20.0,
        max_ga_weeks=40.0,
    )
    passes, delta = check_source_admission("tcd", candidate)
    assert not passes
    assert delta > 0.5

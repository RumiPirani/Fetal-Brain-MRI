"""Tests for the differential diagnosis engine."""

import pytest

from fetal_brain_mri.calculator import evaluate_case
from fetal_brain_mri.inputs import MeasurementInput


def _base(ga: float = 28.0, **kwargs) -> MeasurementInput:
    """Construct a MeasurementInput with values exactly at each source's mean for the given GA."""
    from fetal_brain_mri.registry import get_sources

    _PARAMS = [
        "skull_bpd", "skull_ofd", "brain_bpd", "brain_ofd_l", "brain_ofd_r",
        "atrium_r", "atrium_l", "csp", "cc_length",
        "tcd", "vermis_cc", "vermis_ap", "pons_ap", "third_ventricle",
    ]
    defaults: dict = {}
    for param_id in _PARAMS:
        ga_use = max(ga, get_sources(param_id)[0].min_ga_weeks)
        defaults[param_id] = get_sources(param_id)[0].model.mean(ga_use)
    defaults.update(kwargs)
    return MeasurementInput(ga_weeks=ga, **defaults)


def _card_ids(case) -> set[str]:
    return {c.card_id for c in case.ddx_cards}


# ---------------------------------------------------------------------------
# Normal control — no DDx cards should fire
# ---------------------------------------------------------------------------

def test_normal_control_fires_no_ddx_cards() -> None:
    case = evaluate_case(_base())
    assert case.ddx_cards == ()


# ---------------------------------------------------------------------------
# Ventriculomegaly triggers
# ---------------------------------------------------------------------------

def test_mild_vm_fires_when_atrium_between_10_and_15() -> None:
    case = evaluate_case(_base(atrium_r=11.0))
    assert "mild_ventriculomegaly" in _card_ids(case)
    assert "severe_ventriculomegaly" not in _card_ids(case)


def test_severe_vm_fires_when_atrium_ge_15() -> None:
    case = evaluate_case(_base(atrium_r=15.5))
    assert "severe_ventriculomegaly" in _card_ids(case)
    assert "mild_ventriculomegaly" not in _card_ids(case)


def test_mild_vm_boundary_at_exactly_10mm() -> None:
    case = evaluate_case(_base(atrium_l=10.0))
    assert "mild_ventriculomegaly" in _card_ids(case)


def test_mild_vm_does_not_fire_below_10mm() -> None:
    case = evaluate_case(_base(atrium_r=9.9))
    assert "mild_ventriculomegaly" not in _card_ids(case)


def test_asymmetric_ventricles_fires_when_delta_gt_2mm() -> None:
    case = evaluate_case(_base(atrium_r=9.5, atrium_l=6.5))
    assert "asymmetric_ventricles" in _card_ids(case)


def test_asymmetric_ventricles_does_not_fire_below_2mm_delta() -> None:
    case = evaluate_case(_base(atrium_r=8.0, atrium_l=6.5))  # delta=1.5
    assert "asymmetric_ventricles" not in _card_ids(case)


# ---------------------------------------------------------------------------
# TCD triggers
# ---------------------------------------------------------------------------

def test_small_tcd_fires_when_z_below_minus_1_645() -> None:
    case = evaluate_case(_base(tcd=26.0))  # well below normal
    assert "small_tcd" in _card_ids(case)


def test_large_tcd_fires_when_z_above_1_645() -> None:
    case = evaluate_case(_base(tcd=42.0))
    assert "large_tcd" in _card_ids(case)


# ---------------------------------------------------------------------------
# Vermis triggers
# ---------------------------------------------------------------------------

def test_vermian_hypoplasia_fires_for_small_vermis() -> None:
    case = evaluate_case(_base(vermis_cc=9.0))  # ~z=-4 at 28w
    assert "vermian_hypoplasia" in _card_ids(case)


# ---------------------------------------------------------------------------
# Pons trigger
# ---------------------------------------------------------------------------

def test_small_pons_fires_for_small_pons() -> None:
    case = evaluate_case(_base(pons_ap=5.0))
    assert "small_pons" in _card_ids(case)


# ---------------------------------------------------------------------------
# CSP triggers
# ---------------------------------------------------------------------------

def test_csp_absent_fires_when_csp_absent_flag_set() -> None:
    inp = _base(csp=None)
    inp = MeasurementInput(
        ga_weeks=28.0, skull_bpd=75.5, skull_ofd=102.6,
        brain_bpd=73.2, brain_ofd_l=97.1, brain_ofd_r=97.2,
        atrium_r=7.4, atrium_l=7.4,
        csp_absent=True,
        cc_length=32.5, tcd=34.5, vermis_cc=16.0, vermis_ap=7.3,
        pons_ap=9.5, third_ventricle=1.7,
    )
    case = evaluate_case(inp)
    assert "csp_absent" in _card_ids(case)


def test_csp_enlarged_fires_when_csp_gt_10mm() -> None:
    case = evaluate_case(_base(csp=11.0))
    assert "csp_enlarged" in _card_ids(case)


# ---------------------------------------------------------------------------
# CC triggers
# ---------------------------------------------------------------------------

def test_cc_absent_fires_when_cc_absent_flag_set() -> None:
    inp = MeasurementInput(
        ga_weeks=28.0, skull_bpd=75.5, skull_ofd=102.6,
        brain_bpd=73.2, brain_ofd_l=97.1, brain_ofd_r=97.2,
        atrium_r=7.4, atrium_l=7.4,
        csp=4.4, cc_absent=True,
        tcd=34.5, vermis_cc=16.0, vermis_ap=7.3,
        pons_ap=9.5, third_ventricle=1.7,
    )
    case = evaluate_case(inp)
    assert "cc_absent" in _card_ids(case)


def test_cc_short_fires_when_cc_small() -> None:
    case = evaluate_case(_base(cc_length=18.0))  # well below normal at 28w
    assert "cc_short" in _card_ids(case)


# ---------------------------------------------------------------------------
# Third ventricle trigger
# ---------------------------------------------------------------------------

def test_third_ventricle_wide_fires_above_3_5mm() -> None:
    case = evaluate_case(_base(third_ventricle=4.0))
    assert "third_ventricle_dilatation" in _card_ids(case)


def test_third_ventricle_does_not_fire_below_threshold() -> None:
    case = evaluate_case(_base(third_ventricle=3.4))
    assert "third_ventricle_dilatation" not in _card_ids(case)


# ---------------------------------------------------------------------------
# Skull size triggers
# ---------------------------------------------------------------------------

def test_microcephaly_fires_for_very_small_skull_bpd() -> None:
    case = evaluate_case(_base(skull_bpd=60.0))  # >2 SD below at 28w
    assert "microcephaly_pattern" in _card_ids(case)


def test_macrocephaly_fires_for_very_large_skull_bpd() -> None:
    case = evaluate_case(_base(skull_bpd=92.0))
    assert "macrocephaly_pattern" in _card_ids(case)


# ---------------------------------------------------------------------------
# Combined patterns
# ---------------------------------------------------------------------------

def test_acc_pattern_fires_when_both_cc_and_csp_absent() -> None:
    inp = MeasurementInput(
        ga_weeks=28.0, skull_bpd=75.5, skull_ofd=102.6,
        brain_bpd=73.2, brain_ofd_l=97.1, brain_ofd_r=97.2,
        atrium_r=7.4, atrium_l=7.4,
        csp_absent=True, cc_absent=True,
        tcd=34.5, vermis_cc=16.0, vermis_ap=7.3,
        pons_ap=9.5, third_ventricle=1.7,
    )
    case = evaluate_case(inp)
    assert "acc_pattern" in _card_ids(case)
    assert "csp_absent" in _card_ids(case)
    assert "cc_absent" in _card_ids(case)


def test_hydrocephalus_pattern_fires_with_severe_vm_and_wide_third_ventricle() -> None:
    case = evaluate_case(_base(atrium_r=16.0, atrium_l=16.0, third_ventricle=4.5))
    assert "hydrocephalus_pattern" in _card_ids(case)
    assert "severe_ventriculomegaly" in _card_ids(case)


def test_dwm_pattern_fires_with_small_vermis_and_high_tva() -> None:
    inp = MeasurementInput(
        ga_weeks=28.0, skull_bpd=75.5, skull_ofd=102.6,
        brain_bpd=73.2, brain_ofd_l=97.1, brain_ofd_r=97.2,
        atrium_r=7.4, atrium_l=7.4,
        csp=4.4, cc_length=32.5,
        tcd=34.5, vermis_cc=9.0, vermis_ap=4.0,
        tva_degrees=90.0,
        pons_ap=9.5, third_ventricle=1.7,
    )
    case = evaluate_case(inp)
    assert "dandy_walker_spectrum" in _card_ids(case)
    assert "vermian_hypoplasia" in _card_ids(case)


def test_pch_pattern_fires_with_small_pons_and_small_tcd() -> None:
    case = evaluate_case(_base(pons_ap=4.5, tcd=25.0))
    assert "pch_pattern" in _card_ids(case)
    assert "small_pons" in _card_ids(case)
    assert "small_tcd" in _card_ids(case)


# ---------------------------------------------------------------------------
# Chiari II discriminator
# ---------------------------------------------------------------------------

def test_chiari_ii_fires_when_tdpf_and_csa_both_severely_low() -> None:
    # At GA 28w: TDPF normal ~32mm, so 22mm gives z≈-4; CSA normal ~80°, so 55° gives z≈-4
    inp = MeasurementInput(
        ga_weeks=28.0, skull_bpd=75.5, skull_ofd=102.6,
        brain_bpd=73.2, brain_ofd_l=97.1, brain_ofd_r=97.2,
        atrium_r=7.4, atrium_l=7.4,
        csp=4.4, cc_length=32.5,
        tcd=34.5, vermis_cc=16.0, vermis_ap=7.3,
        pons_ap=9.5, third_ventricle=1.7,
        tdpf=22.0, csa=55.0,
    )
    case = evaluate_case(inp)
    assert "chiari_ii_open_ntd" in _card_ids(case)


def test_chiari_ii_does_not_fire_when_parameters_are_normal() -> None:
    inp = MeasurementInput(
        ga_weeks=28.0, skull_bpd=75.5, skull_ofd=102.6,
        brain_bpd=73.2, brain_ofd_l=97.1, brain_ofd_r=97.2,
        atrium_r=7.4, atrium_l=7.4,
        csp=4.4, cc_length=32.5,
        tcd=34.5, vermis_cc=16.0, vermis_ap=7.3,
        pons_ap=9.5, third_ventricle=1.7,
        tdpf=32.0, csa=80.0,  # normal values
    )
    case = evaluate_case(inp)
    assert "chiari_ii_open_ntd" not in _card_ids(case)

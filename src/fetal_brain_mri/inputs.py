"""Input model for a full calculator case."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeasurementInput:
    """All manual-entry inputs for one fetal brain MRI case."""

    ga_weeks: float

    # Skull
    skull_bpd: float | None = None
    skull_ofd: float | None = None

    # Brain parenchyma
    brain_bpd: float | None = None
    brain_ofd_l: float | None = None
    brain_ofd_r: float | None = None

    # Ventricles
    atrium_r: float | None = None
    atrium_l: float | None = None

    # Midline
    csp: float | None = None
    csp_absent: bool = False       # user-reported qualitative absence
    cc_length: float | None = None
    cc_absent: bool = False        # user-reported qualitative absence

    # Posterior fossa
    tcd: float | None = None
    vermis_cc: float | None = None
    vermis_ap: float | None = None
    tva_degrees: float | None = None   # tegmento-vermian angle, qualitative
    cisterna_magna_depth: float | None = None  # not yet z-scored

    # Brainstem
    pons_ap: float | None = None

    # Other
    third_ventricle: float | None = None

    # Chiari II parameters (SPEC §6.5)
    tdpf: float | None = None
    csa: float | None = None

    # Scan context (non-PHI metadata)
    field_strength_t: float | None = None  # e.g. 1.5, 3.0

    def numeric_parameters(self) -> dict[str, float]:
        """Return all non-None numeric parameter measurements keyed by parameter_id."""

        mapping: dict[str, str] = {
            "skull_bpd": "skull_bpd",
            "skull_ofd": "skull_ofd",
            "brain_bpd": "brain_bpd",
            "brain_ofd_l": "brain_ofd_l",
            "brain_ofd_r": "brain_ofd_r",
            "atrium_r": "atrium_r",
            "atrium_l": "atrium_l",
            "csp": "csp",
            "cc_length": "cc_length",
            "tcd": "tcd",
            "vermis_cc": "vermis_cc",
            "vermis_ap": "vermis_ap",
            "pons_ap": "pons_ap",
            "third_ventricle": "third_ventricle",
            "tdpf": "tdpf",
            "csa": "csa",
        }
        result: dict[str, float] = {}
        for param_id, attr in mapping.items():
            value = getattr(self, attr)
            if value is not None:
                result[param_id] = value
        return result

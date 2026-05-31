"""Normative source registry from SPEC 4.2.2 / 7.3."""

from __future__ import annotations

from fetal_brain_mri.consensus import Source
from fetal_brain_mri.normative_models import (
    LinearMeanConstantSd,
    PerPercentileLinear,
    QuadraticMeanLinearSd,
)

# ---------------------------------------------------------------------------
# Internal registry: parameter_id -> tuple of Source
# Coefficients are verbatim from SPEC §7.3 (byte-identical to Luis 2025
# auto-reporting-brain-biometry.py for Luis entries; from Dovjak 2021 Table 1
# for Dovjak entries; OLS-derived from Woitek 2014 Table 3 for TDPF/CSA).
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Source, ...]] = {
    # -- Skull BPD (§7.3.1) --------------------------------------------------
    "skull_bpd": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0527,
                mean_b=5.7605,
                mean_c=-46.436,
                sd_a=0.0895,
                sd_b=0.1414,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Skull OFD (§7.3.2) --------------------------------------------------
    "skull_ofd": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0984,
                mean_b=8.8526,
                mean_c=-81.605,
                sd_a=0.1511,
                sd_b=-1.3192,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Brain BPD (§7.3.3) --------------------------------------------------
    "brain_bpd": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=0.016,
                mean_b=1.763,
                mean_c=-0.9597,
                sd_a=0.1308,
                sd_b=-1.32,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Brain OFD left (§7.3.4) ---------------------------------------------
    "brain_ofd_l": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0781,
                mean_b=7.7234,
                mean_c=-75.3,
                sd_a=0.1277,
                sd_b=-0.9298,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Brain OFD right (§7.3.4 — same coefficients as left) ----------------
    "brain_ofd_r": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0781,
                mean_b=7.7234,
                mean_c=-75.3,
                sd_a=0.1277,
                sd_b=-0.9298,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Atrial diameter left (§7.3.5 — same coefficients for both sides) ----
    "atrium_l": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=0.0078,
                mean_b=-0.5216,
                mean_c=15.374,
                sd_a=0.0264,
                sd_b=0.5152,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Atrial diameter right (§7.3.5) --------------------------------------
    "atrium_r": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=0.0078,
                mean_b=-0.5216,
                mean_c=15.374,
                sd_a=0.0264,
                sd_b=0.5152,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- CSP width (§7.3.6) --------------------------------------------------
    "csp": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0156,
                mean_b=0.9472,
                mean_c=-6.6953,
                sd_a=0.053,
                sd_b=-0.4388,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- TCD (§7.3.7) — two sources ------------------------------------------
    "tcd": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=0.0051,
                mean_b=1.5165,
                mean_c=-14.584,
                sd_a=0.0343,
                sd_b=0.415,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
        Source(
            label="Dovjak 2021",
            model=PerPercentileLinear(
                p5_k=1.52,
                p5_d=-12.48,
                p95_k=1.85,
                p95_d=-15.23,
            ),
            min_ga_weeks=14.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Vermis cranio-caudal height (§7.3.8) — two sources ------------------
    "vermis_cc": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0138,
                mean_b=1.6136,
                mean_c=-20.065,
                sd_a=0.0354,
                sd_b=-0.1869,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
        Source(
            label="Dovjak 2021",
            model=PerPercentileLinear(
                p5_k=0.72,
                p5_d=-6.83,
                p95_k=0.95,
                p95_d=-8.93,
            ),
            min_ga_weeks=14.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Vermis antero-posterior diameter (§7.3.9) — two sources -------------
    "vermis_ap": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0089,
                mean_b=1.1119,
                mean_c=-14.637,
                sd_a=0.0447,
                sd_b=-0.5126,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
        Source(
            label="Dovjak 2021",
            model=PerPercentileLinear(
                p5_k=0.53,
                p5_d=-5.26,
                p95_k=0.70,
                p95_d=-6.99,
            ),
            min_ga_weeks=14.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Pons AP diameter (§7.3.10) — two sources ----------------------------
    "pons_ap": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=0.002,
                mean_b=0.3144,
                mean_c=-1.2147,
                sd_a=0.0124,
                sd_b=0.261,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
        Source(
            label="Dovjak 2021",
            model=PerPercentileLinear(
                p5_k=0.33,
                p5_d=-0.59,
                p95_k=0.44,
                p95_d=-0.78,
            ),
            min_ga_weeks=14.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Corpus callosum length (§7.3.11) ------------------------------------
    "cc_length": (
        Source(
            label="Luis 2025",
            model=QuadraticMeanLinearSd(
                mean_a=-0.0687,
                mean_b=5.1529,
                mean_c=-57.904,
                sd_a=0.0274,
                sd_b=0.4763,
            ),
            min_ga_weeks=20.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Third ventricle width (§7.3.12) — approximation, cross-modality ----
    # SPEC flags this as an approximation; z-score is ordinal until a
    # verified transcription of Hertzberg 1997 Table 1 replaces this.
    "third_ventricle": (
        Source(
            label="Birnbaum 2018 (approx, cross-modality)",
            model=LinearMeanConstantSd(mean_m=0.02, mean_b=1.2, sd=0.6),
            min_ga_weeks=18.0,
            max_ga_weeks=40.0,
        ),
    ),
    # -- Max transverse diameter of posterior fossa (§7.3.13) ----------------
    # Coefficients OLS-derived from Woitek 2014 Table 3 control cohort.
    "tdpf": (
        Source(
            label="Woitek 2014",
            model=QuadraticMeanLinearSd(
                mean_a=-0.01307,
                mean_b=2.55571,
                mean_c=-21.71,
                sd_a=0.06716,
                sd_b=0.547,
            ),
            min_ga_weeks=21.0,
            max_ga_weeks=37.0,
        ),
    ),
    # -- Clivus-supraocciput angle (§7.3.14) ---------------------------------
    # Coefficients OLS-derived from Woitek 2014 Table 3 control cohort.
    "csa": (
        Source(
            label="Woitek 2014",
            model=QuadraticMeanLinearSd(
                mean_a=-0.04767,
                mean_b=4.20404,
                mean_c=1.73,
                sd_a=0.01814,
                sd_b=5.821,
            ),
            min_ga_weeks=21.0,
            max_ga_weeks=37.0,
        ),
    ),
}


def get_sources(parameter_id: str) -> list[Source]:
    """Return registered sources for a parameter."""

    try:
        return list(_REGISTRY[parameter_id])
    except KeyError as exc:
        raise KeyError(
            f"No normative sources registered for parameter {parameter_id!r}."
        ) from exc


def list_parameter_ids() -> list[str]:
    """Return all registered parameter IDs."""

    return list(_REGISTRY.keys())


def check_source_admission(
    parameter_id: str,
    candidate: Source,
    ga_step: float = 0.5,
) -> tuple[bool, float]:
    """Check SPEC §4.10.1 admission criterion for a candidate source.

    Returns (passes, max_delta) where max_delta is the worst-case
    standardised divergence across the GA overlap with all existing sources.
    Candidate is admitted only if max_delta <= 0.5 for every existing source.
    """

    existing = _REGISTRY.get(parameter_id, ())
    if not existing:
        return True, 0.0

    overall_max = 0.0
    for existing_source in existing:
        overlap_min = max(candidate.min_ga_weeks, existing_source.min_ga_weeks)
        overlap_max = min(candidate.max_ga_weeks, existing_source.max_ga_weeks)
        if overlap_min >= overlap_max:
            continue

        ga = overlap_min
        while ga <= overlap_max:
            mu_new = candidate.model.mean(ga)
            mu_old = existing_source.model.mean(ga)
            sigma_new = candidate.model.sigma(ga)
            sigma_old = existing_source.model.sigma(ga)
            denom = max(sigma_new, sigma_old)
            if denom > 0:
                delta = abs(mu_new - mu_old) / denom
                if delta > overall_max:
                    overall_max = delta
            ga += ga_step

    return overall_max <= 0.5, overall_max

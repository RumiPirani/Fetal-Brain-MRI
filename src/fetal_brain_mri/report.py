"""Deterministic report text helpers."""

from __future__ import annotations

from fetal_brain_mri.calculator import ParameterResult

_PARAMETER_LABELS = {
    "tcd": "TCD",
}


def render_parameter_line(result: ParameterResult) -> str:
    """Render one measured parameter with consensus and source disclosure."""

    label = _PARAMETER_LABELS.get(result.parameter_id, result.parameter_id)
    consensus = result.consensus
    source_text = "; ".join(
        f"{source.label} z={source.z_score:+.2f}"
        f" ({'in-range' if source.in_range else 'extrapolated'})"
        for source in consensus.sources
    )
    percentile = round(consensus.percentile)
    return (
        f"{label}: {result.measurement:.1f} mm "
        f"(Z: {consensus.consensus_z:+.2f}, {percentile}rd percentile, "
        f"band: {result.band}, agreement: {consensus.agreement}; sources: {source_text})"
    )


def render_structured_report(results: list[ParameterResult]) -> str:
    """Render the deterministic report sections currently supported."""

    findings = "\n".join(render_parameter_line(result) for result in results)
    impression = _render_impression(results)
    sections = [
        "METHODOLOGY",
        (
            "Calculator operated in multi-source consensus mode; consensus z-scores are "
            "computed from in-range sources, and the source disagreement threshold is 1.0 SD."
        ),
        "",
        "FINDINGS",
        findings,
    ]
    notes = _render_source_agreement_notes(results)
    if notes:
        sections.extend(["", "SOURCE-AGREEMENT NOTES", notes])
    sections.extend(["", "IMPRESSION", impression])
    return "\n".join(sections)


def _render_impression(results: list[ParameterResult]) -> str:
    if all(result.band == "normal" for result in results):
        return "No abnormal biometric findings."
    abnormal = [
        f"{_PARAMETER_LABELS.get(result.parameter_id, result.parameter_id)} {result.band}"
        for result in results
        if result.band != "normal"
    ]
    return "Abnormal biometric findings: " + "; ".join(abnormal) + "."


def _render_source_agreement_notes(results: list[ParameterResult]) -> str:
    lines = []
    for result in results:
        consensus = result.consensus
        if consensus.agreement != "disagree":
            continue
        label = _PARAMETER_LABELS.get(result.parameter_id, result.parameter_id)
        sources = "; ".join(
            f"{source.label} z={source.z_score:+.2f}" for source in consensus.sources
        )
        lines.append(
            f"{label} disagreement width {consensus.disagreement_width:.2f}: {sources}."
        )
    return "\n".join(lines)

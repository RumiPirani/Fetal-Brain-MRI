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


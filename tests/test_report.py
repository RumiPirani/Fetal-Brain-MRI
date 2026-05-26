from fetal_brain_mri.calculator import evaluate_parameter
from fetal_brain_mri.report import render_parameter_line


def test_render_parameter_line_includes_consensus_and_sources() -> None:
    result = evaluate_parameter(parameter_id="tcd", ga_weeks=28.0, measurement=33.0)

    line = render_parameter_line(result)

    assert "TCD: 33.0 mm" in line
    assert "Z: +0.33" in line
    assert "63rd percentile" in line
    assert "band: normal" in line
    assert "agreement: agree" in line
    assert "Luis 2025 z=+0.82" in line
    assert "Dovjak 2021 z=-0.16" in line


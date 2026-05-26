from fetal_brain_mri.calculator import evaluate_parameter
from fetal_brain_mri.report import render_parameter_line, render_structured_report


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


def test_render_structured_report_includes_methodology_findings_and_impression() -> None:
    result = evaluate_parameter(parameter_id="tcd", ga_weeks=28.0, measurement=33.0)

    report = render_structured_report([result])

    assert "METHODOLOGY" in report
    assert "multi-source consensus mode" in report
    assert "disagreement threshold is 1.0 SD" in report
    assert "FINDINGS" in report
    assert "TCD: 33.0 mm" in report
    assert "IMPRESSION" in report
    assert "No abnormal biometric findings." in report
    assert "SOURCE-AGREEMENT NOTES" not in report


def test_render_structured_report_adds_source_agreement_notes_for_disagreement() -> None:
    result = evaluate_parameter(parameter_id="tcd", ga_weeks=28.0, measurement=34.5)

    report = render_structured_report([result])

    assert "SOURCE-AGREEMENT NOTES" in report
    assert "TCD disagreement width" in report
    assert "Luis 2025 z=+1.91" in report
    assert "Dovjak 2021 z=+0.60" in report

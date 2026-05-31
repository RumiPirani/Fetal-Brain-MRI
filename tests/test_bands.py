from fetal_brain_mri.bands import classify_standard_band


def test_standard_band_classifies_below_third_percentile() -> None:
    assert classify_standard_band(-1.881) == "<3rd"
    assert classify_standard_band(-2.5) == "<3rd"


def test_standard_band_classifies_between_third_and_fifth_percentiles() -> None:
    assert classify_standard_band(-1.880) == "<5th"
    assert classify_standard_band(-1.645) == "<5th"


def test_standard_band_classifies_normal() -> None:
    assert classify_standard_band(-1.644) == "normal"
    assert classify_standard_band(0.0) == "normal"
    assert classify_standard_band(1.644) == "normal"


def test_standard_band_classifies_between_ninety_fifth_and_ninety_seventh() -> None:
    assert classify_standard_band(1.645) == ">95th"
    assert classify_standard_band(1.880) == ">95th"


def test_standard_band_classifies_above_ninety_seventh_percentile() -> None:
    assert classify_standard_band(1.881) == ">97th"
    assert classify_standard_band(2.5) == ">97th"

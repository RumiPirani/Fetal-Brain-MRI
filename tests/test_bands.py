from fetal_brain_mri.bands import classify_standard_band


def test_standard_band_classifies_values_below_fifth_percentile() -> None:
    assert classify_standard_band(-1.645) == "<5th"
    assert classify_standard_band(-2.0) == "<5th"


def test_standard_band_classifies_values_between_fifth_and_ninety_fifth_percentiles() -> None:
    assert classify_standard_band(-1.644) == "normal"
    assert classify_standard_band(0.0) == "normal"
    assert classify_standard_band(1.644) == "normal"


def test_standard_band_classifies_values_above_ninety_fifth_percentile() -> None:
    assert classify_standard_band(1.645) == ">95th"
    assert classify_standard_band(2.0) == ">95th"


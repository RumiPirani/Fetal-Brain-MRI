import math

import pytest

from fetal_brain_mri.gestational_age import GestationalAge, parse_gestational_age


@pytest.mark.parametrize(
    ("raw", "weeks", "days", "decimal"),
    [
        ("24+3", 24, 3, 24 + 3 / 7),
        ("24 w 3 d", 24, 3, 24 + 3 / 7),
        ("28.0 w", 28, 0, 28.0),
        ("28.5", 28, 3.5, 28.5),
    ],
)
def test_parse_gestational_age_accepts_specified_formats(
    raw: str, weeks: int, days: float, decimal: float
) -> None:
    parsed = parse_gestational_age(raw)

    assert parsed.weeks == weeks
    assert parsed.days == days
    assert math.isclose(parsed.decimal_weeks, decimal)


@pytest.mark.parametrize("raw", ["", "24+7", "17+6", "40+1", "abc"])
def test_parse_gestational_age_rejects_invalid_or_unsupported_values(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_gestational_age(raw)


def test_gestational_age_can_be_constructed_from_components() -> None:
    ga = GestationalAge(22, 3)

    assert math.isclose(ga.decimal_weeks, 22 + 3 / 7)
    assert str(ga) == "22+3"


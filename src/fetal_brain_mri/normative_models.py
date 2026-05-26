"""Normative curve model families from SPEC 4.2.1."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Protocol

PERCENTILE_Z_5_TO_95 = 1.6449


class NormativeModel(Protocol):
    """Common interface for a normative source model."""

    def mean(self, ga_weeks: float) -> float:
        """Return expected mean at gestational age."""

    def sigma(self, ga_weeks: float) -> float:
        """Return standard deviation at gestational age."""


class ZScoreMixin:
    """Shared z-score and percentile helpers."""

    def mean(self, ga_weeks: float) -> float:
        raise NotImplementedError

    def sigma(self, ga_weeks: float) -> float:
        raise NotImplementedError

    def z_score(self, ga_weeks: float, measurement: float) -> float:
        sigma = self.sigma(ga_weeks)
        if sigma <= 0:
            raise ValueError("Normative model sigma must be positive.")
        return (measurement - self.mean(ga_weeks)) / sigma

    def percentile(self, ga_weeks: float, measurement: float) -> float:
        return self.percentile_from_z(self.z_score(ga_weeks, measurement))

    @staticmethod
    def percentile_from_z(z_score: float) -> float:
        return 0.5 * (1 + erf(z_score / sqrt(2))) * 100


@dataclass(frozen=True)
class QuadraticMeanLinearSd(ZScoreMixin):
    """Model: mu = a*GA^2 + b*GA + c; sigma = a5*GA + b5."""

    mean_a: float
    mean_b: float
    mean_c: float
    sd_a: float
    sd_b: float

    def mean(self, ga_weeks: float) -> float:
        return self.mean_a * ga_weeks**2 + self.mean_b * ga_weeks + self.mean_c

    def sigma(self, ga_weeks: float) -> float:
        return self.sd_a * ga_weeks + self.sd_b


@dataclass(frozen=True)
class PerPercentileLinear(ZScoreMixin):
    """Model: p5 and p95 are linear in GA; mu and sigma are recovered."""

    p5_k: float
    p5_d: float
    p95_k: float
    p95_d: float

    def mean(self, ga_weeks: float) -> float:
        return (self.p5(ga_weeks) + self.p95(ga_weeks)) / 2

    def sigma(self, ga_weeks: float) -> float:
        return (self.p95(ga_weeks) - self.p5(ga_weeks)) / (2 * PERCENTILE_Z_5_TO_95)

    def p5(self, ga_weeks: float) -> float:
        return self.p5_k * ga_weeks + self.p5_d

    def p95(self, ga_weeks: float) -> float:
        return self.p95_k * ga_weeks + self.p95_d


@dataclass(frozen=True)
class LinearMeanConstantSd(ZScoreMixin):
    """Model: mu = m*GA + b; sigma is constant."""

    mean_m: float
    mean_b: float
    sd: float

    def mean(self, ga_weeks: float) -> float:
        return self.mean_m * ga_weeks + self.mean_b

    def sigma(self, ga_weeks: float) -> float:
        return self.sd


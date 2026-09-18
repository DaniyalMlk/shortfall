"""Value at risk and expected shortfall from a distributional assumption.

**The sign convention, stated once.** Value at risk and expected shortfall are
reported here as *positive losses*. A value at risk of 0.023 means a loss of
2.3% of the portfolio. The underlying return quantile is also reported, signed,
as ``quantile`` — so a caller that wants the other convention has it without
having to guess which one this is. Adding one convention to the other gives a
number wrong by twice the risk, and nothing about the number says so.

A positive mean can make value at risk negative: at 95% confidence a portfolio
with a large enough drift has a *gain* at its fifth percentile, so the "loss" is
negative. That is reported as it comes out rather than clamped to zero. Clamping
would hide the one case where the confidence level chosen is too weak to say
anything about the portfolio at all.

**Which tail.** ``confidence`` is the confidence level, so 0.99 means the 1%
tail, and the tail probability is ``1 - confidence``. They are complements, and
an implementation taking one while documenting the other is wrong by an amount
that grows as the tail thins.

**Three distributional assumptions**, in increasing willingness to admit that
returns are not normal:

*Normal.* Exact, fast, and understates the tail of essentially every financial
return series. Worth having as the baseline everything else is compared to.

*Student-t.* Heavier tails, one extra parameter, and one trap: the raw ``t`` has
variance ``v / (v - 2)``, not 1, so feeding it a volatility without rescaling
produces a distribution whose variance is not the one asked for. The scaling is
applied here and the requirement ``v > 2`` follows from it — below three degrees
of freedom the variance does not exist and there is nothing to match.

*Cornish-Fisher.* Corrects the normal quantile for skewness and excess kurtosis
without committing to a distribution. Its own weakness is sharp rather than
gradual: outside a bounded region of the skewness-kurtosis plane the corrected
quantile stops being monotone in the probability, which means it is not a
quantile function. That is checked and refused rather than returned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .distributions import (
    normal_cdf,
    normal_pdf,
    normal_ppf,
    student_t_pdf,
    student_t_ppf,
)
from .linalg import Matrix, quadratic_form

#: Points at which the Cornish-Fisher mapping is checked for monotonicity, and
#: the range they span. Four standard deviations covers every confidence level
#: anybody quotes — 99.99% is 3.72 — and the expansion is not to be trusted
#: beyond there in any case.
_MONOTONE_LIMIT = 4.0
_MONOTONE_POINTS = 401


class Distribution(str, Enum):
    """Which assumption a risk number was computed under."""

    NORMAL = "normal"
    STUDENT_T = "student-t"
    CORNISH_FISHER = "cornish-fisher"
    #: Not a distributional assumption at all — the sample itself. Carried in
    #: the same enumeration so a historical and a parametric estimate can be put
    #: side by side, which is the comparison worth making: a large gap between
    #: them is the shape assumption being told it is wrong.
    HISTORICAL = "historical"


class NotAQuantileFunction(ValueError):
    """The Cornish-Fisher mapping is not monotone for these moments.

    Which means it does not define a quantile function: two different
    probabilities map to the same value, or a higher probability maps to a lower
    one. Any number read off it is not a quantile of anything.
    """


@dataclass(frozen=True)
class Risk:
    """A risk estimate, with everything needed to know what it means."""

    #: Positive loss. 0.023 is a 2.3% loss of the portfolio.
    value_at_risk: float
    #: Positive loss, and never less than :attr:`value_at_risk`.
    expected_shortfall: float
    #: The signed return quantile the loss was taken from, for callers using the
    #: other sign convention.
    quantile: float
    confidence: float
    distribution: Distribution
    #: Periods the estimate covers. One unless it was scaled to a horizon.
    horizon: float = 1.0
    mean: float = 0.0
    volatility: float = 0.0

    @property
    def tail_probability(self) -> float:
        return 1.0 - self.confidence

    def scaled_to(self, horizon: float) -> Risk:
        """Rescale to a longer horizon under the square-root-of-time rule.

        Which assumes returns are independent across periods and identically
        distributed. Neither holds exactly: volatility clusters, so a ten-day
        risk computed this way understates a crisis and overstates a calm
        stretch. The assumption is named in the result rather than folded in
        silently, and the mean scales linearly while the volatility scales as
        the square root — a distinction that is easy to lose and that flips the
        sign of value at risk for a long enough horizon on a positive drift.
        """
        if horizon <= 0.0:
            raise ValueError(f"a horizon is a positive number of periods, got {horizon!r}")
        return parametric_risk(
            mean=self.mean * horizon,
            volatility=self.volatility * math.sqrt(horizon),
            confidence=self.confidence,
            distribution=self.distribution,
            _horizon=horizon,
        )


def _check_inputs(volatility: float, confidence: float) -> float:
    if volatility < 0.0:
        raise ValueError(f"volatility is non-negative, got {volatility!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence is strictly inside (0, 1), got {confidence!r}. It is the "
            "confidence level, so 0.99 means the 1% tail; pass 0.99 rather than 0.01 "
            "and rather than 99."
        )
    return 1.0 - confidence


# -- normal ------------------------------------------------------------------


def normal_risk(*, mean: float, volatility: float, confidence: float) -> Risk:
    """Value at risk and expected shortfall under a normal assumption.

    Both are closed forms. The expected shortfall is
    ``-mean + volatility * phi(z) / alpha``, where ``z`` is the normal quantile
    at the tail probability ``alpha``: the mean of the tail is the density at its
    edge divided by its probability, which is the one line of this that is worth
    remembering.
    """
    alpha = _check_inputs(volatility, confidence)
    z = normal_ppf(alpha)
    quantile = mean + volatility * z
    shortfall_mean = mean - volatility * normal_pdf(z) / alpha
    return Risk(
        value_at_risk=-quantile,
        expected_shortfall=-shortfall_mean,
        quantile=quantile,
        confidence=confidence,
        distribution=Distribution.NORMAL,
        mean=mean,
        volatility=volatility,
    )


# -- Student-t ---------------------------------------------------------------


def student_t_risk(
    *, mean: float, volatility: float, confidence: float, degrees: float
) -> Risk:
    """Value at risk and expected shortfall under a scaled Student-t.

    The distribution is ``mean + volatility * sqrt((v - 2) / v) * T`` with
    ``T`` a standard Student-t. That scale factor is the whole point: a raw
    ``t`` with ``v`` degrees of freedom has variance ``v / (v - 2)``, so using
    it unscaled gives a distribution whose volatility is not the one supplied —
    by 22% at five degrees of freedom, all of it in the wrong direction.

    ``v > 2`` because the variance does not exist at or below two, and there is
    then nothing for a volatility to match.
    """
    alpha = _check_inputs(volatility, confidence)
    if degrees <= 2.0:
        raise ValueError(
            f"degrees of freedom must exceed 2, got {degrees!r}. At or below 2 the "
            "Student-t has no finite variance, so it cannot be scaled to match a "
            "volatility and its expected shortfall is infinite."
        )
    scale = volatility * math.sqrt((degrees - 2.0) / degrees)
    t = student_t_ppf(alpha, degrees)
    quantile = mean + scale * t
    # E[T | T <= t] = -(v + t^2) / (v - 1) * f(t) / alpha, the t analogue of the
    # normal's density-over-probability.
    tail_mean = -((degrees + t * t) / (degrees - 1.0)) * student_t_pdf(t, degrees) / alpha
    shortfall_mean = mean + scale * tail_mean
    return Risk(
        value_at_risk=-quantile,
        expected_shortfall=-shortfall_mean,
        quantile=quantile,
        confidence=confidence,
        distribution=Distribution.STUDENT_T,
        mean=mean,
        volatility=volatility,
    )


# -- Cornish-Fisher ----------------------------------------------------------


def cornish_fisher_coefficients(skewness: float, excess_kurtosis: float) -> tuple[float, ...]:
    """The corrected quantile as a cubic in the normal quantile.

    ``z + (z^2 - 1) S / 6 + (z^3 - 3z) K / 24 - (2z^3 - 5z) S^2 / 36`` collected
    into ``c0 + c1 z + c2 z^2 + c3 z^3``. Collecting it is not cosmetic: written
    this way the tail integral needed for the expected shortfall has a closed
    form, because the tail moments of the normal do.

    ``excess_kurtosis`` is excess — a normal has zero, not three. The parameter
    is named for it because the two conventions differ by exactly the value a
    normal takes, so a raw kurtosis passed in here would read as heavy tails on
    a distribution that has none.
    """
    s, k = skewness, excess_kurtosis
    return (
        -s / 6.0,
        1.0 - k / 8.0 + 5.0 * s * s / 36.0,
        s / 6.0,
        k / 24.0 - s * s / 18.0,
    )


def cornish_fisher_quantile(z: float, skewness: float, excess_kurtosis: float) -> float:
    """Evaluate the corrected quantile at a normal quantile ``z``."""
    c0, c1, c2, c3 = cornish_fisher_coefficients(skewness, excess_kurtosis)
    return c0 + z * (c1 + z * (c2 + z * c3))


def is_monotone(
    skewness: float,
    excess_kurtosis: float,
    *,
    lower: float = -_MONOTONE_LIMIT,
    upper: float = _MONOTONE_LIMIT,
) -> bool:
    """Whether the corrected mapping increases across ``[lower, upper]`` in ``z``.

    **The range is an argument because the answer depends on it, and the honest
    range is not symmetric.** Whenever the cubic coefficient is negative — which
    it is for any non-zero skewness with no excess kurtosis — the slope goes
    negative far enough out on *both* sides, so no set of moments is monotone
    everywhere and a check over all of ``(-inf, inf)`` would refuse every
    correction there is. What matters is whether the mapping is monotone over
    the quantiles actually read off it, and a left-tail risk number never reads
    the right tail at all. Checking ``[-4, +4]`` for a 99% value at risk refuses
    a skewness of -0.8 because of behaviour at ``z = +4``, which that number
    does not touch.

    The left end is cut off at four standard deviations rather than carried to
    minus infinity. Beyond there lies 3e-5 of the probability, and a three-term
    expansion is not to be trusted that far out in any case; the expected
    shortfall integral does run to minus infinity, so that tail is included in
    the number and excluded from this check, which is a real if tiny
    inconsistency and is better stated than hidden.

    Checked on a grid rather than solved. The derivative is a quadratic whose
    roots could be found exactly, and at this spacing it cannot hide a sign
    change between adjacent points.
    """
    if upper < lower:
        raise ValueError(f"an empty range, [{lower!r}, {upper!r}], is not monotone or not")
    _, c1, c2, c3 = cornish_fisher_coefficients(skewness, excess_kurtosis)
    step = (upper - lower) / (_MONOTONE_POINTS - 1)
    for index in range(_MONOTONE_POINTS):
        z = lower + index * step
        slope = c1 + 2.0 * c2 * z + 3.0 * c3 * z * z
        if slope <= 0.0:
            return False
    return True


def cornish_fisher_risk(
    *,
    mean: float,
    volatility: float,
    confidence: float,
    skewness: float,
    excess_kurtosis: float,
) -> Risk:
    """Value at risk and expected shortfall corrected for skewness and kurtosis.

    The expected shortfall is a closed form, not a quadrature. Writing the
    corrected quantile as a cubic in ``z`` and integrating against the normal
    density over the tail uses the four tail moments of the normal, each of
    which is exact::

        int phi          = Phi(a)
        int z phi        = -phi(a)
        int z^2 phi      = Phi(a) - a phi(a)
        int z^3 phi      = -(a^2 + 2) phi(a)

    So the tail mean of the corrected variable is exact up to rounding, which
    matters because a quadrature over a tail that reaches to minus infinity is
    exactly where a numerical integral is least reliable.

    Refuses moments for which the mapping is not monotone. The number it would
    otherwise return is not a quantile of anything, and there is no way to tell
    from the number itself.
    """
    alpha = _check_inputs(volatility, confidence)
    a = normal_ppf(alpha)
    # Checked over the tail this number is read from, not over a symmetric range
    # around zero: see is_monotone for why the symmetric question is the wrong
    # one to ask of a left-tail estimate.
    if not is_monotone(skewness, excess_kurtosis, upper=a):
        raise NotAQuantileFunction(
            f"the Cornish-Fisher mapping is not monotone for skewness={skewness!r} "
            f"and excess kurtosis={excess_kurtosis!r} over the tail this reads from "
            f"(z from {-_MONOTONE_LIMIT:g} to {a:.4g}): a higher probability maps to "
            "a lower value there. The expansion does not define a quantile function "
            "over that range, so any number taken from it is not a quantile. Use the "
            "Student-t assumption, or a historical estimate, for moments this extreme."
        )

    z_corrected = cornish_fisher_quantile(a, skewness, excess_kurtosis)
    quantile = mean + volatility * z_corrected

    c0, c1, c2, c3 = cornish_fisher_coefficients(skewness, excess_kurtosis)
    phi_a = normal_pdf(a)
    tail_integral = (
        c0 * alpha
        + c1 * (-phi_a)
        + c2 * (alpha - a * phi_a)
        + c3 * (-(a * a + 2.0) * phi_a)
    )
    shortfall_mean = mean + volatility * tail_integral / alpha
    return Risk(
        value_at_risk=-quantile,
        expected_shortfall=-shortfall_mean,
        quantile=quantile,
        confidence=confidence,
        distribution=Distribution.CORNISH_FISHER,
        mean=mean,
        volatility=volatility,
    )


# -- the portfolio entry point ----------------------------------------------


def parametric_risk(
    *,
    mean: float,
    volatility: float,
    confidence: float,
    distribution: Distribution = Distribution.NORMAL,
    degrees: float = 5.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    _horizon: float = 1.0,
) -> Risk:
    """Dispatch to whichever distributional assumption was asked for."""
    if distribution is Distribution.NORMAL:
        result = normal_risk(mean=mean, volatility=volatility, confidence=confidence)
    elif distribution is Distribution.STUDENT_T:
        result = student_t_risk(
            mean=mean, volatility=volatility, confidence=confidence, degrees=degrees
        )
    else:
        result = cornish_fisher_risk(
            mean=mean,
            volatility=volatility,
            confidence=confidence,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
        )
    if _horizon == 1.0:
        return result
    return Risk(
        value_at_risk=result.value_at_risk,
        expected_shortfall=result.expected_shortfall,
        quantile=result.quantile,
        confidence=result.confidence,
        distribution=result.distribution,
        horizon=_horizon,
        mean=result.mean,
        volatility=result.volatility,
    )


def portfolio_risk(
    weights: Sequence[float],
    covariance: Matrix,
    *,
    means: Sequence[float] | None = None,
    confidence: float = 0.99,
    distribution: Distribution = Distribution.NORMAL,
    degrees: float = 5.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> Risk:
    """Risk of a weighted portfolio, from a covariance matrix.

    ``means`` defaults to zero rather than to the sample mean. That is the
    convention in regulatory and most internal practice, and it is the
    conservative one: expected returns are estimated far less reliably than
    covariances — the standard error of a mean falls as the square root of the
    sample while the drift itself does not accumulate — so a value at risk that
    credits an estimated drift is quietly betting on the least trustworthy number
    in the calculation.
    """
    variance = quadratic_form(list(weights), covariance)
    if variance < 0.0:
        # Only reachable from a covariance matrix that is not positive
        # semi-definite, which is a statement about the matrix rather than about
        # the weights, and is better said here than by math.sqrt.
        raise ValueError(
            f"these weights give a portfolio variance of {variance!r}, which is "
            "negative. The covariance matrix is not positive semi-definite; repair "
            "it with nearest_psd or shrink it before taking risk from it."
        )
    drift = 0.0
    if means is not None:
        if len(means) != len(weights):
            raise ValueError(
                f"{len(means)} means against {len(weights)} weights; they must agree"
            )
        drift = sum(w * m for w, m in zip(weights, means, strict=True))
    return parametric_risk(
        mean=drift,
        volatility=math.sqrt(variance),
        confidence=confidence,
        distribution=distribution,
        degrees=degrees,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )


def normal_tail_mean(a: float) -> float:
    """``E[Z | Z <= a]`` for a standard normal, which is ``-phi(a) / Phi(a)``."""
    probability = normal_cdf(a)
    if probability <= 0.0:
        raise ValueError(f"the tail below {a!r} has no probability to average over")
    return -normal_pdf(a) / probability


__all__ = [
    "Distribution",
    "NotAQuantileFunction",
    "Risk",
    "cornish_fisher_coefficients",
    "cornish_fisher_quantile",
    "cornish_fisher_risk",
    "is_monotone",
    "normal_risk",
    "normal_tail_mean",
    "parametric_risk",
    "portfolio_risk",
    "student_t_risk",
]

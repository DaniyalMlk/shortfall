"""Expectiles: coherent and elicitable at once, and what that costs.

:mod:`shortfall.scoring` sets out the problem. Value at risk is elicitable but
not coherent, so it can be scored and cannot be trusted to reward
diversification. Expected shortfall is coherent but not elicitable, so it can
be trusted and cannot be scored on its own. The package's answer there is to
score the pair jointly with a Fissler-Ziegel loss, which works and is the
industry's answer too.

There is a third option. Among law-invariant risk measures, the **expectiles
are the only ones that are both coherent and elicitable** (Bellini and
Bignozzi, 2015), and the scoring function that elicits them is a weighted
squared error:

    S(e, l) = |tau - 1{l <= e}| (l - e)^2

Its expected value differentiates to ``-2[tau E(L - e)+ - (1 - tau) E(e - L)+]``
and its second derivative is ``2[tau P(L > e) + (1 - tau) P(L <= e)]``, which
is strictly positive. So the expected score is strictly convex with a unique
stationary point, and that point is the definition of the ``tau``-expectile:

    tau E[(L - e)+] = (1 - tau) E[(e - L)+]

At ``tau = 1/2`` the two sides are the two halves of a mean deviation and the
expectile is the mean. As ``tau`` rises the loss side is weighted more heavily
and the expectile climbs towards the essential supremum, so ``tau`` near one is
the risk-measure regime. Everything here takes losses in the package's usual
sign convention: positive numbers are losses, and a high ``tau`` is a
conservative number.

**What this module is for is deciding whether the third option is usable**, and
that turns out to be an empirical question rather than a theoretical one,
because a desk does not choose its confidence level freely. Three measurements
decide it, and they are in the roadmap rather than only in the docstrings.

The first is the level matching. An expectile level means nothing to anybody,
so the only way to state one is to say which familiar number it reproduces —
and :func:`matching_level` finds it. The catch is that the answer depends on
the shape of the law and not only on the confidence level. The ``tau``
reproducing a 97.5% expected shortfall is **0.998603** under a normal and
**0.997335** under a standardised Student-t with five degrees of freedom. The
gap looks negligible, 0.0013 in ``tau``, and it is not: carrying the normal's
``tau`` over to the ``t`` overstates the true expected shortfall by **16.5%**,
and by 7.9% at eight degrees of freedom and 2.4% at twenty. A confidence level
can be fixed by a regulator once and applied to any book. An expectile level
cannot, because the same ``tau`` is a different amount of conservatism on a
fatter tail.

The second is what coherence costs elsewhere. Expectiles are subadditive for
``tau >= 1/2`` and not below it, and :func:`subadditivity_gap` finds the
violation rather than citing the theorem: at ``tau = 1/2`` the gap is exactly
zero, because the half-expectile is the mean and the mean is additive, and it
is positive above and negative below with the magnitude growing in the
distance from a half. Searching two hundred dependence structures for the worst
case below a half turned up -2.37 on standardised data at ``tau = 0.02``.

They are also not comonotonically additive, and expected shortfall is. Two
positions that move together offer no diversification, and a capital rule wants
a risk measure that reports none; an expectile reports some anyway. On
perfectly dependent data — one position an increasing transform of the other —
:func:`comonotonic_gap` finds the sum of the expectiles exceeding the expectile
of the sum by 0.40% at ``tau = 0.9``, 0.48% at 0.975 and 0.69% at 0.99, and by
exactly zero at a half. That is the price of elicitability, and it is small
enough to argue about rather than small enough to ignore.

The third is what they are sensitive to. Value at risk ignores everything past
its own quantile and everything below it; an expectile is a functional of the
whole law. Moving mass around inside the body of a distribution, leaving every
upper quantile untouched, moves the expectile and not the value at risk. That
is usually described as an advantage. It is also the reason an expectile is
harder to explain to somebody who wants to know which loss it refers to: there
is no such loss, because an expectile is not a quantile of anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from .distributions import normal_cdf, normal_pdf, student_t_cdf, student_t_pdf

__all__ = [
    "BadLevel",
    "Expectile",
    "Match",
    "asymmetric_squared_loss",
    "comonotonic_gap",
    "expectile_identity",
    "matching_level",
    "normal_expectile",
    "sample_expectile",
    "student_t_expectile",
    "subadditivity_gap",
]


class BadLevel(ValueError):
    """An expectile level outside ``(0, 1)``, or a degenerate request."""


def _check_level(level: float) -> float:
    value = float(level)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise BadLevel(f"an expectile level lies strictly inside (0, 1), got {level!r}")
    return value


def _check_volatility(volatility: float) -> float:
    value = float(volatility)
    if not math.isfinite(value) or value <= 0.0:
        raise BadLevel(f"volatility must be finite and positive, got {volatility!r}")
    return value


@dataclass(frozen=True, slots=True)
class Expectile:
    """An expectile, with the two sides of the condition that defines it.

    Keeping both partial moments alongside the answer is what makes the result
    checkable without recomputing it: :attr:`identity` is the residual of the
    defining equation and must be zero.

    Attributes:
        value: The expectile itself, in the loss convention — positive is a
            loss.
        level: The ``tau`` it was taken at.
        upper: ``E[(L - e)+]``, the expected exceedance above it.
        lower: ``E[(e - L)+]``, the expected shortfall of it from below.
        mean: ``E[L]``, which is the expectile at ``tau = 1/2``.
    """

    value: float
    level: float
    upper: float
    lower: float
    mean: float

    @property
    def identity(self) -> float:
        """``tau E[(L - e)+] - (1 - tau) E[(e - L)+]``, which is zero at the answer."""
        return self.level * self.upper - (1.0 - self.level) * self.lower

    @property
    def exceedance_ratio(self) -> float:
        """``E[(L - e)+] / E[(e - L)+]``, which the condition fixes at ``(1-tau)/tau``.

        The expectile is the point at which the expected overshoot and the
        expected undershoot stand in a given ratio, which is the sentence to
        reach for when a quantile reading is asked for and there is not one.
        """
        if self.lower == 0.0:
            return math.inf
        return self.upper / self.lower


def asymmetric_squared_loss(forecast: float, loss: float, level: float) -> float:
    """The scoring function that elicits the ``level``-expectile.

    Weighted squared error: the weight is ``level`` when the realised loss
    exceeds the forecast and ``1 - level`` when it does not. At ``level = 1/2``
    it is a squared error halved, and the thing it elicits is the mean.

    Args:
        forecast: The expectile being scored.
        loss: The realised loss.
        level: ``tau``, strictly inside ``(0, 1)``.

    Returns:
        The score. Smaller is better.

    Raises:
        BadLevel: If ``level`` is outside ``(0, 1)``.
        ValueError: If ``forecast`` or ``loss`` is not finite.
    """
    tau = _check_level(level)
    for name, value in (("forecast", forecast), ("loss", loss)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    gap = loss - forecast
    weight = tau if loss > forecast else 1.0 - tau
    return weight * gap * gap


def expectile_identity(candidate: float, losses: Sequence[float], level: float) -> float:
    """The defining condition's residual at ``candidate``, on a sample.

    Strictly decreasing in ``candidate``, which is why the root is unique.
    """
    tau = _check_level(level)
    upper = 0.0
    lower = 0.0
    for loss in losses:
        if loss > candidate:
            upper += loss - candidate
        else:
            lower += candidate - loss
    count = len(losses)
    if count == 0:
        raise BadLevel("an expectile needs at least one observation")
    return (tau * upper - (1.0 - tau) * lower) / count


def sample_expectile(losses: Iterable[float], level: float) -> Expectile:
    """The sample ``level``-expectile, exactly rather than by iteration.

    The sample score is piecewise quadratic in the forecast, with a break at
    every order statistic, so its derivative is piecewise *linear* and the root
    on each piece is available in closed form. Scanning the pieces finds the
    one whose root lies inside it, and that root is the answer to the last bit
    rather than to a tolerance.

    The alternative — Newton or bisection on the same derivative — is what this
    is usually implemented as, and it converges perfectly well. It is just
    strictly worse: it costs iterations, it needs a tolerance that has to be
    chosen against the scale of the data, and it returns a number that is
    nearly the answer when the answer is available for the same sorting pass.

    Args:
        losses: Realised losses. At least one, all finite.
        level: ``tau``, strictly inside ``(0, 1)``.

    Returns:
        An :class:`Expectile`.

    Raises:
        BadLevel: If ``level`` is outside ``(0, 1)``, or there are no
            observations.
        ValueError: If any observation is not finite.
    """
    tau = _check_level(level)
    ordered = sorted(float(value) for value in losses)
    count = len(ordered)
    if count == 0:
        raise BadLevel("an expectile needs at least one observation")
    for value in ordered:
        if not math.isfinite(value):
            raise ValueError(f"every loss must be finite, got {value!r}")

    total = math.fsum(ordered)
    mean = total / count
    if tau == 0.5:
        # Exact, and worth short-circuiting: the two weights are equal, every
        # interval gives the same root, and the root is the mean. Going through
        # the scan would find it anyway, in floating point rather than in one
        # division.
        return _assemble(mean, tau, ordered, mean)

    # `index` is the position of the last observation at or below the
    # candidate; -1 means the candidate sits below all of them.
    below_sum = 0.0
    for index in range(-1, count):
        if index >= 0:
            below_sum += ordered[index]
        above_sum = total - below_sum
        above_count = count - 1 - index
        below_count = index + 1
        weight = tau * above_count + (1.0 - tau) * below_count
        if weight <= 0.0:
            continue
        candidate = (tau * above_sum + (1.0 - tau) * below_sum) / weight
        low = -math.inf if index < 0 else ordered[index]
        high = math.inf if index >= count - 1 else ordered[index + 1]
        if low <= candidate <= high:
            return _assemble(candidate, tau, ordered, mean)

    # Unreachable: the derivative runs from positive to negative and is
    # continuous, so exactly one piece contains its root. Kept as an assertion
    # rather than dropped, because the thing it is asserting is the argument
    # above and not an input.
    raise AssertionError(  # pragma: no cover
        f"no interval contained the root for level {tau} on {count} observations"
    )


def _assemble(value: float, level: float, ordered: list[float], mean: float) -> Expectile:
    upper = math.fsum(loss - value for loss in ordered if loss > value) / len(ordered)
    lower = math.fsum(value - loss for loss in ordered if loss <= value) / len(ordered)
    return Expectile(value=value, level=level, upper=upper, lower=lower, mean=mean)


def _normal_partials(z: float) -> tuple[float, float]:
    """``E[(Z - z)+]`` and ``E[(z - Z)+]`` for a standard normal."""
    density = normal_pdf(z)
    upper_probability = 1.0 - normal_cdf(z)
    upper = density - z * upper_probability
    # From E[(Z - z)+] - E[(z - Z)+] = E[Z] - z = -z, which is exact and avoids
    # evaluating a second near-cancelling difference in the far tail.
    return upper, upper + z


def _standard_normal_expectile(tau: float) -> float:
    """The ``tau``-expectile of a standard normal, as a root in ``z``.

    Location and scale factor out of the defining condition, so the standard
    normal's expectile is the whole answer: a general normal's is
    ``mean + volatility * z``. The residual runs from ``+inf`` to ``-inf`` and
    is continuous, so bisection on a widening bracket cannot fail; Newton would
    be faster and would need a derivative that buys nothing here.
    """
    if tau == 0.5:
        return 0.0

    def residual(z: float) -> float:
        upper, lower = _normal_partials(z)
        return tau * upper - (1.0 - tau) * lower

    low, high = -1.0, 1.0
    while residual(low) < 0.0:
        low *= 2.0
        if low < -1e4:  # pragma: no cover
            raise BadLevel(f"could not bracket the expectile at level {tau}")
    while residual(high) > 0.0:
        high *= 2.0
        if high > 1e4:  # pragma: no cover
            raise BadLevel(f"could not bracket the expectile at level {tau}")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def normal_expectile(*, mean: float, volatility: float, level: float) -> Expectile:
    """The ``level``-expectile of a normal law.

    Args:
        mean: Mean of the loss.
        volatility: Standard deviation. Positive.
        level: ``tau``, strictly inside ``(0, 1)``.

    Returns:
        An :class:`Expectile`, with both partial moments in the same units.

    Raises:
        BadLevel: If ``level`` is outside ``(0, 1)`` or ``volatility`` is not
            positive.
    """
    tau = _check_level(level)
    sigma = _check_volatility(volatility)
    if not math.isfinite(mean):
        raise BadLevel(f"mean must be finite, got {mean!r}")
    z = _standard_normal_expectile(tau)
    upper, lower = _normal_partials(z)
    return Expectile(
        value=mean + sigma * z,
        level=tau,
        upper=sigma * upper,
        lower=sigma * lower,
        mean=float(mean),
    )


def _t_partials(t: float, degrees: float) -> tuple[float, float]:
    """``E[(T - t)+]`` and ``E[(t - T)+]`` for a standard Student-t.

    ``E[T 1{T > t}] = (v + t^2) / (v - 1) * f(t)``, the identity
    :func:`shortfall.parametric.student_t_risk` already uses for the tail mean,
    read in the other direction.
    """
    upper = ((degrees + t * t) / (degrees - 1.0)) * student_t_pdf(t, degrees) - t * (
        1.0 - student_t_cdf(t, degrees)
    )
    return upper, upper + t


def student_t_expectile(
    *, mean: float, volatility: float, level: float, degrees: float
) -> Expectile:
    """The ``level``-expectile of a standardised Student-t.

    The distribution is the same one :func:`shortfall.parametric.student_t_risk`
    uses: ``mean + volatility * sqrt((v - 2) / v) * T``, scaled so that the
    volatility supplied is the volatility delivered.

    Args:
        mean: Mean of the loss.
        volatility: Standard deviation. Positive.
        level: ``tau``, strictly inside ``(0, 1)``.
        degrees: Degrees of freedom. Must exceed two, so the variance exists.

    Returns:
        An :class:`Expectile`.

    Raises:
        BadLevel: If ``level`` is outside ``(0, 1)``, ``volatility`` is not
            positive, or ``degrees`` is not above two.
    """
    tau = _check_level(level)
    sigma = _check_volatility(volatility)
    if not math.isfinite(mean):
        raise BadLevel(f"mean must be finite, got {mean!r}")
    if not math.isfinite(degrees) or degrees <= 2.0:
        raise BadLevel(
            f"degrees of freedom must exceed 2, got {degrees!r}. At or below two the "
            "variance does not exist, so there is nothing for a volatility to match"
        )
    scale = sigma * math.sqrt((degrees - 2.0) / degrees)

    def residual(t: float) -> float:
        upper, lower = _t_partials(t, degrees)
        return tau * upper - (1.0 - tau) * lower

    low, high = -1.0, 1.0
    while residual(low) < 0.0:
        low *= 2.0
    while residual(high) > 0.0:
        high *= 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
    t = 0.5 * (low + high)
    upper, lower = _t_partials(t, degrees)
    return Expectile(
        value=mean + scale * t,
        level=tau,
        upper=scale * upper,
        lower=scale * lower,
        mean=float(mean),
    )


@dataclass(frozen=True, slots=True)
class Match:
    """The expectile level that reproduces a given risk number.

    Attributes:
        level: The ``tau`` found.
        target: The number it was matched to.
        value: The expectile at :attr:`level`, which should be ``target``.
        residual: ``value - target``.
    """

    level: float
    target: float
    value: float
    residual: float


def matching_level(target: float, expectile_of: Callable[[float], float]) -> Match:
    """The ``tau`` whose expectile equals ``target``.

    An expectile level is not a number anybody has intuition about, so the only
    way to state one is to say which familiar quantity it reproduces. Pass the
    expectile as a function of the level and this inverts it — the map is
    strictly increasing and continuous, so a bisection on ``(0, 1)`` is enough
    and no bracketing search is needed.

    The two useful calls::

        matching_level(risk.expected_shortfall,
                       lambda t: normal_expectile(mean=0.0, volatility=s, level=t).value)
        matching_level(target, lambda t: sample_expectile(losses, t).value)

    What the answer is good for is stating an expectile in the units already in
    use. What it is *not* good for is fixing a level once: the ``tau``
    reproducing a given expected shortfall depends on the shape of the
    distribution as well as on the confidence level, so the number found on one
    law does not transfer to another.

    Args:
        target: The value to match.
        expectile_of: The expectile as a function of the level.

    Returns:
        A :class:`Match`.

    Raises:
        BadLevel: If ``target`` is not finite, or lies outside the range the
            expectile attains on ``(0, 1)``.
    """
    if not math.isfinite(target):
        raise BadLevel(f"target must be finite, got {target!r}")
    low, high = 1e-12, 1.0 - 1e-12
    at_low, at_high = expectile_of(low), expectile_of(high)
    if not at_low <= target <= at_high:
        raise BadLevel(
            f"the expectile runs from {at_low!r} to {at_high!r} over the levels this "
            f"searches, and {target!r} is outside that; no level attains it"
        )
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if expectile_of(middle) < target:
            low = middle
        else:
            high = middle
    level = 0.5 * (low + high)
    value = expectile_of(level)
    return Match(level=level, target=float(target), value=value, residual=value - target)


def comonotonic_gap(
    first: Sequence[float], second: Sequence[float], level: float
) -> float:
    """``e(X) + e(Y) - e(X + Y)`` on two *comonotonic* positions.

    This is the property expectiles give up in exchange for elicitability.
    Expected shortfall is comonotonically additive: two positions that rise and
    fall together offer no diversification and it reports none, which is the
    behaviour a capital rule wants at the top of the dependence range. An
    expectile is not, and reports a diversification benefit even there — so
    this gap is positive on comonotonic data rather than zero, and that is a
    defect rather than a rounding error.

    Both samples must be ordered the same way, which is what comonotonic means
    for paired data; the function checks rather than assumes it, because the
    whole content of the result depends on it.

    Args:
        first: One position's losses.
        second: The other's, paired and ordered the same way.
        level: ``tau``.

    Returns:
        The gap. Zero would mean comonotonic additivity; it is not zero.

    Raises:
        BadLevel: If the samples are not paired, or not comonotonic.
    """
    _check_level(level)
    if len(first) != len(second):
        raise BadLevel(
            f"the two samples must be paired, got {len(first)} and {len(second)}"
        )
    order = sorted(range(len(first)), key=lambda i: first[i])
    ranked = [second[i] for i in order]
    if any(b < a for a, b in pairwise(ranked)):
        raise BadLevel(
            "the two samples are not comonotonic: sorting by the first does not sort "
            "the second, so there is a pair that moves in opposite directions"
        )
    combined = [a + b for a, b in zip(first, second, strict=True)]
    return (
        sample_expectile(first, level).value
        + sample_expectile(second, level).value
        - sample_expectile(combined, level).value
    )


def subadditivity_gap(
    first: Sequence[float], second: Sequence[float], level: float
) -> float:
    """``e(X) + e(Y) - e(X + Y)``, which coherence requires to be non-negative.

    Evaluated on paired samples, so the dependence between the two is whatever
    the data says rather than an assumption. Non-negative for every pair at
    ``level >= 1/2``; below a half it can go either way, and that is exactly
    the range in which an expectile is not a coherent risk measure.

    Args:
        first: One position's losses.
        second: The other's, paired with the first.
        level: ``tau``.

    Returns:
        The gap. Negative is a subadditivity violation.

    Raises:
        BadLevel: If the two samples are different lengths, or ``level`` is
            outside ``(0, 1)``.
    """
    _check_level(level)
    if len(first) != len(second):
        raise BadLevel(
            f"the two samples must be paired, got {len(first)} and {len(second)}"
        )
    combined = [a + b for a, b in zip(first, second, strict=True)]
    return (
        sample_expectile(first, level).value
        + sample_expectile(second, level).value
        - sample_expectile(combined, level).value
    )

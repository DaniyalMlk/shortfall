"""Risk estimated from the sample rather than from an assumed shape.

The parametric estimates assume a distribution. These assume only that the past
is a draw from the future — a weaker assumption and a different one, not a
better one, since it trades model risk for the accident of which window was
chosen. A historical estimate far from the parametric one is worth more than
either alone: it is the shape assumption being told it is wrong.

Three things here need care, and each is a decision this module makes explicitly
rather than by default.

**Which empirical quantile.** There are nine standard definitions and they
disagree. On 250 daily returns at 99% confidence they select different order
statistics, and on a short window the disagreement *is* the tail. So the method
is an argument, named, with what it interpolates between stated.

**How far to trust the number.** A 99% value at risk over 250 observations is
computed from about two of them. The point estimate says nothing about that, and
a bootstrap interval says it plainly.

**Volatility clustering.** Plain historical simulation treats a return from a
calm month and one from a crisis as equally informative about tomorrow. Filtered
historical simulation divides each past return by an estimate of the volatility
at the time it happened and multiplies by today's — so the *shape* of the
empirical distribution is kept while its scale becomes current. It is the one
change that makes historical simulation responsive rather than merely
backward-looking.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .parametric import Distribution, Risk
from .series import ReturnSeries, TooShort

#: Decay for the exponentially weighted volatility used by filtered historical
#: simulation. RiskMetrics' figure for daily data: the effective memory is about
#: 75 days, long enough to be an estimate and short enough to move in a crisis.
DEFAULT_DECAY = 0.94

#: Resamples in a bootstrap unless the caller says otherwise. Enough that the
#: interval endpoints are stable to about three digits, which is more than the
#: underlying estimate deserves.
DEFAULT_RESAMPLES = 2000


class QuantileMethod(str, Enum):
    """Which of the standard empirical quantile definitions to use.

    Named after what they do rather than after their numbers in Hyndman and
    Fan, because "type 7" tells a reader nothing and is the source of most of
    the confusion about why two implementations disagree.

    All four locate the same position ``h`` in the sorted sample and differ only
    in what they do when ``h`` is not an integer.

    ``LINEAR``
        Interpolates linearly at ``h = (n - 1) p``. This is Hyndman-Fan type 7,
        and what most software means by "the quantile".
    ``LOWER``
        The observation just below that position, ``floor(h)``. Always a return
        that actually happened, never an interpolation, and in the left tail it
        is the more conservative of the two: it takes the worse of the two
        observations the interpolation sits between.
    ``HIGHER``
        The observation just above it, ``ceil(h)``. The less conservative
        counterpart, and worth having precisely so the pair brackets the
        interpolated answer — if the two are far apart, the sample is too thin
        for the confidence level asked for and the interpolation is doing the
        work.
    ``WEIBULL``
        Linear interpolation at position ``(n + 1) p - 1``. Hyndman-Fan type 6.
        Its plotting position is unbiased for the distribution function, which
        makes it the usual choice in hydrology and extreme-value work — and it
        sits further into the tail than ``LINEAR`` for the same ``p``, which for
        a risk number is a real difference rather than a stylistic one.
    """

    LOWER = "lower"
    HIGHER = "higher"
    LINEAR = "linear"
    WEIBULL = "weibull"


def empirical_quantile(
    values: Sequence[float],
    probability: float,
    *,
    method: QuantileMethod = QuantileMethod.LINEAR,
) -> float:
    """The ``probability`` quantile of ``values`` under ``method``.

    Sorted here rather than requiring sorted input. Requiring it would be faster
    and would make an unsorted argument return a number rather than an error,
    which is the wrong trade for a function whose output is a tail.
    """
    if not values:
        raise TooShort("an empirical quantile needs at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"a probability lies in [0, 1], got {probability!r}")
    ordered = sorted(values)
    count = len(ordered)
    if count == 1:
        return ordered[0]

    if method is QuantileMethod.LOWER:
        return ordered[math.floor((count - 1) * probability)]
    if method is QuantileMethod.HIGHER:
        return ordered[math.ceil((count - 1) * probability)]

    position = (
        (count - 1) * probability
        if method is QuantileMethod.LINEAR
        else (count + 1) * probability - 1.0
    )
    position = max(0.0, min(float(count - 1), position))
    below = math.floor(position)
    above = min(below + 1, count - 1)
    weight = position - below
    return ordered[below] * (1.0 - weight) + ordered[above] * weight


def sample_expected_shortfall(
    values: Sequence[float], probability: float
) -> tuple[float, int]:
    """``E[X | X <= q_p]`` from a sample, and how many observations fed it.

    Returns the *mean of the tail* — signed, so a loss is negative — together
    with the number of observations that contributed.

    The exact sample estimator, which is the average of the worst ``n p``
    observations where ``n p`` need not be an integer::

        (1/p) [ (1/n) sum_{i<=m} x_(i) + (p - m/n) x_(m+1) ],   m = floor(n p)

    The last term is the partial observation. At 99% over 250 returns, ``n p``
    is 2.5: two observations count fully and the third counts half. Dropping the
    partial term understates the tail; rounding ``m`` up overstates it; and both
    errors are largest exactly where the sample is thinnest.

    **Not** the mean of every observation at or below the quantile, which is the
    obvious implementation and is wrong whenever the quantile ties with a value
    many observations share. On a series that is flat most days and falls
    occasionally, the quantile *is* the flat value, every observation is at or
    below it, and the "tail mean" becomes the mean of the whole sample. That is
    not a small error: it turned a tail average of -0.399 into -0.0152, and it
    made expected shortfall appear to violate subadditivity, which it cannot.
    """
    count = len(values)
    if count == 0:
        raise TooShort("an expected shortfall needs at least one observation")
    if not 0.0 < probability <= 1.0:
        raise ValueError(f"a tail probability lies in (0, 1], got {probability!r}")

    ordered = sorted(values)
    weight = count * probability
    full = min(math.floor(weight), count)
    total = math.fsum(ordered[:full])
    used = full
    # The partial observation is taken only when it carries weight worth carrying.
    # A bare `weight > full` compares against an exact integer that `count *
    # probability` almost never lands on in binary: 400 * 0.01 is
    # 4.000000000000000444, so the branch fires on a remainder of 4e-16 and the
    # returned count reads five where the arithmetic is four. The estimate itself
    # is unaffected — a term weighted 4e-16 is far below any precision the number
    # has — but the count is reported as how much data is behind the figure, and
    # overstating that is the one direction this library must not be wrong in.
    #
    # The threshold is the rounding error in the product, `count * eps`. That is a
    # bound rather than a tuned constant: it asks whether the remainder is larger
    # than the error made computing it, the same argument the dispersion guard in
    # `ReturnSeries._shape_ratio` uses.
    remainder = weight - full
    if full < count and remainder > count * sys.float_info.epsilon:
        total += remainder * ordered[full]
        used += 1
    return total / weight, max(used, 1)


@dataclass(frozen=True)
class Interval:
    """A bootstrap confidence interval around an estimate."""

    estimate: float
    lower: float
    upper: float
    level: float
    resamples: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        """Width as a fraction of the estimate, which is how it is read.

        An interval of ±0.004 means nothing without knowing whether the estimate
        is 0.01 or 1.0.
        """
        if self.estimate == 0.0:
            return math.inf
        return self.width / abs(self.estimate)


@dataclass(frozen=True)
class HistoricalRisk:
    """A risk estimate taken from a sample, with how much sample it had."""

    value_at_risk: float
    expected_shortfall: float
    quantile: float
    confidence: float
    observations: int
    #: How many observations contributed to the expected shortfall, counting a
    #: partial one as one. Two is not enough, and the number says so where the
    #: estimate cannot.
    tail_observations: int
    method: QuantileMethod

    @property
    def tail_probability(self) -> float:
        return 1.0 - self.confidence

    @property
    def effective_sample(self) -> float:
        """Observations the tail was estimated from, as a count.

        The honest measure of how much data produced the number: 250 daily
        returns at 99% confidence is two and a half observations, not 250.
        """
        return self.observations * self.tail_probability


def historical_risk(
    returns: Sequence[float] | ReturnSeries,
    *,
    confidence: float = 0.99,
    method: QuantileMethod = QuantileMethod.LINEAR,
) -> HistoricalRisk:
    """Value at risk and expected shortfall from the sample itself.

    Same sign convention as the parametric estimates: both are positive losses.
    """
    values = list(returns.values) if isinstance(returns, ReturnSeries) else list(returns)
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence is strictly inside (0, 1), got {confidence!r}; it is the "
            "confidence level, so 0.99 means the 1% tail"
        )
    if len(values) < 2:
        raise TooShort(
            f"a historical estimate needs at least 2 observations, got {len(values)}"
        )
    alpha = 1.0 - confidence
    quantile = empirical_quantile(values, alpha, method=method)
    tail_mean, used = sample_expected_shortfall(values, alpha)
    return HistoricalRisk(
        value_at_risk=-quantile,
        expected_shortfall=-tail_mean,
        quantile=quantile,
        confidence=confidence,
        observations=len(values),
        tail_observations=used,
        method=method,
    )


def bootstrap_interval(
    returns: Sequence[float] | ReturnSeries,
    *,
    confidence: float = 0.99,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    of_expected_shortfall: bool = False,
    method: QuantileMethod = QuantileMethod.LINEAR,
) -> Interval:
    """A percentile bootstrap interval around a historical estimate.

    Resamples with replacement and takes the percentiles of the resampled
    estimates. That assumes the returns are independent across time, which they
    are not — volatility clusters — so the interval is, if anything, optimistic.
    It is still worth having: a 99% value at risk from 250 observations has an
    interval wide enough to make the point on its own.

    ``seed`` is an argument with a default rather than a hidden source of
    randomness, so the same data gives the same interval twice. A risk number
    that moves when nothing moved is worse than a wrong one.
    """
    values = list(returns.values) if isinstance(returns, ReturnSeries) else list(returns)
    if resamples < 2:
        raise ValueError(f"a bootstrap needs at least 2 resamples, got {resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"an interval level lies strictly inside (0, 1), got {level!r}")

    point = historical_risk(values, confidence=confidence, method=method)
    estimate = point.expected_shortfall if of_expected_shortfall else point.value_at_risk

    rng = random.Random(seed)
    count = len(values)
    draws: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        drawn = historical_risk(sample, confidence=confidence, method=method)
        draws.append(drawn.expected_shortfall if of_expected_shortfall else drawn.value_at_risk)

    tail = (1.0 - level) / 2.0
    return Interval(
        estimate=estimate,
        lower=empirical_quantile(draws, tail),
        upper=empirical_quantile(draws, 1.0 - tail),
        level=level,
        resamples=resamples,
    )


def ewma_volatility(
    returns: Sequence[float], *, decay: float = DEFAULT_DECAY
) -> list[float]:
    """Exponentially weighted volatility at each point, using only the past.

    ``s2[t] = decay * s2[t-1] + (1 - decay) * r[t-1]^2``, so the estimate at
    ``t`` is formed from returns strictly before ``t``. The off-by-one is the
    whole point: an estimate that included ``r[t]`` would divide each return by a
    volatility that knew about it, which flattens the standardised series towards
    a constant and makes the filtered tail far too thin. It would also be
    unusable for forecasting, since tomorrow's return is not available today.

    Seeded with the sample variance of the whole series. That does use the
    future, and is the standard treatment: the seed's influence decays like
    ``decay^t`` and is negligible within about a hundred observations, whereas
    seeding from the first return alone makes the early estimates wild.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay lies strictly inside (0, 1), got {decay!r}")
    count = len(returns)
    if count < 2:
        raise TooShort(f"an EWMA volatility needs at least 2 observations, got {count}")

    # math.fsum rather than sum: exactly rounded, so the seed does not depend on
    # the order of a long series, and — with a constant-volatility series — the
    # filtered estimate reduces to the plain one to the last bit rather than to
    # within a few times 1e-15. That identity is a test, so it is worth having
    # exactly.
    mean = math.fsum(returns) / count
    variance = math.fsum((value - mean) ** 2 for value in returns) / count
    out: list[float] = []
    for index in range(count):
        out.append(math.sqrt(variance))
        variance = decay * variance + (1.0 - decay) * returns[index] ** 2
    return out


@dataclass(frozen=True)
class Filtered:
    """A filtered historical simulation and the scaling that produced it."""

    risk: HistoricalRisk
    #: The volatility the standardised returns were scaled back up to.
    current_volatility: float
    #: The average volatility over the window, for comparison. Today above the
    #: average means the filtered estimate is larger than the plain one.
    average_volatility: float
    #: The standardised returns, which are the empirical shape the estimate is
    #: really made of.
    standardised: tuple[float, ...]

    @property
    def scaling(self) -> float:
        return self.current_volatility / self.average_volatility


def filtered_historical_risk(
    returns: Sequence[float] | ReturnSeries,
    *,
    confidence: float = 0.99,
    decay: float = DEFAULT_DECAY,
    method: QuantileMethod = QuantileMethod.LINEAR,
    volatilities: Sequence[float] | None = None,
    current: float | None = None,
) -> Filtered:
    """Historical simulation with each return rescaled to a current volatility.

    Each past return is divided by the volatility estimated at the time it
    happened and multiplied by the current estimate. The empirical *shape* —
    the skew, the fat tail, the particular way this market falls — is kept,
    while the *scale* becomes current.

    When the volatility is constant the two multiplications cancel exactly and
    this reduces to plain historical simulation. That identity is the thing to
    hold on to: it says the filtering is applied and removed in the same units,
    which is the mistake that is otherwise invisible.

    ``volatilities`` supplies the filter instead of :func:`ewma_volatility`, one
    value per return and aligned the same way — element ``i`` estimated from
    returns strictly before ``i``. A fitted
    :class:`~shortfall.volatility.Garch`'s ``volatilities`` goes straight in, and
    it is a better filter than the default for three reasons the volatility
    module sets out: its decay is estimated rather than assumed, a shock decays
    towards a long-run level, and it has a *forecast*.

    ``current`` is the level the standardised returns are scaled back up to. It
    defaults to the filter's last value, which for an exponential weighting is
    all there is. A model that forecasts should pass its one-step-ahead figure
    instead — the default rescales to the volatility of the day that has just
    finished rather than of the day the position is exposed to, and on a series
    where the two differ the gap is the point of having a model at all.
    """
    values = list(returns.values) if isinstance(returns, ReturnSeries) else list(returns)
    if volatilities is None:
        filter_values = ewma_volatility(values, decay=decay)
    else:
        filter_values = list(volatilities)
        if len(filter_values) != len(values):
            raise ValueError(
                f"{len(filter_values)} filter values against {len(values)} returns. "
                "The filter is one value per return, aligned so that element i was "
                "estimated from returns strictly before i; a series of a different "
                "length would divide each return by a volatility belonging to "
                "another date."
            )
        for index, value in enumerate(filter_values):
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(
                    f"filter value {index} is {value!r}. A volatility is finite and "
                    "non-negative."
                )
    current_level = filter_values[-1] if current is None else float(current)
    if current_level <= 0.0:
        raise ValueError(
            "the current volatility estimate is zero, so past returns cannot be "
            "rescaled to it; the series has no variation in its recent history"
        )
    standardised = [
        value / volatility if volatility > 0.0 else 0.0
        for value, volatility in zip(values, filter_values, strict=True)
    ]
    rescaled = [value * current_level for value in standardised]
    return Filtered(
        risk=historical_risk(rescaled, confidence=confidence, method=method),
        current_volatility=current_level,
        average_volatility=math.fsum(filter_values) / len(filter_values),
        standardised=tuple(standardised),
    )


@dataclass(frozen=True)
class Coherence:
    """Whether combining two positions was penalised by the risk measure.

    A coherent risk measure is subadditive: the risk of a combined position is
    never more than the sum of the parts, because diversification cannot hurt.
    Expected shortfall satisfies this. Value at risk does not, and the failure
    is not a technicality — it means a risk limit written in value at risk can
    reward splitting a book into pieces that each sit just inside the limit
    while the whole sits well outside it.
    """

    combined: float
    parts: tuple[float, ...]
    measure: str

    @property
    def sum_of_parts(self) -> float:
        return sum(self.parts)

    @property
    def subadditive(self) -> bool:
        # A tolerance, because two floating-point sums of the same numbers in
        # different orders differ in the last bits and an exact comparison would
        # report a violation of size 1e-18.
        return self.combined <= self.sum_of_parts * (1.0 + 1e-12) + 1e-15

    @property
    def penalty(self) -> float:
        """How much the measure charged for diversifying. Negative is a benefit."""
        return self.combined - self.sum_of_parts


def check_subadditivity(
    parts: Sequence[Sequence[float]],
    *,
    confidence: float = 0.99,
    expected_shortfall: bool = False,
    method: QuantileMethod = QuantileMethod.LINEAR,
) -> Coherence:
    """Compare the risk of a combined position with the sum of its parts.

    The parts are added period by period, so they must be aligned — which is the
    only way the comparison means anything, since subadditivity is a statement
    about one position against another over the same states of the world.
    """
    if len(parts) < 2:
        raise ValueError("subadditivity compares at least two positions")
    lengths = {len(part) for part in parts}
    if len(lengths) != 1:
        raise ValueError(f"the positions must be aligned; lengths are {sorted(lengths)}")

    def measure(values: Sequence[float]) -> float:
        risk = historical_risk(values, confidence=confidence, method=method)
        return risk.expected_shortfall if expected_shortfall else risk.value_at_risk

    combined = [sum(column) for column in zip(*parts, strict=True)]
    return Coherence(
        combined=measure(combined),
        parts=tuple(measure(part) for part in parts),
        measure="expected shortfall" if expected_shortfall else "value at risk",
    )


def as_risk(estimate: HistoricalRisk) -> Risk:
    """Express a historical estimate in the same shape as a parametric one.

    So the two can be put side by side without a caller unpacking two different
    records. The distribution is recorded as the historical one, and the mean and
    volatility are left at zero because a historical estimate does not have them
    — it did not assume a shape, which is the entire point.
    """
    return Risk(
        value_at_risk=estimate.value_at_risk,
        expected_shortfall=estimate.expected_shortfall,
        quantile=estimate.quantile,
        confidence=estimate.confidence,
        distribution=Distribution.HISTORICAL,
    )


__all__ = [
    "DEFAULT_DECAY",
    "DEFAULT_RESAMPLES",
    "Coherence",
    "Filtered",
    "HistoricalRisk",
    "Interval",
    "QuantileMethod",
    "as_risk",
    "bootstrap_interval",
    "check_subadditivity",
    "empirical_quantile",
    "ewma_volatility",
    "filtered_historical_risk",
    "historical_risk",
    "sample_expected_shortfall",
]

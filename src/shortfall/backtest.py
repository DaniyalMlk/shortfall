"""Scoring a risk model against what actually happened.

Everything else in this library produces a forecast. This module is the part
that asks, once the period has passed, whether the forecast was any good.

A value at risk is a quantile of a predictive distribution, so the only thing
it directly claims is a *frequency*: at 99%, a loss worse than the forecast
should happen about one day in a hundred. That claim is testable, and it
decomposes into two independent ones that fail for different reasons.

The first is coverage. Are there roughly the right number of breaches? A model
with the wrong volatility gets this wrong, and Kupiec's test detects it.

The second is independence. Are the breaches spread out? A model that assumes
constant volatility can have exactly the right *number* of breaches over a
year and still be badly wrong, because all of them land in the same fortnight.
That model is useless for the purpose anyone holds capital for, and the count
alone cannot see it. Christoffersen's test can.

Both matter, and a model can pass either one alone. :func:`conditional_coverage`
is the joint test, and :func:`validate` runs the lot.

Expected shortfall is harder and the difference is not a matter of effort.
Value at risk is *elicitable*: there is a scoring function minimised in
expectation by the true quantile, which is what makes a breach count a test.
Expected shortfall is not elicitable on its own, so there is no equivalent
statistic, and a test for it has to compare realised tail losses against the
forecast tail mean directly. :func:`expected_shortfall_test` implements the two
Acerbi-Szekely statistics, which is the standard answer; their null
distribution has no closed form and is simulated.

Sign convention, which is the thing to get right before reading anything else:
returns are signed, so a loss is negative. Forecasts are positive losses, the
same convention :class:`~shortfall.parametric.Risk` uses. A breach is therefore
``observed < -forecast``, and it is strict — a return exactly equal to minus
the forecast is not a breach, because the quantile is the boundary of the
rejection region and not inside it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from .distributions import binomial_cdf, chi_square_sf, normal_ppf, student_t_ppf
from .parametric import Distribution

#: Below this the likelihood-ratio statistics are reported but should not be
#: read as tests: the chi-square limit is asymptotic, and at a 99% level a
#: hundred observations expect one breach.
ADVISORY_MINIMUM: Final = 100

#: The supervisory setup the traffic-light add-on table is published for.
SUPERVISORY_OBSERVATIONS: Final = 250
SUPERVISORY_CONFIDENCE: Final = 0.99

#: Basel's plus factor by breach count, for the setup above. Not a formula:
#: these are tabulated values, which is why they are not extrapolated to any
#: other observation count.
_PLUS_FACTOR: Final = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

#: Zone boundaries, as cumulative binomial probability of the observed count
#: or fewer. Green below the first, red at or above the second.
_GREEN_CEILING: Final = 0.95
_RED_FLOOR: Final = 0.9999


class BadForecast(ValueError):
    """A forecast series that cannot be scored as it stands."""


class Zone(str, Enum):
    """A supervisory traffic-light zone."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class Exceedances:
    """Where a forecast was breached, and how often."""

    #: One flag per observation, in the order given.
    breaches: tuple[bool, ...]
    confidence: float

    @property
    def observations(self) -> int:
        return len(self.breaches)

    @property
    def count(self) -> int:
        return sum(self.breaches)

    @property
    def expected(self) -> float:
        """How many breaches the confidence level implies."""
        return self.observations * (1.0 - self.confidence)

    @property
    def rate(self) -> float:
        """The realised breach frequency."""
        if self.observations == 0:
            return 0.0
        return self.count / self.observations

    @property
    def transitions(self) -> tuple[int, int, int, int]:
        """``(n00, n01, n10, n11)`` over consecutive pairs.

        ``n01`` is a calm day followed by a breach, ``n11`` a breach followed
        by a breach. Independence is the claim that the two rates implied by
        these are the same.
        """
        counts = [0, 0, 0, 0]
        for previous, current in zip(self.breaches, self.breaches[1:], strict=False):
            counts[2 * int(previous) + int(current)] += 1
        return counts[0], counts[1], counts[2], counts[3]


@dataclass(frozen=True)
class CoverageTest:
    """A likelihood-ratio test of one property of a risk model."""

    name: str
    statistic: float
    degrees_of_freedom: int
    p_value: float
    #: What a rejection would mean, in words, so the number is readable alone.
    interpretation: str
    #: True when too few observations for the chi-square limit to be trusted.
    advisory: bool = False

    def rejects_at(self, level: float) -> bool:
        """Whether the model is rejected at this significance level."""
        if not 0.0 < level < 1.0:
            raise BadForecast(f"a significance level lies strictly in (0, 1), got {level!r}")
        return self.p_value < level


@dataclass(frozen=True)
class TrafficLight:
    """The supervisory zone a breach count falls in."""

    zone: Zone
    count: int
    observations: int
    confidence: float
    #: Probability of this many breaches or fewer under a correct model. The
    #: zone is a function of this and nothing else.
    cumulative_probability: float
    #: Basel's capital multiplier add-on, where it is defined. ``None`` outside
    #: the 250-observation, 99% setup the table is published for, because the
    #: values are tabulated and extrapolating them would be an invention.
    plus_factor: float | None


@dataclass(frozen=True)
class ExpectedShortfallTest:
    """The Acerbi-Szekely statistics, which are zero under a correct model."""

    #: Test 1: the realised tail loss against the forecast tail mean, given
    #: that a breach happened. Blind to how many breaches there were.
    conditional: float
    #: Test 2: the same comparison scaled by the expected number of breaches,
    #: so it moves for both a mis-stated tail mean and a mis-stated frequency.
    unconditional: float
    breaches: int
    observations: int
    #: Simulated, when a null distribution was supplied. One-sided: the
    #: alternative worth testing is an understated tail, which is a negative
    #: statistic.
    conditional_p_value: float | None = None
    unconditional_p_value: float | None = None
    replications: int = 0

    @property
    def direction(self) -> str:
        """Which way the tail is wrong, in words."""
        if self.breaches == 0:
            return "no breaches, so the tail mean was never tested"
        if self.conditional < 0.0:
            return "realised tail losses exceeded the forecast: the tail is understated"
        return "realised tail losses fell short of the forecast: the tail is overstated"


@dataclass(frozen=True)
class Validation:
    """Every test this module runs, over one forecast series."""

    exceedances: Exceedances
    unconditional: CoverageTest
    independence: CoverageTest
    conditional: CoverageTest
    traffic_light: TrafficLight
    expected_shortfall: ExpectedShortfallTest | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def rejected_at(self, level: float = 0.05) -> tuple[str, ...]:
        """The names of the tests that reject, so a caller can branch on one."""
        return tuple(
            test.name
            for test in (self.unconditional, self.independence, self.conditional)
            if test.rejects_at(level)
        )


def _check_series(
    observed: Sequence[float],
    forecasts: Sequence[float],
    confidence: float,
) -> None:
    if not 0.0 < confidence < 1.0:
        raise BadForecast(f"a confidence level lies strictly in (0, 1), got {confidence!r}")
    if len(observed) != len(forecasts):
        raise BadForecast(
            f"there must be one forecast per observation, got {len(observed)} returns "
            f"and {len(forecasts)} forecasts. A forecast series is one-step-ahead, so "
            "the first forecast belongs to the first return and neither is offset."
        )
    if not observed:
        raise BadForecast("a backtest needs at least one observation")
    for index, value in enumerate(forecasts):
        if not math.isfinite(value):
            raise BadForecast(f"forecast {index} is {value!r}, which is not a finite number")
        if value <= 0.0:
            raise BadForecast(
                f"forecast {index} is {value!r}. Forecasts are positive losses, the same "
                "convention Risk.value_at_risk uses; a negative one is the sign "
                "convention being crossed rather than a model predicting a gain."
            )
    for index, value in enumerate(observed):
        if not math.isfinite(value):
            raise BadForecast(f"observation {index} is {value!r}, which is not a finite number")


def exceedances(
    observed: Sequence[float],
    forecasts: Sequence[float],
    *,
    confidence: float,
) -> Exceedances:
    """Which observations breached their forecast.

    ``observed`` are signed returns and ``forecasts`` are positive losses, so a
    breach is ``observed[t] < -forecasts[t]``. The comparison is strict; an
    exact tie is not a breach.
    """
    _check_series(observed, forecasts, confidence)
    return Exceedances(
        breaches=tuple(
            value < -forecast for value, forecast in zip(observed, forecasts, strict=True)
        ),
        confidence=confidence,
    )


def _log(value: float) -> float:
    """``log`` with the convention ``0 log 0 = 0`` applied by the caller.

    Returns zero at zero so that a term ``count * _log(rate)`` vanishes when
    the count is zero, which is the limit of the likelihood and not a special
    case invented to avoid the error.
    """
    return math.log(value) if value > 0.0 else 0.0


def unconditional_coverage(breaches: Exceedances) -> CoverageTest:
    """Kupiec's proportion-of-failures test.

    The likelihood ratio between the breach rate the confidence level claims
    and the rate actually observed, against a chi-square on one degree of
    freedom. It sees the count and nothing else, so it cannot distinguish a
    model whose breaches are evenly spread from one whose breaches all arrive
    in the same week.
    """
    n = breaches.observations
    x = breaches.count
    p = 1.0 - breaches.confidence
    rate = x / n if n else 0.0
    restricted = x * _log(p) + (n - x) * _log(1.0 - p)
    unrestricted = x * _log(rate) + (n - x) * _log(1.0 - rate)
    statistic = max(0.0, -2.0 * (restricted - unrestricted))
    direction = "too many" if rate > p else "too few"
    return CoverageTest(
        name="unconditional coverage",
        statistic=statistic,
        degrees_of_freedom=1,
        p_value=chi_square_sf(statistic, 1),
        interpretation=(
            f"{x} breaches in {n} observations against {breaches.expected:.2f} expected"
            f" — {direction}"
        ),
        advisory=n < ADVISORY_MINIMUM,
    )


def independence(breaches: Exceedances) -> CoverageTest:
    """Christoffersen's test that a breach does not make the next one likelier.

    A first-order Markov chain is fitted to the breach indicator and tested
    against the restriction that both transition probabilities are equal. This
    is the clustering test: volatility clusters, so a model that assumes it
    does not produces breaches that arrive together, and that failure is
    invisible to a count.

    Two degenerate cases, both handled by the ``0 log 0 = 0`` convention rather
    than by a branch. With no breaches at all the chain has no information and
    the statistic is zero — correctly, since nothing about dependence has been
    observed. With breaches that never follow one another the fitted ``pi_11``
    is zero, the restricted and unrestricted likelihoods still differ, and the
    statistic is the evidence *against* clustering that it should be.
    """
    n00, n01, n10, n11 = breaches.transitions
    after_calm = n00 + n01
    after_breach = n10 + n11
    total = after_calm + after_breach
    if total == 0:
        return CoverageTest(
            name="independence",
            statistic=0.0,
            degrees_of_freedom=1,
            p_value=1.0,
            interpretation="one observation, so there are no transitions to test",
            advisory=True,
        )
    pooled = (n01 + n11) / total
    from_calm = n01 / after_calm if after_calm else 0.0
    from_breach = n11 / after_breach if after_breach else 0.0

    restricted = (n00 + n10) * _log(1.0 - pooled) + (n01 + n11) * _log(pooled)
    unrestricted = (
        n00 * _log(1.0 - from_calm)
        + n01 * _log(from_calm)
        + n10 * _log(1.0 - from_breach)
        + n11 * _log(from_breach)
    )
    statistic = max(0.0, -2.0 * (restricted - unrestricted))
    if after_breach == 0:
        detail = "no breach was followed by another observation, so clustering is untested"
    else:
        detail = (
            f"a breach follows a calm day {from_calm:.2%} of the time and another "
            f"breach {from_breach:.2%} of the time"
        )
    return CoverageTest(
        name="independence",
        statistic=statistic,
        degrees_of_freedom=1,
        p_value=chi_square_sf(statistic, 1),
        interpretation=detail,
        advisory=breaches.observations < ADVISORY_MINIMUM,
    )


def conditional_coverage(breaches: Exceedances) -> CoverageTest:
    """The joint test: right number of breaches, and spread out.

    The sum of the two statistics above, on two degrees of freedom, because
    the two restrictions are asymptotically independent. Worth reporting
    alongside its parts rather than instead of them: a rejection here does not
    say which half failed, and the answer changes what a caller should do —
    the wrong count is a scaling problem, and clustering is a model that needs
    a volatility process.
    """
    coverage = unconditional_coverage(breaches)
    spread = independence(breaches)
    statistic = coverage.statistic + spread.statistic
    return CoverageTest(
        name="conditional coverage",
        statistic=statistic,
        degrees_of_freedom=2,
        p_value=chi_square_sf(statistic, 2),
        interpretation=f"{coverage.interpretation}; {spread.interpretation}",
        advisory=coverage.advisory or spread.advisory,
    )


def traffic_light(breaches: Exceedances) -> TrafficLight:
    """The supervisory zone, derived from the binomial rather than tabulated.

    The published table is a table of counts, for 250 observations at 99%. The
    rule underneath it is a statement about cumulative probability — green
    below 95%, red at 99.99% or above — and deriving the zones from that
    reproduces the published counts exactly while also answering for a sample
    that is not 250 days long.

    The capital add-on is the other way round. Those numbers are tabulated with
    no formula behind them, so they are returned for the supervisory setup and
    ``None`` anywhere else.
    """
    n = breaches.observations
    x = breaches.count
    p = 1.0 - breaches.confidence
    cumulative = binomial_cdf(x, n, p)
    if cumulative < _GREEN_CEILING:
        zone = Zone.GREEN
    elif cumulative < _RED_FLOOR:
        zone = Zone.YELLOW
    else:
        zone = Zone.RED
    supervisory = n == SUPERVISORY_OBSERVATIONS and breaches.confidence == SUPERVISORY_CONFIDENCE
    plus: float | None = None
    if supervisory:
        plus = 0.0 if zone is Zone.GREEN else _PLUS_FACTOR.get(x, 1.0)
    return TrafficLight(
        zone=zone,
        count=x,
        observations=n,
        confidence=breaches.confidence,
        cumulative_probability=cumulative,
        plus_factor=plus,
    )


def expected_shortfall_test(
    observed: Sequence[float],
    value_at_risk: Sequence[float],
    expected_shortfall: Sequence[float],
    *,
    confidence: float,
) -> ExpectedShortfallTest:
    """The Acerbi-Szekely statistics, both zero in expectation under the null.

    Test 1 averages the ratio of realised loss to forecast tail mean over the
    breaches only. It says whether the tail is the right *size* given that the
    tail was reached, and is deliberately blind to how often it was reached —
    which makes it a clean complement to a coverage test rather than a noisier
    version of one.

    Test 2 divides by the expected number of breaches instead of the realised
    number, so a model that breaches twice as often as it should fails it even
    when each breach is the right size.

    A negative statistic means realised losses were worse than forecast, which
    is the direction that costs money. Neither has a closed-form null
    distribution; :func:`simulate_expected_shortfall_null` supplies one.
    """
    _check_series(observed, value_at_risk, confidence)
    if len(expected_shortfall) != len(observed):
        raise BadForecast(
            f"there must be one expected shortfall per observation, got "
            f"{len(expected_shortfall)} for {len(observed)} returns"
        )
    for index, (var, es) in enumerate(zip(value_at_risk, expected_shortfall, strict=True)):
        if not math.isfinite(es) or es <= 0.0:
            raise BadForecast(
                f"expected shortfall {index} is {es!r}; it is a positive loss, like the "
                "value at risk beside it"
            )
        if es < var:
            raise BadForecast(
                f"expected shortfall {index} is {es!r}, below the value at risk {var!r}. "
                "The tail mean is an average of losses at least as large as the "
                "quantile, so it can never be the smaller of the two."
            )
    n = len(observed)
    p = 1.0 - confidence
    total = math.fsum(
        value / es
        for value, var, es in zip(observed, value_at_risk, expected_shortfall, strict=True)
        if value < -var
    )
    count = sum(1 for value, var in zip(observed, value_at_risk, strict=True) if value < -var)
    return ExpectedShortfallTest(
        conditional=(total / count + 1.0) if count else 0.0,
        unconditional=total / (n * p) + 1.0,
        breaches=count,
        observations=n,
    )


def _standard_draw(rng: random.Random, distribution: Distribution, degrees: float) -> float:
    """One draw from the standardised null, by inverse transform."""
    uniform = rng.random()
    if distribution is Distribution.STUDENT_T:
        # Scaled to unit variance, which is the convention parametric_risk uses.
        return student_t_ppf(uniform, degrees) * math.sqrt((degrees - 2.0) / degrees)
    return normal_ppf(uniform)


def simulate_expected_shortfall_null(
    value_at_risk: Sequence[float],
    expected_shortfall: Sequence[float],
    *,
    confidence: float,
    distribution: Distribution = Distribution.NORMAL,
    degrees: float = 5.0,
    replications: int = 2000,
    seed: int = 0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Simulated null distributions for the two statistics.

    Returns come from the predictive distribution the forecasts were built
    under, scaled at each date by the volatility that forecast implies, so the
    simulation reproduces the same forecast series it is scoring. Under the
    null that is exactly the data-generating process, which is what makes the
    resulting quantiles the right critical values.

    The p-value this supports has a simulation error of its own, about
    ``sqrt(p(1-p)/replications)``: at the default 2000 replications a true 5%
    is estimated to within half a percentage point, which is enough to act on
    and not enough to report to three digits.
    """
    if replications < 1:
        raise BadForecast(f"a simulation needs at least one replication, got {replications!r}")
    if distribution is Distribution.STUDENT_T and degrees <= 2.0:
        raise BadForecast(
            f"a Student-t needs more than two degrees of freedom to have a finite "
            f"variance to standardise by, got {degrees!r}"
        )
    rng = random.Random(seed)
    n = len(value_at_risk)
    p = 1.0 - confidence
    # The volatility each forecast implies, so the simulated returns are the
    # ones that forecast was a quantile of.
    quantile = abs(_standard_quantile(p, distribution, degrees))
    scales = [var / quantile for var in value_at_risk]
    first: list[float] = []
    second: list[float] = []
    for _ in range(replications):
        total = 0.0
        count = 0
        for scale, var, es in zip(scales, value_at_risk, expected_shortfall, strict=True):
            draw = _standard_draw(rng, distribution, degrees) * scale
            if draw < -var:
                total += draw / es
                count += 1
        first.append(total / count + 1.0 if count else 0.0)
        second.append(total / (n * p) + 1.0)
    return tuple(first), tuple(second)


def _standard_quantile(tail: float, distribution: Distribution, degrees: float) -> float:
    if distribution is Distribution.STUDENT_T:
        return student_t_ppf(tail, degrees) * math.sqrt((degrees - 2.0) / degrees)
    return normal_ppf(tail)


def _one_sided_p_value(statistic: float, null: Sequence[float]) -> float:
    """``P(Z <= statistic)`` under the simulated null, with the observed value
    counted in the numerator so that the p-value can never be exactly zero."""
    worse = sum(1 for draw in null if draw <= statistic)
    return (worse + 1) / (len(null) + 1)


def validate(
    observed: Sequence[float],
    value_at_risk: Sequence[float],
    *,
    confidence: float,
    expected_shortfall: Sequence[float] | None = None,
    distribution: Distribution = Distribution.NORMAL,
    degrees: float = 5.0,
    replications: int = 0,
    seed: int = 0,
) -> Validation:
    """Run every test in this module over one forecast series.

    ``replications`` above zero simulates the null for the expected-shortfall
    statistics and attaches p-values to them; it is off by default because the
    simulation costs a second or so and the statistics are readable without it.
    """
    breaches = exceedances(observed, value_at_risk, confidence=confidence)
    warnings: list[str] = []
    if breaches.observations < ADVISORY_MINIMUM:
        warnings.append(
            f"{breaches.observations} observations. The likelihood-ratio tests are "
            f"asymptotic and their sizes are unreliable below about {ADVISORY_MINIMUM}; "
            "the statistics are reported but should be read as descriptive."
        )
    if breaches.count == 0:
        warnings.append(
            "no breaches at all. The independence test has nothing to work with, and "
            "the expected-shortfall statistics are not testing the tail mean."
        )
    shortfall_test: ExpectedShortfallTest | None = None
    if expected_shortfall is not None:
        shortfall_test = expected_shortfall_test(
            observed, value_at_risk, expected_shortfall, confidence=confidence
        )
        if replications > 0:
            first, second = simulate_expected_shortfall_null(
                value_at_risk,
                expected_shortfall,
                confidence=confidence,
                distribution=distribution,
                degrees=degrees,
                replications=replications,
                seed=seed,
            )
            shortfall_test = ExpectedShortfallTest(
                conditional=shortfall_test.conditional,
                unconditional=shortfall_test.unconditional,
                breaches=shortfall_test.breaches,
                observations=shortfall_test.observations,
                conditional_p_value=_one_sided_p_value(shortfall_test.conditional, first),
                unconditional_p_value=_one_sided_p_value(shortfall_test.unconditional, second),
                replications=replications,
            )
    return Validation(
        exceedances=breaches,
        unconditional=unconditional_coverage(breaches),
        independence=independence(breaches),
        conditional=conditional_coverage(breaches),
        traffic_light=traffic_light(breaches),
        expected_shortfall=shortfall_test,
        warnings=tuple(warnings),
    )


__all__ = [
    "ADVISORY_MINIMUM",
    "BadForecast",
    "CoverageTest",
    "Exceedances",
    "ExpectedShortfallTest",
    "TrafficLight",
    "Validation",
    "Zone",
    "conditional_coverage",
    "exceedances",
    "expected_shortfall_test",
    "independence",
    "simulate_expected_shortfall_null",
    "traffic_light",
    "unconditional_coverage",
    "validate",
]

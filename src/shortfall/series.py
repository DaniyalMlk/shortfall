"""Return series, panels of them, and annualisation.

Two conventions are decided here and then relied on everywhere else, because
both are the sort of thing that is obvious to whoever wrote the code and
invisible to whoever reads the number.

**Simple against logarithmic.** A simple return is ``P_t / P_{t-1} - 1``; a log
return is ``ln(P_t / P_{t-1})``. They differ by about half the variance, which
is small enough to overlook and large enough to matter, and they aggregate in
opposite directions: log returns add across time and simple returns add across
a portfolio. Neither is correct in general, so a series carries which one it is
and the routines that care ask.

**Periods per year is an argument, never a guess.** The 252 that appears in most
code is the number of US equity trading days, and it is wrong for weekly data,
for monthly data, for crypto, and for any market with a different holiday
calendar. Annualising with the wrong number scales volatility by the square root
of the ratio, so using 252 on weekly data overstates it by a factor of seven —
which does not look wrong enough to notice.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

#: Periods per year for a few common frequencies. Offered as names so a caller
#: writes ``DAILY`` rather than ``252`` and the assumption is legible in the
#: call. None of them is a default.
DAILY_TRADING = 252
DAILY_CALENDAR = 365
WEEKLY = 52
MONTHLY = 12
QUARTERLY = 4
ANNUAL = 1


class Convention(str, Enum):
    """Which definition of return a series holds."""

    SIMPLE = "simple"
    LOG = "log"


class TooShort(ValueError):
    """The series has too few observations for the estimator asked for."""


class Misaligned(ValueError):
    """Series that must share an index do not."""


@dataclass(frozen=True)
class ReturnSeries:
    """One asset's periodic returns, with the convention they were computed under."""

    name: str
    values: tuple[float, ...]
    convention: Convention = Convention.SIMPLE

    def __post_init__(self) -> None:
        for index, value in enumerate(self.values):
            if value != value or math.isinf(value):
                raise ValueError(f"{self.name}[{index}] is not finite: {value!r}")
            if self.convention is Convention.SIMPLE and value <= -1.0:
                # -1 is a total loss and the end of the series; below it the
                # price went negative, which is a data error rather than a
                # return.
                raise ValueError(
                    f"{self.name}[{index}] is {value!r}; a simple return of -1 is a "
                    "total loss and nothing below it is a return at all"
                )

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def from_prices(
        cls,
        name: str,
        prices: Sequence[float],
        *,
        convention: Convention = Convention.SIMPLE,
    ) -> ReturnSeries:
        """Differences of a price series, under ``convention``.

        ``n`` prices give ``n - 1`` returns. Worth stating because an
        off-by-one here shifts every return by one period against whatever it is
        being compared to, which shows up as a plausible-looking correlation
        rather than as an error.
        """
        if len(prices) < 2:
            raise TooShort(f"{name}: {len(prices)} prices give no returns")
        for index, price in enumerate(prices):
            if price <= 0.0 or price != price or math.isinf(price):
                raise ValueError(f"{name}: price {index} is {price!r}; prices are positive")
        if convention is Convention.LOG:
            values = [
                math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))
            ]
        else:
            values = [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]
        return cls(name=name, values=tuple(values), convention=convention)

    def to_log(self) -> ReturnSeries:
        """The same series expressed as log returns."""
        if self.convention is Convention.LOG:
            return self
        return ReturnSeries(
            name=self.name,
            values=tuple(math.log1p(value) for value in self.values),
            convention=Convention.LOG,
        )

    def to_simple(self) -> ReturnSeries:
        """The same series expressed as simple returns."""
        if self.convention is Convention.SIMPLE:
            return self
        return ReturnSeries(
            name=self.name,
            values=tuple(math.expm1(value) for value in self.values),
            convention=Convention.SIMPLE,
        )

    @property
    def mean(self) -> float:
        if not self.values:
            raise TooShort(f"{self.name} is empty")
        return sum(self.values) / len(self.values)

    def variance(self, *, ddof: int = 1) -> float:
        """Sample variance. ``ddof=1`` is unbiased; ``ddof=0`` is the maximum
        likelihood estimate, which the shrinkage estimator is defined in terms of."""
        count = len(self.values)
        if count - ddof <= 0:
            raise TooShort(
                f"{self.name} has {count} observations, too few for a variance with "
                f"ddof={ddof}"
            )
        centre = self.mean
        return sum((value - centre) ** 2 for value in self.values) / (count - ddof)

    def stdev(self, *, ddof: int = 1) -> float:
        return math.sqrt(self.variance(ddof=ddof))

    def _central_moment(self, order: int) -> float:
        """``(1/n) sum (x - xbar)^order``, the maximum-likelihood moment.

        Divided by ``n`` and not by ``n - 1``: this is the raw ingredient, and the
        finite-sample corrections in the two methods below are written in terms of
        the uncorrected ratios, as their published forms are.
        """
        count = len(self.values)
        centre = self.mean
        return sum((value - centre) ** order for value in self.values) / count

    def _shape_ratio(self, order: int) -> float:
        """The standardised moment ``m_order / m_2 ** (order / 2)``.

        Refuses a series with no dispersion rather than dividing by something
        indistinguishable from zero. A constant series has no shape to describe:
        every standardised moment is ``0 / 0``, and no value for it is more
        defensible than another. Returning zero would be the tempting choice and
        the wrong one, because zero skewness and zero excess kurtosis are exactly
        what a *normal* series reports, so a flat series would come back looking
        better behaved than the data it came from.

        **Testing ``m_2 <= 0`` is not enough**, which is the whole reason this is a
        method and not an inline division. Fifty copies of ``0.004`` do not have a
        mean that is exactly ``0.004`` in binary, so each deviation is around
        ``1e-18`` rather than zero, ``m_2`` comes out near ``1e-36``, and the ratio
        is then whatever the rounding noise happened to be — a number with a
        plausible magnitude and no meaning at all.

        So the comparison is against the rounding error in ``m_2`` itself, which
        is about ``n * eps * sum(x^2)``. That is a threshold with no tuned
        constant in it: it asks whether the dispersion computed is larger than the
        error made computing it, and refuses when it is not.
        """
        count = len(self.values)
        centre = self.mean
        second = sum((value - centre) ** 2 for value in self.values)
        noise = count * sys.float_info.epsilon * sum(value * value for value in self.values)
        if second <= noise:
            raise ValueError(
                f"{self.name} has no usable dispersion across {count} observations: the "
                f"sum of squared deviations is {second!r}, at or below the "
                f"{noise!r} of rounding error incurred computing it. A series that is "
                "constant, or constant to within floating-point resolution, has no "
                "distribution shape to estimate."
            )
        return self._central_moment(order) / math.pow(second / count, order / 2.0)

    def skewness(self, *, corrected: bool = True) -> float:
        """Sample skewness.

        ``corrected=True`` is the bias-corrected estimator, ``G1``, which is the
        default for the same reason ``variance`` defaults to ``ddof=1``: the
        uncorrected ratio is biased towards zero in a short sample, and a risk
        number that understates asymmetry is the wrong way to be wrong. It needs
        three observations.

        ``corrected=False`` is the population moment ratio ``g1 = m3 / m2**1.5``,
        which is what a caller comparing against a formula written in central
        moments will want.

        The correction is ``G1 = g1 * sqrt(n (n - 1)) / (n - 2)``. It matters at
        the sample sizes risk work actually runs on: on sixty observations it is
        a few percent, on twelve it is over ten.
        """
        count = len(self.values)
        minimum = 3 if corrected else 2
        if count < minimum:
            raise TooShort(
                f"{self.name} has {count} observations; skewness needs at least {minimum}"
                + (", or 2 with corrected=False" if corrected else "")
            )
        ratio = self._shape_ratio(3)
        if not corrected:
            return ratio
        return ratio * math.sqrt(count * (count - 1)) / (count - 2)

    def excess_kurtosis(self, *, corrected: bool = True) -> float:
        """Sample excess kurtosis: a normal series reports zero, not three.

        The convention is excess throughout this library — see
        :func:`shortfall.cornish_fisher_coefficients`, which is written in terms
        of it — because the two conventions differ by exactly the value a normal
        takes, so a raw kurtosis passed where excess is wanted reads as heavy
        tails on a distribution that has none.

        ``corrected=True`` is ``G2``, the bias-corrected estimator, and needs four
        observations. ``corrected=False`` is ``g2 = m4 / m2**2 - 3``.

        The correction is ``G2 = (n - 1) / ((n - 2)(n - 3)) * ((n + 1) g2 + 6)``.
        Unlike the skewness correction it is not a simple inflation: on a sample
        that is genuinely normal the uncorrected ratio is biased *downwards*, with
        expectation exactly ``-6 / (n + 1)``, so a short window of well-behaved
        returns looks thin-tailed rather than normal. Note the ``n + 1``: the
        commonly quoted ``-6 / n`` is close but wrong, and the difference is four
        standard errors at twenty observations, which is where it matters most.
        """
        count = len(self.values)
        minimum = 4 if corrected else 2
        if count < minimum:
            raise TooShort(
                f"{self.name} has {count} observations; excess kurtosis needs at least "
                f"{minimum}" + (", or 2 with corrected=False" if corrected else "")
            )
        ratio = self._shape_ratio(4) - 3.0
        if not corrected:
            return ratio
        return (count - 1) / ((count - 2) * (count - 3)) * ((count + 1) * ratio + 6.0)

    def cumulative(self) -> float:
        """Total return over the whole series, as a simple return."""
        if self.convention is Convention.LOG:
            return math.expm1(sum(self.values))
        growth = 1.0
        for value in self.values:
            growth *= 1.0 + value
        return growth - 1.0

    def annualised_return(self, periods_per_year: float) -> float:
        """Geometric annualised return, as a simple return.

        Geometric rather than the arithmetic mean scaled up. The arithmetic mean
        is the larger of the two by roughly half the variance, and it is not a
        return anybody earns: a series that gains 50% and then loses 50% has an
        arithmetic mean of zero and has lost a quarter of its money.
        """
        _check_periods(periods_per_year)
        count = len(self.values)
        if count == 0:
            raise TooShort(f"{self.name} is empty")
        growth = 1.0 + self.cumulative()
        if growth <= 0.0:
            # Reachable only by underflow: every simple return is above -1 by
            # construction, so the product of (1 + r) is strictly positive in
            # exact arithmetic. A long enough run of large losses can still
            # round it to zero, and there is no annual rate that compounds to
            # nothing, so it is refused rather than returned as -1.
            raise ValueError(
                f"{self.name} compounds to a growth factor of {growth!r}; there is no "
                "annualised return for a series that has lost everything"
            )
        # exp(log(g) * k) rather than g ** k: the second is a complex result for
        # a negative base and a fractional exponent, and writing it this way
        # makes it impossible to reach that branch by accident.
        return math.exp(math.log(growth) * (periods_per_year / count)) - 1.0

    def annualised_volatility(self, periods_per_year: float, *, ddof: int = 1) -> float:
        """Volatility scaled by the square root of time.

        Which assumes returns are serially uncorrelated. They are not, quite,
        and the scaling understates the volatility of a trending series and
        overstates that of a mean-reverting one. The assumption is universal and
        worth naming rather than defending.
        """
        _check_periods(periods_per_year)
        return self.stdev(ddof=ddof) * math.sqrt(periods_per_year)


def _check_periods(periods_per_year: float) -> None:
    if periods_per_year <= 0 or periods_per_year != periods_per_year:
        raise ValueError(
            f"periods_per_year must be positive, got {periods_per_year!r}. It is the "
            "number of observations in a year — 252 for US trading days, 52 for "
            "weekly, 12 for monthly — and has no default because guessing it wrong "
            "scales every volatility by the square root of the ratio."
        )


@dataclass(frozen=True)
class Panel:
    """Several series over one shared index.

    Equal length is enforced at construction. The alternative — carrying ragged
    series and aligning at the point of use — puts the alignment decision inside
    every estimator, where it is invisible and easy to get differently wrong in
    each one.
    """

    series: tuple[ReturnSeries, ...]

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError("a panel holds at least one series")
        lengths = {len(one) for one in self.series}
        if len(lengths) != 1:
            report = ", ".join(f"{one.name}={len(one)}" for one in self.series)
            raise Misaligned(f"series have different lengths: {report}")
        conventions = {one.convention for one in self.series}
        if len(conventions) != 1:
            raise Misaligned(
                "series mix simple and log returns; they differ by about half the "
                "variance and must not be estimated from together"
            )
        names = [one.name for one in self.series]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate series names: {names}")

    @property
    def names(self) -> list[str]:
        return [one.name for one in self.series]

    @property
    def assets(self) -> int:
        return len(self.series)

    @property
    def observations(self) -> int:
        return len(self.series[0])

    @property
    def convention(self) -> Convention:
        return self.series[0].convention

    def __len__(self) -> int:
        return self.assets

    def __getitem__(self, name: str) -> ReturnSeries:
        for one in self.series:
            if one.name == name:
                return one
        raise KeyError(name)

    def column(self, index: int) -> ReturnSeries:
        return self.series[index]

    def row(self, index: int) -> list[float]:
        """One period's return across every asset."""
        return [one.values[index] for one in self.series]

    def means(self) -> list[float]:
        return [one.mean for one in self.series]

    def demeaned(self) -> list[list[float]]:
        """Each series with its own mean removed, as rows."""
        return [
            [value - one.mean for value in one.values] for one in self.series
        ]

    def require(self, minimum: int, what: str) -> None:
        """Raise unless there are at least ``minimum`` observations."""
        if self.observations < minimum:
            raise TooShort(
                f"{what} needs at least {minimum} observations, this panel has "
                f"{self.observations}"
            )

    @classmethod
    def from_columns(
        cls,
        columns: Mapping[str, Sequence[float]],
        *,
        convention: Convention = Convention.SIMPLE,
    ) -> Panel:
        """Build from ``{name: returns}``, which must already be aligned."""
        return cls(
            tuple(
                ReturnSeries(name=name, values=tuple(values), convention=convention)
                for name, values in columns.items()
            )
        )

    @classmethod
    def aligned(
        cls,
        columns: Mapping[str, Mapping[Any, float]],
        *,
        convention: Convention = Convention.SIMPLE,
    ) -> Panel:
        """Build from ``{name: {date: return}}``, keeping only shared dates.

        An inner join, not an outer one with holes filled. Filling a missing
        return with zero is the common shortcut and it is a lie with a direction:
        it says the asset did not move, which lowers its estimated volatility and
        drags every correlation involving it towards zero. Dropping the date
        loses information honestly.
        """
        if not columns:
            raise ValueError("a panel holds at least one series")
        shared: set[Any] | None = None
        for index in columns.values():
            keys = set(index)
            shared = keys if shared is None else (shared & keys)
        assert shared is not None
        if not shared:
            raise Misaligned("the series share no dates, so there is nothing to estimate")
        order = sorted(shared)
        return cls.from_columns(
            {name: [index[date] for date in order] for name, index in columns.items()},
            convention=convention,
        )

    @classmethod
    def from_prices(
        cls,
        columns: Mapping[str, Sequence[float]],
        *,
        convention: Convention = Convention.SIMPLE,
    ) -> Panel:
        """Build from aligned price series."""
        return cls(
            tuple(
                ReturnSeries.from_prices(name, prices, convention=convention)
                for name, prices in columns.items()
            )
        )

    def portfolio(self, weights: Sequence[float], *, name: str = "portfolio") -> ReturnSeries:
        """The weighted combination of the panel's series.

        Weights are applied to *simple* returns, converting first if the panel
        holds log returns, because simple returns are the ones that add across a
        portfolio. A weighted sum of log returns is not the log return of the
        weighted portfolio, and the difference is not small for concentrated
        weights.
        """
        if len(weights) != self.assets:
            raise ValueError(
                f"{len(weights)} weights against {self.assets} assets; they must agree"
            )
        simple = [one.to_simple() for one in self.series]
        values = [
            sum(weight * one.values[t] for weight, one in zip(weights, simple, strict=True))
            for t in range(self.observations)
        ]
        return ReturnSeries(name=name, values=tuple(values), convention=Convention.SIMPLE)


def as_panel(series: Iterable[ReturnSeries]) -> Panel:
    return Panel(tuple(series))


__all__ = [
    "ANNUAL",
    "DAILY_CALENDAR",
    "DAILY_TRADING",
    "MONTHLY",
    "QUARTERLY",
    "WEEKLY",
    "Convention",
    "Misaligned",
    "Panel",
    "ReturnSeries",
    "TooShort",
    "as_panel",
]

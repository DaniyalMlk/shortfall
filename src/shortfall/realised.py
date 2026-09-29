"""Volatility from the whole bar, not just the close.

Everything else in this library sees a period as one number, the close-to-close
return. A bar usually carries three more — the open, the high and the low — and
the two extremes in particular say a great deal. A day that closes where it
opened after travelling four per cent in between is not a quiet day, and the
close-to-close estimator records it as one.

Five estimators are here, and the useful thing about them is not that some are
better than others but that each buys its efficiency by assuming something:

===================== ============ ============ ==================
estimator             zero drift?  no gap?      uses
===================== ============ ============ ==================
close to close        not assumed  not assumed  close
Parkinson             assumed      assumed      high, low
Garman-Klass          assumed      assumed      all four
Rogers-Satchell       not assumed  assumed      all four
Garman-Klass-Yang-Zhang assumed    not assumed  all four, previous
Yang-Zhang            not assumed  not assumed  all four, previous
===================== ============ ============ ==================

Which one to reach for is decided by which assumption is false in the data at
hand, and by how much being wrong costs. Both are measured rather than
asserted: :func:`efficiency` measures the variance ratio on the caller's own
sample size, and the README carries the drift and gap sensitivities as a table
produced by the tests.

**The number that matters most is not in the textbooks.** Every range
estimator's efficiency is derived for the continuous high and low. Real ones
come from finitely many trades, and the observed extremes are always inside the
true ones, so every estimator built on the range is biased *down*. The bias is
large at realistic tick counts and it does not shrink with more bars, because
it is a bias and not noise. At a hundred trades a bar Parkinson understates
volatility by about seven per cent; at a thousand, by about two and a half.
That is a systematic error where the efficiency gain is a variance reduction,
and for a short sample the trade can still be worth it — but it has to be made
knowingly, which is why :func:`tick_bias_factor` exists and why the
command-line output prints it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

__all__ = [
    "BadBar",
    "Bar",
    "Bars",
    "Estimator",
    "close_to_close",
    "efficiency",
    "garman_klass",
    "garman_klass_yang_zhang",
    "parkinson",
    "realised_variance",
    "realised_volatility",
    "rogers_satchell",
    "tick_bias_factor",
    "yang_zhang",
]

_LOG_2 = math.log(2.0)

# Garman and Klass's coefficient on the squared open-to-close move. It is not a
# fitted constant: 2 log 2 - 1 is what falls out of minimising the variance of
# a combination of the squared range and the squared close, subject to being
# unbiased.
_GK_CLOSE = 2.0 * _LOG_2 - 1.0

# The discrete extreme's expected overshoot, -zeta(1/2) / sqrt(2 pi).
_OVERSHOOT = 0.5825971579390106

# Two ends of the range shrinking by that, relative to the expected range of a
# continuously observed Brownian motion, which is 2 sqrt(2 / pi) volatilities.
_RANGE_SHORTFALL = 2.0 * _OVERSHOOT / (2.0 * math.sqrt(2.0 / math.pi))


class BadBar(ValueError):
    """A bar that is not a bar.

    Raised rather than repaired. A high below the close and a low above the
    open each break a different estimator in a different way — one returns a
    negative variance, another silently returns something plausible and wrong —
    so the wrong bar is refused where it enters rather than where it bites.
    """


@dataclass(frozen=True, slots=True)
class Bar:
    """One period's open, high, low and close.

    Attributes:
        open: Price at the start of the period. Positive.
        high: Highest traded price. At least the open and the close.
        low: Lowest traded price. At most the open and the close.
        close: Price at the end of the period. Positive.
    """

    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if value != value or math.isinf(value):
                raise BadBar(f"{name} is not finite: {value!r}")
            if value <= 0.0:
                raise BadBar(f"{name} must be positive, got {value!r}")
        if self.high < self.low:
            raise BadBar(f"high {self.high} is below low {self.low}")
        if self.high < self.open or self.high < self.close:
            raise BadBar(
                f"high {self.high} is below the open {self.open} or the close {self.close}"
            )
        if self.low > self.open or self.low > self.close:
            raise BadBar(
                f"low {self.low} is above the open {self.open} or the close {self.close}"
            )

    @property
    def log_range(self) -> float:
        """``log(high / low)``, the quantity Parkinson's estimator is built on."""
        return math.log(self.high / self.low)

    @property
    def log_close_to_open(self) -> float:
        """``log(close / open)``: the move within the period."""
        return math.log(self.close / self.open)

    @property
    def log_high_to_open(self) -> float:
        return math.log(self.high / self.open)

    @property
    def log_low_to_open(self) -> float:
        return math.log(self.low / self.open)


@dataclass(frozen=True, slots=True)
class Bars:
    """A named sequence of bars, in time order."""

    name: str
    bars: tuple[Bar, ...]

    def __post_init__(self) -> None:
        if not self.bars:
            raise BadBar(f"{self.name}: no bars")

    def __len__(self) -> int:
        return len(self.bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self.bars)

    @classmethod
    def from_rows(cls, name: str, rows: Iterable[Sequence[float]]) -> Bars:
        """Build from ``(open, high, low, close)`` rows.

        Args:
            name: What the series is called, used in error messages.
            rows: Four numbers per row.

        Returns:
            The bars.

        Raises:
            BadBar: If a row is the wrong length or is not a bar.
        """
        built: list[Bar] = []
        for index, row in enumerate(rows):
            if len(row) != 4:
                raise BadBar(f"{name}: row {index} has {len(row)} values, expected 4")
            try:
                built.append(Bar(*row))
            except BadBar as error:
                raise BadBar(f"{name}: row {index}: {error}") from error
        return cls(name=name, bars=tuple(built))

    @property
    def closes(self) -> tuple[float, ...]:
        return tuple(bar.close for bar in self.bars)

    def overnight(self) -> list[float]:
        """``log(open_i / close_{i-1})`` for every bar after the first.

        The gap, which is the part of the move no intraday statistic can see.
        """
        return [math.log(later.open / earlier.close) for earlier, later in pairwise(self.bars)]


class Estimator(str, Enum):
    """Which estimator to use."""

    CLOSE_TO_CLOSE = "close-to-close"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman-klass"
    ROGERS_SATCHELL = "rogers-satchell"
    GARMAN_KLASS_YANG_ZHANG = "garman-klass-yang-zhang"
    YANG_ZHANG = "yang-zhang"


def _require(bars: Bars, minimum: int, what: str) -> None:
    if len(bars) < minimum:
        raise BadBar(
            f"{bars.name}: {what} needs at least {minimum} bars, got {len(bars)}"
        )


def close_to_close(bars: Bars) -> float:
    """The sample variance of the close-to-close log return, per bar.

    The baseline. It assumes nothing about the drift or about gaps, because it
    estimates the mean from the sample and the gap is inside the return it
    measures. It also discards three quarters of each row.

    Args:
        bars: At least three bars, since two closes give one return.

    Returns:
        Variance per bar.

    Raises:
        BadBar: If there are too few bars.
    """
    _require(bars, 3, "close-to-close")
    closes = bars.closes
    returns = [math.log(later / earlier) for earlier, later in pairwise(closes)]
    mean = math.fsum(returns) / len(returns)
    return math.fsum((r - mean) ** 2 for r in returns) / (len(returns) - 1)


def parkinson(bars: Bars) -> float:
    """Parkinson's estimator: the squared log range, scaled.

    The expected squared range of a driftless Brownian motion over a unit of
    time is ``4 log 2`` times its variance, so dividing by that makes the
    estimator unbiased. It uses only the high and the low, which is why it is
    blind to a gap: an overnight jump moves the open, the high and the low
    together and leaves the range alone.

    Args:
        bars: At least one bar.

    Returns:
        Variance per bar.
    """
    _require(bars, 1, "Parkinson")
    total = math.fsum(bar.log_range ** 2 for bar in bars.bars)
    return total / (4.0 * _LOG_2 * len(bars))


def garman_klass(bars: Bars) -> float:
    """Garman and Klass's estimator: the range and the close together.

    The minimum-variance unbiased combination of the squared range and the
    squared open-to-close move, under a driftless Brownian motion with no gap.
    More efficient than Parkinson, and no more robust: it makes the same two
    assumptions.

    Args:
        bars: At least one bar.

    Returns:
        Variance per bar.
    """
    _require(bars, 1, "Garman-Klass")
    total = math.fsum(
        0.5 * bar.log_range ** 2 - _GK_CLOSE * bar.log_close_to_open ** 2
        for bar in bars.bars
    )
    return total / len(bars)


def rogers_satchell(bars: Bars) -> float:
    """Rogers and Satchell's estimator, which does not assume the drift is zero.

    The combination ``u(u - c) + d(d - c)`` has the property that its
    expectation is the variance whatever the drift, because the drift terms
    cancel between the high and the low legs. That is worth more than it
    sounds: Parkinson and Garman-Klass read a trending market as a volatile
    one, and a trend is exactly what a risk model is usually looking at.

    It still assumes no gap, since every term is measured from the open.

    Args:
        bars: At least one bar.

    Returns:
        Variance per bar.
    """
    _require(bars, 1, "Rogers-Satchell")
    total = 0.0
    for bar in bars.bars:
        u = bar.log_high_to_open
        d = bar.log_low_to_open
        c = bar.log_close_to_open
        total += u * (u - c) + d * (d - c)
    return total / len(bars)


def garman_klass_yang_zhang(bars: Bars) -> float:
    """Garman-Klass with Yang and Zhang's open-jump term added.

    The squared overnight move is added to the intraday estimate, which makes
    the estimator see the gap. It still assumes the drift is zero within the
    bar.

    Args:
        bars: At least two bars, since the first has no previous close.

    Returns:
        Variance per bar.
    """
    _require(bars, 2, "Garman-Klass-Yang-Zhang")
    gaps = bars.overnight()
    total = math.fsum(
        gap ** 2 + 0.5 * bar.log_range ** 2 - _GK_CLOSE * bar.log_close_to_open ** 2
        for gap, bar in zip(gaps, bars.bars[1:], strict=True)
    )
    return total / len(gaps)


def yang_zhang(bars: Bars) -> float:
    """Yang and Zhang's estimator: drift-independent and gap-aware.

    Three pieces: the variance of the overnight gap, the variance of the
    open-to-close move, and Rogers-Satchell for the path in between, combined
    with a weight chosen to minimise the total variance. The weight depends on
    the sample size and tends to about 0.2 for a long one.

    It is the most efficient of the estimators here on data that actually has
    a drift and a gap, which is what daily equity data has.

    Args:
        bars: At least four bars, since two of the three pieces are sample
            variances over ``n - 1`` gaps.

    Returns:
        Variance per bar.
    """
    _require(bars, 4, "Yang-Zhang")
    gaps = bars.overnight()
    inside = [bar.log_close_to_open for bar in bars.bars[1:]]
    n = len(gaps)

    gap_mean = math.fsum(gaps) / n
    gap_variance = math.fsum((g - gap_mean) ** 2 for g in gaps) / (n - 1)
    inside_mean = math.fsum(inside) / n
    inside_variance = math.fsum((c - inside_mean) ** 2 for c in inside) / (n - 1)

    rest = rogers_satchell(Bars(bars.name, bars.bars[1:]))
    weight = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    return gap_variance + weight * inside_variance + (1.0 - weight) * rest


_ESTIMATORS = {
    Estimator.CLOSE_TO_CLOSE: close_to_close,
    Estimator.PARKINSON: parkinson,
    Estimator.GARMAN_KLASS: garman_klass,
    Estimator.ROGERS_SATCHELL: rogers_satchell,
    Estimator.GARMAN_KLASS_YANG_ZHANG: garman_klass_yang_zhang,
    Estimator.YANG_ZHANG: yang_zhang,
}


def realised_variance(bars: Bars, estimator: Estimator = Estimator.YANG_ZHANG) -> float:
    """Variance per bar, by the named estimator.

    Args:
        bars: The bars.
        estimator: Which estimator.

    Returns:
        Variance per bar. Never negative, and not because it is clamped.

        Garman-Klass subtracts a multiple of the squared open-to-close move
        from half the squared range, which looks as though it could go below
        zero on a bar that closed at its own extreme. It cannot: the high is
        at least the larger of the open and the close and the low is at most
        the smaller, so the log range is at least the absolute log
        open-to-close move, and ``0.5 r^2`` therefore dominates
        ``(2 log 2 - 1) c^2 = 0.386 c^2`` term by term. Rogers-Satchell is a
        sum of two products of same-signed factors, and Yang-Zhang is a
        positive combination of two sample variances and a Rogers-Satchell.
        A clamp here would be unreachable code standing in for a proof, so
        there is a proof, and a test that samples bars looking for a
        counterexample.
    """
    return _ESTIMATORS[estimator](bars)


def realised_volatility(
    bars: Bars,
    periods_per_year: float = 252.0,
    estimator: Estimator = Estimator.YANG_ZHANG,
) -> float:
    """Annualised volatility, by the named estimator.

    Args:
        bars: The bars.
        periods_per_year: Bars in a year.
        estimator: Which estimator.

    Returns:
        Annualised volatility.

    Raises:
        ValueError: If ``periods_per_year`` is not positive.
    """
    if not (periods_per_year > 0.0):
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    return math.sqrt(realised_variance(bars, estimator) * periods_per_year)


def tick_bias_factor(ticks_per_bar: int) -> float:
    """How much a range-based variance falls short, at ``m`` ticks a bar.

    The extreme of a random walk observed at ``m`` points sits inside the
    extreme of the Brownian motion it samples, by an expected
    ``beta * sigma / sqrt(m)`` where ``beta`` is the overshoot constant
    ``-zeta(1/2) / sqrt(2 pi) = 0.5826`` that also appears in the continuity
    correction for a discretely monitored barrier. Both ends shrink, so the
    expected range loses twice that.

    Turning it into a factor needs the range's own scale, and that is the step
    a first attempt at this got wrong: the shortfall has to be divided by the
    expected range of the continuous motion, ``2 sqrt(2 / pi) sigma``, not by
    the volatility. The relative shrink is therefore

    ``2 * 0.5826 / (2 sqrt(2 / pi) sqrt(m)) = 0.7302 / sqrt(m)``

    and a *variance* built on the range is biased by the square of it. Without
    the normalisation the correction is out by a factor of 1.6 and predicts a
    26% bias at twenty ticks where the measured one is 14%.

    Args:
        ticks_per_bar: Price observations within the bar. At least two.

    The expansion is asymptotic in ``m``, so the correction is worth what it
    is measured to be worth and no more: against simulation it recovers the
    bias to within four per cent of itself at sixteen ticks a bar and a
    quarter of one per cent at two hundred and fifty-six. It stays positive
    for every ``m`` at or above two, so there is no degenerate case to guard
    against — at two ticks it claims a 77% shortfall, which is arithmetic
    rather than a prediction, and is why the accuracy above is stated as a
    range of ``m`` rather than as a property.

    Args:
        ticks_per_bar: Price observations within the bar. At least two.

    Returns:
        The factor by which a range-based variance is expected to be too small.
        Multiply an estimate by its reciprocal to correct it.

    Raises:
        ValueError: If fewer than two ticks.
    """
    if ticks_per_bar < 2:
        raise ValueError(f"ticks_per_bar must be at least 2, got {ticks_per_bar}")
    shrink = 1.0 - _RANGE_SHORTFALL / math.sqrt(ticks_per_bar)
    return shrink * shrink


def efficiency(
    samples: Sequence[Bars],
    estimator: Estimator,
    baseline: Estimator = Estimator.CLOSE_TO_CLOSE,
) -> float:
    """How many baseline samples one of these is worth, measured.

    The ratio of the baseline's sampling variance to this estimator's, computed
    across independent samples the caller supplies. That is the definition the
    textbook figures are quoted under, and measuring it on the caller's own
    sample size and data is the only way to find out whether those figures
    survive contact with it — they usually do not, because they assume the
    extremes are continuous.

    Args:
        samples: Independent samples of bars, at least two.
        estimator: The estimator being measured.
        baseline: What to measure it against.

    Returns:
        The variance ratio. Above one means more efficient than the baseline.

    Raises:
        ValueError: If fewer than two samples are given, or the baseline's
            estimates do not vary at all.
    """
    if len(samples) < 2:
        raise ValueError(f"need at least 2 samples to measure a variance, got {len(samples)}")

    def spread(which: Estimator) -> float:
        values = [realised_variance(sample, which) for sample in samples]
        mean = math.fsum(values) / len(values)
        return math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)

    reference = spread(baseline)
    measured = spread(estimator)
    if measured <= 0.0:
        raise ValueError(f"{estimator.value} did not vary across the samples")
    if reference <= 0.0:
        raise ValueError(f"the baseline {baseline.value} did not vary across the samples")
    return reference / measured

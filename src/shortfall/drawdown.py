"""Path statistics: drawdown, time under water, and the ratios built on them.

Everything else in this library treats returns as a bag of observations whose
order does not matter. Shuffle the sample and the volatility, the value at risk
and the expected shortfall are all unchanged. Drawdown is the statistic that
notices the shuffle, and it is the one an investor actually experiences: nobody
redeems because the ninety-ninth percentile of the return distribution is
unattractive, and everybody redeems after being down thirty per cent for two
years.

**Compounded, not summed.** Drawdown is computed on the wealth curve, the
running product of ``1 + r``. Running a cumulative *sum* of simple returns is
the common shortcut and it is a different and smaller number — by about half the
accumulated variance, in the same direction every time, so it systematically
understates exactly the statistic it is quoted to make vivid.

**The wealth curve has one more point than there are returns.** ``n`` returns
move a portfolio between ``n + 1`` valuations, and the first of them matters: a
portfolio that falls in its first period is in drawdown from the start, and a
curve that begins after the first return cannot see it. So an index supplied for
labelling has ``n + 1`` entries — the dates the portfolio was worth something,
not the dates it moved.

**A drawdown that has not ended is not a drawdown of zero length.** The last
one in any sample is usually still open, and its recovery date is unknown rather
than absent. It is reported as censored, and the ratios that would otherwise
average it in say so.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .series import Convention, ReturnSeries, TooShort, _check_periods


def wealth_curve(
    returns: Sequence[float] | ReturnSeries, *, initial: float = 1.0
) -> list[float]:
    """The compounded value of one unit through the return series.

    ``n`` returns give ``n + 1`` values, the first being ``initial`` before
    anything has happened.
    """
    if initial <= 0.0:
        raise ValueError(f"a starting value is positive, got {initial!r}")
    values = _values(returns)
    curve = [initial]
    for value in values:
        curve.append(curve[-1] * (1.0 + value))
    return curve


def _values(returns: Sequence[float] | ReturnSeries) -> tuple[float, ...]:
    """Simple returns from whatever was passed.

    A ``ReturnSeries`` carrying log returns is converted rather than used as is.
    Compounding log returns with ``1 + r`` is wrong by about half the variance
    and looks entirely plausible, so the conversion happens here rather than
    being left to the caller to remember.
    """
    if isinstance(returns, ReturnSeries):
        simple = returns.to_simple() if returns.convention is Convention.LOG else returns
        values = simple.values
    else:
        values = tuple(returns)
    if not values:
        raise TooShort("a path statistic needs at least one return")
    return values


def _index_for(index: Sequence[Any] | None, points: int) -> tuple[Any, ...]:
    if index is None:
        return tuple(range(points))
    if len(index) != points:
        raise ValueError(
            f"{len(index)} index entries against {points} points on the wealth "
            "curve. The curve has one more point than there are returns, because "
            "n returns move a portfolio between n + 1 valuations; the index labels "
            "the valuations."
        )
    return tuple(index)


def drawdown_series(
    returns: Sequence[float] | ReturnSeries, *, initial: float = 1.0
) -> list[float]:
    """Fractional distance below the running peak, at every point of the curve.

    Non-negative, zero at every new high, and ``0.2`` means twenty per cent
    below the best value reached so far. Same length as the wealth curve, so one
    longer than the returns.
    """
    curve = wealth_curve(returns, initial=initial)
    peak = curve[0]
    out = []
    for value in curve:
        peak = max(peak, value)
        out.append(0.0 if peak <= 0.0 else (peak - value) / peak)
    return out


@dataclass(frozen=True)
class Drawdown:
    """One peak-to-trough decline, and what happened after it."""

    #: Fractional decline from the peak, positive. 0.2 is a fall of a fifth.
    depth: float
    #: Positions on the wealth curve, not in the return series.
    peak_position: int
    trough_position: int
    #: Where the curve first regained the peak, or ``None`` if it never did
    #: within the sample.
    recovery_position: int | None
    peak_label: Any
    trough_label: Any
    recovery_label: Any
    peak_value: float
    trough_value: float

    @property
    def recovered(self) -> bool:
        return self.recovery_position is not None

    @property
    def decline_periods(self) -> int:
        """Periods from the peak to the trough."""
        return self.trough_position - self.peak_position

    @property
    def recovery_periods(self) -> int | None:
        """Periods from the trough back to the peak, or ``None`` if still under.

        ``None`` rather than zero, and rather than the number of periods to the
        end of the sample. Both of those are numbers, and averaging them in with
        the recoveries that did happen is how a strategy that has never
        recovered from anything comes to report a short average recovery.
        """
        if self.recovery_position is None:
            return None
        return self.recovery_position - self.trough_position

    @property
    def underwater_periods(self) -> int:
        """Periods spent below the peak, from the peak to recovery.

        For an unrecovered drawdown this is the time under water *so far*, which
        is a lower bound on the eventual figure. :attr:`recovered` says which
        kind of number this is.
        """
        end = (
            self.recovery_position
            if self.recovery_position is not None
            else self.trough_position
        )
        return end - self.peak_position

    @property
    def recovery_return(self) -> float:
        """The gain needed from the trough to get back to the peak.

        Always larger than the depth, and disproportionately so as the depth
        grows: a fall of a half needs a gain of a whole to undo. The asymmetry
        is the reason drawdown is worth a statistic of its own rather than being
        read off the return distribution.
        """
        if self.depth >= 1.0:
            return math.inf
        return self.depth / (1.0 - self.depth)


def maximum_drawdown(
    returns: Sequence[float] | ReturnSeries,
    *,
    index: Sequence[Any] | None = None,
    initial: float = 1.0,
) -> Drawdown:
    """The deepest peak-to-trough decline in the path.

    A single pass. The trap in the obvious two-loop version is not speed but
    correctness: the peak has to be the running maximum *before* the trough, and
    a search that takes the highest and lowest points of the whole path will
    happily report a trough that came first, which is a number no portfolio ever
    experienced.
    """
    curve = wealth_curve(returns, initial=initial)
    labels = _index_for(index, len(curve))

    peak_value = curve[0]
    peak_position = 0
    best_depth = 0.0
    best_peak = 0
    best_trough = 0
    for position, value in enumerate(curve):
        if value > peak_value:
            peak_value = value
            peak_position = position
        depth = 0.0 if peak_value <= 0.0 else (peak_value - value) / peak_value
        if depth > best_depth:
            best_depth = depth
            best_peak = peak_position
            best_trough = position

    recovery = _recovery_after(curve, best_peak, best_trough)
    return Drawdown(
        depth=best_depth,
        peak_position=best_peak,
        trough_position=best_trough,
        recovery_position=recovery,
        peak_label=labels[best_peak],
        trough_label=labels[best_trough],
        recovery_label=labels[recovery] if recovery is not None else None,
        peak_value=curve[best_peak],
        trough_value=curve[best_trough],
    )


def _recovery_after(curve: Sequence[float], peak: int, trough: int) -> int | None:
    target = curve[peak]
    for position in range(trough + 1, len(curve)):
        if curve[position] >= target:
            return position
    return None


def drawdowns(
    returns: Sequence[float] | ReturnSeries,
    *,
    index: Sequence[Any] | None = None,
    initial: float = 1.0,
    minimum_depth: float = 0.0,
) -> list[Drawdown]:
    """Every distinct drawdown episode, deepest first.

    An episode runs from a peak, through its trough, to the point the curve
    regains that peak — and the next episode cannot begin until the previous one
    has ended. That is what makes these separate declines rather than a list of
    overlapping windows: without the rule, the deepest drawdown and the second
    deepest are usually the same fall measured from two nearby peaks.

    The final episode is often still open, with ``recovered`` false.
    """
    curve = wealth_curve(returns, initial=initial)
    labels = _index_for(index, len(curve))
    if minimum_depth < 0.0:
        raise ValueError(f"a minimum depth is non-negative, got {minimum_depth!r}")

    episodes: list[Drawdown] = []
    position = 0
    while position < len(curve) - 1:
        peak_value = curve[position]
        peak_position = position
        # Walk forward while the curve is making new highs; the episode starts
        # at the last high before a fall.
        while position + 1 < len(curve) and curve[position + 1] >= peak_value:
            position += 1
            peak_value = curve[position]
            peak_position = position
        if position + 1 >= len(curve):
            break
        trough_value = peak_value
        trough_position = position
        scan = position + 1
        recovery: int | None = None
        while scan < len(curve):
            if curve[scan] < trough_value:
                trough_value = curve[scan]
                trough_position = scan
            if curve[scan] >= peak_value:
                recovery = scan
                break
            scan += 1
        depth = 0.0 if peak_value <= 0.0 else (peak_value - trough_value) / peak_value
        if depth >= minimum_depth and depth > 0.0:
            episodes.append(
                Drawdown(
                    depth=depth,
                    peak_position=peak_position,
                    trough_position=trough_position,
                    recovery_position=recovery,
                    peak_label=labels[peak_position],
                    trough_label=labels[trough_position],
                    recovery_label=labels[recovery] if recovery is not None else None,
                    peak_value=peak_value,
                    trough_value=trough_value,
                )
            )
        if recovery is None:
            break
        position = recovery
    episodes.sort(key=lambda one: one.depth, reverse=True)
    return episodes


def longest_underwater(
    returns: Sequence[float] | ReturnSeries,
    *,
    index: Sequence[Any] | None = None,
    initial: float = 1.0,
) -> Drawdown:
    """The drawdown that kept the portfolio below its peak for longest.

    Not usually the deepest one, and often the one that matters more. A fund
    that falls thirty per cent and recovers in four months has had a bad
    quarter; a fund that falls twelve per cent and takes five years to get back
    has had a bad decade, and only the second loses its investors.
    """
    episodes = drawdowns(returns, index=index, initial=initial)
    if not episodes:
        return maximum_drawdown(returns, index=index, initial=initial)
    return max(episodes, key=lambda one: (one.underwater_periods, one.depth))


def ulcer_index(
    returns: Sequence[float] | ReturnSeries, *, initial: float = 1.0
) -> float:
    """Root mean square drawdown across the whole path.

    Denominator: the number of points on the wealth curve, which is one more
    than the number of returns. Every point counts, including the ones at a new
    high contributing zero — that is the definition, and restricting the average
    to the underwater points instead measures the depth of drawdowns while
    ignoring how much of the time the portfolio spent in them, which is the
    other half of what this is for.

    Unlike maximum drawdown it uses the whole path rather than its two most
    extreme points, so one unrepeatable crash does not define it and a long
    grinding decline is not flattered by never having a dramatic day.
    """
    series = drawdown_series(returns, initial=initial)
    return math.sqrt(math.fsum(value * value for value in series) / len(series))


def calmar(
    returns: Sequence[float] | ReturnSeries,
    periods_per_year: float,
    *,
    initial: float = 1.0,
) -> float:
    """Annualised return over maximum drawdown.

    Numerator: the *geometric* annualised return, which is the one an investor
    earned. Denominator: the maximum drawdown as a positive fraction.

    The denominator is a single realised extreme, so this ratio is less stable
    than it looks — it moves discontinuously when a new worst drawdown arrives,
    and it improves with no change in behaviour simply by the sample growing
    away from an old crash. :func:`ulcer_index` in the denominator, which is the
    Martin ratio, uses the whole path and does not have that property.
    """
    _check_periods(periods_per_year)
    values = _values(returns)
    series = ReturnSeries(name="path", values=values)
    worst = maximum_drawdown(values, initial=initial)
    if worst.depth <= 0.0:
        raise ValueError(
            "this path never fell below a previous high, so its maximum drawdown is "
            "zero and there is no ratio to take. That is a statement about the "
            "sample being short or the strategy being untested, not about the risk "
            "being absent."
        )
    return series.annualised_return(periods_per_year) / worst.depth


def martin_ratio(
    returns: Sequence[float] | ReturnSeries,
    periods_per_year: float,
    *,
    initial: float = 1.0,
) -> float:
    """Annualised return over the ulcer index."""
    _check_periods(periods_per_year)
    values = _values(returns)
    index = ulcer_index(values, initial=initial)
    if index <= 0.0:
        raise ValueError(
            "this path never fell below a previous high, so its ulcer index is zero "
            "and there is no ratio to take"
        )
    return ReturnSeries(name="path", values=values).annualised_return(
        periods_per_year
    ) / index


def downside_deviation(
    returns: Sequence[float] | ReturnSeries,
    *,
    target: float = 0.0,
    full_sample: bool = True,
) -> float:
    """Root mean square shortfall below ``target``.

    **The denominator is the argument.** Sortino's definition divides the sum of
    squared shortfalls by the count of *all* observations, so periods above the
    target enter as zeros and pull the figure down. The widespread variant
    divides by the count of the downside observations only, which is a different
    statistic: it measures how bad the bad periods were, while the original
    measures how much downside the strategy produced per period.

    They are not close. A strategy that is above target nine periods in ten has
    a full-sample deviation about ``sqrt(10)`` times smaller than the downside-
    only one, so its Sortino ratio is about three times larger — and the two are
    routinely quoted under the same name with no indication of which was used.

    ``full_sample`` defaults to Sortino's own definition, and the other is
    available by asking for it rather than by accident.
    """
    values = _values(returns)
    shortfalls = [min(value - target, 0.0) for value in values]
    squared = math.fsum(value * value for value in shortfalls)
    count = len(values) if full_sample else sum(1 for v in shortfalls if v < 0.0)
    if count == 0:
        raise ValueError(
            f"no observation fell below the target of {target!r}, so a downside-only "
            "deviation has nothing to average over. Pass full_sample=True for the "
            "definition where that case is simply a deviation of zero."
        )
    return math.sqrt(squared / count)


def sortino(
    returns: Sequence[float] | ReturnSeries,
    periods_per_year: float,
    *,
    target: float = 0.0,
    full_sample: bool = True,
) -> float:
    """Excess return over downside deviation, both annualised.

    Numerator: the arithmetic mean return less ``target``, scaled by the number
    of periods in a year. Arithmetic rather than geometric here, because the
    denominator is a per-period second moment and the ratio is only coherent if
    both sides are per-period quantities scaled the same way.

    Denominator: :func:`downside_deviation`, scaled by the square root of the
    periods, with the choice of denominator reported by ``full_sample``.

    ``target`` is the minimum acceptable return *per period*, not annualised.
    """
    _check_periods(periods_per_year)
    values = _values(returns)
    deviation = downside_deviation(values, target=target, full_sample=full_sample)
    if deviation <= 0.0:
        raise ValueError(
            f"no observation fell below the target of {target!r}, so the downside "
            "deviation is zero and the ratio is unbounded. A sample with no downside "
            "has not measured any."
        )
    excess = math.fsum(values) / len(values) - target
    return (excess * periods_per_year) / (deviation * math.sqrt(periods_per_year))


def rolling(
    returns: Sequence[float] | ReturnSeries,
    window: int,
    statistic: Callable[[Sequence[float]], float],
) -> list[float]:
    """Apply ``statistic`` to every consecutive window of ``window`` returns.

    Returns ``len(returns) - window + 1`` values, the first covering the first
    window. Nothing is padded to the original length: a rolling statistic does
    not exist before its window is full, and filling the gap with the first
    available value is how a backtest comes to use information it did not have.
    """
    values = _values(returns)
    if window < 1:
        raise ValueError(f"a window is at least one period, got {window!r}")
    if window > len(values):
        raise TooShort(
            f"a window of {window} does not fit in {len(values)} observations"
        )
    return [
        statistic(values[start : start + window])
        for start in range(len(values) - window + 1)
    ]


def rolling_maximum_drawdown(
    returns: Sequence[float] | ReturnSeries, window: int
) -> list[float]:
    """Maximum drawdown within each window, as a positive fraction.

    Each window is treated as its own path starting from a fresh peak, so a
    drawdown that began before the window does not carry into it. That is the
    right reading of "the worst fall over any six months" and it is not the same
    as slicing the full drawdown series, which would report the distance below a
    peak the window never saw.
    """
    return rolling(returns, window, lambda block: maximum_drawdown(block).depth)


def rolling_ulcer_index(
    returns: Sequence[float] | ReturnSeries, window: int
) -> list[float]:
    """Ulcer index within each window."""
    return rolling(returns, window, ulcer_index)


def rolling_volatility(
    returns: Sequence[float] | ReturnSeries,
    window: int,
    periods_per_year: float,
    *,
    ddof: int = 1,
) -> list[float]:
    """Annualised volatility within each window."""
    _check_periods(periods_per_year)

    def volatility(block: Sequence[float]) -> float:
        return ReturnSeries(name="window", values=tuple(block)).annualised_volatility(
            periods_per_year, ddof=ddof
        )

    return rolling(returns, window, volatility)


__all__ = [
    "Drawdown",
    "calmar",
    "downside_deviation",
    "drawdown_series",
    "drawdowns",
    "longest_underwater",
    "martin_ratio",
    "maximum_drawdown",
    "rolling",
    "rolling_maximum_drawdown",
    "rolling_ulcer_index",
    "rolling_volatility",
    "sortino",
    "ulcer_index",
    "wealth_curve",
]

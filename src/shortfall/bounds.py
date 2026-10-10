"""Value at risk when nothing is assumed about the dependence.

Every other multi-asset route in this package names the dependence. A
covariance matrix names it, a fitted copula names it, a historical panel names
it by having lived it. The number that comes out is then conditional on that
naming, and the usual way of acknowledging that is a sentence in a footnote.

There is a sharper question. Given the marginal loss distributions and
**nothing else** — no correlation, no copula, no panel — what is the set of
values ``VaR_alpha`` of the sum can take? That set is an interval, both of its
ends are computable, and its upper end is not where the intuition puts it.

**The worst case is not the comonotonic coupling.** Lining every loss up so
that they are all large together maximises expected shortfall, because expected
shortfall is comonotonic-additive and that is its maximum. Value at risk is a
quantile, not an average, and it is not subadditive: a coupling that makes the
tail sum *flat* rather than *extreme* can push the quantile above the
comonotonic one, by moving probability out of the very worst outcomes and into
the level just above the quantile. Measured here on Pareto marginals with tail
index 2 at the 99% level, the worst case runs 1.41 times the comonotonic value
at two positions, 1.73 at four, 1.86 at eight, 1.92 at sixteen and 1.93 at
thirty-two, against a limit of ``theta / (theta - 1) = 2`` as positions are
added. On exponential marginals it caps out at 1.217, which is that
distribution's own ratio of expected shortfall to value at risk. The size of
the effect is a statement about tail weight, and it is large enough that a
diversified book's value at risk can nearly double without any marginal
changing.

**Two bounds are inequalities and two are arrangements.** Reading them apart is
the whole design here.

The upper bound on the worst case is a proof: ``VaR_alpha(S) <= ES_alpha(S)``
for any coupling at all, and ``ES_alpha`` is subadditive, so the worst case is
at most the sum of the marginals' own expected shortfalls. Nothing numerical
enters. Symmetrically, the lower bound on the *best* case is the sum of the
marginals' lower tail means.

The other two ends come from the **rearrangement algorithm**: discretise each
marginal's tail onto ``N`` equiprobable points and repeatedly replace one
column with the arrangement that runs against the sum of the others, which
flattens the row sums. The minimum row sum of the result is the value at risk
of an explicit coupling, so it is attained and therefore a bound from below —
provided the discretisation *understates* each marginal, which the
left-endpoint grid does. The right-endpoint grid overstates each marginal, and
its rearrangement is **not** a bound from above, because the algorithm returns
an arrangement rather than the discrete optimum. It is reported, because the
distance between the two grids is the discretisation error and the honest place
for it is in the output.

**Which brackets exist depends on which endpoints are finite, and the two cases
are not symmetric for a loss.** The worst case's right-endpoint grid needs
``q(1)``, so it exists only for a loss bounded above — which a uniform loss is
and a Pareto, exponential or normal loss is not. The best case's left-endpoint
grid needs ``q(0)``, so it exists for any loss bounded below, which all three of
Pareto, exponential and uniform are and a normal is not. A normal loss gets
neither grid check and both proofs; a Pareto loss gets the best case's.

**The exact oracle is complete mixability.** A distribution is
``n``-completely mixable when ``n`` copies of it admit a coupling whose sum is
constant, and the uniform distribution is completely mixable for every ``n >=
2``. The conditional tail of a uniform loss above its ``alpha`` quantile is
again uniform, so for uniform marginals the flat arrangement is attainable and
the worst case **equals** the sum of the tail means exactly — the inequality
above is tight, with a closed form ``d (1 + alpha) / 2`` on the unit interval.
The lower tail is uniform too, so the best case is exactly ``d alpha / 2``.
Both are reproduced here to within one grid step, which is the only way to know
the rearrangement is finding the arrangement and not merely a plausible number.

Away from uniform the inequality is slack, and by how much is a measurement
rather than a guess. At the 99% level with 2000 tail points, exponential
marginals reach 94.5% of the sum of expected shortfalls at two positions,
99.58% at four and 99.96% at eight; Pareto marginals with tail index 2 reach
70.6%, 86.3%, 93.1% and 95.8% at two, four, eight and sixteen. The bound is
asymptotically sharp in the number of positions and badly slack at two, which
is the opposite of the way a bound is usually assumed to behave.

**The algorithm does not always settle, and the stopping rule is on the
quantity asked about.** For three and five uniform columns the sweeps cycle
without a fixed point, visiting the same objective value repeatedly; a rule
that waits for the columns to stop moving would run to the iteration cap and
report failure on a problem it had already solved to one grid step. The rule
here is patience on the objective: stop when the best minimum row sum seen has
not improved for a fixed number of sweeps, and return that best — which belongs
to an arrangement that was actually visited, so it stays attained.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .distributions import normal_pdf, normal_ppf
from .parametric import normal_tail_mean

#: Tail points per marginal when the caller does not choose. Large enough that
#: the uniform oracle is reproduced to 1e-4 at eight positions and small enough
#: that a five-position problem rearranges in well under a second.
DEFAULT_POINTS = 1024

#: Sweeps without an improvement in the objective before stopping. Three is
#: enough to get past the two-sweep cycles the uniform grids fall into.
DEFAULT_PATIENCE = 4

#: Hard cap on sweeps, so a cycling problem terminates.
MAX_SWEEPS = 512

#: Probabilities are kept this far inside ``(0, 1)`` when a quantile function is
#: probed for monotonicity, since many are infinite at the ends.
_PROBE_MARGIN = 1e-9

#: Points used to check a supplied quantile function increases.
_PROBE_POINTS = 64


class NotIncreasing(ValueError):
    """A supplied quantile function decreases somewhere on ``(0, 1)``.

    A quantile function is non-decreasing by definition, so a function that
    falls is not one, and every bound in this module would be computed from a
    marginal that does not exist.
    """


class NoTailMean(ValueError):
    """The marginal was built without the exact tail mean a bound needs.

    The proof-level bounds are sums of the marginals' tail means. Estimating
    those from the same discretisation the rearrangement uses would make the
    bound inherit the discretisation error it is supposed to be independent of,
    so a missing tail mean is refused rather than filled in.
    """


@dataclass(frozen=True)
class LossTail:
    """One position's loss distribution, as a quantile function.

    Losses are positive and large losses are at large ``u``, matching
    :attr:`shortfall.parametric.Risk.value_at_risk` rather than the signed
    return convention used elsewhere in the package. The quantile function is
    the only thing a dependence bound reads off a marginal; the tail means are
    carried alongside because the proof-level bounds need them exactly, and the
    endpoints because which grids exist depends on them.

    Build one with :func:`uniform_tail`, :func:`exponential_tail`,
    :func:`pareto_tail` or :func:`normal_tail` to get the exact means and
    endpoints filled in, or construct it directly from any quantile function and
    accept that the bounds requiring a tail mean will refuse.
    """

    #: ``u -> loss``, non-decreasing on ``(0, 1)``.
    quantile: Callable[[float], float]
    #: ``c -> E[L | L > q(c)]``. The upper bound on the worst case is the sum of
    #: these.
    upper_mean: Callable[[float], float] | None = None
    #: ``c -> E[L | L <= q(c)]``. The lower bound on the best case is the sum of
    #: these.
    lower_mean: Callable[[float], float] | None = None
    #: ``q(1)``. Infinite unless the loss is bounded above.
    upper_endpoint: float = math.inf
    #: ``q(0)``. Negative infinity unless the loss is bounded below.
    lower_endpoint: float = -math.inf
    label: str = ""

    def __post_init__(self) -> None:
        if not self.lower_endpoint <= self.upper_endpoint:
            raise ValueError(
                f"the support [{self.lower_endpoint!r}, {self.upper_endpoint!r}] of "
                f"{self.label or 'a loss'} is empty"
            )

    def value_at_risk(self, confidence: float) -> float:
        """The marginal's own value at risk, which is ``quantile(confidence)``."""
        _check_confidence(confidence)
        return self.quantile(confidence)

    def expected_shortfall(self, confidence: float) -> float:
        """The marginal's own expected shortfall, if it was built with one."""
        _check_confidence(confidence)
        if self.upper_mean is None:
            raise NoTailMean(
                f"{self.label or 'this marginal'} carries no upper tail mean, so the "
                "sum of expected shortfalls that bounds the worst case cannot be "
                "formed. Build it with one of the factories in this module, or pass "
                "upper_mean."
            )
        return self.upper_mean(confidence)

    def lower_tail_mean(self, confidence: float) -> float:
        """The mean of the marginal below its ``confidence`` quantile."""
        _check_confidence(confidence)
        if self.lower_mean is None:
            raise NoTailMean(
                f"{self.label or 'this marginal'} carries no lower tail mean, so the "
                "bound below the best case cannot be formed."
            )
        return self.lower_mean(confidence)

    def check_increases(self) -> None:
        """Probe the quantile function on a grid and refuse one that falls.

        Probed rather than proved, which cannot rule out a dip between adjacent
        points. It catches the mistake this is here for: a quantile function
        written for the signed-return convention, which decreases over the whole
        interval and would otherwise return a bound with the wrong sign.
        """
        previous = self.quantile(_PROBE_MARGIN)
        for index in range(1, _PROBE_POINTS + 1):
            u = _PROBE_MARGIN + (1.0 - 2.0 * _PROBE_MARGIN) * index / _PROBE_POINTS
            current = self.quantile(u)
            if current < previous:
                raise NotIncreasing(
                    f"the quantile function of {self.label or 'a loss'} falls from "
                    f"{previous!r} to {current!r} by u={u!r}. Losses are positive here "
                    "and large losses sit at large u, which is the opposite of the "
                    "signed-return convention the rest of the package uses."
                )
            previous = current


# -- ready-made marginals, with exact tail means -----------------------------


def uniform_tail(low: float = 0.0, high: float = 1.0, *, label: str = "") -> LossTail:
    """A uniform loss on ``[low, high]``, the one case with an exact oracle.

    Both of its conditional tails are uniform and the uniform distribution is
    completely mixable, so the worst and best cases are closed forms and the
    rearrangement has something to be checked against.
    """
    if not high > low:
        raise ValueError(f"a uniform loss needs high > low, got [{low!r}, {high!r}]")
    width = high - low
    return LossTail(
        quantile=lambda u: low + width * u,
        upper_mean=lambda c: low + width * (1.0 + c) / 2.0,
        lower_mean=lambda c: low + width * c / 2.0,
        upper_endpoint=high,
        lower_endpoint=low,
        label=label or f"uniform[{low:g}, {high:g}]",
    )


def exponential_tail(scale: float = 1.0, *, label: str = "") -> LossTail:
    """An exponential loss with mean ``scale``.

    Its expected shortfall exceeds its value at risk by exactly ``scale``,
    whatever the level, which is what caps the superadditivity of the sum.
    """
    if scale <= 0.0:
        raise ValueError(f"an exponential loss needs a positive scale, got {scale!r}")

    def lower_mean(c: float) -> float:
        if c <= 0.0:
            return 0.0
        return scale * (c + (1.0 - c) * math.log1p(-c)) / c

    return LossTail(
        quantile=lambda u: -scale * math.log1p(-u),
        upper_mean=lambda c: scale * (1.0 - math.log1p(-c)),
        lower_mean=lower_mean,
        upper_endpoint=math.inf,
        lower_endpoint=0.0,
        label=label or f"exponential(scale={scale:g})",
    )


def pareto_tail(shape: float, scale: float = 1.0, *, label: str = "") -> LossTail:
    """A Pareto loss on ``[scale, inf)`` with tail index ``shape``.

    The tail mean needs ``shape > 1`` to exist at all, and the ratio of the
    worst case to the comonotonic coupling tends to ``shape / (shape - 1)`` as
    positions are added, so a tail index approaching one is a book whose value
    at risk is unbounded relative to the sum of its parts.
    """
    if shape <= 1.0:
        raise ValueError(
            f"a Pareto loss needs shape > 1 for a finite tail mean, got {shape!r}"
        )
    if scale <= 0.0:
        raise ValueError(f"a Pareto loss needs a positive scale, got {scale!r}")
    mean = shape / (shape - 1.0)

    def lower_mean(c: float) -> float:
        if c <= 0.0:
            return scale
        return scale * mean * (1.0 - (1.0 - c) ** (1.0 - 1.0 / shape)) / c

    return LossTail(
        quantile=lambda u: scale * (1.0 - u) ** (-1.0 / shape),
        upper_mean=lambda c: scale * mean * (1.0 - c) ** (-1.0 / shape),
        lower_mean=lower_mean,
        upper_endpoint=math.inf,
        lower_endpoint=scale,
        label=label or f"pareto(shape={shape:g}, scale={scale:g})",
    )


def normal_tail(mean: float = 0.0, volatility: float = 1.0, *, label: str = "") -> LossTail:
    """A normal loss, the case with no grid check available at either end.

    ``mean`` and ``volatility`` describe the **loss**, so a position with a zero
    expected return and a 2% daily volatility is ``normal_tail(0.0, 0.02)``.
    Reuses :func:`shortfall.parametric.normal_tail_mean` for both ends, which is
    stated there as ``E[Z | Z <= a]`` and gives the upper tail by symmetry.
    """
    if volatility <= 0.0:
        raise ValueError(f"a normal loss needs a positive volatility, got {volatility!r}")
    return LossTail(
        quantile=lambda u: mean + volatility * normal_ppf(u),
        upper_mean=lambda c: mean - volatility * normal_tail_mean(normal_ppf(1.0 - c)),
        lower_mean=lambda c: mean + volatility * normal_tail_mean(normal_ppf(c)),
        upper_endpoint=math.inf,
        lower_endpoint=-math.inf,
        label=label or f"normal(mean={mean:g}, volatility={volatility:g})",
    )


# -- the rearrangement -------------------------------------------------------


@dataclass(frozen=True)
class Arrangement:
    """What the rearrangement did, separately from what it found."""

    points: int
    positions: int
    sweeps: int
    #: Whether the objective stopped improving before the sweep cap.
    settled: bool
    #: Best objective seen, which belongs to an arrangement that was visited.
    objective: float
    #: Row sums of the final arrangement.
    minimum: float
    maximum: float
    #: Mean row sum, which no rearrangement can change: it is the discretised
    #: sum of the marginals' tail means, and the objective's own ceiling.
    mean: float

    @property
    def spread(self) -> float:
        """``maximum - minimum`` of the final arrangement.

        Zero means the discretised tail mixed completely, which is the condition
        under which the inequality bounding the worst case is tight.
        """
        return self.maximum - self.minimum

    @property
    def flatness(self) -> float:
        """:attr:`spread` relative to the mean row sum."""
        if self.mean == 0.0:
            return 0.0
        return self.spread / abs(self.mean)


def rearrange(
    columns: Sequence[Sequence[float]],
    *,
    maximise_minimum: bool = True,
    patience: int = DEFAULT_PATIENCE,
    max_sweeps: int = MAX_SWEEPS,
) -> Arrangement:
    """Rearrange each column against the sum of the others, repeatedly.

    With ``maximise_minimum`` the objective is the smallest row sum and the
    algorithm is pushing it up, which is the worst-case problem. Without it the
    objective is the largest row sum and the algorithm is pushing it down, which
    is the best-case problem. The step is the same either way: a column
    arranged in the opposite order to the sum of the other columns.

    The columns are permuted independently, so every column keeps its own
    multiset of values and the mean row sum never moves. That invariant is what
    makes the mean a ceiling on the minimum and a floor on the maximum, and it
    is asserted in the tests rather than assumed.
    """
    if not columns:
        raise ValueError("a rearrangement needs at least one column")
    count = len(columns[0])
    if count == 0:
        raise ValueError("a rearrangement needs at least one row")
    for index, column in enumerate(columns):
        if len(column) != count:
            raise ValueError(
                f"column {index} has {len(column)} rows against {count} in the first"
            )
    if patience < 1:
        raise ValueError(f"patience is at least one sweep, got {patience!r}")
    if max_sweeps < 1:
        raise ValueError(f"the sweep cap is at least one, got {max_sweeps!r}")

    positions = len(columns)
    grid = [list(column) for column in columns]
    sums = [math.fsum(grid[j][k] for j in range(positions)) for k in range(count)]
    total = math.fsum(sums) / count

    best = min(sums) if maximise_minimum else max(sums)
    stale = 0
    sweeps = 0
    settled = False
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        for j in range(positions):
            column = grid[j]
            others = [sums[k] - column[k] for k in range(count)]
            order = [row for _, row in sorted((others[k], k) for k in range(count))]
            ordered = sorted(column, reverse=True)
            replacement = [0.0] * count
            for rank, row in enumerate(order):
                replacement[row] = ordered[rank]
            grid[j] = replacement
            sums = [others[k] + replacement[k] for k in range(count)]
        current = min(sums) if maximise_minimum else max(sums)
        improved = current > best if maximise_minimum else current < best
        if improved:
            best = current
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            settled = True
            break

    return Arrangement(
        points=count,
        positions=positions,
        sweeps=sweeps,
        settled=settled,
        objective=best,
        minimum=min(sums),
        maximum=max(sums),
        mean=total,
    )


# -- the bounds --------------------------------------------------------------


@dataclass(frozen=True)
class WorstCase:
    """The largest value at risk any coupling of the marginals can produce.

    :attr:`lower` and :attr:`upper` bracket it. The lower end is attained by an
    explicit coupling and the upper end is an inequality, so the bracket is a
    statement and not an estimate; :attr:`overstated` is the same rearrangement
    on a grid that overstates each marginal, reported to show the
    discretisation and not as a bound.
    """

    confidence: float
    #: Minimum row sum after rearranging the understating grid. Attained.
    lower: float
    #: Sum of the marginals' expected shortfalls. ``VaR <= ES`` and ``ES`` is
    #: subadditive, so no coupling exceeds this.
    upper: float
    #: The overstating grid's rearrangement, or ``None`` when some marginal has
    #: no finite upper endpoint.
    overstated: float | None
    #: Sum of the marginals' own values at risk: the comonotonic coupling, which
    #: is a reference point and not an end of the interval.
    comonotonic: float
    arrangement: Arrangement

    @property
    def gap(self) -> float:
        """How much of the inequality the rearrangement did not close."""
        return self.upper - self.lower

    @property
    def relative_gap(self) -> float:
        """:attr:`gap` as a fraction of the bound above."""
        if self.upper == 0.0:
            return 0.0
        return self.gap / abs(self.upper)

    @property
    def superadditivity(self) -> float:
        """The worst case over the comonotonic coupling.

        Above one whenever value at risk fails to be subadditive on these
        marginals, which is almost always. Computed from :attr:`lower`, so it is
        itself a bound from below on the real ratio.
        """
        if self.comonotonic == 0.0:
            return math.inf
        return self.lower / self.comonotonic

    def mixes(self, tolerance: float = 1e-9) -> bool:
        """Whether the inequality is tight to ``tolerance``, relatively.

        True exactly when the discretised tail mixed completely, which for a
        uniform marginal it does at every number of positions and for a Pareto
        or exponential marginal it does not at any.
        """
        return self.relative_gap <= tolerance


@dataclass(frozen=True)
class BestCase:
    """The smallest value at risk any coupling of the marginals can produce.

    The two ends swap roles against :class:`WorstCase`: here the *upper* end is
    attained by an explicit coupling, from the grid that overstates each
    marginal, and the lower end is the inequality.
    """

    confidence: float
    #: Sum of the marginals' lower tail means. No coupling goes below this.
    lower: float
    #: Maximum row sum after rearranging the overstating grid. Attained.
    upper: float
    #: The understating grid's rearrangement, or ``None`` when some marginal has
    #: no finite lower endpoint. Reported, not a bound.
    understated: float | None
    comonotonic: float
    arrangement: Arrangement

    @property
    def gap(self) -> float:
        return self.upper - self.lower

    @property
    def relative_gap(self) -> float:
        if self.lower == 0.0:
            return 0.0
        return self.gap / abs(self.lower)

    @property
    def subadditivity(self) -> float:
        """The best case over the comonotonic coupling, which is below one."""
        if self.comonotonic == 0.0:
            return math.inf
        return self.upper / self.comonotonic


@dataclass(frozen=True)
class Bounds:
    """The whole interval, and the two readings of how wide it is."""

    confidence: float
    worst: WorstCase
    best: BestCase

    @property
    def comonotonic(self) -> float:
        return self.worst.comonotonic

    @property
    def attainable(self) -> tuple[float, float]:
        """The interval both of whose ends a coupling was found for."""
        return (self.best.upper, self.worst.lower)

    @property
    def enclosing(self) -> tuple[float, float]:
        """The interval the proofs put the whole range inside."""
        return (self.best.lower, self.worst.upper)

    @property
    def ratio(self) -> float:
        """Widest over narrowest value at risk, across all couplings.

        Read off :attr:`attainable`, so it understates the real spread.
        """
        low, high = self.attainable
        if low == 0.0:
            return math.inf
        return high / low


def _check_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence is strictly inside (0, 1), got {confidence!r}. It is the "
            "confidence level, so 0.99 means the 1% tail."
        )
    return 1.0 - confidence


def _prepare(tails: Sequence[LossTail], confidence: float) -> None:
    if not tails:
        raise ValueError("a dependence bound needs at least one marginal")
    _check_confidence(confidence)
    for tail in tails:
        tail.check_increases()


def _tail_grid(tail: LossTail, confidence: float, points: int, *, offset: int) -> list[float]:
    """The upper tail on ``points`` equiprobable cells, at one cell edge.

    ``offset`` of zero takes the left edge of each cell, which understates the
    cell's loss; one takes the right edge, which overstates it. The right edge
    of the last cell is ``q(1)``.
    """
    width = (1.0 - confidence) / points
    return [
        tail.quantile(min(1.0, confidence + (index + offset) * width))
        for index in range(points)
    ]


def _lower_grid(tail: LossTail, confidence: float, points: int, *, offset: int) -> list[float]:
    """The same for the part below the quantile; the left edge of the first cell
    is ``q(0)``."""
    width = confidence / points
    return [
        tail.quantile(min(confidence, (index + offset) * width)) for index in range(points)
    ]


def worst_case_value_at_risk(
    tails: Sequence[LossTail],
    *,
    confidence: float = 0.99,
    points: int = DEFAULT_POINTS,
    patience: int = DEFAULT_PATIENCE,
    max_sweeps: int = MAX_SWEEPS,
) -> WorstCase:
    """The largest value at risk the marginals admit, bracketed.

    The lower end rearranges the left-endpoint grid, whose marginals are
    dominated by the real ones, so the coupling it finds is realisable against
    the real marginals at a value at least as large: a bound from below. The
    upper end is the sum of the marginals' expected shortfalls, which requires
    every marginal to carry one.
    """
    _prepare(tails, confidence)
    if points < 2:
        raise ValueError(f"a tail grid needs at least two points, got {points!r}")

    understating = [_tail_grid(tail, confidence, points, offset=0) for tail in tails]
    arrangement = rearrange(
        understating, maximise_minimum=True, patience=patience, max_sweeps=max_sweeps
    )

    overstated: float | None = None
    if all(math.isfinite(tail.upper_endpoint) for tail in tails):
        overstating = [_tail_grid(tail, confidence, points, offset=1) for tail in tails]
        overstated = rearrange(
            overstating, maximise_minimum=True, patience=patience, max_sweeps=max_sweeps
        ).objective

    return WorstCase(
        confidence=confidence,
        lower=arrangement.objective,
        upper=math.fsum(tail.expected_shortfall(confidence) for tail in tails),
        overstated=overstated,
        comonotonic=math.fsum(tail.value_at_risk(confidence) for tail in tails),
        arrangement=arrangement,
    )


def best_case_value_at_risk(
    tails: Sequence[LossTail],
    *,
    confidence: float = 0.99,
    points: int = DEFAULT_POINTS,
    patience: int = DEFAULT_PATIENCE,
    max_sweeps: int = MAX_SWEEPS,
) -> BestCase:
    """The smallest value at risk the marginals admit, bracketed.

    Mirrors :func:`worst_case_value_at_risk` with the roles of the two grids
    exchanged. The attained end is the overstating grid's, so it needs every
    marginal's ``q`` at the cell edges above zero only — which always exist —
    while the reported-not-bounded end needs ``q(0)`` and so is unavailable for
    a loss that is unbounded below.
    """
    _prepare(tails, confidence)
    if points < 2:
        raise ValueError(f"a tail grid needs at least two points, got {points!r}")

    overstating = [_lower_grid(tail, confidence, points, offset=1) for tail in tails]
    arrangement = rearrange(
        overstating, maximise_minimum=False, patience=patience, max_sweeps=max_sweeps
    )

    understated: float | None = None
    if all(math.isfinite(tail.lower_endpoint) for tail in tails):
        understating = [_lower_grid(tail, confidence, points, offset=0) for tail in tails]
        understated = rearrange(
            understating, maximise_minimum=False, patience=patience, max_sweeps=max_sweeps
        ).objective

    return BestCase(
        confidence=confidence,
        lower=math.fsum(tail.lower_tail_mean(confidence) for tail in tails),
        upper=arrangement.objective,
        understated=understated,
        comonotonic=math.fsum(tail.value_at_risk(confidence) for tail in tails),
        arrangement=arrangement,
    )


def dependence_bounds(
    tails: Sequence[LossTail],
    *,
    confidence: float = 0.99,
    points: int = DEFAULT_POINTS,
    patience: int = DEFAULT_PATIENCE,
    max_sweeps: int = MAX_SWEEPS,
) -> Bounds:
    """Both ends of the interval, from one call."""
    return Bounds(
        confidence=confidence,
        worst=worst_case_value_at_risk(
            tails,
            confidence=confidence,
            points=points,
            patience=patience,
            max_sweeps=max_sweeps,
        ),
        best=best_case_value_at_risk(
            tails,
            confidence=confidence,
            points=points,
            patience=patience,
            max_sweeps=max_sweeps,
        ),
    )


def completely_mixable_level(tails: Sequence[LossTail], *, confidence: float = 0.99) -> float:
    """The level a completely mixable tail would sit at.

    Which is the sum of the marginals' expected shortfalls, written as its own
    function because that is the quantity the worst case is being compared
    against and reading it as "the flat level" rather than "a bound" is what
    makes the uniform oracle obvious.
    """
    _prepare(tails, confidence)
    return math.fsum(tail.expected_shortfall(confidence) for tail in tails)


__all__ = [
    "DEFAULT_PATIENCE",
    "DEFAULT_POINTS",
    "MAX_SWEEPS",
    "Arrangement",
    "BestCase",
    "Bounds",
    "LossTail",
    "NoTailMean",
    "NotIncreasing",
    "WorstCase",
    "best_case_value_at_risk",
    "completely_mixable_level",
    "dependence_bounds",
    "exponential_tail",
    "normal_tail",
    "pareto_tail",
    "rearrange",
    "uniform_tail",
    "worst_case_value_at_risk",
]

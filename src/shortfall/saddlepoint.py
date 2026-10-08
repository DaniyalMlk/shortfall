"""The tail of a sum, from the one piece of structure a portfolio actually has.

Every tail measure elsewhere in this library reads a distribution that is
assumed (:mod:`shortfall.parametric`), resampled
(:mod:`shortfall.historical`) or fitted at the extreme
(:mod:`shortfall.extreme`). None of them uses the fact that a portfolio loss is
a **sum**. For a sum of independent pieces the cumulant generating function is
the sum of theirs and is available in closed form even where the density is
not, and that is enough to get the tail.

The construction is a change of measure. Tilt the distribution by ``e^{t L}``
so that its mean lands on the loss level of interest, expand there, and untilt.
The expansion point moves with the question rather than sitting at the centre
of the distribution, which is why the relative error stays bounded as the tail
thins — the property the central limit theorem does not have, and the reason a
normal approximation is worthless at 99.9%.

:func:`tail_probability` is Lugannani and Rice's form of it:

    ``P(L > x) ~ 1 - Phi(w) + phi(w) (1/w - 1/u)``

with ``w = sign(t) sqrt(2 (t x - K(t)))`` and ``u = t sqrt(K''(t))``, at the
saddlepoint ``t`` solving ``K'(t) = x``. ``K'`` is increasing, because ``K''``
is a variance, so the saddlepoint is unique and a bracket plus bisection finds
it without a starting guess.

**There is an exact reference, which is the point.** For exposures sharing a
common unit the loss distribution is a finite convolution:
:func:`exact_distribution` runs one pass per obligor over the lattice and
returns the whole law in three milliseconds for a hundred names. So the
approximation's error is *measured* at every point of the tail against
something containing nothing of the approximation -- no tilt, no expansion, no
normal distribution function.

Measured on a hundred-name portfolio with default probabilities from 0.49% to
7.72%, integer exposures from 1 to 20, a mean loss of 35.84 and a standard
deviation of 20.68:

* **The saddlepoint's error shrinks into the tail and the normal
  approximation's grows to everything.** At exceedance levels of 1e-02, 1e-03,
  1e-04, 1e-05 and 1e-06 the saddlepoint is high by 3.3e-04, 2.5e-04, 2.0e-04,
  1.6e-04 and 1.1e-04 relative, while a normal with the same first two moments
  is **low by 69%, 94%, 99.45%, 99.96% and 99.99%**. That is the whole
  distinction between expanding at the point of interest and expanding at the
  centre, and it is a factor of two thousand at a one-in-a-hundred loss.
* **The normal approximation is not a tail problem.** It first errs by more
  than ten per cent at an exceedance of **0.46** -- essentially at the mean,
  because the portfolio is skewed and a normal is not.
* **The saddlepoint is worst in the body, not the tail.** Over the whole
  lattice its largest relative error is 5.9e-03, at an exceedance of 0.975.
* **The conditional mean is better than the probability it divides by.**
  :func:`saddlepoint_shortfall` is within 1.2e-04 relative of the exact
  conditional mean at a loss of 40 and within 4.3e-05 at 120, which is an order
  better than the tail probability, because the numerator is an exact
  decomposition and only its individual terms are approximated.

**The lattice correction is not a refinement.** Exposures sharing a common unit
leave the loss with no density at all, and the continuous form of the formula
applied to it reads **4.7% to 7.2% high** across those same exceedance
levels -- growing into the tail, which is the shape of failure the saddlepoint
exists to avoid. Correcting it is worth a factor of 150 at a one-in-a-hundred
loss and 700 at a one-in-a-million one. :func:`lattice_span` detects the unit
with exact rational arithmetic rather than leaving it to the caller to
remember.

**The formula has a removable singularity at the mean**, where the saddlepoint
is zero and ``1/w - 1/u`` is ``0/0``. A branch on an exactly-zero saddlepoint
does not catch a nearly-zero one, so the switch is to a neighbourhood whose
width is measured rather than chosen -- see :data:`SADDLE_FLOOR`, where the
generic form's agreement with the limit improves down to a tilt of 1e-06 and
then collapses to an error of 96 by 1e-10.

**And the approximation is not a distribution.** Nothing in the expansion is
monotone and nothing keeps it inside ``[0, 1]``:

* On the hundred-name portfolio it stays inside ``[0, 1]`` to within 6.1e-18
  and is non-monotone at nine of nine hundred lattice points, every one of them
  at a tail probability below 4e-16. That is round-off and not a failure of the
  method, and saying so needs the measurement.
* Two names with a 400-to-1 exposure ratio is a failure of the method. The raw
  value runs from **-3.15 to 4.15**, the sequence is not monotone, and the
  worst relative error is **300%**. There is no asymptotic regime with two
  summands, and :func:`tail_probability` reports this by clamping and setting a
  flag rather than by hiding it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .distributions import normal_cdf, normal_pdf

__all__ = [
    "SADDLE_FLOOR",
    "Exceedance",
    "LossPortfolio",
    "Obligor",
    "SaddlepointError",
    "exact_distribution",
    "exact_shortfall",
    "exact_tail",
    "exceedance_probability",
    "lattice_span",
    "normal_tail",
    "saddlepoint",
    "saddlepoint_shortfall",
    "shortfall_contributions",
    "tail_probability",
]

#: Below this saddlepoint the limiting form at the mean is used instead of the
#: generic one. Not a round number picked for comfort: ``w`` is a difference of
#: two nearly equal quantities under a square root, so the generic form loses
#: digits to cancellation as the saddlepoint goes to zero, while the limiting
#: form's own error grows linearly in it. Swept on the hundred-name portfolio
#: in the module docstring, the gap between the two forms runs 8.2e-02, 8.0e-03,
#: 8.0e-04, 8.0e-05, 7.6e-06 as the tilt goes 1e-02 to 1e-06 -- the limit's
#: error, falling linearly -- and then **rises** to 4.1e-06, 2.8e-03, 1.14 and
#: 96.0 at 1e-07 through 1e-10, which is the cancellation taking over. This is
#: the last decade in which the generic form is still clean, so the limiting
#: form's own error here is 1.7e-05 relative: a hundred times smaller than the
#: approximation's own.
SADDLE_FLOOR = 1.0e-6

_MAX_TILT = 50.0


class SaddlepointError(ValueError):
    """A portfolio or a loss level the approximation has nothing to say about.

    Raised for a loss level outside the support, where no tilt puts the mean
    there, and for a portfolio with no obligor that can lose anything.
    """


@dataclass(frozen=True, slots=True)
class Obligor:
    """One name that either pays or does not.

    Attributes:
        probability: Chance of loss, in ``[0, 1]``.
        exposure: Loss given default, positive.
    """

    probability: float
    exposure: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise SaddlepointError(f"a default probability of {self.probability!r} is not one")
        if self.exposure <= 0.0 or not math.isfinite(self.exposure):
            raise SaddlepointError(f"an exposure of {self.exposure!r} is not one")

    @property
    def mean(self) -> float:
        return self.probability * self.exposure

    @property
    def variance(self) -> float:
        return self.probability * (1.0 - self.probability) * self.exposure**2


def lattice_span(obligors: Sequence[Obligor]) -> float:
    """The largest ``d`` with every exposure a whole multiple of it, or zero.

    A sum of obligors whose exposures share a common unit does not have a
    density at all -- it has mass on a lattice -- and the continuous form of
    the saddlepoint approximation is then wrong by tens of per cent rather
    than slightly. So the span is detected rather than assumed away, and zero
    means "no common unit found, treat as continuous".

    Exact where it matters: the denominators are cleared with
    :class:`fractions.Fraction` so that exposures of 0.25 and 0.1 come back as
    0.05 and not as a float that is nearly it.
    """
    if not obligors:
        return 0.0
    try:
        ratios = [Fraction(one.exposure).limit_denominator(10**6) for one in obligors]
    except (OverflowError, ValueError):  # pragma: no cover - needs a non-finite
        return 0.0
    denominator = 1
    for ratio in ratios:
        denominator = denominator * ratio.denominator // math.gcd(denominator, ratio.denominator)
    integers = []
    for one, ratio in zip(obligors, ratios, strict=True):
        scaled = ratio * denominator
        if scaled.denominator != 1:  # pragma: no cover - cleared by construction
            return 0.0
        if abs(float(ratio) - one.exposure) > 1e-12 * max(1.0, one.exposure):
            return 0.0
        integers.append(int(scaled))
    common = 0
    for value in integers:
        common = math.gcd(common, value)
    return common / denominator


@dataclass(frozen=True)
class LossPortfolio:
    """Independent obligors, and the cumulant generating function of their sum.

    The derivatives are written obligor by obligor rather than differentiated
    numerically, because the saddlepoint solve needs ``K'`` at machine
    precision and the tail formula needs ``K''`` under a square root. The third
    derivative is needed only at the mean, where it is the whole of the
    correction term.

    Attributes:
        obligors: The names. At least one, each able to lose something.
        span: Lattice spacing of the loss, or zero to treat it as continuous.
            Detected by :func:`lattice_span` when built through
            :meth:`detected`, because getting this wrong is a tens-of-per-cent
            error rather than a refinement.
    """

    obligors: tuple[Obligor, ...]
    span: float = 0.0

    def __post_init__(self) -> None:
        if not self.obligors:
            raise SaddlepointError("a portfolio needs at least one obligor")
        if self.maximum <= 0.0:
            raise SaddlepointError(
                "every obligor in this portfolio has a zero default probability, "
                "so the loss is identically zero and has no tail to approximate"
            )

    @classmethod
    def detected(cls, obligors: Sequence[Obligor]) -> LossPortfolio:
        """Build with the lattice span worked out from the exposures."""
        names = tuple(obligors)
        return cls(names, lattice_span(names))

    @classmethod
    def homogeneous(cls, names: int, probability: float, exposure: float = 1.0) -> LossPortfolio:
        """``names`` identical obligors. The binomial case, where the exact law is known."""
        return cls.detected(tuple(Obligor(probability, exposure) for _ in range(names)))

    def without(self, position: int) -> LossPortfolio:
        """The portfolio with one obligor removed, on the same lattice.

        Deliberately the *parent's* span rather than the sub-portfolio's own.
        Dropping a name can only coarsen the common unit -- drop the 3 from
        ``{2, 3}`` and it goes from 1 to 2 -- and the shifted level a
        contribution asks about sits on the parent's lattice, not the coarser
        one.
        """
        return LossPortfolio(self.obligors[:position] + self.obligors[position + 1 :], self.span)

    @property
    def mean(self) -> float:
        """``E[L]``, which is also ``K'(0)``."""
        return sum(one.mean for one in self.obligors)

    @property
    def variance(self) -> float:
        """``Var(L)``, which is also ``K''(0)``."""
        return sum(one.variance for one in self.obligors)

    @property
    def maximum(self) -> float:
        """Largest loss with positive probability.

        An obligor that cannot default does not raise the ceiling, which
        matters: the tail formula is asked about levels up to this and a name
        with a zero probability would move it without being reachable.
        """
        return sum(one.exposure for one in self.obligors if one.probability > 0.0)

    def cumulant(self, tilt: float) -> float:
        """``K(t) = sum_i log(1 + p_i (e^{t e_i} - 1))``."""
        total = 0.0
        for one in self.obligors:
            total += math.log1p(one.probability * math.expm1(tilt * one.exposure))
        return total

    def derivative(self, tilt: float) -> float:
        """``K'(t)``, the tilted mean. Increasing in ``t``."""
        total = 0.0
        for one in self.obligors:
            weight = self._tilted(one, tilt)
            total += weight * one.exposure
        return total

    def second_derivative(self, tilt: float) -> float:
        """``K''(t)``, the tilted variance. Non-negative."""
        total = 0.0
        for one in self.obligors:
            weight = self._tilted(one, tilt)
            total += weight * (1.0 - weight) * one.exposure**2
        return total

    def third_derivative(self, tilt: float) -> float:
        """``K'''(t)``. Needed only at the mean, where it is the whole correction."""
        total = 0.0
        for one in self.obligors:
            weight = self._tilted(one, tilt)
            total += weight * (1.0 - weight) * (1.0 - 2.0 * weight) * one.exposure**3
        return total

    @staticmethod
    def _tilted(one: Obligor, tilt: float) -> float:
        """The obligor's default probability under the tilted measure.

        Written as a logistic in ``t e - log((1-p)/p)`` rather than as
        ``p e^{te} / (1 + p(e^{te} - 1))`` so that a large tilt saturates at
        one instead of overflowing on the way there.
        """
        if one.probability <= 0.0:
            return 0.0
        if one.probability >= 1.0:
            return 1.0
        odds = math.log(one.probability / (1.0 - one.probability))
        argument = tilt * one.exposure + odds
        if argument > _MAX_TILT:
            return 1.0
        if argument < -_MAX_TILT:
            return 0.0
        return 1.0 / (1.0 + math.exp(-argument))


def saddlepoint(portfolio: LossPortfolio, level: float) -> float:
    """The tilt ``t`` with ``K'(t) = level``.

    ``K'`` is increasing because ``K''`` is a variance, so the root is unique
    and a widening bracket plus bisection finds it with no starting guess and
    no derivative of the residual. Newton on a function that saturates at both
    ends is the thing not to use here.

    Raises:
        SaddlepointError: if ``level`` is outside ``(0, maximum)``, where no
            finite tilt moves the mean onto it.
    """
    if level <= 0.0 or level >= portfolio.maximum:
        raise SaddlepointError(
            f"a loss of {level!r} is outside the support (0, "
            f"{portfolio.maximum!r}); no finite tilt puts the mean there"
        )
    low, high = -1.0, 1.0
    while portfolio.derivative(low) > level:
        low *= 2.0
        if low < -_MAX_TILT * 4.0:
            break
    while portfolio.derivative(high) < level:
        high *= 2.0
        if high > _MAX_TILT * 4.0:
            break
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if portfolio.derivative(middle) < level:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True, slots=True)
class Exceedance:
    """An approximate ``P(L > x)``, and what had to be done to report it.

    Attributes:
        level: The loss level.
        probability: The approximation, clamped into ``[0, 1]``.
        raw: Before clamping. Outside ``[0, 1]`` more often than one would
            like, which is why it is kept.
        tilt: The saddlepoint.
        near_mean: True when the limiting form at the mean was used.
    """

    level: float
    probability: float
    raw: float
    tilt: float
    near_mean: bool

    @property
    def clamped(self) -> bool:
        """Whether the raw value had to be pulled into ``[0, 1]``."""
        return self.raw != self.probability


def tail_probability(portfolio: LossPortfolio, level: float) -> Exceedance:
    """Lugannani and Rice's approximation to ``P(L > level)``.

    **The lattice correction is not optional.** A portfolio whose exposures
    share a common unit has no density at all, and the continuous formula
    applied to it is wrong by tens of per cent rather than slightly. Where
    ``portfolio.span`` is positive the saddlepoint is solved at the midpoint
    ``level + span / 2`` and the correction denominator becomes
    ``(2 / d) sinh(t d / 2) sqrt(K'')``, which tends to ``t sqrt(K'')`` as the
    span goes to zero -- so the continuous form is the limit of this one rather
    than a separate branch, and the tests check that limit numerically.

    At the mean the saddlepoint is zero and ``1/w - 1/u`` is ``0/0``. The limit
    there is ``1/2 - K'''(0) / (6 sqrt(2 pi) K''(0)^{3/2})``, and it is used
    throughout a neighbourhood rather than at the single point: ``w`` is a
    difference of two nearly equal quantities under a square root, so the
    generic form loses digits to cancellation long before the saddlepoint is
    exactly zero. See :data:`SADDLE_FLOOR`.

    The result is clamped into ``[0, 1]`` and says so, because nothing in the
    expansion keeps it there.
    """
    span = portfolio.span
    target = level + 0.5 * span if span > 0.0 else level
    tilt = saddlepoint(portfolio, target)
    second = portfolio.second_derivative(tilt)
    if abs(tilt) < SADDLE_FLOOR or second <= 0.0:
        third = portfolio.third_derivative(0.0)
        variance = portfolio.second_derivative(0.0)
        raw = 0.5 - third / (6.0 * math.sqrt(2.0 * math.pi) * variance**1.5)
        near = True
    else:
        inner = 2.0 * (tilt * target - portfolio.cumulant(tilt))
        w = math.copysign(math.sqrt(max(inner, 0.0)), tilt)
        if span > 0.0:
            u = (2.0 / span) * math.sinh(0.5 * tilt * span) * math.sqrt(second)
        else:
            u = tilt * math.sqrt(second)
        # Lugannani and Rice state the approximation for the distribution
        # function, ``F(x) ~ Phi(w) + phi(w) (1/w - 1/u)``, so the correction
        # enters the tail with a minus. With it the wrong way round the answer
        # is still plausible and monotone, and reads 14% high at the mean
        # rising to 65% in the tail -- an error that grows with the thinning
        # tail, which is exactly the shape of failure the saddlepoint exists
        # to avoid, so it cannot be told from a method that does not work.
        raw = 1.0 - normal_cdf(w) - normal_pdf(w) * (1.0 / w - 1.0 / u)
        near = False
    return Exceedance(
        level=level,
        probability=min(1.0, max(0.0, raw)),
        raw=raw,
        tilt=tilt,
        near_mean=near,
    )


def exceedance_probability(portfolio: LossPortfolio, level: float) -> float:
    """``P(L > level)``, approximated, with the ends of the support handled exactly.

    The saddlepoint has nothing to say outside ``(0, maximum)`` and does not
    need to: below zero the probability is one and at or above the maximum it
    is zero, both without approximation. Having this as its own function is
    what lets :func:`shortfall_contributions` call it on sub-portfolios whose
    shifted level routinely lands outside their own support.
    """
    if level < 0.0:
        return 1.0
    if level >= portfolio.maximum:
        return 0.0
    if level == 0.0:
        # P(L > 0) is one less the chance nobody defaults, which is exact.
        survival = 1.0
        for one in portfolio.obligors:
            survival *= 1.0 - one.probability
        return 1.0 - survival
    return tail_probability(portfolio, level).probability


def shortfall_contributions(portfolio: LossPortfolio, level: float) -> tuple[float, ...]:
    """Each obligor's share of ``E[L 1{L > level}]``.

    The decomposition is **exact**, which is the reason to build the shortfall
    this way rather than from a second expansion::

        E[L 1{L>x}] = sum_i e_i E[B_i 1{L>x}]
                    = sum_i e_i p_i P(L^(i) > x - e_i)

    where ``L^(i)`` is the portfolio without obligor ``i``. Conditioning on
    that obligor defaulting replaces its contribution to the loss by a
    constant, and the rest of the portfolio is independent of it. So the only
    approximation is the tail probability of each sub-portfolio, computed by
    the same function as the total -- the two cannot drift apart, and the sum
    is a check on the parts rather than a definition of the whole.

    It also makes the per-obligor numbers a desk actually asks for fall out of
    the total for free, which the second-expansion route does not.
    """
    if not 0.0 <= level < portfolio.maximum:
        raise SaddlepointError(
            f"a loss of {level!r} is outside [0, {portfolio.maximum!r}), where "
            "there is nothing above the level to take a mean of"
        )
    shares = []
    for position, one in enumerate(portfolio.obligors):
        if one.probability <= 0.0:
            shares.append(0.0)
            continue
        if len(portfolio.obligors) > 1:
            tail = exceedance_probability(portfolio.without(position), level - one.exposure)
        else:
            tail = 1.0 if level < one.exposure else 0.0
        shares.append(one.exposure * one.probability * tail)
    return tuple(shares)


def saddlepoint_shortfall(portfolio: LossPortfolio, level: float) -> float:
    """``E[L | L > level]``, from the exact decomposition over its numerator.

    Raises:
        SaddlepointError: if the approximate exceedance probability is zero,
            where a conditional mean divides by it.
    """
    probability = exceedance_probability(portfolio, level)
    if probability <= 0.0:
        raise SaddlepointError(
            f"the approximate exceedance probability at {level!r} is zero, so a "
            "conditional mean divides by it"
        )
    return sum(shortfall_contributions(portfolio, level)) / probability


def exact_distribution(portfolio: LossPortfolio, *, unit: float = 1.0) -> tuple[float, ...]:
    """The loss distribution exactly, by convolution on the exposure lattice.

    One pass per obligor over the lattice, which is the whole algorithm: the
    loss is a sum of independent pieces, so its law is the convolution of
    theirs, and on a lattice a convolution with a two-point law is a shift and
    a weighted add.

    This is the reference the approximation is measured against, and it shares
    nothing with it — no tilt, no expansion, no normal distribution function.

    Args:
        portfolio: The obligors. Every exposure must be a whole number of
            ``unit``, since a lattice cannot represent anything else.
        unit: The lattice spacing.

    Returns:
        ``P(L = k * unit)`` for ``k`` from zero to the total exposure.

    Raises:
        SaddlepointError: if an exposure is not a whole number of ``unit``.
    """
    if unit <= 0.0:
        raise SaddlepointError(f"a lattice spacing of {unit!r} is not one")
    steps = []
    for one in portfolio.obligors:
        ratio = one.exposure / unit
        rounded = round(ratio)
        if abs(ratio - rounded) > 1e-9 * max(1.0, abs(ratio)):
            raise SaddlepointError(
                f"an exposure of {one.exposure!r} is {ratio!r} units of "
                f"{unit!r}, and a lattice holds whole units only. Use the "
                "saddlepoint approximation, which does not need a lattice, or "
                "choose a unit these exposures are multiples of."
            )
        steps.append(rounded)
    size = sum(steps) + 1
    masses = [0.0] * size
    masses[0] = 1.0
    reach = 0
    for one, step in zip(portfolio.obligors, steps, strict=True):
        survive = 1.0 - one.probability
        for position in range(reach, -1, -1):
            mass = masses[position]
            if mass == 0.0:
                continue
            masses[position] = mass * survive
            masses[position + step] += mass * one.probability
        reach += step
    return tuple(masses)


def exact_tail(masses: Sequence[float], level: float, *, unit: float = 1.0) -> float:
    """``P(L > level)`` from an exact distribution.

    Strictly greater, matching :func:`tail_probability`, which matters on a
    lattice where the level itself carries mass.
    """
    threshold = level / unit
    return sum(mass for index, mass in enumerate(masses) if index > threshold + 1e-12)


def exact_shortfall(masses: Sequence[float], level: float, *, unit: float = 1.0) -> float:
    """``E[L | L > level]`` from an exact distribution."""
    threshold = level / unit
    weight = 0.0
    total = 0.0
    for index, mass in enumerate(masses):
        if index > threshold + 1e-12:
            weight += mass
            total += mass * index * unit
    if weight <= 0.0:
        raise SaddlepointError(f"nothing exceeds {level!r}, so there is no mean above it")
    return total / weight


def normal_tail(portfolio: LossPortfolio, level: float) -> float:
    """``P(L > level)`` from a normal with the portfolio's first two moments.

    Here to be beaten. It is the approximation an expansion at the centre of
    the distribution gives, and the comparison against it is the measurement
    that says why the expansion point should move.
    """
    deviation = math.sqrt(portfolio.variance)
    if deviation <= 0.0:
        return 1.0 if level < portfolio.mean else 0.0
    return 1.0 - normal_cdf((level - portfolio.mean) / deviation)

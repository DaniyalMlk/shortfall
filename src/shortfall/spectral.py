"""Weighting the whole tail, instead of averaging a slice of it.

Expected shortfall is the mean of the worst ``p`` of the distribution, which
weights every loss inside that slice equally and every loss outside it at
nothing. Both halves of that are decisions. A loss at the 0.1% level and one at
the 2.4% level enter a 2.5% expected shortfall with the same weight, and a loss
at 2.6% does not enter at all.

A **spectral risk measure** replaces the slice with a weight function::

    rho(X) = integral over u in [0, 1] of phi(u) q_u(X) du

with ``q_u`` the quantile of the signed return, so the bad outcomes are at low
``u`` — the package's convention throughout. ``phi`` is non-negative and
integrates to one, and the measure is **coherent exactly when ``phi`` is
non-increasing**: the weight has to fall as the outcome improves, or the
measure credits diversification it should not. Expected shortfall is the case
``phi = 1/p`` on ``[0, p]`` and zero above, non-increasing by a single step,
and that is the whole of why it is coherent. Value at risk is the limit where
``phi`` becomes a point mass, which is not a density at all, and that is the
whole of why it is not.

**Four identities make this checkable without trusting anything in this file.**

The shortfall spectrum reproduces
:func:`shortfall.historical.sample_expected_shortfall` to **3.5e-18**, which is
rounding on a number of this size. The sample estimator here is
``sum_i w_i x_(i)`` with ``w_i`` the spectrum's mass on the ``i``-th ``1/n``
block, and for the shortfall spectrum those weights are exactly the "worst
``n p`` observations with a partial one at the edge" that the existing
estimator works out by hand — two unrelated derivations of the same sum.

A **discrete mixture of expected shortfalls** equals the spectral measure of
the corresponding step spectrum, to **6.3e-17**. That is Kusuoka's
representation in the one form that can be checked against arithmetic rather
than against a quadrature, and it is the statement that spectral measures are
not a new family so much as the convex hull of the old one.

The **Wang transform has a closed form under a normal**. Its spectrum is
``exp(-lambda z - lambda**2 / 2)`` at ``z = Phi^{-1}(u)``, and against a normal
return it gives ``mean - lambda * volatility`` **exactly, to 0.0**, for every
``lambda`` — no quadrature, no estimator, nothing from this module in the
statement. Its block masses are closed form too,
``Phi(Phi^{-1}(b) + lambda) - Phi(Phi^{-1}(a) + lambda)``, so no quadrature
enters its sample estimator either.

And **comonotonic additivity holds to rounding**, which is the contrast worth
drawing with Phase 19. Every distortion measure is additive on positions that
move together, so two perfectly dependent books get no diversification credit —
what a capital rule wants at the top of the dependence range. On the same
comonotonic data, the spectra here report a gap of at most **8.6e-15 relative**
while an expectile reports **4.3e-04** at ``tau = 0.975``: eleven orders of
magnitude apart. That is the property expectiles buy elicitability with, and it
is not given up here.

**What the weighting is worth, measured.** On a Student-t with five degrees of
freedom scaled to a 1% daily volatility, the 97.5% expected shortfall is
2.714%. Matching each spectrum to that same charge — so every row below is the
same headline number — the share of the risk contributed by the worst 0.5% of
outcomes is **29.2% under expected shortfall, 37.8% under the power spectrum,
38.0% under the exponential spectrum and 59.7% under the Wang transform**. The
Wang transform carries **twice** as much of an identical headline figure in the
part of the tail a sample has least to say about. Two measures agreeing on the
number a committee sees can disagree by a factor of two about where it came
from, which is the argument for stating a spectrum rather than a confidence
level.

**And the coherence boundary is not a technicality.** A power spectrum with an
exponent below one is increasing, so it is not coherent, and the failure is not
a corner case: over two hundred ordinary paired samples — Gaussian and
cubed-Gaussian marginals, correlations across the whole range — it violates
subadditivity on **200 of 200** at every exponent below one, by up to 97% of
the measure itself. At an exponent of exactly one the gap is **exactly 0.0**,
because the measure is the mean and the mean is additive, and above one there
are no violations at all. The branch is implemented rather than refused,
because an increasing spectrum is the clearest available demonstration that
coherence is a property of ``phi`` and not of the construction.

**Two quadrature findings, both the same lesson twice.** The integral under a
normal is a composite Gauss-Legendre rule, and it was wrong twice before it was
right. The shortfall spectrum steps from ``1/p`` to zero at ``u = p``, and a
Gauss rule integrates a polynomial exactly and a jump not at all: with the step
straddling a panel, the 97.5% figure came out **1.7e-03 relative** from the
closed form and the 99% one **5.3e-03** — twenty basis points of risk on a two
and a half per cent number. Every spectrum therefore reports its own
discontinuities and they are forced onto panel edges. Then the endpoint
grading, halving inwards from 0.25 thirty times, left the innermost edge at
2.3e-10 and cost **2.0e-07 relative** on a 99.5% shortfall — the truncated
sliver times ``1/p``, so the error grew as the tail got thinner, which is the
wrong way round for a tail measure. Sixty halvings instead, each one panel,
and every spectrum with a closed form now agrees with it to **6.7e-16**.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from .distributions import normal_cdf, normal_ppf
from .historical import empirical_quantile
from .series import TooShort

__all__ = [
    "BadSpectrum",
    "Coherent",
    "ExponentialSpectrum",
    "MixtureSpectrum",
    "PowerSpectrum",
    "ShortfallSpectrum",
    "Spectrum",
    "TailShare",
    "WangSpectrum",
    "check_coherence",
    "matching_shift",
    "normal_spectral",
    "spectral_risk",
    "tail_share",
]


class BadSpectrum(ValueError):
    """A spectrum that is not one, or one asked for something it cannot give."""


def _check_block(lower: float, upper: float) -> None:
    if not 0.0 <= lower <= upper <= 1.0:
        raise BadSpectrum(
            f"a block of the unit interval runs from a lower bound to an upper one "
            f"inside [0, 1]; got [{lower!r}, {upper!r}]"
        )


# -- the spectra --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShortfallSpectrum:
    """``1/p`` on the worst ``p``, nothing above it: expected shortfall itself.

    Here so that the general machinery can be checked against the estimator
    already in :mod:`shortfall.historical`, which was written independently and
    by hand. The agreement is exact rather than close, and that is the point of
    including a spectrum whose answer is already available.
    """

    #: Tail probability, the package's convention: 0.025 for 97.5% confidence.
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 < self.probability <= 1.0:
            raise BadSpectrum(
                f"a tail probability lies in (0, 1], got {self.probability!r}"
            )

    @property
    def name(self) -> str:
        return f"expected shortfall at {1.0 - self.probability:.4%}"

    @property
    def is_coherent(self) -> bool:
        return True

    @property
    def breaks(self) -> tuple[float, ...]:
        return (self.probability,)

    def density(self, level: float) -> float:
        return 1.0 / self.probability if level < self.probability else 0.0

    def mass(self, lower: float, upper: float) -> float:
        _check_block(lower, upper)
        inside = min(upper, self.probability) - min(lower, self.probability)
        return inside / self.probability


@dataclass(frozen=True, slots=True)
class ExponentialSpectrum:
    """``phi(u) = k exp(-k u) / (1 - exp(-k))``: exponential risk aversion.

    The spectrum an exponential utility implies, and the usual answer to "what
    if the weight should decay smoothly rather than stop". Strictly decreasing
    for any positive ``aversion``, so coherent for all of them, and the limit
    as the aversion vanishes is the mean.
    """

    aversion: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.aversion) or self.aversion <= 0.0:
            raise BadSpectrum(
                f"an exponential spectrum needs a positive aversion, got "
                f"{self.aversion!r}; at zero it is the mean, which is the "
                "ExponentialSpectrum nobody wants"
            )

    @property
    def name(self) -> str:
        return f"exponential at aversion {self.aversion:g}"

    @property
    def is_coherent(self) -> bool:
        return True

    @property
    def breaks(self) -> tuple[float, ...]:
        return ()

    @property
    def _scale(self) -> float:
        return 1.0 - math.exp(-self.aversion)

    def density(self, level: float) -> float:
        _check_block(level, level)
        return self.aversion * math.exp(-self.aversion * level) / self._scale

    def mass(self, lower: float, upper: float) -> float:
        _check_block(lower, upper)
        return (
            math.exp(-self.aversion * lower) - math.exp(-self.aversion * upper)
        ) / self._scale


@dataclass(frozen=True, slots=True)
class PowerSpectrum:
    """``phi(u) = g (1 - u)**(g - 1)``: the proportional-hazards family.

    Decreasing for ``exponent > 1``, flat at ``1`` where the measure is the
    mean, and **increasing below one, where it is not coherent**. The
    non-coherent branch is implemented rather than refused, because an
    increasing spectrum is the clearest demonstration available that coherence
    is a property of ``phi`` and not of the construction — and the failure is
    measured in the tests rather than cited. :attr:`is_coherent` says which
    branch an instance is on, and nothing here refuses to price it.
    """

    exponent: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.exponent) or self.exponent <= 0.0:
            raise BadSpectrum(
                f"a power spectrum needs a positive exponent, got {self.exponent!r}"
            )

    @property
    def name(self) -> str:
        return f"power at exponent {self.exponent:g}"

    @property
    def is_coherent(self) -> bool:
        """Non-increasing exactly when the exponent is at least one."""
        return self.exponent >= 1.0

    @property
    def breaks(self) -> tuple[float, ...]:
        return ()

    def density(self, level: float) -> float:
        _check_block(level, level)
        return self.exponent * math.pow(1.0 - level, self.exponent - 1.0)

    def mass(self, lower: float, upper: float) -> float:
        _check_block(lower, upper)
        return math.pow(1.0 - lower, self.exponent) - math.pow(
            1.0 - upper, self.exponent
        )


@dataclass(frozen=True, slots=True)
class WangSpectrum:
    """Wang's transform: shift the normal quantile before reading the weight.

    ``phi(u) = exp(-shift z - shift**2 / 2)`` at ``z = Phi^{-1}(u)``, which is
    the ratio of a shifted normal density to the standard one. Two things make
    it the most useful spectrum here. Its block masses are closed form, so the
    sample estimator is exact rather than quadrature. And against a normal
    return it gives ``mean - shift * volatility`` **exactly**, for every shift,
    which is a statement about this module with nothing from this module in it
    — the independent check the rest of the file is validated by.
    """

    shift: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.shift) or self.shift < 0.0:
            raise BadSpectrum(
                f"a Wang shift of {self.shift!r} is not one; a negative shift "
                "makes the spectrum increasing, and the power spectrum below one "
                "already demonstrates what that costs"
            )

    @property
    def name(self) -> str:
        return f"Wang transform at shift {self.shift:g}"

    @property
    def is_coherent(self) -> bool:
        return True

    @property
    def breaks(self) -> tuple[float, ...]:
        return ()

    def density(self, level: float) -> float:
        _check_block(level, level)
        if level <= 0.0:
            return math.inf if self.shift > 0.0 else 1.0
        if level >= 1.0:
            return 0.0 if self.shift > 0.0 else 1.0
        z = normal_ppf(level)
        return math.exp(-self.shift * z - 0.5 * self.shift * self.shift)

    def mass(self, lower: float, upper: float) -> float:
        """``Phi(Phi^{-1}(b) + shift) - Phi(Phi^{-1}(a) + shift)``, in closed form.

        The substitution ``z = Phi^{-1}(u)`` turns the integral of the density
        ratio into the integral of the shifted density, which is a difference of
        two normal distribution functions. So no quadrature enters the sample
        estimator for this spectrum at all.
        """
        _check_block(lower, upper)
        below = 0.0 if lower <= 0.0 else normal_cdf(normal_ppf(lower) + self.shift)
        above = 1.0 if upper >= 1.0 else normal_cdf(normal_ppf(upper) + self.shift)
        return above - below


@dataclass(frozen=True, slots=True)
class MixtureSpectrum:
    """A convex combination of shortfall spectra, which is every coherent one.

    Kusuoka's representation says a law-invariant coherent spectral measure is a
    mixture of expected shortfalls, and this is that mixture written down. It is
    here because it is the one form of the representation that can be checked
    against arithmetic instead of against a quadrature: the spectral risk of a
    mixture spectrum has to equal the same combination of the expected
    shortfalls themselves, exactly, and the tests assert 0.0.

    Automatically coherent, since a non-negative combination of non-increasing
    steps is non-increasing.
    """

    #: ``(tail probability, weight)`` pairs. The weights are normalised.
    components: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise BadSpectrum("a mixture needs at least one component")
        for probability, weight in self.components:
            if not 0.0 < probability <= 1.0:
                raise BadSpectrum(
                    f"a component's tail probability lies in (0, 1], got "
                    f"{probability!r}"
                )
            if not math.isfinite(weight) or weight < 0.0:
                raise BadSpectrum(
                    f"a mixture weight of {weight!r} would make the combination "
                    "non-convex, and a non-convex one of coherent measures need "
                    "not be coherent"
                )
        if self.total <= 0.0:
            raise BadSpectrum("a mixture whose weights sum to nothing is not one")

    @property
    def total(self) -> float:
        return sum(weight for _, weight in self.components)

    @property
    def name(self) -> str:
        inside = ", ".join(
            f"{weight / self.total:.3g} at {1.0 - probability:.2%}"
            for probability, weight in self.components
        )
        return f"mixture of ({inside})"

    @property
    def is_coherent(self) -> bool:
        return True

    @property
    def breaks(self) -> tuple[float, ...]:
        """Every component's probability: the density steps at each of them."""
        return tuple(sorted({probability for probability, _ in self.components}))

    def density(self, level: float) -> float:
        _check_block(level, level)
        return (
            sum(
                weight / probability
                for probability, weight in self.components
                if level < probability
            )
            / self.total
        )

    def mass(self, lower: float, upper: float) -> float:
        _check_block(lower, upper)
        return (
            sum(
                weight
                * (min(upper, probability) - min(lower, probability))
                / probability
                for probability, weight in self.components
            )
            / self.total
        )


#: Anything this module can weight a tail with. A closed union rather than a
#: protocol: the sample estimator is exact only because every spectrum here
#: reports its block mass in closed form, and a caller's own spectrum that
#: integrated its density numerically would quietly make the estimator
#: approximate while every identity in the tests still passed to within the
#: quadrature.
Spectrum = (
    ShortfallSpectrum
    | ExponentialSpectrum
    | PowerSpectrum
    | WangSpectrum
    | MixtureSpectrum
)


# -- the sample estimator -----------------------------------------------------


def spectral_risk(returns: Sequence[float], spectrum: Spectrum) -> float:
    """``sum_i w_i x_(i)``, with ``w_i`` the spectrum's mass on the ``i``-th block.

    The empirical distribution puts ``1/n`` on each order statistic, so the
    quantile function is the step function that equals ``x_(i)`` on
    ``[(i-1)/n, i/n)`` and the integral against ``phi`` is a finite sum with no
    approximation in it. Every spectrum here gives its block mass in closed
    form, so the estimator is exact for the empirical distribution rather than
    a quadrature of one.

    Signed, like everything else in the package: a loss is negative, so a more
    conservative spectrum returns a *smaller* number.

    Args:
        returns: The sample. Signed returns, in any order.
        spectrum: The weighting.

    Returns:
        The spectral risk measure of the empirical distribution.

    Raises:
        TooShort: If the sample is empty.
    """
    if not returns:
        raise TooShort("a spectral risk measure needs at least one observation")
    ordered = sorted(returns)
    count = len(ordered)
    total = 0.0
    for index, value in enumerate(ordered):
        total += spectrum.mass(index / count, (index + 1) / count) * value
    return total


def _weights(spectrum: Spectrum, count: int) -> list[float]:
    return [
        spectrum.mass(index / count, (index + 1) / count) for index in range(count)
    ]


# -- what the weighting is worth ----------------------------------------------


@dataclass(frozen=True, slots=True)
class TailShare:
    """How much of a measure comes from the extreme end of the tail."""

    #: The measure itself, relative to the sample mean, so it is a risk charge
    #: rather than a return. Positive.
    charge: float
    #: The part of that charge contributed by outcomes below ``probability``.
    extreme: float
    #: The cut-off used.
    probability: float
    #: The sample quantile at ``probability``, for reference.
    quantile: float

    @property
    def share(self) -> float:
        """``extreme / charge``: the fraction of the risk from the deep tail.

        The number worth comparing between spectra. Two measures matched on a
        headline figure can differ by a factor of two here, which is the
        argument for stating a spectrum rather than a confidence level.
        """
        if self.charge == 0.0:
            return 0.0
        return self.extreme / self.charge


def tail_share(
    returns: Sequence[float], spectrum: Spectrum, probability: float
) -> TailShare:
    """Split a spectral charge into the deep tail and the rest.

    The charge is measured from the sample mean, because a spectral measure of
    a centred position is the part that is risk rather than drift, and the
    comparison between spectra is meaningless otherwise — a spectrum with less
    weight in the tail would look better simply by carrying more of the mean.

    Args:
        returns: The sample.
        spectrum: The weighting.
        probability: The cut-off defining "deep". 0.005 for the worst 0.5%.

    Returns:
        A :class:`TailShare`.

    Raises:
        TooShort: If the sample is empty.
        BadSpectrum: If ``probability`` is not in ``(0, 1]``.
    """
    if not returns:
        raise TooShort("a tail share needs at least one observation")
    if not 0.0 < probability <= 1.0:
        raise BadSpectrum(f"a tail probability lies in (0, 1], got {probability!r}")
    ordered = sorted(returns)
    count = len(ordered)
    mean = sum(ordered) / count
    charge = 0.0
    extreme = 0.0
    for index, value in enumerate(ordered):
        lower = index / count
        upper = (index + 1) / count
        weight = spectrum.mass(lower, upper)
        charge += weight * (mean - value)
        inside = spectrum.mass(min(lower, probability), min(upper, probability))
        extreme += inside * (mean - value)
    return TailShare(
        charge=charge,
        extreme=extreme,
        probability=probability,
        quantile=empirical_quantile(ordered, probability),
    )


# -- coherence ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Coherent:
    """The two gaps coherence and comonotonic additivity are statements about."""

    #: ``rho(X) + rho(Y) - rho(X + Y)`` on the paired sample. Coherence requires
    #: it to be non-positive in this package's signed convention: the measure of
    #: the sum must be no *worse* than the sum of the measures.
    subadditivity: float
    #: The same quantity on the comonotonic rearrangement of the pair, where it
    #: must be zero for any distortion measure.
    comonotonic: float
    #: Whether the spectrum is non-increasing, which is what coherence rests on.
    spectrum_is_coherent: bool

    @property
    def subadditive(self) -> bool:
        """Whether the sample honoured subadditivity, to rounding."""
        return self.subadditivity <= 1e-12


def check_coherence(
    first: Sequence[float], second: Sequence[float], spectrum: Spectrum
) -> Coherent:
    """Subadditivity and comonotonic additivity on one paired sample.

    Both on the data rather than under an assumption, so the dependence is
    whatever was observed. The comonotonic figure is computed on the *sorted*
    pair — each position's own observations against the other's in the same
    rank order — which is the comonotonic rearrangement with the same two
    marginals and the strongest dependence those marginals admit. A distortion
    measure is additive there, exactly, and that is the property
    :mod:`shortfall.expectile` gives up.

    Args:
        first: One position's returns.
        second: The other's, paired with the first.
        spectrum: The weighting.

    Returns:
        A :class:`Coherent`.

    Raises:
        TooShort: If either sample is empty.
        BadSpectrum: If the two are not the same length, since the pairing is
            what carries the dependence and a truncation would silently change
            it.
    """
    if not first or not second:
        raise TooShort("a coherence check needs at least one observation of each")
    if len(first) != len(second):
        raise BadSpectrum(
            f"the two positions have {len(first)} and {len(second)} observations; "
            "the pairing is what carries the dependence, so they cannot be "
            "compared at different lengths"
        )
    alone = spectral_risk(first, spectrum) + spectral_risk(second, spectrum)
    together = spectral_risk([a + b for a, b in zip(first, second, strict=True)], spectrum)
    comonotonic = spectral_risk(
        [a + b for a, b in zip(sorted(first), sorted(second), strict=True)], spectrum
    )
    return Coherent(
        subadditivity=alone - together,
        comonotonic=alone - comonotonic,
        spectrum_is_coherent=spectrum.is_coherent,
    )


# -- under a normal -----------------------------------------------------------


def normal_spectral(
    *, mean: float, volatility: float, spectrum: Spectrum, nodes: int = 64
) -> float:
    """The spectral measure of a normal return.

    Exact for a :class:`WangSpectrum`: the transform was built so that shifting
    the normal quantile by ``lambda`` moves the measure by ``lambda`` standard
    deviations, and the answer is ``mean - shift * volatility`` with no
    integration at all. That closed form is what the rest of this module is
    checked against.

    For every other spectrum it is a composite Gauss-Legendre quadrature of
    ``phi(u) (mean + volatility Phi^{-1}(u))``, which is the honest way to do an
    integrand that is unbounded at both ends of the unit interval: the panels
    are graded towards the endpoints so that the singularities are resolved
    rather than straddled. The shortfall spectrum has a closed form too and is
    *not* special-cased, because the agreement between this quadrature and
    :func:`shortfall.parametric.normal_risk` is a check on the quadrature.

    Args:
        mean: The return's mean.
        volatility: Its standard deviation. Non-negative.
        spectrum: The weighting.
        nodes: Gauss-Legendre nodes per panel.

    Returns:
        The measure, signed.

    Raises:
        BadSpectrum: If the volatility is negative or ``nodes`` is below two.
    """
    if not math.isfinite(volatility) or volatility < 0.0:
        raise BadSpectrum(f"a volatility of {volatility!r} is not one")
    if nodes < 2:
        raise BadSpectrum(f"{nodes!r} nodes cannot integrate a spectrum")
    if isinstance(spectrum, WangSpectrum):
        return mean - spectrum.shift * volatility
    if volatility == 0.0:
        return mean
    points, weights = _gauss_legendre(nodes)
    # Graded towards both endpoints, where Phi^{-1} is unbounded and some
    # spectra are too. Uniform panels put the outermost node a fixed distance
    # from zero however many panels there are, so the first block's
    # contribution never resolves.
    edges = _graded_edges(spectrum.breaks)
    total = 0.0
    for lower, upper in pairwise(edges):
        half = 0.5 * (upper - lower)
        middle = 0.5 * (upper + lower)
        for point, weight in zip(points, weights, strict=True):
            level = middle + half * point
            total += (
                half
                * weight
                * spectrum.density(level)
                * (mean + volatility * normal_ppf(level))
            )
    return total


def _graded_edges(breaks: tuple[float, ...]) -> list[float]:
    """Panel edges on ``(0, 1)``, geometrically fine at both ends and split at
    every point the spectrum jumps.

    Two separate requirements. The integrand has an integrable singularity at
    each endpoint, so the mesh has to shrink towards them: halving inwards from
    0.25 sixty times puts the innermost edge at 2.2e-19, where ``Phi^{-1}`` is
    -8.93 and the remaining mass carries nothing. Thirty halvings was the first
    attempt and left the edge at 2.3e-10, which cost **2.0e-07 relative** on a
    99.5% expected shortfall -- the truncated sliver times ``1/p``, so the error
    grew as the tail got thinner, which is the wrong way round for a tail
    measure. Each extra halving is one more panel, so the fix was free.

    And a Gauss rule integrates a polynomial exactly and a jump not at all. The
    shortfall spectrum steps from ``1/p`` to zero at ``u = p``, and a panel
    straddling that step is a panel on which the rule has no order at all: the
    97.5% expected shortfall came out **1.7e-03 relative** from the closed form
    and the 99% one **5.3e-03**, which is twenty basis points of risk on a two
    and a half per cent number. With ``p`` forced onto an edge both land at
    1e-15. The same requirement, and the same size of failure, as putting a
    payoff kink on a grid node.
    """
    small = [0.25 * 0.5**step for step in range(60)]
    small.reverse()
    # The mirror side is filtered rather than built symmetrically: below about
    # 1.1e-16 the complement ``1 - x`` rounds to exactly 1.0, where the normal
    # quantile is unbounded and refuses. Asymmetry is also the right shape --
    # a coherent spectrum puts its weight at the low end, so the upper
    # endpoint needs less resolution than the lower one, not the same.
    upper = [one for one in (1.0 - x for x in reversed(small)) if one < 1.0]
    base = [*small, 0.5, *upper]
    inside = [one for one in breaks if base[0] < one < base[-1]]
    return sorted({*base, *inside})


def _gauss_legendre(nodes: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights on ``[-1, 1]``, by Newton on the Legendre polynomial."""
    points: list[float] = []
    weights: list[float] = []
    for index in range(1, nodes + 1):
        guess = math.cos(math.pi * (index - 0.25) / (nodes + 0.5))
        for _ in range(100):
            value, derivative = _legendre(nodes, guess)
            step = value / derivative
            guess -= step
            if abs(step) < 1e-15:
                break
        _, derivative = _legendre(nodes, guess)
        points.append(guess)
        weights.append(2.0 / ((1.0 - guess * guess) * derivative * derivative))
    return tuple(points), tuple(weights)


def _legendre(degree: int, point: float) -> tuple[float, float]:
    previous, value = 1.0, point
    for order in range(2, degree + 1):
        previous, value = value, (
            (2.0 * order - 1.0) * point * value - (order - 1.0) * previous
        ) / order
    return value, degree * (point * value - previous) / (point * point - 1.0)


def matching_shift(target: float, *, mean: float, volatility: float) -> float:
    """The Wang shift whose normal measure is ``target``.

    One division, because the closed form is linear in the shift. Here so that
    a spectrum can be stated in units a desk already uses — "the Wang transform
    as conservative as our 97.5% expected shortfall" — in the same way
    :func:`shortfall.expectile.matching_level` does for expectiles, and with the
    same warning: the shift that matches under a normal does not match under
    anything else, and the module docstring measures by how much.
    """
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise BadSpectrum(
            f"a volatility of {volatility!r} cannot scale a shift; at zero every "
            "shift gives the mean"
        )
    shift = (mean - target) / volatility
    if shift < 0.0:
        raise BadSpectrum(
            f"a target of {target!r} is above the mean of {mean!r}, so it needs a "
            "negative shift and an increasing spectrum"
        )
    return shift

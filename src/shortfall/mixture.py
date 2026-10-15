"""Regimes: a loss drawn from a mixture, and the risk that is not a mixture of risks.

Every distribution in :mod:`shortfall.parametric` has one mode. A risk
committee's description of its own book has two: a calm state it is in most of
the time, and a stressed one it is in occasionally, with different volatility
and usually a different mean. A finite mixture is the smallest model that can
say that, and it is also the model whose arithmetic is most reliably done
wrong, because the word "mixture" invites averaging and almost nothing here
averages.

The running example throughout is the smallest interesting one: a stressed
regime happening **5% of the time at four times the volatility**, zero mean in
both. Its variance is 1.75 and its excess kurtosis is **10.47** — fat-tailed by
any measure, from two normals.

## The expected shortfall of a mixture is not the mixture of expected shortfalls

The natural thing to write is ``sum(w_i * ES_i)``: the risk in each regime,
weighted by how often the regime happens. :func:`averaged_shortfall` computes
that, and it is wrong, because it answers a different question — what the loss
would be if a regime were drawn first and *then* a bad day inside it. The tail
of a mixture is not drawn that way. Past any interesting threshold it is
almost entirely the stressed regime, whatever that regime's weight.

| confidence | value at risk | expected shortfall | averaged | error | stressed share of the tail |
|---|---|---|---|---|---|
| 90% | 1.366 | 2.244 | 2.018 | -10.1% | 18.3% |
| 95% | 1.806 | 2.926 | 2.372 | -18.9% | 32.6% |
| 97.5% | 2.279 | 3.844 | 2.688 | -30.1% | 56.9% |
| 99% | 3.445 | 5.607 | 3.065 | -45.3% | 97.3% |
| 99.9% | 8.215 | 9.684 | 3.872 | -60.0% | 100.0% |

The error is one-signed and it grows with the confidence level, so it is worst
exactly where the number is used. At 99% the averaged figure is **45% low**,
and the reason is in the last column: a regime that happens 5% of the time is
responsible for 97.3% of the losses past the 99th percentile.
:func:`responsibilities` is that column, and it is the number a regime report
should carry and almost never does.

## It is a fat tail where you measure it and a normal one in the limit

A mixture of normals has every moment, and far enough out the widest component
dominates, so the mixture's tail is ultimately a normal tail with that
component's scale. The ratio of expected shortfall to value at risk is scale
free for a zero-mean normal — the same number for every ``sigma`` — so the
mixture's excess over it is the whole of the fat-tail signal:

| confidence | mixture | any normal | excess |
|---|---|---|---|
| 99% | 1.6278 | 1.1457 | 42.1% |
| 1 - 1e-4 | 1.1014 | 1.0644 | 3.5% |
| 1 - 1e-6 | 1.0537 | 1.0410 | 1.2% |
| 1 - 1e-12 | 1.0220 | 1.0195 | 0.2% |

So the model is fat-tailed over the range anybody quotes and asymptotically
thin, and an argument that reaches for its limiting behaviour is answering a
question about a part of the distribution no capital rule has ever referred to.

## Value at risk is not subadditive, and the window where it fails is narrow

The textbook counterexample is two independent positions that are usually fine
and occasionally lose a great deal. Mixtures state it exactly — the sum of
independent mixtures is a mixture, by :func:`convolve`, with no simulation
anywhere — so the violation is a fact about the distributions rather than
about a seed. Two copies of a position losing 100 with probability 0.9% and
nothing otherwise, at 99%: each has a value at risk of **0.031** and the pair
has **99.998**, a gap of **+99.94** from diversifying.

What is worth more than the example is the boundary. Sweeping the stressed
weight and bisecting, the violation holds for weights in **(0.004943,
0.010000)** and nowhere else. The upper edge is exactly the tail probability
``1 - c``: above it each position's own value at risk is already in the
stressed region and there is nothing left to discover. The lower edge is where
the pair's chance of at least one stress crosses the same threshold,
``1 - sqrt(c) = 0.0050126``, displaced slightly by the components having width.
So the failure is not generic and it is not rare: it is a window of width half
the tail probability, sitting immediately below it.

## Over a horizon the square-root rule is conservative, not optimistic

Returns drawn under a two-state Markov chain are conditionally normal given
the regime sequence, and the sum depends on that sequence only through how
many periods fell in each state. So the horizon law is an exact mixture with
``periods + 1`` components rather than ``2 ** periods``, and
:func:`occupation_law` accumulates the weights in quadratic time.

With a chain staying calm 98% of the time and stressed 90%, giving a 1/6
stationary stressed share, the horizon value at risk against what the
square-root rule projects from one period:

| periods | 99% confidence | 90% confidence |
|---|---|---|
| 5 | 0.958 | 1.062 |
| 10 | 0.924 | 1.121 |
| 60 | 0.797 | 1.349 |

**The rule's direction depends on the confidence level.** In the far tail it
overstates, by 20% at sixty periods, because aggregation diversifies the
regime itself: the excess kurtosis falls from 7.65 at one period to 1.74 at
sixty, and the one-period tail the rule extrapolates from is not there any
more. In the body it understates, for the same reason running the other way.
The crossing is at **97.41%** confidence at five periods, 97.32% at ten and
97.00% at sixty — a level the usual warning about square-root scaling does not
mention, and close to the 97.5% at which expected shortfall is conventionally
reported.

How much the rule is wrong by is a statement about the chain's persistence
rather than about the regimes: at ten periods and 99%, a chain staying calm
99.9% of the time gives 0.995, one staying calm 98% gives 0.924 and one
staying calm 90% gives 0.792.

## Fitting: the likelihood has no maximum

:func:`fit_mixture` carries a variance floor, and it is part of the estimator
rather than a safeguard. The likelihood of a normal mixture is **unbounded**:
park one component's mean on an observation and shrink its variance, and the
density at that point diverges. On forty standard normal draws, holding a 2.5%
component on the first observation, the log-likelihood goes -55.48, -54.87,
-50.92, -41.71, **-32.50** as that component's deviation goes 1, 1e-2, 1e-4,
1e-8, 1e-12. There is no maximum to find.

An implementation without a floor does not fail loudly. Fitting those same
forty observations with nine components walks straight into the spike, and the
giveaway is that **the answer depends on the floor**: a log-likelihood of
-17.80 at a floor of 1e-8 and -3.99 at 1e-14, from the same data and the same
code. :attr:`Fitted.degenerate` reports it, and it is False on every fit here
that converged to something real — two components on the same forty points
gives -50.44 with a smallest deviation of 0.177.

Where the fit is identified it works: 20,000 draws from an 85/15 mixture
recover a 99% expected shortfall within **1.1%**, and 200 draws within 7.1%.
Starting points are sample quantiles rather than random, so the same data
gives the same answer twice.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .distributions import normal_cdf, normal_pdf, normal_ppf

__all__ = [
    "BadMixture",
    "Component",
    "Fitted",
    "MixtureRisk",
    "NormalMixture",
    "Occupation",
    "averaged_shortfall",
    "convolve",
    "fit_mixture",
    "horizon_mixture",
    "mixture_risk",
    "occupation_law",
    "responsibilities",
    "subadditivity_gap",
    "tail_ratio",
]

#: Weights below this are treated as zero when a mixture is built, so that a
#: convolution of many mixtures does not grow a tail of components that cannot
#: affect any number computed from it.
NEGLIGIBLE = 1e-15

#: The smallest standard deviation :func:`fit_mixture` will let a component
#: take. Without a floor the likelihood is unbounded — see the module
#: docstring — and the fit runs away to it from perfectly ordinary data.
DEFAULT_FLOOR = 1e-8


class BadMixture(ValueError):
    """A mixture that does not describe a distribution."""


@dataclass(frozen=True)
class Component:
    """One regime: how often it happens, and what it looks like when it does.

    In the package's sign convention the mean is a *loss*, so a calm regime
    has a mean near zero or below it and a stressed one has a positive mean.
    """

    weight: float
    mean: float
    deviation: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise BadMixture(
                f"a component's weight is a probability, got {self.weight!r}"
            )
        if not math.isfinite(self.mean):
            raise BadMixture(f"a component's mean must be finite, got {self.mean!r}")
        if not math.isfinite(self.deviation) or self.deviation <= 0.0:
            raise BadMixture(
                f"a component's standard deviation is positive, got "
                f"{self.deviation!r}"
            )

    def cdf(self, loss: float) -> float:
        return normal_cdf((loss - self.mean) / self.deviation)

    def survival(self, loss: float) -> float:
        """``P(L > loss)``, computed from the far side so the tail is exact."""
        return normal_cdf((self.mean - loss) / self.deviation)

    def pdf(self, loss: float) -> float:
        return normal_pdf((loss - self.mean) / self.deviation) / self.deviation

    def partial_mean(self, loss: float) -> float:
        """``E[L 1{L > loss}]``, in closed form.

        ``mu Phibar(z) + sigma phi(z)`` with ``z`` the standardised threshold.
        Not an expectation: it is the expectation times the probability, which
        is the form that adds across components.
        """
        standardised = (loss - self.mean) / self.deviation
        return self.mean * normal_cdf(-standardised) + self.deviation * normal_pdf(
            standardised
        )


@dataclass(frozen=True)
class NormalMixture:
    """A finite mixture of normals, on losses.

    The weights are normalised on construction rather than merely checked, so
    a caller can pass unnormalised ones; what is refused is a set that cannot
    be normalised at all.
    """

    components: tuple[Component, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise BadMixture("a mixture needs at least one component")
        total = math.fsum(one.weight for one in self.components)
        if total <= 0.0:
            raise BadMixture(
                f"the component weights sum to {total!r}, so there is no "
                "distribution to normalise them into"
            )
        if abs(total - 1.0) > 1e-12:
            object.__setattr__(
                self,
                "components",
                tuple(
                    Component(one.weight / total, one.mean, one.deviation)
                    for one in self.components
                ),
            )

    @classmethod
    def of(cls, parts: Iterable[tuple[float, float, float]]) -> NormalMixture:
        """Build from ``(weight, mean, deviation)`` triples."""
        return cls(tuple(Component(*part) for part in parts))

    @classmethod
    def two_regime(
        cls,
        *,
        stress: float,
        calm_mean: float = 0.0,
        calm_deviation: float = 1.0,
        stress_mean: float = 0.0,
        stress_deviation: float = 1.0,
    ) -> NormalMixture:
        """The shape the module is about: one calm regime and one stressed one.

        ``stress`` is how often the stressed regime happens. The components
        come out calm first, so :func:`responsibilities` reads in that order.
        """
        if not 0.0 <= stress <= 1.0:
            raise BadMixture(
                f"the stressed regime's weight is a probability, got {stress!r}"
            )
        parts = []
        if stress < 1.0:
            parts.append((1.0 - stress, calm_mean, calm_deviation))
        if stress > 0.0:
            parts.append((stress, stress_mean, stress_deviation))
        return cls.of(parts)

    # -- moments --------------------------------------------------------------

    @property
    def mean(self) -> float:
        return math.fsum(one.weight * one.mean for one in self.components)

    @property
    def variance(self) -> float:
        """The within-component variance plus the between-component one."""
        centre = self.mean
        return math.fsum(
            one.weight * (one.deviation**2 + (one.mean - centre) ** 2)
            for one in self.components
        )

    @property
    def deviation(self) -> float:
        return math.sqrt(self.variance)

    def central_moment(self, order: int) -> float:
        """The ``order``-th central moment, from the normal's own moments.

        A normal's central moments are ``sigma^k (k-1)!!`` for even ``k`` and
        zero for odd, so a component's moment about the *mixture's* mean is a
        binomial expansion of those around the displacement. Summing that over
        components is exact and needs no integration.
        """
        if order < 0:
            raise BadMixture(f"a moment's order is not negative, got {order!r}")
        centre = self.mean
        total = 0.0
        for one in self.components:
            shift = one.mean - centre
            inner = 0.0
            for term in range(order + 1):
                moment = _normal_central_moment(term, one.deviation)
                if moment == 0.0:
                    continue
                inner += math.comb(order, term) * moment * shift ** (order - term)
            total += one.weight * inner
        return total

    @property
    def skewness(self) -> float:
        return self.central_moment(3) / math.pow(self.variance, 1.5)

    @property
    def excess_kurtosis(self) -> float:
        """Finite, always. A mixture of normals has every moment."""
        return self.central_moment(4) / self.variance**2 - 3.0

    # -- the law --------------------------------------------------------------

    def cdf(self, loss: float) -> float:
        return math.fsum(one.weight * one.cdf(loss) for one in self.components)

    def survival(self, loss: float) -> float:
        """``P(L > loss)``, built from each component's own far tail."""
        return math.fsum(one.weight * one.survival(loss) for one in self.components)

    def pdf(self, loss: float) -> float:
        return math.fsum(one.weight * one.pdf(loss) for one in self.components)

    def partial_mean(self, loss: float) -> float:
        """``E[L 1{L > loss}]``, exactly, as a sum of the components'."""
        return math.fsum(one.weight * one.partial_mean(loss) for one in self.components)

    def quantile(self, probability: float) -> float:
        """The loss ``probability`` of the distribution falls below.

        Bracketed from the components' own quantiles — the mixture's lies
        between the smallest and the largest of them, which is a bound rather
        than a guess — and then solved by a safeguarded Newton step. The
        density is available in closed form, so there is no reason to bisect
        all the way, and there is also no reason to trust Newton alone on a
        bimodal distribution where the density between the modes is small.
        """
        if not 0.0 < probability < 1.0:
            raise BadMixture(
                f"a quantile is taken at a probability strictly between zero "
                f"and one, got {probability!r}"
            )
        edge = normal_ppf(probability)
        low = min(one.mean + edge * one.deviation for one in self.components)
        high = max(one.mean + edge * one.deviation for one in self.components)
        if high - low < 1e-15:
            return 0.5 * (low + high)
        guess = 0.5 * (low + high)
        for _ in range(200):
            value = self.cdf(guess) - probability
            if value > 0.0:
                high = guess
            else:
                low = guess
            density = self.pdf(guess)
            step = guess - value / density if density > 0.0 else 0.5 * (low + high)
            guess = step if low < step < high else 0.5 * (low + high)
            if high - low <= 1e-15 * max(1.0, abs(guess)):
                break
        return guess

    def shifted(self, by: float) -> NormalMixture:
        """The same mixture with every mean moved."""
        return NormalMixture(
            tuple(
                Component(one.weight, one.mean + by, one.deviation)
                for one in self.components
            )
        )

    def scaled(self, by: float) -> NormalMixture:
        """The same mixture with every component scaled about zero."""
        if by <= 0.0:
            raise BadMixture(f"a mixture is scaled by a positive factor, got {by!r}")
        return NormalMixture(
            tuple(
                Component(one.weight, one.mean * by, one.deviation * by)
                for one in self.components
            )
        )


def _normal_central_moment(order: int, deviation: float) -> float:
    if order % 2 == 1:
        return 0.0
    half = order // 2
    double_factorial = 1.0
    for step in range(1, half + 1):
        double_factorial *= 2 * step - 1
    return deviation**order * double_factorial


# -- risk ---------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureRisk:
    """Value at risk and expected shortfall for a mixture, with its workings."""

    confidence: float
    value_at_risk: float
    expected_shortfall: float
    #: ``P(regime i | L > VaR)``, in the mixture's component order.
    responsibility: tuple[float, ...]
    #: The weighted average of the components' own expected shortfalls at the
    #: same confidence level, which is the number people reach for and is not
    #: this one.
    averaged: float

    @property
    def tail_probability(self) -> float:
        return 1.0 - self.confidence

    @property
    def averaging_error(self) -> float:
        """How far the averaged number is from the right one, as a fraction."""
        return self.averaged / self.expected_shortfall - 1.0

    @property
    def tail_ratio(self) -> float:
        """Expected shortfall over value at risk. One for a point mass."""
        return self.expected_shortfall / self.value_at_risk


def mixture_risk(mixture: NormalMixture, confidence: float) -> MixtureRisk:
    """Value at risk and expected shortfall, both exact.

    Only the quantile is solved for. Given it, the expected shortfall is
    ``E[L 1{L > q}] / P(L > q)`` and both halves are closed form, so the tail
    average inherits no error from a quadrature that was never run.
    """
    if not 0.0 < confidence < 1.0:
        raise BadMixture(
            f"a confidence level is strictly between zero and one, got "
            f"{confidence!r}"
        )
    quantile = mixture.quantile(confidence)
    tail = mixture.survival(quantile)
    if tail <= 0.0:
        raise BadMixture(
            f"the mixture puts no probability above its own {confidence!r} "
            "quantile, so there is no tail to average"
        )
    shortfall = mixture.partial_mean(quantile) / tail
    return MixtureRisk(
        confidence=confidence,
        value_at_risk=quantile,
        expected_shortfall=shortfall,
        responsibility=responsibilities(mixture, quantile),
        averaged=averaged_shortfall(mixture, confidence),
    )


def responsibilities(mixture: NormalMixture, loss: float) -> tuple[float, ...]:
    """``P(component i | L > loss)``: which regime a tail loss came from.

    The quantity a regime report should carry and almost never does. A 5%
    regime is 5% of the days and a much larger share of the losses past any
    interesting threshold, and how much larger is not something anybody
    estimates by eye.
    """
    parts = [one.weight * one.survival(loss) for one in mixture.components]
    total = math.fsum(parts)
    if total <= 0.0:
        raise BadMixture(
            f"no component puts any probability above {loss!r}, so the loss "
            "cannot be attributed"
        )
    return tuple(part / total for part in parts)


def averaged_shortfall(mixture: NormalMixture, confidence: float) -> float:
    """The weighted average of the components' expected shortfalls.

    Wrong, and worth computing for that reason. Each component's own expected
    shortfall at ``confidence`` is taken at *its* quantile, not the mixture's,
    so the average answers a question nobody asked: what the loss would be if
    a regime were drawn first and then a bad day inside it. The tail of a
    mixture is not drawn that way — it is overwhelmingly the stressed regime —
    so the average understates whenever the regimes differ.
    """
    if not 0.0 < confidence < 1.0:
        raise BadMixture(
            f"a confidence level is strictly between zero and one, got "
            f"{confidence!r}"
        )
    edge = normal_ppf(confidence)
    total = 0.0
    for one in mixture.components:
        own = one.mean + edge * one.deviation
        total += one.weight * (one.partial_mean(own) / (1.0 - confidence))
    return total


def tail_ratio(mixture: NormalMixture, confidence: float) -> float:
    """Expected shortfall over value at risk at one level."""
    risk = mixture_risk(mixture, confidence)
    return risk.tail_ratio


# -- sums ---------------------------------------------------------------------


def convolve(
    left: NormalMixture, right: NormalMixture, *, negligible: float = NEGLIGIBLE
) -> NormalMixture:
    """The law of the sum of two independent mixtures, exactly.

    Mixtures of normals are closed under convolution: every pairing of
    components contributes one normal whose weight is the product, whose mean
    is the sum and whose variance is the sum. The result has the product of
    the component counts, which is why the weight floor exists.
    """
    parts: list[Component] = []
    for one in left.components:
        for other in right.components:
            weight = one.weight * other.weight
            if weight < negligible:
                continue
            parts.append(
                Component(
                    weight,
                    one.mean + other.mean,
                    math.hypot(one.deviation, other.deviation),
                )
            )
    if not parts:
        raise BadMixture(
            "every pairing fell below the weight floor, which cannot happen "
            "for two normalised mixtures and means one of them was not"
        )
    return NormalMixture(tuple(parts))


def subadditivity_gap(
    left: NormalMixture, right: NormalMixture, confidence: float
) -> float:
    """``VaR(L + R) - VaR(L) - VaR(R)`` for independent ``L`` and ``R``.

    Positive is a violation: diversification made the reported risk worse.
    Computed from the exact convolution rather than from a simulation, so the
    sign is a fact about the distributions and not about a seed.
    """
    joint = convolve(left, right)
    return (
        joint.quantile(confidence)
        - left.quantile(confidence)
        - right.quantile(confidence)
    )


# -- fitting ------------------------------------------------------------------


@dataclass(frozen=True)
class Fitted:
    """A fitted mixture, with what the fit did."""

    mixture: NormalMixture
    log_likelihood: float
    iterations: int
    converged: bool
    #: True when a component hit the variance floor, which means the
    #: likelihood was running away rather than converging.
    degenerate: bool


def fit_mixture(
    sample: Sequence[float],
    components: int = 2,
    *,
    floor: float = DEFAULT_FLOOR,
    tolerance: float = 1e-10,
    iterations: int = 500,
) -> Fitted:
    """Fit by expectation maximisation, with the floor the likelihood needs.

    The likelihood of a normal mixture is **unbounded**: put one component's
    mean on an observation and let its variance go to zero and the density at
    that point diverges, taking the likelihood with it. There is no maximum to
    find, only a good local one, and an implementation with no floor does not
    fail loudly — it reports a component of weight ``1/n`` sitting on a single
    point and a log-likelihood that looks excellent.

    The floor is therefore part of the estimator rather than a safeguard
    bolted on, and :attr:`Fitted.degenerate` says when it bound.

    Starting points are quantiles of the sample, which is deterministic. A
    random start would make the same data give different answers on different
    days, and the local optima here are far enough apart for that to matter.
    """
    data = list(sample)
    if components < 1:
        raise BadMixture(f"a mixture needs at least one component, got {components!r}")
    if len(data) < 2 * components:
        raise BadMixture(
            f"{len(data)} observations cannot identify {components} components; "
            f"at least {2 * components} are needed"
        )
    if any(not math.isfinite(one) for one in data):
        raise BadMixture("the sample contains a value that is not finite")
    if floor <= 0.0:
        raise BadMixture(f"the variance floor is positive, got {floor!r}")

    ordered = sorted(data)
    spread = _sample_deviation(data)
    if spread <= 0.0:
        raise BadMixture(
            "every observation is identical, so no mixture of normals is "
            "identified by it"
        )
    weights = [1.0 / components] * components
    means = [
        ordered[min(len(ordered) - 1, int((index + 0.5) / components * len(ordered)))]
        for index in range(components)
    ]
    deviations = [max(spread, floor)] * components

    previous = -math.inf
    converged = False
    taken = 0
    while taken < iterations:
        taken += 1
        responsibility: list[list[float]] = []
        total_log = 0.0
        for point in data:
            densities = [
                weights[index]
                * normal_pdf((point - means[index]) / deviations[index])
                / deviations[index]
                for index in range(components)
            ]
            mass = math.fsum(densities)
            if mass <= 0.0:
                # Every component is astronomically far from this point. The
                # nearest one takes it whole, which is the limit of the ratio.
                nearest = min(
                    range(components),
                    key=lambda index: abs(point - means[index]) / deviations[index],
                )
                share = [0.0] * components
                share[nearest] = 1.0
                responsibility.append(share)
                total_log += -math.inf
                continue
            responsibility.append([one / mass for one in densities])
            total_log += math.log(mass)
        for index in range(components):
            assigned = math.fsum(row[index] for row in responsibility)
            if assigned <= 0.0:
                continue
            weights[index] = assigned / len(data)
            means[index] = (
                math.fsum(
                    row[index] * point
                    for row, point in zip(responsibility, data, strict=True)
                )
                / assigned
            )
            variance = (
                math.fsum(
                    row[index] * (point - means[index]) ** 2
                    for row, point in zip(responsibility, data, strict=True)
                )
                / assigned
            )
            deviations[index] = max(math.sqrt(max(variance, 0.0)), floor)
        if math.isfinite(total_log) and abs(total_log - previous) <= tolerance * (
            1.0 + abs(total_log)
        ):
            converged = True
            previous = total_log
            break
        previous = total_log

    fitted = NormalMixture.of(zip(weights, means, deviations, strict=True))
    return Fitted(
        mixture=fitted,
        log_likelihood=_log_likelihood(fitted, data),
        iterations=taken,
        converged=converged,
        degenerate=any(one.deviation <= floor * (1.0 + 1e-12) for one in fitted.components),
    )


def _sample_deviation(data: Sequence[float]) -> float:
    if len(data) < 2:
        return 0.0
    centre = math.fsum(data) / len(data)
    return math.sqrt(math.fsum((one - centre) ** 2 for one in data) / (len(data) - 1))


def _log_likelihood(mixture: NormalMixture, data: Sequence[float]) -> float:
    total = 0.0
    for point in data:
        density = mixture.pdf(point)
        total += math.log(density) if density > 0.0 else -math.inf
    return total


# -- horizons -----------------------------------------------------------------


@dataclass(frozen=True)
class Occupation:
    """How many of ``periods`` are spent in the first of two regimes."""

    periods: int
    #: ``probability[k]`` is ``P(exactly k periods in regime 0)``.
    probability: tuple[float, ...]

    @property
    def expected(self) -> float:
        return math.fsum(
            count * mass for count, mass in enumerate(self.probability)
        )


def occupation_law(
    persistence: tuple[float, float], start: tuple[float, float], periods: int
) -> Occupation:
    """The law of the time spent in regime 0, over ``periods`` draws.

    ``persistence`` is ``(P(stay in 0), P(stay in 1))`` and ``start`` is the
    distribution of the first period's regime. There are ``2 ** periods``
    paths and the answer depends only on the occupation count, so the count is
    accumulated directly: a forward recursion over ``(period, count, state)``,
    which is quadratic in the horizon rather than exponential.
    """
    if periods < 1:
        raise BadMixture(f"a horizon is at least one period, got {periods!r}")
    for value in (*persistence, *start):
        if not 0.0 <= value <= 1.0:
            raise BadMixture(f"a transition probability is in [0, 1], got {value!r}")
    if abs(math.fsum(start) - 1.0) > 1e-12:
        raise BadMixture(f"the starting distribution sums to {math.fsum(start)!r}")

    # table[count][state] after the periods drawn so far.
    table = [[0.0, 0.0] for _ in range(periods + 1)]
    table[1][0] = start[0]
    table[0][1] = start[1]
    for _ in range(periods - 1):
        nxt = [[0.0, 0.0] for _ in range(periods + 1)]
        for count in range(periods + 1):
            for state in (0, 1):
                mass = table[count][state]
                if mass == 0.0:
                    continue
                stay = persistence[state]
                to_zero = stay if state == 0 else 1.0 - stay
                if to_zero > 0.0 and count + 1 <= periods:
                    nxt[count + 1][0] += mass * to_zero
                if to_zero < 1.0:
                    nxt[count][1] += mass * (1.0 - to_zero)
        table = nxt
    return Occupation(
        periods=periods,
        probability=tuple(table[count][0] + table[count][1] for count in range(periods + 1)),
    )


def horizon_mixture(
    calm: Component,
    stressed: Component,
    persistence: tuple[float, float],
    start: tuple[float, float],
    periods: int,
) -> NormalMixture:
    """The law of a sum of ``periods`` returns drawn under a regime chain.

    Conditional on the sequence of regimes the sum is normal, and it depends
    on the sequence only through how many periods fell in each regime. So the
    horizon law is an exact mixture with ``periods + 1`` components rather
    than ``2 ** periods``, weighted by :func:`occupation_law`.

    The component weights passed in are ignored: which regime is in force is
    the chain's business, not theirs.
    """
    law = occupation_law(persistence, start, periods)
    parts: list[Component] = []
    for count, mass in enumerate(law.probability):
        if mass < NEGLIGIBLE:
            continue
        others = periods - count
        mean = count * calm.mean + others * stressed.mean
        variance = count * calm.deviation**2 + others * stressed.deviation**2
        parts.append(Component(mass, mean, math.sqrt(variance)))
    if not parts:
        raise BadMixture("the chain reaches no occupation count with any probability")
    return NormalMixture(tuple(parts))

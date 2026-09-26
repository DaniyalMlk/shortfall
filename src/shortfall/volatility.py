"""A conditional volatility model, and the optimiser that fits it.

:func:`~shortfall.backtest.independence` exists to detect a model whose
breaches cluster. When it rejects, the answer is not to rescale anything — it
is that the model has no notion of volatility changing, and needs one. This
module is that notion.

What was here already is :func:`~shortfall.historical.ewma_volatility`, at a
decay of 0.94. That is a filter rather than a model, and the difference matters
in three places. Its decay is assumed rather than estimated. It has no long-run
level, so a shock never decays towards anything. And because it has no long-run
level it cannot forecast: the exponentially weighted forecast at every horizon
is today's estimate, which says a market that is calm today will be exactly as
calm in six months.

GARCH(1,1) adds the missing term::

    variance[t] = omega + alpha * residual[t-1]^2 + beta * variance[t-1]

``alpha`` is how hard yesterday's surprise hits, ``beta`` is how much of
yesterday's level carries over, and ``omega`` sets the level the process
returns to. Their sum is the persistence, and it has to be below one for the
unconditional variance to exist at all. EWMA is the boundary case
``omega = 0``, ``alpha + beta = 1`` — a process with infinite unconditional
variance, which is why it cannot mean-revert.

The fit is by maximum likelihood over a parameterisation that *cannot leave*
the admissible region, rather than by a penalty applied after the optimiser has
already wandered out of it. See :func:`_from_free` for why that choice is not
cosmetic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from .distributions import chi_square_sf, standardised_t_log_pdf
from .parametric import Risk, normal_risk, student_t_risk
from .series import ReturnSeries, TooShort


class Innovation(str, Enum):
    """The distribution the standardised residuals are assumed to follow.

    This is a separate question from the variance recursion, and mixing the two
    up is the mistake the option exists to prevent. The recursion says how the
    *scale* moves; the innovation says what shape is drawn at that scale. Fit a
    fat-tailed series under ``NORMAL`` and the variance path still comes out
    about right — the breaches cluster no worse than they should — while every
    quantile taken from it is too close in, because the 99% point of a
    standardised residual is being read off as 2.326 whatever the residuals
    actually look like.
    """

    NORMAL = "normal"
    STUDENT_T = "student-t"

#: Fewest observations a fit will attempt. Three parameters from fewer than
#: this is not an estimate; the likelihood surface is nearly flat along the
#: persistence direction and the optimiser reports whatever it started near.
MIN_OBSERVATIONS: Final = 100

#: Largest persistence the parameterisation will produce. Exactly one is
#: inadmissible — the unconditional variance is undefined there — and the
#: transform approaches it asymptotically, so this caps it away from the
#: boundary where the long-run variance and the half-life both blow up.
MAX_PERSISTENCE: Final = 0.999_9

#: Fewest degrees of freedom the transform will produce. Below two there is no
#: variance for the density to be standardised against, and the likelihood of a
#: series with a genuinely infinite variance is not something this model should
#: report a fit for. Kept clear of the boundary, where the fourth moment is also
#: gone and the surface is very steep.
MIN_DEGREES: Final = 2.1

#: Most degrees of freedom the transform will produce. Not a modelling claim: at
#: this many the density is within 1e-4 of the normal everywhere inside three
#: standard deviations, so the likelihood is flat above it and any number the
#: optimiser returns there is noise. Hitting the cap is reported rather than
#: passed off as an estimate — see :attr:`Garch.degrees_identified`.
MAX_DEGREES: Final = 1_000.0

#: Above this the estimate is called unidentified. Chosen from the same measured
#: fact: the standardised-t density departs from the normal at order ``1/v``, so
#: at 200 the departure is under a percent even in the tail, and the likelihood
#: cannot separate 200 from 400 on any sample this model is fitted to.
IDENTIFIED_DEGREES: Final = 200.0

_LOG_2PI: Final = math.log(2.0 * math.pi)


class DidNotConverge(RuntimeError):
    """The likelihood optimiser ran out of iterations."""


@dataclass(frozen=True)
class Garch:
    """A fitted GARCH(1,1), and everything needed to use or doubt it."""

    omega: float
    alpha: float
    beta: float
    #: The constant mean removed before fitting. Estimated as the sample mean
    #: rather than jointly, which is standard and costs almost nothing: the
    #: mean and the variance parameters are close to orthogonal in this model,
    #: and estimating a mean jointly with a volatility process on daily data
    #: mostly adds a badly determined parameter.
    mean: float
    #: Conditional variance at each observation, formed from returns strictly
    #: before it. Same off-by-one as :func:`ewma_volatility`, and for the same
    #: reason: a variance that had seen its own return would divide each
    #: residual by something that knew about it.
    variances: tuple[float, ...]
    log_likelihood: float
    observations: int
    iterations: int
    converged: bool
    #: True when the long-run level was fixed to the sample variance rather
    #: than estimated.
    variance_targeted: bool
    #: The shape assumed for the standardised residuals.
    innovation: Innovation = Innovation.NORMAL
    #: Estimated degrees of freedom, or ``None`` under normal innovations.
    degrees_of_freedom: float | None = field(default=None)

    @property
    def degrees_identified(self) -> bool:
        """Whether the degrees of freedom mean anything.

        False under normal innovations, and false when the estimate came back
        above :data:`IDENTIFIED_DEGREES`, where the likelihood is flat: the data
        did not find a fat tail, and the number the optimiser stopped at is not
        evidence of where the tail is. A caller reading
        :attr:`degrees_of_freedom` without checking this can report "our fitted
        tail index is 640" about a series that is simply Gaussian.
        """
        return self.degrees_of_freedom is not None and self.degrees_of_freedom < IDENTIFIED_DEGREES

    @property
    def implied_excess_kurtosis(self) -> float:
        """Excess kurtosis of the *innovations*, not of the returns.

        ``6 / (v - 4)`` for a standardised Student-t, zero under normal
        innovations, and infinite at or below four degrees of freedom, where the
        fourth moment does not exist. The returns have more than this: a GARCH
        process mixes variances, so even Gaussian innovations produce unconditional
        excess kurtosis. Reading this as the kurtosis of the series would double
        count that.
        """
        degrees = self.degrees_of_freedom
        if degrees is None:
            return 0.0
        if degrees <= 4.0:
            return math.inf
        return 6.0 / (degrees - 4.0)

    def risk(self, *, confidence: float = 0.99, last_return: float | None = None) -> Risk:
        """One-period value at risk and expected shortfall, at the current level.

        Closed form in the innovation distribution, at the conditional
        volatility for the next period, with the fitted mean. Under Student-t
        innovations the quantile is the standardised-t quantile, which is the
        whole point of fitting them: the same volatility with a normal quantile
        gives a 99% figure smaller by a factor that grows as the tail fattens.

        One period only, deliberately. Over ``h`` periods the sum of the
        innovations is not a scaled member of the same family — for the
        Student-t it is not a Student-t at all, and even under normal
        innovations it is a variance mixture rather than a normal. What the
        model does give at a horizon is the *variance*, through
        :meth:`horizon_variance`; turning that into a quantile needs the
        distribution of the sum, not an assumption about it.
        """
        variance = (
            self.variances[-1] if last_return is None else self.next_variance(last_return)
        )
        volatility = math.sqrt(variance)
        degrees = self.degrees_of_freedom
        if self.innovation is Innovation.STUDENT_T and degrees is not None:
            return student_t_risk(
                mean=self.mean,
                volatility=volatility,
                confidence=confidence,
                degrees=degrees,
            )
        return normal_risk(mean=self.mean, volatility=volatility, confidence=confidence)

    @property
    def persistence(self) -> float:
        """``alpha + beta``. How much of a shock is still there tomorrow."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """The level the process returns to, ``omega / (1 - persistence)``."""
        return self.omega / (1.0 - self.persistence)

    @property
    def long_run_volatility(self) -> float:
        return math.sqrt(self.long_run_variance)

    @property
    def half_life(self) -> float:
        """Periods for half of a variance shock to decay.

        ``log(0.5) / log(persistence)``. At a persistence of 0.99 this is 69
        periods — a shock to a market with that persistence is still half
        present three months later, which is the fact that makes
        square-root-of-time wrong over any horizon worth caring about.
        """
        return math.log(0.5) / math.log(self.persistence)

    @property
    def volatilities(self) -> tuple[float, ...]:
        return tuple(math.sqrt(value) for value in self.variances)

    def standardised(self, returns: Sequence[float]) -> tuple[float, ...]:
        """Residuals divided by the volatility forecast for their own period.

        If the model is right these are independent and identically
        distributed. They are the honest way to check the fit: the raw squared
        returns of a financial series are strongly autocorrelated, and the
        whole claim of the model is that it has taken that out. What is left in
        them is what the model missed.
        """
        if len(returns) != self.observations:
            raise ValueError(
                f"{len(returns)} returns against {self.observations} the model was "
                f"fitted on; standardising a different series would divide by "
                "variances belonging to other dates"
            )
        return tuple(
            (value - self.mean) / math.sqrt(variance)
            for value, variance in zip(returns, self.variances, strict=True)
        )

    def next_variance(self, last_return: float) -> float:
        """The variance forecast for the period after the fitted sample ends.

        ``omega + alpha * residual^2 + beta * variances[-1]``, and the
        ``variances[-1]`` is the part worth reading twice: it is the *last*
        conditional variance of the series this model was fitted on, not the one
        before ``last_return``. So this is the forecast for the day after the
        sample, and calling it with some earlier day's return in a walk-forward
        loop mixes that day's surprise with the end of the sample's level.

        It produces a plausible number when misused. In a walk-forward
        measurement here the mistake moved a breach rate from 0.76% to 1.74%
        against a nominal 1% and looked like a finding about the model. To step
        through the sample, index :attr:`volatilities` instead — element ``t`` is
        already the forecast made from returns strictly before ``t``.
        """
        residual = last_return - self.mean
        return self.omega + self.alpha * residual * residual + self.beta * self.variances[-1]

    def forecast(self, steps: int, *, last_return: float | None = None) -> tuple[float, ...]:
        """Variance forecasts for the next ``steps`` periods.

        Past one step ahead the squared residual is unknown, and its
        expectation is the variance itself, so the recursion collapses to a
        geometric decay towards the long-run level::

            E[variance[t+h]] = long_run + persistence^(h-1) * (variance[t+1] - long_run)

        This is the term EWMA does not have. An exponentially weighted forecast
        is flat at every horizon, which understates the risk of a long horizon
        in a calm market and overstates it in a turbulent one — and the second
        error is the expensive one, because it arrives exactly when positions
        are being cut.
        """
        if steps < 1:
            raise ValueError(f"a forecast covers at least one period, got {steps!r}")
        first = (
            self.variances[-1] if last_return is None else self.next_variance(last_return)
        )
        level = self.long_run_variance
        return tuple(
            level + self.persistence**step * (first - level) for step in range(steps)
        )

    def horizon_variance(self, steps: int, *, last_return: float | None = None) -> float:
        """Total variance over ``steps`` periods, summing the path.

        This is what a multi-period value at risk needs, and it is not
        ``steps`` times the one-step variance. The gap has a sign: starting
        above the long-run level, mean reversion makes the true figure
        *smaller* than the square-root-of-time rule says, and starting below it
        makes it larger. Scaling a calm day's volatility to ten days
        understates the risk, which is the direction that costs money.
        """
        return math.fsum(self.forecast(steps, last_return=last_return))

    def scaling_against_square_root_of_time(
        self, steps: int, *, last_return: float | None = None
    ) -> float:
        """Horizon volatility over what square-root-of-time would give.

        Below one when today is more volatile than the long run, above one when
        it is calmer. Exactly one only when today sits at the long-run level.
        """
        path = self.forecast(steps, last_return=last_return)
        return math.sqrt(math.fsum(path) / (steps * path[0]))


def garch_variances(
    returns: Sequence[float],
    *,
    omega: float,
    alpha: float,
    beta: float,
    mean: float = 0.0,
    seed: float | None = None,
) -> list[float]:
    """Run the recursion forward, without fitting anything.

    The first variance is the seed — the sample variance unless one is given.
    That does use the whole sample, which is the standard treatment and is the
    same compromise :func:`ewma_volatility` makes: its influence decays like
    ``persistence^t`` and is gone within a few hundred observations, while
    seeding from the first return alone makes the early estimates wild enough
    to dominate the likelihood.
    """
    if omega <= 0.0:
        raise ValueError(f"omega is positive, got {omega!r}; a zero level has no long run")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError(
            f"alpha and beta are non-negative, got alpha={alpha!r}, beta={beta!r}; "
            "a negative one can drive the conditional variance below zero"
        )
    count = len(returns)
    if count < 2:
        raise TooShort(f"the recursion needs at least 2 observations, got {count}")
    if seed is None:
        seed = math.fsum((value - mean) ** 2 for value in returns) / count
    variance = seed
    out: list[float] = []
    for value in returns:
        out.append(variance)
        residual = value - mean
        variance = omega + alpha * residual * residual + beta * variance
    return out


def _negative_log_likelihood(
    returns: Sequence[float],
    *,
    omega: float,
    alpha: float,
    beta: float,
    mean: float,
    seed: float,
    degrees: float | None = None,
) -> float:
    """Minus the log-likelihood, which is what gets minimised.

    Gaussian when ``degrees`` is ``None``, standardised Student-t otherwise. The
    variance recursion is identical either way — that is the design: the
    innovation enters only through the density evaluated at the standardised
    residual, so the two fits are nested and their likelihoods are directly
    comparable.

    Returns infinity rather than raising when the recursion produces a
    non-positive variance. The optimiser reads that as a wall and walks away
    from it, which is the behaviour wanted — an exception would abort a search
    that was merely passing through a bad point.
    """
    variance = seed
    total = 0.0
    for value in returns:
        if variance <= 0.0 or not math.isfinite(variance):
            return math.inf
        residual = value - mean
        if degrees is None:
            total += 0.5 * (_LOG_2PI + math.log(variance) + residual * residual / variance)
        else:
            # log f(r) = log f_Z(r / sigma) - log sigma, the Jacobian of the
            # scaling. Dropping that half-log-variance term would make every
            # likelihood comparison between parameter values meaningless while
            # still producing a plausible-looking surface.
            total -= standardised_t_log_pdf(
                residual / math.sqrt(variance), degrees
            ) - 0.5 * math.log(variance)
        variance = omega + alpha * residual * residual + beta * variance
    return total


def _sigmoid(x: float) -> float:
    """Numerically safe logistic. Written in the branch that does not overflow."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    scaled = math.exp(x)
    return scaled / (1.0 + scaled)


def _from_free(
    free: Sequence[float], *, target: float | None, innovation: Innovation = Innovation.NORMAL
) -> tuple[float, float, float, float | None]:
    """Map unconstrained coordinates to ``(omega, alpha, beta)``.

    The constraints are that all three are positive and that ``alpha + beta``
    is below one. Enforcing them by transform rather than by penalty is the
    decision that makes this fit reliably.

    A penalty lets the optimiser step to ``alpha + beta = 1.4``, evaluate
    something enormous there, and reflect back. The simplex spends its
    iterations exploring a region where the likelihood is not defined, and on a
    persistent series — which is every financial series — the optimum sits
    close enough to the boundary that it never settles. Here the boundary is at
    infinity in the free coordinates and no step can reach it.

    The persistence is split multiplicatively rather than ``alpha`` and
    ``beta`` being transformed separately, because the constraint couples them:
    ``persistence = sigmoid(free[1])`` is the whole of the binding constraint,
    and ``free[2]`` divides it between the two terms.

    The degrees of freedom, when there are any, take the last coordinate and
    are mapped into ``(MIN_DEGREES, MAX_DEGREES)`` by the same device. The cap
    is not a constraint the model needs — it is there because the likelihood
    above it is flat, and a simplex on a flat surface walks until it runs out of
    iterations and then reports not having converged, which would turn a series
    with ordinary thin tails into a failed fit.
    """
    persistence = MAX_PERSISTENCE * _sigmoid(free[1])
    weight = _sigmoid(free[2])
    alpha = persistence * weight
    beta = persistence - alpha
    # Variance targeting: the long-run level is the sample variance, so omega
    # follows from the persistence rather than being searched over.
    omega = target * (1.0 - persistence) if target is not None else math.exp(free[0])
    degrees = (
        MIN_DEGREES + (MAX_DEGREES - MIN_DEGREES) * _sigmoid(free[3])
        if innovation is Innovation.STUDENT_T
        else None
    )
    return omega, alpha, beta, degrees


def _stretch(
    centroid: Sequence[float], worst: Sequence[float], factor: float
) -> list[float]:
    """A point on the line from the worst vertex through the centroid.

    Taken as a free function rather than a closure over the loop's variables.
    A closure would be correct here — it is called within the same iteration it
    is defined in — and it is the shape that becomes a late-binding bug the
    moment anyone stores one of these for later.
    """
    return [value + factor * (value - other) for value, other in zip(centroid, worst, strict=True)]


def _nelder_mead(
    objective: Callable[[Sequence[float]], float],
    start: Sequence[float],
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 4000,
) -> tuple[list[float], float, int, bool]:
    """Nelder-Mead simplex minimisation.

    A derivative-free method, which suits a likelihood whose gradient would
    have to be written out by hand and whose surface is flat along the
    persistence direction near the optimum — the place a gradient method is
    least reliable and a simplex is merely slow.

    The convergence test is on both the spread of the *values* and the size of
    the *simplex*. Testing only the values stops early on exactly the flat
    ridge this surface has: the vertices can agree to twelve digits while still
    being far apart in the persistence direction, so the parameters are wrong
    and the likelihood says they are right.
    """
    call = objective
    n = len(start)
    simplex = [list(start)]
    for index in range(n):
        point = list(start)
        point[index] += 0.5 if point[index] == 0.0 else 0.05 * abs(point[index])
        simplex.append(point)
    values = [call(point) for point in simplex]

    for iteration in range(1, max_iterations + 1):
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        spread = abs(values[-1] - values[0])
        size = max(
            abs(simplex[i][j] - simplex[0][j]) for i in range(1, n + 1) for j in range(n)
        )
        if spread <= tolerance * (abs(values[0]) + tolerance) and size <= 1e-8:
            return simplex[0], values[0], iteration, True

        centroid = [math.fsum(point[j] for point in simplex[:-1]) / n for j in range(n)]

        worst = simplex[-1]
        reflected = _stretch(centroid, worst, 1.0)
        reflected_value = call(reflected)
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[0]:
            expanded = _stretch(centroid, worst, 2.0)
            expanded_value = call(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        contracted = _stretch(centroid, worst, -0.5)
        contracted_value = call(contracted)
        if contracted_value < values[-1]:
            simplex[-1], values[-1] = contracted, contracted_value
            continue
        # Nothing worked: shrink everything towards the best vertex.
        for index in range(1, n + 1):
            simplex[index] = [
                simplex[0][j] + 0.5 * (simplex[index][j] - simplex[0][j]) for j in range(n)
            ]
            values[index] = call(simplex[index])
    best = min(range(n + 1), key=lambda i: values[i])
    return simplex[best], values[best], max_iterations, False


def fit_garch(
    returns: Sequence[float] | ReturnSeries,
    *,
    variance_targeting: bool = False,
    mean: float | None = None,
    strict: bool = True,
    max_iterations: int = 4000,
    innovation: Innovation = Innovation.NORMAL,
) -> Garch:
    """Fit a GARCH(1,1) by maximum likelihood.

    ``innovation`` chooses the density the standardised residuals are assumed to
    follow. Under :attr:`Innovation.STUDENT_T` the degrees of freedom are
    estimated alongside the variance parameters, which is a fourth coordinate in
    the same search rather than a two-stage fit: estimating the variance
    parameters under a normal likelihood and then fitting a tail to the
    residuals gives parameters that are consistent but not efficient, and on a
    fat-tailed series the normal likelihood over-weights the largest residuals
    badly enough to pull ``alpha`` up.

    ``variance_targeting`` fixes the long-run variance to the sample variance
    and estimates only the two dynamic parameters. It is more robust on short
    samples, for a reason worth stating: ``omega`` is the product of the
    long-run level and one minus the persistence, and on a persistent series
    that second factor is small and badly determined, so ``omega`` is the
    product of a number and a nearly unidentified one. Targeting removes it
    from the search and puts the long-run level where the data plainly says it
    is.

    ``strict`` raises when the optimiser runs out of iterations. Setting it
    false returns the best point found with :attr:`Garch.converged` false,
    which is for a caller fitting many series and wanting to see which failed
    rather than losing the batch.
    """
    values = list(returns.values) if isinstance(returns, ReturnSeries) else list(returns)
    count = len(values)
    if count < MIN_OBSERVATIONS:
        raise TooShort(
            f"a GARCH fit needs at least {MIN_OBSERVATIONS} observations, got {count}. "
            "Below that the likelihood is nearly flat along the persistence direction "
            "and the optimiser reports whatever it started near."
        )
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"observation {index} is {value!r}, which is not finite")

    centre = math.fsum(values) / count if mean is None else mean
    sample_variance = math.fsum((value - centre) ** 2 for value in values) / count
    if sample_variance <= 0.0:
        raise ValueError(
            "the returns have no variance at all, so there is no volatility process "
            "to fit; every observation equals the mean"
        )
    target = sample_variance if variance_targeting else None

    def objective(free: Sequence[float]) -> float:
        omega, alpha, beta, degrees = _from_free(free, target=target, innovation=innovation)
        return _negative_log_likelihood(
            values,
            omega=omega,
            alpha=alpha,
            beta=beta,
            mean=centre,
            seed=sample_variance,
            degrees=degrees,
        )

    # Started from alpha near 0.08 and beta near 0.90, which is where daily
    # equity data lands often enough to be a sensible prior and is far from the
    # boundary the transform protects.
    persistence = 0.98
    weight = 0.08 / persistence
    start = [
        math.log(sample_variance * (1.0 - persistence)),
        math.log(persistence / (MAX_PERSISTENCE - persistence)),
        math.log(weight / (1.0 - weight)),
    ]
    if innovation is Innovation.STUDENT_T:
        # Started at eight degrees of freedom: fat enough that the likelihood
        # has a gradient towards either answer, and well inside the identified
        # region. Starting near the cap starts on the flat part.
        share = (8.0 - MIN_DEGREES) / (MAX_DEGREES - MIN_DEGREES)
        start.append(math.log(share / (1.0 - share)))
    best, negative, iterations, converged = _nelder_mead(
        objective, start, max_iterations=max_iterations
    )
    if strict and not converged:
        raise DidNotConverge(
            f"the likelihood optimiser did not converge in {iterations} iterations. "
            "Pass strict=False to receive the best point found, or try "
            "variance_targeting=True, which removes the worst-determined parameter "
            "from the search."
        )

    omega, alpha, beta, degrees = _from_free(best, target=target, innovation=innovation)
    return Garch(
        omega=omega,
        alpha=alpha,
        beta=beta,
        mean=centre,
        variances=tuple(
            garch_variances(
                values, omega=omega, alpha=alpha, beta=beta, mean=centre, seed=sample_variance
            )
        ),
        log_likelihood=-negative,
        observations=count,
        iterations=iterations,
        converged=converged,
        variance_targeted=variance_targeting,
        innovation=innovation,
        degrees_of_freedom=degrees,
    )


@dataclass(frozen=True)
class FatTail:
    """Whether the innovations need a fat tail, by likelihood ratio."""

    #: ``2 * (log L under t - log L under normal)``. Very slightly negative is
    #: possible and is not an error: the normal is the *limit* of the Student-t
    #: family, and :data:`MAX_DEGREES` stops short of it, so on a series with no
    #: fat tail the best admissible t is a hair worse than the normal. Measured
    #: at the cap the cost is about 7e-5 nats per observation, which is -0.10 on
    #: two thousand of them. The p-value clamps it at zero.
    statistic: float
    #: Upper tail of a chi-square with one degree of freedom at the statistic.
    #: **Conservative**, and the reason is structural rather than numerical: the
    #: null puts ``1/v`` at zero, which is the boundary of the parameter space,
    #: so the asymptotic null is the mixture ``0.5 * chi2(0) + 0.5 * chi2(1)``
    #: and this reports about twice the true probability. Measured on 200
    #: Gaussian-innovation samples of 2,000 observations, a nominal 5% test
    #: rejected 2.5% of the time — so a rejection here means what it says, and
    #: a near miss is weaker evidence against a fat tail than the number looks.
    p_value: float
    degrees_of_freedom: float
    identified: bool
    normal_log_likelihood: float
    student_t_log_likelihood: float

    @property
    def fat(self) -> bool:
        """Rejects at 5% *and* the degrees of freedom mean something.

        Both halves are needed. An unidentified estimate up against the cap
        cannot produce a likelihood gain worth rejecting on, so in practice the
        second condition is redundant — but a sample that manages both would be
        reporting a fat tail with no tail index behind it, and that is exactly
        the claim this refuses to make.
        """
        return self.p_value < 0.05 and self.identified


def fat_tail_test(
    returns: Sequence[float] | ReturnSeries,
    *,
    variance_targeting: bool = False,
    mean: float | None = None,
) -> FatTail:
    """Fit both innovation distributions and compare them.

    The models are nested — the standardised Student-t goes to the normal as the
    degrees of freedom grow — so the likelihood ratio is the right test and no
    information criterion is needed to choose between them.

    The reason to run it rather than always fitting the t: a series whose
    innovations really are normal has its degrees of freedom estimated anyway,
    and the estimate lands somewhere large and arbitrary. The value at risk that
    comes out is then slightly too wide by an amount nobody asked for. The test
    says whether the extra parameter is doing work.
    """
    values = list(returns.values) if isinstance(returns, ReturnSeries) else list(returns)
    normal = fit_garch(
        values, variance_targeting=variance_targeting, mean=mean, innovation=Innovation.NORMAL
    )
    student = fit_garch(
        values,
        variance_targeting=variance_targeting,
        mean=mean,
        innovation=Innovation.STUDENT_T,
    )
    statistic = 2.0 * (student.log_likelihood - normal.log_likelihood)
    degrees = student.degrees_of_freedom
    assert degrees is not None  # a Student-t fit always carries one
    return FatTail(
        statistic=statistic,
        p_value=chi_square_sf(max(statistic, 0.0), 1.0),
        degrees_of_freedom=degrees,
        identified=student.degrees_identified,
        normal_log_likelihood=normal.log_likelihood,
        student_t_log_likelihood=student.log_likelihood,
    )


def garch_forecast_series(
    returns: Sequence[float] | ReturnSeries, *, variance_targeting: bool = False
) -> tuple[Garch, tuple[float, ...]]:
    """Fit, and return the one-step-ahead volatility forecast for each period.

    The forecast for period ``t`` uses returns strictly before ``t``, so the
    series can be handed straight to :func:`~shortfall.backtest.validate`
    alongside the returns it was forecasting. That pairing is the point of this
    module: the independence test is what says a constant forecast is not good
    enough, and this is the thing that answers it.
    """
    fitted = fit_garch(returns, variance_targeting=variance_targeting)
    return fitted, fitted.volatilities


__all__ = [
    "IDENTIFIED_DEGREES",
    "MAX_DEGREES",
    "MAX_PERSISTENCE",
    "MIN_DEGREES",
    "MIN_OBSERVATIONS",
    "DidNotConverge",
    "FatTail",
    "Garch",
    "Innovation",
    "fat_tail_test",
    "fit_garch",
    "garch_forecast_series",
    "garch_variances",
]

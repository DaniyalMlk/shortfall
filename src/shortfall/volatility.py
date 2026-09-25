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
from dataclasses import dataclass
from typing import Final

from .series import ReturnSeries, TooShort

#: Fewest observations a fit will attempt. Three parameters from fewer than
#: this is not an estimate; the likelihood surface is nearly flat along the
#: persistence direction and the optimiser reports whatever it started near.
MIN_OBSERVATIONS: Final = 100

#: Largest persistence the parameterisation will produce. Exactly one is
#: inadmissible — the unconditional variance is undefined there — and the
#: transform approaches it asymptotically, so this caps it away from the
#: boundary where the long-run variance and the half-life both blow up.
MAX_PERSISTENCE: Final = 0.999_9

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
        """The one-step-ahead forecast, given the return that just arrived."""
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
) -> float:
    """Minus the Gaussian log-likelihood, which is what gets minimised.

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
        total += _LOG_2PI + math.log(variance) + residual * residual / variance
        variance = omega + alpha * residual * residual + beta * variance
    return 0.5 * total


def _sigmoid(x: float) -> float:
    """Numerically safe logistic. Written in the branch that does not overflow."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    scaled = math.exp(x)
    return scaled / (1.0 + scaled)


def _from_free(
    free: Sequence[float], *, target: float | None
) -> tuple[float, float, float]:
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
    """
    persistence = MAX_PERSISTENCE * _sigmoid(free[1])
    weight = _sigmoid(free[2])
    alpha = persistence * weight
    beta = persistence - alpha
    # Variance targeting: the long-run level is the sample variance, so omega
    # follows from the persistence rather than being searched over.
    omega = target * (1.0 - persistence) if target is not None else math.exp(free[0])
    return omega, alpha, beta


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
) -> Garch:
    """Fit a GARCH(1,1) by maximum likelihood.

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
        omega, alpha, beta = _from_free(free, target=target)
        return _negative_log_likelihood(
            values,
            omega=omega,
            alpha=alpha,
            beta=beta,
            mean=centre,
            seed=sample_variance,
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
    best, negative, iterations, converged = _nelder_mead(objective, start)
    if strict and not converged:
        raise DidNotConverge(
            f"the likelihood optimiser did not converge in {iterations} iterations. "
            "Pass strict=False to receive the best point found, or try "
            "variance_targeting=True, which removes the worst-determined parameter "
            "from the search."
        )

    omega, alpha, beta = _from_free(best, target=target)
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
    "MAX_PERSISTENCE",
    "MIN_OBSERVATIONS",
    "DidNotConverge",
    "Garch",
    "fit_garch",
    "garch_forecast_series",
    "garch_variances",
]

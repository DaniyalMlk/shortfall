"""Risk over a horizon, from a fitted volatility process.

:meth:`~shortfall.volatility.Garch.risk` covers one period and refuses more,
because over ``h`` periods the sum of the innovations is not a member of the
family they were drawn from — for the Student-t it is not a Student-t at all,
and even under normal innovations it is a variance mixture rather than a normal.
The refusal is right and it leaves a gap: a caller who wants a ten-day figure
still needs a quantile, and the only closed form available is the assumption the
one-step method just declined to make.

The honest route is not analytic. Run the recursion forward, drawing an
innovation each step and feeding the squared residual back into the variance, and
take the quantile of the accumulated return across paths. That gets three things
the analytic route cannot:

* The **variance path is stochastic**, not its expectation. A large draw early
  raises the variance for the rest of the horizon, so the distribution of the
  total is skewed and fatter than a mixture over the *expected* path would be.
  Using ``horizon_variance`` with a normal quantile misses this entirely.
* **Mean reversion** is in it, because it is in the recursion.
* The **shape** of the innovations can be the empirical one. Bootstrapping the
  model's own standardised residuals assumes no tail shape at all — it is the
  filtered historical simulation of Barone-Adesi and others, with the filter
  being a model that forecasts rather than an exponential weighting that cannot.

How much that last one is worth depends on the horizon, and it is worth measuring
rather than assuming. On a series with genuinely fat innovations, resampling the
residuals instead of drawing normals raises the expected shortfall by 22% at one
step and by 7% at ten, averaged over four samples of 20,000 paths. Summing ten
draws is a partial central-limit convergence, so the shape of one innovation
matters much less to the total than to a single period — which also means a
one-step tail adjustment does not scale to a horizon, whatever it is multiplied
by.

What it costs is that the answer is an estimate with a standard error, and that
error is reported rather than left to be guessed at. A 99% quantile from a
thousand paths is estimated from ten of them, and the number it produces looks
exactly as precise as one from a hundred thousand.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from .historical import HistoricalRisk, QuantileMethod, historical_risk
from .series import Convention, TooShort
from .volatility import Garch, Innovation

#: Fewest paths a simulation will run. At a hundred paths the 99% quantile is the
#: first order statistic, which is not an estimate of anything.
MIN_PATHS: Final = 1_000

#: Most paths, and most total steps. The work is ``paths * steps`` in Python, so
#: the second cap is the one that binds: a hundred thousand paths over 250 steps
#: is twenty-five million draws.
MAX_PATHS: Final = 200_000
MAX_TOTAL_STEPS: Final = 4_000_000

#: Fewest standardised residuals the bootstrap will draw from. A bootstrap cannot
#: draw beyond the worst residual it has, so the resample's tail is bounded by the
#: sample: at a hundred residuals the worst is the 1st percentile and a 99% figure
#: sits exactly on the boundary of the data. At 250 it is the 0.4th, which puts the
#: quantile inside the resample rather than at its edge. Below this the parametric
#: draw is the honest route — it extrapolates, and says that it does.
MIN_OBSERVED_RESIDUALS: Final = 250

#: Batches the paths are split into to estimate the Monte Carlo error. Twenty
#: gives nineteen degrees of freedom, which is enough for an error bar and few
#: enough that each batch still has paths in its own tail.
BATCHES: Final = 20


class Innovations(str, Enum):
    """Where the simulated innovations come from."""

    #: Drawn from the distribution the model was fitted under — a standard
    #: normal, or a Student-t standardised to unit variance at the fitted degrees
    #: of freedom. Consistent with the likelihood, and only as good as it.
    PARAMETRIC = "parametric"
    #: Resampled from the model's own standardised residuals. Assumes nothing
    #: about the tail: whatever shape is left in the residuals after the variance
    #: process has been taken out is the shape that gets drawn. The cost is that
    #: no draw can exceed the largest residual observed, so the extreme tail is
    #: bounded by the sample rather than by a distribution.
    BOOTSTRAP = "bootstrap"


@dataclass(frozen=True)
class HorizonRisk:
    """A simulated horizon risk estimate, and how much to trust it."""

    #: The estimate itself, taken from the simulated horizon returns exactly as
    #: it would be from realised ones — and in the convention the model was
    #: fitted on, which is the one :attr:`convention` names.
    risk: HistoricalRisk
    #: The same estimate taken from the *sum* of the simulated per-period
    #: returns instead of their accumulation. Under
    #: :attr:`~shortfall.series.Convention.LOG` the sum is the accumulation and
    #: this is :attr:`risk` itself, the same object. Under
    #: :attr:`~shortfall.series.Convention.SIMPLE` it is the figure the
    #: arithmetic shortcut would have produced, kept for two reasons: it is what
    #: :attr:`analytic_volatility` predicts, so the accumulation check survives
    #: under either convention, and the gap between the two is the size of the
    #: convention error rather than something a caller has to take on faith.
    additive_risk: HistoricalRisk
    #: Which accumulation :attr:`risk` was formed under, inherited from the
    #: fitted model.
    convention: Convention
    steps: int
    paths: int
    innovations: Innovations
    #: Standard deviation of the simulated horizon returns, as accumulated.
    simulated_volatility: float
    #: Standard deviation of the summed per-period returns. This is the one
    #: :attr:`analytic_volatility` is a prediction of.
    additive_volatility: float
    #: ``sqrt`` of the analytic horizon variance, which the simulation should
    #: reproduce. Agreement with :attr:`additive_volatility` is evidence the
    #: paths are being accumulated correctly; it says nothing about the
    #: quantile, which is the point of simulating. It is a statement about the
    #: sum and not about the compounded return: the variance of a product is not
    #: the sum of the variances, so under
    #: :attr:`~shortfall.series.Convention.SIMPLE` there is no closed form here
    #: to check :attr:`simulated_volatility` against.
    analytic_volatility: float
    #: What scaling the one-step volatility by ``sqrt(steps)`` would give.
    square_root_of_time_volatility: float
    #: The closed-form one-step value at risk from the same fitted model, at the
    #: same confidence. Carried so the horizon figure can be compared against the
    #: square-root-of-time rule as applied to a *quantile*, which is how the rule
    #: is actually used and is not the same comparison as the one on volatilities.
    one_step_value_at_risk: float
    #: Monte Carlo standard error of the value at risk, from the spread across
    #: :data:`BATCHES` independent batches of paths. Assumption-free — no density
    #: at the quantile has to be estimated — and it is the error of the batch-mean
    #: estimator of the same quantile rather than of the full-sample figure
    #: reported.
    #:
    #: **Read it as a lower bound.** Measured over 30 independent runs at ten
    #: steps: at 20,000 paths it came to 0.89 of the observed spread of the
    #: reported figure, which is inside the 13% precision of a 30-run standard
    #: deviation; at 2,000 paths it came to 0.73 of it, which is not. The reason
    #: is that a batch of 100 paths estimates a 99% quantile from its own single
    #: worst path. The error bar means what it says once ``paths`` times the tail
    #: probability is a couple of hundred.
    standard_error: float
    #: Paths whose compounded value reached zero. A simple return at or below
    #: -1 is ruin rather than a negative price, so the horizon return of such a
    #: path is exactly -1 and stays there; :class:`~shortfall.series.ReturnSeries`
    #: refuses such a value in realised data for the same reason. Always zero
    #: under :attr:`~shortfall.series.Convention.LOG`, where no sum of finite
    #: log returns can wipe a position out, and zero on anything resembling
    #: daily data. Non-zero means the quantile is sitting on a boundary and
    #: :attr:`expected_shortfall` is bounded below by -1 rather than estimated.
    ruined_paths: int

    @property
    def value_at_risk(self) -> float:
        return self.risk.value_at_risk

    @property
    def expected_shortfall(self) -> float:
        return self.risk.expected_shortfall

    @property
    def relative_standard_error(self) -> float:
        """Standard error as a fraction of the value at risk.

        The figure to look at before quoting a result. Below about a percent the
        simulation is not what limits the answer; above five the path count is.
        """
        return self.standard_error / self.value_at_risk if self.value_at_risk else math.inf

    @property
    def scaling_against_square_root_of_time(self) -> float:
        """Simulated horizon volatility over the square-root-of-time figure.

        Below one when today is more volatile than the long run and above one
        when it is calmer, for the same reason as the analytic version — this is
        the same quantity measured through the paths.

        Measured on the *summed* returns, because square-root-of-time is a
        statement about a sum: scaling a one-period volatility by ``sqrt(h)``
        is the variance of an accumulation of ``h`` uncorrelated periods, which
        is arithmetic that only the additive accumulation does. Mixing the
        compounded volatility into this ratio would put the convention gap into
        a number whose whole job is reporting mean reversion.
        """
        return self.additive_volatility / self.square_root_of_time_volatility

    @property
    def compounding_factor(self) -> float:
        """Reported value at risk over the summed one.

        The size of the convention error, as a multiplier. Exactly ``1.0`` under
        :attr:`~shortfall.series.Convention.LOG`, where the two figures are the
        same object rather than two numbers that happen to agree. Under
        :attr:`~shortfall.series.Convention.SIMPLE` it is below one at every
        horizon past the first, because compounding is gentler than addition in
        the loss direction — two periods of -10% compound to -19% and sum to
        -20% — and it falls away with the horizon. Measured on the bundled
        equity series at 99%, over five seeds of 16,000 paths: 0.978 at five
        steps, 0.966 at ten, 0.909 at sixty, 0.829 at two hundred and fifty,
        each within 0.005 across the seeds. Read the other way, adding simple
        returns overstates the loss by 2.2%, 3.5%, 10.0% and 20.6%.

        **At one step it is one to within rounding, and not exactly one.** A
        single return is its own horizon return under either convention, so the
        two quantities are the same real number — but the compounded route
        computes ``(1 + r) - 1``, which loses the low bits of ``r``. The gap came
        to -1.6e-15 on the bundled series and is bounded by a few units in the
        last place of the quantile; it is rounding, not a model difference. The
        exactness that *is* available is the log one above, where there is a
        single list and a single call.

        That one-step agreement is also why every closed-form test in this
        module's suite passed before the convention was carried at all: all of
        them are at one step, where the bug cannot appear.
        """
        additive = self.additive_risk.value_at_risk
        return self.risk.value_at_risk / additive if additive else math.inf

    @property
    def loss_against_square_root_of_time(self) -> float:
        """Reported value at risk over the scaled one-step figure.

        What the caller's own loss is against the rule, which under
        :attr:`~shortfall.series.Convention.SIMPLE` is
        :attr:`quantile_against_square_root_of_time` times
        :attr:`compounding_factor` and under
        :attr:`~shortfall.series.Convention.LOG` is the same number. The two
        effects can point the same way or against each other, and on the bundled
        equity series they very nearly cancel: at ten steps the variance path
        puts the summed ratio at 1.038 and compounding multiplies it by 0.966,
        leaving almost exactly one. The rule looks right there for two reasons
        that are each wrong by about the same amount. At two hundred and fifty
        steps the cancellation is gone — about 1.03 times 0.83 — and the
        compounding half is the larger of the two.
        """
        scaled = math.sqrt(self.steps) * self.one_step_value_at_risk
        return self.risk.value_at_risk / scaled if scaled else math.inf

    @property
    def quantile_against_square_root_of_time(self) -> float:
        """Summed value at risk over ``sqrt(steps)`` times the one-step figure.

        Taken from the sum, for the reason given on
        :attr:`scaling_against_square_root_of_time`: the rule is a statement
        about an accumulation of uncorrelated periods, and this is the ratio that
        isolates what the variance path does to it from what the convention
        does. :attr:`loss_against_square_root_of_time` is the other one.

        The number worth looking at, and it does not agree with
        :attr:`scaling_against_square_root_of_time` — which is the finding.
        Measured over ten independent runs of 40,000 paths at ten steps on one
        fitted GARCH: the horizon volatility came in 0.8% *below* the
        square-root-of-time figure (0.9922, spread 0.004) while the horizon value
        at risk came in 5.2% *above* it (1.0517, spread 0.008). Scaling the
        volatility is conservative there and scaling the quantile is not, in the
        same model on the same day.

        Across *series* rather than across runs the picture is less tidy and the
        untidiness is the point. Over ten samples with normal innovations the ratio
        averaged 1.074 and exceeded one on all ten; over ten with a fitted tail near
        four and a half degrees of freedom it averaged 0.986 and exceeded one on
        four. The direction is dependable in one case and not in the other, so no
        multiplier on a scaled volatility is even consistently wrong.

        The reason is that the variance path is stochastic rather than its own
        expectation. A large draw early in the horizon raises the variance for
        every remaining step, so the accumulated return is leptokurtic even when
        each innovation is normal: the ratio of value at risk to volatility went
        from 2.334 at one step — the normal's 2.326, as it must — to 2.480 at ten.
        An analytic horizon variance cannot see this, because averaging over the
        variance path is exactly what it does.
        """
        scaled = math.sqrt(self.steps) * self.one_step_value_at_risk
        return self.additive_risk.value_at_risk / scaled if scaled else math.inf


def _sample_variance(values: Sequence[float]) -> float:
    """Unbiased sample variance, summed with ``fsum``."""
    average = math.fsum(values) / len(values)
    return math.fsum((value - average) ** 2 for value in values) / (len(values) - 1)


def _standardised_t_draw(rng: random.Random, degrees: float) -> float:
    """A Student-t standardised to unit variance.

    ``normal / sqrt(chi2(v) / v)`` is the definition of a Student-t; dividing by
    ``sqrt(v / (v - 2))`` brings its variance to one so that the variance
    recursion keeps its meaning. The chi-square comes from a gamma with shape
    ``v/2`` and scale two, which works for fractional degrees of freedom where
    summing squared normals would not.
    """
    chi_square = 2.0 * rng.gammavariate(degrees / 2.0, 1.0)
    return (
        rng.gauss(0.0, 1.0)
        / math.sqrt(chi_square / degrees)
        / math.sqrt(degrees / (degrees - 2.0))
    )


def horizon_risk(
    fitted: Garch,
    returns: Sequence[float],
    *,
    steps: int,
    confidence: float = 0.99,
    paths: int = 20_000,
    innovations: Innovations = Innovations.BOOTSTRAP,
    method: QuantileMethod = QuantileMethod.LINEAR,
    seed: int | None = 0,
    convention: Convention | None = None,
) -> HorizonRisk:
    """Value at risk and expected shortfall over ``steps`` periods, by simulation.

    ``returns`` is the series ``fitted`` was fitted on. It is required rather than
    optional for two reasons: the recursion starts from the one-step-ahead
    variance, which needs the last return, and the bootstrap draws from the
    standardised residuals, which need all of them. Passing a different series
    raises rather than silently standardising by variances belonging to other
    dates.

    ``convention`` says how a path is accumulated, and defaults to the one
    ``fitted`` carries — which is the convention of the series it was fitted on,
    so the ordinary call needs nothing here. It is the question this function
    cannot answer on its own and must not guess at, because the two answers are
    different arithmetic on the same draws: under
    :attr:`~shortfall.series.Convention.LOG` the horizon return is the sum of
    the per-period returns, and under
    :attr:`~shortfall.series.Convention.SIMPLE` it is ``prod(1 + r) - 1``.
    Adding simple returns overstates a loss, by 3.5% at ten periods and 20.6% at
    two hundred and fifty on the bundled equity series, and the error does not
    show up at one period, where a return is its own horizon return either way.

    Both are returned regardless — see :attr:`HorizonRisk.additive_risk` — so
    the gap is a number in the result rather than a caveat in the prose.

    ``seed`` defaults to a fixed value so that two identical calls agree. Pass
    ``None`` for a fresh stream, which is what an outer loop measuring this
    function's own error wants.

    The result carries a standard error. Check it before quoting a figure: at a
    thousand paths the 99% quantile is estimated from ten of them and carries an
    error of several percent, while looking exactly as precise as one from a
    hundred thousand.
    """
    if steps < 1:
        raise ValueError(f"a horizon covers at least one period, got {steps!r}")
    if not MIN_PATHS <= paths <= MAX_PATHS:
        raise ValueError(
            f"paths must be between {MIN_PATHS} and {MAX_PATHS}, got {paths!r}. Below "
            f"{MIN_PATHS} the tail of the simulation holds too few paths to be a "
            "quantile of anything."
        )
    if steps * paths > MAX_TOTAL_STEPS:
        raise ValueError(
            f"{steps} steps over {paths} paths is {steps * paths} draws, above the "
            f"limit of {MAX_TOTAL_STEPS}. Lower the path count — the standard error "
            "in the result says how much precision that costs."
        )
    if len(returns) != fitted.observations:
        raise ValueError(
            f"{len(returns)} returns against {fitted.observations} the model was "
            "fitted on. The horizon simulation starts from the one-step-ahead "
            "variance and draws from the standardised residuals, and both belong to "
            "the series that was fitted."
        )

    rng = random.Random(seed)
    residuals: tuple[float, ...] = ()
    degrees = fitted.degrees_of_freedom
    if innovations is Innovations.BOOTSTRAP:
        residuals = fitted.standardised(returns)
        if len(residuals) < MIN_OBSERVED_RESIDUALS:
            raise TooShort(
                f"{len(residuals)} standardised residuals is too few to resample a "
                f"tail from; at least {MIN_OBSERVED_RESIDUALS} are needed. A "
                "bootstrap cannot draw past the worst residual it has, so on a "
                "sample this short the tail of the resample is the sample's own "
                "boundary. Use parametric innovations, which extrapolate and say so."
            )
    elif fitted.innovation is Innovation.STUDENT_T and degrees is None:  # pragma: no cover
        raise ValueError("a Student-t fit carries degrees of freedom; this one does not")

    chosen = fitted.convention if convention is None else convention
    start = fitted.next_variance(returns[-1])
    omega, alpha, beta, mean = fitted.omega, fitted.alpha, fitted.beta, fitted.mean
    parametric_t = innovations is Innovations.PARAMETRIC and (
        fitted.innovation is Innovation.STUDENT_T and degrees is not None
    )

    compounding = chosen is Convention.SIMPLE
    totals: list[float] = []
    accumulated: list[float] = []
    ruined = 0
    for _ in range(paths):
        variance = start
        total = 0.0
        growth = 1.0
        for _step in range(steps):
            if residuals:
                draw = residuals[rng.randrange(len(residuals))]
            elif parametric_t:
                assert degrees is not None  # narrowed by parametric_t
                draw = _standardised_t_draw(rng, degrees)
            else:
                draw = rng.gauss(0.0, 1.0)
            residual = math.sqrt(variance) * draw
            step_return = mean + residual
            total += step_return
            if compounding and growth > 0.0:
                # A simple return at or below -1 is ruin, not a negative price.
                # `ReturnSeries` refuses such a value in realised data; here the
                # path is wiped out and stays wiped out, and the horizon return
                # floors at -1 rather than changing sign. The variance recursion
                # carries on regardless, because the summed total is still a
                # well-defined quantity and is what the analytic variance is a
                # prediction of.
                growth *= 1.0 + step_return
                if growth <= 0.0:
                    growth = 0.0
                    ruined += 1
            variance = omega + alpha * residual * residual + beta * variance
        totals.append(total)
        if compounding:
            accumulated.append(growth - 1.0)

    additive_risk = historical_risk(totals, confidence=confidence, method=method)
    # Under the log convention the sum *is* the accumulation, so the two risks
    # are one object rather than two numbers that agree to some tolerance, and
    # `compounding_factor` is exactly one for that reason.
    risk = (
        historical_risk(accumulated, confidence=confidence, method=method)
        if compounding
        else additive_risk
    )
    reported = accumulated if compounding else totals
    variance_of_totals = _sample_variance(totals)
    variance_of_reported = _sample_variance(reported) if compounding else variance_of_totals

    # Batched, so the error bar needs no density estimate at the quantile. Each
    # batch is a contiguous slice of an independent stream, so the batches are
    # independent by construction rather than by assumption. Batched on the
    # reported accumulation, because the error bar belongs to the figure
    # reported.
    per_batch = paths // BATCHES
    batch_estimates = [
        historical_risk(
            reported[index * per_batch : (index + 1) * per_batch],
            confidence=confidence,
            method=method,
        ).value_at_risk
        for index in range(BATCHES)
    ]
    batch_mean = math.fsum(batch_estimates) / BATCHES
    batch_variance = math.fsum((value - batch_mean) ** 2 for value in batch_estimates) / (
        BATCHES - 1
    )
    standard_error = math.sqrt(batch_variance / BATCHES)

    one_step = math.sqrt(start)
    return HorizonRisk(
        one_step_value_at_risk=fitted.risk(
            confidence=confidence, last_return=returns[-1]
        ).value_at_risk,
        risk=risk,
        additive_risk=additive_risk,
        convention=chosen,
        steps=steps,
        paths=paths,
        innovations=innovations,
        simulated_volatility=math.sqrt(variance_of_reported),
        additive_volatility=math.sqrt(variance_of_totals),
        analytic_volatility=math.sqrt(fitted.horizon_variance(steps, last_return=returns[-1])),
        square_root_of_time_volatility=one_step * math.sqrt(steps),
        standard_error=standard_error,
        ruined_paths=ruined,
    )


__all__ = [
    "BATCHES",
    "MAX_PATHS",
    "MAX_TOTAL_STEPS",
    "MIN_OBSERVED_RESIDUALS",
    "MIN_PATHS",
    "HorizonRisk",
    "Innovations",
    "horizon_risk",
]

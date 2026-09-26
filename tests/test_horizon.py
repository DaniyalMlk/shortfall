"""The horizon simulation, checked against the closed forms it has to reproduce.

A Monte Carlo estimator is the easiest thing in this library to get subtly wrong
and the hardest to notice: it returns a plausible number whatever it does. So the
cases where a closed form exists are the load-bearing tests here. At one step the
simulation has to reproduce :func:`~shortfall.parametric.normal_risk` under normal
innovations and :func:`~shortfall.parametric.student_t_risk` under Student-t ones,
to within the error it reports for itself — and the horizon volatility has to
match the analytic aggregation at every horizon, which pins the accumulation
separately from the quantile.

What is left after that is the part simulating is *for*, and it is measured rather
than asserted: the accumulated return over a horizon is leptokurtic even when
every innovation is normal, because the variance path is stochastic.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from shortfall.historical import HistoricalRisk, QuantileMethod
from shortfall.horizon import (
    BATCHES,
    MAX_PATHS,
    MIN_OBSERVED_RESIDUALS,
    MIN_PATHS,
    HorizonRisk,
    Innovations,
    horizon_risk,
)
from shortfall.parametric import normal_risk, student_t_risk
from shortfall.series import TooShort
from shortfall.volatility import Garch, Innovation, fit_garch

TRUE = (2.0e-6, 0.08, 0.90)
CONFIDENCE = 0.99


def garch_path(count: int = 1500, *, seed: int = 5, burn_in: int = 500) -> list[float]:
    rng = random.Random(seed)
    omega, alpha, beta = TRUE
    variance = omega / (1.0 - alpha - beta)
    out: list[float] = []
    for _ in range(count + burn_in):
        value = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        out.append(value)
        variance = omega + alpha * value * value + beta * variance
    return out[burn_in:]


def student_t_path(
    count: int = 2000, *, degrees: float = 4.5, seed: int = 9, burn_in: int = 500
) -> list[float]:
    rng = random.Random(seed)
    omega, alpha, beta = TRUE
    scale = math.sqrt(degrees / (degrees - 2.0))
    variance = omega / (1.0 - alpha - beta)
    out: list[float] = []
    for _ in range(count + burn_in):
        chi_square = 2.0 * rng.gammavariate(degrees / 2.0, 1.0)
        innovation = rng.gauss(0.0, 1.0) / math.sqrt(chi_square / degrees) / scale
        value = math.sqrt(variance) * innovation
        out.append(value)
        variance = omega + alpha * value * value + beta * variance
    return out[burn_in:]


# -- against the closed forms ------------------------------------------------


def test_one_step_reproduces_the_normal_closed_form() -> None:
    """The whole machinery, checked where the answer is known exactly.

    One step with parametric normal innovations is ``normal_risk`` at the
    one-step-ahead volatility, and nothing about the simulation should change
    that. Inside three standard errors, which at 40,000 paths is about half a
    percent of the figure — so this is a real constraint rather than a formality.
    """
    data = garch_path()
    fitted = fit_garch(data)
    simulated = horizon_risk(
        fitted,
        data,
        steps=1,
        confidence=CONFIDENCE,
        paths=40_000,
        innovations=Innovations.PARAMETRIC,
        seed=1,
    )
    closed = normal_risk(
        mean=fitted.mean,
        volatility=math.sqrt(fitted.next_variance(data[-1])),
        confidence=CONFIDENCE,
    )
    assert abs(simulated.value_at_risk - closed.value_at_risk) < 3.0 * simulated.standard_error
    assert simulated.expected_shortfall == pytest.approx(closed.expected_shortfall, rel=0.02)
    assert simulated.one_step_value_at_risk == pytest.approx(closed.value_at_risk)


def test_one_step_reproduces_the_student_t_closed_form() -> None:
    """The same, with the fat tail in the draw as well as in the likelihood.

    This is the test that would catch the standardisation being dropped from the
    simulated draw: an unstandardised Student-t has variance ``v / (v - 2)``, so
    the simulated quantile would come out 34% wide at four and a half degrees of
    freedom while every other test in this file still passed.
    """
    data = student_t_path()
    fitted = fit_garch(data, innovation=Innovation.STUDENT_T)
    degrees = fitted.degrees_of_freedom
    assert degrees is not None
    simulated = horizon_risk(
        fitted,
        data,
        steps=1,
        confidence=CONFIDENCE,
        paths=60_000,
        innovations=Innovations.PARAMETRIC,
        seed=2,
    )
    closed = student_t_risk(
        mean=fitted.mean,
        volatility=math.sqrt(fitted.next_variance(data[-1])),
        confidence=CONFIDENCE,
        degrees=degrees,
    )
    assert abs(simulated.value_at_risk - closed.value_at_risk) < 4.0 * simulated.standard_error
    assert simulated.value_at_risk == pytest.approx(closed.value_at_risk, rel=0.04)


@pytest.mark.parametrize("steps", [1, 5, 25])
def test_the_simulated_volatility_matches_the_analytic_aggregation(steps: int) -> None:
    """Accumulation pinned separately from the quantile.

    The variance of the total is the sum of the expected variances along the path,
    because the residuals are a martingale difference sequence — so the simulation
    has to reproduce ``horizon_variance`` at every horizon whatever the quantile
    does. Averaged over six independent runs the ratio came to 1.0001; a single run
    of 20,000 paths carries about half a percent of noise in an estimate of a
    standard deviation, so the bound here is 2%.
    """
    data = garch_path()
    fitted = fit_garch(data)
    ratios = []
    for seed in range(6):
        result = horizon_risk(
            fitted,
            data,
            steps=steps,
            paths=20_000,
            innovations=Innovations.PARAMETRIC,
            seed=seed,
        )
        ratios.append(result.simulated_volatility / result.analytic_volatility)
    assert statistics.fmean(ratios) == pytest.approx(1.0, abs=0.01)
    assert all(abs(ratio - 1.0) < 0.02 for ratio in ratios)


# -- what simulating is for --------------------------------------------------


def test_the_horizon_return_is_leptokurtic_even_under_normal_innovations() -> None:
    """The finding, and the reason an analytic horizon variance is not enough.

    Each innovation is normal and the accumulated return is not, because the
    variance path is stochastic: a large draw early raises the variance for every
    remaining step. Measured over ten runs of 40,000 paths, the ratio of value at
    risk to volatility goes from 2.334 at one step — the normal's 2.326, as it must
    — to 2.480 at ten.

    So the two square-root-of-time comparisons point in opposite directions on the
    same day: the horizon volatility came 0.8% below the scaled figure and the
    horizon value at risk 5.2% above it. Scaling a volatility is conservative
    there; scaling a quantile is not.
    """
    data = garch_path()
    fitted = fit_garch(data)
    one = horizon_risk(
        fitted, data, steps=1, paths=40_000, innovations=Innovations.PARAMETRIC, seed=3
    )
    ten = horizon_risk(
        fitted, data, steps=10, paths=40_000, innovations=Innovations.PARAMETRIC, seed=103
    )
    one_ratio = one.value_at_risk / one.simulated_volatility
    ten_ratio = ten.value_at_risk / ten.simulated_volatility
    assert one_ratio == pytest.approx(2.326, abs=0.04)
    assert ten_ratio > one_ratio + 0.08
    assert ten.scaling_against_square_root_of_time < 1.0
    assert ten.quantile_against_square_root_of_time > 1.02
    # The two comparisons disagree, which is the whole content of the test.
    assert ten.quantile_against_square_root_of_time > ten.scaling_against_square_root_of_time


def test_resampling_the_residuals_matters_most_at_one_step() -> None:
    """How much the innovation shape is worth, and how fast it stops being worth it.

    Same fitted variance process, same paths, same seed, on a series with genuinely
    fat innovations: only the shape of the draw differs. Averaged over four samples
    of 20,000 paths, resampling the standardised residuals instead of drawing
    normals raises the expected shortfall by 22% at one step and by 7% at ten.

    The decay is the finding. Summing ten draws is a partial central-limit
    convergence, so the shape of a single innovation matters far less to the total
    than to one period — which also says that a one-step tail adjustment does not
    scale to a horizon, whatever factor it is multiplied by.

    The assertions are on expected shortfall and on the *mean* over samples. The
    value-at-risk ratio came to 1.06 at one step and 0.98 at ten over the same runs,
    and both are inside the sampling noise of a 99% quantile from 20,000 paths, so
    asserting a direction on it would be asserting noise.
    """
    ratios: dict[int, list[float]] = {1: [], 10: []}
    for steps in ratios:
        for seed in range(4):
            data = student_t_path(seed=9 + seed)
            fitted = fit_garch(data)
            resampled = horizon_risk(
                fitted,
                data,
                steps=steps,
                paths=20_000,
                seed=3,
                confidence=CONFIDENCE,
                innovations=Innovations.BOOTSTRAP,
            )
            gaussian = horizon_risk(
                fitted,
                data,
                steps=steps,
                paths=20_000,
                seed=3,
                confidence=CONFIDENCE,
                innovations=Innovations.PARAMETRIC,
            )
            ratios[steps].append(
                resampled.expected_shortfall / gaussian.expected_shortfall
            )
            assert resampled.innovations is Innovations.BOOTSTRAP

    assert statistics.fmean(ratios[1]) > 1.10
    assert all(ratio > 1.0 for ratio in ratios[1])
    assert statistics.fmean(ratios[10]) > 1.0
    assert statistics.fmean(ratios[1]) > statistics.fmean(ratios[10]) + 0.08


def test_mean_reversion_runs_in_both_directions() -> None:
    """Above the long-run level the horizon is below square-root-of-time, and below it above.

    Constructed rather than fitted, because the point is the direction and a fitted
    model sits wherever the data put it. Two records identical except for the
    variance they start from.
    """
    calm = _state(variance=2.0e-5, long_run=8.0e-5)
    stormy = _state(variance=2.0e-4, long_run=8.0e-5)
    series = [0.0] * calm.observations
    quiet = horizon_risk(
        calm, series, steps=50, paths=5_000, innovations=Innovations.PARAMETRIC, seed=4
    )
    loud = horizon_risk(
        stormy, series, steps=50, paths=5_000, innovations=Innovations.PARAMETRIC, seed=4
    )
    assert quiet.scaling_against_square_root_of_time > 1.0
    assert loud.scaling_against_square_root_of_time < 1.0


def _state(*, variance: float, long_run: float, observations: int = 200) -> Garch:
    """A Garch record at a chosen variance, with a chosen long-run level.

    ``omega`` follows from the persistence and the long-run level rather than being
    picked, so the record is internally consistent: the long-run variance really is
    the one named.
    """
    alpha, beta = 0.06, 0.90
    persistence = alpha + beta
    return Garch(
        omega=long_run * (1.0 - persistence),
        alpha=alpha,
        beta=beta,
        mean=0.0,
        variances=(variance,) * observations,
        log_likelihood=0.0,
        observations=observations,
        iterations=1,
        converged=True,
        variance_targeted=False,
    )


# -- the error bar -----------------------------------------------------------


def test_the_standard_error_falls_like_the_square_root_of_the_path_count() -> None:
    """Sixteen times the paths, a quarter of the error — averaged, since a single
    standard-error estimate is itself noisy.

    Measured: 0.00186 at 2,500 paths and 0.00076 at 40,000, a ratio of 2.4 against
    the 4 the rate predicts, on one run each. Over eight runs the ratio settles
    near 4, which is why this averages rather than asserting on one.
    """
    data = garch_path()
    fitted = fit_garch(data)
    errors = {}
    for paths in (2_500, 40_000):
        errors[paths] = statistics.fmean(
            horizon_risk(
                fitted,
                data,
                steps=5,
                paths=paths,
                innovations=Innovations.PARAMETRIC,
                seed=seed,
            ).standard_error
            for seed in range(8)
        )
    assert errors[2_500] / errors[40_000] == pytest.approx(4.0, rel=0.35)


def test_the_reported_error_brackets_the_run_to_run_spread() -> None:
    """The error bar checked against the thing it claims to measure.

    Twelve independent runs, and the observed standard deviation of the reported
    value at risk against the reported error. They agree to within the precision of
    a twelve-run standard deviation, which is about 21%.

    At larger scale the picture is the same and slightly conservative: over 30 runs
    of 20,000 paths at ten steps the reported error was 0.89 of the observed
    spread, and at 2,000 paths 0.73 of it. The reason is that a batch of a hundred
    paths estimates a 99% quantile from its own single worst path, so the docstring
    on the field tells a caller to read it as a lower bound.
    """
    data = garch_path()
    fitted = fit_garch(data)
    estimates = []
    reported = []
    for seed in range(12):
        result = horizon_risk(
            fitted,
            data,
            steps=10,
            paths=10_000,
            innovations=Innovations.PARAMETRIC,
            seed=seed,
        )
        estimates.append(result.value_at_risk)
        reported.append(result.standard_error)
    observed = statistics.stdev(estimates)
    assert statistics.fmean(reported) == pytest.approx(observed, rel=0.6)
    assert statistics.fmean(reported) < observed * 1.3


def test_the_relative_error_is_reported_and_small_at_a_serious_path_count() -> None:
    data = garch_path()
    fitted = fit_garch(data)
    result = horizon_risk(
        fitted, data, steps=10, paths=40_000, innovations=Innovations.PARAMETRIC, seed=6
    )
    assert 0.0 < result.relative_standard_error < 0.03
    assert result.paths == 40_000
    assert result.steps == 10
    assert result.risk.observations == 40_000
    # The tail the estimate really came from: 40,000 paths at 99% is 400 of them.
    assert result.risk.effective_sample == pytest.approx(400.0)


def test_the_error_is_infinite_rather_than_a_division_by_zero() -> None:
    """A value at risk of exactly zero is degenerate, not an occasion to crash."""
    empty = HorizonRisk(
        risk=_zero_risk(),
        steps=1,
        paths=MIN_PATHS,
        innovations=Innovations.PARAMETRIC,
        simulated_volatility=0.0,
        analytic_volatility=0.0,
        square_root_of_time_volatility=0.0,
        one_step_value_at_risk=0.0,
        standard_error=0.0,
    )
    assert empty.relative_standard_error == math.inf
    assert empty.quantile_against_square_root_of_time == math.inf


def _zero_risk() -> HistoricalRisk:
    return HistoricalRisk(
        value_at_risk=0.0,
        expected_shortfall=0.0,
        quantile=0.0,
        confidence=CONFIDENCE,
        observations=MIN_PATHS,
        tail_observations=1,
        method=QuantileMethod.LINEAR,
    )


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("steps", [0, -3])
def test_a_degenerate_horizon_is_refused(steps: int) -> None:
    data = garch_path(400)
    with pytest.raises(ValueError, match="at least one period"):
        horizon_risk(fit_garch(data), data, steps=steps)


@pytest.mark.parametrize("paths", [10, MIN_PATHS - 1, MAX_PATHS + 1])
def test_a_path_count_outside_the_range_is_refused(paths: int) -> None:
    data = garch_path(400)
    with pytest.raises(ValueError, match="paths must be between"):
        horizon_risk(fit_garch(data), data, steps=2, paths=paths)


def test_too_much_total_work_is_refused_rather_than_attempted() -> None:
    """The cap is on the product, because that is what the time is spent on."""
    data = garch_path(400)
    with pytest.raises(ValueError, match="above the limit"):
        horizon_risk(fit_garch(data), data, steps=2_000, paths=MAX_PATHS)


def test_a_different_series_is_refused() -> None:
    """The variances and residuals belong to the series that was fitted.

    Standardising a different series would divide by variances belonging to other
    dates, and the start of the simulation would be the wrong day's forecast. Both
    produce a number.
    """
    data = garch_path(400)
    fitted = fit_garch(data)
    with pytest.raises(ValueError, match="the model was"):
        horizon_risk(fitted, data[:-1], steps=5)


def test_bootstrapping_from_too_few_residuals_is_refused() -> None:
    """A bootstrap cannot draw past the worst residual it holds.

    At a hundred residuals the worst is the 1st percentile, so a 99% figure sits on
    the boundary of the data rather than inside it — and the estimate is then
    bounded by the sample in a way that looks like a number. The parametric draw
    extrapolates instead, which is a different assumption made openly.
    """
    short = garch_path(MIN_OBSERVED_RESIDUALS - 100)
    fitted = fit_garch(short)
    with pytest.raises(TooShort, match="too few"):
        horizon_risk(fitted, short, steps=5, innovations=Innovations.BOOTSTRAP)
    # And the parametric route works on the same sample.
    assert (
        horizon_risk(
            fitted, short, steps=5, paths=2_000, innovations=Innovations.PARAMETRIC
        ).value_at_risk
        > 0.0
    )


# -- reproducibility ---------------------------------------------------------


def test_the_same_seed_gives_the_same_answer() -> None:
    data = garch_path(600)
    fitted = fit_garch(data)
    first = horizon_risk(fitted, data, steps=10, paths=3_000, seed=11)
    second = horizon_risk(fitted, data, steps=10, paths=3_000, seed=11)
    assert first.value_at_risk == second.value_at_risk
    assert first.standard_error == second.standard_error


def test_a_different_seed_gives_a_different_answer_within_the_error() -> None:
    data = garch_path(600)
    fitted = fit_garch(data)
    first = horizon_risk(fitted, data, steps=10, paths=5_000, seed=11)
    second = horizon_risk(fitted, data, steps=10, paths=5_000, seed=12)
    assert first.value_at_risk != second.value_at_risk
    spread = abs(first.value_at_risk - second.value_at_risk)
    assert spread < 6.0 * max(first.standard_error, second.standard_error)


def test_the_batch_count_is_the_one_the_error_is_built_from() -> None:
    """A change to BATCHES has to be a decision, so it is named in one place."""
    assert BATCHES == 20
    data = garch_path(600)
    fitted = fit_garch(data)
    result = horizon_risk(fitted, data, steps=5, paths=4_000, seed=13)
    # Twenty batches of two hundred paths each, all of them used.
    assert result.paths % BATCHES == 0
    assert result.standard_error > 0.0


def test_the_horizon_quantile_crosses_square_root_of_time_in_both_directions() -> None:
    """Two effects pull against each other, and neither is a scaling rule.

    The **stochastic variance path** makes the accumulated return leptokurtic even
    when every innovation is normal, which pushes the horizon quantile *above* the
    square-root-of-time figure. **Aggregation** pulls the total towards normality
    while the one-step quantile keeps the whole of the innovation's own tail, which
    pushes it *below*. Under normal innovations only the first operates; under a fat
    tail the second cancels most of it.

    Measured over ten samples at ten steps and 30,000 paths with parametric draws.
    Under normal innovations: mean 1.074, range 1.011 to 1.150, above one on all
    ten. Under ``t(4.5)``: mean 0.986, range 0.930 to 1.025, above one on four of
    ten.

    So the direction is *reliable in one case and not in the other*, which is a
    stronger argument for simulating than a clean sign change would have been — a
    caller cannot pick a multiplier that is even consistently wrong. The assertions
    below are on three samples each: that the normal case clears one every time,
    that the fat case averages lower, and nothing about the fat case's sign on any
    one series.
    """
    normal_ratios = []
    fat_ratios = []
    for offset in range(3):
        thin = garch_path(count=2000, seed=9 + offset)
        fat = student_t_path(count=2000, seed=9 + offset)
        thin_fit = fit_garch(thin)
        fat_fit = fit_garch(fat, innovation=Innovation.STUDENT_T)
        assert fat_fit.degrees_identified
        normal_ratios.append(
            horizon_risk(
                thin_fit,
                thin,
                steps=10,
                paths=20_000,
                innovations=Innovations.PARAMETRIC,
                seed=3,
            ).quantile_against_square_root_of_time
        )
        fat_ratios.append(
            horizon_risk(
                fat_fit,
                fat,
                steps=10,
                paths=20_000,
                innovations=Innovations.PARAMETRIC,
                seed=3,
            ).quantile_against_square_root_of_time
        )
    assert all(ratio > 1.0 for ratio in normal_ratios)
    assert statistics.fmean(normal_ratios) > statistics.fmean(fat_ratios) + 0.04
    assert statistics.fmean(fat_ratios) < 1.03

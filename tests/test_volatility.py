"""Whether the volatility model recovers what it should, and fixes what it is for.

A fitted model is easy to check against itself and that establishes nothing.
So the parameters are recovered from data simulated with known ones, the
optimiser is checked against a function whose minimum is known in closed form,
and the point of the whole exercise — that forecasts built from this survive
the independence test where a constant forecast does not — is measured across
twenty independent samples rather than demonstrated on one.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import pairwise

import pytest

from shortfall.backtest import validate
from shortfall.distributions import normal_ppf, student_t_ppf
from shortfall.historical import ewma_volatility
from shortfall.parametric import normal_risk, student_t_risk
from shortfall.series import ReturnSeries, TooShort
from shortfall.volatility import (
    IDENTIFIED_DEGREES,
    MAX_DEGREES,
    MAX_PERSISTENCE,
    MIN_OBSERVATIONS,
    DidNotConverge,
    FatTail,
    Garch,
    Innovation,
    _negative_log_likelihood,
    _nelder_mead,
    fat_tail_test,
    fit_garch,
    garch_forecast_series,
    garch_variances,
)

TRUE = (2.0e-6, 0.08, 0.90)
QUANTILE = -normal_ppf(0.01)


def simulate_garch(
    count: int,
    omega: float = TRUE[0],
    alpha: float = TRUE[1],
    beta: float = TRUE[2],
    *,
    seed: int = 7,
    burn_in: int = 500,
) -> list[float]:
    """A GARCH(1,1) path, started at its own unconditional variance.

    The burn-in matters: starting at the unconditional variance is only the
    stationary distribution in expectation, and the first few hundred
    observations still carry the starting point.
    """
    rng = random.Random(seed)
    variance = omega / (1.0 - alpha - beta)
    out: list[float] = []
    for _ in range(count + burn_in):
        value = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        out.append(value)
        variance = omega + alpha * value * value + beta * variance
    return out[burn_in:]


def regime_switching(count: int, seed: int) -> list[float]:
    """Volatility that jumps between calm and turbulent and stays there.

    Deliberately *not* a GARCH process. It is the thing a constant-volatility
    forecast fails on most visibly, and fitting a GARCH to it is the realistic
    case rather than the flattering one.
    """
    rng = random.Random(seed)
    out: list[float] = []
    turbulent = False
    for _ in range(count):
        turbulent = rng.random() < (0.90 if turbulent else 0.02)
        out.append(rng.gauss(0.0, 0.030 if turbulent else 0.006))
    return out


# -- the optimiser, against a known optimum ----------------------------------


def test_the_optimiser_finds_the_minimum_of_a_function_with_a_known_one() -> None:
    """Rosenbrock, whose minimum is at (1, 1) and whose valley is exactly the
    kind of flat curved ridge that stops a lazy convergence test early."""

    def rosenbrock(point: Sequence[float]) -> float:
        x, y = point
        return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

    best, value, iterations, converged = _nelder_mead(rosenbrock, [-1.2, 1.0])
    assert converged
    assert best[0] == pytest.approx(1.0, abs=1e-5)
    assert best[1] == pytest.approx(1.0, abs=1e-5)
    assert value == pytest.approx(0.0, abs=1e-9)
    assert iterations > 10


def test_the_optimiser_reports_failure_rather_than_a_wrong_answer() -> None:
    def rosenbrock(point: Sequence[float]) -> float:
        x, y = point
        return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

    _, _, iterations, converged = _nelder_mead(rosenbrock, [-1.2, 1.0], max_iterations=5)
    assert not converged
    assert iterations == 5


def test_the_optimiser_handles_a_quadratic_in_three_dimensions() -> None:
    def bowl(point: Sequence[float]) -> float:
        targets = (3.0, -1.0, 0.5)
        return sum((value - target) ** 2 for value, target in zip(point, targets, strict=True))

    best, value, _, converged = _nelder_mead(bowl, [0.0, 0.0, 0.0])
    assert converged
    assert best == pytest.approx([3.0, -1.0, 0.5], abs=1e-6)
    assert value == pytest.approx(0.0, abs=1e-12)


# -- parameter recovery ------------------------------------------------------


def test_the_parameters_are_recovered_from_a_long_simulated_series() -> None:
    fitted = fit_garch(simulate_garch(4000))
    assert fitted.converged
    assert fitted.alpha == pytest.approx(TRUE[1], abs=0.02)
    assert fitted.beta == pytest.approx(TRUE[2], abs=0.02)
    assert fitted.omega == pytest.approx(TRUE[0], rel=0.15)
    assert fitted.long_run_volatility == pytest.approx(0.01, rel=0.08)


def test_a_shorter_sample_recovers_the_parameters_less_well() -> None:
    """Recorded rather than hidden. A thousand observations is four years of
    daily data and the persistence still comes back materially low, because
    the likelihood is flat along that direction and the sample has not seen
    enough slow decay to pin it down."""
    short = fit_garch(simulate_garch(1000))
    long = fit_garch(simulate_garch(4000))
    true_persistence = TRUE[1] + TRUE[2]
    assert abs(long.persistence - true_persistence) < abs(short.persistence - true_persistence)
    assert short.persistence < true_persistence


def test_variance_targeting_puts_the_long_run_level_at_the_sample_variance() -> None:
    data = simulate_garch(2000)
    fitted = fit_garch(data, variance_targeting=True)
    sample = math.fsum((value - fitted.mean) ** 2 for value in data) / len(data)
    assert fitted.variance_targeted
    assert fitted.long_run_variance == pytest.approx(sample, rel=1e-9)
    assert fitted.log_likelihood < fit_garch(data).log_likelihood + 1e-6


def test_the_free_fit_reaches_at_least_as_high_a_likelihood_as_the_targeted_one() -> None:
    """It searches a superset of the same space, so anything else would mean
    the optimiser stopped early."""
    data = simulate_garch(2000, seed=9)
    assert fit_garch(data).log_likelihood >= fit_garch(data, variance_targeting=True).log_likelihood


# -- the model's own identities ----------------------------------------------


def test_persistence_long_run_and_half_life_agree_with_each_other() -> None:
    fitted = fit_garch(simulate_garch(2000))
    assert fitted.persistence == pytest.approx(fitted.alpha + fitted.beta)
    assert fitted.long_run_variance == pytest.approx(
        fitted.omega / (1.0 - fitted.persistence)
    )
    assert fitted.persistence**fitted.half_life == pytest.approx(0.5, rel=1e-12)


def test_the_variance_recursion_reproduces_the_fitted_series() -> None:
    data = simulate_garch(1500)
    fitted = fit_garch(data)
    seed = math.fsum((value - fitted.mean) ** 2 for value in data) / len(data)
    rerun = garch_variances(
        data,
        omega=fitted.omega,
        alpha=fitted.alpha,
        beta=fitted.beta,
        mean=fitted.mean,
        seed=seed,
    )
    assert rerun == pytest.approx(list(fitted.variances), rel=1e-12)


def test_each_variance_is_formed_from_returns_strictly_before_it() -> None:
    """The off-by-one that makes the forecast usable. Changing the last return
    must not move any variance in the series — only the one after it, which is
    the forecast."""
    data = simulate_garch(600)
    fitted = fit_garch(data)
    altered = list(data)
    altered[-1] = altered[-1] * 5.0
    rerun = garch_variances(
        altered,
        omega=fitted.omega,
        alpha=fitted.alpha,
        beta=fitted.beta,
        mean=fitted.mean,
        seed=math.fsum((value - fitted.mean) ** 2 for value in data) / len(data),
    )
    assert rerun == pytest.approx(list(fitted.variances), rel=1e-12)
    assert fitted.next_variance(altered[-1]) > fitted.next_variance(data[-1])


def test_standardised_residuals_have_had_the_clustering_taken_out() -> None:
    """The claim of the model, checked on the thing that carries it.

    Squared returns of a volatility-clustered series are strongly
    autocorrelated. If the model has captured the process, the squares of the
    standardised residuals are much less so.
    """
    data = simulate_garch(3000, seed=17)
    fitted = fit_garch(data)
    standardised = fitted.standardised(data)

    def first_autocorrelation(values: list[float]) -> float:
        mean = math.fsum(values) / len(values)
        numerator = math.fsum((a - mean) * (b - mean) for a, b in pairwise(values))
        denominator = math.fsum((value - mean) ** 2 for value in values)
        return numerator / denominator

    raw = first_autocorrelation([value * value for value in data])
    left = first_autocorrelation([value * value for value in standardised])
    assert raw > 0.1, "the simulated series should cluster in the first place"
    assert abs(left) < 0.5 * raw


def test_standardising_a_different_series_is_refused() -> None:
    fitted = fit_garch(simulate_garch(500))
    with pytest.raises(ValueError, match="other dates"):
        fitted.standardised([0.01] * 400)


def test_the_series_may_be_a_return_series() -> None:
    data = simulate_garch(600)
    plain = fit_garch(data)
    wrapped = fit_garch(ReturnSeries(name="x", values=tuple(data)))
    assert wrapped.alpha == pytest.approx(plain.alpha)
    assert wrapped.beta == pytest.approx(plain.beta)


# -- forecasting -------------------------------------------------------------


def fixed_state(variance: float, *, persistence: float = 0.975) -> Garch:
    """A model with a chosen current variance, so a forecast can be probed
    from either side of its long-run level.

    Constructed directly rather than fitted: reaching a *calm* state from a
    fitted model's last variance is not always possible, because the beta term
    alone can already exceed the target and the alpha term cannot be negative.
    """
    long_run = 1e-4
    alpha = 0.08
    beta = persistence - alpha
    return Garch(
        omega=long_run * (1.0 - persistence),
        alpha=alpha,
        beta=beta,
        mean=0.0,
        variances=(variance,),
        log_likelihood=0.0,
        observations=1,
        iterations=0,
        converged=True,
        variance_targeted=False,
    )


def test_a_forecast_decays_towards_the_long_run_level() -> None:
    model = fixed_state(4e-4)
    path = model.forecast(400)
    assert path[0] == pytest.approx(4e-4)
    assert path[0] > path[1] > path[10] > path[100]
    assert path[-1] == pytest.approx(model.long_run_variance, rel=0.02)


def test_a_forecast_from_below_rises_towards_the_long_run_level() -> None:
    model = fixed_state(0.25e-4)
    path = model.forecast(400)
    assert path[0] < path[1] < path[10] < path[100]
    assert path[-1] == pytest.approx(model.long_run_variance, rel=0.02)


def test_a_forecast_from_the_long_run_level_is_flat() -> None:
    model = fixed_state(1e-4)
    path = model.forecast(50)
    assert all(value == pytest.approx(1e-4, rel=1e-12) for value in path)


def test_square_root_of_time_overstates_risk_after_a_shock() -> None:
    """Measured, and the size is the point.

    Starting at four times the long-run variance with a persistence of 0.975,
    the one-year horizon volatility is about 39% below what scaling the current
    volatility by the square root of time would give. Ten days is about 4%
    below. Those are not rounding differences.
    """
    model = fixed_state(4e-4)
    assert model.scaling_against_square_root_of_time(10) == pytest.approx(0.96, abs=0.01)
    assert model.scaling_against_square_root_of_time(250) == pytest.approx(0.61, abs=0.02)


def test_square_root_of_time_understates_risk_in_a_calm_market() -> None:
    """The direction that costs money, because it arrives while positions are
    being put on rather than cut."""
    model = fixed_state(0.25e-4)
    assert model.scaling_against_square_root_of_time(10) > 1.0
    assert model.scaling_against_square_root_of_time(250) > 1.5


def test_the_scaling_is_exactly_one_at_the_long_run_level() -> None:
    model = fixed_state(1e-4)
    for steps in (2, 10, 60, 250):
        assert model.scaling_against_square_root_of_time(steps) == pytest.approx(1.0, rel=1e-12)


def test_the_horizon_variance_is_the_sum_of_the_path() -> None:
    model = fixed_state(2e-4)
    assert model.horizon_variance(30) == pytest.approx(sum(model.forecast(30)))


def test_a_forecast_conditioned_on_a_new_return_moves_the_right_way() -> None:
    data = simulate_garch(800)
    fitted = fit_garch(data)
    quiet = fitted.forecast(5, last_return=0.0)
    shock = fitted.forecast(5, last_return=0.05)
    assert all(a < b for a, b in zip(quiet, shock, strict=True))


@pytest.mark.parametrize("steps", [0, -1])
def test_a_degenerate_horizon_is_refused(steps: int) -> None:
    with pytest.raises(ValueError, match="at least one period"):
        fixed_state(1e-4).forecast(steps)


# -- what EWMA could not do --------------------------------------------------


def test_the_model_mean_reverts_where_an_exponential_weighting_cannot() -> None:
    """The structural difference, stated as a comparison rather than a claim.

    An EWMA has no long-run level, so its multi-period forecast is flat at
    today's estimate whatever the horizon. The fitted persistence being below
    one is exactly what gives this model somewhere to revert to.
    """
    data = regime_switching(2000, seed=3)
    fitted = fit_garch(data)
    assert fitted.persistence < 1.0
    assert 0.0 < fitted.long_run_variance < math.inf
    # An EWMA at the conventional decay is the boundary case: alpha + beta = 1.
    weighted = ewma_volatility(data, decay=0.94)
    assert len(weighted) == len(data)
    # Both track the level, so they should be broadly correlated — this is a
    # check that the fit is tracking volatility at all, not that it differs.
    pairs = list(zip(fitted.volatilities, weighted, strict=True))
    high = [g for g, e in pairs if e > sorted(weighted)[len(weighted) // 2]]
    low = [g for g, e in pairs if e <= sorted(weighted)[len(weighted) // 2]]
    assert sum(high) / len(high) > sum(low) / len(low)


# -- the loop closed ---------------------------------------------------------


def test_a_garch_forecast_survives_the_independence_test_where_a_constant_does_not() -> None:
    """The reason this module exists, measured over twenty independent samples.

    On a regime-switching series a constant forecast set to the unconditional
    volatility has its breaches rejected as clustered most of the time. A
    one-step-ahead GARCH forecast over the same data is rejected once.
    """
    constant_rejections = 0
    garch_rejections = 0
    constant_breaches = 0
    garch_breaches = 0
    samples = 20
    for seed in range(40, 40 + samples):
        data = regime_switching(2000, seed)
        level = math.sqrt(math.fsum(value * value for value in data) / len(data))
        flat = validate(data, [QUANTILE * level] * len(data), confidence=0.99)
        fitted = fit_garch(data)
        conditional = validate(
            data, [QUANTILE * value for value in fitted.volatilities], confidence=0.99
        )
        constant_rejections += flat.independence.rejects_at(0.05)
        garch_rejections += conditional.independence.rejects_at(0.05)
        constant_breaches += flat.exceedances.count
        garch_breaches += conditional.exceedances.count

    assert constant_rejections >= 15, "the constant forecast should mostly be caught"
    assert garch_rejections <= 3, "the conditional forecast should mostly not be"
    # And the count improves too, though it does not reach the nominal 20 per
    # sample: a Gaussian GARCH still understates the tail of a series whose
    # standardised residuals are fat, so this halves the excess rather than
    # removing it. That is the honest result and the reason the two tests are
    # reported separately.
    assert constant_breaches / samples > 45
    assert 24 < garch_breaches / samples < 34


def test_the_independence_test_is_far_weaker_against_a_real_garch_process() -> None:
    """A finding about the test rather than about the model, measured.

    Over thirty samples of 2500 observations, a constant forecast has its
    breaches rejected as clustered 29 times on a regime-switching series and
    13 times on a GARCH(1,1) at realistic parameters — even though the
    volatility plainly clusters in both.

    So a passing independence test is not evidence that volatility is
    constant. It says the breaches did not arrive together, which is a much
    weaker statement, and the gap is what a caller reading a p-value of 0.4
    needs to know. The reason is that a GARCH's volatility wanders smoothly
    while a regime switch holds it high for weeks, and only the second puts
    enough breaches adjacent for a two-state chain fitted on twenty-odd events
    to see.
    """
    samples = 30
    on_garch = 0
    on_regimes = 0
    for offset in range(samples):
        smooth = simulate_garch(2500, seed=11 + offset)
        level = math.sqrt(math.fsum(value * value for value in smooth) / len(smooth))
        on_garch += (
            validate(smooth, [QUANTILE * level] * len(smooth), confidence=0.99)
            .independence.rejects_at(0.05)
        )

        jumpy = regime_switching(2500, 11 + offset)
        level = math.sqrt(math.fsum(value * value for value in jumpy) / len(jumpy))
        on_regimes += (
            validate(jumpy, [QUANTILE * level] * len(jumpy), confidence=0.99)
            .independence.rejects_at(0.05)
        )

    assert on_regimes >= 25, "regime switching should be caught nearly every time"
    assert on_garch <= 20, "a smooth GARCH should be caught far less often"
    assert on_regimes > on_garch + 8


def test_the_forecast_series_helper_pairs_with_validate() -> None:
    data = regime_switching(1500, seed=5)
    fitted, volatilities = garch_forecast_series(data)
    assert len(volatilities) == len(data)
    result = validate(
        data, [QUANTILE * value for value in volatilities], confidence=0.99
    )
    assert result.exceedances.observations == len(data)
    assert fitted.converged


# -- refusals ----------------------------------------------------------------


def test_too_few_observations_is_refused_with_the_reason() -> None:
    with pytest.raises(TooShort, match="nearly flat"):
        fit_garch([0.01, -0.01] * ((MIN_OBSERVATIONS - 2) // 2))


def test_a_series_with_no_variance_is_refused() -> None:
    with pytest.raises(ValueError, match="no variance at all"):
        fit_garch([0.001] * 200)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_observation_is_refused(bad: float) -> None:
    data = simulate_garch(200)
    data[50] = bad
    with pytest.raises(ValueError, match="not finite"):
        fit_garch(data)


def test_the_recursion_refuses_inadmissible_parameters() -> None:
    data = simulate_garch(200)
    with pytest.raises(ValueError, match="omega is positive"):
        garch_variances(data, omega=0.0, alpha=0.1, beta=0.8)
    with pytest.raises(ValueError, match="non-negative"):
        garch_variances(data, omega=1e-6, alpha=-0.1, beta=0.8)


def test_the_recursion_needs_something_to_recurse_over() -> None:
    with pytest.raises(TooShort, match="at least 2"):
        garch_variances([0.01], omega=1e-6, alpha=0.1, beta=0.8)


def test_the_fit_never_leaves_the_admissible_region() -> None:
    """The property the transform exists to guarantee, checked on data chosen
    to push the persistence towards the boundary."""
    for seed in range(3):
        fitted = fit_garch(simulate_garch(1200, 1e-7, 0.05, 0.945, seed=seed))
        assert fitted.omega > 0.0
        assert fitted.alpha >= 0.0
        assert fitted.beta >= 0.0
        assert fitted.persistence < 1.0
        assert fitted.persistence <= MAX_PERSISTENCE
        assert math.isfinite(fitted.long_run_variance)
        assert fitted.long_run_variance > 0.0


def test_a_failure_to_converge_can_be_returned_instead_of_raised() -> None:
    """For a caller fitting many series who would rather see which failed than
    lose the batch."""
    data = simulate_garch(300)
    with pytest.raises(DidNotConverge, match="variance_targeting"):
        fit_garch(data, max_iterations=3)
    relaxed = fit_garch(data, strict=False, max_iterations=3)
    assert relaxed.converged is False
    assert relaxed.iterations == 3
    # Still a usable object rather than a half-built one: the recursion ran to
    # the end with whatever parameters the search had reached.
    assert len(relaxed.variances) == len(data)
    assert relaxed.persistence < 1.0


# -- Student-t innovations ---------------------------------------------------


def simulate_student_t_garch(
    count: int,
    degrees: float = 5.0,
    omega: float = TRUE[0],
    alpha: float = TRUE[1],
    beta: float = TRUE[2],
    *,
    seed: int = 7,
    burn_in: int = 500,
) -> list[float]:
    """A GARCH(1,1) whose innovations are a Student-t standardised to unit variance.

    The draw is ``Z = normal / sqrt(chi2(v) / v)``, which is the definition of a
    Student-t rather than an approximation of one, divided by ``sqrt(v / (v-2))``
    to bring its variance to one. The chi-square comes from a gamma with shape
    ``v/2`` and scale two, which the standard library has and which works for
    fractional degrees of freedom where summing squared normals would not.

    Standardising here is what makes the recovery test meaningful: ``omega``,
    ``alpha`` and ``beta`` below are the same numbers as in the Gaussian case, so
    a fit that comes back with different ones has found something real.
    """
    rng = random.Random(seed)
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


def test_the_student_t_fit_recovers_the_degrees_of_freedom_it_was_given() -> None:
    """Five degrees of freedom simulated, and the variance parameters with them.

    The degrees of freedom are estimated jointly, so getting them right is not
    enough on its own — if the tail absorbed part of the variance dynamics the
    recovered ``alpha`` and ``beta`` would be wrong while ``v`` looked fine.
    """
    data = simulate_student_t_garch(4000, degrees=5.0, seed=17)
    fitted = fit_garch(data, innovation=Innovation.STUDENT_T)
    assert fitted.converged
    assert fitted.innovation is Innovation.STUDENT_T
    assert fitted.degrees_of_freedom == pytest.approx(5.0, rel=0.25)
    assert fitted.degrees_identified
    assert fitted.alpha == pytest.approx(TRUE[1], abs=0.04)
    assert fitted.beta == pytest.approx(TRUE[2], abs=0.05)
    assert fitted.persistence < 1.0


def test_normal_innovations_leave_the_degrees_of_freedom_absent() -> None:
    data = simulate_garch(600, seed=19)
    fitted = fit_garch(data)
    assert fitted.innovation is Innovation.NORMAL
    assert fitted.degrees_of_freedom is None
    assert not fitted.degrees_identified
    assert fitted.implied_excess_kurtosis == 0.0


def test_the_student_t_likelihood_beats_the_normal_one_on_a_fat_tailed_series() -> None:
    """Nested models, so this is the whole content of the comparison.

    The gain has to be large: two thousand observations of ``t(5)`` innovations
    against a normal assumption is not a close call, and a gain of a nat or two
    would mean the extra parameter was fitting noise.
    """
    data = simulate_student_t_garch(2000, degrees=5.0, seed=23)
    normal = fit_garch(data)
    student = fit_garch(data, innovation=Innovation.STUDENT_T)
    assert student.log_likelihood > normal.log_likelihood + 20.0


def test_the_student_t_likelihood_is_the_normal_one_in_the_limit() -> None:
    """At the cap the two densities agree, so the two likelihoods must too.

    This is the nesting written as an equality rather than asserted in prose. It
    also pins the Jacobian term: drop the half-log-variance from the Student-t
    branch and this diverges by a constant times the number of observations,
    while every other test in this file still passes.
    """
    data = simulate_garch(300, seed=29)
    centre = math.fsum(data) / len(data)
    variance = math.fsum((value - centre) ** 2 for value in data) / len(data)
    shared = {
        "omega": TRUE[0],
        "alpha": TRUE[1],
        "beta": TRUE[2],
        "mean": centre,
        "seed": variance,
    }
    gaussian = _negative_log_likelihood(data, **shared)
    nearly = _negative_log_likelihood(data, **shared, degrees=1e7)
    assert nearly == pytest.approx(gaussian, rel=1e-5)


def test_a_fat_tail_is_not_inferred_from_a_series_that_has_none() -> None:
    """Gaussian innovations: the estimate goes to the cap and says so.

    The assertion is on :attr:`Garch.degrees_identified` rather than on the
    number, because the number up there is meaningless by construction — the
    likelihood is flat above :data:`IDENTIFIED_DEGREES`, so which side of 400 the
    optimiser stops on is noise. A caller who reports it as a tail index is
    making a claim the data does not support, which is what the flag exists to
    prevent.
    """
    fitted = fit_garch(simulate_garch(2000, seed=31), innovation=Innovation.STUDENT_T)
    assert fitted.degrees_of_freedom is not None
    assert fitted.degrees_of_freedom > IDENTIFIED_DEGREES
    assert not fitted.degrees_identified
    assert fitted.degrees_of_freedom <= MAX_DEGREES


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(5.0, 6.0), (6.0, 3.0), (10.0, 1.0), (16.0, 0.5)],
)
def test_the_implied_excess_kurtosis_is_six_over_v_minus_four(
    degrees: float, expected: float
) -> None:
    assert _fitted_with(degrees).implied_excess_kurtosis == pytest.approx(expected)


@pytest.mark.parametrize("degrees", [2.5, 3.0, 4.0])
def test_the_implied_excess_kurtosis_is_infinite_without_a_fourth_moment(
    degrees: float,
) -> None:
    """At or below four degrees of freedom there is no fourth moment at all.

    Returning ``inf`` rather than the formula's negative number matters: at three
    degrees of freedom ``6 / (v - 4)`` is -6, and a caller comparing that against
    a sample kurtosis would conclude the innovations were thin-tailed.
    """
    assert _fitted_with(degrees).implied_excess_kurtosis == math.inf


def _fitted_with(degrees: float) -> Garch:
    """A Garch record with given degrees of freedom and nothing else of interest."""
    return Garch(
        omega=1e-6,
        alpha=0.05,
        beta=0.90,
        mean=0.0,
        variances=(1e-4, 1e-4),
        log_likelihood=0.0,
        observations=2,
        iterations=1,
        converged=True,
        variance_targeted=False,
        innovation=Innovation.STUDENT_T,
        degrees_of_freedom=degrees,
    )


# -- risk from the fitted model ----------------------------------------------


def test_the_conditional_risk_is_the_closed_form_at_the_current_volatility() -> None:
    """Delegation checked against the standalone function, not reimplemented."""
    fitted = _fitted_with(6.0)
    volatility = math.sqrt(fitted.variances[-1])
    assert fitted.risk(confidence=0.99) == student_t_risk(
        mean=0.0, volatility=volatility, confidence=0.99, degrees=6.0
    )
    gaussian = fit_garch(simulate_garch(400, seed=37))
    assert gaussian.risk(confidence=0.975) == normal_risk(
        mean=gaussian.mean,
        volatility=math.sqrt(gaussian.variances[-1]),
        confidence=0.975,
    )


def test_a_fat_tail_widens_the_quantile_at_the_same_volatility() -> None:
    """The point of the whole exercise, in one comparison.

    Same volatility, same mean, same confidence: only the innovation differs.
    Measured at five degrees of freedom, the 99% value at risk is 12.0% larger
    and the expected shortfall 29.4% larger. The second gap is the bigger one
    because expected shortfall averages over the whole tail, including the part
    beyond the quantile where the two densities differ by far the most — so a
    normal assumption understates the average loss in a bad tail by more than it
    understates the threshold, and the difference grows as the level rises: at
    99.5% the same two numbers are 21.3% and 40.6%.

    That ordering is the practical content. A desk that checks its value at risk
    against breach counts and never looks at the expected shortfall has been
    measuring the smaller of the two errors.
    """
    thin = _fitted_with(5.0)
    volatility = math.sqrt(thin.variances[-1])
    fat = thin.risk(confidence=0.99)
    flat = normal_risk(mean=0.0, volatility=volatility, confidence=0.99)
    assert fat.value_at_risk / flat.value_at_risk == pytest.approx(1.120, abs=0.002)
    assert fat.expected_shortfall / flat.expected_shortfall == pytest.approx(1.294, abs=0.002)

    deeper = thin.risk(confidence=0.995)
    deeper_flat = normal_risk(mean=0.0, volatility=volatility, confidence=0.995)
    assert deeper.value_at_risk / deeper_flat.value_at_risk == pytest.approx(1.213, abs=0.002)
    assert deeper.expected_shortfall / deeper_flat.expected_shortfall == pytest.approx(
        1.406, abs=0.002
    )


def test_the_conditional_risk_moves_with_the_return_that_just_arrived() -> None:
    """Passing the latest return advances the recursion one step first."""
    fitted = fit_garch(simulate_garch(500, seed=41), innovation=Innovation.STUDENT_T)
    quiet = fitted.risk(confidence=0.99, last_return=0.0)
    shock = fitted.risk(confidence=0.99, last_return=0.10)
    assert shock.value_at_risk > quiet.value_at_risk
    assert quiet.volatility == pytest.approx(math.sqrt(fitted.next_variance(0.0)))


# -- is the fat tail real? ---------------------------------------------------


def test_the_likelihood_ratio_finds_a_fat_tail_that_is_there() -> None:
    """Power, on two thousand observations of ``t(5)`` innovations.

    Measured over thirty independent samples: the test rejected on all thirty
    and the median estimate was 5.07 degrees of freedom. One sample is asserted
    here because thirty fits of two models each is a minute of test time for a
    result that was unanimous.
    """
    verdict = fat_tail_test(simulate_student_t_garch(2000, degrees=5.0, seed=43))
    assert verdict.fat
    assert verdict.identified
    assert verdict.p_value < 1e-6
    assert verdict.statistic > 30.0
    assert verdict.degrees_of_freedom == pytest.approx(5.0, rel=0.3)
    assert verdict.student_t_log_likelihood > verdict.normal_log_likelihood


def test_the_likelihood_ratio_does_not_find_a_fat_tail_that_is_not_there() -> None:
    """Size, and the boundary problem behind the p-value being conservative.

    The null puts ``1/v`` at zero, which is on the edge of the parameter space
    rather than inside it, so the asymptotic null distribution of the statistic
    is the mixture ``0.5 chi2(0) + 0.5 chi2(1)`` and not ``chi2(1)``. Reading it
    against ``chi2(1)`` therefore reports about twice the true tail probability,
    and the test is conservative by roughly a factor of two.

    Measured over 200 Gaussian-innovation samples of 2,000 observations: a
    nominal 5% test rejected 5 times, a rate of 2.5% — which is what the halving
    predicts, to the sample's precision. The degrees of freedom came back
    unidentified on 132 of the 200, so on two thirds of the samples the estimate
    never left the flat region at all.

    Twelve samples are run here. The bound allows two rejections, which at a
    true rate of 2.5% is a generous ceiling and keeps the test from being flaky
    about a fact established at larger scale above.

    Ten of the twelve came back with a slightly *negative* statistic, worst
    -0.098. That is the cap rather than a bug, and it is worth knowing: the
    normal is the limit of the family and :data:`MAX_DEGREES` stops short of it,
    so the best admissible Student-t on a thin-tailed series is a shade worse
    than the normal. At fixed variance parameters the cost of the cap is 0.043
    nats over these 1,200 observations — about 7e-5 each — which accounts for
    the whole of what was observed, so the bound below is written in terms of the
    sample size rather than as a constant.
    """
    rejections = 0
    observations = 1200
    for seed in range(60, 72):
        verdict = fat_tail_test(simulate_garch(observations, seed=seed))
        rejections += verdict.fat
        assert verdict.statistic > -2e-4 * observations
        assert 0.0 <= verdict.p_value <= 1.0
    assert rejections <= 2


def test_an_unidentified_tail_is_never_called_fat() -> None:
    """Both halves of the verdict, with the p-value forced to say yes.

    Constructed rather than sampled: a record whose p-value is zero and whose
    estimate is at the cap. The sampled version of this is vanishingly rare —
    there is no likelihood gain to be had up there — which is exactly why it is
    worth pinning by construction instead of hoping a seed produces it.
    """
    assert not FatTail(
        statistic=50.0,
        p_value=0.0,
        degrees_of_freedom=MAX_DEGREES,
        identified=False,
        normal_log_likelihood=100.0,
        student_t_log_likelihood=125.0,
    ).fat


# -- the level of the breaches, which is what this was for --------------------


def test_fat_innovations_bring_the_breach_count_down_towards_nominal() -> None:
    """The measurement the module was extended for, over twenty samples.

    A Gaussian GARCH on regime-switching data breaches its 99% forecast 28.2
    times per two thousand observations against a nominal 20. Estimating the
    innovation tail as well brings that to 22.45 — the excess over nominal falls
    from 8.2 to 2.45, so about 70% of what was left after the variance model is
    removed by the tail model.

    Not all of it, and the residue is not noise: the standardised residuals of a
    GARCH fitted to a regime-switching series are not identically distributed,
    because the process is not a GARCH. A single tail index for the whole sample
    is closer than one fixed at the normal's, and still an approximation.

    The independence verdict is unchanged at one rejection in twenty, which is
    the point: the quantile moved and the clustering the previous work fixed
    stayed fixed.
    """
    samples = 20
    normal_breaches = 0
    student_breaches = 0
    normal_rejections = 0
    student_rejections = 0
    for seed in range(40, 40 + samples):
        data = regime_switching(2000, seed)
        gaussian = fit_garch(data)
        student = fit_garch(data, innovation=Innovation.STUDENT_T)
        degrees = student.degrees_of_freedom
        assert degrees is not None
        # The standardised-t quantile: the raw t quantile brought back to unit
        # variance. Using the raw one here would widen every forecast by 29% at
        # four degrees of freedom and the breach count would undershoot.
        student_quantile = -student_t_ppf(0.01, degrees) * math.sqrt((degrees - 2.0) / degrees)
        flat = validate(data, [QUANTILE * v for v in gaussian.volatilities], confidence=0.99)
        fat = validate(
            data, [student_quantile * v for v in student.volatilities], confidence=0.99
        )
        normal_breaches += flat.exceedances.count
        student_breaches += fat.exceedances.count
        normal_rejections += flat.independence.rejects_at(0.05)
        student_rejections += fat.independence.rejects_at(0.05)

    assert 24.0 < normal_breaches / samples < 34.0
    assert 19.0 < student_breaches / samples < 26.0
    assert student_breaches < normal_breaches
    assert student_rejections <= 3
    assert normal_rejections <= 3


def test_fat_innovations_do_not_widen_a_forecast_that_was_already_right() -> None:
    """The other half of the claim: no cost when there is no fat tail.

    On genuinely Gaussian GARCH data the breach count is 19.75 per two thousand
    observations under normal innovations and 19.25 under estimated ones, both
    against a nominal 20. Estimating a tail index that turns out not to be needed
    costs half a breach in twenty, because the estimate goes to the cap where the
    density is the normal's to four decimal places.

    Without this the previous test would be satisfied by anything that widens
    forecasts — including multiplying them by 1.1, which would also bring the
    regime-switching count down and would be wrong.
    """
    samples = 10
    normal_breaches = 0
    student_breaches = 0
    for seed in range(3000, 3000 + samples):
        data = simulate_garch(2000, seed=seed)
        gaussian = fit_garch(data)
        student = fit_garch(data, innovation=Innovation.STUDENT_T)
        degrees = student.degrees_of_freedom
        assert degrees is not None
        student_quantile = -student_t_ppf(0.01, degrees) * math.sqrt((degrees - 2.0) / degrees)
        normal_breaches += validate(
            data, [QUANTILE * v for v in gaussian.volatilities], confidence=0.99
        ).exceedances.count
        student_breaches += validate(
            data, [student_quantile * v for v in student.volatilities], confidence=0.99
        ).exceedances.count
    assert 16.0 < normal_breaches / samples < 24.0
    assert 16.0 < student_breaches / samples < 24.0
    assert abs(student_breaches - normal_breaches) / samples < 2.0

"""Parametric value at risk and expected shortfall.

Every expected shortfall here has a closed form, and every closed form is
checked against numerical integration of the density it came from — a wholly
separate route to the same number, which is the only kind of agreement worth
anything.

The integration uses the substitution ``x = q - tan(theta)`` to map the infinite
tail onto ``[0, pi/2)``. Integrating the tail directly by truncating it at some
large negative number is the obvious approach and it quietly loses real mass for
a heavy-tailed distribution: at three degrees of freedom a truncation that looks
generous is wrong in the second decimal, which is larger than any error being
tested for.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from shortfall.distributions import normal_pdf, normal_ppf, student_t_pdf
from shortfall.parametric import (
    Distribution,
    NotAQuantileFunction,
    cornish_fisher_quantile,
    cornish_fisher_risk,
    is_monotone,
    normal_risk,
    normal_tail_mean,
    parametric_risk,
    portfolio_risk,
    student_t_risk,
)

MU = 0.02
SIGMA = 0.15


def tail_mean(
    quantile: float,
    density: Callable[[float], float],
    alpha: float,
    *,
    points: int = 2001,
) -> float:
    """``-E[X | X <= quantile]`` by Simpson over ``x = quantile - tan(theta)``."""
    top = math.pi / 2.0 - 1e-9
    step = top / (points - 1)
    total = 0.0
    for index in range(points):
        theta = index * step
        x = quantile - math.tan(theta)
        weight = 1 if index in (0, points - 1) else (4 if index % 2 else 2)
        total += weight * x * density(x) / math.cos(theta) ** 2
    return -(total * step / 3.0) / alpha


# -- the sign and tail conventions ------------------------------------------


def test_value_at_risk_is_reported_as_a_positive_loss() -> None:
    result = normal_risk(mean=0.0, volatility=SIGMA, confidence=0.99)
    assert result.value_at_risk > 0.0
    assert result.quantile < 0.0
    # And the two are the same number with opposite signs, so a caller using the
    # other convention has it without guessing which one this is.
    assert result.value_at_risk == pytest.approx(-result.quantile, abs=1e-18)


def test_confidence_is_the_confidence_not_the_tail() -> None:
    # 0.99 means the 1% tail. An implementation taking one while documenting the
    # other is wrong by an amount that grows as the tail thins.
    assert normal_risk(mean=0.0, volatility=1.0, confidence=0.99).tail_probability == pytest.approx(
        0.01
    )
    assert normal_risk(mean=0.0, volatility=1.0, confidence=0.99).value_at_risk == pytest.approx(
        -normal_ppf(0.01), rel=1e-14
    )


def test_a_large_enough_drift_makes_the_loss_negative() -> None:
    # Reported as it comes out rather than clamped. Clamping would hide the one
    # case where the confidence chosen says nothing about the portfolio.
    result = normal_risk(mean=0.5, volatility=0.05, confidence=0.95)
    assert result.value_at_risk < 0.0


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 99.0])
def test_a_confidence_outside_the_unit_interval_is_refused(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        normal_risk(mean=0.0, volatility=1.0, confidence=confidence)


def test_a_negative_volatility_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normal_risk(mean=0.0, volatility=-0.1, confidence=0.95)


# -- normal ------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99, 0.999])
def test_the_normal_value_at_risk_is_the_scaled_quantile(confidence: float) -> None:
    result = normal_risk(mean=MU, volatility=SIGMA, confidence=confidence)
    assert result.value_at_risk == pytest.approx(
        -(MU + SIGMA * normal_ppf(1.0 - confidence)), rel=1e-14
    )


@pytest.mark.parametrize("confidence", [0.95, 0.99, 0.999])
def test_the_normal_expected_shortfall_matches_numerical_integration(
    confidence: float,
) -> None:
    alpha = 1.0 - confidence
    result = normal_risk(mean=MU, volatility=SIGMA, confidence=confidence)
    numeric = tail_mean(
        result.quantile, lambda x: normal_pdf((x - MU) / SIGMA) / SIGMA, alpha
    )
    assert result.expected_shortfall == pytest.approx(numeric, rel=1e-9)


def test_the_normal_expected_shortfall_matches_its_published_multiplier() -> None:
    # At 99%, the expected shortfall of a standard normal is phi(z)/alpha,
    # which is 2.665214220345804.
    result = normal_risk(mean=0.0, volatility=1.0, confidence=0.99)
    assert result.expected_shortfall == pytest.approx(2.665214220345804, rel=1e-13)


def test_the_tail_mean_of_a_standard_normal_is_the_density_over_the_probability() -> None:
    assert normal_tail_mean(normal_ppf(0.05)) == pytest.approx(
        -normal_pdf(normal_ppf(0.05)) / 0.05, rel=1e-14
    )


# -- Student-t ---------------------------------------------------------------


@pytest.mark.parametrize("degrees", [2.5, 3.0, 5.0, 8.0, 30.0])
def test_the_student_t_expected_shortfall_matches_numerical_integration(
    degrees: float,
) -> None:
    confidence = 0.99
    alpha = 1.0 - confidence
    result = student_t_risk(
        mean=0.0, volatility=SIGMA, confidence=confidence, degrees=degrees
    )
    scale = SIGMA * math.sqrt((degrees - 2.0) / degrees)
    numeric = tail_mean(
        result.quantile, lambda x: student_t_pdf(x / scale, degrees) / scale, alpha
    )
    # 2.5 degrees of freedom is heavy enough that the quadrature itself is the
    # limiting error, which is why the tolerance is not tighter across the board.
    assert result.expected_shortfall == pytest.approx(numeric, rel=1e-6)


def test_the_student_t_is_scaled_to_the_volatility_it_was_given() -> None:
    # The trap: a raw t with v degrees of freedom has variance v/(v-2), so an
    # unscaled implementation produces a distribution 22% too wide at v = 5 —
    # and 22% is more than the difference the heavy tail is there to capture.
    degrees = 5.0
    result = student_t_risk(mean=0.0, volatility=SIGMA, confidence=0.99, degrees=degrees)
    unscaled_would_be = result.value_at_risk / math.sqrt((degrees - 2.0) / degrees)
    assert unscaled_would_be / result.value_at_risk == pytest.approx(
        math.sqrt(degrees / (degrees - 2.0)), rel=1e-12
    )


def test_heavier_tails_mean_more_risk_at_a_high_confidence() -> None:
    at = {
        degrees: student_t_risk(
            mean=0.0, volatility=SIGMA, confidence=0.99, degrees=degrees
        ).value_at_risk
        for degrees in (3.0, 5.0, 10.0, 50.0)
    }
    assert at[3.0] > at[5.0] > at[10.0] > at[50.0]
    assert at[50.0] > normal_risk(mean=0.0, volatility=SIGMA, confidence=0.99).value_at_risk


def test_many_degrees_of_freedom_converge_on_the_normal() -> None:
    heavy = student_t_risk(mean=MU, volatility=SIGMA, confidence=0.99, degrees=1e7)
    light = normal_risk(mean=MU, volatility=SIGMA, confidence=0.99)
    assert heavy.value_at_risk == pytest.approx(light.value_at_risk, rel=1e-5)
    assert heavy.expected_shortfall == pytest.approx(light.expected_shortfall, rel=1e-5)


@pytest.mark.parametrize("degrees", [2.0, 1.0, 0.5])
def test_two_or_fewer_degrees_of_freedom_are_refused(degrees: float) -> None:
    # No finite variance, so there is nothing for a volatility to match and the
    # expected shortfall is infinite.
    with pytest.raises(ValueError, match="exceed 2"):
        student_t_risk(mean=0.0, volatility=SIGMA, confidence=0.99, degrees=degrees)


# -- Cornish-Fisher -----------------------------------------------------------


def test_zero_skewness_and_kurtosis_reproduce_the_normal_exactly() -> None:
    corrected = cornish_fisher_risk(
        mean=MU, volatility=SIGMA, confidence=0.975, skewness=0.0, excess_kurtosis=0.0
    )
    plain = normal_risk(mean=MU, volatility=SIGMA, confidence=0.975)
    assert corrected.value_at_risk == pytest.approx(plain.value_at_risk, abs=1e-17)
    assert corrected.expected_shortfall == pytest.approx(plain.expected_shortfall, abs=1e-17)


def test_the_kurtosis_parameter_is_excess_not_raw() -> None:
    # The two conventions differ by exactly the value a normal takes, so a raw
    # kurtosis of 3 passed here would read as heavy tails on a normal.
    assert cornish_fisher_quantile(-2.0, 0.0, 0.0) == -2.0
    assert cornish_fisher_quantile(-2.0, 0.0, 3.0) != -2.0


def test_negative_skewness_increases_the_loss() -> None:
    # A left-skewed distribution has a worse left tail, which is the whole point
    # of applying the correction.
    left = cornish_fisher_risk(
        mean=0.0, volatility=SIGMA, confidence=0.99, skewness=-0.8, excess_kurtosis=0.0
    )
    plain = normal_risk(mean=0.0, volatility=SIGMA, confidence=0.99)
    assert left.value_at_risk > plain.value_at_risk


def test_excess_kurtosis_increases_the_loss() -> None:
    heavy = cornish_fisher_risk(
        mean=0.0, volatility=SIGMA, confidence=0.99, skewness=0.0, excess_kurtosis=2.0
    )
    plain = normal_risk(mean=0.0, volatility=SIGMA, confidence=0.99)
    assert heavy.value_at_risk > plain.value_at_risk


@pytest.mark.parametrize(
    ("skewness", "excess_kurtosis"), [(-0.5, 1.0), (0.3, 2.0), (0.0, 0.0), (-1.0, 3.0)]
)
def test_the_cornish_fisher_expected_shortfall_matches_numerical_integration(
    skewness: float, excess_kurtosis: float
) -> None:
    # The closed form comes from the four tail moments of the normal. This
    # integrates the implied quantile against the normal density instead, which
    # shares none of that algebra.
    confidence = 0.99
    alpha = 1.0 - confidence
    result = cornish_fisher_risk(
        mean=0.0,
        volatility=SIGMA,
        confidence=confidence,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
    edge = normal_ppf(alpha)
    lower, points = -40.0, 4001
    step = (edge - lower) / (points - 1)
    total = 0.0
    for index in range(points):
        z = lower + index * step
        weight = 1 if index in (0, points - 1) else (4 if index % 2 else 2)
        total += weight * cornish_fisher_quantile(z, skewness, excess_kurtosis) * normal_pdf(z)
    numeric = -SIGMA * (total * step / 3.0) / alpha
    assert result.expected_shortfall == pytest.approx(numeric, rel=1e-8)


def test_moments_that_break_monotonicity_are_refused() -> None:
    # Where the corrected mapping stops increasing, a higher probability maps to
    # a lower value and the number is not a quantile of anything. There is no
    # way to tell that from the number itself.
    assert not is_monotone(4.0, 0.0, upper=normal_ppf(0.01))
    with pytest.raises(NotAQuantileFunction, match="not monotone"):
        cornish_fisher_risk(
            mean=0.0, volatility=SIGMA, confidence=0.99, skewness=4.0, excess_kurtosis=0.0
        )


def test_the_refusal_names_the_moments_and_the_range_it_checked() -> None:
    with pytest.raises(NotAQuantileFunction, match=r"skewness=4\.0"):
        cornish_fisher_risk(
            mean=0.0, volatility=SIGMA, confidence=0.99, skewness=4.0, excess_kurtosis=0.0
        )
    with pytest.raises(NotAQuantileFunction, match=r"z from -4 to -2\.326"):
        cornish_fisher_risk(
            mean=0.0, volatility=SIGMA, confidence=0.99, skewness=4.0, excess_kurtosis=0.0
        )


def test_monotonicity_is_judged_over_the_tail_the_number_is_read_from() -> None:
    # Not over a symmetric range around zero, and this is the case that shows
    # why. A skewness of -0.8 with no excess kurtosis — an ordinary equity
    # return series — has a negative cubic coefficient, so its slope turns
    # negative far enough out on *both* sides. It fails at z = +4 and holds
    # across the entire left tail a 99% loss is taken from.
    #
    # A symmetric check would refuse it. It would also refuse every pure
    # skewness correction there is, since the cubic coefficient is -S^2/18 and
    # is negative for any non-zero S, which makes the symmetric question the
    # wrong one rather than the strict one.
    assert not is_monotone(-0.8, 0.0)
    assert is_monotone(-0.8, 0.0, upper=normal_ppf(0.01))
    assert cornish_fisher_risk(
        mean=0.0, volatility=SIGMA, confidence=0.99, skewness=-0.8, excess_kurtosis=0.0
    ).value_at_risk > 0.0


def test_an_empty_range_is_refused_rather_than_answered() -> None:
    with pytest.raises(ValueError, match="empty range"):
        is_monotone(0.0, 0.0, lower=1.0, upper=-1.0)


def test_moderate_moments_are_accepted() -> None:
    assert is_monotone(0.0, 0.0)
    assert is_monotone(-0.5, 1.0)
    assert is_monotone(0.4, 2.0)


def test_the_accepted_region_really_is_monotone() -> None:
    # Not just that the flag says so: over the tail each of these is read from,
    # the quantile really must increase with the probability.
    for skewness, kurtosis in ((-0.5, 1.0), (0.3, 2.0), (-1.0, 3.0), (-0.8, 0.0)):
        assert is_monotone(skewness, kurtosis, upper=normal_ppf(0.01))
        previous = -math.inf
        for index in range(1, 100):
            z = -4.0 + index * (normal_ppf(0.01) + 4.0) / 100.0
            value = cornish_fisher_quantile(z, skewness, kurtosis)
            assert value > previous
            previous = value


# -- properties every distribution must satisfy ------------------------------

EVERY = [
    (Distribution.NORMAL, {}),
    (Distribution.STUDENT_T, {"degrees": 4.0}),
    (Distribution.CORNISH_FISHER, {"skewness": -0.4, "excess_kurtosis": 1.5}),
]


@pytest.mark.parametrize(("distribution", "extra"), EVERY)
def test_expected_shortfall_is_never_smaller_than_value_at_risk(
    distribution: Distribution, extra: dict[str, float]
) -> None:
    # The average of the tail cannot be less severe than its edge. This is the
    # single cheapest check that a sign has not been dropped somewhere.
    for confidence in (0.90, 0.95, 0.99, 0.999):
        result = parametric_risk(
            mean=MU,
            volatility=SIGMA,
            confidence=confidence,
            distribution=distribution,
            **extra,
        )
        assert result.expected_shortfall >= result.value_at_risk


@pytest.mark.parametrize(("distribution", "extra"), EVERY)
def test_risk_grows_with_confidence(
    distribution: Distribution, extra: dict[str, float]
) -> None:
    levels = [0.90, 0.95, 0.99, 0.995, 0.999]
    losses = [
        parametric_risk(
            mean=MU, volatility=SIGMA, confidence=c, distribution=distribution, **extra
        ).value_at_risk
        for c in levels
    ]
    assert losses == sorted(losses)


@pytest.mark.parametrize(("distribution", "extra"), EVERY)
def test_risk_is_proportional_to_volatility_when_there_is_no_drift(
    distribution: Distribution, extra: dict[str, float]
) -> None:
    one = parametric_risk(
        mean=0.0, volatility=0.1, confidence=0.99, distribution=distribution, **extra
    )
    two = parametric_risk(
        mean=0.0, volatility=0.3, confidence=0.99, distribution=distribution, **extra
    )
    assert two.value_at_risk == pytest.approx(3.0 * one.value_at_risk, rel=1e-12)
    assert two.expected_shortfall == pytest.approx(3.0 * one.expected_shortfall, rel=1e-12)


@pytest.mark.parametrize(("distribution", "extra"), EVERY)
def test_the_distribution_is_recorded_on_the_result(
    distribution: Distribution, extra: dict[str, float]
) -> None:
    result = parametric_risk(
        mean=MU, volatility=SIGMA, confidence=0.99, distribution=distribution, **extra
    )
    assert result.distribution is distribution
    assert result.horizon == 1.0


# -- horizon scaling ---------------------------------------------------------


def test_the_mean_scales_linearly_and_the_volatility_by_the_square_root() -> None:
    daily = normal_risk(mean=MU, volatility=SIGMA, confidence=0.99)
    ten = daily.scaled_to(10.0)
    assert ten.mean == pytest.approx(10.0 * MU, rel=1e-14)
    assert ten.volatility == pytest.approx(math.sqrt(10.0) * SIGMA, rel=1e-14)
    assert ten.horizon == 10.0


def test_scaling_to_one_period_changes_nothing() -> None:
    daily = normal_risk(mean=MU, volatility=SIGMA, confidence=0.99)
    assert daily.scaled_to(1.0).value_at_risk == pytest.approx(daily.value_at_risk, rel=1e-15)


def test_a_long_enough_horizon_turns_the_loss_into_a_gain() -> None:
    # Because the drift accumulates faster than the volatility. Worth pinning:
    # scaling both by the square root, which is the easy mistake, never does it.
    daily = normal_risk(mean=0.002, volatility=0.02, confidence=0.95)
    assert daily.value_at_risk > 0.0
    assert daily.scaled_to(2000.0).value_at_risk < 0.0


@pytest.mark.parametrize("horizon", [0.0, -1.0])
def test_a_non_positive_horizon_is_refused(horizon: float) -> None:
    with pytest.raises(ValueError, match="positive number of periods"):
        normal_risk(mean=MU, volatility=SIGMA, confidence=0.99).scaled_to(horizon)


# -- the portfolio entry point ------------------------------------------------

COVARIANCE = [[0.04, 0.006], [0.006, 0.09]]


def test_portfolio_risk_uses_the_quadratic_form() -> None:
    weights = [0.6, 0.4]
    variance = 0.36 * 0.04 + 2 * 0.6 * 0.4 * 0.006 + 0.16 * 0.09
    direct = normal_risk(mean=0.0, volatility=math.sqrt(variance), confidence=0.99)
    assert portfolio_risk(weights, COVARIANCE).value_at_risk == pytest.approx(
        direct.value_at_risk, rel=1e-14
    )


def test_the_drift_defaults_to_zero_rather_than_to_an_estimate() -> None:
    # The conservative convention, and the standard one: expected returns are
    # estimated far less reliably than covariances.
    assert portfolio_risk([0.5, 0.5], COVARIANCE).mean == 0.0


def test_supplied_means_are_weighted_into_the_drift() -> None:
    result = portfolio_risk([0.5, 0.5], COVARIANCE, means=[0.01, 0.03])
    assert result.mean == pytest.approx(0.02, rel=1e-15)


def test_means_must_match_the_weights() -> None:
    with pytest.raises(ValueError, match="must agree"):
        portfolio_risk([0.5, 0.5], COVARIANCE, means=[0.01])


def test_a_covariance_matrix_that_is_not_definite_is_reported_as_such() -> None:
    # Better said here than by math.sqrt, since the fault is in the matrix
    # rather than in the weights.
    indefinite = [[1.0, 2.0], [2.0, 1.0]]
    with pytest.raises(ValueError, match="positive semi-definite"):
        portfolio_risk([1.0, -1.0], indefinite)

"""Tests for second-order risk.

Four independent checks, none of which shares arithmetic with the module.

The cumulants are computed in the *original* coordinates from
``tr(Gamma Sigma)``, ``delta' Sigma delta + 0.5 tr((Gamma Sigma)^2)`` and the
two higher traces, which never touches the Cholesky factor or the Jacobi
rotation that the module's whitening depends on. The distribution function goes
against two exact cases: equal eigenvalues with no linear term is a scaled
chi-square, and a single factor is a quadratic in one normal whose distribution
function is elementary. The risk numbers go against a simulation of the
original exposure. And a linear book is checked against
:func:`~shortfall.parametric.normal_risk` for exact equality, which it must have
because the expansion is exact there.

Draws use Box-Muller over ``random.Random`` so the simulation does not depend on
CPython's own normal variate algorithm.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.distributions import chi_square_cdf, normal_cdf, normal_ppf
from shortfall.linalg import Matrix, Vector, cholesky
from shortfall.parametric import normal_risk
from shortfall.quadratic import (
    NEAR_MEAN,
    DidNotSolve,
    Exposure,
    NotQuadratic,
    OutsideSupport,
    QuadraticRisk,
    StandardForm,
    cgf,
    cgf_curvature,
    cgf_slope,
    cumulants,
    delta_normal_quantile,
    domain,
    linear_form,
    negated,
    quadratic_cdf,
    quadratic_quantile,
    quadratic_risk,
    quadratic_shortfall,
    quadratic_tail,
    saddlepoint_argument,
    standard_form,
    support,
)

COVARIANCE: Matrix = [
    [0.0004, 0.00012, 0.00005],
    [0.00012, 0.0009, 0.0002],
    [0.00005, 0.0002, 0.0016],
]
DELTA: Vector = [0.5, -0.2, 0.1]
LONG_GAMMA: Matrix = [[40.0, 5.0, 0.0], [5.0, 30.0, 2.0], [0.0, 2.0, 25.0]]
SHORT_GAMMA: Matrix = [[-value for value in row] for row in LONG_GAMMA]


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return [
        [sum(left[i][k] * right[k][j] for k in range(size)) for j in range(size)]
        for i in range(size)
    ]


def _trace(matrix: Matrix) -> float:
    return sum(matrix[i][i] for i in range(len(matrix)))


def _apply(matrix: Matrix, vector: Vector) -> Vector:
    size = len(matrix)
    return [sum(matrix[i][j] * vector[j] for j in range(size)) for i in range(size)]


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def reference_cumulants(exposure: Exposure, covariance: Matrix) -> tuple[float, ...]:
    """The four cumulants from traces in the factors' own coordinates.

    Writing ``M = Gamma Sigma``, these are ``a + tr(M)/2``,
    ``delta' Sigma delta + tr(M^2)/2``, ``tr(M^3) + 3 delta' Sigma Gamma Sigma delta``
    and ``3 tr(M^4) + 12 delta' Sigma (Gamma Sigma)^2 delta``. No whitening and
    no eigenvalues, which is the point of computing them this way.
    """
    product = _multiply(exposure.gamma, covariance)
    squared = _multiply(product, product)
    cubed = _multiply(squared, product)
    fourth = _multiply(squared, squared)
    shifted = _apply(covariance, exposure.delta)
    first = _dot(shifted, exposure.delta)
    second = _dot(_apply(exposure.gamma, shifted), shifted)
    bracketed = _multiply(_multiply(exposure.gamma, covariance), exposure.gamma)
    third = _dot(_apply(bracketed, shifted), shifted)
    return (
        exposure.constant + 0.5 * _trace(product),
        first + 0.5 * _trace(squared),
        _trace(cubed) + 3.0 * second,
        3.0 * _trace(fourth) + 12.0 * third,
    )


def normals(count: int, seed: int) -> list[float]:
    source = random.Random(seed)
    out: list[float] = []
    while len(out) < count:
        first, second = source.random(), source.random()
        radius = math.sqrt(-2.0 * math.log(1.0 - first))
        angle = 2.0 * math.pi * second
        out.append(radius * math.cos(angle))
        out.append(radius * math.sin(angle))
    return out[:count]


def simulate(exposure: Exposure, covariance: Matrix, paths: int, seed: int) -> list[float]:
    """Draw the profit of the original exposure, sorted ascending."""
    factor = cholesky(covariance)
    size = len(exposure.delta)
    draws = normals(size * (paths // 2), seed)
    out: list[float] = []
    for index in range(paths // 2):
        block = draws[index * size : (index + 1) * size]
        for flip in (1.0, -1.0):
            shock = [flip * value for value in block]
            move = [
                sum(factor[i][k] * shock[k] for k in range(i + 1)) for i in range(size)
            ]
            linear = sum(exposure.delta[i] * move[i] for i in range(size))
            curved = sum(
                move[i] * exposure.gamma[i][j] * move[j]
                for i in range(size)
                for j in range(size)
            )
            out.append(exposure.constant + linear + 0.5 * curved)
    out.sort()
    return out


def theta_funded(gamma: Matrix) -> Exposure:
    """A book whose gamma is paid for by its theta, so the mean profit is zero.

    This is the configuration the comparison is interesting in. A long-gamma
    book with no theta has a large positive expected profit and the question of
    what it can lose is not the question anyone is asking.
    """
    bare = standard_form(Exposure(delta=DELTA, gamma=gamma), COVARIANCE)
    return Exposure(delta=DELTA, gamma=gamma, constant=-0.5 * sum(bare.curvature))


# -- the whitening ------------------------------------------------------------


@pytest.mark.parametrize("gamma", [LONG_GAMMA, SHORT_GAMMA])
def test_the_cumulants_match_the_traces_in_the_original_coordinates(gamma: Matrix) -> None:
    exposure = Exposure(delta=DELTA, gamma=gamma, constant=-0.003)
    form = standard_form(exposure, COVARIANCE)
    expected = reference_cumulants(exposure, COVARIANCE)
    got = cumulants(form, 4)
    for index, (one, other) in enumerate(zip(got, expected, strict=True)):
        assert one == pytest.approx(other, rel=1e-10), index


def test_a_mixed_sign_gamma_also_matches_the_traces() -> None:
    """The case where neither bound exists, so nothing can be hiding in a sign."""
    mixed: Matrix = [[40.0, 5.0, 0.0], [5.0, -30.0, 2.0], [0.0, 2.0, 12.0]]
    exposure = Exposure(delta=DELTA, gamma=mixed)
    form = standard_form(exposure, COVARIANCE)
    assert min(form.curvature) < 0.0 < max(form.curvature)
    assert support(form) == (-math.inf, math.inf)
    for one, other in zip(
        cumulants(form, 4), reference_cumulants(exposure, COVARIANCE), strict=True
    ):
        assert one == pytest.approx(other, rel=1e-10)


def test_the_linear_part_is_a_small_share_of_the_variance_here() -> None:
    """7.8% on this book, which is why a delta-only model is not a small error."""
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    share = cumulants(linear_form(form), 2)[1] / cumulants(form, 2)[1]
    assert share == pytest.approx(0.0785, abs=5e-4)


def test_a_covariance_that_does_not_match_the_exposure_is_refused() -> None:
    with pytest.raises(NotQuadratic, match="covers 3 factors"):
        standard_form(Exposure(delta=DELTA, gamma=LONG_GAMMA), [[0.01, 0.0], [0.0, 0.01]])
    with pytest.raises(NotQuadratic, match="different factor sets"):
        Exposure(delta=[1.0, 2.0], gamma=LONG_GAMMA)
    with pytest.raises(NotQuadratic, match="must be finite"):
        Exposure(delta=[1.0, float("nan"), 0.0], gamma=LONG_GAMMA)
    with pytest.raises(NotQuadratic, match="must be finite"):
        Exposure(delta=DELTA, gamma=[[float("inf"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# -- the cumulants themselves -------------------------------------------------


def test_the_third_and_fourth_cumulants_are_pure_curvature() -> None:
    """A linear book has no skewness and no excess kurtosis whatever its
    covariance is, because every term of both carries a curvature."""
    form = standard_form(Exposure(delta=DELTA, gamma=[[0.0] * 3 for _ in range(3)]), COVARIANCE)
    _, second, third, fourth = cumulants(form, 4)
    assert third == 0.0
    assert fourth == 0.0
    assert second == pytest.approx(_dot(_apply(COVARIANCE, DELTA), DELTA), rel=1e-12)
    assert form.is_linear


def test_the_skewness_takes_the_sign_of_the_gamma() -> None:
    long_form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    short_form = standard_form(theta_funded(SHORT_GAMMA), COVARIANCE)
    assert quadratic_risk(long_form, 0.99).skewness == pytest.approx(1.8872, abs=1e-3)
    assert quadratic_risk(short_form, 0.99).skewness == pytest.approx(-1.8872, abs=1e-3)
    assert quadratic_risk(long_form, 0.99).excess_kurtosis > 0.0


def test_only_four_cumulants_are_offered() -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    assert len(cumulants(form, 1)) == 1
    assert len(cumulants(form, 4)) == 4
    with pytest.raises(NotQuadratic, match="fourth order"):
        cumulants(form, 5)
    with pytest.raises(NotQuadratic, match="fourth order"):
        cumulants(form, 0)


def test_a_mismatched_standard_form_is_refused() -> None:
    with pytest.raises(NotQuadratic, match="against"):
        StandardForm(0.0, (1.0, 2.0), (1.0,))


# -- the generating function --------------------------------------------------


def test_the_generating_functions_derivatives_are_its_derivatives() -> None:
    """Against central differences, which is the only check that the three
    closed forms are derivatives of each other rather than three formulas."""
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    lower, upper = domain(form)
    assert lower == -math.inf
    assert upper == pytest.approx(1.0 / max(form.curvature), rel=1e-15)
    # Each derivative against a central difference of the one above it. A second
    # difference of K itself would be the obvious thing and is the wrong thing:
    # its truncation error is three parts in ten thousand here, which reads as a
    # disagreement rather than as the quadrature it is.
    for argument in (-40.0, -5.0, 0.0, 5.0, 15.0):
        step = 1e-4
        slope = (cgf(form, argument + step) - cgf(form, argument - step)) / (2.0 * step)
        bend = (cgf_slope(form, argument + step) - cgf_slope(form, argument - step)) / (
            2.0 * step
        )
        assert cgf_slope(form, argument) == pytest.approx(slope, rel=1e-7)
        assert cgf_curvature(form, argument) == pytest.approx(bend, rel=1e-7)
        assert cgf_curvature(form, argument) > 0.0
    assert cgf(form, 0.0) == 0.0
    assert cgf_slope(form, 0.0) == pytest.approx(cumulants(form, 1)[0], rel=1e-14)
    assert cgf_curvature(form, 0.0) == pytest.approx(cumulants(form, 2)[1], rel=1e-14)


def test_the_domain_is_set_by_the_two_extreme_eigenvalues() -> None:
    mixed = StandardForm(0.0, (0.1, 0.1, 0.1), (-4.0, 0.0, 2.0))
    assert domain(mixed) == (-0.25, 0.5)
    assert domain(StandardForm(0.0, (1.0,), (0.0,))) == (-math.inf, math.inf)
    for argument in (-0.25, 0.5, 1.0):
        with pytest.raises(NotQuadratic, match="finite on"):
            cgf(mixed, argument)
        with pytest.raises(NotQuadratic, match="finite on"):
            cgf_slope(mixed, argument)
        with pytest.raises(NotQuadratic, match="finite on"):
            cgf_curvature(mixed, argument)


def test_the_saddlepoint_solves_its_own_equation() -> None:
    form = standard_form(theta_funded(SHORT_GAMMA), COVARIANCE)
    for level in (-0.4, -0.1, -0.02, 0.0, 0.02, 0.04):
        argument = saddlepoint_argument(form, level)
        assert cgf_slope(form, argument) == pytest.approx(level, abs=1e-12)


def test_a_level_outside_the_support_has_no_saddlepoint() -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    low, _ = support(form)
    with pytest.raises(DidNotSolve, match="outside the support"):
        saddlepoint_argument(form, low - 0.01)


# -- the two exact oracles ----------------------------------------------------


def test_equal_eigenvalues_reproduce_the_chi_square_and_improve_with_count() -> None:
    """The expansion is asymptotic in the number of independent contributions,
    and the measured worst error over the body falls accordingly: 1.4e-02 at one
    factor, 2.8e-03 at two, 2.6e-04 at five, 5.7e-05 at ten and 1.2e-05 at
    thirty. That is the accuracy statement for everything else here, so it is
    pinned rather than described.
    """
    expected = {1: 1.39e-2, 2: 2.76e-3, 5: 2.62e-4, 10: 5.65e-5, 30: 1.22e-5}
    for count, bound in expected.items():
        scale = 0.3 / math.sqrt(count)
        form = StandardForm(
            0.0, tuple(0.0 for _ in range(count)), tuple(2.0 * scale for _ in range(count))
        )
        mean, variance = cumulants(form, 2)
        spread = math.sqrt(variance)
        worst = 0.0
        for step in range(1, 60):
            level = mean + (-3.0 + 0.1 * step) * spread
            if level <= 0.0:
                continue
            truth = chi_square_cdf(level / scale, count)
            worst = max(worst, abs(quadratic_cdf(form, level) - truth))
        assert worst == pytest.approx(bound, rel=0.05)


def test_a_single_factor_matches_the_elementary_distribution_function() -> None:
    """``a + beta u + 0.5 lambda u^2 <= x`` is an interval in ``u``, so the
    distribution function is a difference of two normal ones. Both signs of the
    curvature, because the interval is the complement in one of them."""

    def exact(constant: float, slope: float, curve: float, level: float) -> float:
        discriminant = slope * slope - 2.0 * curve * (constant - level)
        if discriminant < 0.0:
            return 0.0 if curve > 0.0 else 1.0
        root = math.sqrt(discriminant)
        first, second = (-slope - root) / curve, (-slope + root) / curve
        low, high = min(first, second), max(first, second)
        inside = normal_cdf(high) - normal_cdf(low)
        return inside if curve > 0.0 else 1.0 - inside

    for curve in (0.25, -0.25):
        form = StandardForm(0.0, (0.4,), (curve,))
        worst = 0.0
        for level in (-6.0, -2.0, -0.5, -0.2, 0.0, 0.2, 0.5, 2.0, 6.0):
            low, high = support(form)
            if not low < level < high:
                continue
            worst = max(worst, abs(quadratic_cdf(form, level) - exact(0.0, 0.4, curve, level)))
        # One factor is the worst case for the expansion and it is still inside
        # a per cent, which is the number the chi-square sweep above predicts.
        assert worst < 1.0e-2


# -- the linear case is exact -------------------------------------------------


def test_a_linear_book_reproduces_the_normal_answer_to_the_last_bit() -> None:
    """With no curvature the generating function is exactly Gaussian and the
    expansion is exact, so there is one implementation rather than two."""
    form = StandardForm(0.001, (0.02, 0.015, -0.01), (0.0, 0.0, 0.0))
    mean, variance = cumulants(form, 2)
    spread = math.sqrt(variance)
    for probability in (0.001, 0.01, 0.05, 0.5, 0.95, 0.999):
        assert quadratic_cdf(form, mean + spread * normal_ppf(probability)) == pytest.approx(
            probability, rel=1e-12
        )
        assert quadratic_quantile(form, probability) == pytest.approx(
            mean + spread * normal_ppf(probability), rel=1e-12
        )
    for confidence in (0.95, 0.99, 0.999):
        got = quadratic_risk(form, confidence)
        reference = normal_risk(mean=mean, volatility=spread, confidence=confidence)
        assert got.value_at_risk == pytest.approx(reference.value_at_risk, rel=1e-11)
        assert got.expected_shortfall == pytest.approx(reference.expected_shortfall, rel=1e-9)
        assert got.skewness == 0.0
        assert got.excess_kurtosis == 0.0
        assert got.floor == math.inf
        assert not got.exceeds_the_floor


def test_the_delta_normal_quantile_is_the_linear_forms_own() -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    linear = linear_form(form)
    assert linear.curvature == (0.0, 0.0, 0.0)
    assert linear.linear == form.linear
    for probability in (0.001, 0.01, 0.05):
        assert delta_normal_quantile(form, probability) == pytest.approx(
            quadratic_quantile(linear, probability), rel=1e-11
        )


# -- the floor ----------------------------------------------------------------


def test_the_floor_is_the_completed_square_and_nothing_reaches_it() -> None:
    exposure = theta_funded(LONG_GAMMA)
    form = standard_form(exposure, COVARIANCE)
    low, high = support(form)
    assert high == math.inf
    expected = form.constant - 0.5 * sum(
        slope * slope / curve
        for slope, curve in zip(form.linear, form.curvature, strict=True)
    )
    assert low == pytest.approx(expected, rel=1e-14)
    assert low == pytest.approx(-0.047077, abs=1e-6)
    draws = simulate(exposure, COVARIANCE, 200_000, seed=5)
    assert draws[0] > low
    # And it gets close, so the bound is not loose: four hundred thousand draws
    # come within 2e-05 of it.
    assert draws[0] - low < 2.0e-5


def test_a_short_gamma_book_has_a_ceiling_and_no_floor() -> None:
    form = standard_form(theta_funded(SHORT_GAMMA), COVARIANCE)
    low, high = support(form)
    assert low == -math.inf
    assert high == pytest.approx(0.047077, abs=1e-6)
    assert quadratic_risk(form, 0.99).floor == math.inf


def test_the_value_at_risk_of_a_long_gamma_book_converges_on_its_floor() -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    floor = -support(form)[0]
    previous = 0.0
    for confidence in (0.9, 0.95, 0.99, 0.999, 0.99999):
        risk = quadratic_risk(form, confidence)
        assert previous < risk.value_at_risk < risk.expected_shortfall <= floor
        previous = risk.value_at_risk
    far = quadratic_risk(form, 0.99999)
    assert far.expected_shortfall / floor > 0.999
    assert not far.exceeds_the_floor


def test_the_delta_normal_number_passes_through_the_floor_at_a_measurable_point() -> None:
    """Above 65.9% confidence a delta-normal model on this book quotes a loss
    larger than the book can sustain, and at 99% it is 1.46 times the worst case.

    The reason is not subtle once it is written down: the book pays theta for
    its gamma, a delta-normal model charges the theta and drops the gamma, and
    the gamma is what the theta bought.
    """
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    floor = -support(form)[0]
    for confidence, multiple in ((0.9, 1.21), (0.95, 1.29), (0.99, 1.46), (0.999, 1.64)):
        quoted = -delta_normal_quantile(form, 1.0 - confidence)
        assert quoted / floor == pytest.approx(multiple, abs=5e-3)
    low, high = 0.5, 1.0 - 1e-12
    for _ in range(200):
        middle = 0.5 * (low + high)
        if -delta_normal_quantile(form, 1.0 - middle) < floor:
            low = middle
        else:
            high = middle
    assert 0.5 * (low + high) == pytest.approx(0.6589, abs=1e-3)


def test_the_delta_normal_number_reports_a_short_gamma_book_as_riskless() -> None:
    """A negative value at risk at 95%, against a true loss of 7.6%.

    The same mechanism with the sign reversed: the book receives the theta, a
    delta-normal model credits it and drops the gamma it was paid for taking.
    This is the dangerous direction of the same error.
    """
    form = standard_form(theta_funded(SHORT_GAMMA), COVARIANCE)
    quoted = -delta_normal_quantile(form, 0.05)
    assert quoted < 0.0
    truth = quadratic_risk(form, 0.95)
    assert truth.value_at_risk == pytest.approx(0.0762, abs=1e-3)
    assert quadratic_risk(form, 0.99).value_at_risk == pytest.approx(0.1404, abs=1e-3)
    assert -delta_normal_quantile(form, 0.001) < quadratic_risk(form, 0.999).value_at_risk


def test_a_form_with_no_coefficients_at_all_is_a_point_mass() -> None:
    form = StandardForm(0.03, (0.0, 0.0), (0.0, 0.0))
    assert support(form) == (0.03, 0.03)
    assert quadratic_cdf(form, 0.02) == 0.0
    assert quadratic_cdf(form, 0.04) == 1.0
    assert quadratic_quantile(form, 0.5) == 0.03


def test_a_flat_direction_with_a_delta_in_it_removes_both_bounds() -> None:
    form = StandardForm(0.0, (0.0, 0.1), (2.0, 0.0))
    assert support(form) == (-math.inf, math.inf)
    assert support(StandardForm(0.0, (0.0, 0.0), (2.0, 0.0))) == (0.0, math.inf)


# -- the risk numbers against a simulation ------------------------------------


@pytest.mark.parametrize(
    ("gamma", "label"), [(LONG_GAMMA, "long"), (SHORT_GAMMA, "short")]
)
def test_the_risk_numbers_match_a_simulation_of_the_original_exposure(
    gamma: Matrix, label: str
) -> None:
    exposure = theta_funded(gamma)
    form = standard_form(exposure, COVARIANCE)
    draws = simulate(exposure, COVARIANCE, 200_000, seed=3)
    assert sum(draws) / len(draws) == pytest.approx(0.0, abs=4e-4)
    for confidence, tolerance in ((0.95, 0.03), (0.99, 0.02), (0.999, 0.02)):
        index = int((1.0 - confidence) * len(draws))
        empirical = -draws[index]
        tail_mean = -sum(draws[:index]) / index
        risk = quadratic_risk(form, confidence)
        assert risk.value_at_risk == pytest.approx(empirical, rel=tolerance)
        assert risk.expected_shortfall == pytest.approx(tail_mean, rel=tolerance)
        assert risk.expected_shortfall >= risk.value_at_risk


def test_the_expected_shortfall_is_never_below_the_value_at_risk() -> None:
    for gamma in (LONG_GAMMA, SHORT_GAMMA):
        form = standard_form(theta_funded(gamma), COVARIANCE)
        for confidence in (0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999):
            risk = quadratic_risk(form, confidence)
            assert risk.expected_shortfall >= risk.value_at_risk
            assert risk.quantile == -risk.value_at_risk


def test_the_shortfall_of_a_linear_form_is_the_closed_form() -> None:
    """The quadrature over the distribution function has to reproduce
    ``-mean + sigma phi(z) / alpha``, which is the one case where the tail mean
    is known exactly."""
    form = StandardForm(0.0005, (0.01, 0.008), (0.0, 0.0))
    mean, variance = cumulants(form, 2)
    spread = math.sqrt(variance)
    for alpha in (0.1, 0.05, 0.01, 0.001):
        reference = normal_risk(mean=mean, volatility=spread, confidence=1.0 - alpha)
        assert -quadratic_shortfall(form, alpha) == pytest.approx(
            reference.expected_shortfall, rel=1e-9
        )


# -- the two tails ------------------------------------------------------------


def test_the_upper_tail_is_computed_rather_than_complemented() -> None:
    """Six standard deviations up, ``1 - F`` is identically zero and the tail is
    not. That is the whole reason :func:`negated` exists."""
    # A book whose curvature is small beside its delta, so the upper tail is
    # nearly Gaussian and thin enough for the complement to underflow. The
    # options book is the wrong example here for an instructive reason: a sum of
    # chi-squares has an exponential upper tail, and that book still has 7.8e-05
    # of probability nine standard deviations up, where a normal has 1e-19.
    form = StandardForm(0.0, (1.0, 1.0, 1.0), (0.01, 0.02, 0.03))
    mean, variance = cumulants(form, 2)
    level = mean + 10.0 * math.sqrt(variance)
    assert level < support(form)[1]
    assert 1.0 - quadratic_cdf(form, level) == 0.0
    tail = quadratic_tail(form, level)
    assert 0.0 < tail < 1e-16
    assert tail == pytest.approx(1.29e-21, rel=0.05)


def test_a_long_gamma_books_upper_tail_is_exponential_not_gaussian() -> None:
    """Which is why the complement underflows for one book and not the other: a
    sum of chi-squares keeps 7.8e-05 of its probability nine standard deviations
    above the mean, where a normal keeps 1e-19."""
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    mean, variance = cumulants(form, 2)
    level = mean + 9.0 * math.sqrt(variance)
    assert quadratic_tail(form, level) == pytest.approx(7.79e-5, rel=0.05)


def test_the_two_tails_agree_where_both_are_computable() -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    mean = cumulants(form, 1)[0]
    spread = math.sqrt(cumulants(form, 2)[1])
    # They are two independent expansions, not two sides of one identity, so
    # their sum is one only to the expansion's own accuracy. The worst gap over
    # the body is 1.09e-02 on this three-factor book, which is a lower bound on
    # the error of each and is worth having as a number -- a strongly skewed
    # three-factor book is close to the one-factor end of the accuracy sweep
    # above, not the thirty-factor end.
    worst = 0.0
    for offset in (-1.5, -0.5, 0.0, 0.5, 1.5):
        level = mean + offset * spread
        total = quadratic_cdf(form, level) + quadratic_tail(form, level)
        worst = max(worst, abs(total - 1.0))
    assert worst == pytest.approx(1.09e-2, rel=0.1)


def test_negating_twice_is_the_identity() -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    twice = negated(negated(form))
    assert twice.constant == form.constant
    assert twice.linear == form.linear
    assert twice.curvature == form.curvature


def test_the_distribution_function_is_monotone_and_bounded() -> None:
    for gamma in (LONG_GAMMA, SHORT_GAMMA):
        form = standard_form(theta_funded(gamma), COVARIANCE)
        mean = cumulants(form, 1)[0]
        spread = math.sqrt(cumulants(form, 2)[1])
        values = [
            quadratic_cdf(form, mean + (-5.0 + 0.25 * step) * spread) for step in range(41)
        ]
        assert values == sorted(values)
        assert all(0.0 <= value <= 1.0 for value in values)


def test_the_function_is_continuous_across_the_near_mean_branch() -> None:
    """The branch that was wrong the first time. At the mean the saddlepoint is
    zero and both correction terms vanish; a level a rounding away has a
    saddlepoint of 1e-16 and the direct form then divides two numbers that agree
    to every bit. The two sides of the threshold have to meet."""
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    mean = cumulants(form, 1)[0]
    spread = math.sqrt(cumulants(form, 2)[1])
    at_mean = quadratic_cdf(form, mean)
    # NEAR_MEAN is measured on t sqrt(k2), and t is about (x - mean) / k2 near
    # the mean, so the threshold is at |x - mean| of NEAR_MEAN standard
    # deviations and this level sits just the other side of it.
    outside = mean + 1.5 * NEAR_MEAN * spread
    assert quadratic_cdf(form, outside) == pytest.approx(at_mean, abs=1e-5)
    assert quadratic_cdf(form, mean + 1e-18) == at_mean


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.5])
def test_a_quantile_needs_a_probability(probability: float) -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    with pytest.raises(OutsideSupport, match="strictly inside"):
        quadratic_quantile(form, probability)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 2.0])
def test_a_risk_estimate_needs_a_confidence(confidence: float) -> None:
    form = standard_form(theta_funded(LONG_GAMMA), COVARIANCE)
    with pytest.raises(OutsideSupport, match="strictly inside"):
        quadratic_risk(form, confidence)


def test_a_risk_estimate_is_a_plain_record() -> None:
    record = QuadraticRisk(
        value_at_risk=0.05,
        expected_shortfall=0.06,
        quantile=-0.05,
        confidence=0.99,
        floor=0.04,
        cumulants=(0.0, 0.0004, 1e-6, 1e-8),
    )
    assert record.exceeds_the_floor
    assert record.skewness == pytest.approx(1e-6 / math.pow(0.0004, 1.5), rel=1e-12)
    assert record.excess_kurtosis == pytest.approx(1e-8 / 0.0004**2, rel=1e-12)

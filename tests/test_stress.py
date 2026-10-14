"""The worst a plausible set of factor moves can do.

The reference throughout is a random search over the sphere. It is a weak
optimiser and a strong test: the exact answer has to beat every point it finds,
on every instance, and where the exact answer is a closed form the two agree to
the last few digits. Nothing here compares the module to itself except where
the comparison is between a general solver and a special-case formula.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from shortfall.quadratic import Exposure, NotQuadratic, StandardForm, whitening
from shortfall.stress import (
    MaximumLoss,
    linear_maximum_loss,
    maximum_loss,
    one_at_a_time,
    radius_for,
    worst_case_scenario,
)

COVARIANCE = [[0.04, 0.012, 0.0], [0.012, 0.09, -0.01], [0.0, -0.01, 0.0225]]
DELTA = [1.0, -2.0, 0.5]
GAMMA = [[0.5, 0.1, 0.0], [0.1, -0.3, 0.05], [0.0, 0.05, 0.2]]


def _loss(form: StandardForm, point: list[float]) -> float:
    profit = form.constant + math.fsum(
        form.linear[i] * point[i] + 0.5 * form.curvature[i] * point[i] ** 2
        for i in range(form.size)
    )
    return -profit


def _random_search(form: StandardForm, radius: float, draws: int, seed: int) -> float:
    """The best loss found on the sphere by brute force. An independent reference."""
    generator = random.Random(seed)
    best = -math.inf
    for _ in range(draws):
        direction = [generator.gauss(0.0, 1.0) for _ in range(form.size)]
        norm = math.sqrt(math.fsum(value * value for value in direction))
        point = [radius * value / norm for value in direction]
        best = max(best, _loss(form, point))
    return best


def _spd(generator: random.Random, size: int) -> list[list[float]]:
    noise = [[generator.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    return [
        [
            math.fsum(noise[i][k] * noise[j][k] for k in range(size)) + (0.4 if i == j else 0.0)
            for j in range(size)
        ]
        for i in range(size)
    ]


def _random_book(generator: random.Random, size: int) -> tuple[Exposure, list[list[float]]]:
    covariance = _spd(generator, size)
    delta = [generator.gauss(0.0, 1.0) for _ in range(size)]
    noise = [[generator.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    gamma = [[0.5 * (noise[i][j] + noise[j][i]) for j in range(size)] for i in range(size)]
    return Exposure(delta=delta, gamma=gamma), covariance


# --------------------------------------------------------------------------
# The radius
# --------------------------------------------------------------------------


def test_the_radius_grows_with_the_factor_set() -> None:
    radii = [radius_for(0.99, factors) for factors in (1, 3, 10, 50)]
    assert radii == pytest.approx([2.5758, 3.3682, 4.8176, 8.7266], abs=5e-4)
    assert all(later > earlier for earlier, later in itertools.pairwise(radii))


def test_a_three_sigma_ellipsoid_is_not_one_stress_test() -> None:
    # The whole reason the radius has to be named by a probability: "three
    # standard deviations" covers almost everything over one factor and
    # essentially nothing over fifty.
    from shortfall.distributions import chi_square_cdf

    assert chi_square_cdf(9.0, 1) == pytest.approx(0.9973, abs=1e-4)
    assert chi_square_cdf(9.0, 3) == pytest.approx(0.9707, abs=1e-4)
    assert chi_square_cdf(9.0, 10) == pytest.approx(0.4679, abs=1e-4)
    assert chi_square_cdf(9.0, 50) < 1e-10


def test_the_radius_refuses_an_empty_factor_set() -> None:
    with pytest.raises(NotQuadratic, match="at least one factor"):
        radius_for(0.99, 0)


# --------------------------------------------------------------------------
# The exact maximum, against random search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_the_exact_maximum_beats_random_search_on_every_instance(size: int) -> None:
    generator = random.Random(90 + size)
    closest = 0.0
    for trial in range(60):
        exposure, covariance = _random_book(generator, size)
        form = whitening(exposure, covariance).form
        radius = generator.uniform(0.5, 4.0)
        exact = maximum_loss(form, radius)
        found = _random_search(form, radius, 2000, trial)
        assert exact.loss >= found - 1e-9, (size, trial, exact, found)
        if exact.loss > 1.0:
            closest = max(closest, found / exact.loss)
    # The search does get close on some instance, which is what makes the
    # comparison informative rather than vacuous.
    assert closest > 0.95


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_the_maximum_sits_on_the_boundary_unless_it_is_interior(size: int) -> None:
    generator = random.Random(200 + size)
    for _ in range(60):
        exposure, covariance = _random_book(generator, size)
        form = whitening(exposure, covariance).form
        radius = generator.uniform(0.5, 4.0)
        solved = maximum_loss(form, radius)
        norm = math.sqrt(math.fsum(value * value for value in solved.point))
        if solved.interior:
            assert norm <= radius + 1e-12
            assert solved.multiplier == 0.0
            assert max(form.curvature) < 0.0 or min(form.curvature) > 0.0
        else:
            assert norm == pytest.approx(radius, abs=1e-8)


def test_the_loss_grows_with_the_radius() -> None:
    form = whitening(Exposure(delta=DELTA, gamma=GAMMA), COVARIANCE).form
    losses = [maximum_loss(form, radius).loss for radius in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert all(later > earlier for earlier, later in itertools.pairwise(losses))


def test_the_stationarity_condition_holds_at_the_answer() -> None:
    # (m - b_i) u_i = g_i, in loss coordinates. The equation the solver claims
    # to solve, checked at the point it returns.
    generator = random.Random(31)
    worst = 0.0
    for _ in range(80):
        exposure, covariance = _random_book(generator, 3)
        form = whitening(exposure, covariance).form
        solved = maximum_loss(form, generator.uniform(0.5, 4.0))
        if solved.hard_case:
            continue
        for i in range(form.size):
            curvature = -form.curvature[i]
            gradient = -form.linear[i]
            residual = (solved.multiplier - curvature) * solved.point[i] - gradient
            worst = max(worst, abs(residual))
    assert worst < 1e-9, worst


def test_maximum_loss_validation() -> None:
    form = whitening(Exposure(delta=DELTA, gamma=GAMMA), COVARIANCE).form
    for radius in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(NotQuadratic, match="radius must be positive and finite"):
            maximum_loss(form, radius)
    with pytest.raises(NotQuadratic, match="no factors has no worst case"):
        maximum_loss(StandardForm(0.0, (), ()), 1.0)


# --------------------------------------------------------------------------
# The hard case
# --------------------------------------------------------------------------


HARD = StandardForm(constant=0.0, linear=(0.0, 0.3, -0.1), curvature=(-2.0, -0.5, 0.0))


def test_the_hard_case_is_detected_and_solved() -> None:
    # The direction of fastest loss growth carries no linear exposure, so the
    # secular equation has no root above the floor.
    inside = maximum_loss(HARD, 0.1)
    assert not inside.hard_case
    for radius in (0.3, 1.0, 3.0):
        solved = maximum_loss(HARD, radius)
        assert solved.hard_case
        assert solved.multiplier == pytest.approx(2.0)
        assert math.sqrt(math.fsum(v * v for v in solved.point)) == pytest.approx(
            radius, abs=1e-12
        )
        # And it still beats a dense search.
        assert solved.loss >= _random_search(HARD, radius, 40_000, 5) - 1e-9


def test_the_hard_case_has_a_closed_form_here_and_matches_it() -> None:
    # With the degenerate coordinate carrying no gradient the loss splits: the
    # other coordinates sit at their pinned values and the degenerate one
    # contributes 0.5 * b * (what norm is left).
    for radius in (0.3, 1.0, 3.0, 10.0):
        solved = maximum_loss(HARD, radius)
        others = [0.3 / (2.0 - 0.5), -0.1 / (2.0 - 0.0)]
        used = math.fsum(value * value for value in others)
        spare = radius * radius - used
        expected = (
            0.3 * others[0]
            + 0.5 * 0.5 * others[0] ** 2
            + (-0.1) * others[1]
            + 0.5 * 0.0 * others[1] ** 2
            + 0.5 * 2.0 * spare
        )
        assert solved.loss == pytest.approx(expected, rel=1e-12)


def test_the_hard_case_is_a_delta_hedged_short_gamma_book() -> None:
    # Not a constructed corner: this is the ordinary shape of an option book
    # whose delta has been taken off.
    exposure = Exposure(delta=[0.0, 0.0], gamma=[[-120.0, 0.0], [0.0, -20.0]])
    covariance = [[0.04, 0.0], [0.0, 0.09]]
    radius = radius_for(0.99, 2)
    scenario = worst_case_scenario(exposure, covariance, radius)
    assert scenario.detail.hard_case
    assert scenario.loss == pytest.approx(22.1048, abs=1e-3)
    # The whole move is in the first factor, at the radius in its own sigma.
    assert scenario.standardised[0] == pytest.approx(radius, abs=1e-9)
    assert scenario.standardised[1] == pytest.approx(0.0, abs=1e-12)
    # And here the naive test is exact, because the factors are uncorrelated
    # and the worst direction is an axis. The one shape where it is right.
    naive = one_at_a_time(exposure, covariance, radius)
    assert naive.worst == pytest.approx(scenario.loss, rel=1e-12)
    assert naive.total > scenario.loss


def test_an_interior_maximum_happens_only_for_a_loss_bounded_book() -> None:
    # Loss concave in every direction: positive curvature in the profit, so
    # the loss has an unconstrained maximum and a large enough ellipsoid
    # contains it.
    form = StandardForm(constant=0.0, linear=(0.2, -0.3), curvature=(1.0, 2.0))
    far = maximum_loss(form, 10.0)
    assert far.interior
    assert far.multiplier == 0.0
    near = maximum_loss(form, 0.05)
    assert not near.interior
    # Past the point where the unconstrained maximum fits, the answer stops
    # moving: that is what interior means.
    assert maximum_loss(form, 20.0).loss == pytest.approx(far.loss, rel=1e-12)


# --------------------------------------------------------------------------
# The linear closed form
# --------------------------------------------------------------------------


def test_the_linear_book_is_a_volatility_multiple() -> None:
    zeros = [[0.0] * 3 for _ in range(3)]
    form = whitening(Exposure(delta=DELTA, gamma=zeros), COVARIANCE).form
    variance = math.fsum(
        DELTA[i] * COVARIANCE[i][j] * DELTA[j] for i in range(3) for j in range(3)
    )
    for radius in (1.0, 2.5, radius_for(0.99, 3)):
        general = maximum_loss(form, radius).loss
        closed = linear_maximum_loss(DELTA, COVARIANCE, radius)
        assert general == pytest.approx(closed, rel=1e-12)
        # And the closed form really is the radius times the book's volatility,
        # computed here from the covariance directly rather than through the
        # whitening.
        assert closed == pytest.approx(radius * math.sqrt(variance), rel=1e-12)


def test_the_linear_worst_move_is_the_covariance_times_the_delta() -> None:
    zeros = [[0.0] * 3 for _ in range(3)]
    scenario = worst_case_scenario(Exposure(delta=DELTA, gamma=zeros), COVARIANCE, 2.0)
    direction = [
        -math.fsum(COVARIANCE[i][j] * DELTA[j] for j in range(3)) for i in range(3)
    ]
    norm = math.sqrt(math.fsum(value * value for value in direction))
    unit = [value / norm for value in direction]
    moved = math.sqrt(math.fsum(value * value for value in scenario.move))
    for i in range(3):
        assert scenario.move[i] / moved == pytest.approx(unit[i], abs=1e-9)


def test_linear_maximum_loss_validation() -> None:
    with pytest.raises(NotQuadratic, match="radius must be positive"):
        linear_maximum_loss(DELTA, COVARIANCE, 0.0)


# --------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------


def test_the_scenario_reprices_the_book_at_the_move_it_reports() -> None:
    # The move and the loss have to be the same statement. Evaluating the
    # exposure at the reported factor move must reproduce the reported loss.
    generator = random.Random(44)
    worst = 0.0
    for _ in range(80):
        exposure, covariance = _random_book(generator, 3)
        scenario = worst_case_scenario(exposure, covariance, generator.uniform(0.5, 4.0))
        move = scenario.move
        profit = (
            exposure.constant
            + math.fsum(exposure.delta[i] * move[i] for i in range(3))
            + 0.5
            * math.fsum(
                move[i] * exposure.gamma[i][j] * move[j] for i in range(3) for j in range(3)
            )
        )
        worst = max(worst, abs(-profit - scenario.loss))
    assert worst < 1e-10, worst


def test_the_scenario_move_sits_on_the_ellipsoid() -> None:
    exposure = Exposure(delta=DELTA, gamma=GAMMA)
    radius = radius_for(0.99, 3)
    scenario = worst_case_scenario(exposure, exposure_covariance := COVARIANCE, radius)
    assert scenario.mahalanobis == pytest.approx(radius, abs=1e-8)
    # The standardised move is the move in each factor's own standard
    # deviations, which is how a scenario is read out loud.
    for i in range(3):
        assert scenario.standardised[i] == pytest.approx(
            scenario.move[i] / math.sqrt(exposure_covariance[i][i]), rel=1e-12
        )


# --------------------------------------------------------------------------
# One factor at a time, and what it misses
# --------------------------------------------------------------------------


def test_the_usual_single_factor_move_is_outside_the_ellipsoid() -> None:
    # "Three standard deviations in each risk factor" puts e_i at Mahalanobis
    # length sigma_i sqrt((Sigma^-1)_ii), which is at least one and equals one
    # only for a factor uncorrelated with every other.
    generator = random.Random(13)
    outside = 0
    trials = 60
    for _ in range(trials):
        exposure, covariance = _random_book(generator, 4)
        result = one_at_a_time(exposure, covariance, 3.0, on_the_ellipsoid=False)
        if not result.inside:
            outside += 1
    assert outside == trials
    # Uncorrelated factors are the exception, and then the two agree exactly.
    diagonal = [[0.04, 0.0], [0.0, 0.09]]
    plain = Exposure(delta=[1.0, -1.0], gamma=[[0.0, 0.0], [0.0, 0.0]])
    scaled = one_at_a_time(plain, diagonal, 3.0)
    unscaled = one_at_a_time(plain, diagonal, 3.0, on_the_ellipsoid=False)
    assert unscaled.inside
    assert scaled.moves == pytest.approx(unscaled.moves)


@pytest.mark.parametrize(
    ("size", "low", "high"), [(2, 1.1, 1.8), (3, 1.6, 2.6), (5, 3.0, 5.5), (8, 5.0, 9.0)]
)
def test_one_at_a_time_understates_the_joint_worst_case(
    size: int, low: float, high: float
) -> None:
    # Both searches cover the same ellipsoid, so the only difference is that
    # one of them covers all of it. The ratio is pinned as a band because it
    # is a median over random books.
    generator = random.Random(11 + size)
    ratios = []
    radius = radius_for(0.99, size)
    for _ in range(120):
        exposure, covariance = _random_book(generator, size)
        joint = worst_case_scenario(exposure, covariance, radius).loss
        naive = one_at_a_time(exposure, covariance, radius)
        if naive.worst > 0.0:
            ratios.append(joint / naive.worst)
    ratios.sort()
    median = ratios[len(ratios) // 2]
    assert low < median < high, (size, median)
    # Never the other way round: the joint search includes every single-factor
    # move it is being compared against.
    assert min(ratios) > 1.0 - 1e-9


def test_curvature_is_about_half_the_gap() -> None:
    generator = random.Random(77)
    radius = radius_for(0.99, 5)
    quadratic_ratios: list[float] = []
    linear_ratios: list[float] = []
    for _ in range(120):
        exposure, covariance = _random_book(generator, 5)
        flat = Exposure(delta=exposure.delta, gamma=[[0.0] * 5 for _ in range(5)])
        for book, bucket in ((exposure, quadratic_ratios), (flat, linear_ratios)):
            joint = worst_case_scenario(book, covariance, radius).loss
            naive = one_at_a_time(book, covariance, radius)
            if naive.worst > 0.0:
                bucket.append(joint / naive.worst)
    quadratic_ratios.sort()
    linear_ratios.sort()
    assert quadratic_ratios[len(quadratic_ratios) // 2] > 2.5
    assert 1.5 < linear_ratios[len(linear_ratios) // 2] < 2.5


def test_the_sum_of_single_factor_losses_is_not_a_bound() -> None:
    # The number a report treats as the conservative one. The joint worst case
    # exceeds it on the great majority of random books, and by more as the
    # factor set grows.
    for size, floor in ((3, 0.75), (5, 0.9), (8, 0.95)):
        generator = random.Random(400 + size)
        radius = radius_for(0.99, size)
        exceeded = 0
        counted = 0
        ratios = []
        for _ in range(120):
            exposure, covariance = _random_book(generator, size)
            joint = worst_case_scenario(exposure, covariance, radius).loss
            naive = one_at_a_time(exposure, covariance, radius)
            if naive.total <= 0.0:
                continue
            counted += 1
            ratios.append(joint / naive.total)
            if joint > naive.total:
                exceeded += 1
        assert exceeded / counted > floor, (size, exceeded / counted)
        ratios.sort()
        assert ratios[len(ratios) // 2] > 1.5


def test_one_at_a_time_reports_which_factor_and_how_far() -> None:
    exposure = Exposure(delta=[0.0, 0.0, 0.0], gamma=[[-50.0, 0.0, 0.0]] + [[0.0] * 3] * 2)
    diagonal = [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.0225]]
    result = one_at_a_time(exposure, diagonal, 2.0)
    assert result.worst_factor == 0
    assert result.losses[1] == pytest.approx(0.0)
    assert result.moves[0] == pytest.approx(2.0 * 0.2)
    assert result.worst == pytest.approx(0.5 * 50.0 * (2.0 * 0.2) ** 2)
    assert result.total == pytest.approx(result.worst)


def test_one_at_a_time_validation() -> None:
    exposure = Exposure(delta=DELTA, gamma=GAMMA)
    with pytest.raises(NotQuadratic, match="radius must be positive"):
        one_at_a_time(exposure, COVARIANCE, -1.0)
    with pytest.raises(NotQuadratic, match="covers 3 factors and the covariance matrix 2"):
        one_at_a_time(exposure, [[1.0, 0.0], [0.0, 1.0]], 1.0)


def test_a_maximum_loss_is_frozen() -> None:
    solved = maximum_loss(HARD, 1.0)
    assert isinstance(solved, MaximumLoss)
    with pytest.raises(AttributeError):
        solved.loss = 0.0  # type: ignore[misc]

"""Value at risk bounds under dependence uncertainty.

**One exact oracle carries this file.** The uniform distribution is completely
mixable, so for uniform marginals the worst case is ``d (1 + alpha) / 2`` and
the best case is ``d alpha / 2`` in closed form, with nothing from this module
in either. Everything else — the rearrangement, the gap decomposition, the
superadditivity ratios — is checked against those two expressions first and
measured afterwards.

**The marginals' exact tail means are checked by an identity, not by a
quadrature.** The law of total expectation says
``c * E[L | L <= q(c)] + (1 - c) * E[L | L > q(c)]`` is the unrestricted mean,
whatever ``c`` is, and that holds algebraically for all four families here. It
uses both tail means at once and knows nothing about how either was derived, so
it catches the mistake a numerical check would have to be lucky to catch: a
sign, a missing factor of ``c``, a limit taken at the wrong end.

**The normal marginal is checked against the package's own closed forms.**
:func:`shortfall.parametric.normal_risk` computes value at risk and expected
shortfall from the other sign convention entirely, and the two agree to 7e-18
and exactly.

The rest is the structure the module claims: that the mean row sum is invariant
under rearrangement, that ``gap`` splits into a quadrature part and a mixing
part with no residue, that the attained bound converges as one over the number
of cells, and that which grid is available follows from which endpoint of the
support is finite. Then the measurements, as bands rather than digits, and the
refusals.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise

import pytest

from shortfall.bounds import (
    Arrangement,
    BestCase,
    LossTail,
    NoTailMean,
    NotIncreasing,
    WorstCase,
    best_case_value_at_risk,
    completely_mixable_level,
    dependence_bounds,
    exponential_loss,
    normal_loss,
    pareto_loss,
    rearrange,
    uniform_loss,
    worst_case_value_at_risk,
)
from shortfall.parametric import normal_risk

LEVEL = 0.99
CELLS = 512


def uniform_book(positions: int) -> list[LossTail]:
    return [uniform_loss() for _ in range(positions)]


def worst_uniform_exact(positions: int) -> float:
    """``d (1 + alpha) / 2``, the completely mixable level on the unit interval."""
    return positions * (1.0 + LEVEL) / 2.0


def best_uniform_exact(positions: int) -> float:
    """``d alpha / 2``, the same argument applied to the lower tail."""
    return positions * LEVEL / 2.0


# -- the exact oracle --------------------------------------------------------


@pytest.mark.parametrize("positions", [2, 3, 4, 5, 8, 16])
def test_uniform_bound_above_is_the_closed_form(positions: int) -> None:
    """The proved bound is the mixable level, with no numerics in it."""
    worst = worst_case_value_at_risk(uniform_book(positions), confidence=LEVEL, points=CELLS)
    assert worst.upper == pytest.approx(worst_uniform_exact(positions), abs=1e-13)
    assert completely_mixable_level(uniform_book(positions), confidence=LEVEL) == worst.upper


@pytest.mark.parametrize("positions", [2, 3, 4, 5, 8, 16])
def test_uniform_worst_case_attained_bound_is_within_two_cells(positions: int) -> None:
    """The attained bound is short of the closed form by grid, and only by grid."""
    worst = worst_case_value_at_risk(uniform_book(positions), confidence=LEVEL, points=CELLS)
    exact = worst_uniform_exact(positions)
    cell = positions * (1.0 - LEVEL) / CELLS
    assert worst.lower <= exact
    assert exact - worst.lower <= 2.0 * cell


@pytest.mark.parametrize("positions", [2, 4, 8, 16])
def test_the_corrected_estimate_is_exact_where_the_cells_divide(positions: int) -> None:
    """512 cells split evenly into these counts, and then the answer is rounding.

    This is the sharpest statement in the file: taking the exact bound and
    subtracting only the measured non-mixability does not merely improve on the
    attained bound, it reproduces the closed form to the last few bits, which
    nothing but a correct rearrangement could do.
    """
    assert CELLS % positions == 0
    worst = worst_case_value_at_risk(uniform_book(positions), confidence=LEVEL, points=CELLS)
    exact = worst_uniform_exact(positions)
    assert worst.estimate == pytest.approx(exact, abs=1e-13)
    assert abs(worst.estimate - exact) < abs(worst.lower - exact) / 1000.0


@pytest.mark.parametrize("positions", [3, 5])
def test_where_the_cells_do_not_divide_one_cell_is_the_floor(positions: int) -> None:
    """And the shortfall is one cell, not an arbitrary amount.

    512 points cannot be partitioned into three or five equal row sums, so the
    flattest arrangement is one grid step from flat and the corrected estimate
    inherits exactly that. The two cases are measured apart rather than given a
    single loose tolerance.
    """
    assert CELLS % positions != 0
    worst = worst_case_value_at_risk(uniform_book(positions), confidence=LEVEL, points=CELLS)
    exact = worst_uniform_exact(positions)
    cell = positions * (1.0 - LEVEL) / CELLS
    shortfall = exact - worst.estimate
    assert 0.1 * cell <= shortfall <= 1.1 * cell


@pytest.mark.parametrize("positions", [2, 3, 5, 8])
def test_uniform_best_case_matches_the_lower_closed_form(positions: int) -> None:
    """The lower tail is uniform too, so the best case has a closed form.

    Its cells are ``alpha / N`` wide against ``(1 - alpha) / N`` for the upper
    tail — ninety-nine times as wide at this level — so the same number of
    points buys two fewer digits here, which is a property of where the tail
    sits and not of the algorithm.
    """
    best = best_case_value_at_risk(uniform_book(positions), confidence=LEVEL, points=CELLS)
    exact = best_uniform_exact(positions)
    cell = positions * LEVEL / CELLS
    assert best.lower == pytest.approx(exact, abs=1e-13)
    assert best.upper >= exact
    assert best.upper - exact <= 5.0 * cell
    assert abs(best.estimate - exact) <= 2.0 * cell


@pytest.mark.parametrize("positions", [2, 3, 5, 8])
def test_uniform_brackets_contain_the_exact_answer(positions: int) -> None:
    bounds = dependence_bounds(uniform_book(positions), confidence=LEVEL, points=CELLS)
    worst_exact = worst_uniform_exact(positions)
    best_exact = best_uniform_exact(positions)
    assert bounds.worst.lower <= worst_exact <= bounds.worst.upper
    assert bounds.best.lower <= best_exact <= bounds.best.upper
    low, high = bounds.enclosing
    assert low <= best_exact <= worst_exact <= high


def test_a_single_marginal_has_no_dependence_to_be_uncertain_about() -> None:
    """With one position the worst case is the marginal's own value at risk.

    Which the module has to recover without a special case: a one-column
    rearrangement cannot move anything, the mean row sum is the marginal's
    discretised tail mean, and the corrected estimate lands back on the
    quantile. The bound above is still the expected shortfall, correctly,
    because the inequality does not know there is only one position.
    """
    (tail,) = uniform_book(1)
    worst = worst_case_value_at_risk([tail], confidence=LEVEL, points=CELLS)
    assert worst.lower == pytest.approx(tail.value_at_risk(LEVEL), abs=1e-15)
    assert worst.estimate == pytest.approx(tail.value_at_risk(LEVEL), abs=1e-4)
    assert worst.upper == pytest.approx(tail.expected_shortfall(LEVEL), abs=1e-15)
    assert worst.comonotonic == pytest.approx(tail.value_at_risk(LEVEL), abs=1e-15)
    assert not worst.mixes()


# -- the exact tail means ----------------------------------------------------


@pytest.mark.parametrize(
    ("tail", "mean"),
    [
        (uniform_loss(0.0, 3.0), 1.5),
        (uniform_loss(-1.0, 1.0), 0.0),
        (exponential_loss(2.0), 2.0),
        (pareto_loss(2.5, 1.5), 1.5 * 2.5 / 1.5),
        (pareto_loss(1.2), 1.2 / 0.2),
        (normal_loss(0.3, 0.7), 0.3),
    ],
)
@pytest.mark.parametrize("level", [0.01, 0.2, 0.5, 0.95, 0.99, 0.999])
def test_law_of_total_expectation_ties_the_two_tail_means(
    tail: LossTail, mean: float, level: float
) -> None:
    """``c * lower + (1 - c) * upper == mean``, exactly, at every level.

    An identity between the two closed forms that uses neither derivation, so a
    wrong constant in either one fails it.
    """
    recombined = level * tail.lower_tail_mean(level) + (1.0 - level) * tail.expected_shortfall(
        level
    )
    assert recombined == pytest.approx(mean, abs=1e-11, rel=1e-11)


@pytest.mark.parametrize("level", [0.9, 0.95, 0.99, 0.995])
def test_the_normal_marginal_agrees_with_the_parametric_closed_forms(level: float) -> None:
    """Against :mod:`shortfall.parametric`, which works in signed returns."""
    tail = normal_loss(0.0, 0.02)
    reference = normal_risk(mean=0.0, volatility=0.02, confidence=level)
    assert tail.value_at_risk(level) == pytest.approx(reference.value_at_risk, abs=1e-16)
    assert tail.expected_shortfall(level) == pytest.approx(
        reference.expected_shortfall, abs=1e-16
    )


@pytest.mark.parametrize(
    "tail",
    [uniform_loss(), exponential_loss(), pareto_loss(2.0), normal_loss(0.0, 0.02)],
)
def test_the_tail_mean_exceeds_the_quantile_and_the_lower_mean_does_not(
    tail: LossTail,
) -> None:
    for level in (0.5, 0.9, 0.99):
        assert tail.lower_tail_mean(level) < tail.value_at_risk(level)
        assert tail.expected_shortfall(level) > tail.value_at_risk(level)


# -- the rearrangement itself ------------------------------------------------


def test_rearrangement_leaves_the_mean_row_sum_alone() -> None:
    """The invariant the whole gap decomposition rests on.

    Columns are permuted independently, so each keeps its own multiset of
    values and the total is untouched. If that failed, the mean row sum would
    stop being the discretised tail sum and the split of ``gap`` would be
    meaningless.
    """
    columns = [
        [0.3, 1.5, 2.25, 9.0, 11.5],
        [-2.0, 0.5, 0.75, 4.0, 40.0],
        [1.0, 1.0, 1.0, 2.0, 3.0],
    ]
    before = math.fsum(math.fsum(column) for column in columns) / 5
    for maximise in (True, False):
        arrangement = rearrange(columns, maximise_minimum=maximise)
        assert arrangement.mean == pytest.approx(before, abs=1e-14)
        assert arrangement.minimum <= arrangement.mean <= arrangement.maximum


def test_the_objective_is_on_the_right_side_of_the_mean() -> None:
    columns = [[float(k) for k in range(40)] for _ in range(4)]
    pushing_up = rearrange(columns, maximise_minimum=True)
    pushing_down = rearrange(columns, maximise_minimum=False)
    assert pushing_up.objective <= pushing_up.mean + 1e-12
    assert pushing_down.objective >= pushing_down.mean - 1e-12
    # Forty points over four columns divides, so both directions reach flat.
    assert pushing_up.objective == pytest.approx(pushing_up.mean, abs=1e-12)
    assert pushing_down.objective == pytest.approx(pushing_down.mean, abs=1e-12)


def test_a_flat_set_of_columns_is_already_flat() -> None:
    columns = [[2.0] * 16 for _ in range(3)]
    arrangement = rearrange(columns)
    assert arrangement.spread == 0.0
    assert arrangement.flatness == 0.0
    assert arrangement.objective == pytest.approx(6.0, abs=1e-15)
    assert arrangement.settled


def test_two_columns_of_a_progression_flatten_exactly() -> None:
    """The one case flatness is obvious in: ``k`` against ``n - 1 - k``."""
    columns = [[float(k) for k in range(101)] for _ in range(2)]
    arrangement = rearrange(columns)
    assert arrangement.spread == pytest.approx(0.0, abs=1e-12)
    assert arrangement.objective == pytest.approx(100.0, abs=1e-12)


def test_the_sweep_cap_is_reported_rather_than_hidden() -> None:
    """A cap below the patience cannot settle, and says so."""
    columns = [[math.exp(k / 7.0) for k in range(60)] for _ in range(5)]
    arrangement = rearrange(columns, max_sweeps=1, patience=5)
    assert arrangement.sweeps == 1
    assert not arrangement.settled
    assert arrangement.objective <= arrangement.mean + 1e-12


def test_more_sweeps_never_report_a_worse_objective() -> None:
    columns = [[math.exp(k / 9.0) for k in range(80)] for _ in range(6)]
    short = rearrange(columns, max_sweeps=1, patience=1)
    long = rearrange(columns, max_sweeps=200, patience=8)
    assert long.objective >= short.objective - 1e-12


# -- the structure the module claims -----------------------------------------


@pytest.mark.parametrize(
    "book",
    [
        uniform_book(4),
        [exponential_loss() for _ in range(4)],
        [pareto_loss(2.0) for _ in range(4)],
        [normal_loss(0.0, 0.02) for _ in range(4)],
        [normal_loss(0.0, 0.02), exponential_loss(0.015), pareto_loss(2.5, 0.004)],
    ],
)
def test_the_gap_splits_into_quadrature_and_mixing_with_no_residue(
    book: list[LossTail],
) -> None:
    bounds = dependence_bounds(book, confidence=LEVEL, points=256)
    for side in (bounds.worst, bounds.best):
        assert side.discretisation >= -1e-12
        assert side.mixing_gap >= -1e-12
        assert side.discretisation + side.mixing_gap == pytest.approx(side.gap, abs=1e-12)


@pytest.mark.parametrize(
    "book",
    [
        uniform_book(5),
        [exponential_loss() for _ in range(5)],
        [pareto_loss(2.0) for _ in range(5)],
        [normal_loss(0.01, 0.02) for _ in range(5)],
    ],
)
def test_the_interval_is_ordered_and_contains_the_comonotonic_coupling(
    book: list[LossTail],
) -> None:
    bounds = dependence_bounds(book, confidence=LEVEL, points=256)
    assert bounds.best.lower <= bounds.best.upper
    assert bounds.best.upper <= bounds.comonotonic <= bounds.worst.lower
    assert bounds.worst.lower <= bounds.worst.upper
    attainable_low, attainable_high = bounds.attainable
    enclosing_low, enclosing_high = bounds.enclosing
    assert enclosing_low <= attainable_low <= attainable_high <= enclosing_high
    assert bounds.ratio > 1.0


@pytest.mark.parametrize("positions", [2, 4, 6])
def test_the_attained_bound_converges_as_one_over_the_cells(positions: int) -> None:
    """Halving the cell count doubles the shortfall, within a tight band."""
    exact = worst_uniform_exact(positions)
    errors = []
    for cells in (64, 128, 256, 512, 1024):
        worst = worst_case_value_at_risk(
            uniform_book(positions), confidence=LEVEL, points=cells
        )
        errors.append(exact - worst.lower)
    for coarse, fine in pairwise(errors):
        assert 1.8 <= coarse / fine <= 2.2


@pytest.mark.parametrize(
    ("book", "worst_grid", "best_grid"),
    [
        (uniform_book(2), True, True),
        ([exponential_loss(), exponential_loss()], False, True),
        ([pareto_loss(2.0), pareto_loss(3.0)], False, True),
        ([normal_loss(0.0, 0.02), normal_loss(0.0, 0.01)], False, False),
        ([uniform_loss(), pareto_loss(2.0)], False, True),
    ],
)
def test_which_grid_exists_follows_from_which_endpoint_is_finite(
    book: list[LossTail], worst_grid: bool, best_grid: bool
) -> None:
    """A loss bounded above gets the worst case's check, bounded below the
    best case's, and one unbounded marginal is enough to remove either."""
    bounds = dependence_bounds(book, confidence=LEVEL, points=128)
    assert (bounds.worst.overstated is not None) is worst_grid
    assert (bounds.best.understated is not None) is best_grid


def test_the_reported_grids_sit_the_right_side_of_the_attained_ones() -> None:
    bounds = dependence_bounds(uniform_book(4), confidence=LEVEL, points=256)
    assert bounds.worst.overstated is not None
    assert bounds.best.understated is not None
    assert bounds.worst.overstated >= bounds.worst.lower
    assert bounds.best.understated <= bounds.best.upper


# -- what the measurements say -----------------------------------------------


@pytest.mark.parametrize(
    ("positions", "low", "high"),
    [(2, 1.45, 1.50), (4, 1.77, 1.81), (8, 1.91, 1.94), (16, 1.97, 2.00), (32, 1.99, 2.00)],
)
def test_pareto_superadditivity_climbs_towards_the_tail_index_ratio(
    positions: int, low: float, high: float
) -> None:
    """``theta / (theta - 1) = 2`` at tail index 2, approached from below.

    Banded rather than pinned, because the figure moves in the last digits with
    the cell count and the interpreter's sort. The bands are tight enough that
    the climb itself is the assertion: no two of them overlap.
    """
    worst = worst_case_value_at_risk(
        [pareto_loss(2.0) for _ in range(positions)], confidence=LEVEL, points=CELLS
    )
    assert low <= worst.superadditivity <= high
    assert worst.superadditivity < 2.0


@pytest.mark.parametrize(
    ("shape", "low", "high"), [(1.5, 2.6, 2.8), (3.0, 1.47, 1.50), (5.0, 1.24, 1.25)]
)
def test_a_lighter_tail_reaches_its_own_limit_sooner(
    shape: float, low: float, high: float
) -> None:
    worst = worst_case_value_at_risk(
        [pareto_loss(shape) for _ in range(8)], confidence=LEVEL, points=CELLS
    )
    limit = shape / (shape - 1.0)
    assert low <= worst.superadditivity <= high
    assert worst.superadditivity < limit
    # Eight positions is most of the way there for a light tail and not for a
    # heavy one, which is the whole content of the sweep.
    assert worst.superadditivity / limit > (0.85 if shape >= 3.0 else 0.80)


def test_exponential_superadditivity_caps_at_its_own_shortfall_ratio() -> None:
    """Expected shortfall exceeds value at risk by exactly the scale, so the
    ratio the sum can reach is a closed form of the level alone."""
    tail = exponential_loss()
    limit = tail.expected_shortfall(LEVEL) / tail.value_at_risk(LEVEL)
    assert limit == pytest.approx(1.217147, abs=1e-6)
    ratios = [
        worst_case_value_at_risk(
            [exponential_loss() for _ in range(positions)], confidence=LEVEL, points=CELLS
        ).superadditivity
        for positions in (2, 4, 8, 16)
    ]
    for ratio in ratios:
        assert ratio < limit
    for coarse, fine in pairwise(ratios):
        assert fine > coarse
    assert ratios[-1] / limit > 0.9999


def test_most_of_the_exponential_gap_is_quadrature_not_dependence() -> None:
    """The finding the gap decomposition exists for.

    Reading the ratio of the two bounds on eight exponential marginals would
    say the rearrangement left 0.15% of the bound unexplained and invite the
    reader to call that dependence. Almost all of it is the left-endpoint
    Riemann sum of the tail mean.
    """
    worst = worst_case_value_at_risk(
        [exponential_loss() for _ in range(8)], confidence=LEVEL, points=CELLS
    )
    assert worst.discretisation / worst.gap > 0.9
    assert worst.mixing_gap / worst.gap < 0.1
    assert worst.mixing_gap / worst.arrangement.mean < 1e-3


@pytest.mark.parametrize("positions", [2, 4, 8, 16])
def test_the_uniform_tail_mixes_and_the_pareto_tail_does_not(positions: int) -> None:
    uniform = worst_case_value_at_risk(
        uniform_book(positions), confidence=LEVEL, points=CELLS
    )
    pareto = worst_case_value_at_risk(
        [pareto_loss(2.0) for _ in range(positions)], confidence=LEVEL, points=CELLS
    )
    assert uniform.mixes()
    assert not pareto.mixes()
    assert uniform.mixing_gap / uniform.arrangement.mean < pareto.mixing_gap / (
        pareto.arrangement.mean
    )


def test_a_heavy_tail_becomes_mixable_once_there_are_enough_positions() -> None:
    """Which is the asymptotic sharpness of the bound, seen directly.

    At thirty-two Pareto positions the rearrangement flattens the discretised
    tail to within a grid step, so the inequality that is 27% slack at two
    positions has become an equality to the resolution available.
    """
    relative = [
        worst_case_value_at_risk(
            [pareto_loss(2.0) for _ in range(positions)], confidence=LEVEL, points=CELLS
        ).mixing_gap
        / worst_case_value_at_risk(
            [pareto_loss(2.0) for _ in range(positions)], confidence=LEVEL, points=CELLS
        ).arrangement.mean
        for positions in (2, 4, 8, 16, 32)
    ]
    for coarse, fine in pairwise(relative):
        assert fine < coarse
    assert relative[0] > 0.25
    assert relative[-1] < 1e-3


def test_a_mixed_book_spans_a_factor_of_three() -> None:
    """The headline figure, on marginals of three different shapes."""
    book = [
        normal_loss(0.0, 0.02, label="equity"),
        exponential_loss(0.015, label="credit"),
        pareto_loss(2.5, 0.004, label="operational"),
    ]
    bounds = dependence_bounds(book, confidence=LEVEL, points=CELLS)
    assert 3.2 <= bounds.ratio <= 3.6
    assert 1.20 <= bounds.worst.superadditivity <= 1.26
    assert 0.33 <= bounds.best.subadditivity <= 0.40
    assert bounds.best.upper < bounds.comonotonic < bounds.worst.lower


@pytest.mark.parametrize("level", [0.95, 0.975, 0.99, 0.995])
def test_the_interval_widens_as_the_level_rises(level: float) -> None:
    book = [pareto_loss(2.0) for _ in range(5)]
    bounds = dependence_bounds(book, confidence=level, points=256)
    assert bounds.confidence == level
    assert bounds.worst.superadditivity > 1.0
    assert bounds.best.subadditivity < 1.0


def test_nothing_in_a_result_is_outside_the_floating_point_numbers() -> None:
    """A dependence bound is a payload, so every number in it has to serialise."""
    book = [normal_loss(0.0, 0.02), exponential_loss(0.01), pareto_loss(3.0, 0.002)]
    bounds = dependence_bounds(book, confidence=LEVEL, points=128)
    payload = {
        "confidence": bounds.confidence,
        "attainable": list(bounds.attainable),
        "enclosing": list(bounds.enclosing),
        "estimated": list(bounds.estimated),
        "ratio": bounds.ratio,
        "worst": {
            "lower": bounds.worst.lower,
            "upper": bounds.worst.upper,
            "estimate": bounds.worst.estimate,
            "discretisation": bounds.worst.discretisation,
            "mixing_gap": bounds.worst.mixing_gap,
            "superadditivity": bounds.worst.superadditivity,
            "sweeps": bounds.worst.arrangement.sweeps,
            "flatness": bounds.worst.arrangement.flatness,
        },
        "best": {
            "lower": bounds.best.lower,
            "upper": bounds.best.upper,
            "estimate": bounds.best.estimate,
            "subadditivity": bounds.best.subadditivity,
        },
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_the_same_call_twice_gives_the_same_numbers() -> None:
    book = [pareto_loss(2.0), exponential_loss(1.5), uniform_loss(0.0, 4.0)]
    first = dependence_bounds(book, confidence=LEVEL, points=256)
    second = dependence_bounds(book, confidence=LEVEL, points=256)
    assert first == second


# -- refusals ----------------------------------------------------------------


def test_a_quantile_function_in_the_other_sign_convention_is_refused() -> None:
    """Which is the mistake worth catching: a signed-return quantile falls."""
    returns = LossTail(quantile=lambda u: -2.0 + 4.0 * (1.0 - u))
    with pytest.raises(NotIncreasing, match="falls from"):
        worst_case_value_at_risk([returns, returns], confidence=LEVEL, points=32)


def test_a_marginal_without_a_tail_mean_cannot_bound_the_worst_case() -> None:
    bare = LossTail(quantile=lambda u: u, upper_endpoint=1.0, lower_endpoint=0.0)
    with pytest.raises(NoTailMean, match="no upper tail mean"):
        worst_case_value_at_risk([bare, bare], confidence=LEVEL, points=32)
    with pytest.raises(NoTailMean, match="lower tail mean"):
        best_case_value_at_risk([bare, bare], confidence=LEVEL, points=32)


def test_an_empty_book_and_an_empty_grid_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one marginal"):
        worst_case_value_at_risk([], confidence=LEVEL)
    with pytest.raises(ValueError, match="at least two points"):
        worst_case_value_at_risk(uniform_book(2), confidence=LEVEL, points=1)
    with pytest.raises(ValueError, match="at least two points"):
        best_case_value_at_risk(uniform_book(2), confidence=LEVEL, points=1)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, 99.0])
def test_a_confidence_outside_the_open_unit_interval_is_refused(level: float) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        worst_case_value_at_risk(uniform_book(2), confidence=level, points=32)


def test_the_rearrangement_refuses_a_ragged_grid() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        rearrange([])
    with pytest.raises(ValueError, match="at least one row"):
        rearrange([[], []])
    with pytest.raises(ValueError, match="against 3 in the first"):
        rearrange([[1.0, 2.0, 3.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="patience is at least"):
        rearrange([[1.0, 2.0]], patience=0)
    with pytest.raises(ValueError, match="sweep cap is at least"):
        rearrange([[1.0, 2.0]], max_sweeps=0)


def test_the_factories_refuse_parameters_with_no_distribution_behind_them() -> None:
    with pytest.raises(ValueError, match="high > low"):
        uniform_loss(1.0, 1.0)
    with pytest.raises(ValueError, match="positive scale"):
        exponential_loss(0.0)
    with pytest.raises(ValueError, match="shape > 1"):
        pareto_loss(1.0)
    with pytest.raises(ValueError, match="positive scale"):
        pareto_loss(2.0, -1.0)
    with pytest.raises(ValueError, match="positive volatility"):
        normal_loss(0.0, 0.0)


def test_a_support_the_wrong_way_round_is_refused() -> None:
    with pytest.raises(ValueError, match="is empty"):
        LossTail(quantile=lambda u: u, upper_endpoint=0.0, lower_endpoint=1.0, label="broken")


def test_the_dataclasses_are_frozen() -> None:
    worst = worst_case_value_at_risk(uniform_book(2), confidence=LEVEL, points=32)
    for frozen, field in ((worst, "lower"), (worst.arrangement, "mean")):
        with pytest.raises(AttributeError):
            setattr(frozen, field, 0.0)
    assert isinstance(worst, WorstCase)
    assert isinstance(worst.arrangement, Arrangement)
    best = best_case_value_at_risk(uniform_book(2), confidence=LEVEL, points=32)
    assert isinstance(best, BestCase)


def test_the_figures_printed_in_the_readme_are_the_figures_produced() -> None:
    """The worked example in the README, at the default cell count.

    Pinned to four decimal places because they are in print; a change to the
    numerics that moves them has to move the document in the same commit.
    """
    book = [
        normal_loss(0.0, 0.02, label="equity"),
        exponential_loss(0.015, label="credit"),
        pareto_loss(2.5, 0.004, label="operational"),
    ]
    bounds = dependence_bounds(book, confidence=0.99)
    assert f"{bounds.best.upper:.4%}" == "5.0543%"
    assert f"{bounds.worst.lower:.4%}" == "17.1541%"
    assert f"{bounds.comonotonic:.4%}" == "14.0843%"
    assert f"{bounds.ratio:.2f}" == "3.39"
    assert f"{bounds.worst.superadditivity:.2f}" == "1.22"

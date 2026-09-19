"""Drawdown, time under water, and the ratios built on them.

The path used through most of this file is chosen so every answer can be worked
out on paper::

    wealth:   1.00   1.20   0.90   1.50   1.20   1.35
    returns:      +0.20  -0.25  +2/3   -0.20  +0.125

Its maximum drawdown is the fall from 1.20 to 0.90, which is 25%, recovered one
period later. A second decline from 1.50 to 1.20 is 20% and never recovers
within the sample. Nothing here is a number this code produced and was then
asserted to produce again.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from shortfall.drawdown import (
    calmar,
    downside_deviation,
    drawdown_series,
    drawdowns,
    longest_underwater,
    martin_ratio,
    maximum_drawdown,
    rolling,
    rolling_maximum_drawdown,
    rolling_ulcer_index,
    rolling_volatility,
    sortino,
    ulcer_index,
    wealth_curve,
)
from shortfall.series import Convention, ReturnSeries, TooShort

PATH = [0.20, -0.25, 2.0 / 3.0, -0.20, 0.125]
WEALTH = [1.00, 1.20, 0.90, 1.50, 1.20, 1.35]
LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


# -- the wealth curve --------------------------------------------------------


def test_the_wealth_curve_compounds() -> None:
    curve = wealth_curve(PATH)
    assert curve == pytest.approx(WEALTH)


def test_the_curve_has_one_more_point_than_there_are_returns() -> None:
    assert len(wealth_curve(PATH)) == len(PATH) + 1


def test_the_first_point_is_before_anything_has_happened() -> None:
    assert wealth_curve(PATH, initial=250.0)[0] == 250.0
    assert wealth_curve(PATH, initial=250.0)[1] == pytest.approx(300.0)


def test_log_returns_are_converted_before_compounding() -> None:
    """Compounding log returns with ``1 + r`` is wrong and looks plausible.

    The two agree to first order, so a short series of small returns gives
    almost the same curve and a long one does not. Checked against the exact
    answer, which is the exponential of the running sum.
    """
    simple = ReturnSeries(name="p", values=tuple(PATH))
    logged = simple.to_log()
    assert logged.convention is Convention.LOG
    assert wealth_curve(logged) == pytest.approx(WEALTH)

    running = 0.0
    for index, value in enumerate(logged.values):
        running += value
        assert wealth_curve(logged)[index + 1] == pytest.approx(math.exp(running))


def test_a_raw_sequence_is_taken_as_simple_returns() -> None:
    as_series = wealth_curve(ReturnSeries(name="p", values=tuple(PATH)))
    assert wealth_curve(list(PATH)) == pytest.approx(as_series)


def test_an_empty_path_has_no_statistics() -> None:
    with pytest.raises(TooShort):
        wealth_curve([])


def test_a_starting_value_is_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        wealth_curve(PATH, initial=0.0)


# -- the drawdown series -----------------------------------------------------


def test_the_drawdown_series_is_the_distance_below_the_running_peak() -> None:
    assert drawdown_series(PATH) == pytest.approx([0.0, 0.0, 0.25, 0.0, 0.2, 0.1])


def test_the_drawdown_series_is_zero_at_every_new_high() -> None:
    series = drawdown_series(PATH)
    curve = wealth_curve(PATH)
    running = curve[0]
    for position, value in enumerate(curve):
        running = max(running, value)
        if value >= running:
            assert series[position] == pytest.approx(0.0)


def test_drawdown_is_never_negative_and_never_above_one() -> None:
    rng = random.Random(4)
    path = [rng.gauss(0.0005, 0.02) for _ in range(500)]
    for value in drawdown_series(path):
        assert 0.0 <= value < 1.0


def test_a_monotonically_rising_path_never_draws_down() -> None:
    assert drawdown_series([0.01] * 20) == pytest.approx([0.0] * 21)


# -- the maximum drawdown ----------------------------------------------------


def test_maximum_drawdown_matches_the_hand_computed_path() -> None:
    worst = maximum_drawdown(PATH, index=LABELS)
    assert worst.depth == pytest.approx(0.25)
    assert worst.peak_position == 1
    assert worst.trough_position == 2
    assert worst.recovery_position == 3
    assert worst.peak_value == pytest.approx(1.20)
    assert worst.trough_value == pytest.approx(0.90)
    assert (worst.peak_label, worst.trough_label, worst.recovery_label) == (
        "Tue",
        "Wed",
        "Thu",
    )


def test_maximum_drawdown_never_reports_a_trough_before_its_peak() -> None:
    """The failure mode of taking the highest and lowest points of the path.

    Here the lowest point comes first and the highest last, so a naive
    implementation reports a huge drawdown that never happened: the portfolio
    only ever went up.
    """
    rising = [0.5, 0.5, 0.5, 0.5]
    worst = maximum_drawdown(rising)
    assert worst.depth == pytest.approx(0.0)
    assert worst.peak_position <= worst.trough_position


def test_the_trough_is_always_at_or_after_the_peak() -> None:
    rng = random.Random(8)
    for _ in range(40):
        path = [rng.gauss(0.0, 0.03) for _ in range(60)]
        worst = maximum_drawdown(path)
        assert worst.peak_position <= worst.trough_position
        if worst.recovery_position is not None:
            assert worst.recovery_position > worst.trough_position


def test_the_depth_matches_the_worst_point_of_the_drawdown_series() -> None:
    rng = random.Random(12)
    for _ in range(30):
        path = [rng.gauss(0.0002, 0.02) for _ in range(200)]
        assert maximum_drawdown(path).depth == pytest.approx(max(drawdown_series(path)))


def test_recovery_return_is_larger_than_the_depth() -> None:
    worst = maximum_drawdown(PATH)
    assert worst.recovery_return == pytest.approx(1.0 / 3.0)
    assert worst.recovery_return > worst.depth


@pytest.mark.parametrize(
    ("depth", "needed"), [(0.1, 1.0 / 9.0), (0.2, 0.25), (0.5, 1.0), (0.8, 4.0)]
)
def test_the_recovery_asymmetry(depth: float, needed: float) -> None:
    """A fall of a half needs a gain of a whole, and the gap widens fast."""
    worst = maximum_drawdown([-depth])
    assert worst.depth == pytest.approx(depth)
    assert worst.recovery_return == pytest.approx(needed)


def test_a_total_loss_can_never_be_recovered() -> None:
    worst = maximum_drawdown([-1.0 + 1e-18])
    assert worst.recovery_return > 1e15


def test_an_unrecovered_drawdown_reports_no_recovery() -> None:
    """``None``, not zero and not the distance to the end of the sample."""
    worst = maximum_drawdown([0.5, -0.1, -0.1])
    assert worst.recovery_position is None
    assert worst.recovery_periods is None
    assert not worst.recovered
    # Time under water so far is still available, and is a lower bound.
    assert worst.underwater_periods == 2


def test_the_index_labels_the_valuations_not_the_moves() -> None:
    with pytest.raises(ValueError, match="one more point"):
        maximum_drawdown(PATH, index=["a", "b", "c", "d", "e"])


def test_positions_default_to_integers() -> None:
    worst = maximum_drawdown(PATH)
    assert worst.peak_label == 1
    assert worst.trough_label == 2


# -- separate episodes -------------------------------------------------------


def test_episodes_are_separated_by_a_return_to_the_old_peak() -> None:
    episodes = drawdowns(PATH, index=LABELS)
    assert len(episodes) == 2
    first, second = episodes
    assert first.depth == pytest.approx(0.25)
    assert first.recovered
    assert second.depth == pytest.approx(0.20)
    assert not second.recovered
    assert second.peak_position == 3


def test_episodes_come_back_deepest_first() -> None:
    rng = random.Random(21)
    path = [rng.gauss(0.0003, 0.02) for _ in range(400)]
    depths = [one.depth for one in drawdowns(path)]
    assert depths == sorted(depths, reverse=True)


def test_episodes_do_not_overlap() -> None:
    """The rule that makes these separate declines rather than nearby windows."""
    rng = random.Random(33)
    path = [rng.gauss(0.0003, 0.02) for _ in range(400)]
    episodes = sorted(drawdowns(path), key=lambda one: one.peak_position)
    for earlier, later in pairwise(episodes):
        assert earlier.recovery_position is not None
        assert later.peak_position >= earlier.recovery_position


@pytest.mark.parametrize("seed", range(15))
def test_the_deepest_episode_is_the_maximum_drawdown(seed: int) -> None:
    rng = random.Random(seed)
    path = [rng.gauss(0.0002, 0.02) for _ in range(150)]
    episodes = drawdowns(path)
    assert episodes
    assert episodes[0].depth == pytest.approx(maximum_drawdown(path).depth)


def test_a_minimum_depth_filters_the_small_ones() -> None:
    rng = random.Random(55)
    path = [rng.gauss(0.0003, 0.02) for _ in range(500)]
    every = drawdowns(path)
    big = drawdowns(path, minimum_depth=0.05)
    assert len(big) < len(every)
    assert all(one.depth >= 0.05 for one in big)


def test_a_minimum_depth_is_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        drawdowns(PATH, minimum_depth=-0.1)


def test_a_path_that_only_rises_has_no_episodes() -> None:
    assert drawdowns([0.01] * 10) == []


def test_longest_underwater_is_not_always_the_deepest() -> None:
    """A shallow decline that drags on beats a deep one that snaps back.

    Built so the two are different episodes: a 30% fall recovered in one period,
    and a 10% fall that takes many periods to come back.
    """
    path = [-0.30, 0.50]  # down 30%, straight back above the old peak
    path += [-0.10]  # then down 10%
    path += [0.012] * 9  # and a slow grind back
    deepest = maximum_drawdown(path)
    longest = longest_underwater(path)
    assert deepest.depth == pytest.approx(0.30)
    assert longest.depth == pytest.approx(0.10)
    assert longest.underwater_periods > deepest.underwater_periods


def test_longest_underwater_falls_back_when_there_are_no_episodes() -> None:
    assert longest_underwater([0.01] * 5).depth == pytest.approx(0.0)


# -- the ulcer index ---------------------------------------------------------


def test_the_ulcer_index_is_the_root_mean_square_drawdown() -> None:
    series = drawdown_series(PATH)
    expected = math.sqrt(math.fsum(value * value for value in series) / len(series))
    assert ulcer_index(PATH) == pytest.approx(expected)


def test_the_ulcer_index_averages_over_the_whole_path() -> None:
    """Including the points at a new high, which contribute zero.

    Two paths with the same maximum drawdown, one in it briefly and one in it
    almost always. Maximum drawdown cannot tell them apart; the ulcer index can,
    and that is the entire reason to have it.
    """
    brief = [-0.2, 0.25] + [0.0] * 18
    persistent = [-0.2] + [0.0] * 19
    assert maximum_drawdown(brief).depth == pytest.approx(
        maximum_drawdown(persistent).depth
    )
    assert ulcer_index(persistent) > 4.0 * ulcer_index(brief)


def test_the_ulcer_index_of_a_rising_path_is_zero() -> None:
    assert ulcer_index([0.01] * 10) == pytest.approx(0.0)


def test_the_ulcer_index_never_exceeds_the_maximum_drawdown() -> None:
    """A root mean square cannot exceed the maximum of what it averages."""
    rng = random.Random(66)
    for _ in range(30):
        path = [rng.gauss(0.0, 0.02) for _ in range(150)]
        assert ulcer_index(path) <= maximum_drawdown(path).depth + 1e-15


# -- the ratios --------------------------------------------------------------


def test_calmar_is_annualised_return_over_maximum_drawdown() -> None:
    series = ReturnSeries(name="p", values=tuple(PATH))
    expected = series.annualised_return(12) / maximum_drawdown(PATH).depth
    assert calmar(PATH, 12) == pytest.approx(expected)


def test_martin_is_annualised_return_over_the_ulcer_index() -> None:
    series = ReturnSeries(name="p", values=tuple(PATH))
    expected = series.annualised_return(12) / ulcer_index(PATH)
    assert martin_ratio(PATH, 12) == pytest.approx(expected)


def test_martin_exceeds_calmar_whenever_there_is_any_drawdown() -> None:
    """Because the ulcer index is the smaller denominator, always.

    Worth pinning: the two ratios are quoted interchangeably as "return over
    drawdown" and one of them is systematically the larger.
    """
    rng = random.Random(77)
    compared = 0
    for _ in range(20):
        path = [rng.gauss(0.002, 0.02) for _ in range(100)]
        series = ReturnSeries(name="p", values=tuple(path))
        # The ordering follows from the smaller denominator only when the shared
        # numerator is positive; with a negative annualised return the smaller
        # denominator makes the ratio more negative instead.
        if maximum_drawdown(path).depth <= 0.0 or series.annualised_return(12) <= 0.0:
            continue
        assert ulcer_index(path) < maximum_drawdown(path).depth
        assert martin_ratio(path, 12) > calmar(path, 12)
        compared += 1
    assert compared >= 10


def test_a_path_with_no_drawdown_has_no_drawdown_ratio() -> None:
    with pytest.raises(ValueError, match="never fell below a previous high"):
        calmar([0.01] * 10, 12)
    with pytest.raises(ValueError, match="never fell below a previous high"):
        martin_ratio([0.01] * 10, 12)


def test_the_periods_per_year_is_required_to_be_sensible() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        calmar(PATH, 0)
    with pytest.raises(ValueError, match="periods_per_year"):
        sortino(PATH, -12)


# -- the Sortino denominator -------------------------------------------------

SAMPLE = [0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.02, 0.01, 0.005]


def test_the_two_downside_deviations_differ_by_the_root_of_the_hit_rate() -> None:
    """``full / downside-only == sqrt(k / n)`` exactly, with ``k`` the downside count.

    Three of ten observations are below zero here, so the full-sample figure is
    ``sqrt(0.3)`` of the downside-only one — a factor of 1.83, in a statistic
    that is quoted under one name.
    """
    full = downside_deviation(SAMPLE)
    only = downside_deviation(SAMPLE, full_sample=False)
    downside = sum(1 for value in SAMPLE if value < 0.0)
    assert downside == 3
    assert full / only == pytest.approx(math.sqrt(downside / len(SAMPLE)))
    assert only / full == pytest.approx(1.826, abs=1e-3)


def test_sortino_inherits_the_denominator_choice() -> None:
    assert sortino(SAMPLE, 252) > sortino(SAMPLE, 252, full_sample=False)
    ratio = sortino(SAMPLE, 252) / sortino(SAMPLE, 252, full_sample=False)
    assert ratio == pytest.approx(math.sqrt(len(SAMPLE) / 3), rel=1e-12)


def test_the_documented_factor_of_three_at_one_downside_period_in_ten() -> None:
    """The claim in the docstring, measured rather than asserted."""
    path = [0.01] * 9 + [-0.02]
    full = downside_deviation(path)
    only = downside_deviation(path, full_sample=False)
    assert only / full == pytest.approx(math.sqrt(10.0), rel=1e-12)
    assert sortino(path, 12) / sortino(path, 12, full_sample=False) == pytest.approx(
        math.sqrt(10.0), rel=1e-12
    )


def test_downside_deviation_ignores_everything_above_the_target() -> None:
    below_only = [value for value in SAMPLE if value < 0.0]
    squared = math.fsum(value * value for value in below_only)
    assert downside_deviation(SAMPLE) == pytest.approx(
        math.sqrt(squared / len(SAMPLE))
    )


def test_a_target_moves_what_counts_as_downside() -> None:
    assert downside_deviation(SAMPLE, target=0.02) > downside_deviation(SAMPLE)


def test_no_downside_at_all_is_refused_rather_than_returned_as_infinity() -> None:
    with pytest.raises(ValueError, match="no observation fell below"):
        downside_deviation([0.01] * 5, full_sample=False)
    with pytest.raises(ValueError, match="no observation fell below"):
        sortino([0.01] * 5, 12)


def test_sortino_rises_when_the_downside_shrinks() -> None:
    mild = [0.02, -0.005, 0.03, -0.005, 0.01]
    harsh = [0.02, -0.05, 0.03, -0.05, 0.01]
    assert sortino(mild, 12) > sortino(harsh, 12)


# -- rolling windows ---------------------------------------------------------


def test_a_rolling_window_is_not_padded_to_the_original_length() -> None:
    """Because a rolling statistic does not exist before its window is full.

    Filling the gap is how a backtest comes to use information it did not have
    at the time.
    """
    values = rolling(PATH, 3, lambda block: math.fsum(block))
    assert len(values) == len(PATH) - 3 + 1
    assert values[0] == pytest.approx(math.fsum(PATH[:3]))
    assert values[-1] == pytest.approx(math.fsum(PATH[-3:]))


def test_a_full_width_window_gives_one_value() -> None:
    values = rolling(PATH, len(PATH), lambda block: math.fsum(block))
    assert len(values) == 1
    assert values[0] == pytest.approx(math.fsum(PATH))


def test_a_window_must_fit() -> None:
    with pytest.raises(TooShort, match="does not fit"):
        rolling(PATH, len(PATH) + 1, math.fsum)


def test_a_window_is_at_least_one_period() -> None:
    with pytest.raises(ValueError, match="at least one period"):
        rolling(PATH, 0, math.fsum)


def test_rolling_drawdown_restarts_the_peak_in_each_window() -> None:
    """Each window is its own path, not a slice of the whole drawdown series.

    Here the portfolio falls once at the start and then rises forever. Sliced
    from the full series, late windows would still report the distance below the
    old peak; computed as their own paths they correctly report no drawdown.
    """
    path = [-0.5] + [0.01] * 10
    windows = rolling_maximum_drawdown(path, 4)
    assert windows[0] == pytest.approx(0.5)
    assert windows[-1] == pytest.approx(0.0)
    assert max(drawdown_series(path)[-4:]) > 0.4


def test_rolling_ulcer_index_agrees_with_the_direct_computation() -> None:
    rng = random.Random(88)
    path = [rng.gauss(0.0, 0.02) for _ in range(60)]
    windows = rolling_ulcer_index(path, 20)
    assert windows[0] == pytest.approx(ulcer_index(path[:20]))
    assert windows[-1] == pytest.approx(ulcer_index(path[-20:]))


def test_rolling_volatility_agrees_with_the_series_method() -> None:
    rng = random.Random(99)
    path = [rng.gauss(0.0, 0.02) for _ in range(80)]
    windows = rolling_volatility(path, 30, 252)
    expected = ReturnSeries(name="w", values=tuple(path[:30])).annualised_volatility(252)
    assert windows[0] == pytest.approx(expected)
    assert len(windows) == 80 - 30 + 1


def test_rolling_volatility_checks_its_periods() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        rolling_volatility(PATH, 3, 0)

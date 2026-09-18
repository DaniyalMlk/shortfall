"""Return series, panels and annualisation.

The conventions are what these tests are really about. Each one is a decision
that is invisible in the output — a number annualised with the wrong periods
per year, or a log return treated as a simple one, looks exactly like a correct
number — so each is pinned against an arithmetic identity rather than against a
recorded value.
"""

from __future__ import annotations

import math

import pytest

from shortfall.series import (
    DAILY_TRADING,
    MONTHLY,
    WEEKLY,
    Convention,
    Misaligned,
    Panel,
    ReturnSeries,
    TooShort,
)


def series(*values: float, name: str = "a") -> ReturnSeries:
    return ReturnSeries(name=name, values=tuple(values))


# -- construction -----------------------------------------------------------


def test_n_prices_give_n_minus_one_returns() -> None:
    # An off-by-one here shifts every return one period against whatever it is
    # compared to, which reads as a plausible correlation rather than an error.
    got = ReturnSeries.from_prices("a", [100.0, 110.0, 99.0])
    assert len(got) == 2
    assert got.values == pytest.approx((0.1, -0.1))


def test_log_prices_give_log_returns() -> None:
    got = ReturnSeries.from_prices("a", [100.0, 110.0], convention=Convention.LOG)
    assert got.convention is Convention.LOG
    assert got.values[0] == pytest.approx(math.log(1.1), abs=1e-15)


def test_one_price_is_no_returns() -> None:
    with pytest.raises(TooShort, match="no returns"):
        ReturnSeries.from_prices("a", [100.0])


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_a_price_that_is_not_positive_is_refused(price: float) -> None:
    with pytest.raises(ValueError, match="prices are positive"):
        ReturnSeries.from_prices("a", [100.0, price])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_return_that_is_not_finite_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="not finite"):
        series(0.01, value)


@pytest.mark.parametrize("value", [-1.0, -1.5])
def test_a_simple_return_at_or_below_minus_one_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="total loss"):
        series(value)


def test_a_log_return_below_minus_one_is_fine() -> None:
    # -1.5 as a log return is a 78% fall, which is a bad day and not an error.
    assert ReturnSeries("a", (-1.5,), Convention.LOG).to_simple().values[0] < 0.0


# -- the two conventions ----------------------------------------------------


def test_converting_between_conventions_round_trips() -> None:
    original = series(0.02, -0.013, 0.0004, 0.31)
    assert original.to_log().to_simple().values == pytest.approx(original.values, abs=1e-15)


def test_log_returns_add_across_time() -> None:
    # The property that makes them worth having, and the one that fails for
    # simple returns.
    logs = series(0.05, -0.02, 0.011).to_log()
    assert sum(logs.values) == pytest.approx(math.log1p(logs.cumulative()), abs=1e-15)


def test_simple_returns_compound_across_time() -> None:
    simple = series(0.5, -0.5)
    assert simple.cumulative() == pytest.approx(-0.25, abs=1e-15)


def test_the_two_conventions_agree_on_the_cumulative_return() -> None:
    simple = series(0.07, -0.03, 0.02)
    assert simple.to_log().cumulative() == pytest.approx(simple.cumulative(), abs=1e-15)


def test_simple_returns_add_across_a_portfolio() -> None:
    # A weighted sum of log returns is not the log return of the weighted
    # portfolio, so the panel converts before combining.
    panel = Panel.from_columns({"a": [0.10, -0.04], "b": [0.02, 0.06]})
    combined = panel.portfolio([0.25, 0.75])
    assert combined.values == pytest.approx((0.25 * 0.10 + 0.75 * 0.02, 0.25 * -0.04 + 0.75 * 0.06))


def test_a_log_panel_is_converted_before_it_is_combined() -> None:
    logs = Panel.from_columns({"a": [0.10], "b": [0.02]}, convention=Convention.LOG)
    combined = logs.portfolio([0.5, 0.5])
    naive = 0.5 * 0.10 + 0.5 * 0.02
    correct = 0.5 * math.expm1(0.10) + 0.5 * math.expm1(0.02)
    assert combined.values[0] == pytest.approx(correct, abs=1e-15)
    assert combined.values[0] != pytest.approx(naive, abs=1e-9)


# -- moments and annualisation ----------------------------------------------


def test_the_unbiased_and_maximum_likelihood_variances_differ_by_the_expected_factor() -> None:
    one = series(0.01, -0.02, 0.03, 0.00, -0.01)
    count = len(one)
    assert one.variance(ddof=0) == pytest.approx(
        one.variance(ddof=1) * (count - 1) / count, abs=1e-18
    )


def test_a_variance_needs_more_observations_than_its_degrees_of_freedom() -> None:
    with pytest.raises(TooShort, match="too few"):
        series(0.01).variance(ddof=1)


def test_the_annualised_return_is_geometric_not_arithmetic() -> None:
    # Up 50% then down 50% has an arithmetic mean of zero and has lost a
    # quarter of the money. Annualising the arithmetic mean would report zero.
    one = series(0.5, -0.5)
    assert one.mean == pytest.approx(0.0, abs=1e-18)
    assert one.annualised_return(periods_per_year=2) == pytest.approx(-0.25, abs=1e-14)


def test_a_full_year_of_returns_annualises_to_its_own_total() -> None:
    values = [0.01] * 12
    one = ReturnSeries("a", tuple(values))
    assert one.annualised_return(MONTHLY) == pytest.approx(one.cumulative(), abs=1e-14)


def test_half_a_year_of_returns_is_scaled_up_by_compounding() -> None:
    one = ReturnSeries("a", tuple([0.01] * 6))
    assert one.annualised_return(MONTHLY) == pytest.approx(1.01**12 - 1.0, abs=1e-14)


def test_volatility_scales_by_the_square_root_of_the_periods() -> None:
    one = series(0.01, -0.02, 0.03, 0.00, -0.01)
    assert one.annualised_volatility(DAILY_TRADING) == pytest.approx(
        one.stdev() * math.sqrt(252), abs=1e-18
    )


def test_annualising_weekly_data_as_daily_overstates_it_sevenfold() -> None:
    # The mistake the explicit argument exists to prevent, pinned so the size of
    # it is on the record.
    one = series(0.01, -0.02, 0.03, 0.00, -0.01)
    wrong = one.annualised_volatility(DAILY_TRADING)
    right = one.annualised_volatility(WEEKLY)
    assert wrong / right == pytest.approx(math.sqrt(252 / 52), abs=1e-12)
    assert wrong / right == pytest.approx(2.2, abs=0.01)


@pytest.mark.parametrize("periods", [0, -1, float("nan")])
def test_periods_per_year_must_be_positive(periods: float) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        series(0.01, 0.02).annualised_volatility(periods)


# -- panels -----------------------------------------------------------------


def test_series_of_different_lengths_are_refused() -> None:
    with pytest.raises(Misaligned, match="different lengths"):
        Panel.from_columns({"a": [0.1, 0.2], "b": [0.3]})


def test_mixing_conventions_in_one_panel_is_refused() -> None:
    with pytest.raises(Misaligned, match="simple and log"):
        Panel((series(0.1, name="a"), ReturnSeries("b", (0.1,), Convention.LOG)))


def test_an_empty_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Panel(())


def test_duplicate_names_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Panel((series(0.1, name="a"), series(0.2, name="a")))


def test_alignment_keeps_only_the_shared_dates() -> None:
    panel = Panel.aligned(
        {
            "a": {"2026-01": 0.01, "2026-02": 0.02, "2026-03": 0.03},
            "b": {"2026-02": 0.04, "2026-03": 0.05, "2026-04": 0.06},
        }
    )
    assert panel.observations == 2
    assert panel["a"].values == pytest.approx((0.02, 0.03))
    assert panel["b"].values == pytest.approx((0.04, 0.05))


def test_alignment_sorts_the_shared_dates() -> None:
    # Dictionary order is insertion order, which is whatever the caller happened
    # to build. Two series aligned in different orders would otherwise pair up
    # returns from different periods.
    panel = Panel.aligned(
        {"a": {"c": 3.0, "a": 1.0, "b": 2.0}, "b": {"b": 20.0, "c": 30.0, "a": 10.0}}
    )
    assert panel["a"].values == pytest.approx((1.0, 2.0, 3.0))
    assert panel["b"].values == pytest.approx((10.0, 20.0, 30.0))


def test_series_with_no_shared_dates_are_refused() -> None:
    with pytest.raises(Misaligned, match="share no dates"):
        Panel.aligned({"a": {"x": 0.1}, "b": {"y": 0.2}})


def test_a_panel_reports_its_own_shape() -> None:
    panel = Panel.from_columns({"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6]})
    assert (panel.assets, panel.observations) == (2, 3)
    assert panel.names == ["a", "b"]
    assert panel.row(1) == pytest.approx([0.2, 0.5])
    assert panel.column(0).name == "a"


def test_demeaning_removes_each_series_own_mean() -> None:
    panel = Panel.from_columns({"a": [0.1, 0.3], "b": [-0.2, 0.2]})
    centred = panel.demeaned()
    assert centred[0] == pytest.approx([-0.1, 0.1])
    assert centred[1] == pytest.approx([-0.2, 0.2])
    # A few epsilons, not zero: subtracting a mean that is not exactly
    # representable leaves a residue of about eps times the mean, and demanding
    # better than the arithmetic can deliver would be a test of nothing.
    assert all(sum(row) == pytest.approx(0.0, abs=1e-16) for row in centred)


def test_a_panel_built_from_prices_differences_every_column() -> None:
    panel = Panel.from_prices({"a": [100.0, 110.0], "b": [50.0, 45.0]})
    assert panel.observations == 1
    assert panel["a"].values == pytest.approx((0.1,))
    assert panel["b"].values == pytest.approx((-0.1,))


def test_portfolio_weights_must_match_the_panel() -> None:
    panel = Panel.from_columns({"a": [0.1], "b": [0.2]})
    with pytest.raises(ValueError, match="must agree"):
        panel.portfolio([1.0])


def test_a_panel_refuses_an_estimator_it_is_too_short_for() -> None:
    panel = Panel.from_columns({"a": [0.1], "b": [0.2]})
    with pytest.raises(TooShort, match="needs at least 2"):
        panel.require(2, "an estimator")


def test_an_unknown_name_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        Panel.from_columns({"a": [0.1]})["b"]

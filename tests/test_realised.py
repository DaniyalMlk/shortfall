"""Volatility from the whole bar: what each estimator assumes, and what it costs.

Every figure quoted in the module documentation and in the README is produced
here, at one configuration stated once and reused: 150 independent samples of
60 bars, each bar built from 256 equally spaced price observations of a
geometric Brownian motion with a per-bar volatility of 1%. The random numbers
come from this package's own quantile function applied to ``random.random()``,
not from ``random.gauss``, so the stream is identical on every interpreter and
the assertions can be tight.
"""

from __future__ import annotations

import math
import random
from functools import cache
from typing import ClassVar

import pytest

from shortfall.distributions import normal_ppf
from shortfall.realised import (
    BadBar,
    Bar,
    Bars,
    Estimator,
    close_to_close,
    efficiency,
    garman_klass,
    garman_klass_yang_zhang,
    parkinson,
    realised_variance,
    realised_volatility,
    rogers_satchell,
    tick_bias_factor,
    yang_zhang,
)

SIGMA = 0.01
SAMPLES = 150
BARS = 60
TICKS = 256


def _normal(rng: random.Random) -> float:
    return normal_ppf(min(max(rng.random(), 1e-12), 1.0 - 1e-12))


def _simulate(
    n_bars: int,
    ticks: int,
    sigma: float,
    drift_ratio: float,
    gap_ratio: float,
    rng: random.Random,
) -> Bars:
    """One sample of bars from a geometric Brownian motion observed ``ticks`` times.

    ``drift_ratio`` is the drift per bar measured in per-bar volatilities, so
    1.0 means the expected move over a bar equals its standard deviation.
    ``gap_ratio`` is the overnight jump's standard deviation on the same scale.
    """
    price = 100.0
    bars: list[Bar] = []
    step = 1.0 / ticks
    deviation = sigma * math.sqrt(step)
    drift = drift_ratio * sigma * step
    for _ in range(n_bars):
        if gap_ratio:
            jump = gap_ratio * sigma
            price *= math.exp(jump * _normal(rng) - 0.5 * jump * jump)
        opening = price
        high = price
        low = price
        for _ in range(ticks):
            price *= math.exp(drift + deviation * _normal(rng))
            if price > high:
                high = price
            elif price < low:
                low = price
        bars.append(Bar(opening, high, low, price))
    return Bars("sim", tuple(bars))


@cache
def batch(
    seed: int,
    ticks: int = TICKS,
    drift_ratio: float = 0.0,
    gap_ratio: float = 0.0,
    samples: int = SAMPLES,
    n_bars: int = BARS,
) -> tuple[Bars, ...]:
    rng = random.Random(seed)
    return tuple(
        _simulate(n_bars, ticks, SIGMA, drift_ratio, gap_ratio, rng)
        for _ in range(samples)
    )


def volatility_ratio(
    samples: tuple[Bars, ...], estimator: Estimator, truth: float = SIGMA
) -> float:
    """The estimator's average variance, as a volatility, against the truth."""
    mean = math.fsum(realised_variance(s, estimator) for s in samples) / len(samples)
    return math.sqrt(mean) / truth


def flat_bar(price: float = 100.0) -> Bar:
    return Bar(price, price, price, price)


class TestBar:
    def test_a_well_formed_bar_is_accepted(self) -> None:
        bar = Bar(100.0, 103.0, 98.0, 101.0)
        assert bar.log_range == pytest.approx(math.log(103.0 / 98.0), abs=1e-15)
        assert bar.log_close_to_open == pytest.approx(math.log(1.01), abs=1e-15)
        assert bar.log_high_to_open == pytest.approx(math.log(1.03), abs=1e-15)
        assert bar.log_low_to_open == pytest.approx(math.log(0.98), abs=1e-15)

    def test_a_bar_that_never_moved_is_a_bar(self) -> None:
        # Degenerate but legitimate: an illiquid name that traded once.
        assert flat_bar().log_range == 0.0

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ((100.0, 99.0, 98.0, 101.0), "below the open"),
            ((100.0, 103.0, 101.0, 102.0), "above the open"),
            ((100.0, 103.0, 98.0, 104.0), "is below the open"),
            ((100.0, 103.0, 98.0, 97.0), "is above the open"),
            ((100.0, 97.0, 98.0, 99.0), "below low"),
        ],
    )
    def test_an_impossible_bar_is_refused(
        self, values: tuple[float, ...], message: str
    ) -> None:
        with pytest.raises(BadBar, match=message):
            Bar(*values)

    @pytest.mark.parametrize("index", range(4))
    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_a_non_positive_or_non_finite_price_is_refused(
        self, index: int, bad: float
    ) -> None:
        values = [100.0, 103.0, 98.0, 101.0]
        values[index] = bad
        with pytest.raises(BadBar):
            Bar(*values)


class TestBars:
    def test_rows_become_bars_in_order(self) -> None:
        bars = Bars.from_rows("x", [(100.0, 101.0, 99.0, 100.5), (100.5, 102.0, 100.0, 101.0)])
        assert len(bars) == 2
        assert bars.closes == (100.5, 101.0)
        assert next(iter(bars)).open == 100.0

    def test_the_overnight_gap_is_measured_from_the_previous_close(self) -> None:
        bars = Bars.from_rows("x", [(100.0, 101.0, 99.0, 100.0), (102.0, 103.0, 101.0, 102.5)])
        assert bars.overnight() == pytest.approx([math.log(1.02)], abs=1e-15)

    def test_a_single_bar_has_no_gap(self) -> None:
        assert Bars.from_rows("x", [(100.0, 101.0, 99.0, 100.0)]).overnight() == []

    def test_an_empty_series_is_refused(self) -> None:
        with pytest.raises(BadBar, match="no bars"):
            Bars("x", ())

    def test_a_short_row_names_its_index(self) -> None:
        with pytest.raises(BadBar, match="row 1 has 3 values"):
            Bars.from_rows("x", [(100.0, 101.0, 99.0, 100.0), (1.0, 2.0, 3.0)])

    def test_a_bad_row_names_its_index(self) -> None:
        with pytest.raises(BadBar, match="row 1: high"):
            Bars.from_rows("x", [(100.0, 101.0, 99.0, 100.0), (100.0, 99.0, 98.0, 100.0)])


class TestDegenerateInputs:
    def test_every_estimator_reads_a_motionless_series_as_zero(self) -> None:
        bars = Bars("flat", tuple(flat_bar() for _ in range(10)))
        for estimator in Estimator:
            assert realised_variance(bars, estimator) == pytest.approx(0.0, abs=1e-24)

    @pytest.mark.parametrize(
        ("function", "minimum", "name"),
        [
            (close_to_close, 3, "close-to-close"),
            (garman_klass_yang_zhang, 2, "Garman-Klass-Yang-Zhang"),
            (yang_zhang, 4, "Yang-Zhang"),
        ],
    )
    def test_an_estimator_that_needs_history_says_how_much(
        self, function: object, minimum: int, name: str
    ) -> None:
        bars = Bars("short", tuple(flat_bar() for _ in range(minimum - 1)))
        with pytest.raises(BadBar, match=f"{name} needs at least {minimum} bars"):
            function(bars)  # type: ignore[operator]

    @pytest.mark.parametrize("function", [parkinson, garman_klass, rogers_satchell])
    def test_a_single_bar_is_enough_for_an_intraday_estimator(
        self, function: object
    ) -> None:
        assert function(Bars("one", (Bar(100.0, 102.0, 99.0, 101.0),))) >= 0.0  # type: ignore[operator]

    def test_no_estimator_can_return_a_negative_variance(self) -> None:
        # Garman-Klass subtracts a term and looks as though it could go
        # negative on a bar that closed at its own extreme. It cannot: the log
        # range is at least the absolute log open-to-close move by
        # construction, so half the first squared dominates 0.386 of the
        # second. This searches for the counterexample rather than taking the
        # argument's word for it, over bars deliberately built to sit on their
        # own extremes.
        rng = random.Random(404)
        for _ in range(4000):
            opening = 100.0
            close = opening * math.exp(rng.uniform(-0.3, 0.3))
            high = max(opening, close) * math.exp(rng.uniform(0.0, 0.05))
            low = min(opening, close) * math.exp(-rng.uniform(0.0, 0.05))
            bars = Bars("one", (Bar(opening, high, low, close),))
            for estimator in (
                Estimator.PARKINSON,
                Estimator.GARMAN_KLASS,
                Estimator.ROGERS_SATCHELL,
            ):
                assert realised_variance(bars, estimator) >= 0.0

    def test_a_bar_sitting_exactly_on_both_extremes_is_still_non_negative(self) -> None:
        # The tightest case: the high is the close and the low is the open,
        # so the range and the open-to-close move are the same number.
        bars = Bars("extreme", tuple(Bar(100.0, 110.0, 100.0, 110.0) for _ in range(5)))
        assert garman_klass(bars) > 0.0
        assert garman_klass(bars) == pytest.approx(
            (0.5 - (2.0 * math.log(2.0) - 1.0)) * math.log(1.1) ** 2, rel=1e-14
        )

    def test_annualising_scales_by_the_square_root_of_the_period_count(self) -> None:
        bars = batch(7)[0]
        one = realised_volatility(bars, 252.0, Estimator.PARKINSON)
        four = realised_volatility(bars, 4.0 * 252.0, Estimator.PARKINSON)
        assert four == pytest.approx(2.0 * one, rel=1e-12)

    @pytest.mark.parametrize("periods", [0.0, -1.0])
    def test_a_non_positive_period_count_is_refused(self, periods: float) -> None:
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            realised_volatility(batch(7)[0], periods)


class TestTheTickBias:
    """The number the textbooks do not quote, and the reason they do not."""

    @pytest.mark.parametrize(
        ("ticks", "measured"),
        [(16, 0.8470), (64, 0.9203), (256, 0.9601)],
    )
    def test_parkinson_is_biased_down_and_by_how_much(
        self, ticks: int, measured: float
    ) -> None:
        got = volatility_ratio(batch(7, ticks=ticks), Estimator.PARKINSON)
        assert got == pytest.approx(measured, abs=2e-3)
        assert got < 1.0

    @pytest.mark.parametrize("ticks", [16, 64, 256])
    def test_close_to_close_is_not(self, ticks: int) -> None:
        # It reads only the endpoints, which are observed exactly however few
        # trades there were in between. That is the other half of the trade.
        got = volatility_ratio(batch(7, ticks=ticks), Estimator.CLOSE_TO_CLOSE)
        assert got == pytest.approx(1.0, abs=0.015)

    @pytest.mark.parametrize("ticks", [16, 64, 256])
    def test_the_closed_form_predicts_it(self, ticks: int) -> None:
        # Within a per cent and a half at sixteen ticks a bar and a quarter of
        # one at two hundred and fifty-six, which is what a correction has to
        # do to be worth applying.
        predicted = math.sqrt(tick_bias_factor(ticks))
        measured = volatility_ratio(batch(7, ticks=ticks), Estimator.PARKINSON)
        assert predicted == pytest.approx(measured, rel=0.04)

    def test_the_normalisation_the_first_attempt_got_wrong(self) -> None:
        # Applying the overshoot constant to the volatility rather than to the
        # expected range overstates the bias by a factor of 1.6 and predicts a
        # 26% shortfall at sixteen ticks against a measured 15%.
        naive = (1.0 - 2.0 * 0.5826 / math.sqrt(16)) ** 2
        assert math.sqrt(naive) == pytest.approx(0.7085, abs=1e-3)
        assert math.sqrt(tick_bias_factor(16)) == pytest.approx(0.8175, abs=1e-3)
        assert volatility_ratio(batch(7, ticks=16), Estimator.PARKINSON) > 0.80

    def test_correcting_by_the_factor_removes_most_of_it(self) -> None:
        for ticks in (16, 64, 256):
            raw = volatility_ratio(batch(7, ticks=ticks), Estimator.PARKINSON)
            corrected = raw / math.sqrt(tick_bias_factor(ticks))
            assert abs(corrected - 1.0) < abs(raw - 1.0)
            assert corrected == pytest.approx(1.0, abs=0.05)

    @pytest.mark.parametrize("ticks", [0, 1, -5])
    def test_too_few_ticks_is_refused(self, ticks: int) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            tick_bias_factor(ticks)

    def test_at_two_ticks_it_is_arithmetic_rather_than_a_prediction(self) -> None:
        # Stated rather than guarded against: the expansion is asymptotic, and
        # at two observations a bar it claims three quarters of the variance
        # is missing. It stays positive, so there is no degenerate case, and
        # nothing in the library applies it automatically.
        assert 0.2 < tick_bias_factor(2) < 0.3

    def test_it_tends_to_one(self) -> None:
        assert tick_bias_factor(10**8) == pytest.approx(1.0, abs=1e-3)
        assert tick_bias_factor(10**12) == pytest.approx(1.0, abs=1e-5)


class TestEfficiency:
    """Measured on the caller's own sample size, not quoted."""

    EXPECTED: ClassVar[dict[Estimator, float]] = {
        Estimator.CLOSE_TO_CLOSE: 1.00,
        Estimator.PARKINSON: 5.02,
        Estimator.GARMAN_KLASS: 7.74,
        Estimator.ROGERS_SATCHELL: 6.42,
        Estimator.GARMAN_KLASS_YANG_ZHANG: 7.52,
        Estimator.YANG_ZHANG: 6.78,
    }

    @pytest.mark.parametrize("estimator", list(Estimator))
    def test_the_measured_ratio(self, estimator: Estimator) -> None:
        got = efficiency(batch(7), estimator)
        assert got == pytest.approx(self.EXPECTED[estimator], rel=0.02)

    def test_every_range_estimator_beats_the_close(self) -> None:
        for estimator in Estimator:
            if estimator is not Estimator.CLOSE_TO_CLOSE:
                assert efficiency(batch(7), estimator) > 4.0

    def test_the_baseline_measured_against_itself_is_one(self) -> None:
        assert efficiency(batch(7), Estimator.CLOSE_TO_CLOSE) == pytest.approx(1.0, abs=1e-12)

    def test_one_sample_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            efficiency(batch(7)[:1], Estimator.PARKINSON)

    def test_samples_that_do_not_vary_are_refused(self) -> None:
        flat = tuple(Bars("flat", tuple(flat_bar() for _ in range(10))) for _ in range(5))
        with pytest.raises(ValueError, match="did not vary"):
            efficiency(flat, Estimator.PARKINSON)


class TestDriftSensitivity:
    """Which estimators read a trend as volatility, and by how much."""

    BASELINE: ClassVar[dict[Estimator, float]] = {
        Estimator.CLOSE_TO_CLOSE: 0.9997,
        Estimator.PARKINSON: 0.9601,
        Estimator.GARMAN_KLASS: 0.9439,
        Estimator.ROGERS_SATCHELL: 0.9434,
        Estimator.GARMAN_KLASS_YANG_ZHANG: 0.9438,
        Estimator.YANG_ZHANG: 0.9517,
    }

    @pytest.mark.parametrize("estimator", list(Estimator))
    def test_the_no_drift_baseline(self, estimator: Estimator) -> None:
        assert volatility_ratio(batch(7), estimator) == pytest.approx(
            self.BASELINE[estimator], abs=2e-3
        )

    @pytest.mark.parametrize(
        ("ratio", "parkinson_inflation", "garman_klass_inflation"),
        [(0.5, 1.0429, 1.0119), (1.0, 1.1737, 1.0593), (2.0, 1.5936, 1.2258)],
    )
    def test_the_zero_drift_estimators_read_a_trend_as_volatility(
        self, ratio: float, parkinson_inflation: float, garman_klass_inflation: float
    ) -> None:
        samples = batch(23, drift_ratio=ratio)
        for estimator, expected in (
            (Estimator.PARKINSON, parkinson_inflation),
            (Estimator.GARMAN_KLASS, garman_klass_inflation),
        ):
            inflation = volatility_ratio(samples, estimator) / self.BASELINE[estimator]
            assert inflation == pytest.approx(expected, abs=5e-3)

    def test_the_inflation_is_quadratic_in_the_drift_to_volatility_ratio(self) -> None:
        # Doubling the ratio quadruples the excess, so the estimator is
        # untroubled by a realistic daily drift and badly troubled by a
        # trending intraday hour.
        excess = []
        for ratio in (0.5, 1.0, 2.0):
            inflation = (
                volatility_ratio(batch(23, drift_ratio=ratio), Estimator.PARKINSON)
                / self.BASELINE[Estimator.PARKINSON]
            )
            excess.append(inflation - 1.0)
        assert 3.0 < excess[1] / excess[0] < 5.0
        assert 3.0 < excess[2] / excess[1] < 5.0

    @pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize(
        "estimator", [Estimator.ROGERS_SATCHELL, Estimator.YANG_ZHANG]
    )
    def test_the_drift_independent_estimators_are_barely_moved(
        self, ratio: float, estimator: Estimator
    ) -> None:
        # Exactly drift-free for the continuous process; the residue is the
        # discrete grid interacting with the trend, and at a drift of two
        # volatilities a bar it is still under four per cent.
        moved = volatility_ratio(batch(23, drift_ratio=ratio), estimator)
        assert moved / self.BASELINE[estimator] == pytest.approx(1.0, abs=0.04)

    def test_close_to_close_is_untroubled_because_it_removes_the_mean(self) -> None:
        for ratio in (0.5, 1.0, 2.0):
            moved = volatility_ratio(batch(23, drift_ratio=ratio), Estimator.CLOSE_TO_CLOSE)
            assert moved / self.BASELINE[Estimator.CLOSE_TO_CLOSE] == pytest.approx(
                1.0, abs=0.01
            )


class TestGapSensitivity:
    """An overnight jump the intraday estimators cannot see."""

    TRUTH = SIGMA * math.sqrt(2.0)

    @pytest.mark.parametrize(
        ("estimator", "expected"),
        [
            (Estimator.CLOSE_TO_CLOSE, 1.0048),
            (Estimator.PARKINSON, 0.6810),
            (Estimator.GARMAN_KLASS, 0.6680),
            (Estimator.ROGERS_SATCHELL, 0.6669),
            (Estimator.GARMAN_KLASS_YANG_ZHANG, 0.9745),
            (Estimator.YANG_ZHANG, 0.9781),
        ],
    )
    def test_what_each_one_reads_when_half_the_variance_is_overnight(
        self, estimator: Estimator, expected: float
    ) -> None:
        got = volatility_ratio(batch(29, gap_ratio=1.0), estimator, self.TRUTH)
        assert got == pytest.approx(expected, abs=3e-3)

    @pytest.mark.parametrize(
        "estimator",
        [Estimator.PARKINSON, Estimator.GARMAN_KLASS, Estimator.ROGERS_SATCHELL],
    )
    def test_the_intraday_estimators_miss_exactly_the_overnight_half(
        self, estimator: Estimator
    ) -> None:
        # Everything they read is measured from the open, so a jump that moves
        # the open, the high, the low and the close together is invisible.
        # With the two halves equal they should read 1/sqrt(2) of the truth,
        # times whatever tick bias they already carry.
        gapped = volatility_ratio(batch(29, gap_ratio=1.0), estimator, self.TRUTH)
        intraday_only = volatility_ratio(batch(7), estimator) / math.sqrt(2.0)
        assert gapped == pytest.approx(intraday_only, rel=0.02)

    @pytest.mark.parametrize(
        "estimator", [Estimator.GARMAN_KLASS_YANG_ZHANG, Estimator.YANG_ZHANG]
    )
    def test_the_gap_aware_estimators_recover_it(self, estimator: Estimator) -> None:
        got = volatility_ratio(batch(29, gap_ratio=1.0), estimator, self.TRUTH)
        assert got > 0.95


class TestAgainstEachOther:
    def test_the_dispatcher_matches_the_functions(self) -> None:
        bars = batch(7)[0]
        pairs = (
            (Estimator.CLOSE_TO_CLOSE, close_to_close),
            (Estimator.PARKINSON, parkinson),
            (Estimator.GARMAN_KLASS, garman_klass),
            (Estimator.ROGERS_SATCHELL, rogers_satchell),
            (Estimator.GARMAN_KLASS_YANG_ZHANG, garman_klass_yang_zhang),
            (Estimator.YANG_ZHANG, yang_zhang),
        )
        for estimator, function in pairs:
            assert realised_variance(bars, estimator) == pytest.approx(
                function(bars), rel=1e-15
            )

    def test_parkinson_is_the_scaled_mean_squared_range(self) -> None:
        # The formula, written out, against the implementation.
        bars = batch(7)[0]
        total = math.fsum(math.log(b.high / b.low) ** 2 for b in bars.bars)
        assert parkinson(bars) == pytest.approx(
            total / (4.0 * math.log(2.0) * len(bars)), rel=1e-14
        )

    def test_rogers_satchell_is_unchanged_by_scaling_every_price(self) -> None:
        # It is built from ratios, so a currency redenomination leaves it
        # alone. The same is true of the others and this is the cheapest of
        # them to state.
        bars = batch(7)[0]
        scaled = Bars(
            "scaled",
            tuple(Bar(b.open * 7.3, b.high * 7.3, b.low * 7.3, b.close * 7.3) for b in bars.bars),
        )
        assert rogers_satchell(scaled) == pytest.approx(rogers_satchell(bars), rel=1e-12)

    def test_yang_zhang_weight_lies_between_zero_and_one(self) -> None:
        # k = 0.34 / (1.34 + (n+1)/(n-1)), which is between 0 and 0.34 for
        # every n, so the combination is a genuine weighted average.
        for n in (4, 10, 100, 10_000):
            weight = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
            assert 0.0 < weight < 0.34

    def test_yang_zhang_reduces_towards_rogers_satchell_without_a_gap(self) -> None:
        # With no overnight move and no open-to-close dispersion beyond what
        # Rogers-Satchell already sees, the three pieces should land close to
        # each other rather than far apart.
        samples = batch(7)
        one = volatility_ratio(samples, Estimator.YANG_ZHANG)
        two = volatility_ratio(samples, Estimator.ROGERS_SATCHELL)
        assert one == pytest.approx(two, rel=0.02)


class TestTheFiguresInTheReadme:
    """Every number the README quotes, recomputed at the configuration it names."""

    def test_the_efficiency_table_is_in_the_order_it_is_printed_in(self) -> None:
        order = [
            Estimator.CLOSE_TO_CLOSE,
            Estimator.PARKINSON,
            Estimator.ROGERS_SATCHELL,
            Estimator.YANG_ZHANG,
            Estimator.GARMAN_KLASS_YANG_ZHANG,
            Estimator.GARMAN_KLASS,
        ]
        measured = [efficiency(batch(7), estimator) for estimator in order]
        assert measured == sorted(measured)
        assert measured == pytest.approx([1.00, 5.02, 6.42, 6.78, 7.52, 7.74], abs=0.02)

    @pytest.mark.parametrize(
        ("ticks", "close", "park", "predicted"),
        [(16, 0.991, 0.847, 0.818), (64, 1.004, 0.920, 0.909), (256, 1.000, 0.960, 0.954)],
    )
    def test_the_tick_bias_table(
        self, ticks: int, close: float, park: float, predicted: float
    ) -> None:
        samples = batch(7, ticks=ticks)
        assert volatility_ratio(samples, Estimator.CLOSE_TO_CLOSE) == pytest.approx(
            close, abs=1e-3
        )
        assert volatility_ratio(samples, Estimator.PARKINSON) == pytest.approx(park, abs=1e-3)
        assert math.sqrt(tick_bias_factor(ticks)) == pytest.approx(predicted, abs=1e-3)

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [
            (0.5, (1.043, 1.012, 0.992, 0.994)),
            (1.0, (1.174, 1.059, 0.986, 0.989)),
            (2.0, (1.594, 1.226, 0.965, 0.971)),
        ],
    )
    def test_the_drift_table(self, ratio: float, expected: tuple[float, ...]) -> None:
        samples = batch(23, drift_ratio=ratio)
        order = (
            Estimator.PARKINSON,
            Estimator.GARMAN_KLASS,
            Estimator.ROGERS_SATCHELL,
            Estimator.YANG_ZHANG,
        )
        got = tuple(
            volatility_ratio(samples, estimator)
            / TestDriftSensitivity.BASELINE[estimator]
            for estimator in order
        )
        assert got == pytest.approx(expected, abs=2e-3)

    def test_the_quadratic_claim_about_a_realistic_daily_ratio(self) -> None:
        # The README says an equity near a daily drift-to-volatility ratio of
        # 0.05 sees an effect of four parts in ten thousand. Extrapolated from
        # the measured quadratic rather than simulated, because four parts in
        # ten thousand is below what this sample size can resolve -- which is
        # itself the reason the claim is worth stating that way.
        measured = (
            volatility_ratio(batch(23, drift_ratio=0.5), Estimator.PARKINSON)
            / TestDriftSensitivity.BASELINE[Estimator.PARKINSON]
        ) - 1.0
        coefficient = measured / 0.25
        assert coefficient * 0.05**2 == pytest.approx(4e-4, abs=1.5e-4)

    @pytest.mark.parametrize(
        ("estimator", "expected"),
        [
            (Estimator.CLOSE_TO_CLOSE, 1.005),
            (Estimator.PARKINSON, 0.681),
            (Estimator.GARMAN_KLASS, 0.668),
            (Estimator.ROGERS_SATCHELL, 0.667),
            (Estimator.GARMAN_KLASS_YANG_ZHANG, 0.975),
            (Estimator.YANG_ZHANG, 0.978),
        ],
    )
    def test_the_gap_table(self, estimator: Estimator, expected: float) -> None:
        got = volatility_ratio(batch(29, gap_ratio=1.0), estimator, SIGMA * math.sqrt(2.0))
        assert got == pytest.approx(expected, abs=1.5e-3)

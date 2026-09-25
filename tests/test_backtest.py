"""Whether the risk-model tests detect what they are supposed to detect.

A test of a test has two halves and both are needed. *Size*: a correct model
must not be rejected more often than the significance level says. *Power*: a
wrong model must be rejected. A statistic that never rejects passes the first
half perfectly and is worthless, so neither half alone is evidence.

Both halves are established here by simulation, against models whose defects
are known because they were put there on purpose.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from shortfall.backtest import (
    ADVISORY_MINIMUM,
    BadForecast,
    Exceedances,
    Zone,
    conditional_coverage,
    exceedances,
    expected_shortfall_test,
    independence,
    simulate_expected_shortfall_null,
    traffic_light,
    unconditional_coverage,
    validate,
)
from shortfall.distributions import normal_cdf, normal_pdf, normal_ppf
from shortfall.parametric import Distribution, normal_risk

CONFIDENCE = 0.99
TAIL = 1.0 - CONFIDENCE


def normal_forecast(volatility: float) -> tuple[float, float]:
    """The value at risk and expected shortfall of a zero-mean normal."""
    quantile = -normal_ppf(TAIL)
    return volatility * quantile, volatility * normal_pdf(quantile) / TAIL


def flags(pattern: str) -> Exceedances:
    """``'..X.X'`` as an exceedance series, for the transition arithmetic."""
    return Exceedances(breaches=tuple(char == "X" for char in pattern), confidence=CONFIDENCE)


# --------------------------------------------------------------------------
# The sign convention, which everything else depends on being right.
# --------------------------------------------------------------------------


def test_a_breach_is_a_loss_worse_than_the_forecast() -> None:
    observed = [-0.05, -0.02, 0.03, -0.021, -0.0199]
    forecasts = [0.02] * 5
    result = exceedances(observed, forecasts, confidence=CONFIDENCE)
    assert result.breaches == (True, False, False, True, False)


def test_an_exact_tie_is_not_a_breach() -> None:
    """The quantile bounds the rejection region and is not inside it."""
    result = exceedances([-0.02], [0.02], confidence=CONFIDENCE)
    assert result.breaches == (False,)


def test_a_negative_forecast_is_the_sign_convention_being_crossed() -> None:
    with pytest.raises(BadForecast, match="positive losses"):
        exceedances([0.01], [-0.02], confidence=CONFIDENCE)


def test_mismatched_lengths_say_which_is_which() -> None:
    with pytest.raises(BadForecast, match="one forecast per observation"):
        exceedances([0.01, 0.02], [0.02], confidence=CONFIDENCE)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_inputs_are_refused(bad: float) -> None:
    with pytest.raises(BadForecast, match="finite"):
        exceedances([bad], [0.02], confidence=CONFIDENCE)
    with pytest.raises(BadForecast, match="finite|positive"):
        exceedances([0.01], [bad], confidence=CONFIDENCE)


def test_an_empty_series_is_refused_rather_than_scored() -> None:
    with pytest.raises(BadForecast, match="at least one observation"):
        exceedances([], [], confidence=CONFIDENCE)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
def test_a_confidence_level_outside_the_unit_interval_is_refused(confidence: float) -> None:
    with pytest.raises(BadForecast, match="confidence level"):
        exceedances([0.01], [0.02], confidence=confidence)


# --------------------------------------------------------------------------
# Transition counting, which the independence test is built on.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("....", (3, 0, 0, 0)),
        ("XXXX", (0, 0, 0, 3)),
        (".X.X", (0, 2, 1, 0)),
        ("XX..", (1, 0, 1, 1)),
        ("X", (0, 0, 0, 0)),
        ("..XX..", (2, 1, 1, 1)),
    ],
)
def test_transition_counts(pattern: str, expected: tuple[int, int, int, int]) -> None:
    assert flags(pattern).transitions == expected


def test_the_transitions_account_for_every_consecutive_pair() -> None:
    series = flags("..X..XX...X.")
    assert sum(series.transitions) == series.observations - 1


# --------------------------------------------------------------------------
# Kupiec: the statistic itself, then its size and power.
# --------------------------------------------------------------------------


def test_the_kupiec_statistic_is_zero_when_the_rate_is_exactly_right() -> None:
    """Two breaches in 200 observations at 99% is the null exactly."""
    result = unconditional_coverage(flags("X" * 2 + "." * 198))
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_the_kupiec_statistic_matches_its_formula() -> None:
    n, x, p = 250, 10, TAIL
    rate = x / n
    expected = -2.0 * (
        (x * math.log(p) + (n - x) * math.log(1 - p))
        - (x * math.log(rate) + (n - x) * math.log(1 - rate))
    )
    result = unconditional_coverage(flags("X" * x + "." * (n - x)))
    assert result.statistic == pytest.approx(expected, rel=1e-12)


def test_zero_breaches_gives_a_finite_statistic() -> None:
    """``0 log 0`` is the limit, not an error, and the evidence is real: a
    year with no breaches at 99% is itself mildly surprising."""
    result = unconditional_coverage(flags("." * 250))
    assert result.statistic == pytest.approx(-2.0 * 250 * math.log(1 - TAIL), rel=1e-12)
    assert math.isfinite(result.p_value)


def test_every_observation_breaching_gives_a_finite_statistic() -> None:
    result = unconditional_coverage(flags("X" * 50))
    assert math.isfinite(result.statistic)
    assert result.p_value < 1e-12


def simulate_breach_counts(
    rate: float, observations: int, replications: int, seed: int
) -> list[Exceedances]:
    """Independent breaches at a given rate, which is the Kupiec null when
    ``rate`` is the tail probability."""
    rng = random.Random(seed)
    return [
        Exceedances(
            breaches=tuple(rng.random() < rate for _ in range(observations)),
            confidence=CONFIDENCE,
        )
        for _ in range(replications)
    ]


def test_kupiec_does_not_reject_a_correct_model_too_often() -> None:
    """Size. The nominal level is 5%; the realised rejection rate should be
    near it. The chi-square limit is approximate on a discrete statistic, so
    the band is generous — but it is a band, not a one-sided pass."""
    trials = simulate_breach_counts(TAIL, 500, 400, seed=11)
    rejections = sum(unconditional_coverage(t).rejects_at(0.05) for t in trials)
    assert 0.01 <= rejections / len(trials) <= 0.12


def test_kupiec_rejects_a_model_that_breaches_three_times_too_often() -> None:
    """Power. A 99% forecast that is really a 97% forecast."""
    trials = simulate_breach_counts(3.0 * TAIL, 500, 200, seed=12)
    rejections = sum(unconditional_coverage(t).rejects_at(0.05) for t in trials)
    assert rejections / len(trials) > 0.85


def test_kupiec_says_which_direction_it_failed_in() -> None:
    assert "too many" in unconditional_coverage(flags("X" * 30 + "." * 220)).interpretation
    assert "too few" in unconditional_coverage(flags("." * 250)).interpretation


# --------------------------------------------------------------------------
# Christoffersen: independence.
# --------------------------------------------------------------------------


def test_independence_is_silent_when_there_are_no_breaches() -> None:
    """Nothing about dependence has been observed, so there is no evidence
    either way and the statistic is zero rather than undefined."""
    result = independence(flags("." * 250))
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)
    assert "untested" in result.interpretation


def test_independence_handles_a_single_observation() -> None:
    result = independence(flags("X"))
    assert result.statistic == 0.0
    assert result.advisory


def test_independence_rejects_a_run_of_consecutive_breaches() -> None:
    """Ten breaches in 250 days is the right count for a 96% model and a
    plausible one for a 99% model. All ten in a row is not."""
    clustered = flags("." * 120 + "X" * 10 + "." * 120)
    assert independence(clustered).rejects_at(0.01)
    spread = flags((("." * 24 + "X") * 10) + "." * 0)
    assert not independence(spread).rejects_at(0.05)


def test_independence_does_not_reject_breaches_that_are_merely_frequent() -> None:
    """The point of the test: it is blind to the count, so a model that
    breaches far too often but independently passes it — and is caught by
    Kupiec instead."""
    trials = simulate_breach_counts(5.0 * TAIL, 500, 200, seed=13)
    rejections = sum(independence(t).rejects_at(0.05) for t in trials)
    assert rejections / len(trials) < 0.12
    kupiec = sum(unconditional_coverage(t).rejects_at(0.05) for t in trials)
    assert kupiec / len(trials) > 0.95


def clustered_series(observations: int, rate: float, persistence: float, seed: int) -> Exceedances:
    """A Markov breach process with the right unconditional rate and the
    wrong dependence, which is exactly what independence should catch."""
    rng = random.Random(seed)
    # Solve the two-state chain for the transition out of calm that leaves the
    # stationary rate at `rate` while the transition out of a breach is
    # `persistence`.
    from_calm = rate * (1.0 - persistence) / (1.0 - rate)
    breaches = []
    state = False
    for _ in range(observations):
        threshold = persistence if state else from_calm
        state = rng.random() < threshold
        breaches.append(state)
    return Exceedances(breaches=tuple(breaches), confidence=CONFIDENCE)


def test_independence_rejects_a_clustered_process_with_the_right_average_rate() -> None:
    """Power, on the failure mode a count is nearly blind to.

    The unconditional breach rate is the correct 1%, so the *expected* count
    gives nothing away. Kupiec still rejects about one time in six, and that
    is not a bug in either test: clustering makes the breach count
    overdispersed relative to the binomial its null assumes, so some samples
    land far enough from 1% to be rejected on the count alone. What matters is
    the gap — independence catches this five times as often, and it is the
    test that names the actual defect.
    """
    trials = [clustered_series(1000, TAIL, 0.35, seed=100 + i) for i in range(120)]
    average_rate = sum(t.rate for t in trials) / len(trials)
    assert average_rate == pytest.approx(TAIL, abs=0.003)
    kupiec = sum(unconditional_coverage(t).rejects_at(0.05) for t in trials) / len(trials)
    spread = sum(independence(t).rejects_at(0.05) for t in trials) / len(trials)
    assert spread > 0.80, "clustering should be caught"
    assert kupiec < 0.30, "the count should be much weaker evidence than the pattern"
    assert spread > 3.0 * kupiec


def test_independence_does_not_reject_an_independent_process_too_often() -> None:
    trials = simulate_breach_counts(TAIL, 1000, 300, seed=14)
    rejections = sum(independence(t).rejects_at(0.05) for t in trials)
    assert rejections / len(trials) <= 0.12


# --------------------------------------------------------------------------
# Conditional coverage.
# --------------------------------------------------------------------------


def test_conditional_coverage_is_the_sum_of_its_parts() -> None:
    series = flags("." * 100 + "XX" + "." * 50 + "X" + "." * 97)
    joint = conditional_coverage(series)
    assert joint.statistic == pytest.approx(
        unconditional_coverage(series).statistic + independence(series).statistic
    )
    assert joint.degrees_of_freedom == 2


def test_conditional_coverage_catches_either_failure_alone() -> None:
    too_many = simulate_breach_counts(4.0 * TAIL, 500, 150, seed=15)
    assert sum(conditional_coverage(t).rejects_at(0.05) for t in too_many) / 150 > 0.85
    clustered = [clustered_series(1000, TAIL, 0.35, seed=200 + i) for i in range(120)]
    assert sum(conditional_coverage(t).rejects_at(0.05) for t in clustered) / 120 > 0.70


def test_a_significance_level_must_be_a_probability() -> None:
    test = unconditional_coverage(flags("." * 100))
    with pytest.raises(BadForecast, match="significance level"):
        test.rejects_at(0.0)
    with pytest.raises(BadForecast, match="significance level"):
        test.rejects_at(1.0)


# --------------------------------------------------------------------------
# The supervisory traffic light.
# --------------------------------------------------------------------------

#: Zone and capital add-on by breach count, for 250 observations at 99%.
#: Transcribed from the published supervisory table, not from this code.
PUBLISHED_TABLE: list[tuple[int, str, float]] = [
    (0, "green", 0.00),
    (1, "green", 0.00),
    (2, "green", 0.00),
    (3, "green", 0.00),
    (4, "green", 0.00),
    (5, "yellow", 0.40),
    (6, "yellow", 0.50),
    (7, "yellow", 0.65),
    (8, "yellow", 0.75),
    (9, "yellow", 0.85),
    (10, "red", 1.00),
]


@pytest.mark.parametrize(("count", "zone", "plus"), PUBLISHED_TABLE)
def test_the_traffic_light_reproduces_the_published_table(
    count: int, zone: str, plus: float
) -> None:
    """The zones are derived from the binomial and land on the published
    counts, which is the check that the rule and the table agree."""
    result = traffic_light(flags("X" * count + "." * (250 - count)))
    assert result.zone.value == zone
    assert result.plus_factor == pytest.approx(plus)


def test_the_zone_boundaries_are_where_the_rule_says() -> None:
    assert traffic_light(flags("X" * 4 + "." * 246)).cumulative_probability < 0.95
    assert traffic_light(flags("X" * 5 + "." * 245)).cumulative_probability >= 0.95
    assert traffic_light(flags("X" * 9 + "." * 241)).cumulative_probability < 0.9999
    assert traffic_light(flags("X" * 10 + "." * 240)).cumulative_probability >= 0.9999


def test_beyond_ten_breaches_stays_red_at_the_full_add_on() -> None:
    for count in (11, 20, 100, 250):
        result = traffic_light(flags("X" * count + "." * (250 - count)))
        assert result.zone is Zone.RED
        assert result.plus_factor == pytest.approx(1.0)


def test_the_add_on_is_withheld_outside_the_setup_it_is_published_for() -> None:
    """Extrapolating a tabulated number to a sample length it was never
    computed for would be an invention, so the zone is still returned and the
    multiplier is not."""
    shorter = traffic_light(flags("X" * 3 + "." * 122))
    assert shorter.plus_factor is None
    other_level = Exceedances(breaches=tuple([True] * 6 + [False] * 244), confidence=0.95)
    assert traffic_light(other_level).plus_factor is None


def test_the_zone_follows_the_sample_length_rather_than_the_count() -> None:
    """Three breaches is comfortably green over 250 days and yellow over 125,
    because the expected count halved. A table indexed by count alone cannot
    say this, which is why the zones are derived from the binomial."""
    assert traffic_light(flags("X" * 3 + "." * 247)).zone is Zone.GREEN
    assert traffic_light(flags("X" * 3 + "." * 122)).zone is Zone.YELLOW


def test_the_zone_is_monotone_in_the_breach_count() -> None:
    order = {Zone.GREEN: 0, Zone.YELLOW: 1, Zone.RED: 2}
    previous = -1
    for count in range(0, 40):
        rank = order[traffic_light(flags("X" * count + "." * (250 - count))).zone]
        assert rank >= previous
        previous = rank


# --------------------------------------------------------------------------
# Expected shortfall.
# --------------------------------------------------------------------------


def normal_returns(volatility: float, count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, volatility) for _ in range(count)]


def test_the_shortfall_statistics_are_near_zero_for_a_correct_model() -> None:
    var, es = normal_forecast(0.01)
    observed = normal_returns(0.01, 4000, seed=21)
    result = expected_shortfall_test(
        observed, [var] * 4000, [es] * 4000, confidence=CONFIDENCE
    )
    assert abs(result.conditional) < 0.15
    assert abs(result.unconditional) < 0.25


def theoretical_statistics(
    true_volatility: float, forecast_volatility: float, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """What the two statistics converge to, in closed form.

    For a zero-mean normal truth and a normal forecast, the breach
    probability, the conditional tail mean and hence both statistics are all
    available exactly. Asserting against these rather than against a round
    number chosen to make the test pass is the difference between checking the
    code and checking that the code has not changed.
    """
    tail = 1.0 - confidence
    threshold = forecast_volatility * -normal_ppf(tail)
    forecast_mean = forecast_volatility * normal_pdf(-normal_ppf(tail)) / tail
    standardised = threshold / true_volatility
    breach_probability = normal_cdf(-standardised)
    tail_mean = true_volatility * normal_pdf(standardised) / breach_probability
    first = 1.0 - tail_mean / forecast_mean
    second = 1.0 - breach_probability * tail_mean / (tail * forecast_mean)
    return first, second


def test_the_shortfall_statistics_go_negative_when_the_tail_is_understated() -> None:
    """Returns drawn twice as volatile as the forecast assumed.

    The closed form says test 1 lands at about -0.245, not at some larger
    number: test 1 divides a realised tail mean by a forecast tail mean, and
    doubling the volatility only moves that ratio to 1.24 because the
    conditional tail mean of a normal grows slowly once the threshold is
    already inside the distribution. Test 2 is the one that moves a long way,
    because the breach *count* is what blows out.
    """
    first, second = theoretical_statistics(0.02, 0.01)
    assert first == pytest.approx(-0.245, abs=0.005)
    var, es = normal_forecast(0.01)
    observed = normal_returns(0.02, 4000, seed=22)
    result = expected_shortfall_test(observed, [var] * 4000, [es] * 4000, confidence=CONFIDENCE)
    assert result.conditional == pytest.approx(first, abs=0.05)
    assert result.unconditional == pytest.approx(second, rel=0.10)
    assert result.unconditional < -10.0
    assert "understated" in result.direction


def test_the_shortfall_statistics_go_positive_when_the_tail_is_overstated() -> None:
    """A forecast 30% too wide, which is as far as this test can go.

    Doubling the forecast volatility instead produces *no breaches at all* in
    four thousand observations — the 99% quantile of a distribution twice as
    wide as the truth is a 4.65 sigma event — and a statistic computed from
    zero breaches is not evidence about a tail mean. The overstated case is
    therefore necessarily a mild one, which is itself the finding: test 1 has
    very little power against an over-cautious model, because an over-cautious
    model stops producing the observations the test reads.
    """
    first, second = theoretical_statistics(0.01, 0.013)
    var, es = normal_forecast(0.013)
    observed = normal_returns(0.01, 4000, seed=23)
    result = expected_shortfall_test(observed, [var] * 4000, [es] * 4000, confidence=CONFIDENCE)
    assert 0 < result.breaches < 40
    assert result.unconditional > 0.5
    assert result.unconditional == pytest.approx(second, abs=0.35)
    assert first > 0.0
    assert "overstated" in result.direction


def test_a_forecast_twice_too_wide_produces_no_breaches_to_test_with() -> None:
    """The limit of the previous test, recorded rather than left implicit."""
    var, es = normal_forecast(0.02)
    observed = normal_returns(0.01, 4000, seed=23)
    result = expected_shortfall_test(observed, [var] * 4000, [es] * 4000, confidence=CONFIDENCE)
    assert result.breaches == 0
    assert "never tested" in result.direction


def test_the_conditional_statistic_ignores_the_breach_count() -> None:
    """Which is the whole point of having two. Doubling the number of
    breaches while keeping each one the right size leaves test 1 alone and
    moves test 2."""
    var, es = normal_forecast(0.01)
    observed = normal_returns(0.01, 4000, seed=24)
    # A model that under-states the quantile but states the tail mean of
    # *its own breaches* correctly is caught by test 2 and not by test 1.
    baseline = expected_shortfall_test(
        observed, [var] * 4000, [es] * 4000, confidence=CONFIDENCE
    )
    frequent = expected_shortfall_test(
        observed, [var * 0.82] * 4000, [es] * 4000, confidence=CONFIDENCE
    )
    assert frequent.breaches > baseline.breaches * 1.5
    assert frequent.unconditional < baseline.unconditional - 0.5


def test_an_expected_shortfall_below_its_value_at_risk_is_refused() -> None:
    """The tail mean averages losses at least as large as the quantile, so it
    cannot be the smaller of the two — that is a swapped argument."""
    with pytest.raises(BadForecast, match="below the value at risk"):
        expected_shortfall_test([0.0], [0.03], [0.02], confidence=CONFIDENCE)


def test_a_shortfall_series_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(BadForecast, match="one expected shortfall per observation"):
        expected_shortfall_test([0.0, 0.0], [0.02, 0.02], [0.03], confidence=CONFIDENCE)


def test_no_breaches_leaves_the_conditional_statistic_undefined_but_reported() -> None:
    var, es = normal_forecast(0.01)
    result = expected_shortfall_test([0.0] * 50, [var] * 50, [es] * 50, confidence=CONFIDENCE)
    assert result.breaches == 0
    assert result.conditional == 0.0
    assert "never tested" in result.direction


def test_the_simulated_null_is_centred_on_zero() -> None:
    """Which is the claim the statistics rest on, checked rather than assumed."""
    var, es = normal_forecast(0.01)
    first, second = simulate_expected_shortfall_null(
        [var] * 250, [es] * 250, confidence=CONFIDENCE, replications=800, seed=31
    )
    assert sum(second) / len(second) == pytest.approx(0.0, abs=0.06)
    # Test 1 is only defined on replications that produced a breach; the
    # zero-breach ones are not evidence about the tail mean.
    nonzero = [value for value in first if value != 0.0]
    assert len(nonzero) > 600
    assert sum(nonzero) / len(nonzero) == pytest.approx(0.0, abs=0.08)


def test_the_simulated_p_value_detects_an_understated_tail() -> None:
    var, es = normal_forecast(0.01)
    observed = normal_returns(0.018, 250, seed=32)
    result = validate(
        observed,
        [var] * 250,
        confidence=CONFIDENCE,
        expected_shortfall=[es] * 250,
        replications=600,
        seed=33,
    )
    assert result.expected_shortfall is not None
    assert result.expected_shortfall.unconditional_p_value is not None
    assert result.expected_shortfall.unconditional_p_value < 0.01
    assert result.expected_shortfall.conditional_p_value is not None


def test_a_simulated_p_value_is_never_exactly_zero() -> None:
    """The observed statistic counts itself, so the smallest reportable value
    is 1/(replications + 1). A p-value of zero would claim more than a finite
    simulation can support."""
    var, es = normal_forecast(0.01)
    observed = normal_returns(0.05, 250, seed=34)
    result = validate(
        observed,
        [var] * 250,
        confidence=CONFIDENCE,
        expected_shortfall=[es] * 250,
        replications=200,
        seed=35,
    )
    assert result.expected_shortfall is not None
    assert result.expected_shortfall.unconditional_p_value == pytest.approx(1.0 / 201.0)


def test_the_student_t_null_is_standardised_to_unit_variance() -> None:
    """Otherwise the simulated returns are more volatile than the forecasts
    they are being compared against, and every model looks understated."""
    var, es = normal_forecast(0.01)
    _, second = simulate_expected_shortfall_null(
        [var] * 250,
        [es] * 250,
        confidence=CONFIDENCE,
        distribution=Distribution.STUDENT_T,
        degrees=8.0,
        replications=600,
        seed=36,
    )
    # A t with eight degrees of freedom has a fatter tail than the normal the
    # forecasts assume, so the statistic is negative — but by a moderate
    # amount, not by the factor an unstandardised draw would produce.
    average = sum(second) / len(second)
    assert -1.2 < average < -0.05


def test_a_student_t_null_needs_a_finite_variance() -> None:
    var, es = normal_forecast(0.01)
    with pytest.raises(BadForecast, match="finite\n?\\s*variance|finite variance"):
        simulate_expected_shortfall_null(
            [var] * 10,
            [es] * 10,
            confidence=CONFIDENCE,
            distribution=Distribution.STUDENT_T,
            degrees=2.0,
            replications=5,
        )


def test_a_simulation_needs_at_least_one_replication() -> None:
    var, es = normal_forecast(0.01)
    with pytest.raises(BadForecast, match="at least one replication"):
        simulate_expected_shortfall_null(
            [var], [es], confidence=CONFIDENCE, replications=0
        )


# --------------------------------------------------------------------------
# The whole thing together.
# --------------------------------------------------------------------------


def test_validate_runs_every_test_and_warns_about_a_short_sample() -> None:
    var, es = normal_forecast(0.01)
    observed = normal_returns(0.01, 60, seed=41)
    result = validate(observed, [var] * 60, confidence=CONFIDENCE, expected_shortfall=[es] * 60)
    assert result.exceedances.observations == 60
    assert result.unconditional.advisory
    assert any(str(ADVISORY_MINIMUM) in warning for warning in result.warnings)
    assert result.expected_shortfall is not None
    assert result.expected_shortfall.replications == 0


def test_validate_names_the_tests_that_reject() -> None:
    rng = random.Random(42)
    var, es = normal_forecast(0.01)
    observed = [rng.gauss(0.0, 0.03) for _ in range(500)]
    result = validate(
        observed, [var] * 500, confidence=CONFIDENCE, expected_shortfall=[es] * 500
    )
    rejected = result.rejected_at(0.01)
    assert "unconditional coverage" in rejected
    assert "conditional coverage" in rejected


def test_validate_passes_a_correct_model() -> None:
    var, es = normal_forecast(0.01)
    observed = normal_returns(0.01, 1000, seed=43)
    result = validate(
        observed, [var] * 1000, confidence=CONFIDENCE, expected_shortfall=[es] * 1000
    )
    assert result.rejected_at(0.05) == ()
    assert result.traffic_light.zone is Zone.GREEN
    assert result.warnings == ()


def test_validate_warns_when_nothing_breached() -> None:
    result = validate([0.0] * 300, [0.02] * 300, confidence=CONFIDENCE)
    assert any("no breaches" in warning for warning in result.warnings)
    assert result.expected_shortfall is None


def test_the_forecasts_may_vary_from_day_to_day() -> None:
    """A real risk model produces a different number every day, and nothing
    here assumes otherwise."""
    rng = random.Random(44)
    volatilities = [0.005 + 0.015 * rng.random() for _ in range(800)]
    observed = [rng.gauss(0.0, v) for v in volatilities]
    forecasts = [normal_forecast(v)[0] for v in volatilities]
    shortfalls = [normal_forecast(v)[1] for v in volatilities]
    result = validate(
        observed, forecasts, confidence=CONFIDENCE, expected_shortfall=shortfalls
    )
    assert result.rejected_at(0.05) == ()


def test_the_forecasts_a_risk_estimate_produces_are_scored_directly() -> None:
    """End to end against the library's own estimator rather than a formula
    written out again in the test."""
    estimate = normal_risk(mean=0.0, volatility=0.012, confidence=CONFIDENCE)
    observed = normal_returns(0.012, 750, seed=45)
    result = validate(
        observed,
        [estimate.value_at_risk] * 750,
        confidence=CONFIDENCE,
        expected_shortfall=[estimate.expected_shortfall] * 750,
    )
    assert result.rejected_at(0.05) == ()
    assert result.traffic_light.plus_factor is None  # 750 observations, not 250


def sliding_windows(values: Sequence[float], width: int) -> list[Sequence[float]]:
    return [values[index : index + width] for index in range(len(values) - width + 1)]


def test_a_model_that_ignores_volatility_clustering_is_caught() -> None:
    """The failure this whole module exists for.

    Returns come from a process whose volatility switches between calm and
    turbulent. A constant-volatility forecast set to the unconditional level
    gets roughly the right number of breaches and puts almost all of them in
    the turbulent stretches.
    """
    rng = random.Random(46)
    observed: list[float] = []
    calm, storm = 0.006, 0.030
    turbulent = False
    for _ in range(1500):
        turbulent = rng.random() < (0.90 if turbulent else 0.02)
        observed.append(rng.gauss(0.0, storm if turbulent else calm))
    unconditional_volatility = math.sqrt(sum(value * value for value in observed) / len(observed))
    var, _ = normal_forecast(unconditional_volatility)
    result = validate(observed, [var] * 1500, confidence=CONFIDENCE)
    assert result.independence.rejects_at(0.01), result.independence.interpretation
    assert "conditional coverage" in result.rejected_at(0.01)

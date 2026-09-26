"""Historical and simulated risk.

Two checks here carry most of the weight.

The **subadditivity counterexample** is a worked instance of the property that
distinguishes expected shortfall from value at risk: two independent defaultable
positions whose combination value at risk calls far riskier than the sum of its
parts, while expected shortfall does not. It is arithmetic, not a simulation, so
the numbers are exact and comparable with the theory. It also found a real bug in
the expected shortfall estimator, which is the best argument for having it.

The **constant-volatility identity** is the other: with a flat volatility,
filtered historical simulation must reduce to plain historical simulation to the
last bit, because the two rescalings cancel. Anything else means the filter is
applied and removed in different units, which nothing else would reveal.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.historical import (
    Coherence,
    QuantileMethod,
    bootstrap_interval,
    check_subadditivity,
    empirical_quantile,
    ewma_volatility,
    filtered_historical_risk,
    historical_risk,
    sample_expected_shortfall,
)
from shortfall.parametric import Distribution, normal_risk
from shortfall.series import Convention, ReturnSeries, TooShort
from shortfall.volatility import fit_garch

SMALL = [1.0, 2.0, 3.0, 4.0, 5.0]


def gaussian(count: int, *, seed: int, sigma: float = 0.01) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(count)]


# -- empirical quantiles ----------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        # Position h = (n-1)p = 0.4 for p = 0.1 on five observations.
        (QuantileMethod.LOWER, 1.0),
        (QuantileMethod.HIGHER, 2.0),
        (QuantileMethod.LINEAR, 1.4),
        # Weibull position is (n+1)p - 1 = -0.4, clamped to the first observation.
        (QuantileMethod.WEIBULL, 1.0),
    ],
)
def test_each_method_matches_its_definition(method: QuantileMethod, expected: float) -> None:
    assert empirical_quantile(SMALL, 0.1, method=method) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        # h = 1.6 for p = 0.4.
        (QuantileMethod.LOWER, 2.0),
        (QuantileMethod.HIGHER, 3.0),
        (QuantileMethod.LINEAR, 2.6),
        # (n+1)p - 1 = 1.4.
        (QuantileMethod.WEIBULL, 2.4),
    ],
)
def test_the_methods_genuinely_disagree(method: QuantileMethod, expected: float) -> None:
    # They must, or naming the method would be pointless. The four values here
    # span 2.0 to 3.0 on the same data at the same probability.
    assert empirical_quantile(SMALL, 0.4, method=method) == pytest.approx(expected, abs=1e-15)


def test_the_bracket_encloses_the_interpolation() -> None:
    # LOWER and HIGHER bracket LINEAR by construction, which is what makes the
    # pair a usable measure of how much the interpolation is doing.
    for probability in (0.01, 0.05, 0.2, 0.5, 0.77, 0.99):
        low = empirical_quantile(SMALL, probability, method=QuantileMethod.LOWER)
        mid = empirical_quantile(SMALL, probability, method=QuantileMethod.LINEAR)
        high = empirical_quantile(SMALL, probability, method=QuantileMethod.HIGHER)
        assert low <= mid <= high


@pytest.mark.parametrize("method", list(QuantileMethod))
def test_the_endpoints_are_the_extremes(method: QuantileMethod) -> None:
    assert empirical_quantile(SMALL, 0.0, method=method) == 1.0
    assert empirical_quantile(SMALL, 1.0, method=method) == 5.0


@pytest.mark.parametrize("method", list(QuantileMethod))
def test_the_median_of_an_odd_sample_is_the_middle_observation(
    method: QuantileMethod,
) -> None:
    assert empirical_quantile(SMALL, 0.5, method=method) == 3.0


def test_the_input_does_not_have_to_be_sorted() -> None:
    assert empirical_quantile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0


def test_one_observation_is_its_own_quantile() -> None:
    assert empirical_quantile([0.7], 0.01) == 0.7


def test_an_empty_sample_has_no_quantile() -> None:
    with pytest.raises(TooShort):
        empirical_quantile([], 0.5)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_a_probability_outside_the_unit_interval_is_refused(probability: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        empirical_quantile(SMALL, probability)


# -- the exact sample expected shortfall ------------------------------------


def test_the_partial_observation_is_weighted_rather_than_rounded() -> None:
    # At 99% over 250 returns the tail is 2.5 observations: two count fully and
    # the third counts half. Rounding either way is wrong by a third of the tail.
    values = gaussian(250, seed=2)
    ordered = sorted(values)
    expected = (ordered[0] + ordered[1] + 0.5 * ordered[2]) / 2.5
    mean, used = sample_expected_shortfall(values, 0.01)
    assert mean == pytest.approx(expected, rel=1e-15)
    assert used == 3


def test_a_whole_number_of_observations_needs_no_partial_term() -> None:
    values = gaussian(200, seed=3)
    ordered = sorted(values)
    mean, used = sample_expected_shortfall(values, 0.05)
    assert mean == pytest.approx(math.fsum(ordered[:10]) / 10, rel=1e-15)
    assert used == 10


def test_a_tail_thinner_than_one_observation_is_the_worst_observation() -> None:
    values = gaussian(50, seed=4)
    mean, used = sample_expected_shortfall(values, 0.01)
    assert mean == pytest.approx(min(values), rel=1e-15)
    assert used == 1


def test_the_whole_sample_averages_to_its_mean() -> None:
    values = gaussian(100, seed=5)
    mean, used = sample_expected_shortfall(values, 1.0)
    assert mean == pytest.approx(sum(values) / 100, rel=1e-13)
    assert used == 100


def test_the_tail_mean_is_not_the_mean_of_everything_below_the_quantile() -> None:
    # The obvious implementation, and wrong whenever the quantile ties with a
    # value many observations share. Here nine tenths of the sample sits at the
    # quantile, so "everything at or below it" is nearly the whole sample.
    values = [-1.0] * 10 + [0.0] * 90
    assert empirical_quantile(values, 0.1) == pytest.approx(-0.1, abs=1e-15)
    mean, _ = sample_expected_shortfall(values, 0.1)
    assert mean == pytest.approx(-1.0, abs=1e-15)
    naive = sum(v for v in values if v <= 0.0) / 100
    assert naive == pytest.approx(-0.1, abs=1e-15)


@pytest.mark.parametrize("probability", [0.0, 1.5])
def test_a_tail_probability_outside_its_range_is_refused(probability: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        sample_expected_shortfall(SMALL, probability)


# -- historical risk ---------------------------------------------------------


def test_the_sign_convention_matches_the_parametric_estimates() -> None:
    result = historical_risk(gaussian(1000, seed=6), confidence=0.99)
    assert result.value_at_risk > 0.0
    assert result.quantile < 0.0
    assert result.value_at_risk == pytest.approx(-result.quantile, abs=1e-18)


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("method", list(QuantileMethod))
def test_expected_shortfall_is_never_smaller_than_value_at_risk(
    confidence: float, method: QuantileMethod
) -> None:
    # The average of a tail cannot be less severe than its edge. With LOWER the
    # edge is an observation rather than an interpolation, which is the case
    # where an off-by-one would show up.
    result = historical_risk(gaussian(500, seed=7), confidence=confidence, method=method)
    assert result.expected_shortfall >= result.value_at_risk


def test_risk_grows_with_confidence() -> None:
    values = gaussian(2000, seed=8)
    losses = [
        historical_risk(values, confidence=c).value_at_risk
        for c in (0.90, 0.95, 0.99, 0.995)
    ]
    assert losses == sorted(losses)


def test_a_gaussian_sample_recovers_the_gaussian_answer() -> None:
    # Not a tautology: the historical estimate knows nothing about the normal
    # distribution, so agreeing with the closed form on normal data is evidence
    # both are right. Five thousand observations is enough for about 2%.
    values = gaussian(5000, seed=9, sigma=0.02)
    empirical = historical_risk(values, confidence=0.95)
    sigma = math.sqrt(sum(v * v for v in values) / len(values))
    theoretical = normal_risk(mean=0.0, volatility=sigma, confidence=0.95)
    assert empirical.value_at_risk == pytest.approx(theoretical.value_at_risk, rel=0.05)
    assert empirical.expected_shortfall == pytest.approx(
        theoretical.expected_shortfall, rel=0.05
    )


def test_the_effective_sample_says_how_thin_the_tail_was() -> None:
    # 250 daily returns at 99% is two and a half observations, not 250, and the
    # point estimate cannot say so.
    result = historical_risk(gaussian(250, seed=10), confidence=0.99)
    assert result.observations == 250
    assert result.effective_sample == pytest.approx(2.5)
    assert result.tail_observations == 3


@pytest.mark.parametrize(
    ("count", "confidence", "expected"),
    [
        (400, 0.99, 4),  # n * p = 4.000000000000000444 in binary
        (100, 0.99, 1),  # 1.0000000000000002
        (1000, 0.99, 10),  # 10.000000000000002
        (200, 0.995, 1),  # 1.0000000000000002
        (500, 0.98, 10),  # 10.000000000000002
        (250, 0.99, 3),  # 2.5 exactly: a genuine partial, and it counts
        (1260, 0.975, 32),  # 31.5 exactly
        (37, 0.9, 4),  # 3.7000000000000006: also genuine
    ],
)
def test_the_tail_count_is_not_inflated_by_the_binary_product(
    count: int, confidence: float, expected: int
) -> None:
    """``n * p`` is almost never an exact integer in binary, and the count must not
    care.

    The estimator averages the worst ``n p`` observations, with the last one
    weighted by whatever ``n p`` leaves over. ``400 * 0.01`` is
    ``4.000000000000000444``, so a naive ``remainder > 0`` test takes a partial
    observation weighted ``4e-16`` and reports five observations behind a figure
    that four produced.

    The estimate is untouched by a term that small; the *count* is what breaks, and
    it is reported as how much data stands behind the number. Overstating that is
    the one direction this library must not be wrong in, which is why the cases
    above are the ones where ``n p`` is a whole number in decimal, alongside three
    where it genuinely is not.
    """
    result = historical_risk(gaussian(count, seed=21), confidence=confidence)
    assert result.tail_observations == expected
    assert result.effective_sample == pytest.approx(count * (1.0 - confidence), rel=1e-12)
    assert result.tail_observations == math.ceil(result.effective_sample - 1e-9)


def test_dropping_the_negligible_partial_does_not_move_the_estimate() -> None:
    """The fix is to a count, not to a number, and this is the assertion that says so.

    Recomputes the expected shortfall the way the estimator did before the
    threshold was added — taking the partial observation on any positive remainder
    — and requires the two to agree to within a relative 1e-12. If they did not,
    the fix would be changing an estimate under the guise of correcting a
    diagnostic.
    """
    values = gaussian(400, seed=22)
    result = historical_risk(values, confidence=0.99)

    ordered = sorted(values)
    weight = len(values) * 0.01
    full = math.floor(weight)
    naive = (math.fsum(ordered[:full]) + (weight - full) * ordered[full]) / weight

    assert -result.expected_shortfall == pytest.approx(naive, rel=1e-12)
    # And it is the count that differs, so the test is not vacuous.
    assert result.tail_observations == full


def test_a_return_series_can_be_passed_directly() -> None:
    series = ReturnSeries("a", tuple(gaussian(500, seed=11)), Convention.SIMPLE)
    assert historical_risk(series).value_at_risk == pytest.approx(
        historical_risk(list(series.values)).value_at_risk, abs=1e-18
    )


def test_a_single_observation_is_too_short_to_estimate_from() -> None:
    with pytest.raises(TooShort, match="at least 2"):
        historical_risk([0.01])


@pytest.mark.parametrize("confidence", [0.0, 1.0, 99.0])
def test_a_confidence_outside_the_unit_interval_is_refused(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        historical_risk(gaussian(100, seed=12), confidence=confidence)


# -- bootstrap intervals -----------------------------------------------------


def test_the_interval_brackets_the_point_estimate() -> None:
    interval = bootstrap_interval(gaussian(500, seed=13), confidence=0.95, resamples=400)
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.width > 0.0


def test_the_same_data_and_seed_give_the_same_interval() -> None:
    # A risk number that moves when nothing moved is worse than a wrong one.
    values = gaussian(300, seed=14)
    first = bootstrap_interval(values, resamples=200, seed=7)
    second = bootstrap_interval(values, resamples=200, seed=7)
    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_a_different_seed_gives_a_different_interval() -> None:
    values = gaussian(300, seed=15)
    first = bootstrap_interval(values, resamples=200, seed=1)
    second = bootstrap_interval(values, resamples=200, seed=2)
    assert (first.lower, first.upper) != (second.lower, second.upper)


def test_less_data_gives_a_wider_interval() -> None:
    # The whole reason the interval is worth computing.
    long = bootstrap_interval(gaussian(4000, seed=16), confidence=0.99, resamples=300)
    short = bootstrap_interval(gaussian(250, seed=16), confidence=0.99, resamples=300)
    assert short.relative_width > long.relative_width


def test_a_thinner_tail_gives_a_wider_relative_interval() -> None:
    values = gaussian(500, seed=17)
    at_95 = bootstrap_interval(values, confidence=0.95, resamples=300)
    at_999 = bootstrap_interval(values, confidence=0.999, resamples=300)
    assert at_999.relative_width > at_95.relative_width


def test_the_interval_can_be_taken_around_the_expected_shortfall() -> None:
    values = gaussian(500, seed=18)
    shortfall = bootstrap_interval(values, resamples=300, of_expected_shortfall=True)
    at_risk = bootstrap_interval(values, resamples=300)
    assert shortfall.estimate > at_risk.estimate
    assert shortfall.lower <= shortfall.estimate <= shortfall.upper


def test_a_zero_estimate_has_an_infinite_relative_width() -> None:
    interval = bootstrap_interval([0.0] * 100, resamples=50)
    assert interval.estimate == 0.0
    assert math.isinf(interval.relative_width)


@pytest.mark.parametrize("resamples", [0, 1])
def test_too_few_resamples_is_refused(resamples: int) -> None:
    with pytest.raises(ValueError, match="at least 2 resamples"):
        bootstrap_interval(gaussian(100, seed=19), resamples=resamples)


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_an_interval_level_outside_the_unit_interval_is_refused(level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        bootstrap_interval(gaussian(100, seed=20), level=level, resamples=10)


# -- the volatility filter ---------------------------------------------------


def test_the_volatility_estimate_uses_only_the_past() -> None:
    # The off-by-one that matters. An estimate including today's return would
    # divide it by a volatility that knew about it, flattening the standardised
    # series and making the filtered tail far too thin.
    quiet = [0.001] * 50
    shock = [*quiet, 0.20, *quiet]
    volatilities = ewma_volatility(shock)
    # The estimate at the shock has not seen it; the one after has.
    assert volatilities[51] > volatilities[50]


def test_the_volatility_responds_to_a_change_in_regime() -> None:
    calm = [0.002 if i % 2 else -0.002 for i in range(200)]
    stormy = [0.05 if i % 2 else -0.05 for i in range(200)]
    volatilities = ewma_volatility(calm + stormy)
    assert volatilities[-1] > 10.0 * volatilities[100]


def test_a_constant_volatility_series_has_a_constant_estimate() -> None:
    # Alternating returns of equal magnitude: the squared return never changes,
    # so neither does the exponentially weighted variance.
    values = [0.013 if i % 2 == 0 else -0.013 for i in range(200)]
    volatilities = ewma_volatility(values)
    assert max(volatilities) == min(volatilities)
    assert volatilities[0] == pytest.approx(0.013, rel=1e-15)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.5])
def test_a_decay_outside_the_unit_interval_is_refused(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        ewma_volatility(gaussian(100, seed=21), decay=decay)


def test_a_volatility_needs_more_than_one_observation() -> None:
    with pytest.raises(TooShort, match="at least 2"):
        ewma_volatility([0.01])


# -- filtered historical simulation ------------------------------------------


def test_a_constant_volatility_reduces_the_filter_to_the_identity() -> None:
    # The identity the whole construction rests on: dividing by the volatility
    # at the time and multiplying by today's must cancel exactly when the two
    # are the same. Anything else means the filter is applied and removed in
    # different units, which nothing else here would reveal.
    values = [0.013 if i % 2 == 0 else -0.013 for i in range(200)]
    filtered = filtered_historical_risk(values, confidence=0.95)
    plain = historical_risk(values, confidence=0.95)
    assert filtered.risk.value_at_risk == plain.value_at_risk
    assert filtered.risk.expected_shortfall == plain.expected_shortfall
    assert filtered.scaling == pytest.approx(1.0, abs=1e-14)


def test_filtering_raises_the_estimate_when_today_is_stormier_than_the_window() -> None:
    calm = [0.002 if i % 2 else -0.002 for i in range(300)]
    stormy = [0.04 if i % 2 else -0.04 for i in range(60)]
    values = calm + stormy
    filtered = filtered_historical_risk(values, confidence=0.95)
    plain = historical_risk(values, confidence=0.95)
    assert filtered.current_volatility > filtered.average_volatility
    assert filtered.scaling > 1.0
    assert filtered.risk.value_at_risk > plain.value_at_risk


def test_filtering_lowers_the_estimate_when_today_is_calmer() -> None:
    stormy = [0.04 if i % 2 else -0.04 for i in range(60)]
    calm = [0.002 if i % 2 else -0.002 for i in range(300)]
    values = stormy + calm
    filtered = filtered_historical_risk(values, confidence=0.95)
    plain = historical_risk(values, confidence=0.95)
    assert filtered.scaling < 1.0
    assert filtered.risk.value_at_risk < plain.value_at_risk


def test_the_standardised_returns_are_the_shape_that_was_kept() -> None:
    values = gaussian(400, seed=22)
    filtered = filtered_historical_risk(values)
    assert len(filtered.standardised) == len(values)
    # Standardised by construction: roughly unit variance, whatever the original
    # scale was.
    variance = sum(v * v for v in filtered.standardised) / len(filtered.standardised)
    assert variance == pytest.approx(1.0, rel=0.4)


def test_a_series_with_no_recent_variation_cannot_be_rescaled() -> None:
    with pytest.raises(ValueError, match="current volatility"):
        filtered_historical_risk([0.0] * 100)


# -- coherence ----------------------------------------------------------------
#
# Two independent positions, each half the book, in a bond that pays 1% or
# defaults with probability 4%. Stated as exact frequencies over ten thousand
# periods rather than simulated, so the numbers are comparable with the theory
# to the digit.

PERIODS = 10_000
BOTH_DEFAULT = 16  # 0.04 * 0.04
ONE_DEFAULTS = 384  # 0.04 * 0.96
NEITHER = 9216  # 0.96 * 0.96

FIRST = [-0.5] * (BOTH_DEFAULT + ONE_DEFAULTS) + [0.005] * (ONE_DEFAULTS + NEITHER)
SECOND = (
    [-0.5] * BOTH_DEFAULT
    + [0.005] * ONE_DEFAULTS
    + [-0.5] * ONE_DEFAULTS
    + [0.005] * NEITHER
)


def test_the_counterexample_is_built_as_intended() -> None:
    # If the construction drifts, the two results below stop meaning anything,
    # so the frequencies are asserted rather than assumed.
    assert len(FIRST) == len(SECOND) == PERIODS
    assert sum(1 for value in FIRST if value < 0) == BOTH_DEFAULT + ONE_DEFAULTS
    assert sum(1 for value in SECOND if value < 0) == BOTH_DEFAULT + ONE_DEFAULTS
    both = sum(1 for a, b in zip(FIRST, SECOND, strict=True) if a < 0 and b < 0)
    assert both == BOTH_DEFAULT


def test_value_at_risk_penalises_diversification() -> None:
    # Each half, alone, has a 4% chance of default — inside the 5% tail, so its
    # 95% value at risk is a *gain*. Combine them and the chance that at least
    # one defaults is 7.84%, which reaches into the tail, so the combined value
    # at risk is a large loss. Diversifying made the measured risk worse.
    result = check_subadditivity([FIRST, SECOND], confidence=0.95)
    assert result.parts == pytest.approx((-0.005, -0.005), abs=1e-15)
    assert result.combined == pytest.approx(0.495, abs=1e-12)
    assert not result.subadditive
    assert result.penalty > 0.5


def test_expected_shortfall_does_not() -> None:
    # The same positions, the same confidence, and the property holds — which is
    # the concrete reason a risk limit written in value at risk can be gamed by
    # splitting a book, and expected shortfall cannot.
    result = check_subadditivity([FIRST, SECOND], confidence=0.95, expected_shortfall=True)
    # Each part's tail is 4% at -0.5 and 1% at +0.005, averaging to -0.399.
    assert result.parts == pytest.approx((0.399, 0.399), abs=1e-12)
    assert result.combined == pytest.approx(0.51116, abs=1e-5)
    assert result.subadditive
    assert result.penalty < 0.0


def test_expected_shortfall_is_subadditive_on_ordinary_data_too() -> None:
    first = gaussian(1000, seed=23)
    second = gaussian(1000, seed=24)
    for confidence in (0.90, 0.95, 0.99):
        result = check_subadditivity(
            [first, second], confidence=confidence, expected_shortfall=True
        )
        assert result.subadditive


def test_a_comparison_needs_at_least_two_positions() -> None:
    with pytest.raises(ValueError, match="at least two"):
        check_subadditivity([gaussian(100, seed=25)])


def test_the_positions_must_be_aligned() -> None:
    with pytest.raises(ValueError, match="aligned"):
        check_subadditivity([gaussian(100, seed=26), gaussian(50, seed=27)])


def test_the_measure_is_named_on_the_result() -> None:
    assert (
        check_subadditivity([FIRST, SECOND], confidence=0.95).measure == "value at risk"
    )
    assert (
        check_subadditivity(
            [FIRST, SECOND], confidence=0.95, expected_shortfall=True
        ).measure
        == "expected shortfall"
    )


def test_a_coherence_result_reports_the_sum_it_compared_against() -> None:
    result = Coherence(combined=3.0, parts=(1.0, 1.5), measure="value at risk")
    assert result.sum_of_parts == 2.5
    assert result.penalty == 0.5
    assert not result.subadditive


# -- putting a historical estimate beside a parametric one -------------------


def test_a_historical_estimate_can_be_expressed_as_a_risk() -> None:
    from shortfall.historical import as_risk

    estimate = historical_risk(gaussian(1000, seed=28), confidence=0.99)
    expressed = as_risk(estimate)
    assert expressed.distribution is Distribution.HISTORICAL
    assert expressed.value_at_risk == estimate.value_at_risk
    assert expressed.expected_shortfall == estimate.expected_shortfall
    # No mean and no volatility, because a historical estimate does not have
    # them: it did not assume a shape, which is the entire point.
    assert expressed.mean == 0.0
    assert expressed.volatility == 0.0


# -- a filter the caller supplies --------------------------------------------


def garch_series(count: int = 1200, *, seed: int = 31) -> list[float]:
    rng = random.Random(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    variance = omega / (1.0 - alpha - beta)
    out: list[float] = []
    for _ in range(count + 400):
        value = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        out.append(value)
        variance = omega + alpha * value * value + beta * variance
    return out[400:]


def regime_series(count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    out: list[float] = []
    turbulent = False
    for _ in range(count):
        turbulent = rng.random() < (0.90 if turbulent else 0.02)
        out.append(rng.gauss(0.0, 0.030 if turbulent else 0.006))
    return out


def test_a_supplied_filter_replaces_the_exponential_weighting() -> None:
    values = garch_series(600)
    fitted = fit_garch(values)
    supplied = filtered_historical_risk(
        values, confidence=0.99, volatilities=list(fitted.volatilities)
    )
    default = filtered_historical_risk(values, confidence=0.99)
    assert supplied.current_volatility == pytest.approx(fitted.volatilities[-1])
    assert supplied.risk.value_at_risk != pytest.approx(default.risk.value_at_risk, rel=1e-6)
    assert len(supplied.standardised) == len(values)


def test_a_constant_supplied_filter_still_reduces_to_the_identity() -> None:
    """The invariant the whole construction rests on, under the new argument.

    Dividing by a constant and multiplying by the same constant has to cancel
    exactly, whatever the constant is and wherever it came from. This is the check
    that would catch the supplied filter being applied in different units from the
    level it is rescaled to.
    """
    values = [0.013 if index % 2 == 0 else -0.013 for index in range(200)]
    for level in (0.001, 1.0, 55.0):
        filtered = filtered_historical_risk(
            values, confidence=0.95, volatilities=[level] * len(values)
        )
        plain = historical_risk(values, confidence=0.95)
        assert filtered.risk.value_at_risk == pytest.approx(plain.value_at_risk, rel=1e-12)
        assert filtered.scaling == pytest.approx(1.0, abs=1e-14)


def test_the_current_level_can_be_a_forecast_rather_than_the_filter_s_last_value() -> None:
    """Which is the whole reason to pass a model's filter in.

    An exponentially weighted estimate has no forecast, so its last value is all
    there is. A fitted model does, and the default rescales to the volatility of
    the day that has just finished rather than of the day the position is exposed
    to.
    """
    values = garch_series(600)
    fitted = fit_garch(values)
    ahead = math.sqrt(fitted.next_variance(values[-1]))
    to_forecast = filtered_historical_risk(
        values, confidence=0.99, volatilities=list(fitted.volatilities), current=ahead
    )
    to_last = filtered_historical_risk(
        values, confidence=0.99, volatilities=list(fitted.volatilities)
    )
    assert to_forecast.current_volatility == pytest.approx(ahead)
    ratio = to_forecast.risk.value_at_risk / to_last.risk.value_at_risk
    assert ratio == pytest.approx(ahead / fitted.volatilities[-1], rel=1e-9)


@pytest.mark.parametrize("length", [10, 601])
def test_a_filter_of_the_wrong_length_is_refused(length: int) -> None:
    values = garch_series(600)
    with pytest.raises(ValueError, match="filter is one value per return"):
        filtered_historical_risk(values, volatilities=[0.01] * length)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_a_filter_value_that_is_not_a_volatility_is_refused(bad: float) -> None:
    values = garch_series(200)
    filter_values = [0.01] * len(values)
    filter_values[5] = bad
    with pytest.raises(ValueError, match="finite and"):
        filtered_historical_risk(values, volatilities=filter_values)


def test_filtering_by_either_beats_not_filtering_at_all() -> None:
    """The comparison measured, and it does not say what it was expected to.

    Six regime-switching series, a 500-observation window, and 4,200 one-step 99%
    forecasts scored walk-forward. Plain historical simulation breached 1.29% of
    the time, filtering by an exponential weighting 1.05%, and filtering by a
    fitted GARCH 0.76%, against a nominal 1%.

    So filtering beats not filtering, and between the two filters the *exponential
    weighting lands closest to nominal* while the model is conservative. That is
    not the result this argument is usually made with, and the reasons to prefer
    the model filter are the ones it has anyway: a decay estimated rather than
    assumed, a long-run level to revert to, and a forecast at horizons past one
    step, which an exponential weighting cannot give at all.

    The independence test rejected none of the eighteen runs, which is not evidence
    that nothing clustered: 700 observations at 99% is seven breaches, and the test
    has nothing to work with at that count.
    """
    window = 500
    counts = {"plain": 0, "ewma": 0, "garch": 0}
    total = 0
    for seed in range(40, 46):
        values = regime_series(1200, seed)
        fitted = fit_garch(values)
        model = list(fitted.volatilities)
        weighted = ewma_volatility(values, decay=0.94)
        for index in range(window, len(values)):
            realised = values[index]
            plain = historical_risk(values[:index], confidence=0.99).value_at_risk
            counts["plain"] += realised < -plain
            for name, filter_values in (("ewma", weighted), ("garch", model)):
                estimate = filtered_historical_risk(
                    values[:index],
                    confidence=0.99,
                    volatilities=filter_values[:index],
                    current=filter_values[index],
                ).risk.value_at_risk
                counts[name] += realised < -estimate
            total += 1
    assert total == 4200
    rates = {name: count / total for name, count in counts.items()}
    assert rates["plain"] > rates["ewma"] > rates["garch"]
    assert 0.011 < rates["plain"] < 0.015
    assert 0.009 < rates["ewma"] < 0.012
    assert 0.006 < rates["garch"] < 0.009

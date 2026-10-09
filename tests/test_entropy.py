"""Entropy pooling.

The checks divide into three kinds and they are not equally strong.

**The closed form is the oracle, and there is only one.** Reweighting to hit a
probability on a set has an answer in arithmetic — scale inside, scale outside —
because relative entropy on a partition is minimised cell by cell. So the
solver can be checked against something that is not another solver. A view on a
mean has no closed form, which is why every accuracy assertion here is on a
probability view and the mean views are checked for properties instead.

**The identities are transcription checks.** The optimal dual value is minus the
relative entropy, and the two are computed by different routes from different
quantities. That catches a sign dropped in the objective, which on a convex
problem would otherwise converge smoothly to the wrong point with a small
gradient. It cannot catch the objective being the wrong objective.

**The refusals are the part most likely to be wrong in use.** A target outside
the range the scenarios span, a view that is constant, two views that are
linearly dependent, and two views that are each reachable and jointly
impossible are four different failures with four different causes, and a solver
that reports them all as "did not converge" is no better than one that returns a
number. Each is asserted separately, on the message as well as the type.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.entropy import (
    Posterior,
    View,
    ViewsInfeasible,
    ViewsNotIdentified,
    effective_scenarios,
    mean_view,
    pool,
    probability_view,
    relative_entropy,
    stressed_risk,
    temper,
    weighted_expected_shortfall,
    weighted_quantile,
)
from shortfall.series import Misaligned, TooShort

SCENARIOS = 1000


def sample() -> tuple[list[float], list[float]]:
    """Two correlated series, fixed so that every measured figure is reproducible.

    The second series is 0.6 of the first plus independent noise, so it has a
    population slope on the first of 0.6 and a sample slope near it. Both are
    needed: the interesting checks are about what a view on one does to the
    other.
    """
    rng = random.Random(0)
    first = [rng.gauss(0.0, 1.0) for _ in range(SCENARIOS)]
    second = [0.6 * value + 0.8 * rng.gauss(0.0, 1.0) for value in first]
    return first, second


def uniform(count: int = SCENARIOS) -> tuple[float, ...]:
    return (1.0 / count,) * count


def closed_form_probability(
    flags: list[bool], target: float
) -> tuple[tuple[float, ...], float]:
    """The exact posterior and relative entropy for a probability view.

    Scale the uniform weights inside the set by ``t / p`` and outside by
    ``(1 - t) / (1 - p)``. Nothing else can change: within a cell of a
    partition the prior is already proportional to the posterior, and relative
    entropy is minimised independently on each cell.
    """
    count = len(flags)
    prior = sum(flags) / count
    inside = target / prior / count
    outside = (1.0 - target) / (1.0 - prior) / count
    weights = tuple(inside if flag else outside for flag in flags)
    divergence = target * math.log(target / prior) + (1.0 - target) * math.log(
        (1.0 - target) / (1.0 - prior)
    )
    return weights, divergence


# -- the oracle ----------------------------------------------------------------


def test_probability_views_match_their_closed_form() -> None:
    """Twenty combinations of threshold and target, against arithmetic.

    The thresholds reach from the body of the distribution to a set of two
    scenarios out of a thousand, because the solve's conditioning depends on
    how much prior mass the view is moving and a grid that stays in the body
    would not see it.
    """
    first, _ = sample()
    worst_weight = 0.0
    worst_entropy = 0.0
    checked = 0
    for threshold in (-1.0, -1.5, -2.0, -2.5):
        flags = [value < threshold for value in first]
        prior = sum(flags) / SCENARIOS
        if prior == 0.0:
            continue
        for multiple in (0.5, 1.5, 2.0, 3.0, 5.0):
            target = multiple * prior
            if not 0.0 < target < 1.0:
                continue
            checked += 1
            posterior = pool([probability_view(flags, target)])
            weights, divergence = closed_form_probability(flags, target)
            worst_weight = max(
                worst_weight,
                max(
                    abs(got - want)
                    for got, want in zip(posterior.weights, weights, strict=True)
                ),
            )
            worst_entropy = max(
                worst_entropy,
                abs(posterior.relative_entropy - divergence) / divergence,
            )
            assert posterior.iterations <= 10
    assert checked == 20
    assert worst_weight < 1e-15, worst_weight
    assert worst_entropy < 1e-11, worst_entropy


def test_the_dual_value_is_minus_the_relative_entropy() -> None:
    """Two routes to the same number, from the multipliers and from the weights."""
    first, _ = sample()
    flags = [value < -1.5 for value in first]
    prior = sum(flags) / SCENARIOS
    worst = 0.0
    for target in (0.5 * prior, 2.0 * prior, 4.0 * prior):
        posterior = pool([probability_view(flags, target)])
        worst = max(
            worst,
            abs(-posterior.dual_value - posterior.relative_entropy)
            / posterior.relative_entropy,
        )
    for shift in (-0.25, 0.25, -1.0):
        posterior = pool([mean_view(first, math.fsum(first) / SCENARIOS + shift)])
        worst = max(
            worst,
            abs(-posterior.dual_value - posterior.relative_entropy)
            / posterior.relative_entropy,
        )
    assert worst < 1e-11, worst


def test_the_views_are_met_to_round_off() -> None:
    first, second = sample()
    flags = [value < -1.0 for value in first]
    posterior = pool(
        [
            mean_view(first, math.fsum(first) / SCENARIOS - 0.3, "first mean"),
            probability_view(flags, 2.0 * sum(flags) / SCENARIOS, "first tail"),
            mean_view(second, math.fsum(second) / SCENARIOS - 0.1, "second mean"),
        ]
    )
    assert len(posterior.residuals) == 3
    assert all(abs(residual) < 1e-14 for residual in posterior.residuals)
    assert math.fsum(posterior.weights) == pytest.approx(1.0, abs=1e-14)


# -- the already-true view -----------------------------------------------------


def test_a_satisfied_view_returns_the_prior_exactly() -> None:
    """Not nearly the prior: the same floats.

    A caller who stresses and then compares is going to print the difference as
    a percentage change, and 1e-17 against zero is an answer that reads as a
    real move at four decimal places of a small number.
    """
    first, _ = sample()
    flags = [value < -2.0 for value in first]
    prior = sum(flags) / SCENARIOS
    posterior = pool([probability_view(flags, prior)])
    assert posterior.weights == uniform()
    assert posterior.multipliers == (0.0,)
    assert posterior.relative_entropy == 0.0
    assert posterior.effective_scenarios == pytest.approx(
        float(SCENARIOS), rel=1e-12
    )
    assert posterior.iterations == 0


def test_no_views_returns_the_prior_it_was_given() -> None:
    weights = (0.5, 0.25, 0.25)
    posterior = pool([], weights)
    assert posterior.weights == weights
    assert posterior.relative_entropy == 0.0
    assert posterior.multipliers == ()
    with pytest.raises(TooShort):
        pool([])


def test_a_scenario_the_prior_excludes_stays_excluded() -> None:
    """Zero prior weight is a statement about the scenario, not a starting point."""
    values = [0.0, 1.0, 2.0, 3.0]
    prior = [0.5, 0.0, 0.25, 0.25]
    posterior = pool([mean_view(values, 1.9)], prior)
    assert posterior.weights[1] == 0.0
    assert math.fsum(posterior.weights) == pytest.approx(1.0, abs=1e-15)
    assert math.fsum(
        weight * value
        for weight, value in zip(posterior.weights, values, strict=True)
    ) == pytest.approx(1.9, abs=1e-14)


# -- the four refusals ---------------------------------------------------------


def test_a_target_outside_the_span_is_refused_by_name() -> None:
    values = [-1.0, 0.0, 1.0]
    with pytest.raises(ViewsInfeasible, match="outside the open range"):
        pool([mean_view(values, 1.5, "too far")])
    with pytest.raises(ViewsInfeasible, match="too far"):
        pool([mean_view(values, 1.5, "too far")])
    # The boundary itself is unreachable at finite relative entropy, not merely
    # expensive, so it is refused rather than approximated.
    with pytest.raises(ViewsInfeasible):
        pool([mean_view(values, 1.0)])


def test_a_constant_view_says_nothing_and_is_refused() -> None:
    with pytest.raises(ViewsNotIdentified, match="says nothing"):
        pool([mean_view([2.0, 2.0, 2.0], 2.0, "flat")])
    with pytest.raises(ViewsInfeasible, match=r"constant at 2\.0"):
        pool([mean_view([2.0, 2.0, 2.0], 3.0, "flat")])


def test_linearly_dependent_views_are_refused() -> None:
    """Two views that are the same statement twice, scaled.

    The Hessian is the posterior covariance of the view functions, so a
    dependency makes it singular. That is a property of the views and the
    scenarios together, and the message says so rather than reporting a failed
    step.
    """
    values = [-1.0, 0.0, 1.0, 2.0]
    doubled = [2.0 * value for value in values]
    with pytest.raises(ViewsNotIdentified, match="linearly dependent"):
        pool([mean_view(values, 0.2, "once"), mean_view(doubled, 0.4, "twice")])


def test_views_each_reachable_and_jointly_impossible_are_refused() -> None:
    """The case the cheap feasibility test cannot catch, and says it cannot.

    Two indicators of disjoint sets, each asked for a probability of 0.6. Both
    targets lie strictly inside ``(0, 1)``, which is all a per-view box check
    can see, and no distribution has both because they would sum to 1.2.
    """
    one = [1.0, 0.0, 0.0]
    other = [0.0, 1.0, 0.0]
    with pytest.raises(ViewsInfeasible, match="jointly unreachable"):
        pool([View(tuple(one), 0.6, "first"), View(tuple(other), 0.6, "second")])


def test_malformed_arguments_are_refused() -> None:
    with pytest.raises(TooShort):
        View((), 0.0)
    with pytest.raises(ValueError, match="must be finite"):
        View((1.0, float("inf")), 0.0)
    with pytest.raises(ValueError, match="strictly inside"):
        probability_view([True, False], 0.0)
    with pytest.raises(ValueError, match="strictly inside"):
        probability_view([True, False], 1.0)
    with pytest.raises(Misaligned):
        pool([mean_view([0.0, 1.0], 0.5), mean_view([0.0, 1.0, 2.0], 1.0)])
    with pytest.raises(ValueError, match="sum to one"):
        pool([mean_view([0.0, 1.0, 2.0], 1.0)], [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        pool([mean_view([0.0, 1.0, 2.0], 1.0)], [1.5, -0.5, 0.0])
    with pytest.raises(ValueError, match="tolerance must be positive"):
        pool([mean_view([0.0, 1.0, 2.0], 1.0)], tolerance=0.0)


# -- the diagnostics -----------------------------------------------------------


def test_relative_entropy_is_non_negative_and_zero_only_at_agreement() -> None:
    assert relative_entropy(uniform(4)) == 0.0
    assert relative_entropy((0.5, 0.25, 0.25), (0.25, 0.25, 0.5)) > 0.0
    assert relative_entropy((1.0, 0.0), (0.5, 0.5)) == pytest.approx(math.log(2.0))
    # A posterior that keeps a scenario the prior ruled out is infinitely
    # surprising, which is a different thing from being expensive.
    with pytest.raises(ValueError, match="infinitely surprising"):
        relative_entropy((0.5, 0.5), (1.0, 0.0))
    with pytest.raises(TooShort):
        relative_entropy(())
    with pytest.raises(Misaligned):
        relative_entropy((0.5, 0.5), (1.0,))


def test_effective_scenarios_counts_the_scenarios_doing_the_work() -> None:
    assert effective_scenarios(uniform(10)) == pytest.approx(10.0, rel=1e-14)
    assert effective_scenarios((1.0, 0.0, 0.0)) == pytest.approx(1.0, rel=1e-14)
    assert 1.0 < effective_scenarios((0.7, 0.2, 0.1)) < 3.0
    with pytest.raises(TooShort):
        effective_scenarios(())


def test_a_stronger_view_costs_more_entropy_and_more_of_the_sample() -> None:
    first, _ = sample()
    base = math.fsum(first) / SCENARIOS
    entropies = []
    effective = []
    for shift in (0.1, 0.25, 0.5, 1.0, 1.5):
        posterior = pool([mean_view(first, base - shift)])
        entropies.append(posterior.relative_entropy)
        effective.append(posterior.effective_scenarios)
        assert 0.0 < posterior.concentration <= 1.0
    assert entropies == sorted(entropies)
    assert effective == sorted(effective, reverse=True)
    # And the headline figure: a view of a full standard deviation leaves a
    # thousand-scenario set behaving like rather fewer than a thousand.
    assert 500.0 < effective[3] < 640.0


# -- what a view does to what it did not mention -------------------------------


def test_a_mean_view_moves_an_unmentioned_series_by_its_regression_slope() -> None:
    """The exponential tilt reproduces the least-squares projection, to four figures.

    Worth asserting in both directions. It says the machinery is not inventing
    a relationship that is not in the sample — and it says that if all anybody
    wants is the mean of something else, a regression would have done.
    """
    first, second = sample()
    mean_first = math.fsum(first) / SCENARIOS
    mean_second = math.fsum(second) / SCENARIOS
    variance = math.fsum((value - mean_first) ** 2 for value in first) / SCENARIOS
    covariance = (
        math.fsum(
            (one - mean_first) * (other - mean_second)
            for one, other in zip(first, second, strict=True)
        )
        / SCENARIOS
    )
    slope = covariance / variance
    for shift in (0.1, 0.25, 0.5, 1.0):
        posterior = pool([mean_view(first, mean_first - shift)])
        moved = math.fsum(
            weight * value
            for weight, value in zip(posterior.weights, second, strict=True)
        )
        implied = (moved - mean_second) / -shift
        assert implied == pytest.approx(slope, rel=2e-3)


def test_the_tail_moves_far_less_than_the_regression_would_shift_it() -> None:
    """The finding the module exists for, and the shortcut it contradicts.

    Shifting the loss distribution by beta times the view is what a desk does
    by hand. It gets the mean exactly right and overstates a 99% expected
    shortfall by a factor of between two and three, because the reweighting
    concentrates on scenarios where the *first* series was low and the second
    series' worst scenarios are only partly those.
    """
    first, second = sample()
    mean_first = math.fsum(first) / SCENARIOS
    mean_second = math.fsum(second) / SCENARIOS
    variance = math.fsum((value - mean_first) ** 2 for value in first) / SCENARIOS
    covariance = (
        math.fsum(
            (one - mean_first) * (other - mean_second)
            for one, other in zip(first, second, strict=True)
        )
        / SCENARIOS
    )
    slope = covariance / variance
    unstressed = stressed_risk(second, uniform(), 0.99)
    for shift in (0.25, 0.5, 1.0):
        posterior = pool([mean_view(first, mean_first - shift)])
        stressed = stressed_risk(second, posterior.weights, 0.99)
        predicted = slope * shift
        realised = stressed.expected_shortfall - unstressed.expected_shortfall
        assert realised > 0.0
        assert 0.3 < realised / predicted < 0.5
    # The mean, by contrast, moves by exactly the predicted amount.
    posterior = pool([mean_view(first, mean_first - 0.5)])
    moved = math.fsum(
        weight * value
        for weight, value in zip(posterior.weights, second, strict=True)
    )
    assert mean_second - moved == pytest.approx(slope * 0.5, rel=1e-3)


# -- the confidence blend ------------------------------------------------------


def test_the_blend_is_exactly_optimal_for_a_probability_view() -> None:
    """On a partition the blend is the entropy minimiser, at every confidence.

    Scaling a partition's cells and then blending two such scalings leaves the
    weights within each cell proportional to the prior's, which is the form the
    minimiser takes. So there is nothing to gain by re-solving, and the
    assertion is on equality rather than on an inequality.
    """
    first, _ = sample()
    flags = [value < -2.0 for value in first]
    prior = sum(flags) / SCENARIOS
    full = pool([probability_view(flags, 3.0 * prior)])
    for confidence in (0.25, 0.5, 0.75):
        blended = temper(full, confidence)
        realised = math.fsum(
            weight * (1.0 if flag else 0.0)
            for weight, flag in zip(blended.weights, flags, strict=True)
        )
        resolved = pool([probability_view(flags, realised)])
        assert blended.relative_entropy == pytest.approx(
            resolved.relative_entropy, rel=1e-10
        )


def test_the_blend_costs_entropy_for_a_mean_view_and_most_at_low_confidence() -> None:
    """A blend of two exponential tilts is not an exponential tilt.

    The cost is where it would not be looked for: largest at *low* confidence
    in a *strong* view, and shrinking towards either end of the confidence
    range, where the two distributions coincide.
    """
    first, _ = sample()
    base = math.fsum(first) / SCENARIOS
    excesses = []
    full = pool([mean_view(first, base - 1.0)])
    for confidence in (0.25, 0.5, 0.75):
        blended = temper(full, confidence)
        realised = math.fsum(
            weight * value
            for weight, value in zip(blended.weights, first, strict=True)
        )
        resolved = pool([mean_view(first, realised)])
        assert blended.relative_entropy > resolved.relative_entropy
        excesses.append(blended.relative_entropy / resolved.relative_entropy - 1.0)
    assert excesses == sorted(excesses, reverse=True)
    assert excesses[0] > 0.2
    # A milder view costs far less, so the gap is about the view and not about
    # the blending.
    mild = pool([mean_view(first, base - 0.25)])
    blended = temper(mild, 0.25)
    realised = math.fsum(
        weight * value for weight, value in zip(blended.weights, first, strict=True)
    )
    resolved = pool([mean_view(first, realised)])
    assert 0.0 < blended.relative_entropy / resolved.relative_entropy - 1.0 < 0.05


def test_temper_at_the_ends_returns_the_ends() -> None:
    first, _ = sample()
    base = math.fsum(first) / SCENARIOS
    full = pool([mean_view(first, base - 0.5)])
    assert temper(full, 1.0) is full
    none = temper(full, 0.0)
    assert none.weights == pytest.approx(uniform())
    assert none.relative_entropy == pytest.approx(0.0, abs=1e-15)
    assert none.multipliers == ()
    with pytest.raises(ValueError, match=r"confidence lies in \[0, 1\]"):
        temper(full, 1.5)


# -- reading risk off the posterior -------------------------------------------


def test_the_weighted_quantile_is_an_order_statistic_under_uniform_weights() -> None:
    """Uninterpolated on purpose, so the answer is an observation.

    Under uniform weights the cumulative weight reaches ``p`` at the
    ``ceil(p n)``-th smallest observation, and that is what comes back — not a
    blend of two of them.
    """
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    ordered = sorted(values)
    for probability in (0.05, 0.2, 0.25, 0.5, 0.8, 1.0):
        position = max(1, math.ceil(probability * len(values)))
        assert weighted_quantile(values, uniform(5), probability) == ordered[
            position - 1
        ]
    assert weighted_quantile(values, uniform(5), 0.0) == ordered[0]


def test_the_weighted_quantile_follows_the_weights() -> None:
    values = [-3.0, -1.0, 0.0, 1.0]
    # Only 1% of the mass sits at or below -1, so the 5% quantile is not there.
    assert weighted_quantile(values, (0.001, 0.009, 0.5, 0.49), 0.05) == 0.0
    assert weighted_quantile(values, (0.001, 0.2, 0.3, 0.499), 0.05) == -1.0
    assert weighted_quantile(values, (0.5, 0.2, 0.2, 0.1), 0.05) == -3.0
    with pytest.raises(TooShort):
        weighted_quantile([], (), 0.5)
    with pytest.raises(Misaligned):
        weighted_quantile([1.0], (0.5, 0.5), 0.5)
    with pytest.raises(ValueError, match=r"probability lies in \[0, 1\]"):
        weighted_quantile(values, uniform(4), 1.5)


def test_the_weighted_shortfall_reports_the_mass_it_actually_used() -> None:
    """Which is not the tail probability, and the gap grows after a tilt.

    The quantile lands on an observation carrying weight, so the tail up to and
    including it holds more mass than was asked for. The average divides by
    what it used and says what that was.
    """
    first, _ = sample()
    average, mass = weighted_expected_shortfall(first, uniform(), 0.05)
    assert mass == pytest.approx(0.05, abs=2e-3)
    assert average < 0.0
    flags = [value < -1.0 for value in first]
    posterior = pool([probability_view(flags, 3.0 * sum(flags) / SCENARIOS)])
    tilted_average, tilted_mass = weighted_expected_shortfall(
        first, posterior.weights, 0.05
    )
    assert tilted_mass >= 0.05
    assert tilted_average < average
    with pytest.raises(ValueError, match=r"tail probability lies in \(0, 1\]"):
        weighted_expected_shortfall(first, uniform(), 0.0)
    with pytest.raises(TooShort):
        weighted_expected_shortfall([], (), 0.5)


def test_stressed_risk_is_coherent_in_its_own_terms() -> None:
    first, _ = sample()
    flags = [value < -1.0 for value in first]
    posterior = pool([probability_view(flags, 2.0 * sum(flags) / SCENARIOS)])
    for confidence in (0.9, 0.95, 0.99):
        risk = stressed_risk(first, posterior.weights, confidence)
        assert risk.expected_shortfall >= risk.value_at_risk
        assert risk.value_at_risk == -risk.quantile
        assert risk.confidence == confidence
        assert risk.distribution.value == "historical"
        assert risk.volatility > 0.0
    unstressed = stressed_risk(first, uniform(), 0.99)
    stressed = stressed_risk(first, posterior.weights, 0.99)
    assert stressed.value_at_risk > unstressed.value_at_risk
    assert stressed.mean < unstressed.mean
    with pytest.raises(ValueError, match="strictly inside"):
        stressed_risk(first, uniform(), 1.0)
    with pytest.raises(TooShort):
        stressed_risk([], (), 0.99)


def test_the_posterior_reports_what_it_is() -> None:
    first, _ = sample()
    posterior = pool([mean_view(first, math.fsum(first) / SCENARIOS - 0.2)])
    assert isinstance(posterior, Posterior)
    assert posterior.scenarios == SCENARIOS
    assert 0.0 < posterior.concentration < 1.0
    assert len(posterior.multipliers) == 1
    assert posterior.multipliers[0] != 0.0
    assert posterior.iterations >= 1


def three_assets() -> tuple[list[float], list[float]]:
    """A driver and a portfolio built on it, fixed, with a modest loading.

    Separate from :func:`sample` because the finding below needs a portfolio
    whose own tail is only partly driven by the series the view is placed on.
    With a loading near one the two tails coincide and the comparison has
    nothing to say.
    """
    rng = random.Random(7)
    driver: list[float] = []
    portfolio: list[float] = []
    for _ in range(SCENARIOS):
        first = rng.gauss(0.0003, 0.011)
        second = 0.45 * first + 0.0004 + 0.006 * rng.gauss(0.0, 1.0)
        third = -0.2 * first + 0.0001 + 0.003 * rng.gauss(0.0, 1.0)
        driver.append(first)
        portfolio.append((first + second + third) / 3.0)
    return driver, portfolio


def captured_fraction(
    driver: list[float], portfolio: list[float], view: View
) -> float:
    """How much of the parallel-shift prediction the reweighting actually delivers."""
    count = len(driver)
    flat = uniform(count)
    mean_driver = math.fsum(driver) / count
    before = stressed_risk(portfolio, flat, 0.99)
    variance = math.fsum((value - mean_driver) ** 2 for value in driver) / count
    covariance = (
        math.fsum(
            (one - mean_driver) * (other - before.mean)
            for one, other in zip(driver, portfolio, strict=True)
        )
        / count
    )
    slope = covariance / variance
    posterior = pool([view])
    after = stressed_risk(portfolio, posterior.weights, 0.99)
    move = (
        math.fsum(
            weight * value
            for weight, value in zip(posterior.weights, driver, strict=True)
        )
        - mean_driver
    )
    return (after.expected_shortfall - before.expected_shortfall) / (-slope * move)


def test_the_shortcut_misses_in_both_directions() -> None:
    """Which is worse than missing in one, because no factor corrects it.

    Shifting the loss distribution by beta times the view's move is the
    comparison this module exists to beat, and it is not beaten by a fixed
    amount. A view on a mean spreads its weight over the whole sample and moves
    the far tail by *less* than the shift predicts; a view on the probability of
    a tail event puts its weight where the losses already are and moves it by
    *more*. On one sample, with one portfolio, the same two kinds of view land
    on opposite sides of the prediction.
    """
    driver, portfolio = three_assets()
    mean_driver = math.fsum(driver) / SCENARIOS
    on_mean = captured_fraction(
        driver, portfolio, mean_view(driver, mean_driver - 0.01)
    )
    flags = [value < -0.02 for value in driver]
    observed = sum(flags) / SCENARIOS
    assert observed > 0.0
    on_tail = captured_fraction(
        driver, portfolio, probability_view(flags, 3.0 * observed)
    )
    assert on_mean < 1.0 < on_tail, (on_mean, on_tail)

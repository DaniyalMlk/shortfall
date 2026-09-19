"""Euler allocation of portfolio risk.

Two things are worth testing about a risk allocation and they are different
things. The first is the summation identity — that the components add to the
total — which is what makes the pieces meaningful. The second is that each
piece is actually the derivative it claims to be, which the identity does *not*
establish: any vector scaled to sum to the total satisfies the identity, and a
wrong gradient scaled that way would pass a summation check while being wrong
about every individual position.

So every gradient here is checked against a central finite difference of the
estimator itself — a route to the number that shares no code with the analytic
one — and the identity is checked separately on top of it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

import pytest

from shortfall.contributions import (
    Allocation,
    NoRiskToAllocate,
    historical_shortfall_contributions,
    risk_contributions,
    unit_risk,
    volatility_contributions,
)
from shortfall.parametric import Distribution, parametric_risk, portfolio_risk
from shortfall.series import Panel

#: A covariance with a negative entry, so the hedging case is exercised rather
#: than only the comfortable all-positive one.
COVARIANCE = [
    [0.0400, 0.0120, -0.0040],
    [0.0120, 0.0900, 0.0060],
    [-0.0040, 0.0060, 0.0225],
]
WEIGHTS = [0.5, 0.3, 0.2]
NAMES = ["A", "B", "C"]


def central_difference(
    measure: Callable[[Sequence[float]], float],
    weights: Sequence[float],
    index: int,
    step: float = 1e-6,
) -> float:
    """``d measure / d w_index`` by a symmetric difference.

    Symmetric rather than forward: the forward difference has an error linear in
    the step, and at a step small enough to make that error negligible the
    subtraction has lost most of its significant digits. The symmetric form is
    quadratic in the step and lets the step stay large enough to be safe.
    """
    up = list(weights)
    down = list(weights)
    up[index] += step
    down[index] -= step
    return (measure(up) - measure(down)) / (2.0 * step)


def assert_sums_to_total(allocation: Allocation) -> None:
    assert allocation.identity_error == pytest.approx(0.0, abs=1e-14)
    assert math.fsum(allocation.percentage) == pytest.approx(1.0, abs=1e-12)


# -- volatility --------------------------------------------------------------


def test_volatility_components_sum_to_the_volatility() -> None:
    allocation = volatility_contributions(WEIGHTS, COVARIANCE, names=NAMES)
    expected = math.sqrt(
        sum(
            WEIGHTS[i] * COVARIANCE[i][j] * WEIGHTS[j]
            for i in range(3)
            for j in range(3)
        )
    )
    assert allocation.total == pytest.approx(expected)
    assert_sums_to_total(allocation)


def test_volatility_marginals_are_the_gradient() -> None:
    def volatility(weights: Sequence[float]) -> float:
        return math.sqrt(
            sum(
                weights[i] * COVARIANCE[i][j] * weights[j]
                for i in range(3)
                for j in range(3)
            )
        )

    allocation = volatility_contributions(WEIGHTS, COVARIANCE, names=NAMES)
    for index in range(3):
        assert allocation.marginal[index] == pytest.approx(
            central_difference(volatility, WEIGHTS, index), rel=1e-7
        )


def test_volatility_allocation_is_homogeneous_of_degree_one() -> None:
    """Doubling every weight doubles each component and leaves each share alone.

    Which is the property Euler's theorem needs, checked on the implementation
    rather than assumed from the algebra.
    """
    single = volatility_contributions(WEIGHTS, COVARIANCE)
    double = volatility_contributions([2.0 * w for w in WEIGHTS], COVARIANCE)
    assert double.total == pytest.approx(2.0 * single.total)
    for one, two in zip(single.component, double.component, strict=True):
        assert two == pytest.approx(2.0 * one)
    for one, two in zip(single.percentage, double.percentage, strict=True):
        assert two == pytest.approx(one)


def test_marginal_contribution_is_beta_times_portfolio_volatility() -> None:
    """The same number by the other route it is usually quoted as.

    An asset's marginal contribution to volatility is ``cov(r_i, r_p) / sigma_p``,
    which is ``beta_i * sigma_p`` where ``beta_i`` is its regression coefficient
    on the portfolio. Both are computed here from the covariance matrix without
    going through the allocation.
    """
    allocation = volatility_contributions(WEIGHTS, COVARIANCE)
    variance = sum(
        WEIGHTS[i] * COVARIANCE[i][j] * WEIGHTS[j] for i in range(3) for j in range(3)
    )
    for index in range(3):
        covariance_with_portfolio = sum(
            COVARIANCE[index][j] * WEIGHTS[j] for j in range(3)
        )
        beta = covariance_with_portfolio / variance
        assert allocation.marginal[index] == pytest.approx(beta * math.sqrt(variance))


def test_a_single_asset_portfolio_contributes_all_of_its_own_risk() -> None:
    allocation = volatility_contributions([1.0, 0.0, 0.0], COVARIANCE, names=NAMES)
    assert allocation.total == pytest.approx(math.sqrt(COVARIANCE[0][0]))
    assert allocation.component[0] == pytest.approx(allocation.total)
    assert allocation.component[1] == pytest.approx(0.0)
    assert allocation.component[2] == pytest.approx(0.0)


def test_a_hedge_contributes_negative_risk() -> None:
    """A position anticorrelated with the rest removes risk, and says so.

    Reported as a negative component rather than clamped. The identity still
    holds — the negative piece is exactly how much risk the position takes out.
    """
    covariance = [[0.04, -0.03], [-0.03, 0.04]]
    allocation = volatility_contributions([0.9, 0.35], covariance, names=["long", "hedge"])
    assert allocation.component[1] < 0.0
    assert allocation.has_negative_contribution
    assert_sums_to_total(allocation)


def test_zero_weights_have_no_gradient_to_take() -> None:
    with pytest.raises(NoRiskToAllocate):
        volatility_contributions([0.0, 0.0, 0.0], COVARIANCE)


def test_weights_in_the_null_space_have_no_gradient_either() -> None:
    """A singular covariance can give a genuinely riskless combination.

    Two assets that are the same asset: any long-short pair of equal size has no
    volatility at all, and so no volatility to allocate.
    """
    covariance = [[0.04, 0.04], [0.04, 0.04]]
    with pytest.raises(NoRiskToAllocate):
        volatility_contributions([1.0, -1.0], covariance)


def test_a_covariance_that_is_not_positive_semidefinite_is_refused() -> None:
    covariance = [[0.04, 0.09], [0.09, 0.04]]
    with pytest.raises(ValueError, match="not positive semi-definite"):
        volatility_contributions([1.0, -1.0], covariance)


def test_weights_must_match_the_covariance() -> None:
    with pytest.raises(ValueError, match="against a 3x3"):
        volatility_contributions([0.5, 0.5], COVARIANCE)


def test_names_must_match_the_assets() -> None:
    with pytest.raises(ValueError, match="names against"):
        volatility_contributions(WEIGHTS, COVARIANCE, names=["A", "B"])


def test_names_default_to_positions() -> None:
    allocation = volatility_contributions(WEIGHTS, COVARIANCE)
    assert allocation.names == ("asset 0", "asset 1", "asset 2")


# -- the location-scale constant ---------------------------------------------


@pytest.mark.parametrize(
    ("distribution", "extra"),
    [
        (Distribution.NORMAL, {}),
        (Distribution.STUDENT_T, {"degrees": 6.0}),
        (Distribution.STUDENT_T, {"degrees": 3.5}),
        (Distribution.CORNISH_FISHER, {"skewness": -0.4, "excess_kurtosis": 1.2}),
    ],
)
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99, 0.999])
def test_risk_is_affine_in_mean_and_volatility(
    distribution: Distribution, extra: dict[str, float], confidence: float
) -> None:
    """``rho(mean, vol) == -mean + vol * rho(0, 1)``, which the allocation relies on.

    If this ever stopped holding for some distribution, the Euler allocation
    below would be allocating a measure that is not the one being reported. It
    is the assumption the whole module rests on, so it is tested directly rather
    than inferred from the allocation identity that follows from it.
    """
    unit = unit_risk(confidence=confidence, distribution=distribution, **extra)
    mean, volatility = 0.004, 0.027
    actual = parametric_risk(
        mean=mean,
        volatility=volatility,
        confidence=confidence,
        distribution=distribution,
        **extra,
    )
    assert actual.value_at_risk == pytest.approx(
        -mean + volatility * unit.value_at_risk, rel=1e-13
    )
    assert actual.expected_shortfall == pytest.approx(
        -mean + volatility * unit.expected_shortfall, rel=1e-13
    )


# -- parametric value at risk and expected shortfall -------------------------


@pytest.mark.parametrize("shortfall_instead", [False, True])
@pytest.mark.parametrize(
    ("distribution", "extra"),
    [
        (Distribution.NORMAL, {}),
        (Distribution.STUDENT_T, {"degrees": 5.0}),
        (Distribution.CORNISH_FISHER, {"skewness": -0.3, "excess_kurtosis": 0.8}),
    ],
)
def test_parametric_components_sum_to_the_reported_total(
    shortfall_instead: bool, distribution: Distribution, extra: dict[str, float]
) -> None:
    means = [0.0008, 0.0015, 0.0004]
    allocation = risk_contributions(
        WEIGHTS,
        COVARIANCE,
        means=means,
        confidence=0.975,
        distribution=distribution,
        of_expected_shortfall=shortfall_instead,
        names=NAMES,
        **extra,
    )
    reported = portfolio_risk(
        WEIGHTS,
        COVARIANCE,
        means=means,
        confidence=0.975,
        distribution=distribution,
        **extra,
    )
    expected = (
        reported.expected_shortfall if shortfall_instead else reported.value_at_risk
    )
    assert allocation.total == pytest.approx(expected, rel=1e-13)
    assert_sums_to_total(allocation)


@pytest.mark.parametrize("shortfall_instead", [False, True])
def test_parametric_marginals_are_the_gradient(shortfall_instead: bool) -> None:
    means = [0.0008, 0.0015, 0.0004]

    def measure(weights: Sequence[float]) -> float:
        risk = portfolio_risk(weights, COVARIANCE, means=means, confidence=0.99)
        return risk.expected_shortfall if shortfall_instead else risk.value_at_risk

    allocation = risk_contributions(
        WEIGHTS,
        COVARIANCE,
        means=means,
        confidence=0.99,
        of_expected_shortfall=shortfall_instead,
    )
    for index in range(3):
        assert allocation.marginal[index] == pytest.approx(
            central_difference(measure, WEIGHTS, index), rel=1e-6
        )


def test_value_at_risk_allocation_is_the_volatility_allocation_scaled() -> None:
    """With no drift the two differ only by the distributional constant.

    Worth pinning down because it is the reason a value-at-risk allocation
    carries no information a volatility allocation does not, once the drift is
    zero — a point that is easy to lose when the two are quoted separately.
    """
    volatility = volatility_contributions(WEIGHTS, COVARIANCE)
    at_risk = risk_contributions(WEIGHTS, COVARIANCE, confidence=0.99)
    scale = unit_risk(confidence=0.99).value_at_risk
    for one, two in zip(volatility.component, at_risk.component, strict=True):
        assert two == pytest.approx(scale * one)
    for one, two in zip(volatility.percentage, at_risk.percentage, strict=True):
        assert two == pytest.approx(one)


def test_drift_makes_the_two_allocations_differ() -> None:
    """The shares are only equal without a drift; with one they must not be."""
    volatility = volatility_contributions(WEIGHTS, COVARIANCE)
    at_risk = risk_contributions(
        WEIGHTS, COVARIANCE, means=[0.01, 0.03, 0.002], confidence=0.99
    )
    assert at_risk.percentage != pytest.approx(volatility.percentage)
    assert_sums_to_total(at_risk)


def test_a_large_expected_return_can_make_a_contribution_negative() -> None:
    """A position whose drift outweighs its risk share reduces value at risk.

    Not an error: at a low enough confidence level the expected gain on a
    position genuinely exceeds the loss it contributes at that quantile.
    """
    allocation = risk_contributions(
        WEIGHTS, COVARIANCE, means=[0.0, 0.0, 0.5], confidence=0.90
    )
    assert allocation.component[2] < 0.0
    assert_sums_to_total(allocation)


def test_expected_shortfall_allocates_more_than_value_at_risk() -> None:
    at_risk = risk_contributions(WEIGHTS, COVARIANCE, confidence=0.99)
    shortfall_of = risk_contributions(
        WEIGHTS, COVARIANCE, confidence=0.99, of_expected_shortfall=True
    )
    assert shortfall_of.total > at_risk.total
    for one, two in zip(at_risk.component, shortfall_of.component, strict=True):
        assert abs(two) > abs(one)


def test_the_measure_is_recorded_on_the_result() -> None:
    assert volatility_contributions(WEIGHTS, COVARIANCE).measure == "volatility"
    assert (
        risk_contributions(WEIGHTS, COVARIANCE, confidence=0.99).measure
        == "normal value at risk"
    )
    assert (
        risk_contributions(
            WEIGHTS,
            COVARIANCE,
            confidence=0.99,
            distribution=Distribution.STUDENT_T,
            of_expected_shortfall=True,
        ).measure
        == "student-t expected shortfall"
    )


def test_means_must_match_the_assets() -> None:
    with pytest.raises(ValueError, match="means against"):
        risk_contributions(WEIGHTS, COVARIANCE, means=[0.1, 0.2])


def test_parametric_allocation_needs_volatility() -> None:
    with pytest.raises(NoRiskToAllocate):
        risk_contributions([0.0, 0.0, 0.0], COVARIANCE)


# -- sample expected shortfall -----------------------------------------------


def sample_panel(seed: int = 7, count: int = 400) -> Panel:
    rng = random.Random(seed)
    return Panel.from_columns(
        {
            "A": [rng.gauss(0.0004, 0.010) for _ in range(count)],
            "B": [rng.gauss(0.0006, 0.021) for _ in range(count)],
            "C": [rng.gauss(0.0002, 0.015) for _ in range(count)],
        }
    )


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_sample_components_sum_to_the_sample_expected_shortfall(
    confidence: float,
) -> None:
    """And to the same number the one-dimensional estimator gives.

    This is the check that the fractional tail weighting matches. Rounding the
    tail to a whole number of observations in either place leaves an error that
    is invisible at 90% over 400 days and grows as the tail thins.
    """
    from shortfall.historical import sample_expected_shortfall

    panel = sample_panel()
    allocation = historical_shortfall_contributions(
        panel, WEIGHTS, confidence=confidence
    )
    tail_mean, _ = sample_expected_shortfall(
        panel.portfolio(WEIGHTS).values, 1.0 - confidence
    )
    assert allocation.total == pytest.approx(-tail_mean, rel=1e-13)
    assert_sums_to_total(allocation)


def test_sample_allocation_matches_the_historical_estimator() -> None:
    from shortfall.historical import historical_risk

    panel = sample_panel()
    allocation = historical_shortfall_contributions(panel, WEIGHTS, confidence=0.95)
    estimate = historical_risk(panel.portfolio(WEIGHTS), confidence=0.95)
    assert allocation.total == pytest.approx(estimate.expected_shortfall, rel=1e-13)


def test_sample_marginal_is_the_tail_conditional_mean() -> None:
    """With an integral number of tail observations the weighting is unambiguous.

    200 observations at 95% puts exactly ten in the tail with no fractional
    part, so the marginal must be the plain average of each asset's return on
    those ten days — computed here by sorting and averaging, with none of the
    allocation's machinery.
    """
    panel = sample_panel(seed=11, count=200)
    portfolio = panel.portfolio(WEIGHTS)
    order = sorted(range(200), key=lambda t: portfolio.values[t])
    worst = order[:10]
    allocation = historical_shortfall_contributions(panel, WEIGHTS, confidence=0.95)
    for index, asset in enumerate(panel.series):
        plain = -sum(asset.values[t] for t in worst) / 10.0
        assert allocation.marginal[index] == pytest.approx(plain, rel=1e-12)


def test_sample_allocation_uses_the_portfolio_tail_not_each_asset_own() -> None:
    """The distinction the implementation exists to get right.

    Averaging each asset over *its own* worst days gives a larger number for
    every asset, because no asset has its worst days exactly when the portfolio
    does. That version would not sum to the portfolio's expected shortfall, and
    it would overstate every position's contribution.
    """
    panel = sample_panel()
    allocation = historical_shortfall_contributions(panel, WEIGHTS, confidence=0.95)
    for index, asset in enumerate(panel.series):
        own_worst = sorted(asset.values)[:20]
        own_tail = -sum(own_worst) / 20.0
        assert allocation.marginal[index] < own_tail


def test_sample_allocation_reflects_tail_dependence_the_covariance_misses() -> None:
    """Two assets with the same volatility and correlation, differing only in the tail.

    Built so the covariance matrix cannot tell them apart to better than
    sampling error, while one of them is the one that falls when the portfolio
    falls. The historical allocation separates them; a Gaussian one cannot.
    """
    rng = random.Random(3)
    base = [rng.gauss(0.0, 0.01) for _ in range(600)]
    # ``twin`` moves with the base only in the tail; ``mirror`` moves with it
    # only in the body. Both are scaled to a similar overall covariance.
    twin = []
    mirror = []
    for value in base:
        noise = rng.gauss(0.0, 0.008)
        if value < -0.012:
            twin.append(value + noise)
            mirror.append(-0.5 * value + noise)
        else:
            twin.append(0.2 * value + noise)
            mirror.append(1.6 * value + noise)
    panel = Panel.from_columns({"base": base, "twin": twin, "mirror": mirror})
    allocation = historical_shortfall_contributions(
        panel, [0.4, 0.3, 0.3], confidence=0.95
    )
    assert allocation.of("twin") > allocation.of("mirror")
    assert_sums_to_total(allocation)


def test_sample_weights_must_match_the_panel() -> None:
    with pytest.raises(ValueError, match="against 3 assets"):
        historical_shortfall_contributions(sample_panel(), [0.5, 0.5])


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.01, 99.0])
def test_sample_confidence_must_be_a_confidence(confidence: float) -> None:
    """The endpoints are excluded as well as the values outside.

    At a confidence of one the tail has no probability and the conditional mean
    is an average over nothing; at zero the "tail" is the whole sample and the
    number is not a tail statistic. 99 is the common mistake of passing a
    percentage, and 0.01 is the other one — passing the tail probability
    instead of the confidence level — which is *not* rejected here because it is
    a perfectly good confidence level, just not the one the caller meant. That
    one is caught by the convention being stated, not by a check.
    """
    with pytest.raises(ValueError, match="strictly inside"):
        historical_shortfall_contributions(
            sample_panel(), WEIGHTS, confidence=confidence
        )


def test_sample_allocation_takes_its_names_from_the_panel() -> None:
    allocation = historical_shortfall_contributions(sample_panel(), WEIGHTS)
    assert allocation.names == ("A", "B", "C")


# -- the result object -------------------------------------------------------


def test_largest_contributor_is_reported_by_name() -> None:
    allocation = volatility_contributions(WEIGHTS, COVARIANCE, names=NAMES)
    name, component = allocation.largest()
    assert name == "A"
    assert component == pytest.approx(max(allocation.component))


def test_lookup_by_name() -> None:
    allocation = volatility_contributions(WEIGHTS, COVARIANCE, names=NAMES)
    assert allocation.of("B") == pytest.approx(allocation.component[1])
    with pytest.raises(KeyError):
        allocation.of("D")


def test_an_allocation_with_mismatched_lengths_is_refused() -> None:
    with pytest.raises(ValueError, match="one entry per asset"):
        Allocation(
            names=("A", "B"),
            weights=(0.5,),
            marginal=(0.1, 0.2),
            component=(0.05, 0.1),
            total=0.15,
            measure="volatility",
        )


def test_length_is_the_number_of_positions() -> None:
    assert len(volatility_contributions(WEIGHTS, COVARIANCE)) == 3


def test_shares_of_a_zero_total_are_refused() -> None:
    allocation = Allocation(
        names=("A",),
        weights=(0.0,),
        marginal=(0.0,),
        component=(0.0,),
        total=0.0,
        measure="volatility",
    )
    with pytest.raises(NoRiskToAllocate):
        _ = allocation.percentage

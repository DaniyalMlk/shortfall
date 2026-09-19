"""Risk parity, and the diversification measures built on an allocation.

Risk parity has closed forms in two cases and they are both used here, because
an iterative solver that agrees with its own stopping rule has established
nothing. The cases are worth stating:

*Two assets.* Equating the two risk contributions gives ``w1^2 s1^2 = w2^2
s2^2`` — the cross terms are the same on both sides and cancel — so the answer
is inverse volatility **whatever the correlation is**. A solver that quietly
depends on the correlation for two assets is wrong, and testing at one
correlation would not show it.

*Constant correlation, any number of assets.* Inverse volatility again, exactly.
Substituting ``w_i = c / s_i`` into a matrix with ``S_ij = rho s_i s_j`` gives
every contribution equal to ``c^2 (1 + rho (n - 1))``.

Beyond those, correctness is checked against the defining condition directly —
that the risk shares equal the budgets — which is independent of how the
weights were found.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.contributions import (
    DidNotConverge,
    NegativeContribution,
    NoRiskToAllocate,
    concentration,
    diversification_ratio,
    effective_bets,
    principal_bets,
    risk_parity,
    volatility_contributions,
)


def constant_correlation(volatilities: list[float], rho: float) -> list[list[float]]:
    size = len(volatilities)
    return [
        [
            volatilities[i] * volatilities[j] * (1.0 if i == j else rho)
            for j in range(size)
        ]
        for i in range(size)
    ]


def random_covariance(size: int, *, seed: int, draws: int | None = None) -> list[list[float]]:
    """A full-rank covariance from a random sample, so it is a real matrix.

    ``draws`` above ``size`` keeps it positive definite; the estimator is the
    plain second-moment matrix because the test wants a matrix, not an estimate.
    """
    rng = random.Random(seed)
    draws = draws if draws is not None else 4 * size
    rows = [[rng.gauss(0.0, 0.02) for _ in range(draws)] for _ in range(size)]
    return [
        [math.fsum(rows[i][k] * rows[j][k] for k in range(draws)) / draws for j in range(size)]
        for i in range(size)
    ]


# -- the closed forms --------------------------------------------------------


@pytest.mark.parametrize("rho", [-0.95, -0.5, -0.1, 0.0, 0.3, 0.7, 0.99])
def test_two_asset_risk_parity_is_inverse_volatility_at_any_correlation(
    rho: float,
) -> None:
    first, second = 0.12, 0.31
    covariance = constant_correlation([first, second], rho)
    result = risk_parity(covariance)
    total = 1.0 / first + 1.0 / second
    assert result.weights[0] == pytest.approx((1.0 / first) / total, rel=1e-12)
    assert result.weights[1] == pytest.approx((1.0 / second) / total, rel=1e-12)
    assert result.converged


@pytest.mark.parametrize("fraction", [-0.9, -0.4, 0.0, 0.4, 0.85])
@pytest.mark.parametrize("size", [3, 5, 8])
def test_constant_correlation_risk_parity_is_inverse_volatility(
    fraction: float, size: int
) -> None:
    """``fraction`` is a share of the range of correlations a matrix of this size
    can actually have.

    A constant-correlation matrix has eigenvalues ``1 + (n-1) rho`` and
    ``1 - rho``, so it stops being positive definite below ``-1/(n-1)`` — which
    at eight assets is only -0.143. Writing a fixed negative correlation into
    this parametrisation builds an indefinite matrix at the larger sizes and
    tests the solver on a problem that has no solution.
    """
    rho = fraction * (1.0 / (size - 1) if fraction < 0 else 1.0) * 0.99
    volatilities = [0.05 + 0.07 * index for index in range(size)]
    result = risk_parity(constant_correlation(volatilities, rho))
    total = math.fsum(1.0 / one for one in volatilities)
    for index, one in enumerate(volatilities):
        assert result.weights[index] == pytest.approx((1.0 / one) / total, rel=1e-11)


def test_identical_assets_get_identical_weights() -> None:
    covariance = constant_correlation([0.2, 0.2, 0.2, 0.2], 0.35)
    result = risk_parity(covariance)
    for weight in result.weights:
        assert weight == pytest.approx(0.25, rel=1e-12)


# -- the defining condition --------------------------------------------------


@pytest.mark.parametrize("size", [2, 4, 6, 9])
def test_equal_budgets_give_equal_risk_contributions(size: int) -> None:
    """The condition risk parity is defined by, checked on a general matrix.

    Not against a reference implementation — against the property itself, which
    is what the weights are supposed to have.
    """
    covariance = random_covariance(size, seed=size * 13)
    result = risk_parity(covariance)
    shares = result.allocation.percentage
    for share in shares:
        assert share == pytest.approx(1.0 / size, abs=1e-11)
    assert result.budget_error < 1e-11
    assert result.equal_risk


def test_risk_parity_weights_are_all_positive_and_sum_to_one() -> None:
    covariance = random_covariance(7, seed=99)
    result = risk_parity(covariance)
    assert all(weight > 0.0 for weight in result.weights)
    assert math.fsum(result.weights) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize(
    "budgets",
    [
        [0.5, 0.3, 0.2],
        [0.8, 0.1, 0.1],
        [1.0, 1.0, 8.0],
        [0.01, 0.01, 0.98],
    ],
)
def test_unequal_budgets_are_met(budgets: list[float]) -> None:
    covariance = random_covariance(3, seed=4)
    result = risk_parity(covariance, budgets=budgets)
    total = math.fsum(budgets)
    for share, budget in zip(result.allocation.percentage, budgets, strict=True):
        assert share == pytest.approx(budget / total, abs=1e-11)
    assert not result.equal_risk


def test_budgets_are_normalised_not_required_to_sum_to_one() -> None:
    covariance = random_covariance(4, seed=21)
    scaled = risk_parity(covariance, budgets=[2.0, 4.0, 6.0, 8.0])
    shares = risk_parity(covariance, budgets=[0.1, 0.2, 0.3, 0.4])
    for one, two in zip(scaled.weights, shares.weights, strict=True):
        assert one == pytest.approx(two, rel=1e-12)


def test_a_larger_budget_buys_a_larger_weight() -> None:
    covariance = random_covariance(4, seed=8)
    even = risk_parity(covariance)
    tilted = risk_parity(covariance, budgets=[0.7, 0.1, 0.1, 0.1])
    assert tilted.weights[0] > even.weights[0]


def test_risk_parity_is_not_inverse_volatility_in_general() -> None:
    """The two coincide under constant correlation and not otherwise.

    Worth a test of its own: inverse volatility is the standard cheap substitute
    for risk parity, and if this implementation returned it always, every
    closed-form test above would still pass.
    """
    covariance = random_covariance(5, seed=77)
    result = risk_parity(covariance)
    volatilities = [math.sqrt(covariance[i][i]) for i in range(5)]
    total = math.fsum(1.0 / one for one in volatilities)
    inverse = [(1.0 / one) / total for one in volatilities]
    assert max(
        abs(a - b) for a, b in zip(result.weights, inverse, strict=True)
    ) > 1e-4


def test_convergence_evidence_is_reported() -> None:
    result = risk_parity(random_covariance(5, seed=31))
    assert result.converged
    assert result.sweeps >= 1
    assert result.final_change <= 1e-14
    assert result.budget_error < 1e-11
    assert result.allocation.measure == "volatility"


def test_running_out_of_sweeps_raises_and_carries_the_evidence() -> None:
    """Stopping short is an error, not a quietly approximate answer.

    Weights that are nearly risk parity are indistinguishable from weights that
    are risk parity by looking at them, so the failure has to be loud. The
    partial result rides along on the exception so the caller can see how far it
    got.
    """
    covariance = random_covariance(6, seed=12)
    with pytest.raises(DidNotConverge) as raised:
        risk_parity(covariance, max_sweeps=2)
    assert raised.value.result.sweeps == 2
    assert not raised.value.result.converged
    assert raised.value.result.budget_error > 0.0


def test_names_are_carried_through_to_the_allocation() -> None:
    covariance = constant_correlation([0.1, 0.2, 0.3], 0.2)
    result = risk_parity(covariance, names=["gilts", "credit", "equity"])
    assert result.names == ("gilts", "credit", "equity")
    assert result.allocation.of("equity") == pytest.approx(
        result.allocation.total / 3.0, rel=1e-11
    )


def test_a_riskless_asset_is_refused() -> None:
    covariance = [[0.04, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError, match="variance of"):
        risk_parity(covariance)


def test_budgets_must_match_the_assets() -> None:
    with pytest.raises(ValueError, match="budgets against"):
        risk_parity(random_covariance(3, seed=1), budgets=[0.5, 0.5])


@pytest.mark.parametrize("bad", [0.0, -0.2])
def test_a_non_positive_budget_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="strictly"):
        risk_parity(random_covariance(3, seed=1), budgets=[0.5, 0.5, bad])


# -- diversification ratio ---------------------------------------------------


def test_diversification_ratio_of_uncorrelated_equal_assets_is_root_n() -> None:
    size, volatility = 6, 0.2
    covariance = [
        [volatility * volatility if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]
    ratio = diversification_ratio([1.0 / size] * size, covariance)
    assert ratio == pytest.approx(math.sqrt(size))


def test_diversification_ratio_of_perfectly_correlated_assets_is_one() -> None:
    size, volatility = 5, 0.2
    covariance = [[volatility * volatility] * size for _ in range(size)]
    assert diversification_ratio([1.0 / size] * size, covariance) == pytest.approx(1.0)


def test_diversification_ratio_of_a_single_asset_is_one() -> None:
    covariance = random_covariance(4, seed=6)
    assert diversification_ratio([1.0, 0.0, 0.0, 0.0], covariance) == pytest.approx(1.0)


def test_diversification_ratio_rises_as_correlation_falls() -> None:
    volatilities = [0.1, 0.2, 0.3, 0.15]
    weights = [0.25] * 4
    ratios = [
        diversification_ratio(weights, constant_correlation(volatilities, rho))
        for rho in (0.9, 0.5, 0.1, -0.1)
    ]
    assert ratios == sorted(ratios)
    assert all(ratio >= 1.0 for ratio in ratios)


def test_diversification_ratio_refuses_a_short_position() -> None:
    with pytest.raises(ValueError, match="long-only"):
        diversification_ratio([1.4, -0.4], constant_correlation([0.1, 0.2], 0.3))


def test_diversification_ratio_needs_volatility() -> None:
    with pytest.raises(NoRiskToAllocate):
        diversification_ratio([0.0, 0.0], constant_correlation([0.1, 0.2], 0.3))


# -- effective number of bets ------------------------------------------------


def test_equal_contributions_give_a_bet_per_position() -> None:
    size, volatility = 6, 0.2
    covariance = [
        [volatility * volatility if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]
    allocation = volatility_contributions([1.0 / size] * size, covariance)
    assert effective_bets(allocation) == pytest.approx(float(size))
    assert concentration(allocation) == pytest.approx(1.0 / size)


def test_one_position_carrying_everything_is_one_bet() -> None:
    covariance = random_covariance(4, seed=15)
    allocation = volatility_contributions([1.0, 0.0, 0.0, 0.0], covariance)
    assert effective_bets(allocation) == pytest.approx(1.0)
    assert concentration(allocation) == pytest.approx(1.0)


def test_effective_bets_never_exceeds_the_number_of_positions() -> None:
    covariance = random_covariance(7, seed=44)
    allocation = volatility_contributions([1.0 / 7] * 7, covariance)
    assert 1.0 <= effective_bets(allocation) <= 7.0


def test_risk_parity_maximises_bets_over_positions() -> None:
    """Equal risk shares are the entropy maximum, so parity is the most bets.

    A direct consequence of the definition rather than an empirical claim, and
    a cheap check that the two pieces agree with each other.
    """
    # A positive constant correlation, so that every long-only portfolio has
    # strictly positive contributions and the entropy is defined for all of
    # them. On a matrix with strong negative correlations even a mild tilt can
    # put a position into hedging the rest, and then there is no distribution to
    # take an entropy of and nothing to compare.
    covariance = constant_correlation([0.10, 0.15, 0.20, 0.25, 0.30], 0.35)
    parity = risk_parity(covariance)
    assert effective_bets(parity.allocation) == pytest.approx(5.0, abs=1e-10)
    for tilt in (
        [0.5, 0.15, 0.15, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.3, 0.3, 0.2, 0.1, 0.1],
    ):
        tilted = volatility_contributions(tilt, covariance)
        assert not tilted.has_negative_contribution
        assert effective_bets(tilted) < effective_bets(parity.allocation)


@pytest.mark.parametrize("seed", [2, 19, 37, 58])
def test_the_entropy_count_never_reports_fewer_bets_than_the_herfindahl_one(
    seed: int,
) -> None:
    """``exp(entropy) >= 1 / sum p^2``, always, which is why both are offered.

    The two are Renyi entropies of order one and two, and Renyi entropy does not
    increase in its order — so the exponential-entropy count is the more
    generous of the two for every portfolio that is not perfectly balanced, and
    the gap between them widens the more concentrated the portfolio is.

    That is the practical point. Quoting "effective number of bets" without
    saying which one is a free choice between a larger number and a smaller one
    for the same portfolio, and it is always available in the flattering
    direction.
    """
    size = 6
    covariance = random_covariance(size, seed=seed)
    parity = risk_parity(covariance)
    for weights in (
        list(parity.weights),
        [0.4, 0.15, 0.15, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.2, 0.15, 0.1, 0.05],
    ):
        allocation = volatility_contributions(weights, covariance)
        if allocation.has_negative_contribution:
            continue
        assert effective_bets(allocation) >= 1.0 / concentration(allocation) - 1e-12


def test_the_two_counts_agree_only_when_the_portfolio_is_balanced() -> None:
    """Equality in that inequality is exactly the uniform case.

    So a portfolio where the two counts coincide is balanced, and one where they
    differ is not — and the size of the gap is itself a concentration measure.
    """
    size = 8
    covariance = [[0.04 if i == j else 0.0 for j in range(size)] for i in range(size)]
    # For a diagonal matrix with equal variances the risk shares are the squared
    # weights normalised, so the shares can be set directly.
    balanced = volatility_contributions([math.sqrt(1.0 / size)] * size, covariance)
    assert effective_bets(balanced) == pytest.approx(1.0 / concentration(balanced))

    lopsided = volatility_contributions(
        [math.sqrt(x) for x in [0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]],
        covariance,
    )
    assert effective_bets(lopsided) > 1.0 / concentration(lopsided) + 0.1


def test_a_negative_contribution_has_no_entropy() -> None:
    covariance = [[0.04, -0.03], [-0.03, 0.04]]
    allocation = volatility_contributions([0.9, 0.35], covariance)
    assert allocation.has_negative_contribution
    with pytest.raises(NegativeContribution):
        effective_bets(allocation)
    with pytest.raises(NegativeContribution):
        concentration(allocation)


# -- principal components ----------------------------------------------------


def test_principal_bets_of_uncorrelated_equal_assets_is_the_count() -> None:
    size, volatility = 6, 0.2
    covariance = [
        [volatility * volatility if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]
    bets = principal_bets([1.0 / size] * size, covariance)
    assert bets.effective_bets == pytest.approx(float(size))
    assert bets.concentration == pytest.approx(1.0 / size)


def test_principal_bets_of_perfectly_correlated_assets_is_one() -> None:
    size, volatility = 5, 0.2
    covariance = [[volatility * volatility] * size for _ in range(size)]
    bets = principal_bets([1.0 / size] * size, covariance)
    assert bets.effective_bets == pytest.approx(1.0)
    assert bets.distribution[0] == pytest.approx(1.0)


def test_the_principal_distribution_is_a_distribution() -> None:
    covariance = random_covariance(6, seed=55)
    bets = principal_bets([0.4, -0.2, 0.3, 0.2, 0.2, 0.1], covariance)
    assert all(share >= 0.0 for share in bets.distribution)
    assert math.fsum(bets.distribution) == pytest.approx(1.0, abs=1e-12)


def test_principal_variances_come_back_largest_first() -> None:
    covariance = random_covariance(5, seed=23)
    bets = principal_bets([0.2] * 5, covariance)
    assert list(bets.variances) == sorted(bets.variances, reverse=True)


def test_principal_bets_counts_correlated_positions_as_fewer() -> None:
    """The reason the principal measure exists.

    Four positions, two pairs, each pair nearly the same asset. Spread evenly,
    the measure over positions reports close to four bets; the measure over
    principal components reports close to two, which is the truth.
    """
    volatilities = [0.2] * 4
    size = 4
    pairs = {(0, 1), (1, 0), (2, 3), (3, 2)}
    covariance = [
        [
            volatilities[i] * volatilities[j]
            * (1.0 if i == j else (0.995 if (i, j) in pairs else 0.0))
            for j in range(size)
        ]
        for i in range(size)
    ]
    weights = [0.25] * 4
    over_positions = effective_bets(volatility_contributions(weights, covariance))
    over_components = principal_bets(weights, covariance).effective_bets
    assert over_positions == pytest.approx(4.0, abs=1e-9)
    assert over_components < 2.05
    assert over_components > 1.95


def test_principal_bets_is_defined_where_the_position_measure_is_not() -> None:
    covariance = [[0.04, -0.03], [-0.03, 0.04]]
    weights = [0.9, 0.35]
    with pytest.raises(NegativeContribution):
        effective_bets(volatility_contributions(weights, covariance))
    bets = principal_bets(weights, covariance)
    assert 1.0 <= bets.effective_bets <= 2.0


def test_principal_bets_needs_variance() -> None:
    covariance = [[0.04, 0.04], [0.04, 0.04]]
    with pytest.raises(NoRiskToAllocate):
        principal_bets([1.0, -1.0], covariance)


def test_principal_bets_weights_must_match() -> None:
    with pytest.raises(ValueError, match="against a 3x3"):
        principal_bets([0.5, 0.5], random_covariance(3, seed=2))


def test_an_indefinite_covariance_is_refused() -> None:
    """A constant-correlation matrix below ``-1/(n-1)`` is not a covariance.

    The iteration would otherwise run to completion on it and return weights,
    because nothing in the coordinate update notices: the log barrier keeps them
    positive and the updates keep shrinking. The weights would be an answer to a
    problem that has none.
    """
    from shortfall.linalg import NotPositiveDefinite

    covariance = constant_correlation([0.1] * 8, -0.2)
    with pytest.raises(NotPositiveDefinite):
        risk_parity(covariance)

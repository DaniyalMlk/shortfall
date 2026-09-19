"""Factor exposures, the implied covariance, and risk attribution.

The regression is checked three ways, because a least squares routine that only
agrees with itself is worth nothing:

*Against a published worked example.* Anscombe's first dataset, whose fit —
intercept 3.00, slope 0.500, R-squared 0.667 — has been printed in statistics
texts since 1973 and is not something this code could have influenced.

*Against exact rational arithmetic.* The normal equations solved over
``fractions.Fraction`` have no rounding error at all, so the difference between
that answer and this one is exactly the floating-point error, and it can be
required to be tiny rather than merely plausible.

*Against the defining property.* Least squares residuals are orthogonal to
every column of the design. That holds for the true solution and for nothing
else, so it is a check on the answer rather than on the method.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from shortfall.contributions import volatility_contributions
from shortfall.factors import (
    Regression,
    attribute_risk,
    fit_factor_model,
    regress,
)
from shortfall.linalg import RankDeficient, least_squares, upper_inverse_gram
from shortfall.series import Panel, ReturnSeries, TooShort

#: Anscombe's first dataset. The famous quartet: four sets with the same fitted
#: line and wildly different shapes. This is the one that actually looks linear.
ANSCOMBE_X = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
ANSCOMBE_Y = [
    8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68,
]


def exact_coefficients(
    design: list[list[float]], target: list[float]
) -> list[float]:
    """Solve the normal equations over the rationals, with no rounding at all.

    Every input here is a float, so it is a rational number exactly, and
    ``Fraction`` carries the whole computation without error. The result is the
    true least squares solution of this exact problem — which makes the gap
    between it and the QR answer a measurement of the floating-point error and
    not an estimate of it.
    """
    rows = len(design)
    size = len(design[0])
    exact = [[Fraction(value) for value in row] for row in design]
    goal = [Fraction(value) for value in target]
    gram = [
        [sum((exact[t][i] * exact[t][j] for t in range(rows)), Fraction(0)) for j in range(size)]
        for i in range(size)
    ]
    rhs = [sum((exact[t][i] * goal[t] for t in range(rows)), Fraction(0)) for i in range(size)]
    augmented = [gram[i] + [rhs[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column] / augmented[column][column]
            augmented[row] = [
                augmented[row][k] - factor * augmented[column][k]
                for k in range(size + 1)
            ]
    return [float(augmented[i][size] / augmented[i][i]) for i in range(size)]


def factor_panel(seed: int = 42, periods: int = 500) -> Panel:
    rng = random.Random(seed)
    return Panel.from_columns(
        {
            "market": [rng.gauss(0.0004, 0.011) for _ in range(periods)],
            "value": [rng.gauss(0.0001, 0.006) for _ in range(periods)],
        }
    )


def built_panel(
    factors: Panel,
    truth: dict[str, tuple[float, float, float]],
    *,
    seed: int = 7,
) -> Panel:
    """Assets built to a known exposure and a known specific volatility."""
    rng = random.Random(seed)
    periods = factors.observations
    columns = {}
    for name, (market, value, specific) in truth.items():
        columns[name] = [
            market * factors["market"].values[t]
            + value * factors["value"].values[t]
            + rng.gauss(0.0, specific)
            for t in range(periods)
        ]
    return Panel.from_columns(columns)


TRUTH = {"A": (1.10, 0.30, 0.004), "B": (0.85, -0.45, 0.007), "C": (1.40, 0.05, 0.010)}


# -- least squares -----------------------------------------------------------


def test_least_squares_reproduces_anscombes_published_fit() -> None:
    design = [[1.0, x] for x in ANSCOMBE_X]
    coefficients, _ = least_squares(design, ANSCOMBE_Y)
    assert coefficients[0] == pytest.approx(3.00, abs=5e-3)
    assert coefficients[1] == pytest.approx(0.500, abs=5e-3)


def test_least_squares_matches_exact_rational_arithmetic() -> None:
    rng = random.Random(3)
    design = [[1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(80)]
    target = [
        2.0 + 0.5 * row[1] - 1.25 * row[2] + rng.gauss(0.0, 0.4) for row in design
    ]
    coefficients, _ = least_squares(design, target)
    exact = exact_coefficients(design, target)
    for computed, true in zip(coefficients, exact, strict=True):
        assert computed == pytest.approx(true, abs=1e-12)


def test_residuals_are_orthogonal_to_every_column_of_the_design() -> None:
    """The property that defines the least squares solution.

    If the residual had any component along a column, moving that coefficient
    would reduce the sum of squares, so the fit would not be least squares. It
    holds for the right answer and for no other.
    """
    rng = random.Random(11)
    design = [
        [1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)]
        for _ in range(60)
    ]
    target = [rng.gauss(0.0, 1.0) for _ in range(60)]
    coefficients, _ = least_squares(design, target)
    scale = max(abs(value) for value in target)
    for column in range(4):
        projection = math.fsum(
            design[t][column]
            * (
                target[t]
                - math.fsum(
                    design[t][k] * coefficients[k] for k in range(4)
                )
            )
            for t in range(60)
        )
        assert abs(projection) < 1e-12 * scale * 60


def test_an_exact_relationship_is_recovered_exactly() -> None:
    rng = random.Random(5)
    design = [[1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(40)]
    target = [3.0 + 2.0 * row[1] - 1.5 * row[2] for row in design]
    coefficients, _ = least_squares(design, target)
    assert coefficients == pytest.approx([3.0, 2.0, -1.5], abs=1e-12)


def test_qr_beats_the_normal_equations_on_an_ill_conditioned_design() -> None:
    """The Lauchli matrix, where forming ``X'X`` loses the problem entirely.

    With ``eps`` below the square root of the machine epsilon, ``1 + eps^2``
    rounds to exactly ``1``, so ``X'X`` comes back singular and the normal
    equations have no answer at all — while the design itself is perfectly well
    determined and QR solves it to full precision. This is the case that
    justifies the extra factor of two in arithmetic, so it is measured rather
    than asserted.
    """
    eps = 1e-9
    design = [[1.0, 1.0], [eps, 0.0], [0.0, eps]]
    target = [2.0, eps, eps]
    coefficients, _ = least_squares(design, target)
    assert coefficients == pytest.approx([1.0, 1.0], abs=1e-9)

    # What the normal equations would have produced, computed in floating point
    # exactly as a Cholesky-based solver would form it.
    gram = [
        [math.fsum(design[t][i] * design[t][j] for t in range(3)) for j in range(2)]
        for i in range(2)
    ]
    assert gram[0][0] == 1.0  # 1 + eps^2 has rounded away
    assert gram[0][1] == 1.0
    determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    assert determinant == 0.0  # singular, so there is nothing to solve


def test_dependent_columns_are_refused() -> None:
    with pytest.raises(RankDeficient, match="linear combination"):
        least_squares([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])


def test_a_duplicated_factor_is_refused() -> None:
    rng = random.Random(9)
    design = []
    for _ in range(50):
        first = rng.gauss(0.0, 1.0)
        design.append([1.0, first, first])
    with pytest.raises(RankDeficient):
        least_squares(design, [rng.gauss(0.0, 1.0) for _ in range(50)])


def test_more_unknowns_than_equations_is_refused() -> None:
    with pytest.raises(RankDeficient, match="more unknowns than equations"):
        least_squares([[1.0, 2.0, 3.0]], [1.0])


def test_a_zero_design_is_refused() -> None:
    with pytest.raises(RankDeficient, match="entirely zero"):
        least_squares([[0.0], [0.0], [0.0]], [1.0, 2.0, 3.0])


def test_targets_must_match_the_design() -> None:
    with pytest.raises(ValueError, match="targets against"):
        least_squares([[1.0], [1.0]], [1.0])


def test_a_ragged_design_is_refused() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        least_squares([[1.0, 2.0], [1.0]], [1.0, 2.0])


def test_the_gram_inverse_is_the_inverse_of_the_gram_matrix() -> None:
    rng = random.Random(17)
    design = [[1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(45)]
    _, upper = least_squares(design, [rng.gauss(0.0, 1.0) for _ in range(45)])
    inverse = upper_inverse_gram(upper)
    gram = [
        [math.fsum(design[t][i] * design[t][j] for t in range(45)) for j in range(3)]
        for i in range(3)
    ]
    for i in range(3):
        for j in range(3):
            entry = math.fsum(inverse[i][k] * gram[k][j] for k in range(3))
            assert entry == pytest.approx(1.0 if i == j else 0.0, abs=1e-12)


# -- regression diagnostics --------------------------------------------------


def test_regression_reproduces_anscombes_fit_and_r_squared() -> None:
    factors = Panel.from_columns({"x": ANSCOMBE_X})
    asset = ReturnSeries(name="y", values=tuple(ANSCOMBE_Y))
    fit = regress(asset, factors)
    assert fit.alpha == pytest.approx(3.00, abs=5e-3)
    assert fit.beta("x") == pytest.approx(0.500, abs=5e-3)
    assert fit.r_squared == pytest.approx(0.667, abs=5e-3)
    assert fit.observations == 11
    assert fit.parameters == 2
    assert fit.degrees_of_freedom == 9


def test_simple_regression_matches_its_closed_form() -> None:
    """One factor has a closed form, so the general routine can be checked on it."""
    factors = factor_panel(periods=200)
    market = factors["market"]
    one = Panel(tuple([market]))
    asset = built_panel(factors, {"A": (1.2, 0.0, 0.005)})["A"]
    fit = regress(asset, one)

    count = len(asset)
    mean_x = math.fsum(market.values) / count
    mean_y = math.fsum(asset.values) / count
    covariance = math.fsum(
        (market.values[t] - mean_x) * (asset.values[t] - mean_y) for t in range(count)
    )
    variance = math.fsum((market.values[t] - mean_x) ** 2 for t in range(count))
    assert fit.betas[0] == pytest.approx(covariance / variance, rel=1e-12)
    assert fit.alpha == pytest.approx(
        mean_y - (covariance / variance) * mean_x, rel=1e-10
    )


def test_r_squared_is_the_squared_correlation_of_fitted_and_actual() -> None:
    factors = factor_panel(periods=300)
    asset = built_panel(factors, {"A": (0.9, 0.4, 0.008)})["A"]
    fit = regress(asset, factors)
    fitted = [
        fit.fitted(factors.row(t)) for t in range(factors.observations)
    ]
    count = len(fitted)
    mean_f = math.fsum(fitted) / count
    mean_a = math.fsum(asset.values) / count
    top = math.fsum(
        (fitted[t] - mean_f) * (asset.values[t] - mean_a) for t in range(count)
    )
    bottom = math.sqrt(
        math.fsum((fitted[t] - mean_f) ** 2 for t in range(count))
        * math.fsum((asset.values[t] - mean_a) ** 2 for t in range(count))
    )
    assert fit.r_squared == pytest.approx((top / bottom) ** 2, rel=1e-10)


def test_known_exposures_are_recovered() -> None:
    factors = factor_panel(periods=4000)
    panel = built_panel(factors, TRUTH, seed=2)
    model = fit_factor_model(panel, factors)
    for name, (market, value, specific) in TRUTH.items():
        fit = model.regression(name)
        assert fit.beta("market") == pytest.approx(market, abs=0.03)
        assert fit.beta("value") == pytest.approx(value, abs=0.06)
        assert fit.specific_volatility == pytest.approx(specific, rel=0.08)


def test_an_exposure_that_is_really_there_is_significant() -> None:
    factors = factor_panel(periods=1000)
    panel = built_panel(factors, TRUTH, seed=4)
    model = fit_factor_model(panel, factors)
    fit = model.regression("A")
    # intercept, market, value
    assert abs(fit.t_statistics[1]) > 10.0
    assert abs(fit.t_statistics[2]) > 5.0
    # There is no alpha in the data, so the intercept should not look real.
    assert abs(fit.t_statistics[0]) < 3.0


def noisy_pair(
    seed: int, *, extra: int = 20, periods: int = 120
) -> tuple[Regression, Regression]:
    """The same asset fitted with and without ``extra`` factors of pure noise."""
    # A separate stream from the one behind the factors. Seeding both the same
    # way reproduces the factor columns exactly inside the "noise", which makes
    # the padded design rank deficient rather than merely wasteful.
    rng = random.Random(seed * 7919 + 13)
    factors = factor_panel(seed=seed, periods=periods)
    asset = built_panel(factors, {"A": (1.0, 0.2, 0.01)}, seed=seed + 1)["A"]
    padded = Panel.from_columns(
        {
            "market": list(factors["market"].values),
            "value": list(factors["value"].values),
            **{
                f"noise {k}": [rng.gauss(0.0, 0.01) for _ in range(periods)]
                for k in range(extra)
            },
        }
    )
    return regress(asset, factors), regress(asset, padded)


@pytest.mark.parametrize("seed", [23, 24, 25, 26, 27])
def test_adding_factors_never_lowers_raw_r_squared(seed: int) -> None:
    """Which is the flaw the adjusted figure exists to correct.

    Deterministic: the larger design contains the smaller one, so its best fit
    cannot be worse. Selecting a model on raw R-squared therefore always picks
    the largest one on offer.
    """
    lean, fat = noisy_pair(seed)
    assert fat.r_squared >= lean.r_squared


@pytest.mark.parametrize("seed", [23, 24, 25, 26, 27])
def test_adjusted_r_squared_never_exceeds_the_raw_figure(seed: int) -> None:
    lean, fat = noisy_pair(seed)
    assert lean.adjusted_r_squared <= lean.r_squared
    assert fat.adjusted_r_squared <= fat.r_squared
    # And the penalty is heavier for the model that spent more parameters.
    assert (fat.r_squared - fat.adjusted_r_squared) > (
        lean.r_squared - lean.adjusted_r_squared
    )


def test_adjusted_r_squared_estimates_the_population_figure_either_way() -> None:
    """The property that actually holds, which is stronger and less often stated.

    The intuitive claim — that adding useless factors lowers adjusted
    R-squared — is not right, and measuring it says so: over forty independent
    samples the padded model's mean adjusted R-squared came out marginally
    *above* the lean model's. What is true is better than that. Adjusted
    R-squared is approximately unbiased for the population R-squared whether or
    not useless factors are present, so both models recover the same population
    number while the raw figure of the padded model is inflated well past it.

    The population value follows from the construction: the signal variance is
    ``1.0^2 * 0.011^2 + 0.2^2 * 0.006^2`` against a specific variance of
    ``0.01^2``, giving 0.5504.
    """
    signal = 1.0**2 * 0.011**2 + 0.2**2 * 0.006**2
    population = signal / (signal + 0.01**2)
    assert population == pytest.approx(0.5504, abs=1e-4)

    pairs = [noisy_pair(seed) for seed in range(200, 240)]
    lean_adjusted = math.fsum(one.adjusted_r_squared for one, _ in pairs) / len(pairs)
    fat_adjusted = math.fsum(two.adjusted_r_squared for _, two in pairs) / len(pairs)
    assert lean_adjusted == pytest.approx(population, abs=0.03)
    assert fat_adjusted == pytest.approx(population, abs=0.03)

    # The raw figure of the padded model is not an estimate of anything: twenty
    # noise factors on 120 observations buy about 20/120 of the remaining
    # unexplained variance for nothing.
    raw_fat = math.fsum(two.r_squared for _, two in pairs) / len(pairs)
    assert raw_fat > population + 0.05


def test_adjusted_r_squared_goes_negative_on_pure_noise() -> None:
    rng = random.Random(31)
    periods = 40
    noise = Panel.from_columns(
        {f"noise {k}": [rng.gauss(0.0, 0.01) for _ in range(periods)] for k in range(15)}
    )
    asset = ReturnSeries(
        name="A", values=tuple(rng.gauss(0.0, 0.01) for _ in range(periods))
    )
    fit = regress(asset, noise)
    assert fit.r_squared > 0.0
    assert fit.adjusted_r_squared < 0.0


def test_durbin_watson_is_near_two_for_independent_residuals() -> None:
    factors = factor_panel(periods=2000)
    asset = built_panel(factors, {"A": (1.0, 0.3, 0.01)}, seed=13)["A"]
    fit = regress(asset, factors)
    assert fit.durbin_watson == pytest.approx(2.0, abs=0.12)


def test_durbin_watson_detects_positive_serial_correlation() -> None:
    """Residuals built to trend, which ordinary least squares cannot see.

    The exposures come back fine — least squares is still unbiased — while the
    standard errors do not, so the statistic that catches this is the one worth
    reporting alongside them.
    """
    factors = factor_panel(periods=1000)
    rng = random.Random(29)
    drift = 0.0
    values = []
    for t in range(1000):
        drift = 0.92 * drift + rng.gauss(0.0, 0.004)
        values.append(1.0 * factors["market"].values[t] + drift)
    asset = ReturnSeries(name="A", values=tuple(values))
    fit = regress(asset, factors)
    assert fit.durbin_watson < 1.0
    assert fit.beta("market") == pytest.approx(1.0, abs=0.05)


def test_dropping_the_intercept_forces_the_fit_through_the_origin() -> None:
    factors = factor_panel(periods=200)
    asset = built_panel(factors, {"A": (1.0, 0.2, 0.01)})["A"]
    fit = regress(asset, factors, intercept=False)
    assert fit.alpha == 0.0
    assert fit.parameters == 2
    assert fit.degrees_of_freedom == 198


def test_a_regression_needs_more_observations_than_coefficients() -> None:
    factors = factor_panel(periods=3)
    asset = ReturnSeries(name="A", values=(0.01, -0.02, 0.005))
    with pytest.raises(TooShort, match="cannot estimate"):
        regress(asset, factors)


def test_asset_and_factors_must_be_aligned() -> None:
    factors = factor_panel(periods=100)
    asset = ReturnSeries(name="A", values=tuple([0.01] * 90))
    with pytest.raises(ValueError, match="a regression needs them aligned"):
        regress(asset, factors)


def test_an_unknown_factor_is_a_key_error() -> None:
    factors = factor_panel(periods=100)
    fit = regress(built_panel(factors, {"A": (1.0, 0.0, 0.01)})["A"], factors)
    with pytest.raises(KeyError):
        fit.beta("momentum")


def test_fitted_needs_one_return_per_factor() -> None:
    factors = factor_panel(periods=100)
    fit = regress(built_panel(factors, {"A": (1.0, 0.0, 0.01)})["A"], factors)
    with pytest.raises(ValueError, match="against 2 exposures"):
        fit.fitted([0.01])


# -- the implied covariance --------------------------------------------------


def test_implied_covariance_off_diagonal_is_exposures_through_the_factors() -> None:
    """``B F B'`` off the diagonal, with the specific variance only on it.

    Computed here the slow explicit way, which is a different route through the
    same algebra to the blocked one the model uses.
    """
    factors = factor_panel(periods=300)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    implied = model.implied_covariance()
    for i in range(model.assets):
        for j in range(model.assets):
            through = math.fsum(
                model.exposures[i][k]
                * model.factor_covariance[k][m]
                * model.exposures[j][m]
                for k in range(model.factors)
                for m in range(model.factors)
            )
            expected = through + (model.specific_variances[i] if i == j else 0.0)
            assert implied[i][j] == pytest.approx(expected, rel=1e-11, abs=1e-18)


def test_implied_covariance_is_symmetric_and_positive_definite() -> None:
    from shortfall.linalg import cholesky

    factors = factor_panel(periods=300)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    implied = model.implied_covariance()
    for i in range(model.assets):
        for j in range(model.assets):
            assert implied[i][j] == implied[j][i]
    cholesky(implied)


def test_implied_covariance_is_full_rank_with_fewer_periods_than_assets() -> None:
    """The reason factor models are used on a large universe at all.

    Twelve assets over eight periods: the sample covariance is necessarily
    singular, and the factor-implied one is not, because it is a small matrix
    plus a strictly positive diagonal.
    """
    from shortfall.linalg import cholesky

    rng = random.Random(61)
    periods = 8
    factors = Panel.from_columns(
        {"market": [rng.gauss(0.0, 0.01) for _ in range(periods)]}
    )
    panel = Panel.from_columns(
        {
            f"asset {i}": [
                rng.gauss(0.0, 0.5) * factors["market"].values[t] + rng.gauss(0.0, 0.01)
                for t in range(periods)
            ]
            for i in range(12)
        }
    )
    model = fit_factor_model(panel, factors)
    assert model.assets == 12
    cholesky(model.implied_covariance())


def test_a_single_factor_implies_a_one_factor_covariance() -> None:
    periods = 400
    rng = random.Random(71)
    factors = Panel.from_columns(
        {"market": [rng.gauss(0.0, 0.012) for _ in range(periods)]}
    )
    panel = Panel.from_columns(
        {
            name: [
                beta * factors["market"].values[t] + rng.gauss(0.0, 0.006)
                for t in range(periods)
            ]
            for name, beta in (("A", 1.2), ("B", 0.7), ("C", -0.4))
        }
    )
    model = fit_factor_model(panel, factors)
    implied = model.implied_covariance()
    variance = model.factor_covariance[0][0]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            assert implied[i][j] == pytest.approx(
                model.exposures[i][0] * model.exposures[j][0] * variance, rel=1e-11
            )


def test_residual_correlation_is_reported_not_assumed_away() -> None:
    """An injected common residual shows up in the diagnostic.

    Two assets given a shared shock the factors cannot see. The model still
    fits, the R-squared values are unremarkable, and the one number that reveals
    the problem is the residual correlation.
    """
    periods = 600
    rng = random.Random(83)
    factors = Panel.from_columns(
        {"market": [rng.gauss(0.0, 0.011) for _ in range(periods)]}
    )
    shared = [rng.gauss(0.0, 0.009) for _ in range(periods)]
    panel = Panel.from_columns(
        {
            "clean": [
                1.0 * factors["market"].values[t] + rng.gauss(0.0, 0.009)
                for t in range(periods)
            ],
            "paired one": [
                0.9 * factors["market"].values[t] + shared[t] + rng.gauss(0.0, 0.003)
                for t in range(periods)
            ],
            "paired two": [
                1.1 * factors["market"].values[t] + shared[t] + rng.gauss(0.0, 0.003)
                for t in range(periods)
            ],
        }
    )
    model = fit_factor_model(panel, factors)
    first, second, worst = model.worst_residual_correlation()
    assert {first, second} == {"paired one", "paired two"}
    assert worst > 0.8

    correlations = model.residual_correlations()
    for i in range(3):
        assert correlations[i][i] == pytest.approx(1.0)


def test_a_clean_model_has_small_residual_correlations() -> None:
    factors = factor_panel(periods=1500)
    model = fit_factor_model(built_panel(factors, TRUTH, seed=19), factors)
    _, _, worst = model.worst_residual_correlation()
    assert abs(worst) < 0.12


def test_assets_and_factors_must_be_aligned_for_a_model() -> None:
    factors = factor_panel(periods=100)
    panel = built_panel(factor_panel(periods=90), {"A": (1.0, 0.0, 0.01)})
    with pytest.raises(ValueError, match="must be aligned"):
        fit_factor_model(panel, factors)


# -- attribution -------------------------------------------------------------


WEIGHTS = [0.5, 0.3, 0.2]


def test_attribution_reconciles_to_the_total() -> None:
    factors = factor_panel(periods=500)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    attribution = attribute_risk(WEIGHTS, model)
    assert attribution.identity_error < 1e-14 * attribution.total
    assert attribution.sum_of_parts == pytest.approx(attribution.total, rel=1e-14)


def test_attribution_total_is_the_implied_covariance_volatility() -> None:
    """The cross-check between the two routes to the same number.

    One goes through the factor representation and one through the assembled
    covariance matrix and the Euler allocation in ``contributions``. They share
    no code path, and a disagreement would mean the factor decomposition is not
    decomposing the risk the model actually implies.
    """
    factors = factor_panel(periods=500)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    attribution = attribute_risk(WEIGHTS, model)
    allocation = volatility_contributions(WEIGHTS, model.implied_covariance())
    assert attribution.total == pytest.approx(allocation.total, rel=1e-12)


def test_variance_splits_between_factors_and_residuals_with_nothing_left_over() -> None:
    factors = factor_panel(periods=400)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    attribution = attribute_risk(WEIGHTS, model)
    assert (
        attribution.factor_volatility**2 + attribution.specific_volatility**2
    ) == pytest.approx(attribution.total**2, rel=1e-13)


def test_standalone_volatilities_do_not_add_but_contributions_do() -> None:
    """The distinction the result object exists to keep straight."""
    factors = factor_panel(periods=400)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    attribution = attribute_risk(WEIGHTS, model)
    naive = attribution.factor_volatility + attribution.specific_volatility
    assert naive > attribution.total
    assert (
        attribution.factor_contribution + attribution.specific_contribution
    ) == pytest.approx(attribution.total, rel=1e-14)


def test_per_factor_components_sum_to_the_factor_contribution() -> None:
    factors = factor_panel(periods=400)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    attribution = attribute_risk(WEIGHTS, model)
    assert math.fsum(attribution.components) == pytest.approx(
        attribution.factor_contribution, rel=1e-13
    )


def test_portfolio_exposures_are_the_weighted_asset_exposures() -> None:
    factors = factor_panel(periods=300)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    attribution = attribute_risk(WEIGHTS, model)
    for k, name in enumerate(model.factor_names):
        expected = math.fsum(
            WEIGHTS[i] * model.regression(model.asset_names[i]).beta(name)
            for i in range(model.assets)
        )
        assert attribution.exposures[k] == pytest.approx(expected, rel=1e-12)


def test_a_market_dominated_portfolio_attributes_its_risk_to_the_market() -> None:
    factors = factor_panel(periods=800)
    model = fit_factor_model(built_panel(factors, TRUTH, seed=27), factors)
    attribution = attribute_risk(WEIGHTS, model)
    name, component = attribution.largest()
    assert name == "market"
    assert component > 0.0
    assert attribution.of("market") == component
    assert 0.0 < attribution.factor_share < 1.0
    assert attribution.factor_share > 0.8


def test_a_portfolio_of_pure_specific_risk_attributes_none_to_the_factors() -> None:
    """Weights chosen to have zero net exposure to every factor.

    Two assets with the same exposures, held long and short in equal size. The
    factor part of the risk vanishes exactly and everything that remains is
    specific — which the identity still has to reconcile.
    """
    periods = 400
    rng = random.Random(97)
    factors = Panel.from_columns(
        {"market": [rng.gauss(0.0, 0.011) for _ in range(periods)]}
    )
    shared_beta = 1.0
    panel = Panel.from_columns(
        {
            name: [
                shared_beta * factors["market"].values[t] + rng.gauss(0.0, 0.008)
                for t in range(periods)
            ]
            for name in ("A", "B")
        }
    )
    model = fit_factor_model(panel, factors)
    # Solve for the long-short pair with exactly no net exposure.
    first, second = model.exposures[0][0], model.exposures[1][0]
    weights = [1.0, -first / second]
    attribution = attribute_risk(weights, model)
    assert attribution.exposures[0] == pytest.approx(0.0, abs=1e-15)
    assert attribution.factor_volatility == pytest.approx(0.0, abs=1e-15)
    assert attribution.factor_share == pytest.approx(0.0, abs=1e-25)
    assert attribution.specific_contribution == pytest.approx(
        attribution.total, rel=1e-14
    )
    assert attribution.identity_error < 1e-14 * attribution.total


def test_a_factor_contribution_can_be_negative() -> None:
    """A short exposure to a factor correlated with the long ones removes risk.

    Constructed with two positively correlated factors and a portfolio long the
    first and short the second, so the second's contribution has to come out
    below zero. The identity has to hold through it.
    """
    periods = 600
    rng = random.Random(53)
    first = [rng.gauss(0.0, 0.012) for _ in range(periods)]
    second = [0.8 * first[t] + rng.gauss(0.0, 0.007) for t in range(periods)]
    factors = Panel.from_columns({"first": first, "second": second})
    panel = Panel.from_columns(
        {
            "long": [1.0 * first[t] + rng.gauss(0.0, 0.004) for t in range(periods)],
            "short": [1.0 * second[t] + rng.gauss(0.0, 0.004) for t in range(periods)],
        }
    )
    model = fit_factor_model(panel, factors)
    attribution = attribute_risk([1.0, -0.6], model)
    assert attribution.of("second") < 0.0
    assert attribution.identity_error < 1e-14 * attribution.total


def test_attribution_weights_must_match_the_model() -> None:
    factors = factor_panel(periods=200)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    with pytest.raises(ValueError, match="weights against 3 assets"):
        attribute_risk([0.5, 0.5], model)


def test_an_unknown_factor_name_is_a_key_error() -> None:
    factors = factor_panel(periods=200)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    attribution = attribute_risk(WEIGHTS, model)
    with pytest.raises(KeyError):
        attribution.of("momentum")
    with pytest.raises(KeyError):
        model.regression("D")


def test_attribution_is_homogeneous_in_the_weights() -> None:
    factors = factor_panel(periods=300)
    model = fit_factor_model(built_panel(factors, TRUTH), factors)
    single = attribute_risk(WEIGHTS, model)
    double = attribute_risk([2.0 * w for w in WEIGHTS], model)
    assert double.total == pytest.approx(2.0 * single.total, rel=1e-13)
    for one, two in zip(single.components, double.components, strict=True):
        assert two == pytest.approx(2.0 * one, rel=1e-13)
    assert double.factor_share == pytest.approx(single.factor_share, rel=1e-13)

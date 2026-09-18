"""Covariance estimation and Ledoit-Wolf shrinkage.

The shrinkage intensity has no closed form to check against, so it is checked
against what the theory predicts it will do. The optimum trades the noise in the
sample against the bias in the target, so:

- data whose true structure *is* constant correlation should shrink hard,
  because the target costs almost no bias;
- data with block structure, which the target cannot represent, should shrink
  little when there is enough of it to estimate the blocks;
- the same block-structured data observed for a tenth as long should shrink
  more, because the sample got noisier while the bias stayed the same.

Those are three independent predictions with the wrong answer in three different
directions, and an implementation that got the ``rho`` term wrong — the term a
naive derivation drops — fails at least one of them.

Everything else here is an identity: the diagonal is preserved exactly at every
intensity, the blend is linear in the intensity, and a singular sample becomes
factorisable.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.covariance import (
    average_correlation,
    blend,
    constant_correlation_target,
    correlation,
    diagnose,
    ledoit_wolf,
    portfolio_variance,
    sample_covariance,
)
from shortfall.linalg import NotPositiveDefinite, cholesky, condition_number
from shortfall.series import Panel, TooShort


def one_factor(assets: int, periods: int, *, seed: int, noise: float = 0.008) -> Panel:
    """Assets driven by one common factor: true structure near constant correlation."""
    rng = random.Random(seed)
    factor = [rng.gauss(0.0, 0.01) for _ in range(periods)]
    return Panel.from_columns(
        {
            f"a{i}": [factor[t] + rng.gauss(0.0, noise) for t in range(periods)]
            for i in range(assets)
        }
    )


def two_blocks(per_block: int, periods: int, *, seed: int) -> Panel:
    """Two uncorrelated groups: structure the constant-correlation target cannot hold."""
    rng = random.Random(seed)
    first = [rng.gauss(0.0, 0.01) for _ in range(periods)]
    second = [rng.gauss(0.0, 0.01) for _ in range(periods)]
    columns: dict[str, list[float]] = {}
    for i in range(per_block):
        columns[f"x{i}"] = [first[t] + rng.gauss(0.0, 0.003) for t in range(periods)]
        columns[f"y{i}"] = [second[t] + rng.gauss(0.0, 0.003) for t in range(periods)]
    return Panel.from_columns(columns)


# -- sample covariance -------------------------------------------------------


def test_the_covariance_of_one_asset_with_itself_is_its_variance() -> None:
    panel = Panel.from_columns({"a": [0.01, -0.02, 0.03, 0.00]})
    assert sample_covariance(panel)[0][0] == pytest.approx(panel["a"].variance(), abs=1e-18)


def test_the_covariance_matrix_is_symmetric() -> None:
    matrix = sample_covariance(one_factor(5, 30, seed=1))
    for i in range(5):
        for j in range(5):
            assert matrix[i][j] == matrix[j][i]


def test_a_hand_computed_covariance() -> None:
    # a = (1, 2, 3, 4), b = (2, 4, 5, 9). Means 2.5 and 5. Deviations
    # (-1.5, -0.5, 0.5, 1.5) and (-3, -1, 0, 4). Cross products sum to
    # 4.5 + 0.5 + 0 + 6 = 11, over 3 degrees of freedom.
    panel = Panel.from_columns({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 5.0, 9.0]})
    matrix = sample_covariance(panel)
    assert matrix[0][1] == pytest.approx(11.0 / 3.0, abs=1e-14)
    assert matrix[0][0] == pytest.approx(5.0 / 3.0, abs=1e-14)
    assert matrix[1][1] == pytest.approx(26.0 / 3.0, abs=1e-14)


def test_the_two_degrees_of_freedom_conventions_differ_by_the_expected_factor() -> None:
    panel = one_factor(4, 25, seed=2)
    unbiased = sample_covariance(panel, ddof=1)
    likelihood = sample_covariance(panel, ddof=0)
    for i in range(4):
        for j in range(4):
            assert likelihood[i][j] == pytest.approx(unbiased[i][j] * 24 / 25, abs=1e-18)


def test_a_single_observation_has_no_unbiased_covariance() -> None:
    with pytest.raises(TooShort, match="ddof=1"):
        sample_covariance(Panel.from_columns({"a": [0.01], "b": [0.02]}))


def test_the_variance_of_a_portfolio_matches_the_series_it_describes() -> None:
    # The quadratic form and the realised portfolio series are computed by
    # entirely separate routes, so agreement is evidence the matrix is right.
    panel = one_factor(4, 120, seed=11)
    weights = [0.4, 0.3, -0.2, 0.5]
    through_matrix = portfolio_variance(weights, sample_covariance(panel))
    through_series = panel.portfolio(weights).variance()
    assert through_matrix == pytest.approx(through_series, abs=1e-18)


# -- correlation and the target ----------------------------------------------


def test_correlation_has_a_unit_diagonal_and_lives_in_minus_one_to_one() -> None:
    matrix = correlation(sample_covariance(one_factor(6, 50, seed=4)))
    for i in range(6):
        assert matrix[i][i] == 1.0
        for j in range(6):
            assert -1.0 <= matrix[i][j] <= 1.0


def test_a_perfectly_correlated_pair_has_a_correlation_of_one() -> None:
    panel = Panel.from_columns({"a": [0.01, 0.02, 0.03], "b": [0.02, 0.04, 0.06]})
    assert correlation(sample_covariance(panel))[0][1] == pytest.approx(1.0, abs=1e-14)


def test_a_dead_series_correlates_with_nothing_rather_than_producing_a_nan() -> None:
    # 0/0. A nan here would propagate through the average correlation and turn
    # one flat series into a matrix of nothing.
    panel = Panel.from_columns({"a": [0.01, -0.02, 0.03], "b": [0.0, 0.0, 0.0]})
    matrix = correlation(sample_covariance(panel))
    assert matrix[0][1] == 0.0
    assert not math.isnan(average_correlation(sample_covariance(panel)))


def test_the_target_keeps_the_variances_and_averages_the_correlations() -> None:
    sample = sample_covariance(one_factor(5, 40, seed=6))
    target = constant_correlation_target(sample)
    mean = average_correlation(sample)
    for i in range(5):
        assert target[i][i] == pytest.approx(sample[i][i], abs=1e-20)
        for j in range(i + 1, 5):
            implied = target[i][j] / math.sqrt(target[i][i] * target[j][j])
            assert implied == pytest.approx(mean, abs=1e-14)


def test_the_target_of_a_constant_correlation_matrix_is_itself() -> None:
    sample = [[4.0, 1.2, 1.8], [1.2, 1.0, 0.9], [1.8, 0.9, 9.0]]
    # Correlations 0.6, 0.3, 0.3 — not constant, so the target differs.
    assert constant_correlation_target(sample) != sample
    equal = [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]
    rebuilt = constant_correlation_target(equal)
    for i in range(3):
        for j in range(3):
            assert rebuilt[i][j] == pytest.approx(equal[i][j], abs=1e-15)


# -- blending ----------------------------------------------------------------


def test_blending_is_linear_in_the_intensity() -> None:
    sample = sample_covariance(one_factor(4, 30, seed=8))
    target = constant_correlation_target(sample)
    half = blend(sample, target, 0.5)
    for i in range(4):
        for j in range(4):
            assert half[i][j] == pytest.approx((sample[i][j] + target[i][j]) / 2.0, abs=1e-18)


@pytest.mark.parametrize("intensity", [0.0, 1.0])
def test_the_endpoints_of_a_blend_are_the_inputs(intensity: float) -> None:
    sample = sample_covariance(one_factor(4, 30, seed=9))
    target = constant_correlation_target(sample)
    wanted = target if intensity == 1.0 else sample
    got = blend(sample, target, intensity)
    for i in range(4):
        for j in range(4):
            assert got[i][j] == pytest.approx(wanted[i][j], abs=1e-18)


@pytest.mark.parametrize("intensity", [-0.01, 1.01])
def test_an_intensity_outside_the_unit_interval_is_refused(intensity: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        blend([[1.0]], [[1.0]], intensity)


# -- Ledoit-Wolf: what the theory predicts -----------------------------------


def test_data_the_target_describes_well_is_shrunk_hard() -> None:
    # One common factor and equal idiosyncratic noise is constant correlation,
    # so the target costs almost no bias and the optimum should be near one.
    result = ledoit_wolf(one_factor(8, 250, seed=21))
    assert result.intensity > 0.5


def test_data_the_target_cannot_describe_is_shrunk_little() -> None:
    # Two uncorrelated blocks. A single average correlation cannot represent
    # them, so the bias is large and, with plenty of data, shrinkage is small.
    result = ledoit_wolf(two_blocks(5, 400, seed=22))
    assert result.intensity < 0.1


def test_the_same_structure_observed_for_less_time_is_shrunk_more() -> None:
    # The sample got noisier and the target is exactly as wrong as before, so
    # the optimum has to move up. This is the prediction that fails if the rho
    # term — the one a naive derivation drops — is missing or mis-signed.
    plenty = ledoit_wolf(two_blocks(5, 400, seed=23))
    scarce = ledoit_wolf(two_blocks(5, 40, seed=23))
    assert scarce.intensity > plenty.intensity


def test_the_intensity_is_always_in_the_unit_interval() -> None:
    for seed in range(12):
        result = ledoit_wolf(one_factor(6, 20, seed=seed))
        assert 0.0 <= result.intensity <= 1.0


def test_the_unclamped_optimum_is_reported_even_when_it_is_out_of_range() -> None:
    # Seeing that the estimate is being driven by the clamp rather than by the
    # data is the point of reporting it.
    result = ledoit_wolf(one_factor(8, 250, seed=21))
    assert result.unclamped_intensity >= result.intensity
    assert result.clamped == (result.unclamped_intensity != result.intensity)


# -- Ledoit-Wolf: identities -------------------------------------------------


def test_shrinkage_preserves_the_variances_exactly() -> None:
    # Sample and target share a diagonal, so every blend of them does too —
    # whatever the intensity. This is the identity that catches a target built
    # under the wrong degrees-of-freedom convention, which would leave the
    # diagonal off by T/(T-1).
    panel = one_factor(7, 45, seed=31)
    result = ledoit_wolf(panel)
    unbiased = sample_covariance(panel, ddof=1)
    for i in range(7):
        assert result.matrix[i][i] == pytest.approx(unbiased[i][i], abs=1e-20)


def test_the_reported_parts_reproduce_the_reported_matrix() -> None:
    result = ledoit_wolf(one_factor(5, 60, seed=32))
    rebuilt = blend(result.sample, result.target, result.intensity)
    for i in range(5):
        for j in range(5):
            assert result.matrix[i][j] == pytest.approx(rebuilt[i][j], abs=1e-20)


def test_the_shrunk_matrix_is_symmetric_and_positive_semidefinite() -> None:
    result = ledoit_wolf(one_factor(9, 15, seed=33))
    assert result.diagnostics.positive_semidefinite
    for i in range(9):
        for j in range(9):
            assert result.matrix[i][j] == pytest.approx(result.matrix[j][i], abs=1e-20)


def test_shrinkage_never_worsens_the_conditioning_of_a_short_sample() -> None:
    panel = one_factor(10, 14, seed=34)
    sample = sample_covariance(panel)
    result = ledoit_wolf(panel)
    assert condition_number(result.matrix) < condition_number(sample)


def test_a_singular_sample_becomes_factorisable() -> None:
    # Fewer observations than assets: the sample is rank-deficient by
    # construction and cannot be factored, which is the practical failure
    # shrinkage exists to prevent.
    panel = one_factor(12, 6, seed=35)
    with pytest.raises(NotPositiveDefinite):
        cholesky(sample_covariance(panel))
    cholesky(ledoit_wolf(panel).matrix)


def test_shrinkage_needs_more_than_one_observation() -> None:
    with pytest.raises(TooShort, match="Ledoit-Wolf"):
        ledoit_wolf(Panel.from_columns({"a": [0.01], "b": [0.02]}))


def test_a_sample_that_already_is_the_target_shrinks_to_itself() -> None:
    # gamma is zero, so the optimum is a division by zero. Taking the target
    # costs nothing here because it is the sample, which is what is reported.
    panel = Panel.from_columns({"a": [0.01, -0.01, 0.02, -0.02]})
    result = ledoit_wolf(panel)
    assert result.intensity == 1.0
    assert result.matrix[0][0] == pytest.approx(result.sample[0][0], abs=1e-20)


# -- diagnostics -------------------------------------------------------------


def test_a_short_sample_is_reported_as_numerically_singular() -> None:
    # Not by testing the condition number for infinity: a rank-deficient sample
    # arrives with a smallest eigenvalue around 1e-18 rather than 0, so that
    # test would call it well conditioned.
    panel = one_factor(12, 6, seed=41)
    report = diagnose(sample_covariance(panel), observations=6)
    assert report.numerically_singular
    assert not math.isinf(report.condition_number)


def test_a_healthy_sample_is_not_reported_as_singular() -> None:
    panel = one_factor(4, 500, seed=42)
    assert not diagnose(sample_covariance(panel), observations=500).numerically_singular


def test_observations_per_parameter_falls_as_the_universe_grows() -> None:
    # 5 assets over 100 periods is 500 numbers for 15 parameters; 50 assets over
    # the same 100 periods is 5000 numbers for 1275. The data grew tenfold and
    # the ratio fell.
    small = diagnose(sample_covariance(one_factor(5, 100, seed=43)), observations=100)
    large = diagnose(sample_covariance(one_factor(50, 100, seed=43)), observations=100)
    assert small.observations_per_parameter > large.observations_per_parameter
    assert small.observations_per_parameter == pytest.approx(500 / 15, abs=1e-12)

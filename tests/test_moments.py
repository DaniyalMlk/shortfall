"""Skewness and excess kurtosis.

Pinned against closed forms and identities rather than against recorded values,
for the reason the rest of this suite gives: a transcribed constant tests the
transcription, and if the constant and the code are wrong in the same way
nothing fails at all.

Two distributions carry the weight here because both have exactly known shape
and both can be represented as a finite sample with no sampling error:

*The discrete uniform on n equally spaced points* has zero skewness and excess
kurtosis exactly ``-(6/5)(n^2 + 1)/(n^2 - 1)``. It approaches ``-1.2`` from
below as n grows, which is the continuous uniform's value, so the same formula
also checks that the estimator converges to the right thing rather than merely
to something stable.

*A two-point sample* is a Bernoulli trial, whose skewness is
``(1 - 2p) / sqrt(p(1 - p))`` and whose excess kurtosis is
``(1 - 6p(1 - p)) / (p(1 - p))``. Unlike the uniform it is asymmetric, and its
skewness diverges as p leaves the middle, so it exercises the third moment over
a wide range without any randomness.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.parametric import cornish_fisher_risk, normal_risk
from shortfall.series import ReturnSeries, TooShort


def series(*values: float, name: str = "a") -> ReturnSeries:
    return ReturnSeries(name=name, values=tuple(values))


def uniform(n: int, *, step: float = 0.001) -> ReturnSeries:
    """n equally spaced returns. Scaled small enough to be plausible returns."""
    return series(*(step * i for i in range(n)), name=f"uniform{n}")


def uniform_excess_kurtosis(n: int) -> float:
    """Exact excess kurtosis of the discrete uniform on n equally spaced points."""
    return -(6.0 / 5.0) * (n * n + 1) / (n * n - 1)


def two_point(n: int, k: int, *, high: float = 0.02) -> ReturnSeries:
    """``k`` observations at ``high`` and ``n - k`` at zero: a Bernoulli sample."""
    return series(*([high] * k + [0.0] * (n - k)), name=f"bernoulli{k}of{n}")


# -- closed forms ------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 5, 8, 20, 100, 501])
def test_uniform_excess_kurtosis_matches_its_closed_form(n: int) -> None:
    got = uniform(n).excess_kurtosis(corrected=False)
    assert got == pytest.approx(uniform_excess_kurtosis(n), abs=1e-13)


@pytest.mark.parametrize("n", [4, 5, 8, 20, 100, 501])
def test_a_symmetric_sample_has_no_skewness(n: int) -> None:
    """Exactly zero up to cancellation error, not merely small.

    The third central moment of a symmetric sample is a sum that cancels term by
    term, so what is left is floating-point noise around a true zero. The bound
    is on that noise, scaled by the second moment the ratio divides by.
    """
    assert uniform(n).skewness(corrected=False) == pytest.approx(0.0, abs=1e-12)
    assert uniform(n).skewness(corrected=True) == pytest.approx(0.0, abs=1e-12)


def test_the_uniform_approaches_the_continuous_value_from_below() -> None:
    """-1.2 is the continuous uniform's excess kurtosis, and n gets there.

    Worth asserting the direction as well as the limit. The discrete value is
    always below -1.2, so a sign slip in the closed form would still converge to
    something near -1.2 and only the approach would betray it.
    """
    values = [uniform(n).excess_kurtosis(corrected=False) for n in (4, 10, 50, 250, 1000)]
    assert all(value < -1.2 for value in values)
    assert values == sorted(values)
    assert values[-1] == pytest.approx(-1.2, abs=1e-5)


@pytest.mark.parametrize(("n", "k"), [(10, 3), (20, 7), (100, 11), (50, 25), (40, 37)])
def test_two_point_moments_match_the_bernoulli_forms(n: int, k: int) -> None:
    sample = two_point(n, k)
    p = k / n
    spread = p * (1.0 - p)
    assert sample.skewness(corrected=False) == pytest.approx(
        (1.0 - 2.0 * p) / math.sqrt(spread), rel=1e-12
    )
    assert sample.excess_kurtosis(corrected=False) == pytest.approx(
        (1.0 - 6.0 * spread) / spread, rel=1e-12
    )


def test_a_balanced_two_point_sample_is_symmetric_and_maximally_thin_tailed() -> None:
    """p = 1/2 gives skewness 0 and excess kurtosis exactly -2.

    -2 is the theoretical minimum for any distribution, so this is the one case
    where the estimator's output can be checked against a bound rather than a
    formula.
    """
    sample = two_point(40, 20)
    assert sample.skewness(corrected=False) == pytest.approx(0.0, abs=1e-12)
    assert sample.excess_kurtosis(corrected=False) == pytest.approx(-2.0, rel=1e-12)


# -- invariances -------------------------------------------------------------


def _random_returns(seed: int, n: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0004, 0.011) + 0.3 * rng.expovariate(200.0) for _ in range(n)]


@pytest.mark.parametrize("corrected", [True, False])
def test_both_moments_are_invariant_under_shift_and_positive_scale(corrected: bool) -> None:
    """Standardised moments do not move when the units do.

    This is the identity that catches a missing division by the second moment,
    or a correction applied to the moment rather than to the ratio: either one
    leaves a quantity that changes when the series is rescaled.
    """
    values = _random_returns(1, 200)
    base = series(*values)
    moved = series(*(0.002 + 0.5 * value for value in values))
    assert moved.skewness(corrected=corrected) == pytest.approx(
        base.skewness(corrected=corrected), rel=1e-11
    )
    assert moved.excess_kurtosis(corrected=corrected) == pytest.approx(
        base.excess_kurtosis(corrected=corrected), rel=1e-11
    )


@pytest.mark.parametrize("corrected", [True, False])
def test_negating_a_series_flips_the_skewness_and_leaves_the_kurtosis(corrected: bool) -> None:
    values = _random_returns(2, 200)
    base = series(*values)
    flipped = series(*(-value for value in values))
    assert flipped.skewness(corrected=corrected) == pytest.approx(
        -base.skewness(corrected=corrected), rel=1e-11
    )
    assert flipped.excess_kurtosis(corrected=corrected) == pytest.approx(
        base.excess_kurtosis(corrected=corrected), rel=1e-11
    )


@pytest.mark.parametrize("n", [5, 12, 60, 252])
def test_the_skewness_correction_is_the_published_factor(n: int) -> None:
    """``G1 / g1 = sqrt(n(n-1)) / (n-2)``, exactly."""
    sample = series(*_random_returns(3, n))
    ratio = sample.skewness(corrected=True) / sample.skewness(corrected=False)
    assert ratio == pytest.approx(math.sqrt(n * (n - 1)) / (n - 2), rel=1e-12)


@pytest.mark.parametrize("n", [6, 12, 60, 252])
def test_the_kurtosis_correction_is_the_published_transformation(n: int) -> None:
    """``G2 = (n-1)/((n-2)(n-3)) * ((n+1) g2 + 6)``, exactly.

    Not a ratio: unlike the skewness correction this one is affine rather than
    multiplicative, which is why ``G2`` can have the opposite sign to ``g2``.
    """
    sample = series(*_random_returns(4, n))
    uncorrected = sample.excess_kurtosis(corrected=False)
    expected = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * uncorrected + 6.0)
    assert sample.excess_kurtosis(corrected=True) == pytest.approx(expected, rel=1e-12)


def test_the_corrections_vanish_as_the_sample_grows() -> None:
    """Both estimators agree in the limit, and disagree where it matters.

    The point of the default is that the gap is material at the sample sizes risk
    work runs on. A year of daily data puts the skewness correction under one
    percent; a year of monthly data puts it near fifteen.
    """
    monthly = series(*_random_returns(5, 12))
    daily = series(*_random_returns(5, 252))

    monthly_gap = abs(monthly.skewness() / monthly.skewness(corrected=False) - 1.0)
    daily_gap = abs(daily.skewness() / daily.skewness(corrected=False) - 1.0)
    assert monthly_gap == pytest.approx(0.1489, abs=5e-4)
    assert daily_gap == pytest.approx(0.0060, abs=5e-4)


# -- the bias the correction exists to remove --------------------------------


def test_the_uncorrected_kurtosis_is_biased_downwards_on_normal_data() -> None:
    """``E[g2] = -6/(n+1)`` where the truth is zero; ``G2`` has no such bias.

    This is the whole argument for the default: on a short window of perfectly
    well-behaved returns the uncorrected ratio reports thin tails, which for a
    risk number is being wrong in the comfortable direction.

    The divisor is ``n + 1``, not ``n``. The commonly quoted ``-6/n`` is close
    enough to pass a loose test and is wrong: at twenty observations the two
    differ by 0.0143, which was four standard errors of the mean over forty
    thousand replications when this was measured, and the measurement picked
    ``n + 1``. The tolerances below are three standard errors of the mean at each
    sample size — wide enough not to flake on a seed, narrow enough that they
    would not have accepted ``-6/n`` at ``n = 20``.
    """
    rng = random.Random(90210)
    replications = 4000
    for n, tolerance in ((20, 0.012), (60, 0.009)):
        uncorrected = []
        corrected = []
        for _ in range(replications):
            values = tuple(rng.gauss(0.0, 0.01) for _ in range(n))
            sample = ReturnSeries(name="normal", values=values)
            uncorrected.append(sample.excess_kurtosis(corrected=False))
            corrected.append(sample.excess_kurtosis(corrected=True))

        mean_uncorrected = sum(uncorrected) / replications
        mean_corrected = sum(corrected) / replications
        assert mean_uncorrected == pytest.approx(-6.0 / (n + 1), abs=tolerance)
        assert abs(mean_corrected) < abs(mean_uncorrected) / 2.0
        assert mean_corrected == pytest.approx(0.0, abs=tolerance)

    # The claim that the tolerance discriminates, rather than being wide enough
    # to have accepted either form. Asserted rather than left to the comment.
    assert abs(-6.0 / 20 - -6.0 / 21) > 0.012


# -- refusals ----------------------------------------------------------------


def test_a_flat_series_has_no_shape() -> None:
    """Refused, not reported as zero.

    Zero skewness and zero excess kurtosis are what a *normal* series reports, so
    returning them for a constant series would make the degenerate case
    indistinguishable from the well-behaved one.
    """
    flat = series(*([0.004] * 50))
    for call in (flat.skewness, flat.excess_kurtosis):
        with pytest.raises(ValueError, match="no usable dispersion"):
            call()


def test_a_series_flat_only_in_binary_is_refused_too() -> None:
    """The guard is against rounding noise, not against an exact zero.

    Fifty copies of 0.004 do not average to exactly 0.004 in binary, so the
    deviations are about 1e-18 and the sum of their squares is around 1e-34 —
    strictly positive. A ``m_2 <= 0`` test passes it straight through and the
    standardised moments come back as whatever the rounding happened to be: a
    number of plausible size and no meaning. Comparing against the rounding error
    in ``m_2`` is what catches it.
    """
    flat = series(*([0.004] * 50))
    with pytest.raises(ValueError, match=r"sum of squared deviations is [0-9.e-]+, at or below"):
        flat.skewness(corrected=False)

    # And a series with genuine, if tiny, dispersion is *not* refused: the guard
    # must not swallow a real answer to be sure of rejecting a fake one.
    faint = series(*[0.004 + (1e-9 if index % 2 else -1e-9) for index in range(50)])
    assert faint.excess_kurtosis(corrected=False) == pytest.approx(-2.0, rel=1e-9)


@pytest.mark.parametrize(
    ("n", "method", "corrected"),
    [
        (2, "skewness", True),
        (1, "skewness", False),
        (3, "excess_kurtosis", True),
        (1, "excess_kurtosis", False),
    ],
)
def test_too_few_observations_is_refused_with_the_minimum_named(
    n: int, method: str, corrected: bool
) -> None:
    sample = series(*_random_returns(6, n))
    with pytest.raises(TooShort, match="needs at least"):
        getattr(sample, method)(corrected=corrected)


def test_the_uncorrected_estimators_reach_further_down_than_the_corrected_ones() -> None:
    """Two observations are enough for a ratio and not for a correction.

    The correction divides by ``n - 2`` and ``n - 3``, so it has a hard floor the
    ratio does not. Offering the ratio there is not a loophole: it is the honest
    answer to a different question.
    """
    pair = series(0.01, -0.01)
    assert pair.skewness(corrected=False) == pytest.approx(0.0, abs=1e-12)
    assert pair.excess_kurtosis(corrected=False) == pytest.approx(-2.0, rel=1e-12)
    with pytest.raises(TooShort):
        pair.skewness()


# -- the reason these exist --------------------------------------------------


def test_the_moments_feed_the_cornish_fisher_estimator() -> None:
    """The wiring these methods were added for, end to end.

    A left-skewed sample must produce a larger value at risk under the
    Cornish-Fisher correction than under the normal assumption, because that is
    what the correction is for. Asserting the direction rather than a value keeps
    this a test of the wiring — that the estimated moments arrive with the right
    signs and the excess convention intact — rather than a second test of the
    expansion, which has its own.
    """
    rng = random.Random(4242)
    # Negative jumps on top of a normal core: a return series with a left tail.
    values = tuple(
        rng.gauss(0.0005, 0.008) - (0.05 if rng.random() < 0.03 else 0.0) for _ in range(750)
    )
    sample = ReturnSeries(name="skewed", values=values)

    skewness = sample.skewness()
    excess = sample.excess_kurtosis()
    assert skewness < -0.5, skewness
    assert excess > 1.0, excess

    normal = normal_risk(mean=sample.mean, volatility=sample.stdev(), confidence=0.99)
    adjusted = cornish_fisher_risk(
        mean=sample.mean,
        volatility=sample.stdev(),
        confidence=0.99,
        skewness=skewness,
        excess_kurtosis=excess,
    )
    assert adjusted.value_at_risk > normal.value_at_risk
    assert adjusted.expected_shortfall > normal.expected_shortfall

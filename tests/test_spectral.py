"""Spectral risk measures.

Four identities carry this file, and none of them is checked against a number
this module produced.

**The shortfall spectrum reproduces the estimator already in the package.**
:func:`shortfall.historical.sample_expected_shortfall` computes the average of
the worst ``n p`` observations with a partial one at the edge, by hand. The
estimator here is a weighted sum of order statistics with the weights taken
from the spectrum's block masses. They are unrelated derivations of the same
quantity and they agree to rounding, which is a real check rather than a
tautology.

**A discrete mixture of expected shortfalls equals the step spectrum's
measure.** Kusuoka's representation in the only form that can be checked
against arithmetic instead of against a quadrature.

**The Wang transform has a closed form under a normal.** ``mean - shift *
volatility``, exactly, for every shift. There is nothing from this module in
that statement, so it is what validates the quadrature and the block masses
together.

**Comonotonic additivity holds to rounding**, and it is asserted beside
:func:`shortfall.expectile.comonotonic_gap` on the same data, because that is
precisely the property Phase 19 gave up and the contrast is the point.

The rest is the measured behaviour — where in the tail each spectrum puts its
weight, and how badly the non-coherent branch fails — and the refusals.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from shortfall.expectile import comonotonic_gap, sample_expectile
from shortfall.historical import sample_expected_shortfall
from shortfall.parametric import normal_risk
from shortfall.series import TooShort
from shortfall.spectral import (
    BadSpectrum,
    ExponentialSpectrum,
    MixtureSpectrum,
    PowerSpectrum,
    ShortfallSpectrum,
    Spectrum,
    WangSpectrum,
    check_coherence,
    matching_shift,
    normal_spectral,
    spectral_risk,
    tail_share,
)

MEAN = 0.0004
VOLATILITY = 0.011
PROBABILITIES = (0.001, 0.005, 0.01, 0.025, 0.05, 0.10)

COHERENT: tuple[Spectrum, ...] = (
    ShortfallSpectrum(0.025),
    ExponentialSpectrum(50.0),
    PowerSpectrum(8.0),
    WangSpectrum(2.0),
    MixtureSpectrum(((0.01, 1.0), (0.10, 1.0))),
)


def gaussian(count: int = 2000, seed: int = 5) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(MEAN, VOLATILITY) for _ in range(count)]


def student_t(count: int = 40_000, seed: int = 11) -> list[float]:
    """A t with five degrees of freedom, scaled to a 1% volatility.

    Built from a normal over the root of a chi-square rather than drawn from a
    library, so the sample has the tail the claims below are about.
    """
    rng = random.Random(seed)
    raw = []
    for _ in range(count):
        z = rng.gauss(0.0, 1.0)
        chi = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(5))
        raw.append(z / math.sqrt(chi / 5.0))
    scale = math.sqrt(sum(one * one for one in raw) / count)
    return [0.01 * one / scale for one in raw]


def paired(seed: int) -> tuple[list[float], list[float]]:
    """Two positions with a dependence drawn from the seed.

    Gaussian or cubed-Gaussian marginals and a correlation anywhere in range,
    so a sweep over seeds is a sweep over ordinary dependence structures rather
    than over constructed counterexamples.
    """
    rng = random.Random(seed)
    count = 400
    correlation = rng.uniform(-1.0, 1.0)
    power = rng.choice([1, 3])
    first = [rng.gauss(0.0, 1.0) for _ in range(count)]
    residual = math.sqrt(max(1.0 - correlation * correlation, 0.0))
    second = [
        correlation * one + residual * rng.gauss(0.0, 1.0) for one in first
    ]
    if power > 1:
        first = [one * abs(one) ** (power - 1) for one in first]
        second = [one * abs(one) ** (power - 1) for one in second]
    return first, second


# -- the spectra are spectra --------------------------------------------------


@pytest.mark.parametrize("spectrum", COHERENT)
def test_a_spectrum_integrates_to_one(spectrum: Spectrum) -> None:
    """Over a thousand blocks, which is how the estimator uses it."""
    count = 1000
    total = sum(
        spectrum.mass(index / count, (index + 1) / count) for index in range(count)
    )
    assert total == pytest.approx(1.0, abs=1e-13)
    assert spectrum.mass(0.0, 1.0) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("spectrum", COHERENT)
def test_a_coherent_spectrum_is_non_increasing(spectrum: Spectrum) -> None:
    """The property coherence rests on, checked rather than asserted in a comment."""
    assert spectrum.is_coherent
    levels = [0.001 + 0.01 * step for step in range(99)]
    densities = [spectrum.density(level) for level in levels]
    assert densities == sorted(densities, reverse=True)


@pytest.mark.parametrize("exponent", [0.3, 0.5, 0.7, 0.9])
def test_the_power_spectrum_below_one_is_increasing_and_says_so(
    exponent: float,
) -> None:
    spectrum = PowerSpectrum(exponent)
    assert not spectrum.is_coherent
    densities = [spectrum.density(0.01 * step) for step in range(1, 100)]
    assert densities == sorted(densities)


def test_the_power_spectrum_at_one_is_flat() -> None:
    """Which makes the measure the mean, and the mean is where coherence begins."""
    spectrum = PowerSpectrum(1.0)
    assert spectrum.is_coherent
    assert all(
        spectrum.density(0.01 * step) == pytest.approx(1.0, abs=1e-15)
        for step in range(100)
    )
    sample = gaussian()
    assert spectral_risk(sample, spectrum) == pytest.approx(
        sum(sample) / len(sample), rel=1e-13
    )


@pytest.mark.parametrize("spectrum", COHERENT)
def test_the_block_masses_are_consistent_with_each_other(
    spectrum: Spectrum,
) -> None:
    """Splitting a block has to give the same total, or the estimator is wrong."""
    for lower, upper in ((0.0, 0.03), (0.02, 0.4), (0.3, 1.0)):
        middle = 0.5 * (lower + upper)
        assert spectrum.mass(lower, upper) == pytest.approx(
            spectrum.mass(lower, middle) + spectrum.mass(middle, upper), abs=1e-15
        )


# -- identity one: the existing estimator -------------------------------------


@pytest.mark.parametrize("probability", PROBABILITIES)
def test_the_shortfall_spectrum_is_the_existing_estimator(
    probability: float,
) -> None:
    """Two unrelated derivations of the same sum, agreeing to rounding.

    The existing estimator works out the average of the worst ``n p``
    observations with a partial one at the edge, directly. This one integrates
    a step spectrum against the empirical quantile. Neither was written with
    the other in view.
    """
    sample = gaussian()
    spectral = spectral_risk(sample, ShortfallSpectrum(probability))
    existing, _ = sample_expected_shortfall(sample, probability)
    assert spectral == pytest.approx(existing, abs=1e-16)


def test_the_whole_distribution_is_the_mean() -> None:
    """``p = 1`` leaves every observation weighted ``1/n``."""
    sample = gaussian()
    assert spectral_risk(sample, ShortfallSpectrum(1.0)) == pytest.approx(
        sum(sample) / len(sample), rel=1e-14
    )


# -- identity two: Kusuoka, as arithmetic -------------------------------------


@pytest.mark.parametrize(
    "components",
    [
        (((0.01, 0.3), (0.05, 0.5), (0.25, 0.2))),
        (((0.025, 1.0),)),
        (((0.001, 0.9), (1.0, 0.1))),
    ],
)
def test_a_mixture_is_the_same_mixture_of_expected_shortfalls(
    components: tuple[tuple[float, float], ...],
) -> None:
    """The representation in the one form checkable against arithmetic.

    A quadrature of the continuous representation would only ever be as good
    as the quadrature. A discrete mixture has an exact answer, and the exact
    answer is the same combination of numbers the package already computes.
    """
    sample = gaussian()
    total = sum(weight for _, weight in components)
    combination = sum(
        weight / total * sample_expected_shortfall(sample, probability)[0]
        for probability, weight in components
    )
    assert spectral_risk(sample, MixtureSpectrum(components)) == pytest.approx(
        combination, abs=1e-16
    )


def test_a_mixture_normalises_its_weights() -> None:
    """So a caller can pass relative weights without scaling them first."""
    sample = gaussian()
    scaled = MixtureSpectrum(((0.01, 3.0), (0.05, 5.0)))
    unit = MixtureSpectrum(((0.01, 0.375), (0.05, 0.625)))
    assert spectral_risk(sample, scaled) == pytest.approx(
        spectral_risk(sample, unit), rel=1e-14
    )


# -- identity three: the Wang transform under a normal ------------------------


@pytest.mark.parametrize("shift", [0.0, 0.25, 0.5, 1.0, 1.96, 3.0, 5.0])
def test_the_wang_transform_of_a_normal_is_exact(shift: float) -> None:
    """``mean - shift * volatility``, to 0.0, with nothing from this module in it.

    The transform is built so that shifting the normal quantile by ``lambda``
    moves the measure by ``lambda`` standard deviations. That is the statement
    the quadrature and the block masses are both validated against.
    """
    assert normal_spectral(
        mean=MEAN, volatility=VOLATILITY, spectrum=WangSpectrum(shift)
    ) == pytest.approx(MEAN - shift * VOLATILITY, abs=1e-18)


def test_a_zero_wang_shift_is_the_mean() -> None:
    spectrum = WangSpectrum(0.0)
    assert all(
        spectrum.density(0.01 * step) == pytest.approx(1.0, abs=1e-15)
        for step in range(1, 100)
    )
    sample = gaussian()
    assert spectral_risk(sample, spectrum) == pytest.approx(
        sum(sample) / len(sample), rel=1e-13
    )


@pytest.mark.parametrize("shift", [0.5, 2.0])
def test_the_wang_sample_estimator_converges_at_the_root_n_rate(
    shift: float,
) -> None:
    """The estimator against the closed form, at the rate rather than at a point.

    The first version of this compared one draw at each size and asserted the
    error fell. It does not, reliably: the 2,000-observation draw came in at
    2.3e-05 and the 32,000 one at 1.0e-04, which is one lucky sample and one
    ordinary one, not a failure to converge. A single-draw comparison is a
    coin flip dressed as a convergence test, and this package has walked into
    it before.

    So the quantity asserted is the root-mean-square error over twelve
    independent samples at each size, against the prediction that it falls as
    one over the root of the count: a quadrupling of the sample should halve
    it. Independent seeds everywhere -- a seed reused across sizes correlates
    the rows and destroys the comparison.
    """
    spectrum = WangSpectrum(shift)
    exact = MEAN - shift * VOLATILITY
    seed = 900
    rms = []
    for count in (2_000, 8_000, 32_000):
        total = 0.0
        for _ in range(12):
            rng = random.Random(seed)
            seed += 1
            sample = [rng.gauss(MEAN, VOLATILITY) for _ in range(count)]
            total += (spectral_risk(sample, spectrum) - exact) ** 2
        rms.append(math.sqrt(total / 12))
    assert rms == sorted(rms, reverse=True)
    for coarse, fine in pairwise(rms):
        assert 1.4 < coarse / fine < 2.8


# -- the quadrature, and the two things that broke it -------------------------


@pytest.mark.parametrize("probability", PROBABILITIES)
def test_the_quadrature_matches_the_closed_form_normal_shortfall(
    probability: float,
) -> None:
    """To machine precision, which it took two fixes to reach.

    The shortfall spectrum steps at ``u = p``, and a Gauss rule integrates a
    jump not at all: straddling it cost 1.7e-03 relative at 97.5% and 5.3e-03
    at 99%. Every spectrum now reports its discontinuities and they are forced
    onto panel edges. The endpoint grading then left the innermost edge at
    2.3e-10, which cost 2.0e-07 on a 99.5% shortfall -- the truncated sliver
    times ``1/p``, so the error grew as the tail thinned, the wrong way round
    for a tail measure.

    The shortfall spectrum is deliberately *not* special-cased in
    :func:`normal_spectral`, because this comparison is the only check on the
    quadrature that the other spectra then inherit.
    """
    quadrature = normal_spectral(
        mean=MEAN, volatility=VOLATILITY, spectrum=ShortfallSpectrum(probability)
    )
    closed = -normal_risk(
        mean=MEAN, volatility=VOLATILITY, confidence=1.0 - probability
    ).expected_shortfall
    assert quadrature == pytest.approx(closed, rel=5e-15)


def test_the_quadrature_matches_a_mixture_of_closed_forms() -> None:
    """Which exercises a spectrum with three discontinuities rather than one."""
    components = ((0.01, 0.3), (0.05, 0.5), (0.25, 0.2))
    combination = sum(
        weight
        * -normal_risk(
            mean=MEAN, volatility=VOLATILITY, confidence=1.0 - probability
        ).expected_shortfall
        for probability, weight in components
    )
    assert normal_spectral(
        mean=MEAN, volatility=VOLATILITY, spectrum=MixtureSpectrum(components)
    ) == pytest.approx(combination, rel=5e-14)


def test_the_degenerate_spectra_integrate_to_the_mean() -> None:
    """A flat spectrum has to come back with the mean and nothing else."""
    assert normal_spectral(
        mean=MEAN, volatility=VOLATILITY, spectrum=PowerSpectrum(1.0)
    ) == pytest.approx(MEAN, rel=1e-12)
    assert normal_spectral(
        mean=MEAN, volatility=VOLATILITY, spectrum=ShortfallSpectrum(1.0)
    ) == pytest.approx(MEAN, rel=1e-12)


def test_a_zero_volatility_normal_is_its_mean() -> None:
    for spectrum in COHERENT:
        assert normal_spectral(
            mean=MEAN, volatility=0.0, spectrum=spectrum
        ) == pytest.approx(MEAN, abs=1e-15)


@pytest.mark.parametrize("nodes", [8, 16, 64])
def test_the_quadrature_does_not_depend_on_the_node_count(nodes: int) -> None:
    """Once the breaks are on edges, the rule has order and more nodes buy nothing.

    Worth asserting: a quadrature whose answer moves with the node count has
    not converged, and the failure above looked exactly like that.
    """
    closed = -normal_risk(
        mean=MEAN, volatility=VOLATILITY, confidence=0.975
    ).expected_shortfall
    assert normal_spectral(
        mean=MEAN,
        volatility=VOLATILITY,
        spectrum=ShortfallSpectrum(0.025),
        nodes=nodes,
    ) == pytest.approx(closed, rel=1e-12)


# -- identity four: comonotonic additivity ------------------------------------


@pytest.mark.parametrize("spectrum", COHERENT)
def test_a_distortion_measure_is_comonotonically_additive(
    spectrum: Spectrum,
) -> None:
    """To rounding, which is what a capital rule wants at full dependence.

    Two positions that move together offer no diversification and a distortion
    measure reports none.
    """
    first = student_t(count=4_000, seed=11)
    second = [0.6 * first[i] + 0.4 * first[(i * 7) % len(first)] for i in range(len(first))]
    result = check_coherence(first, second, spectrum)
    alone = spectral_risk(sorted(first), spectrum) + spectral_risk(
        sorted(second), spectrum
    )
    assert abs(result.comonotonic / alone) < 1e-12


def test_the_expectile_reports_a_benefit_where_a_spectrum_reports_none() -> None:
    """The contrast with Phase 19, on one sample, in relative terms.

    Expectiles are coherent and elicitable, and comonotonic additivity is what
    they pay for the second. Eleven orders of magnitude between the two gaps,
    on the same data.
    """
    first = sorted(student_t(count=4_000, seed=11))
    second = sorted(
        [0.6 * first[i] + 0.4 * first[(i * 7) % len(first)] for i in range(len(first))]
    )
    spectrum = ShortfallSpectrum(0.025)
    alone = spectral_risk(first, spectrum) + spectral_risk(second, spectrum)
    spectral_gap = abs(
        check_coherence(first, second, spectrum).comonotonic / alone
    )

    level = 0.975
    expectiles = (
        sample_expectile(first, level).value + sample_expectile(second, level).value
    )
    expectile_gap = abs(comonotonic_gap(first, second, level) / expectiles)

    assert spectral_gap < 1e-12
    assert expectile_gap > 1e-5
    assert expectile_gap > 1_000_000.0 * spectral_gap


# -- coherence ----------------------------------------------------------------


@pytest.mark.parametrize("spectrum", COHERENT)
def test_a_coherent_spectrum_is_subadditive_on_every_sample(
    spectrum: Spectrum,
) -> None:
    """Two hundred ordinary dependence structures, no violations."""
    for seed in range(200):
        first, second = paired(seed)
        result = check_coherence(first, second, spectrum)
        assert result.spectrum_is_coherent
        assert result.subadditive


@pytest.mark.parametrize("exponent", [0.3, 0.5, 0.7, 0.9])
def test_the_non_coherent_branch_fails_on_every_sample(exponent: float) -> None:
    """Not a corner case: 200 of 200, by up to 97% of the measure itself.

    Which is the argument for implementing the branch rather than refusing it.
    A reader can be told that coherence needs a non-increasing spectrum, or
    shown what happens without one on data that was not built to break it.
    """
    spectrum = PowerSpectrum(exponent)
    violations = 0
    worst = -math.inf
    for seed in range(200):
        first, second = paired(seed)
        result = check_coherence(first, second, spectrum)
        assert not result.spectrum_is_coherent
        if not result.subadditive:
            violations += 1
        alone = spectral_risk(first, spectrum) + spectral_risk(second, spectrum)
        worst = max(worst, result.subadditivity / abs(alone))
    assert violations == 200
    assert worst > 0.5


def test_the_mean_is_exactly_additive() -> None:
    """The boundary between the two branches, and it is an exact zero."""
    first, second = paired(0)
    result = check_coherence(first, second, PowerSpectrum(1.0))
    assert result.subadditivity == pytest.approx(0.0, abs=1e-13)
    assert result.comonotonic == pytest.approx(0.0, abs=1e-13)
    assert result.subadditive


# -- what the weighting is worth ----------------------------------------------


def test_matched_spectra_disagree_about_where_the_risk_comes_from() -> None:
    """The practical finding, and the argument for stating a spectrum.

    Every row here is matched to the same headline charge, so a committee
    reading the number alone could not tell them apart. The share of it coming
    from the worst 0.5% of outcomes runs from 29.2% to 59.7% — the Wang
    transform carrying twice as much of an identical figure in the part of the
    tail a sample has least to say about.
    """
    sample = student_t()
    target = tail_share(sample, ShortfallSpectrum(0.025), 0.005)
    assert -target.quantile > 0.0
    assert target.charge == pytest.approx(0.027169, abs=5e-5)

    def matched(make: object, low: float, high: float) -> Spectrum:
        """Bisect the spectrum's own parameter onto the shortfall charge."""
        for _ in range(80):
            middle = 0.5 * (low + high)
            built = make(middle)  # type: ignore[operator]
            if tail_share(sample, built, 0.005).charge < target.charge:
                low = middle
            else:
                high = middle
        return make(0.5 * (low + high))  # type: ignore[operator, no-any-return]

    shares = {"shortfall": target.share}
    for name, make, low, high in (
        ("wang", WangSpectrum, 0.01, 6.0),
        ("exponential", ExponentialSpectrum, 0.1, 400.0),
        ("power", PowerSpectrum, 1.0, 400.0),
    ):
        spectrum = matched(make, low, high)
        split = tail_share(sample, spectrum, 0.005)
        assert split.charge == pytest.approx(target.charge, rel=1e-6)
        shares[name] = split.share

    assert shares["shortfall"] == pytest.approx(0.292, abs=0.01)
    assert shares["power"] == pytest.approx(0.378, abs=0.01)
    assert shares["exponential"] == pytest.approx(0.380, abs=0.01)
    assert shares["wang"] == pytest.approx(0.597, abs=0.015)
    assert shares["wang"] > 1.9 * shares["shortfall"]


def test_a_tail_share_is_bounded_by_the_whole_charge() -> None:
    sample = gaussian()
    for spectrum in COHERENT:
        split = tail_share(sample, spectrum, 0.01)
        assert 0.0 <= split.extreme <= split.charge
        assert 0.0 <= split.share <= 1.0


def test_the_whole_tail_is_the_whole_charge() -> None:
    """A cut-off of one leaves nothing outside it."""
    sample = gaussian()
    split = tail_share(sample, ShortfallSpectrum(0.05), 1.0)
    assert split.share == pytest.approx(1.0, rel=1e-13)


def test_a_more_conservative_spectrum_returns_a_worse_number() -> None:
    """Monotone in each family's own parameter, which is the least it must be."""
    sample = gaussian()
    for make, parameters in (
        (WangSpectrum, (0.0, 0.5, 1.0, 2.0, 3.0)),
        (ExponentialSpectrum, (1.0, 10.0, 50.0, 200.0)),
        (PowerSpectrum, (1.0, 2.0, 8.0, 50.0)),
    ):
        values = [spectral_risk(sample, make(one)) for one in parameters]
        assert values == sorted(values, reverse=True)


# -- stating a spectrum in units already in use -------------------------------


def test_a_wang_shift_can_be_matched_to_a_target() -> None:
    """One division, because the closed form is linear in the shift."""
    target = -normal_risk(
        mean=MEAN, volatility=VOLATILITY, confidence=0.975
    ).expected_shortfall
    shift = matching_shift(target, mean=MEAN, volatility=VOLATILITY)
    # A float round trip through a division and a multiplication, so the last
    # bit rather than the exactness the closed form itself has.
    assert normal_spectral(
        mean=MEAN, volatility=VOLATILITY, spectrum=WangSpectrum(shift)
    ) == pytest.approx(target, rel=1e-15)
    assert shift == pytest.approx(2.3375, abs=5e-4)


def test_a_matched_shift_does_not_transfer_between_distributions() -> None:
    """The same warning Phase 19 ends on, and for the same reason.

    A shift calibrated on a normal is a different amount of conservatism on a
    fatter tail, because it is a number of standard deviations and nothing
    about the shape.
    """
    normal_sample = gaussian(count=40_000, seed=21)
    heavy = student_t()
    target = -normal_risk(
        mean=MEAN, volatility=VOLATILITY, confidence=0.975
    ).expected_shortfall
    shift = matching_shift(target, mean=MEAN, volatility=VOLATILITY)
    spectrum = WangSpectrum(shift)
    on_normal = tail_share(normal_sample, spectrum, 0.005)
    on_heavy = tail_share(heavy, spectrum, 0.005)
    assert on_heavy.charge > on_normal.charge
    assert on_heavy.share > on_normal.share


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: ShortfallSpectrum(0.0), "lies in"),
        (lambda: ShortfallSpectrum(1.5), "lies in"),
        (lambda: ExponentialSpectrum(0.0), "positive aversion"),
        (lambda: ExponentialSpectrum(-1.0), "positive aversion"),
        (lambda: PowerSpectrum(0.0), "positive exponent"),
        (lambda: WangSpectrum(-0.5), "is not one"),
        (lambda: MixtureSpectrum(()), "at least one component"),
        (lambda: MixtureSpectrum(((0.0, 1.0),)), "lies in"),
        (lambda: MixtureSpectrum(((0.05, -1.0),)), "non-convex"),
        (lambda: MixtureSpectrum(((0.05, 0.0),)), "sum to nothing"),
    ],
)
def test_a_spectrum_that_is_not_one_is_refused(
    build: object, message: str
) -> None:
    with pytest.raises(BadSpectrum, match=message):
        build()  # type: ignore[operator]


def test_an_empty_sample_is_refused() -> None:
    with pytest.raises(TooShort, match="at least one observation"):
        spectral_risk([], ShortfallSpectrum(0.025))
    with pytest.raises(TooShort, match="at least one observation"):
        tail_share([], ShortfallSpectrum(0.025), 0.01)
    with pytest.raises(TooShort, match="at least one observation"):
        check_coherence([], [1.0], ShortfallSpectrum(0.025))


def test_mismatched_lengths_are_refused_rather_than_truncated() -> None:
    """The pairing carries the dependence, so a truncation would change it."""
    with pytest.raises(BadSpectrum, match="cannot be compared at different lengths"):
        check_coherence([1.0, 2.0, 3.0], [1.0, 2.0], ShortfallSpectrum(0.025))


def test_a_block_outside_the_unit_interval_is_refused() -> None:
    spectrum = PowerSpectrum(4.0)
    for lower, upper in ((-0.1, 0.5), (0.5, 1.1), (0.6, 0.4)):
        with pytest.raises(BadSpectrum, match="unit interval"):
            spectrum.mass(lower, upper)


def test_a_bad_normal_integration_is_refused() -> None:
    with pytest.raises(BadSpectrum, match="is not one"):
        normal_spectral(
            mean=0.0, volatility=-1.0, spectrum=ShortfallSpectrum(0.025)
        )
    with pytest.raises(BadSpectrum, match="cannot integrate"):
        normal_spectral(
            mean=0.0, volatility=0.01, spectrum=ShortfallSpectrum(0.025), nodes=1
        )


def test_a_bad_tail_cut_off_is_refused() -> None:
    with pytest.raises(BadSpectrum, match="lies in"):
        tail_share(gaussian(count=10), ShortfallSpectrum(0.025), 0.0)


def test_a_target_above_the_mean_is_refused() -> None:
    """It needs a negative shift, which is an increasing spectrum."""
    with pytest.raises(BadSpectrum, match="negative shift"):
        matching_shift(MEAN + 0.01, mean=MEAN, volatility=VOLATILITY)
    with pytest.raises(BadSpectrum, match="cannot scale a shift"):
        matching_shift(0.0, mean=MEAN, volatility=0.0)


def test_the_spectra_report_themselves() -> None:
    assert "97.5" in ShortfallSpectrum(0.025).name
    assert "aversion 50" in ExponentialSpectrum(50.0).name
    assert "exponent 4" in PowerSpectrum(4.0).name
    assert "shift 2" in WangSpectrum(2.0).name
    assert "mixture of" in MixtureSpectrum(((0.01, 1.0),)).name
    assert ShortfallSpectrum(0.025).breaks == (0.025,)
    assert PowerSpectrum(4.0).breaks == ()
    assert MixtureSpectrum(((0.05, 1.0), (0.01, 1.0))).breaks == (0.01, 0.05)

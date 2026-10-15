"""Tests for finite normal mixtures.

Every closed form here is checked against an independent route rather than
against itself: the distribution function against a quadrature of the density,
the tail mean against a quadrature of ``x f(x)``, the moments against the same
quadrature, the convolution against a numerical convolution integral, and the
horizon law against a brute-force enumeration of every regime path.

The measured claims in the module docstring are pinned, including the ones
that are negative results — the averaging error, the window in which value at
risk fails to be subadditive, the confidence level at which the square-root
rule changes direction, and the fact that a degenerate fit's log-likelihood is
a function of the variance floor rather than of the data.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from itertools import pairwise
from typing import ClassVar

import pytest

from shortfall.distributions import normal_ppf
from shortfall.mixture import (
    BadMixture,
    Component,
    NormalMixture,
    averaged_shortfall,
    convolve,
    diversification_gap,
    fit_mixture,
    horizon_mixture,
    mixture_risk,
    occupation_law,
    responsibilities,
    stationary,
    tail_ratio,
)

# The running example of the module docstring.
STRESSED = NormalMixture.two_regime(stress=0.05, stress_deviation=4.0)


def integrate(
    function: Callable[[float], float], low: float, high: float, panels: int = 20000
) -> float:
    """Composite Simpson, on a grid fine enough to be the reference.

    Deliberately crude and deliberately not the method under test. Every
    integrand here is smooth and the interval is chosen to hold essentially
    all of the mass, so the only question is whether the closed form agrees.
    """
    if panels % 2:
        panels += 1
    width = (high - low) / panels
    total: float = function(low) + function(high)
    for index in range(1, panels):
        total += (4.0 if index % 2 else 2.0) * function(low + index * width)
    return float(total * width / 3.0)


def draw(mixture: NormalMixture, count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(count):
        point = rng.random()
        running = 0.0
        chosen = mixture.components[-1]
        for one in mixture.components:
            running += one.weight
            if point <= running:
                chosen = one
                break
        out.append(rng.gauss(chosen.mean, chosen.deviation))
    return out


class TestTheLaw:
    """Closed forms against a quadrature of the density."""

    def test_the_density_integrates_to_one(self) -> None:
        assert integrate(STRESSED.pdf, -60.0, 60.0) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("at", [-5.0, -1.0, 0.0, 0.5, 3.0, 9.0])
    def test_the_distribution_function_is_the_integral(self, at: float) -> None:
        assert STRESSED.cdf(at) == pytest.approx(
            integrate(STRESSED.pdf, -60.0, at), abs=1e-12
        )

    @pytest.mark.parametrize("at", [-2.0, 0.0, 2.0, 6.0])
    def test_the_survival_is_the_complement(self, at: float) -> None:
        assert STRESSED.cdf(at) + STRESSED.survival(at) == pytest.approx(1.0, abs=1e-15)

    def test_the_survival_survives_the_far_tail(self) -> None:
        # Where 1 - cdf is zero or rounding, the far-side form is still a
        # number. This is the reason survival is not written as 1 - cdf.
        assert 1.0 - STRESSED.cdf(60.0) == 0.0
        assert 0.0 < STRESSED.survival(60.0) < 1e-50

    @pytest.mark.parametrize("at", [-1.0, 0.0, 2.0, 5.0])
    def test_the_partial_mean_is_the_integral(self, at: float) -> None:
        assert STRESSED.partial_mean(at) == pytest.approx(
            integrate(lambda x: x * STRESSED.pdf(x), at, 60.0), abs=1e-11
        )

    def test_the_moments(self) -> None:
        assert STRESSED.mean == pytest.approx(0.0, abs=1e-16)
        assert STRESSED.variance == pytest.approx(1.75, rel=1e-15)
        assert STRESSED.skewness == pytest.approx(0.0, abs=1e-12)
        assert STRESSED.excess_kurtosis == pytest.approx(10.4694, abs=1e-4)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_central_moments_against_a_quadrature(self, order: int) -> None:
        centre = STRESSED.mean
        want = integrate(
            lambda x: (x - centre) ** order * STRESSED.pdf(x), -80.0, 80.0, 60000
        )
        assert STRESSED.central_moment(order) == pytest.approx(want, abs=1e-6)

    def test_a_skewed_mixture_is_skewed(self) -> None:
        skewed = NormalMixture.of([(0.9, -0.2, 1.0), (0.1, 1.8, 3.0)])
        assert skewed.mean == pytest.approx(0.0, abs=1e-15)
        assert skewed.skewness > 1.0
        centre = skewed.mean
        want = integrate(lambda x: (x - centre) ** 3 * skewed.pdf(x), -80.0, 80.0, 60000)
        assert skewed.central_moment(3) == pytest.approx(want, abs=1e-6)

    @pytest.mark.parametrize("probability", [1e-6, 0.01, 0.3, 0.5, 0.9, 0.99, 1 - 1e-9])
    def test_the_quantile_inverts_the_distribution_function(
        self, probability: float
    ) -> None:
        at = STRESSED.quantile(probability)
        assert STRESSED.cdf(at) == pytest.approx(probability, rel=1e-12, abs=1e-15)

    def test_the_quantile_of_one_component_is_the_normal_one(self) -> None:
        single = NormalMixture.of([(1.0, 0.4, 2.5)])
        for probability in (0.01, 0.5, 0.975, 0.9999):
            assert single.quantile(probability) == pytest.approx(
                0.4 + 2.5 * normal_ppf(probability), rel=1e-12, abs=1e-14
            )

    def test_shifting_and_scaling(self) -> None:
        moved = STRESSED.shifted(1.5).scaled(2.0)
        assert moved.mean == pytest.approx(3.0, abs=1e-15)
        assert moved.variance == pytest.approx(4.0 * 1.75, rel=1e-14)


class TestRisk:
    """The quantities the module exists for."""

    PUBLISHED: ClassVar[dict[float, tuple[float, ...]]] = {
        0.90: (1.366, 2.244, 2.018, -0.101, 0.183),
        0.95: (1.806, 2.926, 2.372, -0.189, 0.326),
        0.975: (2.279, 3.844, 2.688, -0.301, 0.569),
        0.99: (3.445, 5.607, 3.065, -0.453, 0.973),
        0.999: (8.215, 9.684, 3.872, -0.600, 1.000),
    }

    @pytest.mark.parametrize("confidence", sorted(PUBLISHED))
    def test_the_published_table(self, confidence: float) -> None:
        want = self.PUBLISHED[confidence]
        got = mixture_risk(STRESSED, confidence)
        assert got.value_at_risk == pytest.approx(want[0], abs=0.001)
        assert got.expected_shortfall == pytest.approx(want[1], abs=0.001)
        assert got.averaged == pytest.approx(want[2], abs=0.001)
        assert got.averaging_error == pytest.approx(want[3], abs=0.001)
        assert got.responsibility[1] == pytest.approx(want[4], abs=0.001)

    @pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99, 0.999])
    def test_the_shortfall_is_the_conditional_mean(self, confidence: float) -> None:
        got = mixture_risk(STRESSED, confidence)
        quantile = got.value_at_risk
        want = integrate(
            lambda x: x * STRESSED.pdf(x), quantile, 80.0, 40000
        ) / STRESSED.survival(quantile)
        assert got.expected_shortfall == pytest.approx(want, rel=1e-9)

    def test_the_averaging_error_is_one_signed_and_grows(self) -> None:
        errors = [
            mixture_risk(STRESSED, confidence).averaging_error
            for confidence in (0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
        ]
        assert all(one < 0.0 for one in errors)
        assert errors == sorted(errors, reverse=True)

    def test_there_is_no_error_when_the_regimes_are_the_same(self) -> None:
        # The averaged figure is only wrong because the regimes differ. Two
        # identical components make the two calculations the same question.
        same = NormalMixture.of([(0.3, 0.2, 1.4), (0.7, 0.2, 1.4)])
        got = mixture_risk(same, 0.99)
        assert got.averaging_error == pytest.approx(0.0, abs=1e-12)

    def test_responsibilities_sum_to_one_and_follow_the_threshold(self) -> None:
        for at in (-2.0, 0.0, 1.0, 4.0, 12.0):
            shares = responsibilities(STRESSED, at)
            assert math.fsum(shares) == pytest.approx(1.0, abs=1e-14)
        climbing = [responsibilities(STRESSED, at)[1] for at in (0.0, 1.0, 2.0, 4.0, 8.0)]
        assert climbing == sorted(climbing)
        assert climbing[0] == pytest.approx(0.05, abs=0.002)
        assert climbing[-1] > 0.9999

    def test_responsibility_against_the_definition(self) -> None:
        # P(component | L > x) is the component's own tail mass over the
        # mixture's, written out here rather than reused.
        at = 3.0
        calm, stress = STRESSED.components
        total = calm.weight * calm.survival(at) + stress.weight * stress.survival(at)
        assert responsibilities(STRESSED, at)[1] == pytest.approx(
            stress.weight * stress.survival(at) / total, rel=1e-14
        )

    def test_a_single_component_agrees_with_the_normal(self) -> None:
        single = NormalMixture.of([(1.0, 0.0, 1.0)])
        got = mixture_risk(single, 0.975)
        assert got.value_at_risk == pytest.approx(normal_ppf(0.975), rel=1e-12)
        assert got.averaging_error == pytest.approx(0.0, abs=1e-14)
        assert got.responsibility == (1.0,)


class TestAsymptoticallyNormal:
    """Fat where it is measured, thin in the limit."""

    PUBLISHED: ClassVar[dict[float, float]] = {
        0.99: 1.6278,
        1 - 1e-4: 1.1014,
        1 - 1e-6: 1.0537,
        1 - 1e-12: 1.0220,
    }

    @pytest.mark.parametrize("confidence", sorted(PUBLISHED))
    def test_the_published_ratios(self, confidence: float) -> None:
        assert tail_ratio(STRESSED, confidence) == pytest.approx(
            self.PUBLISHED[confidence], abs=5e-4
        )

    def test_a_normals_ratio_does_not_depend_on_its_scale(self) -> None:
        narrow = tail_ratio(NormalMixture.of([(1.0, 0.0, 0.3)]), 0.99)
        wide = tail_ratio(NormalMixture.of([(1.0, 0.0, 400.0)]), 0.99)
        assert narrow == pytest.approx(wide, rel=1e-12)
        assert narrow == pytest.approx(1.1457, abs=1e-4)

    def test_the_excess_decays_towards_zero(self) -> None:
        excess = []
        for confidence in (0.99, 1 - 1e-4, 1 - 1e-6, 1 - 1e-12):
            normal = tail_ratio(NormalMixture.of([(1.0, 0.0, 1.0)]), confidence)
            excess.append(tail_ratio(STRESSED, confidence) / normal - 1.0)
        assert excess == sorted(excess, reverse=True)
        assert excess[0] == pytest.approx(0.421, abs=0.002)
        assert excess[-1] < 0.005


class TestSums:
    """Convolution, and what it says about subadditivity."""

    def test_against_a_numerical_convolution(self) -> None:
        left = NormalMixture.of([(0.7, -0.3, 1.0), (0.3, 1.0, 2.0)])
        right = NormalMixture.of([(0.4, 0.1, 0.5), (0.6, -0.2, 3.0)])
        joint = convolve(left, right)
        assert len(joint.components) == 4
        for at in (-4.0, -1.0, 0.0, 2.5, 7.0):

            def integrand(x: float, at: float = at) -> float:
                return left.pdf(x) * right.pdf(at - x)

            want = integrate(integrand, -70.0, 70.0, 40000)
            assert joint.pdf(at) == pytest.approx(want, abs=1e-12)

    def test_the_moments_add(self) -> None:
        left = NormalMixture.of([(0.7, -0.3, 1.0), (0.3, 1.0, 2.0)])
        right = NormalMixture.of([(0.4, 0.1, 0.5), (0.6, -0.2, 3.0)])
        joint = convolve(left, right)
        assert joint.mean == pytest.approx(left.mean + right.mean, abs=1e-14)
        assert joint.variance == pytest.approx(left.variance + right.variance, rel=1e-14)
        # Third cumulants add under independence; fourth do not without the
        # correction, so the central moments themselves must not be asserted.
        assert joint.central_moment(3) == pytest.approx(
            left.central_moment(3) + right.central_moment(3), abs=1e-12
        )

    def test_the_weight_floor_keeps_a_long_chain_finite(self) -> None:
        running = NormalMixture.of([(0.5, 0.0, 1.0), (0.5, 0.0, 2.0)])
        for _ in range(12):
            running = convolve(running, STRESSED, negligible=1e-10)
        # Twelve doublings from two components is 8192; the floor removes
        # the ones that cannot affect any number computed from the result.
        assert len(running.components) == 6604
        assert math.fsum(one.weight for one in running.components) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_value_at_risk_is_not_subadditive(self) -> None:
        position = NormalMixture.of([(0.991, 0.0, 0.01), (0.009, 100.0, 0.01)])
        assert position.quantile(0.99) == pytest.approx(0.031, abs=0.002)
        assert convolve(position, position).quantile(0.99) == pytest.approx(
            99.998, abs=0.002
        )
        assert diversification_gap(position, position, 0.99) == pytest.approx(
            99.936, abs=0.01
        )

    def test_expected_shortfall_is_subadditive_where_value_at_risk_is_not(self) -> None:
        position = NormalMixture.of([(0.991, 0.0, 0.01), (0.009, 100.0, 0.01)])
        joint = mixture_risk(convolve(position, position), 0.99).expected_shortfall
        alone = mixture_risk(position, 0.99).expected_shortfall
        assert joint <= 2.0 * alone + 1e-9

    def test_the_window_the_violation_lives_in(self) -> None:
        def gap(weight: float) -> float:
            position = NormalMixture.of(
                [(1.0 - weight, 0.0, 0.01), (weight, 100.0, 0.01)]
            )
            return diversification_gap(position, position, 0.99)

        def bisect(low: float, high: float, want_positive_above: bool) -> float:
            for _ in range(200):
                middle = 0.5 * (low + high)
                inside = gap(middle) > 0.0
                if inside == want_positive_above:
                    high = middle
                else:
                    low = middle
            return 0.5 * (low + high)

        lower = bisect(0.002, 0.009, True)
        upper = bisect(0.009, 0.02, False)
        assert lower == pytest.approx(0.004943, abs=2e-6)
        assert upper == pytest.approx(0.010000, abs=2e-6)
        # The upper edge is the tail probability itself, and the lower one is
        # where the pair's chance of at least one stress reaches it.
        assert upper == pytest.approx(1.0 - 0.99, abs=1e-5)
        assert lower == pytest.approx(1.0 - math.sqrt(0.99), abs=1e-4)
        assert gap(0.5 * (lower + upper)) > 90.0
        assert gap(lower - 1e-4) < 0.0
        assert gap(upper + 1e-4) < 0.0


class TestHorizons:
    """The occupation-time reduction, against enumerating the paths."""

    @pytest.mark.parametrize("periods", [1, 2, 3, 6, 9])
    def test_against_a_brute_force_enumeration(self, periods: int) -> None:
        persistence, start = (0.8, 0.6), (0.7, 0.3)
        law = occupation_law(persistence, start, periods)
        brute = [0.0] * (periods + 1)
        for path in range(2 ** periods):
            states = [(path >> step) & 1 for step in range(periods)]
            mass = start[states[0]]
            for before, after in pairwise(states):
                stay = persistence[before]
                mass *= stay if after == before else 1.0 - stay
            brute[sum(1 for state in states if state == 0)] += mass
        for count in range(periods + 1):
            assert law.probability[count] == pytest.approx(brute[count], abs=1e-14)
        assert math.fsum(law.probability) == pytest.approx(1.0, abs=1e-14)

    def test_the_expected_occupation_of_a_stationary_start(self) -> None:
        # Stay-calm 0.98 and stay-stressed 0.90 has a 5/6 calm stationary
        # share, so a chain started there spends 5/6 of any horizon calm.
        law = occupation_law((0.98, 0.90), (5.0 / 6.0, 1.0 / 6.0), 30)
        assert law.expected == pytest.approx(25.0, rel=1e-12)

    def test_the_horizon_law_against_the_paths(self) -> None:
        calm, stress = Component(1.0, 0.0, 1.0), Component(1.0, 0.5, 4.0)
        persistence, start, periods = (0.9, 0.7), (0.6, 0.4), 8
        built = horizon_mixture(calm, stress, persistence, start, periods)
        assert len(built.components) == periods + 1
        total = 0.0
        variance = 0.0
        for path in range(2 ** periods):
            states = [(path >> step) & 1 for step in range(periods)]
            mass = start[states[0]]
            for before, after in pairwise(states):
                stay = persistence[before]
                mass *= stay if after == before else 1.0 - stay
            parts = [calm if state == 0 else stress for state in states]
            mean = sum(one.mean for one in parts)
            spread = sum(one.deviation**2 for one in parts)
            total += mass * mean
            variance += mass * (spread + mean**2)
        assert built.mean == pytest.approx(total, abs=1e-13)
        assert built.variance == pytest.approx(variance - total**2, rel=1e-12)

    def test_the_weights_ignore_the_components_own(self) -> None:
        # The chain decides which regime is in force, so a component's weight
        # is not an input here and changing it must not change the answer.
        chain, start = (0.95, 0.8), (0.5, 0.5)
        one = horizon_mixture(Component(1.0, 0.0, 1.0), Component(1.0, 0.0, 3.0), chain, start, 7)
        other = horizon_mixture(
            Component(0.01, 0.0, 1.0), Component(0.99, 0.0, 3.0), chain, start, 7
        )
        assert one.variance == pytest.approx(other.variance, rel=1e-15)


class TestTheSquareRootRule:
    """Which way it is wrong, and where it changes its mind."""

    CHAIN = (0.98, 0.90)
    START = (5.0 / 6.0, 1.0 / 6.0)
    CALM = Component(1.0, 0.0, 1.0)
    STRESS = Component(1.0, 0.0, 4.0)

    def scaled(self, confidence: float, periods: int) -> float:
        one = mixture_risk(
            horizon_mixture(self.CALM, self.STRESS, self.CHAIN, self.START, 1), confidence
        ).value_at_risk
        many = mixture_risk(
            horizon_mixture(self.CALM, self.STRESS, self.CHAIN, self.START, periods),
            confidence,
        ).value_at_risk
        return many / (one * math.sqrt(periods))

    @pytest.mark.parametrize(
        ("confidence", "periods", "want"),
        [
            (0.99, 5, 0.958),
            (0.99, 10, 0.924),
            (0.99, 60, 0.797),
            (0.90, 5, 1.062),
            (0.90, 10, 1.121),
            (0.90, 60, 1.349),
        ],
    )
    def test_the_published_ratios(
        self, confidence: float, periods: int, want: float
    ) -> None:
        assert self.scaled(confidence, periods) == pytest.approx(want, abs=0.001)

    @pytest.mark.parametrize("periods", [5, 10, 60])
    def test_the_direction_changes_at_a_level_worth_knowing(self, periods: int) -> None:
        low, high = 0.90, 0.995
        for _ in range(80):
            middle = 0.5 * (low + high)
            if self.scaled(middle, periods) > 1.0:
                low = middle
            else:
                high = middle
        crossing = 0.5 * (low + high)
        assert 0.969 < crossing < 0.975
        expected = {5: 0.97410, 10: 0.97319, 60: 0.96996}[periods]
        assert crossing == pytest.approx(expected, abs=2e-4)

    def test_aggregation_diversifies_the_regime(self) -> None:
        kurtosis = [
            horizon_mixture(
                self.CALM, self.STRESS, self.CHAIN, self.START, periods
            ).excess_kurtosis
            for periods in (1, 5, 10, 20, 60)
        ]
        assert kurtosis == sorted(kurtosis, reverse=True)
        assert kurtosis[0] == pytest.approx(7.652, abs=0.002)
        assert kurtosis[-1] == pytest.approx(1.739, abs=0.002)

    @pytest.mark.parametrize(
        ("stay_calm", "stay_stressed", "want"),
        [(0.90, 0.50, 0.792), (0.98, 0.90, 0.924), (0.999, 0.995, 0.995)],
    )
    def test_persistence_decides_how_wrong_it_is(
        self, stay_calm: float, stay_stressed: float, want: float
    ) -> None:
        chain = (stay_calm, stay_stressed)
        one = mixture_risk(
            horizon_mixture(self.CALM, self.STRESS, chain, self.START, 1), 0.99
        ).value_at_risk
        ten = mixture_risk(
            horizon_mixture(self.CALM, self.STRESS, chain, self.START, 10), 0.99
        ).value_at_risk
        assert ten / (one * math.sqrt(10.0)) == pytest.approx(want, abs=0.001)


class TestFitting:
    """The likelihood has no maximum, and the estimator says so."""

    def test_the_likelihood_is_unbounded(self) -> None:
        sample = draw(NormalMixture.of([(1.0, 0.0, 1.0)]), 40, seed=5)
        climbing = []
        for deviation in (1.0, 1e-2, 1e-4, 1e-8, 1e-12):
            spike = NormalMixture.of([(0.975, 0.0, 1.0), (0.025, sample[0], deviation)])
            climbing.append(math.fsum(math.log(spike.pdf(one)) for one in sample))
        assert climbing == sorted(climbing)
        assert climbing[0] == pytest.approx(-55.478, abs=0.002)
        assert climbing[-1] == pytest.approx(-32.498, abs=0.002)

    def test_a_degenerate_fit_depends_on_the_floor(self) -> None:
        sample = draw(NormalMixture.of([(1.0, 0.0, 1.0)]), 40, seed=5)
        loose = fit_mixture(sample, 9, floor=1e-8)
        tight = fit_mixture(sample, 9, floor=1e-14)
        assert loose.degenerate and tight.degenerate
        # The giveaway: the same data and the same code, and the reported
        # quality of the fit is a property of the safeguard.
        assert tight.log_likelihood > loose.log_likelihood + 10.0
        assert loose.log_likelihood == pytest.approx(-17.80, abs=0.02)
        assert tight.log_likelihood == pytest.approx(-3.99, abs=0.02)

    def test_an_identified_fit_is_not_flagged(self) -> None:
        sample = draw(NormalMixture.of([(1.0, 0.0, 1.0)]), 40, seed=5)
        honest = fit_mixture(sample, 2)
        assert not honest.degenerate
        assert honest.converged
        assert honest.log_likelihood == pytest.approx(-50.440, abs=0.005)
        assert min(one.deviation for one in honest.mixture.components) == pytest.approx(
            0.1768, abs=0.0005
        )

    def test_it_recovers_a_mixture_it_was_given(self) -> None:
        truth = NormalMixture.of([(0.85, 0.0, 1.0), (0.15, 2.0, 3.0)])
        sample = draw(truth, 20000, seed=11)
        fitted = fit_mixture(sample, 2)
        assert fitted.converged
        assert not fitted.degenerate
        want = mixture_risk(truth, 0.99).expected_shortfall
        got = mixture_risk(fitted.mixture, 0.99).expected_shortfall
        assert abs(got / want - 1.0) < 0.02

    def test_a_small_sample_is_worse_and_still_the_right_shape(self) -> None:
        truth = NormalMixture.of([(0.85, 0.0, 1.0), (0.15, 2.0, 3.0)])
        fitted = fit_mixture(draw(truth, 200, seed=11), 2)
        want = mixture_risk(truth, 0.99).expected_shortfall
        got = mixture_risk(fitted.mixture, 0.99).expected_shortfall
        assert 0.02 < abs(got / want - 1.0) < 0.12

    def test_it_is_deterministic(self) -> None:
        sample = draw(NormalMixture.of([(0.6, 0.0, 1.0), (0.4, 3.0, 1.0)]), 400, seed=2)
        first, second = fit_mixture(sample, 2), fit_mixture(sample, 2)
        assert first == second

    def test_one_component_is_the_sample_moments(self) -> None:
        sample = draw(NormalMixture.of([(1.0, 1.0, 2.0)]), 500, seed=4)
        fitted = fit_mixture(sample, 1)
        mean = math.fsum(sample) / len(sample)
        variance = math.fsum((one - mean) ** 2 for one in sample) / len(sample)
        assert fitted.mixture.components[0].mean == pytest.approx(mean, rel=1e-12)
        assert fitted.mixture.components[0].deviation == pytest.approx(
            math.sqrt(variance), rel=1e-12
        )


class TestRefusals:
    """What the module declines to answer."""

    def test_a_negative_weight(self) -> None:
        with pytest.raises(BadMixture, match="weight is a probability"):
            Component(-0.1, 0.0, 1.0)

    def test_a_zero_deviation(self) -> None:
        with pytest.raises(BadMixture, match="standard deviation is positive"):
            Component(0.5, 0.0, 0.0)

    def test_an_infinite_mean(self) -> None:
        with pytest.raises(BadMixture, match="mean must be finite"):
            Component(0.5, math.inf, 1.0)

    def test_no_components(self) -> None:
        with pytest.raises(BadMixture, match="at least one component"):
            NormalMixture(())

    def test_weights_that_cannot_be_normalised(self) -> None:
        with pytest.raises(BadMixture, match="no distribution to normalise"):
            NormalMixture.of([(0.0, 0.0, 1.0), (0.0, 1.0, 1.0)])

    def test_weights_are_normalised_rather_than_refused(self) -> None:
        built = NormalMixture.of([(2.0, 0.0, 1.0), (6.0, 1.0, 1.0)])
        assert built.components[0].weight == pytest.approx(0.25, rel=1e-15)
        assert built.components[1].weight == pytest.approx(0.75, rel=1e-15)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 2.0])
    def test_a_confidence_outside_the_unit_interval(self, confidence: float) -> None:
        with pytest.raises(BadMixture, match="strictly between zero and one"):
            mixture_risk(STRESSED, confidence)
        with pytest.raises(BadMixture, match="strictly between zero and one"):
            averaged_shortfall(STRESSED, confidence)

    def test_a_quantile_at_an_endpoint(self) -> None:
        with pytest.raises(BadMixture, match="strictly between zero and one"):
            STRESSED.quantile(1.0)

    def test_a_stressed_weight_outside_the_unit_interval(self) -> None:
        with pytest.raises(BadMixture, match="is a probability"):
            NormalMixture.two_regime(stress=1.5)

    def test_a_sample_too_small_to_identify(self) -> None:
        with pytest.raises(BadMixture, match="cannot identify"):
            fit_mixture([0.1, 0.2, 0.3], 2)

    def test_a_constant_sample(self) -> None:
        with pytest.raises(BadMixture, match="every observation is identical"):
            fit_mixture([1.0] * 30, 2)

    def test_a_sample_with_a_hole_in_it(self) -> None:
        with pytest.raises(BadMixture, match="not finite"):
            fit_mixture([0.1, 0.2, math.nan, 0.4, 0.5, 0.6], 2)

    def test_a_horizon_of_nothing(self) -> None:
        with pytest.raises(BadMixture, match="at least one period"):
            occupation_law((0.9, 0.9), (0.5, 0.5), 0)

    def test_a_starting_distribution_that_is_not_one(self) -> None:
        with pytest.raises(BadMixture, match="starting distribution sums"):
            occupation_law((0.9, 0.9), (0.5, 0.6), 4)

    def test_a_transition_outside_the_unit_interval(self) -> None:
        with pytest.raises(BadMixture, match="transition probability"):
            occupation_law((1.4, 0.9), (0.5, 0.5), 4)

    def test_a_negative_scale(self) -> None:
        with pytest.raises(BadMixture, match="positive factor"):
            STRESSED.scaled(-1.0)

    def test_a_moment_of_negative_order(self) -> None:
        with pytest.raises(BadMixture, match="not negative"):
            STRESSED.central_moment(-2)

    def test_a_threshold_nothing_can_reach(self) -> None:
        with pytest.raises(BadMixture, match="cannot be attributed"):
            responsibilities(STRESSED, 1e6)


class TestTheStationaryStart:
    """The default start, and what starting elsewhere does."""

    def test_it_is_the_fixed_point(self) -> None:
        chain = (0.98, 0.90)
        share = stationary(chain)
        forward = (
            share[0] * chain[0] + share[1] * (1.0 - chain[1]),
            share[0] * (1.0 - chain[0]) + share[1] * chain[1],
        )
        assert forward == pytest.approx(share, abs=1e-15)
        assert share[0] == pytest.approx(5.0 / 6.0, rel=1e-15)

    def test_the_occupation_share_settles_there(self) -> None:
        # Started away from the stationary distribution, the running share
        # approaches it from below and the shortfall decays as 1/n: the
        # transient is a fixed amount of time spent in the wrong state, and
        # averaging it over more periods is the only thing that dilutes it.
        chain = (0.98, 0.90)
        gaps = [
            (5.0 / 6.0 - occupation_law(chain, (0.2, 0.8), periods).expected / periods)
            * periods
            for periods in (50, 100, 200, 400)
        ]
        for one in gaps:
            assert one == pytest.approx(5.28, abs=0.02)

    def test_an_absorbing_chain_has_no_single_one(self) -> None:
        with pytest.raises(BadMixture, match="never leaves either state"):
            stationary((1.0, 1.0))

    @pytest.mark.parametrize(
        ("stressed_mean", "want"), [(0.0, 0.9235), (0.5, 1.0427), (2.0, 1.3608)]
    )
    def test_a_regime_mean_changes_the_direction(
        self, stressed_mean: float, want: float
    ) -> None:
        chain = (0.98, 0.90)
        start = stationary(chain)
        calm = Component(1.0, 0.0, 1.0)
        stressed = Component(1.0, stressed_mean, 4.0)
        one = mixture_risk(
            horizon_mixture(calm, stressed, chain, start, 1), 0.99
        ).value_at_risk
        ten = mixture_risk(
            horizon_mixture(calm, stressed, chain, start, 10), 0.99
        ).value_at_risk
        assert ten / (one * math.sqrt(10.0)) == pytest.approx(want, abs=0.001)

    def test_where_the_mean_makes_the_rule_exact(self) -> None:
        chain = (0.98, 0.90)
        start = stationary(chain)

        def ratio(stressed_mean: float) -> float:
            calm = Component(1.0, 0.0, 1.0)
            stressed = Component(1.0, stressed_mean, 4.0)
            one = mixture_risk(
                horizon_mixture(calm, stressed, chain, start, 1), 0.99
            ).value_at_risk
            ten = mixture_risk(
                horizon_mixture(calm, stressed, chain, start, 10), 0.99
            ).value_at_risk
            return ten / (one * math.sqrt(10.0))

        low, high = 0.0, 2.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            if ratio(middle) < 1.0:
                low = middle
            else:
                high = middle
        assert 0.5 * (low + high) == pytest.approx(0.3189, abs=5e-4)

"""Expectiles: the definition, the score that elicits them, and what they cost.

Three groups carry the argument.

:class:`TestConsistency` is the one that matters for the claim. The asymmetric
squared loss is asserted to be strictly consistent for the expectile by
minimising its *exact* expected value under a normal law on a grid, rather than
by simulating and hoping. The same method :mod:`tests.test_scoring` uses on the
Fissler-Ziegel loss, for the same reason: a simulation of a consistency claim
measures the simulation.

:class:`TestExactSampleEstimator` checks that the closed-form sample expectile
is the minimiser of the sample score, by minimising the sample score
independently and comparing. If the interval scan picked the wrong piece, the
two would differ and nothing else in the suite would notice.

:class:`TestWhatItCosts` holds the measurements. A level calibrated on one law
does not transfer to another; subadditivity fails below a half; comonotonic
additivity fails everywhere but a half. Each is a number in the source
docstrings and each is asserted here, so a change that moves one has to move
the documentation with it.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.distributions import normal_pdf
from shortfall.expectile import (
    BadLevel,
    Expectile,
    asymmetric_squared_loss,
    comonotonic_gap,
    expectile_identity,
    matching_level,
    normal_expectile,
    sample_expectile,
    student_t_expectile,
    subadditivity_gap,
)
from shortfall.parametric import normal_risk, student_t_risk

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999]


def _sample(count: int, seed: int, *, scale: float = 1.0, shift: float = 0.0) -> list[float]:
    generator = random.Random(seed)
    return [shift + scale * generator.gauss(0.0, 1.0) for _ in range(count)]


def _expected_score(forecast: float, level: float, sigma: float, nodes: int = 40001) -> float:
    """``E[S(forecast, L)]`` for ``L ~ N(0, sigma)``, by quadrature on the density.

    Not a simulation. The integrand is a weighted parabola against a normal
    density, smooth away from the single kink at the forecast, and the kink is
    placed on a node so that the rule does not straddle it.
    """
    reach = 12.0 * sigma
    # Put the kink on a node: split the interval at the forecast and use an
    # even number of panels either side, so Simpson's rule never spans it.
    total = 0.0
    for low, high in ((-reach, forecast), (forecast, reach)):
        panels = nodes // 2 * 2
        width = (high - low) / panels
        if width == 0.0:
            continue
        running = 0.0
        for index in range(panels + 1):
            point = low + index * width
            weight = 1.0 if index in (0, panels) else (4.0 if index % 2 else 2.0)
            density = normal_pdf(point / sigma) / sigma
            running += weight * asymmetric_squared_loss(forecast, point, level) * density
        total += running * width / 3.0
    return total


class TestDefinition:
    """The condition, and the things it forces."""

    def test_the_half_expectile_is_the_mean(self) -> None:
        for mean, sigma in ((0.0, 1.0), (0.004, 0.02), (-0.01, 0.5)):
            analytic = normal_expectile(mean=mean, volatility=sigma, level=0.5)
            assert analytic.value == mean
            assert analytic.identity == 0.0

    def test_the_half_expectile_of_a_sample_is_its_mean(self) -> None:
        for seed in range(5):
            losses = _sample(500, seed)
            found = sample_expectile(losses, 0.5)
            assert found.value == pytest.approx(math.fsum(losses) / len(losses), rel=1e-14)

    @pytest.mark.parametrize("level", LEVELS)
    def test_the_defining_condition_holds(self, level: float) -> None:
        analytic = normal_expectile(mean=0.001, volatility=0.02, level=level)
        assert analytic.identity == pytest.approx(0.0, abs=1e-12)
        empirical = sample_expectile(_sample(800, 3, scale=0.02), level)
        assert empirical.identity == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("level", LEVELS)
    def test_the_exceedance_ratio_is_fixed_by_the_level(self, level: float) -> None:
        """``E[(L-e)+] / E[(e-L)+] = (1-tau)/tau``, which is the reading to give.

        An expectile is not a quantile of anything, so "the loss exceeded 2.5%
        of the time" has no analogue. This is the sentence that does: the point
        at which expected overshoot and expected undershoot stand in a fixed
        ratio.
        """
        analytic = normal_expectile(mean=0.0, volatility=0.02, level=level)
        assert analytic.exceedance_ratio == pytest.approx((1.0 - level) / level, rel=1e-9)

    @pytest.mark.parametrize("level", LEVELS)
    def test_location_and_scale_pass_straight_through(self, level: float) -> None:
        base = normal_expectile(mean=0.0, volatility=1.0, level=level)
        moved = normal_expectile(mean=0.03, volatility=0.25, level=level)
        assert moved.value == pytest.approx(0.03 + 0.25 * base.value, rel=1e-12)

    def test_the_expectile_rises_with_the_level(self) -> None:
        values = [normal_expectile(mean=0.0, volatility=0.02, level=t).value for t in LEVELS]
        assert values == sorted(values)
        losses = _sample(600, 17, scale=0.02)
        empirical = [sample_expectile(losses, t).value for t in LEVELS]
        assert empirical == sorted(empirical)

    def test_the_identity_residual_is_decreasing_in_the_candidate(self) -> None:
        losses = _sample(300, 9)
        found = sample_expectile(losses, 0.9).value
        grid = [found - 1.0, found - 0.1, found, found + 0.1, found + 1.0]
        residuals = [expectile_identity(point, losses, 0.9) for point in grid]
        assert residuals == sorted(residuals, reverse=True)
        assert residuals[2] == pytest.approx(0.0, abs=1e-15)

    def test_a_degenerate_sample_gives_its_one_value(self) -> None:
        for level in (0.1, 0.5, 0.99):
            found = sample_expectile([0.03] * 7, level)
            assert found.value == pytest.approx(0.03)
            assert found.upper == 0.0
            assert found.lower == 0.0
            assert found.exceedance_ratio == math.inf


class TestConsistency:
    """Strict consistency, against the exact expected score rather than a simulation."""

    @pytest.mark.parametrize("level", [0.75, 0.9, 0.975])
    def test_the_exact_expected_score_is_minimised_at_the_expectile(
        self, level: float
    ) -> None:
        sigma = 1.0
        truth = normal_expectile(mean=0.0, volatility=sigma, level=level).value
        at_truth = _expected_score(truth, level, sigma)
        for offset in (-0.4, -0.1, -0.01, 0.01, 0.1, 0.4):
            assert _expected_score(truth + offset, level, sigma) > at_truth

    def test_the_minimiser_of_the_exact_score_is_the_expectile(self) -> None:
        """Found by search rather than assumed, to the resolution of the search."""
        level, sigma = 0.9, 1.0
        truth = normal_expectile(mean=0.0, volatility=sigma, level=level).value
        low, high = truth - 0.5, truth + 0.5
        for _ in range(40):
            left = low + (high - low) / 3.0
            right = high - (high - low) / 3.0
            if _expected_score(left, level, sigma, nodes=4001) < _expected_score(
                right, level, sigma, nodes=4001
            ):
                high = right
            else:
                low = left
        assert 0.5 * (low + high) == pytest.approx(truth, abs=1e-4)

    def test_the_score_weights_the_two_sides_as_documented(self) -> None:
        assert asymmetric_squared_loss(1.0, 3.0, 0.9) == pytest.approx(0.9 * 4.0)
        assert asymmetric_squared_loss(1.0, -1.0, 0.9) == pytest.approx(0.1 * 4.0)
        # At a half it is a halved squared error, and so elicits the mean.
        assert asymmetric_squared_loss(1.0, 3.0, 0.5) == pytest.approx(2.0)
        assert asymmetric_squared_loss(1.0, -1.0, 0.5) == pytest.approx(2.0)

    def test_the_score_is_zero_only_on_a_perfect_forecast(self) -> None:
        assert asymmetric_squared_loss(0.5, 0.5, 0.9) == 0.0
        assert asymmetric_squared_loss(0.5, 0.5000001, 0.9) > 0.0

    def test_a_truthful_forecaster_beats_a_shaded_one_on_average(self) -> None:
        """The property the score exists for, on a sample large enough to show it."""
        level, sigma = 0.95, 0.02
        truth = normal_expectile(mean=0.0, volatility=sigma, level=level).value
        losses = _sample(40000, 404, scale=sigma)
        honest = math.fsum(asymmetric_squared_loss(truth, loss, level) for loss in losses)
        for shade in (0.8, 0.9, 1.1, 1.25):
            shaded = math.fsum(
                asymmetric_squared_loss(truth * shade, loss, level) for loss in losses
            )
            assert shaded > honest


class TestExactSampleEstimator:
    """The interval scan, against an independent minimisation of the same score."""

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("count", [1, 2, 7, 50, 301])
    def test_it_minimises_the_sample_score(self, level: float, count: int) -> None:
        losses = _sample(count, count * 31 + 7)
        found = sample_expectile(losses, level).value

        def total(forecast: float) -> float:
            return math.fsum(asymmetric_squared_loss(forecast, loss, level) for loss in losses)

        low, high = min(losses) - 2.0, max(losses) + 2.0
        for _ in range(200):
            left = low + (high - low) / 3.0
            right = high - (high - low) / 3.0
            if total(left) < total(right):
                high = right
            else:
                low = left
        # 1e-7 and not tighter, because the *reference* cannot do better. The
        # score is quadratic at its minimum, so two forecasts a distance d
        # apart differ in score by O(d^2), and once d falls to the square root
        # of machine epsilon the comparison the search depends on is noise.
        # The closed form has no such floor: its residual in the defining
        # condition is at the rounding level, around 1e-17, which is the
        # argument for computing it rather than searching for it.
        assert found == pytest.approx(0.5 * (low + high), abs=1e-7)
        assert expectile_identity(found, losses, level) == pytest.approx(0.0, abs=1e-14)

    def test_it_lands_between_the_order_statistics_that_bracket_it(self) -> None:
        losses = _sample(200, 77)
        ordered = sorted(losses)
        for level in LEVELS:
            found = sample_expectile(losses, level).value
            assert ordered[0] <= found <= ordered[-1]

    def test_ties_in_the_data_do_not_break_the_scan(self) -> None:
        losses = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        for level in LEVELS:
            found = sample_expectile(losses, level)
            assert found.identity == pytest.approx(0.0, abs=1e-15)
            assert 0.0 <= found.value <= 2.0

    def test_it_is_unbiased_and_attains_its_asymptotic_standard_error(self) -> None:
        """Against a prediction derived independently, not against "it got smaller".

        The estimating equation is ``E[psi(L, e)] = 0`` with
        ``psi(l, e) = |tau - 1{l <= e}| (l - e)``, so the sandwich formula gives
        an asymptotic standard error of ``sqrt(Var psi) / E[dpsi/de]`` over
        ``sqrt(n)``, both evaluated at the true expectile. For a standard
        normal at ``tau = 0.95`` that is ``1.4285 / sqrt(n)``, and the root
        mean squared error over independent samples has to match it.

        Every row uses its own seeds. One seed shared across the rows makes
        that draw's sampling error look like a systematic bias — an earlier
        version of this test did exactly that and read a 1.5-sigma draw at the
        largest sample size as a failure to converge.
        """
        level, sigma = 0.95, 1.0
        truth = normal_expectile(mean=0.0, volatility=sigma, level=level).value
        predicted = 1.428462
        for count, replications in ((500, 120), (4000, 120)):
            errors = []
            for index in range(replications):
                generator = random.Random(index * 7919 + count * 13)
                losses = [generator.gauss(0.0, sigma) for _ in range(count)]
                errors.append(sample_expectile(losses, level).value - truth)
            root_mean_square = math.sqrt(math.fsum(e * e for e in errors) / replications)
            expected = predicted / math.sqrt(count)
            assert root_mean_square == pytest.approx(expected, rel=0.2)
            # And no bias: the mean error is small against the spread of the
            # mean itself, which is the standard error over sqrt(replications).
            bias = math.fsum(errors) / replications
            assert abs(bias) < 3.0 * expected / math.sqrt(replications)

    def test_the_result_is_frozen(self) -> None:
        found = sample_expectile(_sample(20, 1), 0.9)
        assert isinstance(found, Expectile)
        with pytest.raises((AttributeError, TypeError)):
            found.value = 0.0  # type: ignore[misc]


class TestStudentT:
    """The fatter-tailed closed form, and its agreement with its own sample."""

    @pytest.mark.parametrize("level", [0.5, 0.9, 0.975, 0.99])
    @pytest.mark.parametrize("degrees", [3.0, 5.0, 10.0, 50.0])
    def test_the_condition_holds(self, level: float, degrees: float) -> None:
        found = student_t_expectile(
            mean=0.002, volatility=0.02, level=level, degrees=degrees
        )
        assert found.identity == pytest.approx(0.0, abs=1e-12)

    def test_the_half_expectile_is_still_the_mean(self) -> None:
        for degrees in (3.0, 5.0, 30.0):
            found = student_t_expectile(
                mean=0.002, volatility=0.02, level=0.5, degrees=degrees
            )
            assert found.value == pytest.approx(0.002, abs=1e-12)

    def test_a_fatter_tail_gives_a_larger_expectile(self) -> None:
        values = [
            student_t_expectile(mean=0.0, volatility=0.02, level=0.99, degrees=d).value
            for d in (50.0, 20.0, 10.0, 5.0, 3.0)
        ]
        assert values == sorted(values)
        assert values[0] > normal_expectile(mean=0.0, volatility=0.02, level=0.99).value

    def test_many_degrees_of_freedom_approach_the_normal(self) -> None:
        normal = normal_expectile(mean=0.0, volatility=0.02, level=0.975).value
        far = student_t_expectile(
            mean=0.0, volatility=0.02, level=0.975, degrees=5000.0
        ).value
        assert far == pytest.approx(normal, rel=2e-3)


class TestMatching:
    """Stating an expectile in the units a desk already uses."""

    @pytest.mark.parametrize("confidence", [0.95, 0.975, 0.99])
    def test_the_level_matching_an_expected_shortfall_is_found_exactly(
        self, confidence: float
    ) -> None:
        risk = normal_risk(mean=0.0, volatility=0.02, confidence=confidence)
        match = matching_level(
            risk.expected_shortfall,
            lambda t: normal_expectile(mean=0.0, volatility=0.02, level=t).value,
        )
        assert match.residual == pytest.approx(0.0, abs=1e-12)
        assert 0.5 < match.level < 1.0
        assert match.value == pytest.approx(risk.expected_shortfall, rel=1e-12)

    def test_the_normal_level_for_a_97_5_shortfall(self) -> None:
        """The number in the module docstring, asserted so it cannot drift."""
        risk = normal_risk(mean=0.0, volatility=0.02, confidence=0.975)
        match = matching_level(
            risk.expected_shortfall,
            lambda t: normal_expectile(mean=0.0, volatility=0.02, level=t).value,
        )
        assert match.level == pytest.approx(0.998603, abs=5e-6)

    def test_matching_on_a_sample_works_the_same_way(self) -> None:
        losses = _sample(2000, 55, scale=0.02)
        target = sample_expectile(losses, 0.97).value
        match = matching_level(target, lambda t: sample_expectile(losses, t).value)
        assert match.level == pytest.approx(0.97, abs=1e-6)

    def test_a_target_nothing_attains_is_refused(self) -> None:
        losses = _sample(100, 3)
        with pytest.raises(BadLevel, match="no level attains it"):
            matching_level(max(losses) + 10.0, lambda t: sample_expectile(losses, t).value)
        with pytest.raises(BadLevel, match="no level attains it"):
            matching_level(min(losses) - 10.0, lambda t: sample_expectile(losses, t).value)

    def test_a_non_finite_target_is_refused(self) -> None:
        with pytest.raises(BadLevel, match="target must be finite"):
            matching_level(math.nan, lambda t: t)


class TestWhatItCosts:
    """The three measurements. Each is a figure in the source docstrings."""

    def test_a_level_calibrated_on_one_law_does_not_transfer(self) -> None:
        """The headline: 0.0013 in ``tau`` is 16.5% in the risk number.

        This is the practical objection to expectiles as a capital measure. A
        confidence level is fixed once by a rule and means the same thing on
        every book. An expectile level does not: the same ``tau`` is a
        different amount of conservatism on a fatter tail, so it has to be
        recalibrated per distribution, and the recalibration is exactly the
        model assumption the exercise was supposed to avoid.
        """
        sigma = 0.02
        normal = matching_level(
            normal_risk(mean=0.0, volatility=sigma, confidence=0.975).expected_shortfall,
            lambda t: normal_expectile(mean=0.0, volatility=sigma, level=t).value,
        ).level
        overstatement = {}
        for degrees in (5.0, 8.0, 20.0):
            truth = student_t_risk(
                mean=0.0, volatility=sigma, confidence=0.975, degrees=degrees
            ).expected_shortfall
            transplanted = student_t_expectile(
                mean=0.0, volatility=sigma, level=normal, degrees=degrees
            ).value
            overstatement[degrees] = transplanted / truth - 1.0
        assert overstatement[5.0] == pytest.approx(0.165, abs=0.005)
        assert overstatement[8.0] == pytest.approx(0.0788, abs=0.004)
        assert overstatement[20.0] == pytest.approx(0.0241, abs=0.003)
        # Monotone in the tail thickness, which is the shape of the problem: the
        # transplant is worst exactly where the measure was meant to help.
        assert overstatement[5.0] > overstatement[8.0] > overstatement[20.0] > 0.0

    def test_subadditivity_holds_above_a_half_and_fails_below(self) -> None:
        generator = random.Random(11)
        first = [generator.gauss(0.0, 1.0) for _ in range(4000)]
        second = [0.3 * value + 0.95 * generator.gauss(0.0, 1.0) for value in first]
        for level in (0.6, 0.9, 0.99):
            assert subadditivity_gap(first, second, level) > 0.0
        for level in (0.05, 0.2, 0.4, 0.49):
            assert subadditivity_gap(first, second, level) < 0.0
        # Exactly zero at a half, because the half-expectile is the mean and
        # the mean is additive. That is the boundary, not an approximation to it.
        assert subadditivity_gap(first, second, 0.5) == pytest.approx(0.0, abs=1e-14)

    def test_the_violation_grows_with_the_distance_below_a_half(self) -> None:
        generator = random.Random(404)
        first = [generator.gauss(0.0, 1.0) for _ in range(2000)]
        second = [-0.6 * value + 0.8 * generator.gauss(0.0, 1.0) for value in first]
        gaps = [subadditivity_gap(first, second, t) for t in (0.45, 0.3, 0.15, 0.05)]
        assert all(gap < 0.0 for gap in gaps)
        assert gaps == sorted(gaps, reverse=True)

    def test_expectiles_are_not_comonotonically_additive(self) -> None:
        """The property traded away for elicitability, measured.

        Expected shortfall is comonotonically additive, so two positions that
        move together get no diversification credit. An expectile gives some
        anyway.
        """
        generator = random.Random(5)
        base = sorted(generator.gauss(0.0, 1.0) for _ in range(2000))
        transformed = [math.exp(0.5 * value) for value in base]
        assert comonotonic_gap(base, transformed, 0.5) == pytest.approx(0.0, abs=1e-13)
        for level, expected in ((0.9, 0.00399), (0.975, 0.00476), (0.99, 0.00688)):
            gap = comonotonic_gap(base, transformed, level)
            total = (
                sample_expectile(base, level).value
                + sample_expectile(transformed, level).value
            )
            assert gap > 0.0
            assert gap / total == pytest.approx(expected, abs=2e-4)

    def test_comonotonic_data_is_checked_rather_than_assumed(self) -> None:
        generator = random.Random(8)
        first = [generator.gauss(0.0, 1.0) for _ in range(200)]
        second = [-value for value in first]
        with pytest.raises(BadLevel, match="not comonotonic"):
            comonotonic_gap(first, second, 0.9)

    def test_an_expectile_sees_the_body_and_value_at_risk_does_not(self) -> None:
        """Mass moved inside the body leaves every upper quantile alone.

        Which is usually stated as an advantage of expectiles and is also the
        reason there is no loss level to point at when one is quoted.
        """
        lower = [-3.0] * 100 + [0.0] * 800 + [4.0] * 100
        tighter = [-1.0] * 100 + [0.0] * 800 + [4.0] * 100
        # Every observation at or above the 90th percentile is untouched, so
        # the 95% value at risk and expected shortfall are identical.
        assert sorted(lower)[900:] == sorted(tighter)[900:]
        assert sample_expectile(lower, 0.95).value != sample_expectile(tighter, 0.95).value
        assert sample_expectile(tighter, 0.95).value > sample_expectile(lower, 0.95).value


class TestValidation:
    """What gets refused."""

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf])
    def test_levels_outside_the_open_interval(self, level: float) -> None:
        with pytest.raises(BadLevel, match="strictly inside"):
            sample_expectile([0.1, 0.2], level)
        with pytest.raises(BadLevel, match="strictly inside"):
            normal_expectile(mean=0.0, volatility=1.0, level=level)
        with pytest.raises(BadLevel, match="strictly inside"):
            asymmetric_squared_loss(0.0, 0.0, level)

    def test_an_empty_sample(self) -> None:
        with pytest.raises(BadLevel, match="at least one observation"):
            sample_expectile([], 0.9)
        with pytest.raises(BadLevel, match="at least one observation"):
            expectile_identity(0.0, [], 0.9)

    def test_a_non_finite_observation(self) -> None:
        with pytest.raises(ValueError, match="every loss must be finite"):
            sample_expectile([0.1, math.nan, 0.2], 0.9)
        with pytest.raises(ValueError, match="every loss must be finite"):
            sample_expectile([0.1, math.inf], 0.9)

    @pytest.mark.parametrize("volatility", [0.0, -0.01, math.nan, math.inf])
    def test_a_bad_volatility(self, volatility: float) -> None:
        with pytest.raises(BadLevel, match="volatility"):
            normal_expectile(mean=0.0, volatility=volatility, level=0.9)
        with pytest.raises(BadLevel, match="volatility"):
            student_t_expectile(
                mean=0.0, volatility=volatility, level=0.9, degrees=5.0
            )

    def test_a_non_finite_mean(self) -> None:
        with pytest.raises(BadLevel, match="mean"):
            normal_expectile(mean=math.inf, volatility=1.0, level=0.9)
        with pytest.raises(BadLevel, match="mean"):
            student_t_expectile(mean=math.nan, volatility=1.0, level=0.9, degrees=5.0)

    @pytest.mark.parametrize("degrees", [2.0, 1.5, 0.0, -3.0, math.nan])
    def test_degrees_of_freedom_at_or_below_two(self, degrees: float) -> None:
        with pytest.raises(BadLevel, match="must exceed 2"):
            student_t_expectile(mean=0.0, volatility=0.02, level=0.9, degrees=degrees)

    def test_unpaired_samples(self) -> None:
        with pytest.raises(BadLevel, match="paired"):
            subadditivity_gap([0.1, 0.2], [0.1], 0.9)
        with pytest.raises(BadLevel, match="paired"):
            comonotonic_gap([0.1, 0.2], [0.1], 0.9)

    def test_a_non_finite_score_input(self) -> None:
        with pytest.raises(ValueError, match="forecast must be finite"):
            asymmetric_squared_loss(math.nan, 0.0, 0.9)
        with pytest.raises(ValueError, match="loss must be finite"):
            asymmetric_squared_loss(0.0, math.inf, 0.9)

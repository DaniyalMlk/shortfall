"""Scoring rules: consistency proved exactly, and the obvious score's failure.

A scoring function is worth something only if it is strictly consistent — the
expected score minimised, uniquely, at the true value. That is a statement
about an expectation, so for a normal law it can be checked *exactly* rather
than simulated: every expectation here is elementary, and the file computes
them in closed form and minimises them on a refining grid.

The useful half of the file is the negative result. The score anybody would
build first — the pinball loss for the quantile plus a squared error on the
breaches for the shortfall — is not consistent, its optimum sits about a third
above the truth, and it ranks a shaded forecaster above a truthful one. The
tests measure all three.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from statistics import NormalDist
from typing import ClassVar

import pytest

from shortfall.scoring import (
    BadScore,
    Scores,
    compare,
    fz0_loss,
    quantile_loss,
)

NORMAL = NormalDist()
LEVELS = (0.95, 0.975, 0.99)


def truth(confidence: float, sigma: float = 1.0) -> tuple[float, float]:
    """The value at risk and expected shortfall of a centred normal loss."""
    alpha = 1.0 - confidence
    z = NORMAL.inv_cdf(confidence)
    return sigma * z, sigma * NORMAL.pdf(z) / alpha


def expected_pinball(var: float, confidence: float, sigma: float = 1.0) -> float:
    """``E[(L - v)(c - 1{L < v})]`` for ``L ~ N(0, sigma)``."""
    z = var / sigma
    return -confidence * var + sigma * NORMAL.pdf(z) + var * NORMAL.cdf(z)


def expected_tail(var: float, sigma: float = 1.0) -> float:
    """``E[(L - v)+]``, which both joint scores are built out of."""
    z = var / sigma
    return sigma * NORMAL.pdf(z) + var * (NORMAL.cdf(z) - 1.0)


def expected_fz0(
    var: float, shortfall: float, confidence: float, sigma: float = 1.0
) -> float:
    alpha = 1.0 - confidence
    return (
        expected_tail(var, sigma) / (alpha * shortfall)
        + var / shortfall
        + math.log(shortfall)
        - 1.0
    )


def expected_obvious(
    var: float, shortfall: float, confidence: float, sigma: float = 1.0
) -> float:
    """Pinball for the quantile plus squared error on the breaches for the mean.

    The construction anybody reaches for first, written out exactly: the second
    term is ``E[1{L >= v}(L - e)**2] / alpha``, and every piece of it is a
    truncated normal moment.
    """
    alpha = 1.0 - confidence
    z = var / sigma
    tail_probability = 1.0 - NORMAL.cdf(z)
    second = sigma * sigma * (tail_probability + z * NORMAL.pdf(z))
    cross = -2.0 * shortfall * sigma * NORMAL.pdf(z)
    square = shortfall * shortfall * tail_probability
    return expected_pinball(var, confidence, sigma) + (second + cross + square) / alpha


def minimise(
    objective: Callable[[float, float], float],
    start: tuple[float, float],
    span: float = 0.9,
    rounds: int = 90,
    points: int = 41,
) -> tuple[float, float]:
    """Refine a grid around the running best. Deterministic, and good to 1e-8."""
    low_v, high_v = start[0] * (1.0 - span), start[0] * (1.0 + span)
    low_e, high_e = start[1] * (1.0 - span), start[1] * (1.0 + span)
    best = start
    for _ in range(rounds):
        grid_v = [low_v + (high_v - low_v) * i / (points - 1) for i in range(points)]
        grid_e = [low_e + (high_e - low_e) * i / (points - 1) for i in range(points)]
        value = math.inf
        for candidate_v in grid_v:
            for candidate_e in grid_e:
                if candidate_v <= 0.0 or candidate_e <= 0.0:
                    continue
                current = objective(candidate_v, candidate_e)
                if current < value:
                    value, best = current, (candidate_v, candidate_e)
        step_v = (high_v - low_v) / (points - 1)
        step_e = (high_e - low_e) / (points - 1)
        low_v, high_v = best[0] - step_v, best[0] + step_v
        low_e, high_e = best[1] - step_e, best[1] + step_e
        if step_v < 1e-12 and step_e < 1e-12:
            break
    return best


class TestConsistency:
    """The expected score, in closed form, minimised at the truth."""

    @pytest.mark.parametrize("confidence", LEVELS)
    def test_the_pinball_loss_elicits_the_quantile(self, confidence: float) -> None:
        """Its derivative is ``Phi(v/sigma) - c``, zero exactly at the quantile."""
        var, _ = truth(confidence)
        step = 1e-6
        slope = (
            expected_pinball(var + step, confidence)
            - expected_pinball(var - step, confidence)
        ) / (2.0 * step)
        assert slope == pytest.approx(0.0, abs=1e-9)
        # And it is a minimum, not a stationary point of another kind.
        assert expected_pinball(var, confidence) < expected_pinball(var * 1.1, confidence)
        assert expected_pinball(var, confidence) < expected_pinball(var * 0.9, confidence)

    @pytest.mark.parametrize("confidence", LEVELS)
    def test_fissler_ziegel_elicits_the_pair(self, confidence: float) -> None:
        var, shortfall = truth(confidence)
        found = minimise(
            lambda v, e: expected_fz0(v, e, confidence), (var * 1.4, shortfall * 0.7)
        )
        assert found[0] == pytest.approx(var, rel=1e-6)
        assert found[1] == pytest.approx(shortfall, rel=1e-6)

    @pytest.mark.parametrize("confidence", LEVELS)
    def test_its_first_order_conditions_are_the_definitions(
        self, confidence: float
    ) -> None:
        """Both conditions reduce to what the two quantities are."""
        alpha = 1.0 - confidence
        var, shortfall = truth(confidence)
        # d/dv: (Phi(v) - 1) / (alpha e) + 1 / e
        in_var = (NORMAL.cdf(var) - 1.0) / (alpha * shortfall) + 1.0 / shortfall
        assert in_var == pytest.approx(0.0, abs=1e-14)
        # d/de: 1/e - [E(L-v)+ / alpha + v] / e^2
        in_shortfall = (
            1.0 / shortfall
            - (expected_tail(var) / alpha + var) / (shortfall * shortfall)
        )
        assert in_shortfall == pytest.approx(0.0, abs=1e-14)
        # The second is the definition of the shortfall, rearranged.
        assert var + expected_tail(var) / alpha == pytest.approx(shortfall, rel=1e-14)

    @pytest.mark.parametrize("confidence", LEVELS)
    def test_rescaling_shifts_every_score_by_the_same_constant(
        self, confidence: float
    ) -> None:
        """Which is what makes the *ranking* scale free, not the score.

        Rescaling by ``k`` moves the score by exactly ``log k``, from the
        ``log e`` term, and that shift does not depend on the forecast. So
        every difference between two forecasts is invariant and the ordering
        cannot move when the currency does — which is the reason for this
        member of the family and is a weaker statement than the score itself
        being unchanged. The first draft of this test asserted the stronger
        one and failed by precisely ``log 1000``.
        """
        returns = [0.004, -0.019, 0.011, -0.031, 0.002, -0.0005]
        var, shortfall = truth(confidence, sigma=0.01)
        factor = 1000.0
        plain = fz0_loss(
            returns, [var] * len(returns), [shortfall] * len(returns), confidence=confidence
        )
        scaled = fz0_loss(
            [value * factor for value in returns],
            [var * factor] * len(returns),
            [shortfall * factor] * len(returns),
            confidence=confidence,
        )
        for first, second in zip(plain.values, scaled.values, strict=True):
            assert second - first == pytest.approx(math.log(factor), rel=1e-12)

        # And so a comparison is untouched: a second, worse model shifts by the
        # same constant and the difference between the two does not move.
        worse = [1.6 * var] * len(returns), [1.6 * shortfall] * len(returns)
        plain_worse = fz0_loss(returns, *worse, confidence=confidence)
        scaled_worse = fz0_loss(
            [value * factor for value in returns],
            [value * factor for value in worse[0]],
            [value * factor for value in worse[1]],
            confidence=confidence,
        )
        assert scaled_worse.mean - scaled.mean == pytest.approx(
            plain_worse.mean - plain.mean, rel=1e-12
        )


class TestTheObviousScoreIsNot:
    """The negative result, which is the point of the module."""

    @pytest.mark.parametrize("confidence", LEVELS)
    def test_its_derivative_at_the_truth_is_strictly_negative(
        self, confidence: float
    ) -> None:
        """``-phi(z) (VaR - ES)**2 / alpha``: it pushes the quantile upwards.

        Zero only where the shortfall equals the quantile, which is to say
        never, so the failure is structural rather than a matter of degree.
        """
        alpha = 1.0 - confidence
        var, shortfall = truth(confidence)
        predicted = -NORMAL.pdf(var) * (var - shortfall) ** 2 / alpha
        step = 1e-6
        measured = (
            expected_obvious(var + step, shortfall, confidence)
            - expected_obvious(var - step, shortfall, confidence)
        ) / (2.0 * step)
        assert measured == pytest.approx(predicted, rel=1e-6)
        assert measured < -0.3

    @pytest.mark.parametrize(
        ("confidence", "var_excess", "shortfall_excess"),
        [(0.95, 0.4880, 0.3453), (0.975, 0.4598, 0.3493), (0.99, 0.4431, 0.3571)],
    )
    def test_its_optimum_is_about_a_third_above_the_truth(
        self, confidence: float, var_excess: float, shortfall_excess: float
    ) -> None:
        var, shortfall = truth(confidence)
        found = minimise(
            lambda v, e: expected_obvious(v, e, confidence), (var * 1.4, shortfall * 0.9)
        )
        assert found[0] / var - 1.0 == pytest.approx(var_excess, abs=0.001)
        assert found[1] / shortfall - 1.0 == pytest.approx(shortfall_excess, abs=0.001)

    def test_given_the_quantile_it_does_elicit_the_tail_mean(self) -> None:
        """Which is why it looks right: the failure is in the other argument."""
        confidence = 0.975
        var, _ = truth(confidence)
        conditional = NORMAL.pdf(var) / (1.0 - NORMAL.cdf(var))
        step = 1e-7
        slope = (
            expected_obvious(var, conditional + step, confidence)
            - expected_obvious(var, conditional - step, confidence)
        ) / (2.0 * step)
        assert slope == pytest.approx(0.0, abs=1e-8)
        # And at the true quantile the conditional tail mean *is* the shortfall.
        assert conditional == pytest.approx(truth(confidence)[1], rel=1e-12)

    def test_the_two_scores_rank_the_same_pair_in_opposite_orders(self) -> None:
        """A truthful forecaster against one shading to the obvious optimum.

        Fissler-Ziegel prefers the truthful one, 0.84921 against 1.06369. The
        obvious score prefers the shaded one, 0.07849 against 0.17513. A desk
        choosing its model with the obvious score picks the wrong model, and
        the comparison looks decisive either way round.
        """
        confidence = 0.975
        var, shortfall = truth(confidence)
        shaded = minimise(
            lambda v, e: expected_obvious(v, e, confidence), (var * 1.4, shortfall * 0.9)
        )
        honest_fz = expected_fz0(var, shortfall, confidence)
        shaded_fz = expected_fz0(shaded[0], shaded[1], confidence)
        honest_obvious = expected_obvious(var, shortfall, confidence)
        shaded_obvious = expected_obvious(shaded[0], shaded[1], confidence)
        assert honest_fz < shaded_fz
        assert shaded_obvious < honest_obvious
        assert honest_fz == pytest.approx(0.84921, abs=0.0005)
        assert shaded_fz == pytest.approx(1.06369, abs=0.0005)
        assert honest_obvious == pytest.approx(0.17513, abs=0.0005)
        assert shaded_obvious == pytest.approx(0.07849, abs=0.0005)


class TestAgainstTheImplementation:
    """The closed forms above, against what the module computes on a sample."""

    @staticmethod
    def sample(count: int, sigma: float, seed: int) -> list[float]:
        rng = random.Random(seed)
        return [rng.gauss(0.0, sigma) for _ in range(count)]

    def test_the_sample_pinball_mean_tracks_its_expectation(self) -> None:
        confidence, sigma = 0.975, 0.01
        var, _ = truth(confidence, sigma)
        returns = self.sample(200_000, sigma, 5)
        scored = quantile_loss(returns, [var] * len(returns), confidence=confidence)
        assert scored.mean == pytest.approx(
            expected_pinball(var, confidence, sigma), rel=0.02
        )
        assert scored.name == "pinball"
        assert len(scored) == len(returns)

    def test_the_sample_fz0_mean_tracks_its_expectation(self) -> None:
        confidence, sigma = 0.975, 0.01
        var, shortfall = truth(confidence, sigma)
        returns = self.sample(200_000, sigma, 6)
        scored = fz0_loss(
            returns,
            [var] * len(returns),
            [shortfall] * len(returns),
            confidence=confidence,
        )
        assert scored.mean == pytest.approx(
            expected_fz0(var, shortfall, confidence, sigma), rel=0.01
        )
        assert scored.name == "fissler-ziegel"

    def test_the_truthful_model_scores_better_on_a_sample(self) -> None:
        confidence, sigma = 0.975, 0.01
        var, shortfall = truth(confidence, sigma)
        returns = self.sample(50_000, sigma, 7)
        honest = fz0_loss(
            returns, [var] * len(returns), [shortfall] * len(returns), confidence=confidence
        )
        for factor in (0.7, 0.85, 1.2, 1.5):
            other = fz0_loss(
                returns,
                [var * factor] * len(returns),
                [shortfall * factor] * len(returns),
                confidence=confidence,
            )
            assert honest.mean < other.mean


class TestComparison:
    @staticmethod
    def simulate(
        persistence: float, count: int = 4_000, seed: int = 7
    ) -> tuple[list[float], list[float]]:
        """Returns from a log-volatility process, and the volatilities."""
        rng = random.Random(seed)
        returns: list[float] = []
        volatilities: list[float] = []
        state = 0.0
        for _ in range(count):
            state = persistence * state + math.sqrt(
                1.0 - persistence * persistence
            ) * rng.gauss(0.0, 0.5)
            sigma = 0.01 * math.exp(state)
            volatilities.append(sigma)
            returns.append(rng.gauss(0.0, sigma))
        return returns, volatilities

    @staticmethod
    def ewma(returns: list[float], seed_variance: float, decay: float) -> list[float]:
        out: list[float] = []
        variance = seed_variance
        for value in returns:
            out.append(math.sqrt(variance))
            variance = decay * variance + (1.0 - decay) * value * value
        return out

    def scored(self, returns: list[float], sigmas: list[float], confidence: float) -> Scores:
        z = NORMAL.inv_cdf(confidence)
        alpha = 1.0 - confidence
        return fz0_loss(
            returns,
            [sigma * z for sigma in sigmas],
            [sigma * NORMAL.pdf(z) / alpha for sigma in sigmas],
            confidence=confidence,
        )

    @pytest.mark.parametrize(
        ("persistence", "ratio"), [(0.0, 0.9896), (0.95, 1.1041), (0.995, 1.0556)]
    )
    def test_the_robust_variance_is_real_and_modest(
        self, persistence: float, ratio: float
    ) -> None:
        """Measured, because "the naive one understates it" is an assumption.

        At zero persistence the robust standard error comes out *below* the
        naive one, which is the Bartlett estimator's own finite-sample noise
        rather than a correction in the wrong direction.
        """
        returns, volatilities = self.simulate(persistence)
        first = self.scored(returns, self.ewma(returns, volatilities[0] ** 2, 0.94), 0.975)
        second = self.scored(returns, self.ewma(returns, volatilities[0] ** 2, 0.995), 0.975)
        result = compare(first, second)
        assert result.standard_error / result.naive_standard_error == pytest.approx(
            ratio, abs=0.002
        )
        assert result.lags == 9
        assert result.observations == 4_000

    def test_a_model_that_knows_the_volatility_wins(self) -> None:
        returns, volatilities = self.simulate(0.98)
        knowing = self.scored(returns, volatilities, 0.975)
        unconditional = math.sqrt(
            math.fsum(sigma * sigma for sigma in volatilities) / len(volatilities)
        )
        blind = self.scored(returns, [unconditional] * len(returns), 0.975)
        result = compare(knowing, blind)
        assert result.difference < 0.0
        assert result.better == 1
        assert result.statistic < -3.0
        assert result.p_value < 0.01

    def test_a_model_compared_with_itself_is_a_tie(self) -> None:
        returns, volatilities = self.simulate(0.9)
        scored = self.scored(returns, volatilities, 0.975)
        result = compare(scored, scored)
        assert result.difference == 0.0
        assert result.better is None
        assert result.statistic == 0.0
        assert result.p_value == pytest.approx(1.0)

    def test_the_bandwidth_can_be_set(self) -> None:
        returns, volatilities = self.simulate(0.95)
        first = self.scored(returns, self.ewma(returns, volatilities[0] ** 2, 0.94), 0.975)
        second = self.scored(returns, self.ewma(returns, volatilities[0] ** 2, 0.995), 0.975)
        assert compare(first, second, lags=0).standard_error == pytest.approx(
            compare(first, second, lags=0).naive_standard_error, rel=1e-12
        )
        assert compare(first, second, lags=40).lags == 40

    def test_refuses_series_that_cannot_be_compared(self) -> None:
        returns, volatilities = self.simulate(0.5, count=100)
        a = self.scored(returns, volatilities, 0.975)
        b = self.scored(returns[:50], volatilities[:50], 0.975)
        with pytest.raises(BadScore, match="same observations"):
            compare(a, b)
        other_level = self.scored(returns, volatilities, 0.99)
        with pytest.raises(BadScore, match="same confidence"):
            compare(a, other_level)
        pinball = quantile_loss(
            returns, [sigma * 2.0 for sigma in volatilities], confidence=0.975
        )
        with pytest.raises(BadScore, match="same scoring function"):
            compare(a, pinball)
        with pytest.raises(BadScore, match="at least two observations"):
            compare(
                Scores((1.0,), 0.975, "pinball"), Scores((2.0,), 0.975, "pinball")
            )
        with pytest.raises(BadScore, match="bandwidth"):
            compare(a, a, lags=-1)


class TestValidation:
    RETURNS: ClassVar[list[float]] = [0.01, -0.02, 0.005, -0.04]

    def test_refuses_a_confidence_outside_the_open_interval(self) -> None:
        for level in (0.0, 1.0, -0.1, 1.2):
            with pytest.raises(BadScore, match="strictly in"):
                quantile_loss(self.RETURNS, [0.03] * 4, confidence=level)

    def test_refuses_mismatched_lengths(self) -> None:
        with pytest.raises(BadScore, match="one value at risk per observation"):
            quantile_loss(self.RETURNS, [0.03] * 3, confidence=0.975)

    def test_refuses_an_empty_series(self) -> None:
        with pytest.raises(BadScore, match="at least one observation"):
            quantile_loss([], [], confidence=0.975)

    def test_refuses_a_non_positive_forecast(self) -> None:
        with pytest.raises(BadScore, match="sign error"):
            quantile_loss(self.RETURNS, [0.03, 0.0, 0.03, 0.03], confidence=0.975)
        with pytest.raises(BadScore, match="sign error"):
            quantile_loss(self.RETURNS, [0.03, -0.01, 0.03, 0.03], confidence=0.975)

    def test_refuses_a_non_finite_value(self) -> None:
        with pytest.raises(BadScore, match="not a finite number"):
            quantile_loss([math.nan, *self.RETURNS[1:]], [0.03] * 4, confidence=0.975)
        with pytest.raises(BadScore, match="not a finite number"):
            quantile_loss(self.RETURNS, [0.03, math.inf, 0.03, 0.03], confidence=0.975)

    def test_refuses_a_shortfall_below_its_own_quantile(self) -> None:
        with pytest.raises(BadScore, match="cannot be smaller than the quantile"):
            fz0_loss(
                self.RETURNS,
                [0.03] * 4,
                [0.04, 0.04, 0.02, 0.04],
                confidence=0.975,
            )

    def test_accepts_a_shortfall_equal_to_its_quantile(self) -> None:
        """The boundary of the region the score is consistent on, not outside it."""
        scored = fz0_loss(self.RETURNS, [0.03] * 4, [0.03] * 4, confidence=0.975)
        assert len(scored) == 4
        assert all(math.isfinite(value) for value in scored.values)

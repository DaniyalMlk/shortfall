"""Copulas: the closed forms, the special cases, and the refusals.

Simulation-backed assertions here use small path counts and loose tolerances on
purpose. The tight numbers quoted in the module docstring were measured with
40,000 paths over several seeds, which is minutes of pure Python; a test suite
that took that long would not be run. What is asserted here is the direction and
the order of magnitude, plus every closed form exactly.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall import (
    MAX_COPULA_DEGREES,
    MIN_COPULA_PATHS,
    CopulaRisk,
    Family,
    FittedCopula,
    Marginal,
    Misaligned,
    Panel,
    TooShort,
    copula_risk,
    fit_copula,
    gaussian_copula_log_likelihood,
    joint_exceedance_rate,
    simulate_copula,
    student_t_copula_log_likelihood,
    tail_dependence_coefficient,
)
from shortfall.distributions import student_t_cdf


def one_factor_panel(
    seed: int,
    *,
    observations: int = 600,
    assets: int = 4,
    tau: float = 0.35,
    degrees: float | None = None,
    scale: float = 0.012,
) -> Panel:
    """A panel with a known equicorrelated copula.

    ``degrees=None`` gives a Gaussian copula; a number gives a t copula by the
    usual construction — one chi-square shared across the assets, which is the
    mechanism the fitted degrees of freedom are estimating.
    """
    rng = random.Random(seed)
    rho = math.sin(math.pi * tau / 2.0)
    columns: dict[str, list[float]] = {f"a{i}": [] for i in range(assets)}
    for _ in range(observations):
        common = rng.gauss(0.0, 1.0)
        mixing = (
            1.0
            if degrees is None
            else math.sqrt(degrees / rng.gammavariate(degrees / 2.0, 2.0))
        )
        for i in range(assets):
            idiosyncratic = rng.gauss(0.0, 1.0)
            z = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * idiosyncratic
            columns[f"a{i}"].append(scale * z * mixing)
    return Panel.from_columns(columns)

_PANELS: dict[tuple[object, ...], Panel] = {}
_FITS: dict[tuple[object, ...], FittedCopula] = {}


def panel_for(
    seed: int,
    *,
    observations: int = 500,
    assets: int = 3,
    tau: float = 0.35,
    degrees: float | None = None,
    scale: float = 0.012,
) -> Panel:
    """Cached :func:`one_factor_panel`, keyed by its arguments."""
    key = (seed, observations, assets, tau, degrees, scale)
    if key not in _PANELS:
        _PANELS[key] = one_factor_panel(
            seed,
            observations=observations,
            assets=assets,
            tau=tau,
            degrees=degrees,
            scale=scale,
        )
    return _PANELS[key]


def fit_for(panel: Panel, **kwargs: object) -> FittedCopula:
    """Cached :func:`fit_copula`. The fit is deterministic, so this is free."""
    key = (id(panel), tuple(sorted(kwargs.items())))
    if key not in _FITS:
        _FITS[key] = fit_copula(panel, **kwargs)  # type: ignore[arg-type]
    return _FITS[key]


class TestTailDependenceCoefficient:
    def test_gaussian_is_exactly_zero(self) -> None:
        for rho in (-0.9, 0.0, 0.5, 0.99, 0.999999):
            assert tail_dependence_coefficient(rho, None) == 0.0

    def test_gaussian_at_perfect_correlation_is_one(self) -> None:
        assert tail_dependence_coefficient(1.0, None) == 1.0

    def test_matches_the_closed_form(self) -> None:
        for rho in (0.0, 0.3, 0.7, 0.95):
            for degrees in (2.5, 4.0, 10.0, 50.0):
                expected = 2.0 * student_t_cdf(
                    -math.sqrt((degrees + 1.0) * (1.0 - rho) / (1.0 + rho)),
                    degrees + 1.0,
                )
                assert tail_dependence_coefficient(rho, degrees) == pytest.approx(
                    expected
                )

    def test_known_values(self) -> None:
        # A t copula at 4 degrees of freedom and a correlation of 0.5 puts a
        # quarter of the mass in joint tail arrival; the Gaussian value at the
        # same correlation is zero, which is the whole point. And a correlation
        # of *zero* still gives 7.6%, because the shared mixing variable makes
        # uncorrelated t variates arrive in their tails together.
        assert tail_dependence_coefficient(0.5, 4.0) == pytest.approx(
            0.25317000, abs=1e-7
        )
        assert tail_dependence_coefficient(0.0, 4.0) == pytest.approx(
            0.07558682, abs=1e-7
        )

    def test_agrees_with_an_elementary_antiderivative(self) -> None:
        # At five degrees of freedom the t distribution function has a closed
        # form in arctangents, which is an independent route to the coefficient:
        # it shares no code with the series the library evaluates. Checked here
        # because these were the two numbers a first draft of this test got
        # wrong, from memory rather than from arithmetic.
        def student_five(x: float) -> float:
            u = x / math.sqrt(5.0)
            t = 1.0 / (1.0 + u * u)
            return 0.5 + (math.atan(u) + u * t * (1.0 + 2.0 * t / 3.0)) / math.pi

        for rho in (-0.5, 0.0, 0.25, 0.6, 0.9):
            argument = -math.sqrt(5.0 * (1.0 - rho) / (1.0 + rho))
            assert tail_dependence_coefficient(rho, 4.0) == pytest.approx(
                2.0 * student_five(argument), abs=1e-9
            )

    def test_rises_with_correlation_and_falls_with_degrees(self) -> None:
        rising = [tail_dependence_coefficient(rho, 5.0) for rho in (0.0, 0.3, 0.6, 0.9)]
        assert rising == sorted(rising)
        falling = [
            tail_dependence_coefficient(0.5, degrees)
            for degrees in (3.0, 6.0, 15.0, 60.0)
        ]
        assert falling == sorted(falling, reverse=True)

    def test_vanishes_as_degrees_grow(self) -> None:
        # Not zero at any finite degrees of freedom, and small enough at the upper
        # bound of the fit that the distinction stops being measurable.
        assert tail_dependence_coefficient(0.5, MAX_COPULA_DEGREES) < 0.01
        assert tail_dependence_coefficient(0.5, MAX_COPULA_DEGREES) > 0.0

    def test_perfect_negative_correlation_has_no_upper_tail_dependence(self) -> None:
        assert tail_dependence_coefficient(-1.0, 4.0) == 0.0

    def test_rejects_a_correlation_outside_the_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            tail_dependence_coefficient(1.4, 4.0)


class TestLogLikelihood:
    def _uniforms(self, seed: int, n: int = 400) -> list[list[float]]:
        rng = random.Random(seed)
        return [[rng.uniform(0.02, 0.98) for _ in range(3)] for _ in range(n)]

    def test_t_converges_to_gaussian_as_degrees_grow(self) -> None:
        # The Student-t copula contains the Gaussian one as a limit. At 100,000
        # degrees of freedom the two log-likelihoods should agree to well within
        # a unit of likelihood over 400 observations.
        uniforms = self._uniforms(1)
        correlation = [[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]]
        gaussian = gaussian_copula_log_likelihood(uniforms, correlation)
        heavy = student_t_copula_log_likelihood(uniforms, correlation, 100_000.0)
        assert heavy == pytest.approx(gaussian, abs=0.5)

    def test_independence_likelihood_is_zero(self) -> None:
        # With an identity correlation matrix the copula density is 1 everywhere,
        # so its log is 0 exactly — for the Gaussian, and for the t only in the
        # limit, because a t copula at finite degrees of freedom is dependent even
        # with a diagonal matrix.
        uniforms = self._uniforms(2)
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert gaussian_copula_log_likelihood(uniforms, identity) == pytest.approx(
            0.0, abs=1e-9
        )
        assert student_t_copula_log_likelihood(uniforms, identity, 4.0) != pytest.approx(
            0.0, abs=1.0
        )

    def test_correlated_data_prefers_a_correlated_copula(self) -> None:
        panel = panel_for(3, tau=0.4)
        fitted = fit_for(panel, family=Family.GAUSSIAN)
        from shortfall.copula import _pseudo_observations

        uniforms = _pseudo_observations(panel)
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert fitted.log_likelihood > gaussian_copula_log_likelihood(
            uniforms, identity
        )

    def test_rejects_non_positive_degrees(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            student_t_copula_log_likelihood(
                self._uniforms(4), [[1.0, 0.0], [0.0, 1.0]], 0.0
            )


class TestFit:
    def test_recovers_the_degrees_of_freedom(self) -> None:
        # 1,500 observations of a four-asset t copula at 4 degrees of freedom.
        # The profile likelihood should land near the truth; the tolerance is wide
        # because the estimate is a tail parameter from a single sample.
        panel = panel_for(5, observations=1200, degrees=4.0)
        fitted = fit_for(panel)
        assert fitted.degrees_of_freedom is not None
        assert 3.0 < fitted.degrees_of_freedom < 6.0

    def test_recovers_the_correlation(self) -> None:
        panel = panel_for(6, observations=1200, tau=0.35)
        fitted = fit_for(panel)
        expected = math.sin(math.pi * 0.35 / 2.0)
        for i in range(3):
            for j in range(i + 1, 3):
                assert fitted.correlation[i][j] == pytest.approx(expected, abs=0.06)

    def test_gaussian_data_pushes_the_degrees_of_freedom_up(self) -> None:
        # The important negative result: no tail dependence in the data should
        # not produce a fit that says there is some.
        panel = panel_for(7, observations=1200, degrees=None)
        fitted = fit_for(panel)
        assert fitted.degrees_of_freedom is not None
        assert fitted.degrees_of_freedom > 12.0

    def test_likelihood_ratio_separates_the_two_cases(self) -> None:
        heavy = fit_for(panel_for(8, observations=1200, degrees=3.0))
        light = fit_for(panel_for(7, observations=1200, degrees=None))
        assert heavy.likelihood_ratio > 100.0
        assert light.likelihood_ratio < 15.0

    def test_gaussian_family_has_no_degrees_of_freedom(self) -> None:
        fitted = fit_for(panel_for(10), family=Family.GAUSSIAN)
        assert fitted.degrees_of_freedom is None
        assert fitted.log_likelihood == fitted.gaussian_log_likelihood
        assert fitted.likelihood_ratio == 0.0

    def test_fixed_degrees_of_freedom_are_used_not_fitted(self) -> None:
        panel = panel_for(11, degrees=None)
        fitted = fit_for(panel, degrees_of_freedom=3.0)
        assert fitted.degrees_of_freedom == 3.0
        # And the likelihood at an imposed value cannot beat the fitted one.
        assert fitted.log_likelihood <= fit_for(panel).log_likelihood + 1e-9

    def test_tail_dependence_lists_every_pair(self) -> None:
        fitted = fit_for(panel_for(12, degrees=4.0))
        pairs = fitted.tail_dependence()
        assert len(pairs) == 3
        assert {(p.first, p.second) for p in pairs} == {
            ("a0", "a1"),
            ("a0", "a2"),
            ("a1", "a2"),
        }
        for pair in pairs:
            assert pair.coefficient > 0.0
            assert not pair.is_asymptotically_independent

    def test_gaussian_pairs_are_asymptotically_independent(self) -> None:
        fitted = fit_for(panel_for(13), family=Family.GAUSSIAN)
        for pair in fitted.tail_dependence():
            assert pair.coefficient == 0.0
            assert pair.is_asymptotically_independent


    def test_the_search_beats_a_brute_force_grid(self) -> None:
        # A profile search that settles below the maximum is the failure mode
        # this code is most likely to have, and it does not announce itself: a
        # slightly wrong degrees of freedom still produces a plausible number.
        # So the search is checked against a grid of the same likelihood, and it
        # has to come out at least as high.
        from shortfall.copula import (
            _pseudo_observations,
            student_t_copula_log_likelihood,
        )
        from shortfall.ranks import elliptical_correlation, kendall_matrix

        panel = panel_for(33, observations=800, degrees=4.0)
        fitted = fit_for(panel)
        assert fitted.degrees_of_freedom is not None
        uniforms = _pseudo_observations(panel)
        correlation, _ = elliptical_correlation(kendall_matrix(panel))
        at_fit = student_t_copula_log_likelihood(
            uniforms, correlation, fitted.degrees_of_freedom
        )
        for step in range(8, 61, 4):
            candidate = step / 2.0
            assert at_fit >= student_t_copula_log_likelihood(
                uniforms, correlation, candidate
            ) - 1e-9


class TestFitRefusals:
    def test_needs_two_assets(self) -> None:
        single = Panel.from_columns({"a": [0.001 * i for i in range(300)]})
        with pytest.raises(TooShort, match="at least two assets"):
            fit_copula(single)

    def test_needs_enough_observations(self) -> None:
        with pytest.raises(TooShort, match="at least 200 observations"):
            fit_copula(one_factor_panel(14, observations=150))

    def test_gaussian_family_rejects_fixed_degrees(self) -> None:
        with pytest.raises(ValueError, match="no degrees of freedom"):
            fit_copula(
                panel_for(15), family=Family.GAUSSIAN, degrees_of_freedom=5.0
            )

    def test_degrees_out_of_bounds(self) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            fit_copula(panel_for(16), degrees_of_freedom=1.0)
        with pytest.raises(ValueError, match="must lie in"):
            fit_copula(panel_for(16), degrees_of_freedom=500.0)


class TestSimulation:
    def test_draws_are_uniform_on_each_margin(self) -> None:
        fitted = fit_for(panel_for(18, degrees=5.0))
        draws = simulate_copula(fitted, paths=6_000, seed=1)
        for asset in range(fitted.assets):
            column = [row[asset] for row in draws]
            assert all(0.0 <= value <= 1.0 for value in column)
            assert sum(column) / len(column) == pytest.approx(0.5, abs=0.03)

    def test_joint_tail_arrival_beats_the_gaussian_copula(self) -> None:
        # Same panel, same correlation matrix, one extra parameter. The fraction
        # of draws with every asset below its own 5% point is the finite-sample
        # face of tail dependence, and it is the number a portfolio feels.
        panel = panel_for(19, degrees=3.0)
        heavy = simulate_copula(fit_for(panel), paths=12_000, seed=2)
        light = simulate_copula(
            fit_for(panel, family=Family.GAUSSIAN), paths=12_000, seed=2
        )
        heavy_rate = joint_exceedance_rate(heavy, quantile=0.05)
        light_rate = joint_exceedance_rate(light, quantile=0.05)
        assert heavy_rate > light_rate
        assert heavy_rate > 5.0 * 0.05**3

    def test_independence_rate_is_the_product(self) -> None:
        rng = random.Random(20)
        independent = [[rng.random() for _ in range(3)] for _ in range(150_000)]
        assert joint_exceedance_rate(independent, quantile=0.1) == pytest.approx(
            0.001, abs=0.0005
        )

    def test_exceedance_rate_refusals(self) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            joint_exceedance_rate([[0.5]], quantile=0.0)
        with pytest.raises(TooShort, match="no draws"):
            joint_exceedance_rate([], quantile=0.01)

    def test_path_bounds(self) -> None:
        fitted = fit_for(panel_for(21))
        with pytest.raises(ValueError, match="paths must lie in"):
            simulate_copula(fitted, paths=MIN_COPULA_PATHS - 1)

    def test_seed_makes_it_reproducible(self) -> None:
        fitted = fit_for(panel_for(22, degrees=4.0))
        first = simulate_copula(fitted, paths=MIN_COPULA_PATHS, seed=7)
        second = simulate_copula(fitted, paths=MIN_COPULA_PATHS, seed=7)
        assert first == second


class TestPortfolioRisk:
    def test_t_copula_costs_more_than_the_gaussian_one(self) -> None:
        panel = panel_for(23, degrees=4.0)
        result = copula_risk(
            panel, [1 / 3] * 3, confidence=0.99, paths=4_000, seed=3
        )
        assert result.expected_shortfall > result.gaussian_expected_shortfall
        assert result.tail_dependence_premium > 0.02
        assert result.value_at_risk > 0.0
        assert result.standard_error > 0.0

    def test_gaussian_family_matches_its_own_baseline_exactly(self) -> None:
        # Under a Gaussian copula the fitted figure and the comparison figure are
        # the same computation on the same draws, so they must agree bit for bit.
        # Anything else means the shared-draw pairing is broken.
        panel = panel_for(24)
        result = copula_risk(
            panel,
            [1 / 3] * 3,
            family=Family.GAUSSIAN,
            paths=MIN_COPULA_PATHS,
            seed=4,
        )
        assert result.value_at_risk == result.gaussian_value_at_risk
        assert result.expected_shortfall == result.gaussian_expected_shortfall
        assert result.tail_dependence_premium == 0.0

    def test_spliced_marginal_answers_across_the_whole_unit_interval(self) -> None:
        # The defect this pins: the splice point has to be the fit's *realised*
        # exceedance fraction, not the tail fraction it was asked for. A
        # threshold at the 5% point of 800 losses is exceeded by 39 of them, so
        # the fit describes the worst 4.875% and refuses anything shallower.
        # Splicing at 5% sends a thin band of probabilities to a tail that
        # declines to answer, and since those probabilities only turn up in the
        # simulation occasionally, it is the kind of failure that appears on one
        # seed in five.
        from shortfall.copula import _build_marginals

        panel = panel_for(34, observations=800, degrees=4.0)
        for model in _build_marginals(panel, Marginal.EXTREME_VALUE):
            previous = model.quantile(1e-6)
            for step in range(1, 1000):
                current = model.quantile(step / 1000.0)
                assert current >= previous - 1e-12
                previous = current
            assert math.isfinite(model.quantile(0.999999))

    def test_extreme_value_marginals_are_at_least_as_severe(self) -> None:
        # Empirical marginals cannot exceed the worst observation for any single
        # asset; a fitted tail can. So splicing one on should not reduce the
        # portfolio figure, and generally raises it.
        panel = panel_for(25, observations=800, degrees=4.0)
        empirical = copula_risk(
            panel,
            [1 / 3] * 3,
            confidence=0.99,
            marginal=Marginal.EMPIRICAL,
            paths=4_000,
            seed=5,
        )
        spliced = copula_risk(
            panel,
            [1 / 3] * 3,
            confidence=0.99,
            marginal=Marginal.EXTREME_VALUE,
            paths=4_000,
            seed=5,
        )
        assert spliced.expected_shortfall > empirical.expected_shortfall * 0.98
        assert spliced.marginal is Marginal.EXTREME_VALUE

    def test_result_carries_the_fit_it_used(self) -> None:
        panel = panel_for(26, degrees=4.0)
        result = copula_risk(panel, [1 / 3] * 3, paths=MIN_COPULA_PATHS, seed=6)
        assert isinstance(result, CopulaRisk)
        assert isinstance(result.copula, FittedCopula)
        assert result.copula.names == ("a0", "a1", "a2")
        assert result.paths == MIN_COPULA_PATHS
        assert result.weights == (1 / 3, 1 / 3, 1 / 3)

    def test_concentrated_weights_are_riskier_than_spread_ones(self) -> None:
        panel = panel_for(27, tau=0.2, degrees=5.0)
        spread = copula_risk(
            panel, [1 / 3] * 3, confidence=0.99, paths=4_000, seed=8
        )
        concentrated = copula_risk(
            panel, [1.0, 0.0, 0.0], confidence=0.99, paths=4_000, seed=8
        )
        assert concentrated.value_at_risk > spread.value_at_risk

    def test_every_number_survives_strict_json(self) -> None:
        # Non-finite values are not JSON and a strict parser rejects the whole
        # document; `json.loads` accepts them, so a round trip does not catch it.
        import json

        panel = panel_for(28, degrees=4.0)
        result = copula_risk(panel, [1 / 3] * 3, paths=MIN_COPULA_PATHS, seed=9)
        payload = {
            "value_at_risk": result.value_at_risk,
            "expected_shortfall": result.expected_shortfall,
            "gaussian_value_at_risk": result.gaussian_value_at_risk,
            "gaussian_expected_shortfall": result.gaussian_expected_shortfall,
            "premium": result.tail_dependence_premium,
            "standard_error": result.standard_error,
            "degrees_of_freedom": result.copula.degrees_of_freedom,
            "likelihood_ratio": result.copula.likelihood_ratio,
            "correlation": result.copula.correlation,
            "tail_dependence": [
                pair.coefficient for pair in result.copula.tail_dependence()
            ],
        }
        json.dumps(payload, allow_nan=False)


    def test_the_premium_falls_as_the_correlation_rises(self) -> None:
        """The finding, at a tolerance a short test can afford.

        Measured properly in the module: the premium over a Gaussian copula runs
        +19.7% at a tau of 0.05 down to -1.0% at 0.90. The direction is the point
        and it is the opposite of the obvious expectation, so it is pinned here —
        loosely, because at 4,000 paths the figure has a couple of points of
        noise in it and the two ends are what separate.
        """
        weak = panel_for(40, observations=700, assets=3, tau=0.05, degrees=4.0)
        strong = panel_for(41, observations=700, assets=3, tau=0.9, degrees=4.0)
        diversified = copula_risk(
            weak, [1 / 3] * 3, confidence=0.99, paths=4_000, seed=11
        )
        concentrated = copula_risk(
            strong, [1 / 3] * 3, confidence=0.99, paths=4_000, seed=11
        )
        assert (
            diversified.tail_dependence_premium
            > concentrated.tail_dependence_premium
        )
        assert diversified.tail_dependence_premium > 0.05

    def test_uncorrelated_t_margins_still_arrive_together(self) -> None:
        # The mechanism behind the sweep above: the shared mixing variable does
        # not consult the correlation, so a t copula has tail dependence at a
        # correlation of exactly zero while a Gaussian one has none at all. This
        # is why the Gaussian promise of tail diversification is the one that
        # fails.
        assert tail_dependence_coefficient(0.0, 4.0) == pytest.approx(
            0.07558682, abs=1e-7
        )
        assert tail_dependence_coefficient(0.0, None) == 0.0


    def test_value_at_risk_can_move_the_other_way_from_expected_shortfall(
        self,
    ) -> None:
        """Measured: -4.0% against +6.0% at 95% on a weakly correlated book.

        Tail dependence moves mass from the near tail to the far tail and the
        total is one, so a quantile close to the body has less beyond it and
        comes in lower while the mean of what is beyond comes in higher. A
        reader taking the 95% value at risk alone would conclude the assumption
        made the portfolio safer, which is the reason both figures are in the
        result. Asserted as the gap between the two rather than as the sign of
        the first, because at 4,000 paths the sign of a 4% move is not reliable
        and the ordering is.
        """
        panel = panel_for(42, observations=900, assets=3, tau=0.05, degrees=4.0)
        shallow = copula_risk(
            panel, [1 / 3] * 3, confidence=0.95, paths=8_000, seed=13
        )
        var_change = (
            shallow.value_at_risk / shallow.gaussian_value_at_risk - 1.0
        )
        assert shallow.tail_dependence_premium > var_change

    def test_the_gap_between_the_two_measures_closes_further_out(self) -> None:
        # Measured: value at risk runs -4.0% at 95% and +29.4% at 99.9% while
        # expected shortfall runs +6.0% and +35.2%. Both rise, and the quantile
        # catches up, because far enough out it is inside the region the mass
        # moved to.
        panel = panel_for(42, observations=900, assets=3, tau=0.05, degrees=4.0)
        near = copula_risk(panel, [1 / 3] * 3, confidence=0.95, paths=8_000, seed=14)
        far = copula_risk(panel, [1 / 3] * 3, confidence=0.995, paths=8_000, seed=14)
        near_var = near.value_at_risk / near.gaussian_value_at_risk - 1.0
        far_var = far.value_at_risk / far.gaussian_value_at_risk - 1.0
        assert far_var > near_var


class TestRiskRefusals:
    def test_weight_count_must_match(self) -> None:
        panel = panel_for(29, observations=300)
        with pytest.raises(Misaligned, match="2 weights for 3 assets"):
            copula_risk(panel, [0.5, 0.5], paths=MIN_COPULA_PATHS)

    def test_path_bounds(self) -> None:
        panel = panel_for(29, observations=300)
        with pytest.raises(ValueError, match="paths must lie in"):
            copula_risk(panel, [1 / 3] * 3, paths=10)

    def test_confidence_bounds(self) -> None:
        panel = panel_for(29, observations=300)
        with pytest.raises(ValueError, match=r"\[0.5, 1\)"):
            copula_risk(
                panel, [1 / 3] * 3, confidence=1.0, paths=MIN_COPULA_PATHS
            )

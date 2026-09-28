"""Rank correlation, against the definition it is an optimisation of."""

from __future__ import annotations

import math
import random

import pytest

from shortfall import (
    Misaligned,
    Panel,
    ReturnSeries,
    TooShort,
    correlation_from_kendall,
    correlation_from_spearman,
    elliptical_correlation,
    kendall_matrix,
    kendall_tau,
    kendall_tau_quadratic,
    ranks_of,
    spearman_matrix,
    spearman_rho,
)
from shortfall.linalg import is_positive_semidefinite


def _sample(
    seed: int, n: int, *, rho: float = 0.0, digits: int | None = None
) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    x = []
    y = []
    for _ in range(n):
        a = rng.gauss(0.0, 1.0)
        b = rho * a + math.sqrt(1.0 - rho * rho) * rng.gauss(0.0, 1.0)
        if digits is not None:
            a = round(a, digits)
            b = round(b, digits)
        x.append(a)
        y.append(b)
    return x, y


class TestAgainstTheDefinition:
    def test_matches_pair_counting_without_ties(self) -> None:
        for seed in range(40):
            x, y = _sample(seed, 20 + seed)
            assert kendall_tau(x, y) == pytest.approx(
                kendall_tau_quadratic(x, y), abs=1e-12
            )

    def test_matches_pair_counting_with_heavy_ties(self) -> None:
        # Rounding to one decimal makes ties the rule rather than the exception,
        # which is the case the block handling and the tau-b denominator exist
        # for. Independent series are used so the sign varies across samples.
        for seed in range(40):
            x, y = _sample(1000 + seed, 30 + seed, digits=1)
            assert kendall_tau(x, y) == pytest.approx(
                kendall_tau_quadratic(x, y), abs=1e-12
            )

    def test_matches_pair_counting_under_dependence(self) -> None:
        for seed in range(20):
            x, y = _sample(2000 + seed, 50, rho=0.8, digits=1)
            assert kendall_tau(x, y) == pytest.approx(
                kendall_tau_quadratic(x, y), abs=1e-12
            )

    def test_perfectly_concordant_is_one(self) -> None:
        values = [float(i) for i in range(40)]
        assert kendall_tau(values, values) == pytest.approx(1.0)
        assert kendall_tau(values, [v * 3.0 + 7.0 for v in values]) == pytest.approx(
            1.0
        )

    def test_perfectly_discordant_is_minus_one(self) -> None:
        values = [float(i) for i in range(40)]
        assert kendall_tau(values, list(reversed(values))) == pytest.approx(-1.0)

    def test_invariant_under_monotone_transforms(self) -> None:
        # The whole reason to use a rank statistic: squaring a positive series
        # and taking logs of another changes every deviation and no ordering.
        x, y = _sample(5, 200)
        shifted_x = [v + 10.0 for v in x]
        shifted_y = [v + 10.0 for v in y]
        plain = kendall_tau(shifted_x, shifted_y)
        transformed = kendall_tau(
            [v**2 for v in shifted_x], [math.log(v) for v in shifted_y]
        )
        assert transformed == pytest.approx(plain)

    def test_symmetric_in_its_arguments(self) -> None:
        x, y = _sample(9, 120, rho=0.5)
        assert kendall_tau(x, y) == pytest.approx(kendall_tau(y, x))


class TestOutlierSensitivity:
    def test_one_joint_outlier_moves_tau_by_at_most_two_over_n(self) -> None:
        # Measured in the module docstring as 0.0066 at n = 300 against 0.175 for
        # a Pearson correlation. The bound asserted is the derived one: appending
        # a point adds n pairs to a numerator of n(n-1)/2 and n to the
        # denominator, so the move cannot exceed 2 * (1 + |tau|) / (n + 1).
        rng = random.Random(3)
        for _ in range(20):
            x = [rng.gauss(0.0, 1.0) for _ in range(300)]
            y = [rng.gauss(0.0, 1.0) for _ in range(300)]
            before = kendall_tau(x, y)
            after = kendall_tau([*x, 8.0], [*y, 8.0])
            bound = 2.0 * (1.0 + abs(before)) / 301.0
            assert abs(after - before) <= bound + 1e-12


class TestRefusals:
    def test_mismatched_lengths(self) -> None:
        with pytest.raises(Misaligned, match="same length"):
            kendall_tau([1.0] * 30, [1.0] * 29)
        with pytest.raises(Misaligned, match="same length"):
            spearman_rho([1.0] * 30, [1.0] * 29)

    def test_too_few_observations(self) -> None:
        with pytest.raises(TooShort, match="at least 20"):
            kendall_tau([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        with pytest.raises(TooShort, match="at least 20"):
            spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_constant_series_is_refused_not_answered_with_zero(self) -> None:
        # Every pair is tied, so the tau-b denominator is zero. "There is no
        # association" and "association is undefined here" are different
        # statements and a constant series is the second one.
        values = [float(i) for i in range(30)]
        with pytest.raises(TooShort, match="constant"):
            kendall_tau(values, [1.0] * 30)
        with pytest.raises(TooShort, match="no rank variation"):
            spearman_rho(values, [1.0] * 30)

    def test_inversion_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            correlation_from_kendall(1.5)
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            correlation_from_spearman(-1.2)


class TestRanks:
    def test_ties_share_the_average_rank(self) -> None:
        assert ranks_of([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]
        assert ranks_of([5.0, 5.0, 5.0]) == [2.0, 2.0, 2.0]

    def test_ranks_are_a_permutation_when_distinct(self) -> None:
        rng = random.Random(4)
        values = [rng.gauss(0.0, 1.0) for _ in range(50)]
        assert sorted(ranks_of(values)) == [float(i) for i in range(1, 51)]

    def test_spearman_of_ranks_is_one(self) -> None:
        rng = random.Random(6)
        values = [rng.gauss(0.0, 1.0) for _ in range(60)]
        assert spearman_rho(values, ranks_of(values)) == pytest.approx(1.0)


class TestEllipticalInversion:
    def test_known_values(self) -> None:
        # sin(pi/4) and sin(pi/20): the inversion is not a rescaling, and a tau
        # read as a linear correlation understates dependence throughout.
        assert correlation_from_kendall(0.5) == pytest.approx(0.7071067811865476)
        assert correlation_from_kendall(0.1) == pytest.approx(0.15643446504023087)
        assert correlation_from_kendall(0.0) == 0.0
        assert correlation_from_kendall(1.0) == pytest.approx(1.0)
        assert correlation_from_kendall(-1.0) == pytest.approx(-1.0)

    def test_spearman_inversion_known_values(self) -> None:
        assert correlation_from_spearman(0.0) == 0.0
        assert correlation_from_spearman(1.0) == pytest.approx(1.0)
        assert correlation_from_spearman(0.5) == pytest.approx(
            2.0 * math.sin(math.pi / 12.0)
        )

    def test_recovers_the_generating_correlation(self) -> None:
        # A Gaussian pair with a known rho: the Kendall route should land on it
        # to within sampling error at 4,000 observations.
        for rho in (-0.6, 0.0, 0.3, 0.85):
            x, y = _sample(int(abs(rho) * 100) + 1, 4000, rho=rho)
            recovered = correlation_from_kendall(kendall_tau(x, y))
            assert recovered == pytest.approx(rho, abs=0.03)

    def test_identity_is_left_alone(self) -> None:
        result, repaired = elliptical_correlation([[1.0, 0.0], [0.0, 1.0]])
        assert not repaired
        assert result == [[1.0, 0.0], [0.0, 1.0]]

    def test_inconsistent_pairwise_estimates_are_projected(self) -> None:
        # Three pairwise taus that no joint distribution can produce: each pair
        # strongly positive except one strongly negative.
        rank_matrix = [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
        result, repaired = elliptical_correlation(rank_matrix)
        assert repaired
        assert is_positive_semidefinite(result, tolerance=1e-8)
        for i in range(3):
            assert result[i][i] == pytest.approx(1.0)
            for j in range(3):
                assert result[i][j] == pytest.approx(result[j][i])
                assert -1.0 <= result[i][j] <= 1.0


class TestMatrices:
    def _panel(self, seed: int, n: int = 500) -> Panel:
        rng = random.Random(seed)
        factor = [rng.gauss(0.0, 1.0) for _ in range(n)]
        # Scaled to plausible daily returns: a simple return below -1 is refused
        # at the boundary, and rightly.
        columns = []
        for k in range(3):
            load = 0.3 * (k + 1)
            columns.append(
                ReturnSeries(
                    f"a{k}",
                    tuple(
                        0.01
                        * (
                            load * factor[i]
                            + math.sqrt(1.0 - load * load) * rng.gauss(0.0, 1.0)
                        )
                        for i in range(n)
                    ),
                )
            )
        return Panel(tuple(columns))

    def test_kendall_matrix_is_symmetric_with_unit_diagonal(self) -> None:
        matrix = kendall_matrix(self._panel(12))
        assert len(matrix) == 3
        for i in range(3):
            assert matrix[i][i] == 1.0
            for j in range(3):
                assert matrix[i][j] == pytest.approx(matrix[j][i])
                assert -1.0 <= matrix[i][j] <= 1.0

    def test_spearman_matrix_agrees_in_sign_and_ordering(self) -> None:
        panel = self._panel(13)
        tau = kendall_matrix(panel)
        rho = spearman_matrix(panel)
        # Both measure monotone association, so a stronger pair under one is a
        # stronger pair under the other; rho is the larger in magnitude for a
        # Gaussian dependence, which is the usual relation.
        assert rho[0][2] > tau[0][2] > 0.0
        assert rho[1][2] > tau[1][2] > 0.0

    def test_matrix_entries_match_the_pairwise_function(self) -> None:
        panel = self._panel(14)
        matrix = kendall_matrix(panel)
        direct = kendall_tau(panel.column(0).values, panel.column(2).values)
        assert matrix[0][2] == pytest.approx(direct)

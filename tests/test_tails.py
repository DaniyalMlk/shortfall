"""The chi-square and binomial tails the coverage tests are scored against.

The references here are exact rather than borrowed. A binomial probability is
a rational number, so :mod:`fractions` computes it to the last digit and the
implementation is compared against that instead of against another floating
point routine that might be wrong in the same direction. The chi-square values
are the published critical points, which is what a reader would check against.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from shortfall.distributions import (
    OutOfDomain,
    binomial_cdf,
    binomial_sf,
    chi_square_cdf,
    chi_square_sf,
    regularised_incomplete_gamma,
)

#: Upper-tail critical points, from the standard table. ``(degrees, tail, x)``.
CRITICAL_POINTS: list[tuple[int, float, float]] = [
    (1, 0.10, 2.70554),
    (1, 0.05, 3.84146),
    (1, 0.01, 6.63490),
    (1, 0.001, 10.82757),
    (2, 0.10, 4.60517),
    (2, 0.05, 5.99146),
    (2, 0.01, 9.21034),
    (2, 0.001, 13.81551),
    (3, 0.05, 7.81473),
    (4, 0.05, 9.48773),
    (5, 0.05, 11.07050),
    (10, 0.05, 18.30704),
    (20, 0.05, 31.41043),
]


@pytest.mark.parametrize(("degrees", "tail", "x"), CRITICAL_POINTS)
def test_chi_square_survival_matches_the_published_critical_points(
    degrees: int, tail: float, x: float
) -> None:
    # The table is quoted to five decimals, so the probability it implies is
    # only good to about that; the tolerance is the table's, not the code's.
    assert chi_square_sf(x, degrees) == pytest.approx(tail, rel=2e-5)


@pytest.mark.parametrize(("degrees", "tail", "x"), CRITICAL_POINTS)
def test_the_two_chi_square_tails_are_complements(degrees: int, tail: float, x: float) -> None:
    assert chi_square_cdf(x, degrees) + chi_square_sf(x, degrees) == pytest.approx(1.0, abs=1e-15)


def test_chi_square_on_two_degrees_is_the_exponential() -> None:
    """A closed form the general routine has no special case for."""
    for x in (0.25, 1.0, 4.0, 12.5, 40.0, 120.0):
        assert chi_square_sf(x, 2) == pytest.approx(math.exp(-0.5 * x), rel=1e-14)


def test_chi_square_on_one_degree_is_the_folded_normal() -> None:
    for x in (0.1, 1.0, 3.84146, 25.0, 60.0):
        assert chi_square_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(0.5 * x)), rel=1e-12)


def test_the_survival_function_keeps_digits_the_complement_has_lost() -> None:
    """The reason :func:`chi_square_sf` is not ``1 - chi_square_cdf``.

    At sixty on one degree of freedom the true probability is 9.4857e-15. The
    distribution function has not rounded to exactly one yet, so the complement
    still returns something — but it is wrong in the second digit, and further
    out it becomes wrong in the first and then returns zero.
    """
    direct = chi_square_sf(60.0, 1)
    complement = 1.0 - chi_square_cdf(60.0, 1)
    exact = math.erfc(math.sqrt(30.0))
    assert direct == pytest.approx(exact, rel=1e-12)
    assert abs(complement / exact - 1.0) > 1e-3
    # Far enough out the complement has nothing left at all.
    assert 1.0 - chi_square_cdf(200.0, 1) == 0.0
    assert chi_square_sf(200.0, 1) > 0.0


def test_chi_square_is_monotone_and_bounded() -> None:
    previous = 1.0
    for x in [i * 0.37 for i in range(1, 300)]:
        value = chi_square_sf(x, 4)
        assert 0.0 <= value <= previous
        previous = value


def test_both_incomplete_gamma_branches_match_a_closed_form() -> None:
    """``P(1, x) = 1 - exp(-x)``, which straddles the crossover at ``x = 2``.

    Comparing the two branches to each other at *nearly* the same point is the
    tempting version of this test and it cannot pass: the branch is chosen by
    ``x``, so the two calls are at different arguments and the function has a
    derivative. Either side of ``x = 2`` the values differ by the step times
    the density, not by rounding. Each branch is therefore held to the closed
    form at its own argument instead.
    """
    for x in (0.1, 0.5, 1.0, 1.99, 2.01, 3.0, 8.0, 30.0):
        assert regularised_incomplete_gamma(x, 1.0) == pytest.approx(-math.expm1(-x), rel=1e-13)


def test_the_incomplete_gamma_is_continuous_across_the_crossover() -> None:
    """The step across the branch change is the density's, and no larger."""
    for a in (0.5, 1.0, 2.5, 7.0):
        step = 1e-9
        below = regularised_incomplete_gamma(a + 1.0 - step, a)
        above = regularised_incomplete_gamma(a + 1.0 + step, a)
        # The gamma density at x = a + 1, which is what the gap should be.
        x = a + 1.0
        density = math.exp((a - 1.0) * math.log(x) - x - math.lgamma(a))
        assert above - below == pytest.approx(2.0 * step * density, rel=1e-5)


def test_degenerate_arguments() -> None:
    assert chi_square_cdf(0.0, 3) == 0.0
    assert chi_square_sf(0.0, 3) == 1.0
    assert chi_square_sf(-1.0, 3) == 1.0
    assert regularised_incomplete_gamma(0.0, 2.0) == 0.0
    with pytest.raises(OutOfDomain):
        regularised_incomplete_gamma(-1.0, 2.0)
    with pytest.raises(OutOfDomain):
        chi_square_sf(1.0, 0.0)


def exact_binomial_cdf(successes: int, trials: int, probability: Fraction) -> Fraction:
    """``P(X <= successes)`` in exact rational arithmetic."""
    return sum(
        (
            Fraction(math.comb(trials, count))
            * probability**count
            * (1 - probability) ** (trials - count)
            for count in range(0, successes + 1)
        ),
        Fraction(0),
    )


BINOMIAL_CASES: list[tuple[int, Fraction]] = [
    (250, Fraction(1, 100)),
    (250, Fraction(5, 100)),
    (500, Fraction(1, 1000)),
    (60, Fraction(1, 2)),
    (30, Fraction(3, 10)),
]


@pytest.mark.parametrize(("trials", "probability"), BINOMIAL_CASES)
def test_binomial_matches_exact_rational_arithmetic(trials: int, probability: Fraction) -> None:
    worst = 0.0
    for successes in range(0, min(trials, 40) + 1):
        exact = float(exact_binomial_cdf(successes, trials, probability))
        got = binomial_cdf(successes, trials, float(probability))
        worst = max(worst, abs(got - exact))
    # Measured, not aspirational: the incomplete beta carries the error of a
    # lgamma of the trial count, which is a few parts in 1e13 at these sizes.
    assert worst < 1e-12


@pytest.mark.parametrize(("trials", "probability"), BINOMIAL_CASES)
def test_the_binomial_tails_are_complements(trials: int, probability: Fraction) -> None:
    for successes in range(0, min(trials, 40) + 1):
        total = binomial_cdf(successes, trials, float(probability)) + binomial_sf(
            successes + 1, trials, float(probability)
        )
        assert total == pytest.approx(1.0, abs=1e-12)


def test_a_far_binomial_tail_is_returned_rather_than_rounded_away() -> None:
    """Twenty breaches in 250 days at 1% is not impossible, it is 1.9e-12."""
    exact = 1.0 - float(exact_binomial_cdf(19, 250, Fraction(1, 100)))
    got = binomial_sf(20, 250, 0.01)
    assert got > 0.0
    assert got == pytest.approx(exact, rel=1e-9)


def test_binomial_boundaries() -> None:
    assert binomial_cdf(-1, 10, 0.3) == 0.0
    assert binomial_cdf(10, 10, 0.3) == 1.0
    assert binomial_sf(0, 10, 0.3) == 1.0
    assert binomial_sf(11, 10, 0.3) == 0.0
    assert binomial_cdf(3, 10, 0.0) == 1.0
    assert binomial_sf(1, 10, 0.0) == 0.0
    assert binomial_cdf(3, 10, 1.0) == 0.0
    assert binomial_sf(1, 10, 1.0) == 1.0
    with pytest.raises(OutOfDomain):
        binomial_cdf(1, -1, 0.5)
    with pytest.raises(OutOfDomain):
        binomial_sf(1, 10, 1.5)

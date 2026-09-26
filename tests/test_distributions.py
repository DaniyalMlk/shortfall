"""The normal and Student-t primitives.

Checked against values that exist independently of this code: published
quantiles, the closed forms the Student-t has at one and two degrees of freedom,
and the exact values of the incomplete beta at integer shape parameters. The
round trips are there too, but a round trip only shows that two functions agree
with each other, so it is never the only evidence for anything here.
"""

from __future__ import annotations

import math

import pytest

from shortfall.distributions import (
    OutOfDomain,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    standardised_t_log_pdf,
    regularised_incomplete_beta,
    student_t_cdf,
    student_t_pdf,
    student_t_ppf,
)

# -- the normal -------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.5, 0.0),
        (0.75, 0.6744897501960817),
        (0.95, 1.6448536269514722),
        (0.975, 1.9599639845400545),
        (0.99, 2.3263478740408408),
        (0.995, 2.5758293035489004),
        (0.999, 3.090232306167813),
        (0.9999, 3.719016485455709),
    ],
)
def test_the_normal_quantile_matches_published_values(p: float, expected: float) -> None:
    assert normal_ppf(p) == pytest.approx(expected, rel=1e-14)


def test_the_normal_quantile_is_antisymmetric() -> None:
    for p in (0.001, 0.01, 0.1, 0.3):
        assert normal_ppf(p) == pytest.approx(-normal_ppf(1.0 - p), rel=1e-13)


@pytest.mark.parametrize("p", [1e-15, 1e-10, 1e-6, 0.001, 0.05, 0.5, 0.95, 0.999999])
def test_the_normal_quantile_inverts_the_distribution_function(p: float) -> None:
    # Relative, not absolute. An absolute tolerance is met trivially by any
    # function that returns something small when asked for a small probability,
    # which is the whole difficulty in the tail.
    assert normal_cdf(normal_ppf(p)) == pytest.approx(p, rel=1e-12)


def test_the_distribution_function_is_accurate_far_into_the_tail() -> None:
    # The erf form of this computes a small number as the difference of two
    # numbers near one and has no correct digits left by here.
    assert normal_cdf(-6.0) == pytest.approx(9.865876450376946e-10, rel=1e-12)
    assert normal_cdf(-10.0) == pytest.approx(7.619853024160525e-24, rel=1e-11)


def test_the_density_matches_its_closed_form() -> None:
    assert normal_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), rel=1e-15)
    assert normal_pdf(1.0) == pytest.approx(0.24197072451914337, rel=1e-14)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.1])
def test_a_probability_outside_the_open_unit_interval_is_refused(p: float) -> None:
    with pytest.raises(OutOfDomain, match=r"\(0, 1\)"):
        normal_ppf(p)


# -- the incomplete beta ----------------------------------------------------


@pytest.mark.parametrize(
    ("x", "a", "b", "expected"),
    [
        # I_x(1, 1) = x, since the beta(1,1) is uniform.
        (0.3, 1.0, 1.0, 0.3),
        (0.75, 1.0, 1.0, 0.75),
        # I_x(2, 1) = x^2 and I_x(1, 2) = 1 - (1 - x)^2.
        (0.4, 2.0, 1.0, 0.16),
        (0.4, 1.0, 2.0, 0.64),
        # I_x(2, 3) = x^2 (6 - 8x + 3x^2); at x = 1/2 that is 0.6875.
        (0.5, 2.0, 3.0, 0.6875),
        (0.0, 2.0, 3.0, 0.0),
        (1.0, 2.0, 3.0, 1.0),
    ],
)
def test_the_incomplete_beta_matches_its_closed_forms(
    x: float, a: float, b: float, expected: float
) -> None:
    assert regularised_incomplete_beta(x, a, b) == pytest.approx(expected, rel=1e-14, abs=1e-15)


def test_the_incomplete_beta_satisfies_its_reflection_identity() -> None:
    # I_x(a, b) = 1 - I_{1-x}(b, a). The implementation uses this to stay on the
    # fast side of the continued fraction, so it must hold on both sides.
    for x in (0.05, 0.2, 0.5, 0.8, 0.95):
        assert regularised_incomplete_beta(x, 2.5, 4.5) == pytest.approx(
            1.0 - regularised_incomplete_beta(1.0 - x, 4.5, 2.5), rel=1e-13
        )


def test_the_incomplete_beta_is_defined_on_the_unit_interval_only() -> None:
    with pytest.raises(OutOfDomain, match=r"\[0, 1\]"):
        regularised_incomplete_beta(1.5, 1.0, 1.0)


# -- the Student-t ----------------------------------------------------------


@pytest.mark.parametrize("p", [0.6, 0.75, 0.9, 0.95, 0.99, 0.999])
def test_the_cauchy_case_matches_its_closed_form(p: float) -> None:
    # One degree of freedom is the Cauchy, whose quantile is tan(pi (p - 1/2)).
    assert student_t_ppf(p, 1.0) == pytest.approx(math.tan(math.pi * (p - 0.5)), rel=1e-12)


@pytest.mark.parametrize("p", [0.6, 0.75, 0.9, 0.975, 0.995])
def test_the_two_degree_case_matches_its_closed_form(p: float) -> None:
    # At two degrees of freedom the quantile is (2p - 1) sqrt(2 / (1 - (2p-1)^2)).
    shifted = 2.0 * p - 1.0
    expected = shifted * math.sqrt(2.0 / (1.0 - shifted * shifted))
    assert student_t_ppf(p, 2.0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("p", "degrees", "expected"),
    [
        (0.95, 5.0, 2.015048373),
        (0.975, 10.0, 2.228138852),
        (0.99, 20.0, 2.527977003),
        (0.975, 2.0, 4.302652730),
    ],
)
def test_the_quantile_matches_published_tables(
    p: float, degrees: float, expected: float
) -> None:
    assert student_t_ppf(p, degrees) == pytest.approx(expected, rel=1e-8)


def test_the_quantile_is_symmetric_about_zero() -> None:
    assert student_t_ppf(0.5, 7.0) == 0.0
    for p in (0.01, 0.1, 0.3):
        assert student_t_ppf(p, 7.0) == pytest.approx(-student_t_ppf(1.0 - p, 7.0), rel=1e-12)


@pytest.mark.parametrize("degrees", [1.0, 2.0, 3.5, 10.0, 100.0])
@pytest.mark.parametrize("p", [0.001, 0.05, 0.4, 0.9, 0.995])
def test_the_quantile_inverts_the_distribution_function(p: float, degrees: float) -> None:
    assert student_t_cdf(student_t_ppf(p, degrees), degrees) == pytest.approx(p, rel=1e-11)


def test_the_distribution_function_is_a_half_at_zero() -> None:
    for degrees in (1.0, 3.0, 50.0):
        assert student_t_cdf(0.0, degrees) == 0.5


def test_the_tails_are_heavier_than_the_normal_and_converge_to_it() -> None:
    # The defining property, and the reason the t is offered at all.
    assert student_t_ppf(0.99, 3.0) > student_t_ppf(0.99, 30.0) > normal_ppf(0.99)
    assert student_t_ppf(0.99, 1e6) == pytest.approx(normal_ppf(0.99), rel=1e-5)


def test_the_density_does_not_overflow_at_large_degrees_of_freedom() -> None:
    # Both gamma functions in the normalising constant overflow above about 170
    # degrees of freedom while their ratio does not, so this is computed in logs.
    assert student_t_pdf(0.0, 1000.0) == pytest.approx(normal_pdf(0.0), rel=1e-3)
    assert student_t_pdf(0.0, 1e7) == pytest.approx(normal_pdf(0.0), rel=1e-6)


def test_the_density_integrates_against_its_own_distribution_function() -> None:
    # A central difference of the distribution function must be the density.
    # Separate code paths — a continued fraction against a closed form — so
    # agreement is evidence rather than a tautology.
    #
    # The step is 1e-4 because that is where the two error sources cross. Below
    # it the subtraction of two distribution values differing in the fifth
    # decimal amplifies their rounding by 1/(2h); above it the truncation error
    # of the difference itself dominates. Measured across the points below, the
    # agreement is 3e-7 at 1e-3, 7e-9 at 1e-4, 4e-8 at 1e-5 and 7e-5 at 1e-6 —
    # the curve of a finite difference against an accurate function, which is
    # what this is really evidence of.
    step = 1e-4
    for x in (-2.0, -0.5, 0.0, 1.3):
        slope = (student_t_cdf(x + step, 6.0) - student_t_cdf(x - step, 6.0)) / (2.0 * step)
        assert slope == pytest.approx(student_t_pdf(x, 6.0), rel=1e-7)


@pytest.mark.parametrize("degrees", [0.0, -1.0])
def test_non_positive_degrees_of_freedom_are_refused(degrees: float) -> None:
    with pytest.raises(OutOfDomain, match="degrees of freedom"):
        student_t_ppf(0.9, degrees)


@pytest.mark.parametrize("p", [0.0, 1.0, 2.0])
def test_a_student_t_probability_outside_the_unit_interval_is_refused(p: float) -> None:
    with pytest.raises(OutOfDomain, match=r"\(0, 1\)"):
        student_t_ppf(p, 5.0)


# -- the standardised Student-t density --------------------------------------


@pytest.mark.parametrize("degrees", [2.5, 4.0, 5.0, 30.0, 300.0])
def test_the_standardised_density_is_the_change_of_variables(degrees: float) -> None:
    """``f_Z(z) = c f_T(c z)`` exactly, with ``c = sqrt(v / (v - 2))``.

    Written independently here: the implementation folds the scale into the
    normalising constant, this multiplies it out.
    """
    scale = math.sqrt(degrees / (degrees - 2.0))
    for z in (-4.0, -1.0, 0.0, 0.5, 2.2):
        assert math.exp(standardised_t_log_pdf(z, degrees)) == pytest.approx(
            scale * student_t_pdf(scale * z, degrees), rel=1e-12
        )


@pytest.mark.parametrize("degrees", [3.0, 6.0, 25.0])
def test_the_standardised_density_integrates_to_one_and_has_unit_variance(
    degrees: float,
) -> None:
    """The two properties that make it usable in a variance-parameterised model.

    Simpson's rule over a symmetric interval, which truncates rather than
    approximates: everything outside the interval is simply missing, so both
    integrals come out *below* their true value and never above it.

    The mass converges quickly enough that the truncation is invisible at 1e-6.
    The second moment does not, and the reason is the fat tail itself. At three
    degrees of freedom the standardised density falls off like ``z^-4``, so the
    integrand of the variance falls off like ``z^-2`` and the tail beyond
    ``limit`` carries of order ``1 / limit`` of the total. Measured at
    ``limit = 400`` the second moment is 0.99682, a deficit of 1.3/limit, and
    the bound below allows 2/limit in that direction and 1e-6 in the other.
    That asymmetry is the assertion: a fat tail can only be under-counted here.
    """
    limit, steps = 400.0, 400_000
    step = 2.0 * limit / steps
    mass = 0.0
    second = 0.0
    for index in range(steps + 1):
        z = -limit + index * step
        weight = 1.0 if index in (0, steps) else (4.0 if index % 2 else 2.0)
        density = math.exp(standardised_t_log_pdf(z, degrees))
        mass += weight * density
        second += weight * z * z * density
    mass *= step / 3.0
    second *= step / 3.0
    assert mass == pytest.approx(1.0, abs=1e-6)
    assert 1.0 - 2.0 / limit <= second <= 1.0 + 1e-6


def test_the_standardised_density_approaches_the_normal_at_the_rate_1_over_v() -> None:
    """The departure from normal shrinks like ``1 / v``, and is measured saying so.

    Not "agrees to six figures", which would be wrong: at 10,000 degrees of
    freedom the relative departure is 7.5e-5 at the centre and 7.5e-4 at three
    standard deviations, because the implied excess kurtosis is ``6 / (v - 4)``
    and the standardising factor is ``1 + 1/v + O(v^-2)``. Both are first order
    in ``1/v``, so raising ``v`` tenfold divides the departure by ten — which is
    what this asserts, at two decades apart.

    This is the fact behind reporting an unidentified fit rather than a number:
    the likelihood at ``v = 1000`` and at ``v = 10,000`` differ by a thousandth
    of a nat per observation, so the optimiser has nothing to find up there.
    """
    for z in (-3.0, -1.0, 0.0, 2.0):
        reference = normal_pdf(z)
        near = abs(math.exp(standardised_t_log_pdf(z, 1_000.0)) - reference) / reference
        far = abs(math.exp(standardised_t_log_pdf(z, 10_000.0)) - reference) / reference
        assert near == pytest.approx(10.0 * far, rel=0.02)
        assert far < 1e-3


@pytest.mark.parametrize("degrees", [2.0, 1.5, 0.0, -3.0])
def test_the_standardised_density_refuses_degrees_without_a_variance(
    degrees: float,
) -> None:
    with pytest.raises(OutOfDomain, match="greater than 2"):
        standardised_t_log_pdf(0.0, degrees)

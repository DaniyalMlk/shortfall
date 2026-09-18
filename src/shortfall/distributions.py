"""The normal and Student-t primitives, to full double precision.

Python ships ``statistics.NormalDist``, which would cover half of this, and
nothing at all for the Student-t. Since the library has no dependencies, both
are built here — and building them means they can be held to the accuracy the
risk numbers need rather than to whatever the nearest available routine happens
to give.

That accuracy matters more in the tail than anywhere else, which is exactly
where risk lives. A quantile function that is good to six digits in the body and
four in the tail is fine for plotting and useless for a 99.9% value at risk.

Each routine says what it is accurate to, and each is checked against published
values in the tests rather than against itself.
"""

from __future__ import annotations

import math
from typing import Final

#: 1 / sqrt(2*pi), the normal density's normalising constant.
_INV_SQRT_2PI: Final = 0.3989422804014327

class OutOfDomain(ValueError):
    """An argument outside the range a distribution function is defined on."""


def normal_pdf(x: float) -> float:
    """The standard normal density."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def normal_cdf(x: float) -> float:
    """The standard normal distribution function.

    Written with ``erfc`` rather than ``erf``. They are equivalent in exact
    arithmetic and not in floating point: for ``x`` well below zero,
    ``(1 + erf(x/sqrt(2)))/2`` computes a small number as the difference of two
    numbers near one and loses every significant digit, while ``erfc`` computes
    it directly. At ``x = -6`` the first form has no correct digits left; the
    second is exact to the last bit.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def normal_ppf(p: float) -> float:
    """The standard normal quantile function.

    Acklam's rational approximation, which is good to about 1.15e-9 in relative
    terms, followed by one step of Halley's method against :func:`normal_cdf`.
    Halley is cubically convergent, so a starting point accurate to 1e-9 lands
    at full double precision in a single step and a second step would only move
    the last bit around.

    The refinement is what makes this worth writing out rather than using the
    approximation directly: 1e-9 is ample for a chart and not for a 99.99%
    quantile, where the difference between the approximation and the true value
    is a real amount of money.
    """
    if not 0.0 < p < 1.0:
        raise OutOfDomain(
            f"a probability is strictly inside (0, 1), got {p!r}; the normal "
            "quantile is unbounded at both ends"
        )
    estimate = _acklam(p)
    # One Halley step on F(x) - p = 0, using F'' = -x F'.
    error = normal_cdf(estimate) - p
    density = normal_pdf(estimate)
    if density > 0.0:
        step = error / density
        estimate -= step / (1.0 + 0.5 * estimate * step)
    return estimate


# Acklam's coefficients. The central region is a rational function of
# (p - 1/2)^2; the tails are a rational function of sqrt(-2 ln p).
_A: Final = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B: Final = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C: Final = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D: Final = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_LOW: Final = 0.02425


def _acklam(p: float) -> float:
    if p < _LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p > 1.0 - _LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    )


# -- Student-t --------------------------------------------------------------


def _check_degrees(degrees: float, *, minimum: float = 0.0, what: str = "degrees") -> None:
    if not degrees > minimum:
        raise OutOfDomain(
            f"{what} of freedom must be greater than {minimum:g}, got {degrees!r}"
        )


def student_t_pdf(x: float, degrees: float) -> float:
    """The standard Student-t density with ``degrees`` degrees of freedom.

    Computed through ``lgamma`` rather than ``gamma``. The two gamma functions
    in the normalising constant both overflow above about 170 degrees of
    freedom, and their ratio does not, so taking logs is the difference between
    a correct answer and an ``OverflowError`` for a distribution that is by then
    almost exactly normal.
    """
    _check_degrees(degrees)
    log_norm = (
        math.lgamma((degrees + 1.0) / 2.0)
        - math.lgamma(degrees / 2.0)
        - 0.5 * math.log(degrees * math.pi)
    )
    return math.exp(log_norm - ((degrees + 1.0) / 2.0) * math.log1p(x * x / degrees))


def student_t_cdf(x: float, degrees: float) -> float:
    """The standard Student-t distribution function.

    Through the regularised incomplete beta function, which is the exact
    relation rather than an approximation of one. The two halves of the
    distribution are handled separately so the small tail is always computed as
    a small number, never as one minus a number near one.
    """
    _check_degrees(degrees)
    if x == 0.0:
        return 0.5
    half = 0.5 * regularised_incomplete_beta(degrees / (degrees + x * x), degrees / 2.0, 0.5)
    return half if x < 0.0 else 1.0 - half


def student_t_ppf(p: float, degrees: float) -> float:
    """The standard Student-t quantile function.

    There is no closed form, so this brackets the root and then bisects with a
    guaranteed-correct interval rather than running Newton from a guess. Newton
    is faster and can leave the bracket entirely on a flat tail, which turns a
    slow answer into a wrong one; the bracket here cannot be left, and the loop
    is over after sixty halvings whatever the input.
    """
    _check_degrees(degrees)
    if not 0.0 < p < 1.0:
        raise OutOfDomain(
            f"a probability is strictly inside (0, 1), got {p!r}; the Student-t "
            "quantile is unbounded at both ends"
        )
    if p == 0.5:
        return 0.0

    # Start from the normal quantile, which is the right shape, and widen until
    # the root is enclosed. The t is heavier-tailed, so the true quantile is
    # always further out than the normal one and the widening only goes one way.
    guess = normal_ppf(p)
    lower, upper = (guess - 1.0, 0.0) if p < 0.5 else (0.0, guess + 1.0)
    while student_t_cdf(lower, degrees) > p:
        lower *= 2.0
        if lower < -1e12:  # pragma: no cover - unreachable for p above 1e-300
            break
    while student_t_cdf(upper, degrees) < p:
        upper *= 2.0 if upper != 0.0 else 1.0
        upper = upper if upper != 0.0 else 1.0
        if upper > 1e12:  # pragma: no cover
            break

    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if middle in (lower, upper):
            break
        if student_t_cdf(middle, degrees) < p:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """``I_x(a, b)``, by the continued fraction of Lentz as modified by Thompson.

    The fraction converges quickly on one side of ``x = (a+1)/(a+b+2)`` and
    slowly on the other, so the symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)`` is
    used to stay on the fast side. Without that the evaluation is not merely
    slower: it stops converging within any sensible iteration count near the
    ends, which is precisely the region a tail quantile asks about.
    """
    if not 0.0 <= x <= 1.0:
        raise OutOfDomain(f"the incomplete beta is defined on [0, 1], got x={x!r}")
    _check_degrees(a, what="the first shape parameter, a,")
    _check_degrees(b, what="the second shape parameter, b,")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_fraction(x, a, b) / a
    return 1.0 - math.exp(log_front) * _beta_fraction(1.0 - x, b, a) / b


_TINY: Final = 1e-300
_FRACTION_TOLERANCE: Final = 1e-16
_FRACTION_STEPS: Final = 300


def _beta_fraction(x: float, a: float, b: float) -> float:
    """The continued fraction for the incomplete beta, by the modified Lentz method.

    Each iteration applies *two* of the fraction's terms — the even one and the
    odd one — because they have different shapes. Folding them into a single
    loop over one index is the obvious simplification and it is wrong in a way
    that is easy to miss: the m=0 odd term is already applied as the
    initialisation of ``d`` below, so a single loop starting from it applies
    that term twice and every quantile comes back plausible and incorrect.
    """
    c = 1.0
    # This is 1 + aa*1 for the m=0 odd term, so the loop must start after it.
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    result = d

    for m in range(1, _FRACTION_STEPS + 1):
        two_m = 2.0 * m

        even = m * (b - m) * x / ((a + two_m - 1.0) * (a + two_m))
        d = 1.0 + even * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + even / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        result *= d * c

        odd = -(a + m) * (a + b + m) * x / ((a + two_m) * (a + two_m + 1.0))
        d = 1.0 + odd * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + odd / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        factor = d * c
        result *= factor

        if abs(factor - 1.0) < _FRACTION_TOLERANCE:
            return result
    # Unreachable for the arguments a Student-t produces; kept so that a caller
    # reaching it learns the fraction did not converge rather than receiving a
    # partial sum that looks like an answer.
    raise ArithmeticError(  # pragma: no cover
        f"the incomplete beta continued fraction did not converge in "
        f"{_FRACTION_STEPS} steps at x={x!r}, a={a!r}, b={b!r}"
    )


__all__ = [
    "OutOfDomain",
    "normal_cdf",
    "normal_pdf",
    "normal_ppf",
    "regularised_incomplete_beta",
    "student_t_cdf",
    "student_t_pdf",
    "student_t_ppf",
]

"""Risk when the portfolio is not linear in its factors.

A book of options is not a vector of exposures. To second order its profit and
loss is

    L = a + delta' x + 0.5 x' Gamma x

for a factor move ``x`` drawn from ``N(0, Sigma)``, and the quadratic term is
not a correction to be bolted on: it is where every asymmetry in the
distribution comes from. Whiten the factors with a Cholesky factor ``C`` of
``Sigma`` and diagonalise ``C' Gamma C``, and the problem becomes a sum of
independent pieces,

    L = a + sum_i (beta_i u_i + 0.5 lambda_i u_i^2),    u ~ N(0, I)

which is what :func:`standard_form` produces and everything else here reads.
The eigenvalues are the book's gamma in its own principal directions; the
linear coefficients are its delta in the same frame.

**The cumulants are closed forms, and the third one is pure gamma.** From the
cumulant generating function

    K(t) = a t + sum_i [ -0.5 log(1 - lambda_i t) + 0.5 beta_i^2 t^2 / (1 - lambda_i t) ]

the first four are

    k1 = a + 0.5 sum lambda
    k2 = sum (0.5 lambda^2 + beta^2)
    k3 = sum (lambda^3 + 3 beta^2 lambda)
    k4 = sum (3 lambda^4 + 12 beta^2 lambda^2)

Every term of ``k3`` and ``k4`` carries a ``lambda``, so a linear book has no
skewness and no excess kurtosis whatever its covariance is. There is no
"skewness of the deltas"; the asymmetry is the gamma.

**A long-gamma book cannot lose more than a fixed amount, and that bound is
computable.** Completing the square in each coordinate gives
``inf L = a - 0.5 sum beta_i^2 / lambda_i`` when every eigenvalue is positive:
the loss is bounded below, full stop, by an amount :func:`support` returns. With
every eigenvalue negative the same expression is the *ceiling* and the loss is
unbounded, which is the short-gamma case and the one worth being frightened of.

**And that is where a delta-only model fails, in both directions.** Measured on
a three-factor book whose gamma is paid for by its theta, so the expected profit
is zero: the floor is 4.71% of notional, the exact 99% value at risk is 4.53%,
and a delta-normal number is 6.85% — 1.46 times a loss the book cannot sustain.
It passes through the floor at a confidence of 65.9% and never comes back. Flip
the sign of the gamma and the same arithmetic runs the other way: the book now
*receives* the theta, a delta-normal model credits it, drops the gamma it was
paid for taking, and reports a **negative** value at risk at 95% against a true
loss of 7.6% and 14.0% at 99%. The mechanism is the same in both cases — the
model charges or credits the theta and ignores what the theta bought — and only
one of the two errors is in the safe direction. On that book the curvature also
carries 92% of the variance and all of the skewness, which is 1.89 in absolute
value.

**The tail is a saddlepoint approximation, and its error is measured.** The
Lugannani-Rice expansion of the distribution function is used directly in
whichever tail is asked for, by negating the form rather than complementing the
other tail — ``1 - F`` is zero once the true tail is below 1e-16, so a complement
is not a tail. Two exact cases pin the accuracy. Equal eigenvalues with no linear
term is a scaled chi-square, and against it the worst absolute error over the
body is

========= ==========
factors   worst error
========= ==========
1         1.4e-02
2         2.8e-03
5         2.6e-04
10        5.7e-05
30        1.2e-05
========= ==========

so the expansion is asymptotic in the number of independent contributions rather
than uniformly accurate, and a book dominated by one direction of gamma is at the
top of that table. A single factor is also a quadratic in one normal whose
distribution function is elementary, and the agreement there is inside a per
cent, which is what the first row predicts.

**Two places where the arithmetic had to be written carefully.** The
cumulant generating function takes ``log1p(-lambda t)`` rather than
``log(1 - lambda t)``: the logarithm's argument is near one for a small
saddlepoint, ``1.0 - x`` there carries an absolute error of an ulp of one while
the answer is of size ``x``, and the quantity that error lands in is
``t K'(t) - K(t)``, itself a cancellation of size ``t^2``. With ``log`` the
distribution function was wrong by a per cent within a thousandth of a standard
deviation of the mean; with ``log1p`` it holds to 6e-05 at a millionth.

At the mean itself both correction terms vanish and their ratio does not, so the
difference is taken as a limit, and the switch is made on the dimensionless
``t sqrt(k2)`` rather than on the level equalling the mean — a level one rounding
from the mean has a saddlepoint of 1e-16, which is not zero, and the direct form
then divides two quantities that agree to every bit they have. Before that test
existed the distribution function read exactly ``1.0`` at the mean of a
one-dimensional form, which the clamp turned into a plausible number instead of a
visible one. The threshold is at the balance of the two errors and the
discontinuity across it is 4e-06, two orders inside the expansion's own error.

**The delta-normal comparison runs through the same code.** :func:`linear_form`
zeroes the curvature, and with no curvature the generating function is exactly
Gaussian and the expansion is exact, so :func:`quadratic_risk` on a linearised
form *is* the delta-normal answer and agrees with
:func:`~shortfall.parametric.normal_risk` to the last bit. There is no second
implementation to disagree with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .distributions import normal_cdf, normal_pdf, normal_ppf
from .linalg import Matrix, Vector, check_symmetric, cholesky, eigh

__all__ = [
    "NEAR_MEAN",
    "DidNotSolve",
    "Exposure",
    "NotQuadratic",
    "OutsideSupport",
    "QuadraticRisk",
    "StandardForm",
    "cgf",
    "cgf_curvature",
    "cgf_slope",
    "cumulants",
    "delta_normal_quantile",
    "domain",
    "linear_form",
    "negated",
    "quadratic_cdf",
    "quadratic_quantile",
    "quadratic_risk",
    "quadratic_shortfall",
    "quadratic_tail",
    "saddlepoint_argument",
    "standard_form",
    "support",
]

#: Where to stop evaluating the expansion's correction directly and take its
#: limit instead, measured on the dimensionless ``t sqrt(k2)``. At this value
#: the direct form still has eleven digits and the limit costs under 1e-6 of the
#: answer, which is two orders inside the expansion's own error.
NEAR_MEAN = 1e-5

#: How close to the edge of the generating function's domain a solve may go.
#: The expansion needs ``K''`` there and ``K''`` diverges at the edge, so the
#: last fraction of the interval is not usable.
EDGE = 1.0 - 1e-12


class NotQuadratic(ValueError):
    """Inputs that do not describe a second-order exposure."""


class OutsideSupport(ValueError):
    """A level the distribution puts no mass on either side of."""


class DidNotSolve(RuntimeError):
    """The saddlepoint equation did not converge inside the domain."""


@dataclass(frozen=True, slots=True)
class Exposure:
    """A book's first and second derivatives with respect to its risk factors.

    Attributes:
        delta: One entry per factor, in profit per unit of factor move.
        gamma: The symmetric matrix of second derivatives, in profit per unit
            squared. A matrix of zeros is a linear book and is allowed.
        constant: Profit at an unchanged factor level. Usually zero; carrying
            it means a theta or a carry term can be included without the caller
            having to shift the answer afterwards.
    """

    delta: Vector
    gamma: Matrix
    constant: float = 0.0

    def __post_init__(self) -> None:
        size = check_symmetric(self.gamma)
        if len(self.delta) != size:
            raise NotQuadratic(
                f"the delta vector has {len(self.delta)} entries and the gamma matrix "
                f"is {size} by {size}; they describe different factor sets"
            )
        for value in (*self.delta, self.constant):
            if not math.isfinite(value):
                raise NotQuadratic(f"every exposure must be finite, got {value!r}")
        for row in self.gamma:
            for value in row:
                if not math.isfinite(value):
                    raise NotQuadratic(f"every exposure must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class StandardForm:
    """A quadratic form in independent standard normals.

    ``L = constant + sum_i (linear_i u_i + 0.5 curvature_i u_i^2)``, with the
    curvatures in ascending order because that is the order
    :func:`~shortfall.linalg.eigh` returns and because the extremes of the list
    are the two numbers that decide whether the loss is bounded.

    Attributes:
        constant: The value at a zero factor move.
        linear: One coefficient per independent direction.
        curvature: The eigenvalue in each direction.
    """

    constant: float
    linear: tuple[float, ...]
    curvature: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.linear) != len(self.curvature):
            raise NotQuadratic(
                f"{len(self.linear)} linear coefficients against "
                f"{len(self.curvature)} curvatures"
            )

    @property
    def size(self) -> int:
        return len(self.linear)

    @property
    def is_linear(self) -> bool:
        """True when there is no curvature anywhere, so the form is normal."""
        return all(value == 0.0 for value in self.curvature)


def standard_form(exposure: Exposure, covariance: Matrix) -> StandardForm:
    """Whiten the factors and diagonalise the curvature.

    With ``Sigma = C C'`` the move ``x = C u`` has ``u`` standard normal, so the
    linear term becomes ``C' delta`` and the quadratic one ``C' Gamma C``.
    Diagonalising the latter with an orthogonal ``Q`` leaves
    ``beta = Q' C' delta`` and the eigenvalues as the curvatures. The rotation
    is applied to the linear term rather than the other way round, because a
    rotation of standard normals is standard normal and a rotation of the
    eigenvectors would not leave the quadratic form diagonal.

    Args:
        exposure: The book's derivatives.
        covariance: The factors' covariance matrix. Must be positive definite;
            a singular one is refused by the Cholesky factorisation with a
            message about shrinking or repairing it, because a factor set with
            an exact linear dependence has a direction in which the risk is
            undefined rather than zero.

    Returns:
        The form in independent coordinates.

    Raises:
        NotQuadratic: If the covariance and the exposure describe different
            factor sets.
    """
    size = check_symmetric(covariance)
    if size != len(exposure.delta):
        raise NotQuadratic(
            f"the exposure covers {len(exposure.delta)} factors and the covariance "
            f"matrix {size}"
        )
    factor = cholesky(covariance)
    # C' delta, with C lower triangular: row j of C' is column j of C.
    rotated_delta = [
        sum(factor[row][column] * exposure.delta[row] for row in range(column, size))
        for column in range(size)
    ]
    # C' Gamma C, formed as (C' Gamma) C so each product is one pass.
    left = [
        [
            sum(factor[row][i] * exposure.gamma[row][column] for row in range(i, size))
            for column in range(size)
        ]
        for i in range(size)
    ]
    middle: Matrix = [
        [
            sum(left[i][row] * factor[row][j] for row in range(j, size))
            for j in range(size)
        ]
        for i in range(size)
    ]
    # Jacobi needs exact symmetry and the two triangles differ in the last bit.
    for i in range(size):
        for j in range(i + 1, size):
            average = 0.5 * (middle[i][j] + middle[j][i])
            middle[i][j] = middle[j][i] = average
    values, vectors = eigh(middle)
    linear = tuple(
        sum(vectors[index][i] * rotated_delta[i] for i in range(size)) for index in range(size)
    )
    return StandardForm(exposure.constant, linear, tuple(values))


def negated(form: StandardForm) -> StandardForm:
    """The form of ``-L``, which is how the other tail is reached.

    Every coefficient changes sign. Reaching the upper tail this way rather than
    by complementing the lower one is the whole reason it exists: ``1 - F`` is
    identically zero once the true tail is below the rounding of one, so a
    complement is not a tail.
    """
    return StandardForm(
        -form.constant,
        tuple(-value for value in form.linear),
        tuple(-value for value in reversed(form.curvature)),
    )


def linear_form(form: StandardForm) -> StandardForm:
    """The same form with its curvature removed: the delta-normal view.

    With no curvature the generating function is exactly that of a normal and
    every approximation here is exact, so passing this to
    :func:`quadratic_risk` gives the delta-normal answer through the same code
    rather than through a second implementation of it.
    """
    return StandardForm(form.constant, form.linear, tuple(0.0 for _ in form.curvature))


def cumulants(form: StandardForm, order: int = 4) -> tuple[float, ...]:
    """The first ``order`` cumulants, in closed form.

    Each term of the third and fourth carries a factor of the curvature, which
    is the statement that a linear book has no skewness and no excess kurtosis
    however its factors are distributed.

    Args:
        form: The standardised quadratic form.
        order: How many cumulants to return. Between 1 and 4; beyond the fourth
            nothing here uses them and a general recursion would be a different
            piece of work.

    Returns:
        ``(k1, ..., k_order)``.

    Raises:
        NotQuadratic: If ``order`` is outside 1 to 4.
    """
    if not 1 <= order <= 4:
        raise NotQuadratic(f"cumulants are available to fourth order, asked for {order!r}")
    first = form.constant + 0.5 * sum(form.curvature)
    out = [first]
    if order >= 2:
        out.append(
            sum(
                0.5 * curve * curve + slope * slope
                for curve, slope in zip(form.curvature, form.linear, strict=True)
            )
        )
    if order >= 3:
        out.append(
            sum(
                curve**3 + 3.0 * slope * slope * curve
                for curve, slope in zip(form.curvature, form.linear, strict=True)
            )
        )
    if order >= 4:
        out.append(
            sum(
                3.0 * curve**4 + 12.0 * slope * slope * curve * curve
                for curve, slope in zip(form.curvature, form.linear, strict=True)
            )
        )
    return tuple(out)


def domain(form: StandardForm) -> tuple[float, float]:
    """The open interval on which the generating function is finite.

    ``1 - lambda t > 0`` for every eigenvalue, so the positive eigenvalues cap
    the argument above and the negative ones cap it below. A form with no
    curvature is finite everywhere and the interval is returned as infinite
    rather than as a large number.
    """
    lower = -math.inf
    upper = math.inf
    for value in form.curvature:
        if value > 0.0:
            upper = min(upper, 1.0 / value)
        elif value < 0.0:
            lower = max(lower, 1.0 / value)
    return lower, upper


def support(form: StandardForm) -> tuple[float, float]:
    """The range of values the form can actually take.

    With every eigenvalue positive the loss is bounded below by
    ``constant - 0.5 sum beta^2 / lambda``, which is the minimum of the
    completed squares. With every eigenvalue negative the same expression is an
    upper bound. A direction with zero curvature and a non-zero linear
    coefficient is unbounded both ways and removes both bounds; a direction with
    neither contributes nothing.
    """
    floor_ok = True
    ceiling_ok = True
    offset = 0.0
    for curve, slope in zip(form.curvature, form.linear, strict=True):
        if curve == 0.0:
            if slope != 0.0:
                floor_ok = ceiling_ok = False
            continue
        if curve > 0.0:
            ceiling_ok = False
        else:
            floor_ok = False
        offset += 0.5 * slope * slope / curve
    low = form.constant - offset if floor_ok else -math.inf
    high = form.constant - offset if ceiling_ok else math.inf
    if floor_ok and ceiling_ok:
        # Every coefficient is zero: the form is the constant.
        return form.constant, form.constant
    return low, high


def cgf(form: StandardForm, argument: float) -> float:
    """``K(t)``, the cumulant generating function.

    Raises:
        NotQuadratic: If the argument is outside :func:`domain`.
    """
    lower, upper = domain(form)
    if not lower < argument < upper:
        raise NotQuadratic(
            f"the generating function is finite on ({lower!r}, {upper!r}) and was "
            f"asked for {argument!r}"
        )
    total = form.constant * argument
    for curve, slope in zip(form.curvature, form.linear, strict=True):
        shrink = 1.0 - curve * argument
        # log1p, not log(1 - x). The logarithm is evaluated at an argument near
        # one for a small saddlepoint, and `1.0 - x` there carries an absolute
        # error of an ulp of one while the answer is of size x. Summed over the
        # directions that is enough error to swamp `t K'(t) - K(t)`, which is
        # itself a cancellation of size t^2 -- and that quantity is the
        # expansion's own argument, so the loss shows up as a distribution
        # function that is wrong by a per cent near the mean rather than as
        # anything that looks like rounding.
        total += -0.5 * math.log1p(-curve * argument)
        total += 0.5 * slope * slope * argument * argument / shrink
    return total


def cgf_slope(form: StandardForm, argument: float) -> float:
    """``K'(t)``, which is the mean of the exponentially tilted distribution."""
    lower, upper = domain(form)
    if not lower < argument < upper:
        raise NotQuadratic(
            f"the generating function is finite on ({lower!r}, {upper!r}) and was "
            f"asked for {argument!r}"
        )
    total = form.constant
    for curve, slope in zip(form.curvature, form.linear, strict=True):
        shrink = 1.0 - curve * argument
        total += 0.5 * curve / shrink + 0.5 * slope * slope * argument * (
            2.0 - curve * argument
        ) / (shrink * shrink)
    return total


def cgf_curvature(form: StandardForm, argument: float) -> float:
    """``K''(t)``, the tilted variance. Strictly positive unless the form is a point."""
    lower, upper = domain(form)
    if not lower < argument < upper:
        raise NotQuadratic(
            f"the generating function is finite on ({lower!r}, {upper!r}) and was "
            f"asked for {argument!r}"
        )
    total = 0.0
    for curve, slope in zip(form.curvature, form.linear, strict=True):
        shrink = 1.0 - curve * argument
        total += 0.5 * curve * curve / (shrink * shrink) + slope * slope / shrink**3
    return total


def _bracket(form: StandardForm, level: float) -> tuple[float, float]:
    """A sign-changing bracket for ``K'(t) = level`` inside the domain."""
    lower, upper = domain(form)
    # Start from a scale set by the distribution rather than from 1: a book
    # whose cumulants are in units of 1e-4 has a saddlepoint of order 1e4.
    second = cumulants(form, 2)[1]
    scale = 1.0 / math.sqrt(second) if second > 0.0 else 1.0
    low = max(lower * EDGE, -scale * 1e8) if math.isfinite(lower) else -scale * 1e8
    high = min(upper * EDGE, scale * 1e8) if math.isfinite(upper) else scale * 1e8
    if cgf_slope(form, low) > level or cgf_slope(form, high) < level:
        raise DidNotSolve(
            f"the level {level!r} is not reachable by K' inside the domain "
            f"({lower!r}, {upper!r}); it sits outside the support"
        )
    return low, high


def saddlepoint_argument(form: StandardForm, level: float) -> float:
    """Solve ``K'(t) = level`` for ``t``, by bisection on a monotone function.

    ``K''`` is strictly positive, so ``K'`` is strictly increasing and the
    bracket has exactly one root. Bisection rather than Newton because the
    derivative blows up at the edges of the domain and a Newton step from near
    one of them leaves the interval, where ``K'`` is not defined at all.

    Raises:
        DidNotSolve: If the level is not in the open support.
    """
    low, high = _bracket(form, level)
    for _ in range(300):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if cgf_slope(form, middle) < level:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def quadratic_cdf(form: StandardForm, level: float) -> float:
    """``P(L <= level)``, by the Lugannani-Rice expansion.

    At the mean the expansion's two terms each diverge and their difference does
    not; the limit is ``0.5 + k3 / (6 sqrt(2 pi) k2^(3/2))``, and the branch is
    taken on the saddlepoint being numerically zero rather than on the level
    equalling the mean, because the second test fails on a level a rounding
    away from the mean while the first does not.

    Outside the support the answer is exact rather than approximate: zero below
    a floor and one above a ceiling.
    """
    low, high = support(form)
    if level <= low:
        return 0.0
    if level >= high:
        return 1.0
    if form.is_linear:
        mean, variance = cumulants(form, 2)
        if variance == 0.0:
            return 1.0 if level >= mean else 0.0
        return normal_cdf((level - mean) / math.sqrt(variance))

    argument = saddlepoint_argument(form, level)
    moments = cumulants(form, 3)
    middle = 0.5 + moments[2] / (6.0 * math.sqrt(2.0 * math.pi) * math.pow(moments[1], 1.5))
    # Near the mean both correction terms go to zero like the saddlepoint and
    # their difference does not, so the difference is evaluated as a limit. The
    # test is on the dimensionless t sqrt(k2) rather than on the level being the
    # mean: a level one rounding away from the mean has a saddlepoint of 1e-16,
    # which is not zero, and the direct form then divides two quantities that
    # agree to every bit they have. That is how this went wrong the first time
    # -- the distribution function read exactly 1.0 at the mean of a
    # one-dimensional form, which the clamp turned into a plausible number
    # rather than a visible one.
    if abs(argument) * math.sqrt(moments[1]) < NEAR_MEAN:
        return middle
    spread = cgf_curvature(form, argument)
    exponent = argument * level - cgf(form, argument)
    radical = math.sqrt(2.0 * max(exponent, 0.0))
    signed = radical if argument > 0.0 else -radical
    scaled = argument * math.sqrt(spread)
    if signed == 0.0 or scaled == 0.0:  # pragma: no cover - the branch above fires first
        return middle
    value = normal_cdf(signed) + normal_pdf(signed) * (1.0 / signed - 1.0 / scaled)
    return min(max(value, 0.0), 1.0)


def quadratic_tail(form: StandardForm, level: float) -> float:
    """``P(L > level)``, computed in its own right rather than as ``1 - F``.

    The form is negated and the lower tail of ``-L`` is taken, so the answer is
    a small number computed from small numbers. Complementing the distribution
    function instead would return exactly zero for every level beyond about six
    standard deviations.
    """
    return quadratic_cdf(negated(form), -level)


def quadratic_quantile(form: StandardForm, probability: float) -> float:
    """The level ``q`` with ``P(L <= q) = probability``.

    Found by bisection on the distribution function, which is monotone by
    construction. The bracket starts from the support when it is finite and
    otherwise walks outwards from the mean in units of the standard deviation,
    which is bounded because the distribution function is continuous and
    strictly monotone on the support.

    Raises:
        OutsideSupport: If the probability is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < probability < 1.0:
        raise OutsideSupport(
            f"a quantile needs a probability strictly inside (0, 1), got {probability!r}"
        )
    mean, variance = cumulants(form, 2)
    if variance == 0.0:
        return mean
    spread = math.sqrt(variance)
    low, high = support(form)
    left = low if math.isfinite(low) else mean - 40.0 * spread
    right = high if math.isfinite(high) else mean + 40.0 * spread
    # The support's endpoint is attained with probability zero, so step just
    # inside it before bisecting.
    width = right - left
    left += 1e-13 * width
    right -= 1e-13 * width
    for _ in range(200):
        middle = 0.5 * (left + right)
        if middle in (left, right):
            break
        if quadratic_cdf(form, middle) < probability:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right)


def quadratic_shortfall(form: StandardForm, probability: float, *, panels: int = 512) -> float:
    """``E[L | L <= q]`` at the quantile ``q`` for the given tail probability.

    Written as ``q - E[(q - L)+] / alpha`` and the expectation as
    ``int F(x) dx`` from the bottom of the support up to ``q``. Integrating the
    distribution function rather than a density is the point: there is no
    density here, only a saddlepoint expansion of ``F``, and differentiating an
    approximation to get one would add an order of error for nothing.

    When the support is bounded below the integral is over a finite interval and
    is exact up to the quadrature. When it is not, the lower limit is walked
    outwards until the distribution function there contributes less than a
    relative 1e-12 of the integral so far, and the walk is refused if it does
    not get there — an unconverged tail integral reported as a risk number is
    worse than an error.
    """
    quantile = quadratic_quantile(form, probability)
    low, _ = support(form)
    if math.isfinite(low):
        integral = _simpson(form, low, quantile, panels)
    else:
        spread = math.sqrt(cumulants(form, 2)[1])
        integral = 0.0
        edge = quantile
        for _ in range(80):
            step = max(spread, abs(edge) * 0.5)
            piece = _simpson(form, edge - step, edge, panels)
            integral += piece
            edge -= step
            if piece <= 1e-12 * integral or quadratic_cdf(form, edge) == 0.0:
                break
        else:  # pragma: no cover - needs a distribution with no left tail decay
            raise DidNotSolve(
                "the tail integral for the expected shortfall did not converge; the "
                "distribution function is not decaying towards the left"
            )
    return quantile - integral / probability


def _simpson(form: StandardForm, low: float, high: float, panels: int) -> float:
    """Composite Simpson of the distribution function over ``[low, high]``."""
    if high <= low:
        return 0.0
    panels = panels + (panels % 2)
    step = (high - low) / panels
    total = quadratic_cdf(form, low) + quadratic_cdf(form, high)
    for index in range(1, panels):
        weight = 4.0 if index % 2 else 2.0
        total += weight * quadratic_cdf(form, low + index * step)
    return total * step / 3.0


@dataclass(frozen=True, slots=True)
class QuadraticRisk:
    """A risk estimate for a second-order book, with the bound it cannot pass.

    Attributes:
        value_at_risk: Positive loss, the negated quantile.
        expected_shortfall: Positive loss, and never below
            :attr:`value_at_risk`.
        quantile: The signed profit quantile the loss was taken from.
        confidence: The confidence the estimate was taken at.
        floor: The worst outcome the book admits, as a positive loss, or
            infinity when the loss is unbounded. A long-gamma book has a finite
            one and a short-gamma book does not.
        cumulants: The first four cumulants of the profit distribution.
    """

    value_at_risk: float
    expected_shortfall: float
    quantile: float
    confidence: float
    floor: float
    cumulants: tuple[float, ...]

    @property
    def skewness(self) -> float:
        """``k3 / k2^(3/2)``, which is zero exactly when the book is linear."""
        return self.cumulants[2] / math.pow(self.cumulants[1], 1.5)

    @property
    def excess_kurtosis(self) -> float:
        return self.cumulants[3] / self.cumulants[1] ** 2

    @property
    def exceeds_the_floor(self) -> bool:
        """True when the quoted loss is larger than the book can actually lose.

        Never true of an estimate from this module, and the reason the field is
        here: it is what a delta-normal number does at a high enough confidence
        on a long-gamma book, and the comparison is only meaningful if both
        sides report it.
        """
        return self.value_at_risk > self.floor


def quadratic_risk(form: StandardForm, confidence: float) -> QuadraticRisk:
    """Value at risk and expected shortfall for a second-order book.

    Args:
        form: The standardised quadratic form, from :func:`standard_form`.
        confidence: Strictly inside ``(0, 1)``; 0.99 for a 99% estimate.

    Returns:
        The estimate, the cumulants behind it, and the loss the book cannot
        exceed.

    Raises:
        OutsideSupport: If the confidence is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise OutsideSupport(
            f"a confidence must lie strictly inside (0, 1), got {confidence!r}"
        )
    alpha = 1.0 - confidence
    quantile = quadratic_quantile(form, alpha)
    tail_mean = quadratic_shortfall(form, alpha)
    low, _ = support(form)
    floor = math.inf if low == -math.inf else -low
    return QuadraticRisk(
        value_at_risk=-quantile,
        expected_shortfall=-tail_mean,
        quantile=quantile,
        confidence=confidence,
        floor=floor,
        cumulants=cumulants(form, 4),
    )


def delta_normal_quantile(form: StandardForm, probability: float) -> float:
    """The quantile the linearised book would quote, for comparison.

    Exactly ``mu + sigma z`` on the *linear* part's own mean and variance —
    which is not the full form's variance, because the curvature contributes to
    that too. Quoting the full variance with a normal shape would be a third
    thing again, and is not what a delta-normal system does.
    """
    linear = linear_form(form)
    mean, variance = cumulants(linear, 2)
    return mean + math.sqrt(variance) * normal_ppf(probability)

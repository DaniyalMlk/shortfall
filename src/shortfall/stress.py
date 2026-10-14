"""The worst a plausible set of factor moves can do, and what those moves are.

Every other risk number here is a quantile or a tail average of a distribution.
A risk committee also asks something else: *what is the worst this book can lose
if the factors move in a way we would not call absurd, and what would they have
to do?* That is not a quantile. It has an answer, the answer is exact, and it
comes with a scenario attached.

The plausible set is the Mahalanobis ellipsoid ``x' Sigma^-1 x <= radius**2``.
It is the natural choice because it is the set the covariance itself defines:
every point on its boundary is equally unlikely under a normal model, so no
direction is privileged by the shape of the set rather than by the book. Under
that model ``x' Sigma^-1 x`` is chi-square with one degree of freedom per
factor, which is how :func:`radius_for` turns a probability into a radius.

In the coordinates :func:`~shortfall.quadratic.whitening` produces -- factors
whitened, curvature diagonalised -- the ellipsoid *is* the Euclidean ball, and
maximising a quadratic over a ball is the trust-region subproblem. With the
Hessian already diagonal its solution is a scalar equation in one unknown, so
nothing here is an optimiser.

**For a linear book the whole exercise is a volatility multiple, and saying so
is more use than hiding it.** The worst loss is ``radius * sqrt(delta' Sigma
delta)``, attained in the single direction ``-Sigma delta`` — which is
parametric value at risk with a different constant in front. So for a book
with no curvature a worst-case stress test and a VaR are the same calculation
and the only live question is which multiplier to use. The framing starts
earning its keep exactly where the curvature does.

**One factor at a time understates the joint worst case by a factor that grows
with the factor set.** Comparing the two over the *same* ellipsoid, so that
the only difference is that one search covers all of it, the median ratio of
the joint worst case to the worst single-factor loss is 1.35 over two factors,
2.04 over three, 4.09 over five and 6.76 over eight, across four hundred
random books each. For books with no curvature the same medians are 1.15,
1.43, 1.94 and 2.52, so about half the gap is the correlation and half is the
curvature.

**And it understates the *sum* of its own per-factor losses too, which is the
number a report treats as the conservative one.** The joint worst case exceeds
that sum in 239 of 275 random three-factor books, 275 of 288 at five factors
and 278 of 279 at eight, by a median factor of 1.84, 2.93 and 4.78. Adding up
independent stresses is not a bound on anything: it is a different number that
no single point in the factor space produces.

**The usual version of the test is not even inside the set.** "Move each
factor by three of its own standard deviations" places ``e_i`` at Mahalanobis
length ``sigma_i sqrt((Sigma^-1)_ii)``, which is at least one and equals one
only when the factor is uncorrelated with every other. Over random correlated
covariances that happens never: the move was outside the ellipsoid in 300 of
300 draws at every factor count tried. So the common stress test is neither a
lower bound nor a subset, and it still comes out smaller.

**The hard case is an ordinary book, not a corner.** When the direction of
fastest loss growth carries no linear exposure the secular equation has no
root, and that is precisely a delta-hedged short-gamma position. A two-factor
book with gamma ``diag(-120, -20)`` and no delta loses 22.10 at the 99%
radius, with the whole move in the first factor and the second untouched; and
there the one-at-a-time test is *exact*, because the factors are uncorrelated
and the worst direction is an axis. That is the one shape where the naive test
is right, and it is worth knowing which one it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .distributions import chi_square_ppf
from .linalg import Matrix, Vector, cholesky
from .quadratic import Exposure, NotQuadratic, StandardForm, whitening

__all__ = [
    "MaximumLoss",
    "OneAtATime",
    "Scenario",
    "linear_maximum_loss",
    "maximum_loss",
    "one_at_a_time",
    "radius_for",
    "worst_case_scenario",
]

# The secular equation is solved by bisection on a bracket grown outwards. The
# count is a guard against a non-finite input slipping through, not a tuning
# parameter: 200 halvings exhaust a double long before this matters.
_BISECTIONS = 200

# Curvatures within this multiple of the largest, in absolute value, count as
# the same eigenvalue when deciding whether the hard case has been reached.
# A relative tolerance rather than an absolute one, because the curvatures
# carry the units of the book.
_DEGENERATE = 1e-12


def radius_for(probability: float, factors: int) -> float:
    """The Mahalanobis radius of the ellipsoid containing this much probability.

    Under a normal model ``x' Sigma^-1 x`` is chi-square on ``factors``
    degrees of freedom, so the radius is the square root of its quantile.

    The number grows with the factor set, and that is the point rather than an
    artefact: the 99% radius is 2.58 over one factor, 3.37 over three, 4.82
    over ten and 8.73 over fifty. Read the other way round, "a three standard
    deviation move" names a set holding 99.73% of the probability over one
    factor, 97.07% over three, 46.79% over ten and 1.9e-11 over fifty. Those
    are not the same stress test, and the phrase does not distinguish them.
    """
    if factors < 1:
        raise NotQuadratic(f"a factor set needs at least one factor, got {factors}")
    return math.sqrt(chi_square_ppf(probability, float(factors)))


@dataclass(frozen=True, slots=True)
class MaximumLoss:
    """The worst loss over an ellipsoid, and the point that attains it.

    Attributes:
        loss: The largest loss, as a positive number for a loss.
        point: The worst move in the independent coordinates. Its norm is the
            radius unless the maximum is interior.
        radius: The radius it was computed at.
        multiplier: The Lagrange multiplier on the radius constraint. Zero
            exactly when the maximum is interior.
        interior: True when the worst point is strictly inside the ellipsoid,
            which happens only for a book with loss-bounded curvature in every
            direction.
        hard_case: True when the multiplier had to be pinned at the negative of
            the extreme curvature because the secular equation had no root --
            the classic hard case, reached whenever the most dangerous
            direction carries no linear exposure.
    """

    loss: float
    point: Vector
    radius: float
    multiplier: float
    interior: bool
    hard_case: bool


def _loss_at(form: StandardForm, point: Vector) -> float:
    """The loss -- positive for a loss -- at a point in the independent coordinates."""
    profit = form.constant + math.fsum(
        form.linear[i] * point[i] + 0.5 * form.curvature[i] * point[i] ** 2
        for i in range(form.size)
    )
    return -profit


def maximum_loss(form: StandardForm, radius: float) -> MaximumLoss:
    """The exact worst loss over the ball of this radius, through the secular equation.

    Maximising the loss ``-constant + sum(g_i u_i + 0.5 b_i u_i**2)`` over
    ``||u|| <= radius``, with ``b_i = -k_i`` the loss curvature and
    ``g_i = -l_i`` the loss gradient, is a trust-region subproblem. The
    stationarity condition is ``(m - b_i) u_i = g_i`` for a multiplier ``m``
    with ``m >= max(0, max(b))``, so that the shifted Hessian has the sign a
    maximum needs, and ``m`` fixed by ``||u|| = radius``. The whole problem is
    then the scalar equation

        ``sum_i g_i**2 / (m - b_i)**2 = radius**2``

    whose left side decreases in ``m``, so one bisection settles it.

    **The hard case is not a corner to be guarded against, it is a book shape.**
    The binding eigenvalue is the *largest* loss curvature -- the direction in
    which the loss grows fastest. If that direction carries no linear exposure
    then ``g_i = 0`` there, the term drops out of the sum, and the equation can
    have no root: the left side at ``m = max(b)`` is already below
    ``radius**2``. The answer is then to pin ``m`` at ``max(b)``, take the
    solution in the other coordinates, and spend whatever norm is left along
    the degenerate direction. A short gamma position with no delta is exactly
    that book, so this is the ordinary case for an option book that has been
    delta-hedged rather than an edge case.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise NotQuadratic(f"the radius must be positive and finite, got {radius!r}")
    size = form.size
    if size == 0:
        raise NotQuadratic("a form with no factors has no worst case")
    # In loss coordinates: L(u) = -constant + sum(g_i u_i + 0.5 b_i u_i**2).
    curvature = [-value for value in form.curvature]
    gradient = [-value for value in form.linear]
    largest = max(curvature)
    scale = max(abs(value) for value in curvature) + 1.0

    def point_at(multiplier: float) -> Vector:
        return [
            gradient[i] / (multiplier - curvature[i])
            if multiplier - curvature[i] != 0.0
            else 0.0
            for i in range(size)
        ]

    def norm_at(multiplier: float) -> float:
        return math.sqrt(math.fsum(value * value for value in point_at(multiplier)))

    # Interior: only when the loss is concave in every direction, so that an
    # unconstrained maximum exists at all.
    if largest < 0.0 and norm_at(0.0) <= radius:
        interior_point = point_at(0.0)
        return MaximumLoss(
            loss=_loss_at(form, interior_point),
            point=interior_point,
            radius=radius,
            multiplier=0.0,
            interior=True,
            hard_case=False,
        )

    floor = max(0.0, largest)
    degenerate = [i for i in range(size) if abs(curvature[i] - largest) <= _DEGENERATE * scale]
    unexposed = all(gradient[i] == 0.0 for i in degenerate)
    if unexposed and norm_at(floor) <= radius:
        pinned = point_at(floor)
        for index in degenerate:
            pinned[index] = 0.0
        used = math.fsum(value * value for value in pinned)
        # The whole remaining norm goes along one degenerate direction; the
        # loss is the same for any split between them and for either sign.
        pinned[degenerate[0]] = math.sqrt(max(0.0, radius * radius - used))
        return MaximumLoss(
            loss=_loss_at(form, pinned),
            point=pinned,
            radius=radius,
            multiplier=floor,
            interior=False,
            hard_case=True,
        )

    low = floor
    high = floor + scale
    for _ in range(_BISECTIONS):
        if norm_at(high) < radius:
            break
        high += max(scale, high - floor)
    else:  # pragma: no cover - the norm falls like 1/m, so growing always wins
        raise NotQuadratic(f"could not bracket the secular equation at radius {radius!r}")
    for _ in range(_BISECTIONS):
        middle = 0.5 * (low + high)
        if norm_at(middle) > radius:
            low = middle
        else:
            high = middle
    multiplier = 0.5 * (low + high)
    solved = point_at(multiplier)
    return MaximumLoss(
        loss=_loss_at(form, solved),
        point=solved,
        radius=radius,
        multiplier=multiplier,
        interior=False,
        hard_case=False,
    )


def linear_maximum_loss(delta: Vector, covariance: Matrix, radius: float) -> float:
    """The worst loss over the ellipsoid for a book with no curvature.

    ``radius * sqrt(delta' Sigma delta)`` -- the radius times the book's own
    volatility, attained in the single direction ``-Sigma delta`` normalised.

    Which is the honest answer to whether the ellipsoid framing buys anything
    for a linear book: it does not. Parametric value at risk is the same
    number with a different constant in front, so for a linear book a
    worst-case stress test and a VaR are the same calculation and the only
    question is which multiplier to use. The framing starts to matter exactly
    where the curvature does, which is what :func:`maximum_loss` is for.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise NotQuadratic(f"the radius must be positive and finite, got {radius!r}")
    form = whitening(Exposure(delta=delta, gamma=_zeros(len(delta))), covariance).form
    return radius * math.sqrt(math.fsum(value * value for value in form.linear))


def _zeros(size: int) -> Matrix:
    return [[0.0] * size for _ in range(size)]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A worst case with the factor move that produces it.

    Attributes:
        loss: The largest loss over the ellipsoid, positive for a loss.
        move: The factor move that attains it, in the factors' own units.
        standardised: The same move divided by each factor's standard
            deviation, which is how a stress scenario is read: "oil down two
            and a half sigma, rates up one".
        detail: The solution in the independent coordinates, including whether
            the hard case was reached.
    """

    loss: float
    move: Vector
    standardised: Vector
    detail: MaximumLoss

    @property
    def mahalanobis(self) -> float:
        """The move's distance in standard deviations of the joint distribution."""
        return math.sqrt(math.fsum(value * value for value in self.detail.point))


def worst_case_scenario(exposure: Exposure, covariance: Matrix, radius: float) -> Scenario:
    """The worst loss over the ellipsoid, and the factor move that causes it.

    The move is the point the trust-region solution picks out, mapped back
    through the whitening. Reporting it is most of the value of the exercise:
    a committee that is told the number learns how bad things can get, and a
    committee that is told the move learns what to watch.

    The sign is not determined in one case. In the hard case the loss is even
    in the degenerate coordinate, so the move and its mirror image lose
    exactly the same amount, and the one returned is the positive branch.
    """
    transform = whitening(exposure, covariance)
    solved = maximum_loss(transform.form, radius)
    move = transform.to_factors(solved.point)
    deviations = [math.sqrt(covariance[i][i]) for i in range(len(move))]
    return Scenario(
        loss=solved.loss,
        move=move,
        standardised=[
            move[i] / deviations[i] if deviations[i] > 0.0 else 0.0 for i in range(len(move))
        ],
        detail=solved,
    )


@dataclass(frozen=True, slots=True)
class OneAtATime:
    """What a stress test that moves one factor at a time finds.

    Attributes:
        losses: The worse of the two signed moves, per factor.
        worst: The largest single-factor loss.
        worst_factor: Which factor produced it.
        total: The sum of the per-factor losses, which is the other number
            such a report usually carries. It is not a loss the book can
            suffer: no single point in the factor space produces it.
        moves: The size of the move applied to each factor, in its own units.
        inside: Whether every applied move lay inside the ellipsoid.
    """

    losses: Vector
    worst: float
    worst_factor: int
    total: float
    moves: Vector
    inside: bool


def _inverse_diagonal(covariance: Matrix) -> Vector:
    """The diagonal of the inverse covariance, through the Cholesky factor.

    ``(Sigma^-1)_ii = e_i' Sigma^-1 e_i``, obtained by forward- and
    back-substituting the unit vector rather than forming an inverse.
    """
    factor = cholesky(covariance)
    size = len(factor)
    diagonal: Vector = []
    for index in range(size):
        forward = [0.0] * size
        for row in range(size):
            target = 1.0 if row == index else 0.0
            forward[row] = (
                target - math.fsum(factor[row][k] * forward[k] for k in range(row))
            ) / factor[row][row]
        back = [0.0] * size
        for row in range(size - 1, -1, -1):
            back[row] = (
                forward[row] - math.fsum(factor[k][row] * back[k] for k in range(row + 1, size))
            ) / factor[row][row]
        diagonal.append(back[index])
    return diagonal


def one_at_a_time(
    exposure: Exposure,
    covariance: Matrix,
    radius: float,
    *,
    on_the_ellipsoid: bool = True,
) -> OneAtATime:
    """The stress test a risk report actually contains, for comparison.

    Each factor is moved on its own, up and down, and the worse of the two is
    recorded. The question is what size of move.

    ``on_the_ellipsoid`` scales each single-factor move so that it lands
    exactly on the boundary of the same ellipsoid :func:`maximum_loss` works
    over: ``radius / sqrt((Sigma^-1)_ii)``. That makes the comparison fair,
    because both tests are then searching the same set and the only difference
    is that one of them searches all of it.

    With it false, each factor is moved ``radius`` of its *own* standard
    deviations, which is what a report means by "a three sigma move in each
    risk factor". Those points are generally **outside** the ellipsoid --
    ``e_i`` has Mahalanobis length ``sigma_i sqrt((Sigma^-1)_ii) >= 1``, with
    equality only when the factor is uncorrelated with every other -- so that
    version is not a lower bound on the joint worst case by construction. It
    is still smaller in practice, which is the finding worth having rather
    than the one worth assuming.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise NotQuadratic(f"the radius must be positive and finite, got {radius!r}")
    size = len(exposure.delta)
    if len(covariance) != size:
        raise NotQuadratic(
            f"the exposure covers {size} factors and the covariance matrix {len(covariance)}"
        )
    inverse = _inverse_diagonal(covariance)
    losses: Vector = []
    moves: Vector = []
    inside = True
    for index in range(size):
        if on_the_ellipsoid:
            step = radius / math.sqrt(inverse[index])
        else:
            step = radius * math.sqrt(covariance[index][index])
            if step * step * inverse[index] > radius * radius * (1.0 + 1e-12):
                inside = False
        best = -math.inf
        for sign in (1.0, -1.0):
            shift = sign * step
            profit = (
                exposure.constant
                + exposure.delta[index] * shift
                + 0.5 * exposure.gamma[index][index] * shift * shift
            )
            best = max(best, -profit)
        losses.append(best)
        moves.append(step)
    worst_factor = max(range(size), key=lambda index: losses[index])
    return OneAtATime(
        losses=losses,
        worst=losses[worst_factor],
        worst_factor=worst_factor,
        total=math.fsum(losses),
        moves=moves,
        inside=inside,
    )

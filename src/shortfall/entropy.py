"""Putting a view on a scenario set without throwing the scenario set away.

Every other estimator here reads a sample and takes it as given. That leaves no
way to ask what the risk would be *if* something were true, and the usual answer
— keep the scenarios that match the story and drop the rest — destroys the thing
that made the sample worth using. Dropping the two thirds of a history that are
not recessions does not produce a recession scenario set; it produces a third of
a sample, with the dependence between assets re-estimated on whatever is left.

Entropy pooling reweights instead. Among all distributions on the *existing*
scenarios that satisfy the view, take the one closest to the original in
relative entropy. Nothing is discarded, the view holds exactly, and everything
the view did not mention moves only as far as the sample's own dependence
implies.

**The dual is the problem to solve, and that is the decision this module turns
on.** The primal has one unknown per scenario — a thousand, ten thousand — with
``k + 1`` equality constraints. Its dual has one unknown per *view*, which is
almost always fewer than five. Better than the dimension: the dual's gradient is
the view residual itself and its Hessian is the posterior covariance of the view
functions, so Newton's method is available, the Hessian is positive semidefinite
by construction rather than by hope, and the stopping test is a statement about
the thing the caller asked for rather than about a step length. Every solve in
the tests converges in five to eight steps.

The identity that makes the two sides line up is worth stating, because it is
also the module's cheapest check:

    relative entropy of the solution  =  minus the optimal dual value

Both are computed, from the weights and from the multipliers, and they agree to
**2.6e-13** across the test grid — the relative entropy's own cancellation,
since it sums a thousand terms of mixed sign to reach a number near 1e-03. That
catches a sign dropped in the objective, which on a convex problem would
otherwise still converge: to the wrong point, smoothly, with a small gradient.

**A probability view has a closed form, and it is the only real check
available.** Reweighting to make ``P(S) = t`` scales the weights inside ``S`` by
``t / P(S)`` and outside by ``(1 - t) / (1 - P(S))``, and nothing else can
change, because relative entropy is minimised cell by cell on a partition. So
the solver's answer on that problem is checkable against arithmetic rather than
against a second optimiser: over twenty combinations of threshold and target it
agrees to **1.4e-16** on the weights and **5.1e-13** on the relative entropy.
Views on a mean have no closed form, which is why the probability case is where
the implementation is pinned.

**A step that only has to lower the objective stops three orders of magnitude
early.** The textbook line search accepts a step when the objective strictly
decreases, and the dual value of a small-probability view is itself around
1e-04: it goes flat to the last bit of a double while the residual is still
1e-09. Accepting a step that improves the objective *or* the residual — the
residual being what the caller asked about — takes the same solves from a worst
residual of 1e-09 to 1.4e-16, and the relative entropy from 2.3e-08 relative to
5.1e-13. Before that, the non-strict version of the same test was worse again:
at the optimum the objective stops changing at all, so ``<=`` accepts
zero-progress steps until the iteration budget runs out and a converged solve is
reported as a failure to converge.

**Four failures, four messages.** A target outside the range the scenarios
span, a view that is constant, two views that are linearly dependent, and two
views that are each reachable and jointly impossible are different faults with
different fixes, and a solver that reports them all as "did not converge" is
not much better than one that returns a number. The first two are caught before
the solve starts. The last two both arrive as a singular Hessian, and telling
them apart took a test: a Hessian that is singular *at the prior* means the view
functions are dependent over the scenarios and no tilt separates them, while one
that goes singular later means the tilt has already driven the weights onto a
face of the simplex, where functions independent over the whole set no longer
are — which is what happens on the way out of the feasible region. The residual
is what distinguishes them, and the first version of this module reported two
disjoint sets each asked for a probability of 0.6 as linearly dependent views.

**An already-satisfied view returns the prior exactly.** The multipliers are
zero, so the exponential tilt is ``exp(0)`` and the weights are the prior's own
floats — not a copy that round-tripped through a normalisation. A caller
comparing a stressed number against an unstressed one gets ``0.0`` for the
difference rather than 1e-17, which matters because the comparison is usually
printed as a percentage change.

**What a mean view does to a quantity it never mentioned is the sample's own
regression coefficient, to four figures.** On a thousand scenarios of two
correlated series, a view that moves the first series' mean down by ``d`` moves
the second's by the least-squares slope times ``d``: the slope is 0.652444 and
the implied ratios at ``d`` of 0.1, 0.25, 0.5 and 1.0 standard deviations are
0.652141, 0.651922, 0.652120 and 0.653246 — agreement to between 0.05% and
0.12% over a tenfold range of view strength. Nothing linear was assumed
anywhere; the exponential tilt reproduces the projection. Which is worth knowing
in both directions: it says the machinery is not inventing a relationship, and
it says that for *means* a regression would have done.

**The tail does not follow the mean, and that is what the machinery is for.**
The same views move the second series' expected shortfall by far less than a
parallel shift of its distribution would. At a view of half a standard deviation
the mean moves by 0.3261 — the regression amount — while the 99% expected
shortfall moves by 0.1263, which is **0.39** of it, and the 95% by 0.81 of it.
The attenuation deepens both further into the tail and as the view strengthens:
at a full standard deviation the figures are 0.35 and 0.63. So the shortcut
everybody reaches for — shift the loss distribution by beta times the view —
overstates a 99% expected shortfall by between **2.2 and 2.9 times** here. The
reweighting concentrates on scenarios where the first series was low, and the
second series' worst scenarios are only partly those.

**The usual confidence blend is optimal exactly when the view is on a
partition, and not otherwise.** Taking ``(1 - c) p + c q`` is what partial
confidence usually means, and it produces a distribution whose view value is
``c`` of the way from the prior to the target. For a *probability* view it is
also the entropy-minimising distribution for that intermediate target, exactly
and at every confidence — measured at 0.0% excess entropy — because scaling a
partition's cells and then blending two such scalings leaves the weights within
each cell still proportional. For a *mean* view it is not, because a blend of
two exponential tilts is not an exponential tilt, and the cost is largest where
nobody would look for it: at **low** confidence in a **strong** view. On a view
moving the mean a full standard deviation the blend carries **31.8%** more
relative entropy than re-solving at the value it actually produced when
``c = 0.25``, 13.8% at 0.5 and 4.1% at 0.75; on a quarter-standard-deviation
view the same three are 1.9%, 0.85% and 0.22%. :func:`temper` is provided
because the blend is what people mean, and it reports its own entropy so the
comparison can be made rather than assumed away.

**The effective number of scenarios is the diagnostic that decides whether to
believe the answer.** A view in the body of the distribution costs almost
nothing; a view in the tail spends the sample. On a thousand scenarios drawn
from a standard normal, doubling the probability of a loss worse than two
standard deviations leaves **993** effective scenarios and tripling it leaves
976, but doubling the probability of a loss worse than *one* standard deviation
leaves 934 and tripling it 785 — and a mean view of a full standard deviation
leaves **569**, a set reporting a 99% expected shortfall from rather fewer
observations than its length suggests. The number is returned with every solve
rather than offered as an option.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .linalg import NotPositiveDefinite, cholesky
from .parametric import Distribution, Risk
from .series import Misaligned, TooShort

__all__ = [
    "MAX_POOL_ITERATIONS",
    "POOL_TOLERANCE",
    "STALL_TOLERANCE",
    "Posterior",
    "View",
    "ViewsInfeasible",
    "ViewsNotIdentified",
    "effective_scenarios",
    "mean_view",
    "pool",
    "probability_view",
    "relative_entropy",
    "stressed_risk",
    "temper",
    "weighted_expected_shortfall",
    "weighted_quantile",
]

#: Newton iterations before the solve gives up. The dual is convex and smooth
#: and the Hessian is exact, so a feasible problem converges in well under ten;
#: a hundred is a bound on pathology rather than a budget to be spent.
MAX_POOL_ITERATIONS = 100

#: Default stopping tolerance, on the view residual divided by the span of the
#: view function over the scenarios. Dimensionless on purpose: a view on a
#: return and a view on a probability are not measured in the same units, and a
#: tolerance stated in the units of either one is wrong for the other.
#:
#: Set at round-off rather than at a comfortable margin, because the relative
#: entropy's sensitivity to the residual is the multiplier itself — a view met
#: to 1e-12 with a multiplier of a hundred leaves the entropy wrong in its
#: eleventh digit, and the entropy is what the caller reads to decide whether
#: the view was expensive. Newton converges quadratically, so the last digits
#: cost one extra iteration. A tolerance this tight is not always reachable,
#: and :data:`STALL_TOLERANCE` is what distinguishes having arrived from being
#: unable to move.
POOL_TOLERANCE = 1e-15

#: How small the residual has to be for a step that cannot descend to count as
#: convergence rather than as an unbounded dual. The two look identical from
#: inside a line search: no fraction of the Newton step reduces the objective.
#: They are told apart by the gradient, which goes to zero at an optimum and
#: stays bounded away from it when the views are unreachable, and this is where
#: the line is drawn.
STALL_TOLERANCE = 1e-9


class ViewsInfeasible(ValueError):
    """A view that no distribution on these scenarios can satisfy."""


class ViewsNotIdentified(ValueError):
    """Views that do not pin down a single answer on these scenarios."""


@dataclass(frozen=True, slots=True)
class View:
    """A linear statement about the posterior: ``E_q[f] = target``.

    The view function is given by its values at the scenarios rather than as a
    callable, because that is how it arrives — a column of returns, an
    indicator of which scenarios are in a set — and because a callable would
    have to be evaluated once per scenario on every iteration for no gain.

    Attributes:
        values: The view function at each scenario, in the scenarios' order.
        target: What its posterior expectation should be.
        label: A name, used in error messages so that an infeasible view can be
            pointed at. Defaults to the view's position.
    """

    values: tuple[float, ...]
    target: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.values:
            raise TooShort("a view needs at least one scenario")
        for value in (*self.values, self.target):
            if value != value or math.isinf(value):
                raise ValueError(f"a view must be finite, got {value}")

    @property
    def span(self) -> tuple[float, float]:
        """The smallest and largest value the view function takes."""
        return min(self.values), max(self.values)

    def named(self, position: int) -> str:
        return self.label or f"view {position}"


def mean_view(values: Sequence[float], target: float, label: str = "") -> View:
    """A view on the mean of a series: ``E_q[x] = target``.

    Args:
        values: The quantity, one value per scenario.
        target: Its posterior mean.
        label: A name for error messages.
    """
    return View(tuple(float(x) for x in values), float(target), label)


def probability_view(
    inside: Sequence[bool], target: float, label: str = ""
) -> View:
    """A view on the probability of a set: ``P_q(S) = target``.

    The only view in this module with a closed-form answer, because relative
    entropy on a partition is minimised cell by cell. That is what the solver
    is checked against.

    Args:
        inside: Whether each scenario is in the set.
        target: The posterior probability of the set, strictly between 0 and 1.
        label: A name for error messages.

    Raises:
        ValueError: If ``target`` is not strictly inside ``(0, 1)``. A
            probability of exactly zero or one is not reachable at finite
            relative entropy unless the prior already has it, and a caller who
            means "drop these scenarios" is not asking for this function.
    """
    if not 0.0 < float(target) < 1.0:
        raise ValueError(
            f"a probability view needs a target strictly inside (0, 1), got {target!r}"
        )
    return View(tuple(1.0 if flag else 0.0 for flag in inside), float(target), label)


@dataclass(frozen=True, slots=True)
class Posterior:
    """The reweighted scenario set, and everything needed to judge it.

    Attributes:
        weights: The posterior probability of each scenario, summing to one.
        multipliers: The dual solution, one per view. A multiplier's sign says
            which way the view pushed, and its size is in the reciprocal units
            of the view function.
        relative_entropy: ``sum q log(q / p)``, in nats. Zero when the view was
            already true, and the price of the view otherwise.
        effective_scenarios: ``exp`` of the posterior's entropy. Equal to the
            scenario count for a uniform posterior, and smaller to the extent
            the view concentrated the weight. The number to look at before
            reading a tail statistic off the result.
        residuals: ``E_q[f] - target`` for each view, in the view's own units.
            Returned rather than asserted to be zero, because a caller is
            entitled to see how exactly the view was met.
        iterations: Newton steps taken.
        dual_value: The optimal value of the dual objective, which is minus
            :attr:`relative_entropy`. Kept because the two are computed by
            different routes and their agreement is the cheapest check on the
            solve.
    """

    weights: tuple[float, ...]
    multipliers: tuple[float, ...]
    relative_entropy: float
    effective_scenarios: float
    residuals: tuple[float, ...]
    iterations: int
    dual_value: float

    @property
    def scenarios(self) -> int:
        return len(self.weights)

    @property
    def concentration(self) -> float:
        """Effective scenarios as a fraction of the scenarios there are.

        One when the view cost nothing. A number well below one says the tail
        statistics of this posterior rest on fewer observations than the
        scenario count implies, whatever the length of the sample.
        """
        return self.effective_scenarios / len(self.weights)


def relative_entropy(
    weights: Sequence[float], prior: Sequence[float] | None = None
) -> float:
    """``sum q log(q / p)``, in nats, against a uniform prior by default.

    Non-negative, and zero only when the two agree. Scenarios with zero
    posterior weight contribute nothing, which is the continuous extension of
    ``q log q`` at zero rather than a convention chosen here; a scenario with
    zero *prior* weight and positive posterior weight is infinitely surprising
    and raises instead.

    Args:
        weights: The posterior. Non-negative and summing to one.
        prior: The prior. Uniform if omitted.

    Returns:
        The divergence in nats.

    Raises:
        Misaligned: If the two have different lengths.
        TooShort: If ``weights`` is empty.
        ValueError: If a weight is negative, the weights do not sum to one, or a
            scenario carries posterior weight where the prior has none.
    """
    count = len(weights)
    if count == 0:
        raise TooShort("a relative entropy needs at least one scenario")
    reference = _uniform(count) if prior is None else _checked(prior, count)
    _checked(weights, count)
    terms = []
    for weight, base in zip(weights, reference, strict=True):
        if weight == 0.0:
            continue
        if base == 0.0:
            raise ValueError(
                "a scenario carries posterior weight where the prior has none, "
                "which is infinitely surprising"
            )
        terms.append(weight * math.log(weight / base))
    return math.fsum(terms)


def effective_scenarios(weights: Sequence[float]) -> float:
    """``exp`` of the entropy of ``weights``: how many scenarios are doing the work.

    Equal to the length for a uniform set, one for a set that has collapsed onto
    a single scenario, and in between otherwise. The same statistic
    :func:`shortfall.contributions.effective_bets` applies to risk shares, read
    here on probabilities.
    """
    count = len(weights)
    if count == 0:
        raise TooShort("an effective count needs at least one scenario")
    _checked(weights, count)
    entropy = -math.fsum(
        weight * math.log(weight) for weight in weights if weight > 0.0
    )
    return math.exp(entropy)


def _uniform(count: int) -> tuple[float, ...]:
    return (1.0 / count,) * count


def _checked(weights: Sequence[float], count: int) -> tuple[float, ...]:
    if len(weights) != count:
        raise Misaligned(
            f"expected {count} weights to match the scenarios, got {len(weights)}"
        )
    total = math.fsum(weights)
    for weight in weights:
        if weight < 0.0:
            raise ValueError(f"a probability is non-negative, got {weight}")
        if weight != weight or math.isinf(weight):
            raise ValueError(f"a probability must be finite, got {weight}")
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to one, got {total!r}")
    return tuple(weights)


def _check_feasible(views: Sequence[View], prior: Sequence[float]) -> None:
    """Refuse a target the scenarios cannot reach, before the solve diverges.

    For a single view this is exactly the feasibility condition: the posterior
    mean of a function is a convex combination of its values, so it lies
    strictly inside their range unless all the weight goes to one end, which
    costs infinite relative entropy. For several views the condition is that the
    target vector lies in the relative interior of the convex hull of the rows,
    and that is *not* implied by each target lying in its own range — two views
    each individually reachable can be jointly impossible. So this catches the
    easy case with a message that names the view, and the solve's own divergence
    catches the rest.
    """
    support = [index for index, weight in enumerate(prior) if weight > 0.0]
    if not support:
        raise ValueError("the prior puts no weight on any scenario")
    for position, view in enumerate(views, start=1):
        reachable = [view.values[index] for index in support]
        low, high = min(reachable), max(reachable)
        name = view.named(position)
        if low == high:
            if view.target == low:
                raise ViewsNotIdentified(
                    f"{name} is constant at {low} over every scenario the prior "
                    "reaches, so it says nothing and leaves the answer undetermined"
                )
            raise ViewsInfeasible(
                f"{name} targets {view.target} but is constant at {low} over "
                "every scenario the prior reaches"
            )
        if not low < view.target < high:
            raise ViewsInfeasible(
                f"{name} targets {view.target}, outside the open range "
                f"({low}, {high}) the scenarios span: no distribution on them "
                "has that expectation at finite relative entropy"
            )


def _tilt(
    design: Sequence[tuple[float, ...]],
    prior: Sequence[float],
    multipliers: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    """The tilted weights and ``log Z``, shifted so the exponential cannot overflow.

    The exponent is ``-lambda . f`` and nothing bounds it: a view on a return of
    a few per cent gives multipliers in the hundreds, and the product reaches
    where ``exp`` overflows long before the weights become unreasonable.
    Subtracting the largest exponent before exponentiating leaves the weights
    unchanged, since they are normalised, and moves the shift into ``log Z``
    where it belongs.
    """
    exponents = [
        -math.fsum(
            multiplier * value
            for multiplier, value in zip(multipliers, row, strict=True)
        )
        for row in design
    ]
    largest = max(
        exponent
        for exponent, weight in zip(exponents, prior, strict=True)
        if weight > 0.0
    )
    unnormalised = [
        weight * math.exp(exponent - largest) if weight > 0.0 else 0.0
        for weight, exponent in zip(prior, exponents, strict=True)
    ]
    total = math.fsum(unnormalised)
    if total <= 0.0:
        raise ViewsInfeasible(
            "the tilt put no weight on any scenario, which means the views "
            "cannot be met on this scenario set"
        )
    return tuple(value / total for value in unnormalised), largest + math.log(total)


def _objective(
    design: Sequence[tuple[float, ...]],
    prior: Sequence[float],
    targets: Sequence[float],
    multipliers: Sequence[float],
) -> tuple[float, tuple[float, ...]]:
    """The dual value and the weights it implies."""
    weights, log_partition = _tilt(design, prior, multipliers)
    offset = math.fsum(
        multiplier * target
        for multiplier, target in zip(multipliers, targets, strict=True)
    )
    return log_partition + offset, weights


def _solve_spd(matrix: list[list[float]], target: list[float]) -> list[float]:
    """Solve a symmetric positive-definite system by Cholesky.

    Written here rather than in :mod:`shortfall.linalg` because that module's
    ``solve_upper`` is the back substitution and this needs the forward one as
    well; the pair is eight lines and does not earn a public name.
    """
    lower = cholesky(matrix)
    size = len(target)
    forward = [0.0] * size
    for i in range(size):
        total = target[i] - math.fsum(
            lower[i][j] * forward[j] for j in range(i)
        )
        forward[i] = total / lower[i][i]
    back = [0.0] * size
    for i in reversed(range(size)):
        total = forward[i] - math.fsum(
            lower[j][i] * back[j] for j in range(i + 1, size)
        )
        back[i] = total / lower[i][i]
    return back


def pool(
    views: Sequence[View],
    prior: Sequence[float] | None = None,
    *,
    tolerance: float = POOL_TOLERANCE,
    max_iterations: int = MAX_POOL_ITERATIONS,
) -> Posterior:
    """The distribution closest to ``prior`` in relative entropy that meets ``views``.

    Solved in the dual, by Newton with a backtracking line search. The dual has
    one unknown per view, its gradient is the view residual and its Hessian is
    the posterior covariance of the view functions — so the Hessian is positive
    semidefinite by construction, and a singular one means the views are
    linearly dependent on this scenario set rather than that the step failed.

    Args:
        views: The views, all of which hold exactly in the result. An empty
            sequence returns the prior.
        prior: The prior probability of each scenario. Uniform if omitted.
        tolerance: Stopping test on each view's residual divided by the span of
            its view function over the scenarios, which makes it dimensionless
            and so comparable across views measured in different units.
        max_iterations: Newton steps before giving up.

    Returns:
        A :class:`Posterior`.

    Raises:
        ViewsInfeasible: If a view targets a value outside the range its
            function spans over the scenarios the prior reaches, or if the solve
            diverges because the targets are jointly unreachable.
        ViewsNotIdentified: If a view function is constant, or if two views are
            linearly dependent over the scenarios.
        Misaligned: If the views disagree about how many scenarios there are, or
            the prior does not match them.
        TooShort: If there are no scenarios.
        ValueError: If the prior is not a probability distribution, or
            ``tolerance`` is not positive.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    if not views:
        count = 0 if prior is None else len(prior)
        if count == 0:
            raise TooShort("with no views, a prior is needed to have anything to return")
        weights = _checked(prior if prior is not None else (), count)
        return Posterior(
            weights=weights,
            multipliers=(),
            relative_entropy=0.0,
            effective_scenarios=effective_scenarios(weights),
            residuals=(),
            iterations=0,
            dual_value=0.0,
        )

    count = len(views[0].values)
    for position, view in enumerate(views, start=1):
        if len(view.values) != count:
            raise Misaligned(
                f"{view.named(position)} covers {len(view.values)} scenarios "
                f"where the first covers {count}"
            )
    base = _uniform(count) if prior is None else _checked(prior, count)
    _check_feasible(views, base)

    design = tuple(
        tuple(view.values[index] for view in views) for index in range(count)
    )
    targets = [view.target for view in views]
    spans = [max(high - low, 1e-300) for low, high in (view.span for view in views)]
    size = len(views)
    multipliers = [0.0] * size

    value, weights = _objective(design, base, targets, multipliers)
    for iteration in range(1, max_iterations + 1):
        means = [
            math.fsum(
                weight * row[j]
                for weight, row in zip(weights, design, strict=True)
            )
            for j in range(size)
        ]
        gradient = [targets[j] - means[j] for j in range(size)]
        if all(
            abs(gradient[j]) <= tolerance * spans[j] for j in range(size)
        ):
            return _finish(
                weights, multipliers, base, gradient, iteration - 1, value
            )

        worst = max(abs(gradient[j]) / spans[j] for j in range(size))
        hessian = [
            [
                math.fsum(
                    weight * (row[j] - means[j]) * (row[m] - means[m])
                    for weight, row in zip(weights, design, strict=True)
                )
                for m in range(size)
            ]
            for j in range(size)
        ]
        try:
            step = _solve_spd(hessian, [-entry for entry in gradient])
        except NotPositiveDefinite as error:
            # A singular Hessian at the prior and a singular Hessian later are
            # two different diagnoses. At the prior the views are genuinely
            # linearly dependent over the scenarios and no amount of tilting
            # separates them. Later it means the tilt has already driven the
            # weights onto a face of the simplex, where functions that were
            # independent over the whole scenario set are no longer so — which
            # happens when the targets cannot be met and the solve is on its
            # way out of the feasible region. The residual tells them apart.
            if iteration == 1 or worst <= STALL_TOLERANCE:
                raise ViewsNotIdentified(
                    "the views are linearly dependent over these scenarios, so "
                    "the posterior that meets them is not unique"
                ) from error
            raise ViewsInfeasible(
                "the tilt collapsed onto a subset of the scenarios while the "
                f"views are still missed by up to {worst:.3e} of their span, "
                "which means they are jointly unreachable on this scenario set"
            ) from error

        # Backtracking, accepting a step that improves *either* the objective
        # or the residual. Objective decrease alone is the textbook condition
        # and it stops too early here: the dual value of a small-probability
        # view is itself around 1e-04, so it goes flat to the last bit while
        # the residual is still 1e-09, and a search that insists on a strictly
        # lower objective then declares victory three orders of magnitude short
        # of what the gradient can resolve. The residual is what the caller
        # asked about, so it gets a vote. Halving also guards against a step so
        # long that the tilt overflows.
        scale = 1.0
        for _ in range(60):
            trial = [multipliers[j] + scale * step[j] for j in range(size)]
            try:
                candidate, tilted = _objective(design, base, targets, trial)
            except (OverflowError, ViewsInfeasible):
                scale *= 0.5
                continue
            moved = [
                math.fsum(
                    weight * row[j]
                    for weight, row in zip(tilted, design, strict=True)
                )
                for j in range(size)
            ]
            reached = max(
                abs(targets[j] - moved[j]) / spans[j] for j in range(size)
            )
            if candidate < value or reached < worst:
                multipliers = trial
                value, weights = candidate, tilted
                break
            scale *= 0.5
        else:
            if worst <= STALL_TOLERANCE:
                # At the optimum to within double precision: neither the
                # objective nor the residual can be improved, and the residual
                # is already at round-off.
                return _finish(
                    weights, multipliers, base, gradient, iteration, value
                )
            raise ViewsInfeasible(
                "no step improved either the dual objective or the residual "
                f"while the views are still missed by up to {worst:.3e} of "
                "their span, which on a convex problem means they are jointly "
                "unreachable on this scenario set"
            )

    raise ViewsNotIdentified(
        f"the dual did not converge in {max_iterations} Newton steps"
    )


def _finish(
    weights: tuple[float, ...],
    multipliers: Sequence[float],
    prior: Sequence[float],
    gradient: Sequence[float],
    iterations: int,
    value: float,
) -> Posterior:
    """Assemble the result, with the relative entropy taken the direct way.

    The dual value is minus the relative entropy at the optimum, so the two are
    available by different routes. Both are kept: the direct sum is what the
    caller wants and the dual value is what checks it.
    """
    if all(multiplier == 0.0 for multiplier in multipliers):
        # The view was already true, so the tilt is exp(0) and the posterior is
        # the prior. Returning the prior's own floats rather than the normalised
        # tilt makes the difference exactly zero instead of nearly.
        weights = tuple(prior)
        divergence = 0.0
    else:
        divergence = relative_entropy(weights, prior)
    return Posterior(
        weights=weights,
        multipliers=tuple(multipliers),
        relative_entropy=divergence,
        effective_scenarios=effective_scenarios(weights),
        residuals=tuple(-entry for entry in gradient),
        iterations=iterations,
        dual_value=value,
    )


def temper(
    posterior: Posterior,
    confidence: float,
    prior: Sequence[float] | None = None,
) -> Posterior:
    """Blend a posterior back towards the prior: ``(1 - c) p + c q``.

    What partial confidence in a view usually means. For a view on a
    *probability* it is also the entropy-minimising distribution for the
    intermediate view value it produces, exactly and at every confidence, since
    blending two per-cell scalings of a partition leaves each cell's weights
    proportional. For a view on a *mean* it is not, because a blend of two
    exponential tilts is not an exponential tilt: re-solving :func:`pool` at the
    view value the blend actually produced gives a different distribution with
    strictly less relative entropy, by up to 31.8% on the example in this
    module's docstring — and the gap is widest at low confidence in a strong
    view, which is the opposite end from where it would be looked for. The
    returned object reports its own entropy and effective scenario count so
    that the comparison can be made.

    Args:
        posterior: The full-confidence solution.
        confidence: Weight on it, in ``[0, 1]``. Zero returns the prior.
        prior: The prior the posterior was solved against. Uniform if omitted.

    Returns:
        A :class:`Posterior` whose multipliers are empty, because a blend is not
        an exponential tilt and reporting the original's multipliers beside
        blended weights would be a lie. Its residuals are the blend's own,
        which are not zero and are not meant to be.

    Raises:
        ValueError: If ``confidence`` is outside ``[0, 1]``.
        Misaligned: If the prior does not match the posterior.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence lies in [0, 1], got {confidence!r}")
    count = len(posterior.weights)
    base = _uniform(count) if prior is None else _checked(prior, count)
    if confidence == 1.0:
        return posterior
    blended = tuple(
        (1.0 - confidence) * one + confidence * other
        for one, other in zip(base, posterior.weights, strict=True)
    )
    return Posterior(
        weights=blended,
        multipliers=(),
        relative_entropy=relative_entropy(blended, base),
        effective_scenarios=effective_scenarios(blended),
        residuals=(),
        iterations=0,
        dual_value=float("nan"),
    )


def weighted_quantile(
    values: Sequence[float], weights: Sequence[float], probability: float
) -> float:
    """The ``probability`` quantile of a weighted empirical distribution.

    The lower quantile: the smallest observed value whose cumulative weight
    reaches ``probability``. Deliberately not interpolated, unlike
    :func:`shortfall.historical.empirical_quantile`, and the reason is the
    weights. Interpolating between two observations presumes they carry
    comparable mass; after a tilt they routinely differ by a factor of ten, and
    a linear blend across that gap invents a value the reweighted sample does
    not support. A caller who wants a smooth quantile should smooth the sample,
    not the quantile.

    Args:
        values: The observations.
        weights: Their probabilities, summing to one.
        probability: In ``[0, 1]``.

    Returns:
        The quantile.

    Raises:
        TooShort: If there are no observations.
        Misaligned: If the lengths differ.
        ValueError: If the weights are not a distribution, or ``probability``
            is outside ``[0, 1]``.
    """
    count = len(values)
    if count == 0:
        raise TooShort("a weighted quantile needs at least one observation")
    checked = _checked(weights, count)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"a probability lies in [0, 1], got {probability!r}")
    order = sorted(range(count), key=lambda index: values[index])
    cumulative = 0.0
    for index in order:
        cumulative += checked[index]
        if cumulative >= probability:
            return values[index]
    return values[order[-1]]


def weighted_expected_shortfall(
    values: Sequence[float], weights: Sequence[float], probability: float
) -> tuple[float, float]:
    """``E[X | X <= q_p]`` under ``weights``, and the mass it averaged over.

    The mass is returned because it is not ``probability``. The quantile lands
    on an observation whose weight can be substantial, so the tail up to and
    including it carries more mass than asked for; the average divides by what
    it actually used, and the caller is told what that was. On an unweighted
    sample of a thousand the difference is a tenth of a per cent, and after a
    tilt that concentrates the tail it can be several per cent — enough to
    matter to a number quoted to three figures.

    Args:
        values: The observations.
        weights: Their probabilities.
        probability: The tail probability, strictly inside ``(0, 1]``.

    Returns:
        ``(mean of the tail, mass used)``. The mean is signed, so a loss is
        negative.

    Raises:
        TooShort: If there are no observations.
        Misaligned: If the lengths differ.
        ValueError: If the weights are not a distribution, or ``probability``
            is outside ``(0, 1]``.
    """
    count = len(values)
    if count == 0:
        raise TooShort("a weighted expected shortfall needs at least one observation")
    checked = _checked(weights, count)
    if not 0.0 < probability <= 1.0:
        raise ValueError(
            f"a tail probability lies in (0, 1], got {probability!r}"
        )
    order = sorted(range(count), key=lambda index: values[index])
    mass = 0.0
    total = 0.0
    for index in order:
        mass += checked[index]
        total += checked[index] * values[index]
        if mass >= probability:
            break
    if mass <= 0.0:
        return values[order[0]], 0.0
    return total / mass, mass


def stressed_risk(
    returns: Sequence[float],
    weights: Sequence[float],
    confidence: float,
) -> Risk:
    """Value at risk and expected shortfall from a reweighted sample.

    The counterpart of :func:`shortfall.historical.historical_risk` for a
    posterior. The distribution is reported as
    :attr:`~shortfall.parametric.Distribution.HISTORICAL` because that is what
    it is: the sample itself, with the observations carrying different
    probabilities. No shape has been assumed, so putting it beside the
    unstressed historical number is the comparison that says what the view did.

    Args:
        returns: Signed returns, one per scenario.
        weights: Their posterior probabilities.
        confidence: The confidence level, strictly inside ``(0, 1)``.

    Returns:
        A :class:`~shortfall.parametric.Risk`, with the mean and volatility
        taken under the same weights so that every field describes one
        distribution.

    Raises:
        TooShort: If there are no observations.
        Misaligned: If the lengths differ.
        ValueError: If the weights are not a distribution, or ``confidence`` is
            outside ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence lies strictly inside (0, 1), got {confidence!r}"
        )
    count = len(returns)
    if count == 0:
        raise TooShort("a stressed risk estimate needs at least one observation")
    checked = _checked(weights, count)
    tail = 1.0 - confidence
    quantile = weighted_quantile(returns, checked, tail)
    shortfall, _ = weighted_expected_shortfall(returns, checked, tail)
    mean = math.fsum(
        weight * value for weight, value in zip(checked, returns, strict=True)
    )
    variance = math.fsum(
        weight * (value - mean) ** 2
        for weight, value in zip(checked, returns, strict=True)
    )
    return Risk(
        value_at_risk=-quantile,
        expected_shortfall=-shortfall,
        quantile=quantile,
        confidence=confidence,
        distribution=Distribution.HISTORICAL,
        mean=mean,
        volatility=math.sqrt(max(variance, 0.0)),
    )

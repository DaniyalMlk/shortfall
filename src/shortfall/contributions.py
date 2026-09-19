"""Where a portfolio's risk comes from: Euler allocation, risk parity, diversification.

A portfolio risk number is one figure, and one figure does not say which
position produced it. Splitting it up is only meaningful if the split obeys a
rule, and the rule used here is the Euler allocation.

**Why Euler, and what it needs.** Volatility, value at risk and expected
shortfall are all *positively homogeneous of degree one* in the weights: double
every position and each measure doubles. Euler's theorem then says the measure
equals the sum of the weights times its own partial derivatives::

    rho(w) = sum_i w_i * d rho / d w_i

So the component contributions add up to the total exactly — not approximately,
and not as a normalisation applied afterwards. That identity is the whole
justification for calling the pieces "contributions", and it is asserted in the
tests rather than trusted.

Homogeneity is a real requirement and not a formality. It is why variance is
*not* allocated here: variance is homogeneous of degree two, so its Euler sum is
twice the variance, and a "variance contribution" that silently halves itself to
make the total work is not a derivative of anything.

**The allocation never re-derives a distributional constant.** Every
location-scale risk measure in this package has the form ``-mean + k *
volatility`` for a constant ``k`` that depends on the distribution and the
confidence level and on nothing else. Rather than reimplementing ``k`` for the
normal, the Student-t and the Cornish-Fisher cases — three more places to get a
tail convention wrong — the allocation reads it off the very function that
produces the total, by asking that function for the risk of a unit-volatility,
zero-mean portfolio. If the total is ever wrong, the allocation is wrong in
exactly the same way, which is the failure mode worth having: the identity
between them cannot quietly break.

**Negative contributions are real.** A position that hedges the rest of the
portfolio has a negative component contribution: it removes risk. That is
information, not an error, and it is returned as it comes out. But it breaks
every definition built on treating contributions as a probability distribution —
an entropy, a Herfindahl concentration — and those refuse rather than returning
a number computed from negative "probabilities".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .linalg import Matrix, cholesky, dimension, eigh, matrix_vector, quadratic_form
from .parametric import Distribution, Risk, parametric_risk
from .series import Panel

#: Weights whose portfolio volatility falls at or below this are treated as
#: having no risk to allocate. The Euler gradient divides by the volatility, so
#: there is no finite answer there rather than a large one.
_ZERO_VOLATILITY = 1e-300


class NoRiskToAllocate(ValueError):
    """The portfolio has no volatility, so there is no gradient to take."""


class NegativeContribution(ValueError):
    """A measure that treats contributions as a distribution met a negative one.

    Which is not a bug in the portfolio: a hedge genuinely contributes negative
    risk. It means the requested statistic — an entropy or a concentration index
    — is not defined for this allocation, because it is defined over weights
    that sum to one and are non-negative, and these do not.
    """


@dataclass(frozen=True)
class Allocation:
    """A risk total split across positions, with the pieces that produced it."""

    names: tuple[str, ...]
    weights: tuple[float, ...]
    #: ``d rho / d w_i``: the change in total risk per unit change in weight.
    #: The number to read when asking whether to add to a position.
    marginal: tuple[float, ...]
    #: ``w_i * marginal_i``: the share of the total this position accounts for.
    #: These sum to :attr:`total`.
    component: tuple[float, ...]
    #: The portfolio risk being allocated, from the estimator itself rather than
    #: from summing the components.
    total: float
    #: Which measure was allocated, for a result that is read later.
    measure: str

    def __post_init__(self) -> None:
        sizes = {len(self.names), len(self.weights), len(self.marginal), len(self.component)}
        if len(sizes) != 1:
            raise ValueError(
                f"an allocation has one entry per asset; got names={len(self.names)}, "
                f"weights={len(self.weights)}, marginal={len(self.marginal)}, "
                f"component={len(self.component)}"
            )

    def __len__(self) -> int:
        return len(self.names)

    @property
    def sum_of_components(self) -> float:
        """What the components add to, which Euler says is :attr:`total`."""
        return math.fsum(self.component)

    @property
    def identity_error(self) -> float:
        """How far the components are from adding to the total, in absolute terms.

        Exposed rather than merely tested. It is the cheapest possible check
        that an allocation means anything, and a caller who has repaired a
        covariance matrix or hand-built a gradient can read it without importing
        the test suite.
        """
        return abs(self.sum_of_components - self.total)

    @property
    def percentage(self) -> tuple[float, ...]:
        """Each component as a share of the total. These sum to one.

        May be negative, and may exceed one, whenever a position hedges: the
        shares still sum to one, but they are not a distribution over anything.
        """
        if self.total == 0.0:
            raise NoRiskToAllocate(
                "the total risk is zero, so there are no shares of it; the component "
                "contributions are still available and all of them are zero too"
            )
        return tuple(value / self.total for value in self.component)

    @property
    def has_negative_contribution(self) -> bool:
        """Whether any position removes risk rather than adding it."""
        return any(value < 0.0 for value in self.component)

    def largest(self) -> tuple[str, float]:
        """The name and component of the position contributing most risk."""
        index = max(range(len(self.component)), key=lambda i: self.component[i])
        return self.names[index], self.component[index]

    def of(self, name: str) -> float:
        """One named position's component contribution."""
        for index, held in enumerate(self.names):
            if held == name:
                return self.component[index]
        raise KeyError(name)


def _names_for(names: Sequence[str] | None, size: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"asset {index}" for index in range(size))
    if len(names) != size:
        raise ValueError(f"{len(names)} names against {size} assets; they must agree")
    return tuple(names)


def _portfolio_volatility(weights: Sequence[float], covariance: Matrix) -> float:
    variance = quadratic_form(list(weights), covariance)
    if variance < 0.0:
        raise ValueError(
            f"these weights give a portfolio variance of {variance!r}, which is "
            "negative. The covariance matrix is not positive semi-definite; repair it "
            "with nearest_psd or shrink it before allocating risk from it."
        )
    return math.sqrt(variance)


def volatility_contributions(
    weights: Sequence[float],
    covariance: Matrix,
    *,
    names: Sequence[str] | None = None,
) -> Allocation:
    """Euler allocation of portfolio volatility.

    The gradient of ``sqrt(w' S w)`` is ``S w / sigma``, so the marginal
    contribution of a position is its covariance with the portfolio divided by
    the portfolio volatility — which is the position's beta to the portfolio
    times the portfolio volatility, written the other way round.

    The components sum to the volatility exactly, because ``sum_i w_i (Sw)_i /
    sigma`` is ``w'Sw / sigma`` is ``sigma``.
    """
    size = dimension(covariance)
    if len(weights) != size:
        raise ValueError(f"{len(weights)} weights against a {size}x{size} covariance")
    volatility = _portfolio_volatility(weights, covariance)
    if volatility <= _ZERO_VOLATILITY:
        raise NoRiskToAllocate(
            "the portfolio volatility is zero, so the risk gradient is not defined. "
            "Either every weight is zero or the weights lie in the null space of the "
            "covariance matrix."
        )
    covariances = matrix_vector(covariance, list(weights))
    marginal = tuple(value / volatility for value in covariances)
    component = tuple(w * m for w, m in zip(weights, marginal, strict=True))
    return Allocation(
        names=_names_for(names, size),
        weights=tuple(weights),
        marginal=marginal,
        component=component,
        total=volatility,
        measure="volatility",
    )


def unit_risk(
    *,
    confidence: float,
    distribution: Distribution = Distribution.NORMAL,
    degrees: float = 5.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> Risk:
    """The risk of a zero-mean, unit-volatility position.

    Which is the constant ``k`` in ``rho = -mean + k * volatility``, for both
    value at risk and expected shortfall, read straight off the estimator rather
    than re-derived. Every distribution here is location-scale in ``(mean,
    volatility)`` with the shape parameters held fixed, so that one call
    characterises the measure completely.
    """
    return parametric_risk(
        mean=0.0,
        volatility=1.0,
        confidence=confidence,
        distribution=distribution,
        degrees=degrees,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )


def risk_contributions(
    weights: Sequence[float],
    covariance: Matrix,
    *,
    means: Sequence[float] | None = None,
    confidence: float = 0.99,
    distribution: Distribution = Distribution.NORMAL,
    degrees: float = 5.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    of_expected_shortfall: bool = False,
    names: Sequence[str] | None = None,
) -> Allocation:
    """Euler allocation of parametric value at risk, or of expected shortfall.

    Under any location-scale assumption the measure is ``-w'mu + k * sigma(w)``,
    whose gradient is ``-mu_i + k * (Sw)_i / sigma``. So the allocation is the
    volatility allocation scaled by ``k``, less each position's own expected
    return — and a position with a large enough expected return can contribute
    negative risk on that account alone.

    ``means`` defaults to zero, as it does in :func:`~shortfall.portfolio_risk`
    and for the same reason: crediting an estimated drift inside a risk number
    bets on the least reliably estimated input in the calculation.

    Set ``of_expected_shortfall`` to allocate expected shortfall instead. The
    only thing that changes is which constant is read from :func:`unit_risk`,
    which is the point of taking it from there.
    """
    size = dimension(covariance)
    if len(weights) != size:
        raise ValueError(f"{len(weights)} weights against a {size}x{size} covariance")
    if means is not None and len(means) != size:
        raise ValueError(f"{len(means)} means against {size} assets; they must agree")

    volatility = _portfolio_volatility(weights, covariance)
    if volatility <= _ZERO_VOLATILITY:
        raise NoRiskToAllocate(
            "the portfolio volatility is zero, so the risk gradient is not defined. "
            "A portfolio with no volatility has a risk of minus its expected return, "
            "and that number has no gradient to allocate."
        )
    unit = unit_risk(
        confidence=confidence,
        distribution=distribution,
        degrees=degrees,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
    scale = unit.expected_shortfall if of_expected_shortfall else unit.value_at_risk
    drifts = list(means) if means is not None else [0.0] * size

    covariances = matrix_vector(covariance, list(weights))
    marginal = tuple(
        scale * covariances[index] / volatility - drifts[index] for index in range(size)
    )
    component = tuple(w * m for w, m in zip(weights, marginal, strict=True))
    total = scale * volatility - math.fsum(
        w * d for w, d in zip(weights, drifts, strict=True)
    )
    return Allocation(
        names=_names_for(names, size),
        weights=tuple(weights),
        marginal=marginal,
        component=component,
        total=total,
        measure=(
            f"{distribution.value} expected shortfall"
            if of_expected_shortfall
            else f"{distribution.value} value at risk"
        ),
    )


def historical_shortfall_contributions(
    panel: Panel,
    weights: Sequence[float],
    *,
    confidence: float = 0.99,
    names: Sequence[str] | None = None,
) -> Allocation:
    """Euler allocation of *sample* expected shortfall, with no shape assumption.

    Expected shortfall is differentiable in the weights even without a
    distribution, and its derivative has a reading anybody can check::

        d ES / d w_i = -E[ r_i | the portfolio is in its tail ]

    So a position's contribution is minus its average return on the days the
    portfolio lost most. No normality, no covariance matrix, and the answer
    reflects whatever actually co-moved in the tail — which is the reason to
    compute it this way, since correlations in a crash are not the correlations
    in the covariance matrix.

    The tail is weighted exactly as :func:`~shortfall.sample_expected_shortfall`
    weights it, including the fractional last observation. That is what makes
    the components sum to the sample expected shortfall of the portfolio rather
    than to something close to it: the two are the same weighted average, taken
    over the assets in one case and over their weighted sum in the other. Using
    a rounded tail here and a fractional one there would leave an identity error
    that grows as the sample thins, exactly where anybody would blame the
    estimator instead of the allocation.

    Value at risk is deliberately not offered from a sample. Its Euler
    derivative is an expectation conditioned on the portfolio being *exactly* at
    its quantile, which no finite sample observes; estimating it needs a kernel
    whose bandwidth changes the answer, and a number that depends on an
    undisclosed smoothing choice is worse than no number.
    """
    if len(weights) != panel.assets:
        raise ValueError(
            f"{len(weights)} weights against {panel.assets} assets; they must agree"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence is strictly inside (0, 1), got {confidence!r}. It is the "
            "confidence level, so 0.99 means the 1% tail."
        )
    probability = 1.0 - confidence
    portfolio = panel.portfolio(list(weights))
    count = len(portfolio)

    # Order the *periods* by the portfolio's return, not each asset by its own:
    # the tail being averaged over is the portfolio's tail, and an asset's
    # contribution is its behaviour on those particular days.
    order = sorted(range(count), key=lambda t: portfolio.values[t])
    mass = count * probability
    full = min(math.floor(mass), count)
    tail_weights = [0.0] * count
    for position in range(full):
        tail_weights[order[position]] = 1.0
    if full < count and mass > full:
        tail_weights[order[full]] = mass - full

    simple = [one.to_simple() for one in panel.series]
    # -E[r_i | portfolio in its tail]; the sign flip is the package convention
    # that risk is a positive loss.
    marginal = tuple(
        -math.fsum(
            tail_weights[t] * asset.values[t] for t in range(count) if tail_weights[t]
        )
        / mass
        for asset in simple
    )
    component = tuple(w * m for w, m in zip(weights, marginal, strict=True))
    total = (
        -math.fsum(
            tail_weights[t] * portfolio.values[t] for t in range(count) if tail_weights[t]
        )
        / mass
    )
    return Allocation(
        names=_names_for(names if names is not None else panel.names, panel.assets),
        weights=tuple(weights),
        marginal=marginal,
        component=component,
        total=total,
        measure="historical expected shortfall",
    )


# -- risk parity -------------------------------------------------------------


class DidNotConverge(RuntimeError):
    """The risk parity iteration ran out of sweeps before meeting its tolerance.

    Carries the unconverged :attr:`result` so the evidence can be inspected
    rather than guessed at. Raised rather than returned quietly, because weights
    that are nearly risk parity look exactly like weights that are risk parity.
    """

    def __init__(self, message: str, result: RiskParity) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class RiskParity:
    """Weights whose risk contributions match a set of budgets, and the evidence."""

    names: tuple[str, ...]
    weights: tuple[float, ...]
    #: The requested shares of risk, normalised to sum to one.
    budgets: tuple[float, ...]
    #: The allocation these weights actually produce, so the claim can be read.
    allocation: Allocation
    sweeps: int
    #: Largest absolute change in any coordinate on the final sweep. What the
    #: tolerance was tested against.
    final_change: float
    #: Largest absolute gap between an achieved risk share and its budget. The
    #: number that says whether the answer is right, as opposed to whether the
    #: iteration stopped moving.
    budget_error: float
    converged: bool

    @property
    def equal_risk(self) -> bool:
        """Whether this is the equal-contribution portfolio specifically."""
        first = self.budgets[0]
        return all(abs(b - first) <= 1e-12 for b in self.budgets)


def _stable_positive_root(a: float, c: float, b: float) -> float:
    """The positive root of ``a x^2 + c x - b = 0`` for ``a > 0``, ``b > 0``.

    Both closed forms for a quadratic root lose precision on one sign of ``c``,
    and this iteration hits both signs: ``c`` is the covariance of the coordinate
    with everything else, which is negative exactly when the asset hedges. So the
    branch is chosen rather than picked once and hoped for. Writing only
    ``(-c + sqrt(c^2 + 4ab)) / 2a`` silently loses digits for a strongly
    positively correlated asset, which is most of them.
    """
    discriminant = math.sqrt(c * c + 4.0 * a * b)
    if c >= 0.0:
        return 2.0 * b / (c + discriminant)
    return (-c + discriminant) / (2.0 * a)


def risk_parity(
    covariance: Matrix,
    *,
    budgets: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    tolerance: float = 1e-14,
    max_sweeps: int = 2000,
) -> RiskParity:
    """Long-only weights whose risk contributions match ``budgets``.

    Equal budgets by default, which is the portfolio usually meant by "risk
    parity": every position accounts for the same share of total volatility.

    **Solved as a convex problem rather than as a root find.** The naive
    approach is to throw the nonlinear system ``w_i (S w)_i = b_i`` at a solver,
    which is not convex, has spurious solutions with negative weights, and fails
    on exactly the ill-conditioned matrices that make the answer interesting.
    Instead this minimises

        f(x) = (1/2) x' S x - sum_i b_i log(x_i)

    over positive ``x``, which is strictly convex for a positive definite ``S``
    — a requirement checked rather than assumed, since the iteration terminates
    happily on an indefinite matrix and returns weights for a problem that has
    no solution —
    and whose stationary condition ``x_i (S x)_i = b_i`` is the risk parity
    condition up to scale. The logarithmic barrier keeps every weight positive
    without a constraint, and the solution is normalised to sum to one at the
    end — the condition is scale invariant, so that costs nothing.

    Each coordinate then has a closed-form update: holding the others fixed,
    ``f`` is a quadratic plus a log in one variable, and its minimiser is a root
    of ``S_ii x^2 + c x - b_i`` with ``c`` the coordinate's covariance with the
    rest. So no line search, no step size, and no tuning.

    ``tolerance`` is tested against the largest coordinate movement in a sweep,
    and the result also reports :attr:`~RiskParity.budget_error`, which is what
    actually matters: an iteration can stop moving without having arrived.
    """
    size = dimension(covariance)
    for index in range(size):
        if covariance[index][index] <= 0.0:
            raise ValueError(
                f"asset {index} has a variance of {covariance[index][index]!r}; risk "
                "parity needs every asset to carry some risk, since an asset with "
                "none would take an unbounded weight to reach its budget"
            )
    # The objective is strictly convex only for a positive definite matrix, and
    # this is where that is established rather than assumed. On an indefinite
    # matrix the coordinate updates still produce positive numbers and the
    # iteration still terminates, so the failure is silent: the weights come
    # back looking like an answer to a problem that has none.
    cholesky(covariance)
    if budgets is None:
        shares = [1.0 / size] * size
    else:
        if len(budgets) != size:
            raise ValueError(f"{len(budgets)} budgets against {size} assets")
        for index, budget in enumerate(budgets):
            if budget <= 0.0:
                raise ValueError(
                    f"budget {index} is {budget!r}; every budget is strictly "
                    "positive. A budget of zero asks for a position that holds none "
                    "of the asset, which is a smaller problem — drop the asset."
                )
        total_budget = math.fsum(budgets)
        shares = [budget / total_budget for budget in budgets]

    # Start from inverse volatility, which is the exact answer whenever the
    # correlations are all equal and a good one otherwise, so the iteration
    # usually has little left to do.
    x = [
        share / math.sqrt(covariance[index][index])
        for index, share in enumerate(shares)
    ]

    sweeps = 0
    change = math.inf
    for sweeps in range(1, max_sweeps + 1):  # noqa: B007
        change = 0.0
        for index in range(size):
            cross = math.fsum(
                covariance[index][j] * x[j] for j in range(size) if j != index
            )
            updated = _stable_positive_root(covariance[index][index], cross, shares[index])
            change = max(change, abs(updated - x[index]))
            x[index] = updated
        if change <= tolerance:
            break

    total = math.fsum(x)
    weights = tuple(value / total for value in x)
    allocation = volatility_contributions(weights, covariance, names=names)
    achieved = allocation.percentage
    budget_error = max(
        abs(one - two) for one, two in zip(achieved, shares, strict=True)
    )
    result = RiskParity(
        names=allocation.names,
        weights=weights,
        budgets=tuple(shares),
        allocation=allocation,
        sweeps=sweeps,
        final_change=change,
        budget_error=budget_error,
        converged=change <= tolerance,
    )
    if not result.converged:
        raise DidNotConverge(
            f"risk parity did not converge in {max_sweeps} sweeps: the last sweep "
            f"still moved a coordinate by {change:.3e} against a tolerance of "
            f"{tolerance:.3e}, and the worst risk share is {budget_error:.3e} from "
            "its budget. The covariance matrix is probably near-singular; shrink it "
            "or repair it with nearest_psd first.",
            result,
        )
    return result


# -- diversification ---------------------------------------------------------


def diversification_ratio(weights: Sequence[float], covariance: Matrix) -> float:
    """Weighted average volatility over portfolio volatility.

    One where the portfolio is a single asset or a set of perfectly correlated
    ones, and larger the more the positions offset each other. It is the factor
    by which diversification has reduced the risk of holding the same positions
    separately.

    Defined for long-only weights. With a short position the numerator mixes
    signs while the denominator cannot, and the ratio stops measuring anything —
    so that case is refused rather than returned.
    """
    size = dimension(covariance)
    if len(weights) != size:
        raise ValueError(f"{len(weights)} weights against a {size}x{size} covariance")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(
            "the diversification ratio is defined for long-only weights; with a "
            "short position its numerator adds volatilities that the portfolio "
            "subtracts, and the ratio no longer measures diversification"
        )
    volatility = _portfolio_volatility(weights, covariance)
    if volatility <= _ZERO_VOLATILITY:
        raise NoRiskToAllocate("a portfolio with no volatility has no ratio to take")
    weighted = math.fsum(
        weight * math.sqrt(covariance[index][index])
        for index, weight in enumerate(weights)
    )
    return weighted / volatility


def _as_distribution(shares: Sequence[float], what: str) -> list[float]:
    for index, share in enumerate(shares):
        if share < 0.0:
            raise NegativeContribution(
                f"{what} treats the risk shares as a distribution, and share {index} "
                f"is {share!r}. A negative share means that position removes risk, "
                "which is a real and useful thing to know and is not a quantity an "
                "entropy is defined over."
            )
    return list(shares)


def effective_bets(allocation: Allocation) -> float:
    """The exponential of the entropy of the risk shares.

    Answers "how many equally-sized independent positions would carry this much
    risk concentration?" — ``n`` for ``n`` positions contributing equally, and
    one for a portfolio whose risk all comes from a single position. It is a
    count, and unlike the weight count it responds to what the positions
    actually do.

    Measured over risk contributions and not over weights, which is the whole
    point: ten equally weighted positions in the same sector are ten weights and
    close to one bet.
    """
    shares = _as_distribution(allocation.percentage, "the effective number of bets")
    entropy = -math.fsum(
        share * math.log(share) for share in shares if share > 0.0
    )
    return math.exp(entropy)


def concentration(allocation: Allocation) -> float:
    """Herfindahl index of the risk shares: ``sum_i p_i^2``.

    Between ``1/n`` and one. The reciprocal is another effective count, and it
    differs from :func:`effective_bets` by being far more sensitive to the
    largest share and far less to the long tail of small ones. Both are offered
    because which of those two behaviours is wanted depends on the question, and
    quoting one under the other's name is a common way to make a concentrated
    portfolio look diversified.
    """
    shares = _as_distribution(allocation.percentage, "the concentration index")
    return math.fsum(share * share for share in shares)


@dataclass(frozen=True)
class PrincipalBets:
    """The risk of a portfolio spread across uncorrelated principal components."""

    #: Share of portfolio variance carried by each principal component, largest
    #: eigenvalue first. Non-negative and summing to one, always — unlike the
    #: risk shares across positions, which need not be either.
    distribution: tuple[float, ...]
    #: Eigenvalues of the covariance matrix, in the same order.
    variances: tuple[float, ...]
    #: ``exp`` of the entropy of :attr:`distribution`.
    effective_bets: float

    @property
    def concentration(self) -> float:
        return math.fsum(share * share for share in self.distribution)


def principal_bets(weights: Sequence[float], covariance: Matrix) -> PrincipalBets:
    """Effective number of bets measured across principal components.

    The measure over positions has a flaw that is easy to miss: positions are
    not independent, so spreading risk evenly across ten correlated positions
    reports ten bets when there is really one. Rotating into the principal
    components of the covariance matrix fixes that by construction, because the
    components are uncorrelated — so the shares are shares of genuinely separate
    sources of risk.

    It is also better behaved. Variance shares over an orthogonal basis are
    non-negative whatever the weights do, so this is defined for a portfolio
    with shorts, where :func:`effective_bets` is not.

    The cost is interpretability: a principal component is a combination of
    assets and not a thing anybody holds, and the components are determined by
    the estimated covariance, so a noisy matrix produces confident-looking bets
    on directions that are mostly estimation error. This is Meucci's
    diversification distribution over the principal-component basis; the
    minimum-torsion basis he later proposed trades some of the orthogonality for
    components that stay close to the original assets.
    """
    size = dimension(covariance)
    if len(weights) != size:
        raise ValueError(f"{len(weights)} weights against a {size}x{size} covariance")
    values, vectors = eigh(covariance)
    order = sorted(range(size), key=lambda k: values[k], reverse=True)
    variance = 0.0
    contributions: list[float] = []
    eigenvalues: list[float] = []
    for k in order:
        # The portfolio's loading on this component, then the variance it
        # carries. Negative eigenvalues can only come from a matrix that is not
        # positive semi-definite; they are floored rather than propagated into a
        # negative "share".
        # ``eigh`` returns eigenvectors as rows, so the vector for values[k] is
        # vectors[k] and not the k-th column. Indexing it the other way is a
        # silent transpose: it still produces a plausible non-negative
        # distribution summing to one, and it is wrong about every component.
        loading = math.fsum(
            vectors[k][row] * weights[row] for row in range(size)
        )
        carried = max(values[k], 0.0) * loading * loading
        contributions.append(carried)
        eigenvalues.append(values[k])
        variance += carried
    if variance <= _ZERO_VOLATILITY:
        raise NoRiskToAllocate(
            "the portfolio carries no variance in any principal direction, so there "
            "is nothing to spread across them"
        )
    distribution = tuple(value / variance for value in contributions)
    entropy = -math.fsum(
        share * math.log(share) for share in distribution if share > 0.0
    )
    return PrincipalBets(
        distribution=distribution,
        variances=tuple(eigenvalues),
        effective_bets=math.exp(entropy),
    )


__all__ = [
    "Allocation",
    "DidNotConverge",
    "NegativeContribution",
    "NoRiskToAllocate",
    "PrincipalBets",
    "RiskParity",
    "concentration",
    "diversification_ratio",
    "effective_bets",
    "historical_shortfall_contributions",
    "principal_bets",
    "risk_contributions",
    "risk_parity",
    "unit_risk",
    "volatility_contributions",
]

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

from .linalg import Matrix, dimension, matrix_vector, quadratic_form
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


__all__ = [
    "Allocation",
    "NegativeContribution",
    "NoRiskToAllocate",
    "historical_shortfall_contributions",
    "risk_contributions",
    "unit_risk",
    "volatility_contributions",
]

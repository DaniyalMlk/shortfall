"""Factor models: exposures by regression, specific risk, and risk attribution.

A covariance matrix treats assets as the unit of risk and says nothing about
what the risk is *about*. A factor model answers that, by writing each asset's
return as a linear response to a handful of shared drivers plus whatever is left
over::

    r_i = a_i + sum_k B_ik f_k + e_i

The portfolio's risk then splits into the part explained by the factors and the
part that is not, and the first part splits again across individual factors. So
"why is this portfolio risky" gets an answer with names on it.

**The two assumptions, both of them consequential.**

*Linearity.* Exposures are constant through the sample. They are not: a
credit-sensitive position's beta to rates changes with the level of rates, and
every exposure changes in a crisis. The model is a local approximation over the
window it was fitted on, which is why the window is a visible argument and not
a default.

*Diagonal specific risk.* The residuals are assumed uncorrelated across assets,
which is what makes the implied covariance cheap: ``B F B' + D`` needs only the
small factor covariance and a diagonal, whatever the number of assets. It is
also the assumption that fails quietly — if a sector the factors do not span
moves together, its residuals are correlated, and the model reports a
diversification that the portfolio does not have. So the fit reports the largest
residual correlation it left behind rather than only its R-squared, because that
number is the one that says whether the model may be believed.

**Fitted per asset, not jointly.** Each asset is regressed on the same factors
separately, which gives the same coefficients as a joint fit whenever the design
is shared — and it is, here — while keeping the diagnostics per asset, where
they can be read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .linalg import (
    Matrix,
    Vector,
    least_squares,
    symmetrise,
    upper_inverse_gram,
)
from .series import Panel, ReturnSeries, TooShort


@dataclass(frozen=True)
class Regression:
    """One asset regressed on the factors, with everything needed to judge it."""

    name: str
    #: Intercept. The part of the return the factors do not explain on average —
    #: "alpha" when it is flattering and "the model is missing a factor" when it
    #: is not.
    alpha: float
    #: Exposure to each factor, in the factors' order.
    betas: tuple[float, ...]
    factor_names: tuple[str, ...]
    residuals: tuple[float, ...]
    observations: int
    #: Coefficients estimated, intercept included. The residual degrees of
    #: freedom are ``observations - parameters``.
    parameters: int
    #: Residual variance with the degrees of freedom taken out. This is the
    #: specific variance the factor model carries for this asset.
    specific_variance: float
    #: Standard error of each coefficient, intercept first.
    standard_errors: tuple[float, ...]
    r_squared: float

    @property
    def degrees_of_freedom(self) -> int:
        return self.observations - self.parameters

    @property
    def specific_volatility(self) -> float:
        return math.sqrt(self.specific_variance)

    @property
    def adjusted_r_squared(self) -> float:
        """R-squared penalised for the parameters spent reaching it.

        The unadjusted figure rises whenever a factor is added, including a
        factor of pure noise, so comparing models on it always selects the
        largest one. This can go negative, which is the honest report that the
        factors explain less than the asset's own mean does.
        """
        if self.degrees_of_freedom <= 0:
            raise TooShort(f"{self.name}: no residual degrees of freedom to adjust by")
        return 1.0 - (1.0 - self.r_squared) * (self.observations - 1) / (
            self.degrees_of_freedom
        )

    @property
    def t_statistics(self) -> tuple[float, ...]:
        """Each coefficient over its standard error, intercept first."""
        coefficients = (self.alpha, *self.betas)
        return tuple(
            (value / error if error > 0.0 else math.inf if value != 0.0 else 0.0)
            for value, error in zip(coefficients, self.standard_errors, strict=True)
        )

    @property
    def durbin_watson(self) -> float:
        """Serial correlation in the residuals: near 2 is none, near 0 is positive.

        Worth reading because ordinary least squares is unbiased under serial
        correlation but its *standard errors* are not, and they are wrong in the
        flattering direction — a t-statistic of 3 on autocorrelated residuals
        may be a t-statistic of 1. Financial residuals are frequently
        autocorrelated, so a large exposure with an impressive t-statistic and a
        Durbin-Watson of 1.2 has not been established at all.
        """
        if len(self.residuals) < 2:
            raise TooShort(f"{self.name}: serial correlation needs two residuals")
        bottom = math.fsum(value * value for value in self.residuals)
        if bottom == 0.0:
            raise ValueError(
                f"{self.name}: the residuals are all zero, so the fit is exact and "
                "there is no serial correlation to measure"
            )
        top = math.fsum(
            (self.residuals[i] - self.residuals[i - 1]) ** 2
            for i in range(1, len(self.residuals))
        )
        return top / bottom

    def beta(self, factor: str) -> float:
        for index, held in enumerate(self.factor_names):
            if held == factor:
                return self.betas[index]
        raise KeyError(factor)

    def fitted(self, exposures: Sequence[float]) -> float:
        """The return this model predicts for one period's factor returns."""
        if len(exposures) != len(self.betas):
            raise ValueError(
                f"{len(exposures)} factor returns against {len(self.betas)} exposures"
            )
        return self.alpha + math.fsum(
            b * f for b, f in zip(self.betas, exposures, strict=True)
        )


def regress(
    asset: ReturnSeries, factors: Panel, *, intercept: bool = True
) -> Regression:
    """Least squares exposure of one asset to a set of factors.

    ``intercept`` on by default. Dropping it forces the fit through the origin,
    which is occasionally right — for excess returns over the same risk-free
    rate the factors are measured against — and much more often a way to make
    R-squared look better than it is, because without an intercept the statistic
    is measured against zero rather than against the asset's own mean and is not
    the same quantity at all.
    """
    if len(asset) != factors.observations:
        raise ValueError(
            f"{asset.name} has {len(asset)} observations against {factors.observations} "
            "for the factors; a regression needs them aligned"
        )
    parameters = factors.assets + (1 if intercept else 0)
    if len(asset) <= parameters:
        raise TooShort(
            f"{asset.name}: {len(asset)} observations cannot estimate {parameters} "
            "coefficients and leave anything to measure the fit with"
        )
    design = [
        ([1.0] if intercept else []) + factors.row(t) for t in range(factors.observations)
    ]
    target = list(asset.values)
    coefficients, upper = least_squares(design, target)
    alpha = coefficients[0] if intercept else 0.0
    betas = tuple(coefficients[1:] if intercept else coefficients)

    fitted = [
        math.fsum(
            value * coefficient
            for value, coefficient in zip(design[t], coefficients, strict=True)
        )
        for t in range(len(target))
    ]
    residuals = tuple(target[t] - fitted[t] for t in range(len(target)))
    residual_sum = math.fsum(value * value for value in residuals)
    freedom = len(target) - parameters
    specific_variance = residual_sum / freedom

    gram = upper_inverse_gram(upper)
    standard_errors = tuple(
        math.sqrt(max(specific_variance * gram[i][i], 0.0)) for i in range(parameters)
    )
    if not intercept:
        standard_errors = (0.0, *standard_errors)

    # Measured against the asset's own mean, so that R-squared is the share of
    # variance explained rather than the share of the raw second moment. Without
    # an intercept the two differ and the second is the one that flatters.
    centre = math.fsum(target) / len(target)
    total = math.fsum((value - centre) ** 2 for value in target)
    r_squared = 1.0 - residual_sum / total if total > 0.0 else 1.0

    return Regression(
        name=asset.name,
        alpha=alpha,
        betas=betas,
        factor_names=tuple(factors.names),
        residuals=residuals,
        observations=len(target),
        parameters=parameters,
        specific_variance=specific_variance,
        standard_errors=standard_errors,
        r_squared=r_squared,
    )


@dataclass(frozen=True)
class FactorModel:
    """Exposures, factor covariance and specific risk for a set of assets."""

    asset_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    #: ``exposures[i][k]`` is asset ``i``'s loading on factor ``k``.
    exposures: Matrix
    #: Covariance of the factors themselves.
    factor_covariance: Matrix
    #: Residual variance per asset — the diagonal of ``D``.
    specific_variances: tuple[float, ...]
    regressions: tuple[Regression, ...]

    @property
    def assets(self) -> int:
        return len(self.asset_names)

    @property
    def factors(self) -> int:
        return len(self.factor_names)

    def regression(self, name: str) -> Regression:
        for held in self.regressions:
            if held.name == name:
                return held
        raise KeyError(name)

    def portfolio_exposures(self, weights: Sequence[float]) -> Vector:
        """``B' w``: the portfolio's loading on each factor.

        The number an allocator acts on. A portfolio's exposure to a factor is
        the weighted sum of its holdings' exposures, and it can be large while
        no single holding's is.
        """
        if len(weights) != self.assets:
            raise ValueError(
                f"{len(weights)} weights against {self.assets} assets; they must agree"
            )
        return [
            math.fsum(
                weights[i] * self.exposures[i][k] for i in range(self.assets)
            )
            for k in range(self.factors)
        ]

    def implied_covariance(self) -> Matrix:
        """``B F B' + D``, the covariance the model implies between assets.

        Always positive semi-definite when the factor covariance is, whatever
        the sample length — which is the practical reason factor models are used
        on large universes. A sample covariance over ``n`` assets and ``T``
        periods is singular whenever ``T <= n``, and with a thousand assets it
        essentially always is; this one is built from a small matrix and a
        strictly positive diagonal and cannot be.
        """
        size = self.assets
        # B F, then (B F) B'. Done in that order so the work is proportional to
        # assets times factors rather than to assets squared times factors.
        loaded = [
            [
                math.fsum(
                    self.exposures[i][k] * self.factor_covariance[k][j]
                    for k in range(self.factors)
                )
                for j in range(self.factors)
            ]
            for i in range(size)
        ]
        out = [
            [
                math.fsum(
                    loaded[i][k] * self.exposures[j][k] for k in range(self.factors)
                )
                for j in range(size)
            ]
            for i in range(size)
        ]
        for i in range(size):
            out[i][i] += self.specific_variances[i]
        return symmetrise(out)

    def residual_correlations(self) -> Matrix:
        """Correlations between the residuals the model assumes are uncorrelated."""
        size = self.assets
        residuals = [one.residuals for one in self.regressions]
        count = len(residuals[0])
        centred = [
            [value - math.fsum(row) / count for value in row] for row in residuals
        ]
        norms = [math.sqrt(math.fsum(v * v for v in row)) for row in centred]
        out = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if norms[i] == 0.0 or norms[j] == 0.0:
                    out[i][j] = 1.0 if i == j else 0.0
                else:
                    out[i][j] = (
                        math.fsum(
                            centred[i][t] * centred[j][t] for t in range(count)
                        )
                        / (norms[i] * norms[j])
                    )
        return out

    def worst_residual_correlation(self) -> tuple[str, str, float]:
        """The pair of assets whose residuals are most correlated, and by how much.

        The diagnostic that says whether the diagonal specific-risk assumption
        can be believed for this data. A large value means the factors have
        missed something those two assets share, and the implied covariance is
        understating how much they move together — so a portfolio holding both
        looks better diversified than it is. It is reported signed, because a
        strongly *negative* residual correlation is the same failure.
        """
        worst = 0.0
        pair = (self.asset_names[0], self.asset_names[0])
        correlations = self.residual_correlations()
        for i in range(self.assets):
            for j in range(i + 1, self.assets):
                if abs(correlations[i][j]) > abs(worst):
                    worst = correlations[i][j]
                    pair = (self.asset_names[i], self.asset_names[j])
        return pair[0], pair[1], worst


def fit_factor_model(
    assets: Panel, factors: Panel, *, intercept: bool = True
) -> FactorModel:
    """Regress every asset on the same factors and assemble the model."""
    if assets.observations != factors.observations:
        raise ValueError(
            f"{assets.observations} asset observations against "
            f"{factors.observations} factor observations; they must be aligned"
        )
    regressions = tuple(
        regress(one, factors, intercept=intercept) for one in assets.series
    )
    exposures = [list(one.betas) for one in regressions]
    factor_covariance = _covariance_of(factors)
    return FactorModel(
        asset_names=tuple(assets.names),
        factor_names=tuple(factors.names),
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variances=tuple(one.specific_variance for one in regressions),
        regressions=regressions,
    )


def _covariance_of(panel: Panel) -> Matrix:
    """Unbiased sample covariance of a panel, as a plain matrix."""
    panel.require(2, "a factor covariance")
    rows = panel.demeaned()
    count = panel.observations
    size = panel.assets
    return symmetrise(
        [
            [
                math.fsum(rows[i][t] * rows[j][t] for t in range(count)) / (count - 1)
                for j in range(size)
            ]
            for i in range(size)
        ]
    )


@dataclass(frozen=True)
class Attribution:
    """Portfolio risk split between the factors and what they do not explain.

    **Two different quantities live here and they are kept apart on purpose.**

    A *standalone volatility* is what the portfolio's risk would be if only that
    source existed: :attr:`factor_volatility` and :attr:`specific_volatility`.
    These are the intuitive numbers and they do **not** add up — squares add, so
    a 12% factor volatility and a 5% specific volatility make 13%, not 17%.

    A *contribution* is that source's share of the total under the Euler
    allocation: :attr:`factor_contribution`, :attr:`specific_contribution` and
    the per-factor :attr:`components`. These do add up, exactly, which is what
    makes them allocatable — but no single one of them is a volatility anybody
    would recognise on its own.

    Reporting one under the other's name is the ordinary way an attribution
    stops reconciling, and the reader cannot tell from the numbers which kind
    they were handed.
    """

    #: Total portfolio volatility implied by the model.
    total: float
    #: Standalone volatility from the factors alone, ``sqrt(x' F x)``.
    factor_volatility: float
    #: Standalone volatility from the residuals alone, ``sqrt(sum w_i^2 d_i)``.
    specific_volatility: float
    factor_names: tuple[str, ...]
    #: Per-factor Euler contribution. These sum to :attr:`factor_contribution`.
    components: tuple[float, ...]
    #: The portfolio's loading on each factor.
    exposures: tuple[float, ...]

    @property
    def factor_contribution(self) -> float:
        """The factors' share of the total volatility, which is what adds."""
        return self.factor_volatility * self.factor_volatility / self.total

    @property
    def specific_contribution(self) -> float:
        """The residuals' share of the total volatility."""
        return self.specific_volatility * self.specific_volatility / self.total

    @property
    def sum_of_parts(self) -> float:
        return math.fsum(self.components) + self.specific_contribution

    @property
    def identity_error(self) -> float:
        return abs(self.sum_of_parts - self.total)

    @property
    def factor_share(self) -> float:
        """Share of *variance* explained by the factors, between zero and one.

        Variance rather than volatility, because variance is the thing that
        splits additively between the two sources. It is also the same quantity
        as the contribution share — ``factor_contribution / total`` is exactly
        this — which is worth knowing, because it means the Euler contributions
        answer the variance question while being denominated in volatility.
        """
        if self.total <= 0.0:
            raise ValueError("a portfolio with no risk has no share of it to report")
        return (self.factor_volatility * self.factor_volatility) / (
            self.total * self.total
        )

    def of(self, factor: str) -> float:
        for index, held in enumerate(self.factor_names):
            if held == factor:
                return self.components[index]
        raise KeyError(factor)

    def largest(self) -> tuple[str, float]:
        index = max(
            range(len(self.components)), key=lambda k: abs(self.components[k])
        )
        return self.factor_names[index], self.components[index]


def attribute_risk(weights: Sequence[float], model: FactorModel) -> Attribution:
    """Split a portfolio's model-implied volatility across factors and residuals.

    The variance decomposes without any choice being made::

        w' (B F B' + D) w  =  x' F x  +  sum_i w_i^2 d_i,    x = B' w

    — the first term the factors, the second the residuals, and nothing left
    over. Dividing each piece by the total volatility turns them into
    contributions that sum to the volatility, which is the Euler allocation
    applied to this representation and agrees with
    :func:`~shortfall.volatility_contributions` on the implied covariance.

    The per-factor split is ``x_k (F x)_k / sigma``, which can be negative: a
    factor the portfolio is short of, and which is positively correlated with
    the factors it is long of, genuinely reduces total risk. That is reported
    rather than squared away, for the same reason a hedging position's
    contribution is.
    """
    if len(weights) != model.assets:
        raise ValueError(
            f"{len(weights)} weights against {model.assets} assets; they must agree"
        )
    exposures = model.portfolio_exposures(weights)
    loaded = [
        math.fsum(
            model.factor_covariance[k][j] * exposures[j] for j in range(model.factors)
        )
        for k in range(model.factors)
    ]
    factor_variance = math.fsum(
        exposures[k] * loaded[k] for k in range(model.factors)
    )
    specific_variance = math.fsum(
        weights[i] * weights[i] * model.specific_variances[i]
        for i in range(model.assets)
    )
    total_variance = factor_variance + specific_variance
    if total_variance <= 0.0:
        raise ValueError(
            f"the model implies a portfolio variance of {total_variance!r}; the "
            "factor covariance is not positive semi-definite"
        )
    total = math.sqrt(total_variance)
    return Attribution(
        total=total,
        factor_volatility=math.sqrt(max(factor_variance, 0.0)),
        specific_volatility=math.sqrt(max(specific_variance, 0.0)),
        factor_names=model.factor_names,
        components=tuple(
            exposures[k] * loaded[k] / total for k in range(model.factors)
        ),
        exposures=tuple(exposures),
    )


__all__ = [
    "Attribution",
    "FactorModel",
    "Regression",
    "attribute_risk",
    "fit_factor_model",
    "regress",
]

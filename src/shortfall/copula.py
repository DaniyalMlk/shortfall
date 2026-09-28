"""Portfolio risk when the dependence is not elliptical.

Every other multi-asset route in this library ties the joint distribution to a
covariance matrix, and that is a stronger assumption than it looks.

Under a multivariate normal, the probability that two assets are *both* beyond
their own ``q`` quantile, divided by ``q``, goes to zero as ``q`` falls —
whatever the correlation. A correlation of 0.9 buys a lot of co-movement in the
body and asymptotically none in the tail. Under a multivariate Student-t the
limit is positive, which is better, but it is forced to be the same number for
every pair and the same in the upper tail as the lower. Neither describes a
portfolio that is mildly correlated day to day and moves as one in a crash, and
that portfolio is the reason the estimate is being computed.

Sklar's theorem gives the decomposition that separates the two questions. Any
joint distribution can be written as a copula — the joint distribution of the
marginals' own probability transforms — applied to marginals that are otherwise
unconstrained. So the tail *shape of each asset* and the *way they arrive
together* can be estimated separately, from statistics suited to each, and put
back together by simulation.

What that buys, and what it costs:

* The **dependence parameter comes from the ranks**, so it does not move when a
  marginal is re-estimated, and one joint outlier cannot drag it (see
  :mod:`shortfall.ranks`).
* The **degrees of freedom of the copula are fitted**, not assumed, by profile
  likelihood on the copula density with the correlation matrix held at its rank
  estimate. Low degrees of freedom are how joint tail arrivals get into the
  model, so leaving the number to a default would be choosing the answer.
* The **marginals can carry a fitted tail**. With empirical marginals no
  simulated draw for a single asset can exceed that asset's worst observation,
  so all the extrapolation is in the dependence. Splicing a generalised Pareto
  tail onto the empirical body (:class:`Marginal.EXTREME_VALUE`) puts it back —
  deliberately, and separately from the copula, so the two assumptions can be
  turned on and off one at a time.
* The **cost is Monte Carlo error**, which is reported, and a two-step estimator
  that is consistent but not efficient. Fitting the correlation matrix and the
  degrees of freedom jointly by maximum likelihood would be efficient under the
  model and would also let a misspecified tail contaminate the correlation.

**Measured, on the case the module exists for.** A five-asset equally weighted
portfolio, 1,500 observations simulated from a t copula at 4 degrees of freedom
with every pairwise tau at 0.35, empirical marginals from that same sample,
20,000 paths, over three samples and three simulation seeds each: the fitted
copula puts 99% expected shortfall 10.6% of itself above what the Gaussian
copula gives on the identical marginals, the identical correlation matrix and
the identical normal draws — 0.0433 against 0.0391. The spread of that figure
across the nine runs is 2.6 percentage points and its range is 6.7% to 13.7%,
which is the honest precision of it. At 99.5% the gap is 13.7%, and it keeps
widening as the quantile falls: the asymptotic statement showing up at quantiles
that can still be simulated. The fitted degrees of freedom came to 4.2, in a
range of 4.0 to 4.5.

**Where it matters is the opposite of where it is expected to.** Sweeping the
pairwise tau on the same construction, with the fitted copula against the
Gaussian one at 99% over three simulation seeds each:

=====  ===============  =======
tau    correlation      premium
=====  ===============  =======
0.05   0.078            +19.7%
0.15   0.233            +17.6%
0.30   0.454            +12.0%
0.50   0.707             +5.9%
0.70   0.891             +1.5%
0.90   0.988             -1.0%
=====  ===============  =======

The premium is *largest when the correlation is lowest* and gone by the time the
correlation approaches one. The usual intuition — that tail dependence matters
most for assets that are already correlated — has it backwards, and the reason is
worth following. At a correlation near one the Gaussian copula already moves
everything together, so the portfolio behaves like a single asset and no choice of
copula can change that asset's own marginal tail. At a correlation near zero the
Gaussian copula promises real diversification in the extremes, and that promise is
exactly what is illusory: a t copula at four degrees of freedom has a tail
dependence coefficient of 0.0756 at a correlation of *zero*, because the shared
mixing variable is what makes large moves arrive together and it does not care
about the correlation at all.

So the portfolio this module exists for is the one that looks diversified. A book
of assets correlated at 0.08 is the case where the covariance matrix is most
reassuring and most wrong.

The mechanism is easier to see in the copula alone. Taking the fitted t copula
and the Gaussian copula on the same correlation matrix, the fraction of draws
with *all five* assets below their own 5% point is 0.42% against 0.14%, a factor
of 3; below their own 1% point it is 0.057% against 0.005%, a factor of 11.
Independence would give 3.1e-7 and 1e-10. The factor grows as the quantile falls
because one of the two limits is zero.

**And on the case the module does not exist for.** Fed 1,500 observations from a
genuine Gaussian copula, the same procedure fits 92 degrees of freedom — at or
near the upper bound, which is how this reports "no tail dependence found" — and
puts 99% expected shortfall 0.2% above the Gaussian figure over the same nine
runs, with a spread of 0.2 percentage points and a range of -0.1% to +0.5%. The
method does not manufacture tail dependence that is not in the data, which is
the property that makes the 10.6% above worth anything.

Sign convention, as everywhere else here: results are stated in losses, and a
value at risk of 0.04 is a 4% loss.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from .distributions import normal_cdf, normal_ppf, student_t_cdf, student_t_ppf
from .extreme import (
    GeneralisedPareto,
    TailMethod,
    fit_generalised_pareto,
    threshold_for,
)
from .historical import HistoricalRisk, QuantileMethod, historical_risk
from .linalg import Matrix, NotPositiveDefinite, cholesky
from .ranks import elliptical_correlation, kendall_matrix, ranks_of
from .series import Misaligned, Panel, TooShort

#: Fewest paths a copula simulation will run, and the most. Same reasoning as
#: :mod:`shortfall.horizon`: a 99% quantile from a thousand paths is estimated
#: from ten of them, and the work is ``paths * assets`` in Python.
MIN_COPULA_PATHS: Final = 2_000
MAX_COPULA_PATHS: Final = 200_000

#: Most assets a fit will accept. The rank matrix is ``O(n^2 T log T)`` and the
#: simulation is ``O(paths * n^2)`` through the factor, so a fifty-asset panel at
#: forty thousand paths is a hundred million multiply-adds in pure Python. The
#: cap is about runtime and says so; nothing about the method breaks above it.
MAX_COPULA_ASSETS: Final = 40

#: Bounds on the fitted degrees of freedom. Below two the copula is still a
#: perfectly good copula — it has no moments, but a copula never uses them — and
#: the profile likelihood is flat enough down there that the number stops being
#: identified. Above the upper bound the t copula and the Gaussian one are
#: indistinguishable at any sample size available, so the fit reports the bound
#: and the caller should read it as "no tail dependence found".
MIN_COPULA_DEGREES: Final = 2.0
MAX_COPULA_DEGREES: Final = 100.0

#: Batches used for the Monte Carlo standard error.
COPULA_BATCHES: Final = 20

#: Points in the geometric grid the profile likelihood is scanned on, and how
#: finely the golden-section step then refines. Every evaluation is a pass over
#: the whole sample, so both numbers are runtime; neither is precision that the
#: estimate can support.
GRID_POINTS: Final = 14
DEGREES_TOLERANCE: Final = 0.1

#: Fewest observations the fit will accept. Two hundred is already thin for a
#: 99% figure through empirical marginals — the second-worst observation — and
#: the rank matrix needs enough to pin down the pairwise taus.
MIN_COPULA_OBSERVATIONS: Final = 200


class Family(str, Enum):
    """Which elliptical copula to fit."""

    #: No tail dependence at any correlation below one. The right choice when the
    #: data says so, and the wrong one by default.
    GAUSSIAN = "gaussian"
    #: One extra parameter, the degrees of freedom, which controls how often the
    #: assets arrive in their tails together. Symmetric between the upper and
    #: lower tail, which is a real limitation: equity crashes are not symmetric.
    STUDENT_T = "student_t"


class Marginal(str, Enum):
    """Where each asset's own distribution comes from."""

    #: The asset's own order statistics, interpolated. Assumes nothing about
    #: shape and cannot produce a draw beyond the sample for any single asset.
    EMPIRICAL = "empirical"
    #: Empirical through the body, and a generalised Pareto fitted to the worst
    #: :data:`TAIL_FRACTION` of each asset's losses beyond that. Extrapolates,
    #: and says so.
    EXTREME_VALUE = "extreme_value"


#: Fraction of each marginal's losses used for the spliced tail. 5% matches the
#: default in :mod:`shortfall.extreme`, where the threshold trade-off is measured.
TAIL_FRACTION: Final = 0.05


class NotIdentified(ValueError):
    """The profile likelihood gave no interior maximum for the degrees of freedom."""


@dataclass(frozen=True)
class TailDependence:
    """The coefficient of tail dependence for one pair, and what it means."""

    first: str
    second: str
    #: Correlation parameter of the copula for this pair.
    correlation: float
    #: ``lim P(U_2 > q | U_1 > q)`` as ``q`` goes to one. Zero for a Gaussian
    #: copula at any correlation below one; for a t copula,
    #: ``2 * t_{v+1}(-sqrt((v+1)(1-rho)/(1+rho)))``.
    coefficient: float

    @property
    def is_asymptotically_independent(self) -> bool:
        return self.coefficient <= 0.0


@dataclass(frozen=True)
class FittedCopula:
    """A copula fitted to the ranks of a panel."""

    family: Family
    names: tuple[str, ...]
    #: Correlation matrix, from the pairwise Kendall inversion.
    correlation: Matrix
    #: Kendall's tau matrix it was inverted from, carried so the estimate can be
    #: read alongside the statistic it came from.
    kendall: Matrix
    #: ``None`` for a Gaussian copula.
    degrees_of_freedom: float | None
    observations: int
    #: Copula log-likelihood at the fitted parameters.
    log_likelihood: float
    #: Copula log-likelihood of the Gaussian copula on the same correlation
    #: matrix. The difference is one degree of freedom, so twice it is a
    #: likelihood ratio statistic against a chi-square with one degree of
    #: freedom — though the null sits on the boundary of the parameter space, so
    #: read the size with care.
    gaussian_log_likelihood: float
    #: Whether the pairwise inversion had to be projected to be a valid
    #: correlation matrix. ``True`` means the reported correlations are not
    #: exactly the ones estimated.
    projected: bool

    @property
    def assets(self) -> int:
        return len(self.names)

    @property
    def likelihood_ratio(self) -> float:
        """``2 * (fitted - gaussian)``, against the Gaussian special case."""
        return 2.0 * (self.log_likelihood - self.gaussian_log_likelihood)

    def tail_dependence(self) -> tuple[TailDependence, ...]:
        """Every pair, in the order the panel gave them."""
        result = []
        for i in range(self.assets):
            for j in range(i + 1, self.assets):
                rho = self.correlation[i][j]
                result.append(
                    TailDependence(
                        first=self.names[i],
                        second=self.names[j],
                        correlation=rho,
                        coefficient=tail_dependence_coefficient(
                            rho, self.degrees_of_freedom
                        ),
                    )
                )
        return tuple(result)


@dataclass(frozen=True)
class CopulaRisk:
    """Simulated portfolio risk under a fitted copula."""

    #: Read off the simulated portfolio losses exactly as it would be off
    #: realised ones.
    risk: HistoricalRisk
    copula: FittedCopula
    marginal: Marginal
    paths: int
    weights: tuple[float, ...]
    #: The same figure with the same marginals and the same correlation matrix
    #: under a Gaussian copula. The difference is the dependence assumption and
    #: nothing else: the marginals, the weights, the correlation matrix and the
    #: random draws are shared.
    gaussian_value_at_risk: float
    gaussian_expected_shortfall: float
    #: Monte Carlo standard error of the value at risk, from the spread across
    #: :data:`COPULA_BATCHES` independent batches. Read it as a lower bound for
    #: the same reason as in :mod:`shortfall.horizon`: a batch of 2,000 paths
    #: estimates a 99% quantile from its own worst twenty.
    standard_error: float

    @property
    def value_at_risk(self) -> float:
        return self.risk.value_at_risk

    @property
    def expected_shortfall(self) -> float:
        return self.risk.expected_shortfall

    @property
    def tail_dependence_premium(self) -> float:
        """Expected shortfall relative to the Gaussian copula's, as a fraction.

        ``0.15`` means the fitted copula puts expected shortfall 15% above what
        the same marginals under a Gaussian copula give. Zero is the honest
        answer when the data has no tail dependence in it, and the estimate does
        return roughly zero there — measured in the module docstring.
        """
        if self.gaussian_expected_shortfall == 0.0:
            raise NoGaussianBaseline(
                "the Gaussian copula's expected shortfall is zero, so a relative "
                "premium is a division by zero; compare the levels instead"
            )
        return (
            self.expected_shortfall - self.gaussian_expected_shortfall
        ) / self.gaussian_expected_shortfall


class NoGaussianBaseline(ZeroDivisionError):
    """Asked for a relative premium against a zero baseline."""


def tail_dependence_coefficient(
    correlation: float, degrees_of_freedom: float | None
) -> float:
    """Coefficient of upper (equivalently lower) tail dependence.

    For the Gaussian copula this is zero for every correlation strictly below
    one, and the function returns that rather than a small number — it is a limit
    that is exactly zero, not an approximation. The finite-sample consequence is
    different and worth stating: at a correlation of 0.9 the *empirical* fraction
    of joint 1% exceedances under a Gaussian copula is far from zero, so a
    caller who checks the model against 2,000 observations will not see the
    asymptotic statement. It bites at the quantiles nobody has data for, which
    are the ones being extrapolated to.
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must lie in [-1, 1], got {correlation}")
    if degrees_of_freedom is None:
        return 1.0 if correlation >= 1.0 else 0.0
    if correlation >= 1.0:
        return 1.0
    if correlation <= -1.0:
        return 0.0
    argument = -math.sqrt(
        (degrees_of_freedom + 1.0) * (1.0 - correlation) / (1.0 + correlation)
    )
    return 2.0 * student_t_cdf(argument, degrees_of_freedom + 1.0)


def _log_determinant_and_solver(
    correlation: Matrix,
) -> tuple[float, Matrix]:
    """``log det R`` and the Cholesky factor, or a refusal."""
    try:
        lower = cholesky(correlation)
    except NotPositiveDefinite as error:  # pragma: no cover - guarded upstream
        raise NotPositiveDefinite(
            f"the copula correlation matrix is not positive definite: {error}"
        ) from error
    log_determinant = 2.0 * sum(math.log(lower[i][i]) for i in range(len(lower)))
    return log_determinant, lower


def _quadratic_through_inverse(lower: Matrix, vector: Sequence[float]) -> float:
    """``z' R^-1 z`` from the lower Cholesky factor, without inverting anything.

    ``R = L L'``, so ``z' R^-1 z = |L^-1 z|^2`` and the inner solve is a forward
    substitution. Forming the inverse and then a quadratic form would be three
    times the work and less accurate.
    """
    size = len(lower)
    solved = [0.0] * size
    for i in range(size):
        total = vector[i] - sum(lower[i][k] * solved[k] for k in range(i))
        solved[i] = total / lower[i][i]
    return sum(value * value for value in solved)


def gaussian_copula_log_likelihood(
    uniforms: Sequence[Sequence[float]], correlation: Matrix
) -> float:
    """Log-likelihood of a Gaussian copula at observed probability transforms.

    ``log c(u) = -0.5 log det R - 0.5 z'(R^-1 - I)z`` with ``z = Phi^-1(u)``. The
    ``-I`` is the marginal normal densities dividing out, and dropping it is the
    classic error: it leaves a term that grows with the sample and makes every
    correlation look better than independence.
    """
    log_determinant, lower = _log_determinant_and_solver(correlation)
    cache: dict[float, float] = {}
    total = 0.0
    for row in uniforms:
        z = []
        for value in row:
            quantile = cache.get(value)
            if quantile is None:
                quantile = normal_ppf(value)
                cache[value] = quantile
            z.append(quantile)
        quadratic = _quadratic_through_inverse(lower, z)
        total += -0.5 * quadratic + 0.5 * sum(value * value for value in z)
    return total - 0.5 * log_determinant * len(uniforms)


def student_t_copula_log_likelihood(
    uniforms: Sequence[Sequence[float]],
    correlation: Matrix,
    degrees_of_freedom: float,
) -> float:
    """Log-likelihood of a Student-t copula at observed probability transforms.

    The density is the multivariate t density at ``z_i = t_v^-1(u_i)`` divided by
    the product of the univariate t densities at the same points, which leaves

    ``lgamma((v+d)/2) + (d-1) lgamma(v/2) - d lgamma((v+1)/2)
      - 0.5 log det R - ((v+d)/2) log(1 + z'R^-1 z / v)
      + ((v+1)/2) sum log(1 + z_i^2 / v)``

    per observation. Everything but the two logarithms is constant across the
    sample, which is what makes a profile search over ``v`` affordable.
    """
    if degrees_of_freedom <= 0.0:
        raise ValueError(
            f"degrees of freedom must be positive, got {degrees_of_freedom}"
        )
    log_determinant, lower = _log_determinant_and_solver(correlation)
    dimension = len(correlation)
    half = degrees_of_freedom / 2.0
    constant = (
        math.lgamma(half + dimension / 2.0)
        + (dimension - 1) * math.lgamma(half)
        - dimension * math.lgamma(half + 0.5)
        - 0.5 * log_determinant
    )
    # The quantile function is the expensive part — a root-find per call — and on
    # pseudo-observations the same arguments recur across assets: the ranks of
    # every column are a permutation of 1..T, so a panel of T observations and d
    # assets presents T distinct probabilities rather than T*d. Memoising within
    # the call is exact and saves a factor of d; measured on 800 observations of
    # four assets it took one evaluation from 835ms to 201ms.
    cache: dict[float, float] = {}
    total = 0.0
    for row in uniforms:
        z = []
        for value in row:
            quantile = cache.get(value)
            if quantile is None:
                quantile = student_t_ppf(value, degrees_of_freedom)
                cache[value] = quantile
            z.append(quantile)
        quadratic = _quadratic_through_inverse(lower, z)
        total += constant
        total -= (degrees_of_freedom + dimension) / 2.0 * math.log1p(
            quadratic / degrees_of_freedom
        )
        total += sum(
            (degrees_of_freedom + 1.0)
            / 2.0
            * math.log1p(value * value / degrees_of_freedom)
            for value in z
        )
    return total


def _pseudo_observations(panel: Panel) -> list[list[float]]:
    """Ranks scaled into ``(0, 1)`` by ``rank / (T + 1)``.

    The ``T + 1`` rather than ``T`` is not cosmetic: dividing by ``T`` maps the
    worst observation to exactly 1, whose normal and t quantiles are infinite, and
    the likelihood is then undefined rather than large.
    """
    observations = panel.observations
    columns = [
        ranks_of(list(panel.column(i).values)) for i in range(panel.assets)
    ]
    return [
        [columns[asset][t] / (observations + 1.0) for asset in range(panel.assets)]
        for t in range(observations)
    ]


def fit_copula(
    panel: Panel,
    *,
    family: Family = Family.STUDENT_T,
    degrees_of_freedom: float | None = None,
) -> FittedCopula:
    """Fit an elliptical copula to a panel's ranks.

    Two steps, deliberately. The correlation matrix comes from the pairwise
    Kendall inversion, which holds for every member of the elliptical family and
    so does not depend on getting the tail right. The degrees of freedom then come
    from a profile likelihood with that matrix fixed. Estimating both at once
    would be more efficient under the model and would let a misspecified tail
    move the correlations, which is the trade this takes the other side of.

    Pass ``degrees_of_freedom`` to fix it rather than fit it — useful for a
    stress scenario, where the point is to ask what a heavier tail would do
    rather than what the sample supports.
    """
    if panel.assets < 2:
        raise TooShort(
            f"a copula describes the dependence between at least two assets, "
            f"got {panel.assets}"
        )
    if panel.assets > MAX_COPULA_ASSETS:
        raise TooShort(
            f"at most {MAX_COPULA_ASSETS} assets, got {panel.assets}: the rank "
            f"matrix and the simulation are both quadratic in the count"
        )
    if panel.observations < MIN_COPULA_OBSERVATIONS:
        raise TooShort(
            f"a copula fit needs at least {MIN_COPULA_OBSERVATIONS} observations, "
            f"got {panel.observations}"
        )

    kendall = kendall_matrix(panel)
    correlation, projected = elliptical_correlation(kendall)
    uniforms = _pseudo_observations(panel)
    gaussian = gaussian_copula_log_likelihood(uniforms, correlation)

    if family is Family.GAUSSIAN:
        if degrees_of_freedom is not None:
            raise ValueError(
                "a Gaussian copula has no degrees of freedom; pass "
                "family=Family.STUDENT_T to fix one"
            )
        return FittedCopula(
            family=family,
            names=tuple(panel.names),
            correlation=correlation,
            kendall=kendall,
            degrees_of_freedom=None,
            observations=panel.observations,
            log_likelihood=gaussian,
            gaussian_log_likelihood=gaussian,
            projected=projected,
        )

    if degrees_of_freedom is None:
        fitted = _profile_degrees_of_freedom(uniforms, correlation)
    else:
        if not MIN_COPULA_DEGREES <= degrees_of_freedom <= MAX_COPULA_DEGREES:
            raise ValueError(
                f"degrees of freedom must lie in "
                f"[{MIN_COPULA_DEGREES}, {MAX_COPULA_DEGREES}], "
                f"got {degrees_of_freedom}"
            )
        fitted = degrees_of_freedom
    log_likelihood = student_t_copula_log_likelihood(uniforms, correlation, fitted)
    return FittedCopula(
        family=family,
        names=tuple(panel.names),
        correlation=correlation,
        kendall=kendall,
        degrees_of_freedom=fitted,
        observations=panel.observations,
        log_likelihood=log_likelihood,
        gaussian_log_likelihood=gaussian,
        projected=projected,
    )


def _profile_degrees_of_freedom(
    uniforms: Sequence[Sequence[float]], correlation: Matrix
) -> float:
    """Maximise the copula likelihood over the degrees of freedom.

    A coarse grid, then a golden-section refinement inside the best bracket. The
    grid is geometric because the likelihood is close to flat in ``v`` at the top
    of the range and steep at the bottom: twenty points evenly spaced between 2
    and 100 would put most of them where the function is not changing.

    A maximum at the upper bound is returned as the bound rather than raised as a
    failure. It is the correct answer to "how heavy is the joint tail" when the
    answer is "not measurably heavy", and the caller can see the value and the
    likelihood ratio and read it that way.
    """
    # Geometric, so `**` on floats — annotated because typeshed types that as
    # returning Any, a float base being allowed to produce a complex result.
    grid: list[float] = [
        MIN_COPULA_DEGREES
        * (MAX_COPULA_DEGREES / MIN_COPULA_DEGREES) ** (k / (GRID_POINTS - 1.0))
        for k in range(GRID_POINTS)
    ]
    scores = [
        student_t_copula_log_likelihood(uniforms, correlation, value)
        for value in grid
    ]
    best = max(range(len(grid)), key=lambda k: scores[k])
    if best == 0:
        return grid[0]
    if best == len(grid) - 1:
        return grid[-1]

    low, high = grid[best - 1], grid[best + 1]
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - inverse_phi * (high - low)
    right = low + inverse_phi * (high - low)
    score_left = student_t_copula_log_likelihood(uniforms, correlation, left)
    score_right = student_t_copula_log_likelihood(uniforms, correlation, right)
    for _ in range(40):
        # A tenth of a degree of freedom. The standard error of this estimate is
        # of order one on a sample this size, so refining further buys nothing
        # and each step is a full pass over the sample.
        if high - low < DEGREES_TOLERANCE:
            break
        if score_left < score_right:
            low, left, score_left = left, right, score_right
            right = low + inverse_phi * (high - low)
            score_right = student_t_copula_log_likelihood(
                uniforms, correlation, right
            )
        else:
            high, right, score_right = right, left, score_left
            left = high - inverse_phi * (high - low)
            score_left = student_t_copula_log_likelihood(uniforms, correlation, left)
    return (low + high) / 2.0


@dataclass(frozen=True)
class _EmpiricalMarginal:
    """Inverts a probability to a return through the order statistics."""

    sorted_values: tuple[float, ...]

    def quantile(self, probability: float) -> float:
        n = len(self.sorted_values)
        position = probability * (n - 1)
        low = math.floor(position)
        if low >= n - 1:
            return self.sorted_values[-1]
        weight = position - low
        return (
            self.sorted_values[low] * (1.0 - weight)
            + self.sorted_values[low + 1] * weight
        )


@dataclass(frozen=True)
class _SplicedMarginal:
    """Empirical in the body, generalised Pareto in the lower tail.

    Only the loss tail is spliced. The upper tail of a return distribution is not
    where portfolio risk lives, and fitting it would double the number of
    extrapolations for no gain to the number being asked for.
    """

    body: _EmpiricalMarginal
    tail: GeneralisedPareto
    #: Probability below which the fitted tail takes over, in return space.
    #:
    #: This is the fit's *realised* exceedance fraction, not the tail fraction
    #: that was asked for, and the difference is not cosmetic. A threshold at the
    #: 5% point of 800 losses is exceeded by 39 of them, not 40, so the fit
    #: describes the worst 4.875% and refuses anything shallower — it knows it
    #: was told nothing about the body. Splicing at the nominal 5% therefore
    #: routes a thin band of probabilities to a tail that declines to answer for
    #: them.
    crossover: float

    def quantile(self, probability: float) -> float:
        if probability >= self.crossover:
            return self.body.quantile(probability)
        # In loss space the tail probability is `probability`; the fit's own
        # quantile function is stated in losses, hence the negation.
        loss_confidence = 1.0 - probability
        return -self.tail.quantile(loss_confidence)


_MarginalModel = _EmpiricalMarginal | _SplicedMarginal


def _build_marginals(panel: Panel, marginal: Marginal) -> list[_MarginalModel]:
    models: list[_MarginalModel] = []
    for index in range(panel.assets):
        values = list(panel.column(index).values)
        body = _EmpiricalMarginal(tuple(sorted(values)))
        if marginal is Marginal.EMPIRICAL:
            models.append(body)
            continue
        losses = [-value for value in values]
        threshold = threshold_for(losses, TAIL_FRACTION)
        excesses = [loss - threshold for loss in losses if loss > threshold]
        fit = fit_generalised_pareto(
            excesses,
            threshold=threshold,
            observations=len(losses),
            method=TailMethod.MAXIMUM_LIKELIHOOD,
        )
        models.append(
            _SplicedMarginal(
                body=body, tail=fit, crossover=1.0 - fit.lowest_confidence
            )
        )
    return models


def _draw_uniforms(
    rng: random.Random,
    lower: Matrix,
    degrees_of_freedom: float | None,
) -> list[float]:
    """One draw of probability transforms from the copula.

    Gaussian: ``u = Phi(L e)``. Student-t: divide the same correlated normal by
    ``sqrt(w / v)`` for a chi-square ``w``, then map through the univariate t.
    The single scalar ``w`` shared across assets is the whole mechanism — it is
    what makes large moves arrive together rather than independently.
    """
    size = len(lower)
    normals = [rng.gauss(0.0, 1.0) for _ in range(size)]
    correlated = [
        sum(lower[i][k] * normals[k] for k in range(i + 1)) for i in range(size)
    ]
    if degrees_of_freedom is None:
        return [normal_cdf(value) for value in correlated]
    # chi-square with v degrees of freedom is gamma(v/2, scale=2).
    mixing = rng.gammavariate(degrees_of_freedom / 2.0, 2.0)
    scale = math.sqrt(degrees_of_freedom / mixing)
    return [
        student_t_cdf(value * scale, degrees_of_freedom) for value in correlated
    ]


def copula_risk(
    panel: Panel,
    weights: Sequence[float],
    *,
    confidence: float = 0.99,
    family: Family = Family.STUDENT_T,
    marginal: Marginal = Marginal.EMPIRICAL,
    degrees_of_freedom: float | None = None,
    paths: int = 40_000,
    seed: int | None = None,
    method: QuantileMethod = QuantileMethod.LINEAR,
) -> CopulaRisk:
    """Portfolio value at risk and expected shortfall under a fitted copula.

    The Gaussian comparison in the result is computed from the *same* standard
    normal draws as the fitted copula, differing only in whether they are divided
    by the chi-square mixing variable. That makes the two figures paired rather
    than independent, and their difference a cleaner read on the assumption than
    two separate simulations would give.

    How much cleaner is worth measuring rather than asserting, because the answer
    is "about half" and not "almost all". Over six simulation seeds at 20,000
    paths the spread of the relative difference is 2.4 percentage points paired
    against 4.6 unpaired on a sample drawn at four degrees of freedom, and 1.3
    against 2.8 on the bundled example series. The reason it is not better is
    structural: the normals are shared and the chi-square mixing draw is not, and
    it cannot be — that variable is the entire difference between the two
    copulas, so there is nothing to share.
    """
    if len(weights) != panel.assets:
        raise Misaligned(
            f"{len(weights)} weights for {panel.assets} assets"
        )
    if not MIN_COPULA_PATHS <= paths <= MAX_COPULA_PATHS:
        raise ValueError(
            f"paths must lie in [{MIN_COPULA_PATHS}, {MAX_COPULA_PATHS}], "
            f"got {paths}"
        )
    if not 0.5 <= confidence < 1.0:
        raise ValueError(f"confidence must lie in [0.5, 1), got {confidence}")

    fitted = fit_copula(
        panel, family=family, degrees_of_freedom=degrees_of_freedom
    )
    models = _build_marginals(panel, marginal)
    _, lower = _log_determinant_and_solver(fitted.correlation)
    rng = random.Random(seed)
    size = panel.assets

    fitted_losses: list[float] = []
    gaussian_losses: list[float] = []
    for _ in range(paths):
        normals = [rng.gauss(0.0, 1.0) for _ in range(size)]
        correlated = [
            sum(lower[i][k] * normals[k] for k in range(i + 1))
            for i in range(size)
        ]
        gaussian_uniforms = [normal_cdf(value) for value in correlated]
        gaussian_losses.append(
            -_portfolio_return(gaussian_uniforms, models, weights)
        )
        if fitted.degrees_of_freedom is None:
            fitted_losses.append(gaussian_losses[-1])
            continue
        mixing = rng.gammavariate(fitted.degrees_of_freedom / 2.0, 2.0)
        scale = math.sqrt(fitted.degrees_of_freedom / mixing)
        uniforms = [
            student_t_cdf(value * scale, fitted.degrees_of_freedom)
            for value in correlated
        ]
        fitted_losses.append(-_portfolio_return(uniforms, models, weights))

    risk = historical_risk(
        [-loss for loss in fitted_losses], confidence=confidence, method=method
    )
    gaussian = historical_risk(
        [-loss for loss in gaussian_losses], confidence=confidence, method=method
    )
    return CopulaRisk(
        risk=risk,
        copula=fitted,
        marginal=marginal,
        paths=paths,
        weights=tuple(weights),
        gaussian_value_at_risk=gaussian.value_at_risk,
        gaussian_expected_shortfall=gaussian.expected_shortfall,
        standard_error=_batch_standard_error(
            fitted_losses, confidence=confidence, method=method
        ),
    )


def _portfolio_return(
    uniforms: Sequence[float],
    models: Sequence[_MarginalModel],
    weights: Sequence[float],
) -> float:
    return sum(
        weight * model.quantile(u)
        for weight, model, u in zip(weights, models, uniforms, strict=True)
    )


def _batch_standard_error(
    losses: Sequence[float], *, confidence: float, method: QuantileMethod
) -> float:
    """Standard error of the value at risk across independent batches."""
    size = len(losses) // COPULA_BATCHES
    if size < 2:  # pragma: no cover - MIN_COPULA_PATHS makes this unreachable
        return float("nan")
    estimates = []
    for batch in range(COPULA_BATCHES):
        slice_ = losses[batch * size : (batch + 1) * size]
        estimates.append(
            historical_risk(
                [-loss for loss in slice_], confidence=confidence, method=method
            ).value_at_risk
        )
    mean = sum(estimates) / len(estimates)
    variance = sum((value - mean) ** 2 for value in estimates) / (
        len(estimates) - 1
    )
    return math.sqrt(variance / len(estimates))


def simulate_copula(
    fitted: FittedCopula,
    *,
    paths: int = 10_000,
    seed: int | None = None,
) -> list[list[float]]:
    """Probability transforms drawn from a fitted copula.

    Exported because the useful thing to do with a copula is not always to price
    a portfolio with it. Feeding these through any quantile function gives a
    joint sample with that copula's dependence and whatever marginals are wanted,
    including ones this library knows nothing about.
    """
    if not MIN_COPULA_PATHS <= paths <= MAX_COPULA_PATHS:
        raise ValueError(
            f"paths must lie in [{MIN_COPULA_PATHS}, {MAX_COPULA_PATHS}], "
            f"got {paths}"
        )
    _, lower = _log_determinant_and_solver(fitted.correlation)
    rng = random.Random(seed)
    return [
        _draw_uniforms(rng, lower, fitted.degrees_of_freedom)
        for _ in range(paths)
    ]


def joint_exceedance_rate(
    uniforms: Sequence[Sequence[float]], *, quantile: float = 0.01
) -> float:
    """Fraction of draws where *every* asset is below its own ``quantile``.

    The finite-sample counterpart of the tail dependence coefficient, and the
    diagnostic worth looking at because it is the thing a portfolio actually
    experiences. Under independence this is ``quantile ** assets``, which for
    five assets at 1% is one in ten billion; a fitted t copula puts it several
    orders of magnitude above that, and the ratio is the clearest single number
    for what the dependence assumption is doing.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}")
    if not uniforms:
        raise TooShort("no draws to count exceedances in")
    hits = sum(1 for row in uniforms if all(value <= quantile for value in row))
    return hits / len(uniforms)

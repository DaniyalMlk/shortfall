"""Covariance estimation, and the diagnostics that say how far to trust it.

Every risk number downstream of here reads a covariance matrix, so the quality
of the risk is bounded by the quality of this estimate — and the sample
covariance is a worse estimate than its familiarity suggests.

With ``T`` observations of ``n`` assets it fits ``n(n+1)/2`` parameters from
``nT`` numbers. When ``T`` is not large relative to ``n`` the eigenvalues spread
out: the largest are biased up and the smallest are biased down, towards zero
and past it once ``T < n``, where the matrix is singular outright. That is not a
uniform loss of accuracy. An optimiser looking for low-variance directions finds
exactly the eigenvectors whose variance was most understated, and loads up on
them, so the error concentrates precisely where the portfolio is about to.

Shrinkage is the standard answer and a good one. Pull the sample matrix towards
a structured target with far fewer parameters, and trade a little bias for a
large reduction in variance. Ledoit and Wolf's contribution is that the optimal
amount to pull can be estimated from the data rather than tuned, which turns a
judgement call into an estimate with a standard error.

Two targets are offered. The default is constant correlation: every pairwise
correlation replaced by the average of them all, variances left alone. It is the
one from *Honey, I Shrunk the Sample Covariance Matrix* (Ledoit and Wolf, 2003),
and it suits equities, where the dominant structure really is that everything is
correlated with everything else by roughly the same amount.

The other is the scaled identity of *A Well-Conditioned Estimator for
Large-Dimensional Covariance Matrices* (Ledoit and Wolf, 2004): every variance
replaced by their average and every covariance by zero. It suits assets on
comparable scales with little correlation between them, and nothing else: it
pulls the variances together as well as the correlations, which constant
correlation deliberately does not. Measured on ten assets and forty
observations, against the true matrix in squared Frobenius distance, it has
about a seventh of constant correlation's error when the assets are
uncorrelated and equally volatile, and more than twice it either when they are
uncorrelated with differing volatilities or when they share a common factor.
``tests/test_shrinkage_targets.py`` holds those three cases.

The intensity is reported rather than hidden. A shrinkage of 0.05 says the
sample was informative; one of 0.9 says the answer is mostly the target and the
data contributed little, and a caller who cannot see which of those happened
cannot tell a risk estimate from a prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .linalg import (
    Matrix,
    Vector,
    condition_number,
    eigenvalues,
    is_positive_semidefinite,
)
from .series import Panel, TooShort


@dataclass(frozen=True)
class Diagnostics:
    """What an estimated matrix looks like from the outside."""

    assets: int
    observations: int
    condition_number: float
    smallest_eigenvalue: float
    largest_eigenvalue: float
    positive_semidefinite: bool
    #: Observations per estimated parameter. Below about 2 the sample covariance
    #: is not worth using unshrunk, and below 1 it is singular by construction.
    observations_per_parameter: float

    #: Below this ratio of smallest to largest eigenvalue, the matrix is
    #: rank-deficient as far as double precision can tell. It is not a tuning
    #: knob: it is a little above the point where a Cholesky factorisation of
    #: the matrix stops being meaningful.
    SINGULAR_RATIO = 1e-14

    @property
    def numerically_singular(self) -> bool:
        """Whether the matrix is rank-deficient as far as double precision can tell.

        Tested as a ratio rather than as ``isinf(condition_number)``. A sample
        covariance from fewer observations than assets is singular in exact
        arithmetic and arrives here with a smallest eigenvalue around 1e-18
        rather than 0, so testing for infinity would call it well conditioned.
        """
        if self.largest_eigenvalue <= 0.0:
            return True
        return self.smallest_eigenvalue <= self.largest_eigenvalue * self.SINGULAR_RATIO


def diagnose(matrix: Matrix, *, observations: int) -> Diagnostics:
    """Describe an estimated matrix: conditioning, definiteness, how much data fed it."""
    values = eigenvalues(matrix)
    assets = len(values)
    parameters = assets * (assets + 1) / 2
    return Diagnostics(
        assets=assets,
        observations=observations,
        condition_number=condition_number(matrix),
        smallest_eigenvalue=min(values),
        largest_eigenvalue=max(values),
        positive_semidefinite=is_positive_semidefinite(matrix),
        observations_per_parameter=(assets * observations) / parameters,
    )


def sample_covariance(panel: Panel, *, ddof: int = 1) -> Matrix:
    """The sample covariance matrix.

    ``ddof=1`` is Bessel's correction and gives the unbiased estimate; ``ddof=0``
    gives the maximum likelihood one. The shrinkage estimator below is defined
    in terms of the second, so both exist rather than one being hidden inside
    the other.
    """
    count = panel.observations
    if count - ddof <= 0:
        raise TooShort(
            f"a covariance with ddof={ddof} needs more than {ddof} observations, "
            f"this panel has {count}"
        )
    centred = panel.demeaned()
    size = panel.assets
    out = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            total = sum(centred[i][t] * centred[j][t] for t in range(count))
            value = total / (count - ddof)
            out[i][j] = value
            out[j][i] = value
    return out


def correlation(covariance: Matrix) -> Matrix:
    """Rescale a covariance matrix to unit diagonal.

    An asset with zero estimated variance has no correlation with anything —
    the quantity is 0/0 — so it is reported as zero off the diagonal and one on
    it, rather than as ``nan``. A ``nan`` would propagate silently through an
    average and turn one dead series into a matrix of nothing.
    """
    size = len(covariance)
    scale = [math.sqrt(covariance[i][i]) if covariance[i][i] > 0 else 0.0 for i in range(size)]
    out = [[0.0] * size for _ in range(size)]
    for i in range(size):
        out[i][i] = 1.0
        for j in range(i + 1, size):
            if scale[i] == 0.0 or scale[j] == 0.0:
                continue
            value = covariance[i][j] / (scale[i] * scale[j])
            out[i][j] = value
            out[j][i] = value
    return out


def average_correlation(covariance: Matrix) -> float:
    """The mean of the off-diagonal correlations, which is the shrinkage target's
    single structural parameter."""
    size = len(covariance)
    if size < 2:
        return 0.0
    correlations = correlation(covariance)
    total = sum(correlations[i][j] for i in range(size) for j in range(i + 1, size))
    return total / (size * (size - 1) / 2)


def constant_correlation_target(covariance: Matrix) -> Matrix:
    """The constant-correlation matrix with the same variances as ``covariance``.

    Same diagonal, and every off-diagonal correlation replaced by the average of
    them. Keeping the variances is what makes this a good target: individual
    variances are estimated from ``T`` numbers each and are comparatively
    reliable, whereas the ``n(n-1)/2`` correlations are where the parameter count
    lives and where the noise is.
    """
    size = len(covariance)
    mean_correlation = average_correlation(covariance)
    scale = [math.sqrt(max(covariance[i][i], 0.0)) for i in range(size)]
    out = [[0.0] * size for _ in range(size)]
    for i in range(size):
        out[i][i] = covariance[i][i]
        for j in range(i + 1, size):
            value = mean_correlation * scale[i] * scale[j]
            out[i][j] = value
            out[j][i] = value
    return out


def scaled_identity_target(covariance: Matrix) -> Matrix:
    """The average variance times the identity: ``(trace / n) * I``.

    The target of Ledoit and Wolf (2004). It shares the sample's total variance
    and nothing else, so the blend keeps the trace of the sample — the sum of the
    eigenvalues — while pulling every eigenvalue towards their mean. That is the
    sense in which it is well conditioned: the smallest eigenvalue of the blend
    is at least ``intensity`` times the average variance.
    """
    size = len(covariance)
    level = sum(covariance[i][i] for i in range(size)) / size if size else 0.0
    return [[level if i == j else 0.0 for j in range(size)] for i in range(size)]


class ShrinkageTarget(str, Enum):
    """What :func:`ledoit_wolf` pulls the sample covariance towards."""

    #: Sample variances, and every correlation replaced by their average.
    CONSTANT_CORRELATION = "constant_correlation"
    #: The average variance on the diagonal and zero elsewhere.
    SCALED_IDENTITY = "scaled_identity"


def blend(sample: Matrix, target: Matrix, intensity: float) -> Matrix:
    """``intensity * target + (1 - intensity) * sample``."""
    if not 0.0 <= intensity <= 1.0:
        raise ValueError(f"intensity must be in [0, 1], got {intensity!r}")
    size = len(sample)
    return [
        [intensity * target[i][j] + (1.0 - intensity) * sample[i][j] for j in range(size)]
        for i in range(size)
    ]


@dataclass(frozen=True)
class Shrunk:
    """A shrunk covariance matrix, with everything that went into it.

    The parts are kept rather than discarded because the intensity alone does
    not say whether to believe the answer. A caller comparing the sample and the
    target can see how far apart they were, and therefore how much the shrinkage
    actually changed.
    """

    matrix: Matrix
    intensity: float
    sample: Matrix
    target: Matrix
    average_correlation: float
    #: The unclamped optimum. It falls outside [0, 1] on small samples, and
    #: seeing that it did is the signal that the estimate is being driven by the
    #: clamp rather than by the data.
    unclamped_intensity: float
    diagnostics: Diagnostics
    #: Which target :attr:`target` is. :attr:`average_correlation` is the
    #: sample's under either, and is the target's own parameter only under
    #: constant correlation.
    target_kind: ShrinkageTarget = ShrinkageTarget.CONSTANT_CORRELATION

    @property
    def clamped(self) -> bool:
        return self.unclamped_intensity != self.intensity


def ledoit_wolf(
    panel: Panel, *, target: ShrinkageTarget | str = ShrinkageTarget.CONSTANT_CORRELATION
) -> Shrunk:
    """Shrink the sample covariance towards a structured target.

    ``target`` chooses it: constant correlation by default, following Ledoit and
    Wolf (2003), or the scaled identity of Ledoit and Wolf (2004). The
    derivation below is the 2003 one; the identity is the special case noted
    under ``rho``. The optimal intensity minimises the expected
    squared Frobenius distance to the true covariance, and decomposes into three
    estimated quantities:

    ``pi``
        The summed asymptotic variance of the sample covariance entries — how
        noisy the sample estimate is. Large ``pi`` argues for shrinking more.
    ``rho``
        The summed asymptotic covariance between the sample entries and the
        target's, since the target is itself estimated from the same data. This
        is the term a naive derivation omits, and omitting it over-shrinks.
        Under the scaled identity it is zero: the 2004 estimator treats the
        average variance as known, since it is one number estimated from all
        ``n T`` observations, and the intensity reduces to the paper's
        ``b^2 / d^2``.
    ``gamma``
        The squared distance between sample and target — the bias shrinkage
        would introduce. Large ``gamma`` argues for shrinking less.

    The optimum is ``(pi - rho) / gamma / T``, clamped to ``[0, 1]``.

    Everything here uses the maximum likelihood covariance, dividing by ``T``
    rather than ``T - 1``, because that is what the asymptotics are written
    against. Mixing the two conventions inside the formula is a subtle way to
    get an intensity that is wrong by a factor of ``T/(T-1)``.
    """
    kind = ShrinkageTarget(target)
    panel.require(2, "Ledoit-Wolf shrinkage")
    count = panel.observations
    size = panel.assets

    sample_ml = sample_covariance(panel, ddof=0)
    unbiased = sample_covariance(panel, ddof=1)
    if kind is ShrinkageTarget.SCALED_IDENTITY:
        return _towards_identity(panel, sample_ml, unbiased)
    target_ml = constant_correlation_target(sample_ml)
    mean_correlation = average_correlation(sample_ml)
    centred = panel.demeaned()

    # pi: summed variance of the sample entries.
    pi_matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            total = sum(
                (centred[i][t] * centred[j][t] - sample_ml[i][j]) ** 2 for t in range(count)
            )
            value = total / count
            pi_matrix[i][j] = value
            pi_matrix[j][i] = value
    pi = sum(pi_matrix[i][j] for i in range(size) for j in range(size))

    # rho: the diagonal terms, plus the cross terms that come from the target
    # sharing the sample's variances.
    rho = sum(pi_matrix[i][i] for i in range(size))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            var_i = sample_ml[i][i]
            var_j = sample_ml[j][j]
            if var_i <= 0.0 or var_j <= 0.0:
                # A series with no variance contributes no correlation to the
                # target, so it contributes no covariance with it either.
                continue
            theta_ii = (
                sum(
                    (centred[i][t] ** 2 - var_i)
                    * (centred[i][t] * centred[j][t] - sample_ml[i][j])
                    for t in range(count)
                )
                / count
            )
            theta_jj = (
                sum(
                    (centred[j][t] ** 2 - var_j)
                    * (centred[i][t] * centred[j][t] - sample_ml[i][j])
                    for t in range(count)
                )
                / count
            )
            rho += (mean_correlation / 2.0) * (
                math.sqrt(var_j / var_i) * theta_ii + math.sqrt(var_i / var_j) * theta_jj
            )

    # gamma: the bias shrinkage would introduce.
    gamma = sum(
        (target_ml[i][j] - sample_ml[i][j]) ** 2 for i in range(size) for j in range(size)
    )

    # gamma == 0 means sample and target coincide, so shrinkage changes nothing
    # and the optimum is undefined rather than infinite. Reported as 1: taking
    # the target costs nothing when it *is* the sample, and the alternative is
    # dividing by zero to say the same thing.
    unclamped = 1.0 if gamma == 0.0 else (pi - rho) / gamma / count
    intensity = min(1.0, max(0.0, unclamped))

    rescaled = _rescale_target(target_ml, unbiased)
    matrix = blend(unbiased, rescaled, intensity)
    return Shrunk(
        matrix=matrix,
        intensity=intensity,
        sample=unbiased,
        target=rescaled,
        average_correlation=mean_correlation,
        unclamped_intensity=unclamped,
        diagnostics=diagnose(matrix, observations=count),
    )


def _summed_entry_variance(panel: Panel, sample_ml: Matrix) -> Matrix:
    """``pi_ij``: the asymptotic variance of each sample covariance entry."""
    count = panel.observations
    size = panel.assets
    centred = panel.demeaned()
    out = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            total = sum(
                (centred[i][t] * centred[j][t] - sample_ml[i][j]) ** 2 for t in range(count)
            )
            value = total / count
            out[i][j] = value
            out[j][i] = value
    return out


def _towards_identity(panel: Panel, sample_ml: Matrix, unbiased: Matrix) -> Shrunk:
    """Ledoit and Wolf (2004): ``pi / gamma / T`` with ``rho`` nil.

    The target is rescaled by multiplying, not through correlations as the
    constant-correlation one is: its diagonal is the *average* variance, so
    rebuilding it on the sample's own variances would turn the scaled identity
    into the diagonal of the sample — a different target with a different
    optimum. Both conventions' averages differ by exactly ``T / (T - 1)``,
    which is the factor the unbiased target carries.
    """
    count = panel.observations
    size = panel.assets
    target_ml = scaled_identity_target(sample_ml)
    pi_matrix = _summed_entry_variance(panel, sample_ml)
    pi = sum(pi_matrix[i][j] for i in range(size) for j in range(size))
    gamma = sum(
        (target_ml[i][j] - sample_ml[i][j]) ** 2 for i in range(size) for j in range(size)
    )
    # gamma == 0 is a sample that already is a multiple of the identity, read as
    # under constant correlation: taking the target costs nothing.
    unclamped = 1.0 if gamma == 0.0 else pi / gamma / count
    intensity = min(1.0, max(0.0, unclamped))
    rescaled = scaled_identity_target(unbiased)
    matrix = blend(unbiased, rescaled, intensity)
    return Shrunk(
        matrix=matrix,
        intensity=intensity,
        sample=unbiased,
        target=rescaled,
        average_correlation=average_correlation(sample_ml),
        unclamped_intensity=unclamped,
        diagnostics=diagnose(matrix, observations=count),
        target_kind=ShrinkageTarget.SCALED_IDENTITY,
    )


def _rescale_target(target_ml: Matrix, unbiased: Matrix) -> Matrix:
    """Express the target against the unbiased sample's variances.

    The intensity is derived under the maximum likelihood convention, but the
    matrix returned to a caller should be the unbiased one — that is what every
    other estimator in this library and everywhere else means by "the sample
    covariance". The two differ by the constant factor ``T/(T-1)``, so the
    target is rescaled by the same factor and the blend stays consistent: the
    result has exactly the unbiased variances on its diagonal for every
    intensity, which is the identity the tests assert.
    """
    size = len(target_ml)
    out = [[0.0] * size for _ in range(size)]
    for i in range(size):
        out[i][i] = unbiased[i][i]
        for j in range(i + 1, size):
            if target_ml[i][i] <= 0.0 or target_ml[j][j] <= 0.0:
                continue
            # Rescale through the correlation, which is convention-free.
            rho_ij = target_ml[i][j] / math.sqrt(target_ml[i][i] * target_ml[j][j])
            value = rho_ij * math.sqrt(unbiased[i][i] * unbiased[j][j])
            out[i][j] = value
            out[j][i] = value
    return out


def portfolio_variance(weights: Vector, covariance: Matrix) -> float:
    """``w^T S w``. Re-exported here so callers need not reach into linalg."""
    from .linalg import quadratic_form

    return quadratic_form(weights, covariance)


__all__ = [
    "Diagnostics",
    "ShrinkageTarget",
    "Shrunk",
    "average_correlation",
    "blend",
    "constant_correlation_target",
    "correlation",
    "diagnose",
    "ledoit_wolf",
    "portfolio_variance",
    "sample_covariance",
    "scaled_identity_target",
]

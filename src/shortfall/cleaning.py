"""Which directions of a covariance estimate are noise, and what to do with them.

:func:`sample_covariance` is unbiased and, at the dimensions a book actually
has, close to useless. Sixty assets and a hundred and twenty observations is
two hundred and forty days of data against 1,830 parameters, and a
minimum-variance portfolio built on the sample estimate realises about twice the
variance of one built on the truth. :func:`ledoit_wolf` repairs that by pulling
the whole matrix towards a target, which is one scalar decision applied to every
direction at once.

This module asks the other question: *which* directions carry information.

**The noise band is not a heuristic.** For ``T`` observations of ``N``
independent unit-variance series, the eigenvalues of the sample correlation
matrix do not cluster near one — they spread, and in the limit with
``q = N / T`` held fixed they fill ``[(1 - sqrt(q))**2, (1 + sqrt(q))**2]`` with
the Marchenko-Pastur density

    rho(x) = sqrt((upper - x) (x - lower)) / (2 pi q x).

At ``q = 0.5`` that band runs from 0.086 to 2.914. So a sample eigenvalue of 2.5
in a sixty-by-hundred-and-twenty estimate is not a factor explaining two and a
half times its share of the variance; it is what pure noise produces, routinely.
The density is checked here against quadrature — it integrates to one and has
mean one — rather than asserted.

**Cleaning.** :func:`clip_spectrum` works on the *correlation* matrix, which is
where the law applies with no scale to fit, replaces every eigenvalue at or
below the upper edge with the average of that group, and leaves the ones above
it alone. The average rather than zero, so the trace is preserved exactly: the
total variance was measured well even though its distribution across directions
was not. The diagonal is restored to one afterwards, because rebuilding from a
modified spectrum perturbs it, and the standard deviations are put back at the
end.

**A factor can be too small to see at all.** This is the part worth the phase.
Put one factor of strength ``s`` into the truth, so its true eigenvalue is
``1 + s``. If ``s > sqrt(q)`` the largest sample eigenvalue converges to
``(1 + s) (1 + q / s)`` — above the band, detectable, and *biased upward*: at
``q = 0.25`` and ``s = 0.8`` the truth is 1.8 and the sample reads 2.36, which
is 31% high. If ``s < sqrt(q)`` the largest sample eigenvalue converges to the
band edge **regardless of s**. Measured at ``q = 0.25``, where the threshold is
0.5: a factor of strength 0.4 — a direction with 40% more variance than every
other — produces a top eigenvalue of 2.21 against an edge of 2.25, and an
eigenvector whose squared overlap with the truth is 0.11 at sixty assets and
0.05 at two hundred, falling towards zero. The factor is not weakly estimated.
It is absent from the estimate, and no amount of cleaning recovers it.

:func:`detection_threshold`, :func:`spike_location`, :func:`implied_spike` and
:func:`eigenvector_overlap` are the four statements of that, and the last two
are what let a caller turn a sample eigenvalue back into a claim about the
world instead of reading it as one.

**Whether cleaning beats shrinking depends on the truth, and two of the three
obvious tests do not test it.** Forty assets, eighty training observations, the
true variance of the resulting minimum-variance portfolio in units of 1e-4:

| truth | sample | shrunk | clipped | equal weight | truth |
|---|---|---|---|---|---|
| isotropic | 0.0506 | 0.0257 | 0.0257 | 0.0250 | 0.0250 |
| one common factor, equal loadings | 1.3516 | 0.7615 | 0.8027 | 0.6650 | 0.6650 |
| three factors, unequal loadings | 0.0561 | 0.0358 | 0.0299 | 0.0506 | 0.0274 |

The first two rows are rigged and the fourth column says so: when the truth is
exchangeable across assets, the equal-weight portfolio is *exactly* optimal, so
nothing is being estimated and the shrinkage target happens to be the truth.
Shrinkage winning the second row by 5% is a statement about the test, not about
the method. Only the third row has a truth that distinguishes assets, and there
cleaning closes 91% of the gap between the sample estimate and the truth against
shrinkage's 71%, while equal weighting closes 20%.

So neither dominates, :func:`compare_estimators` exists instead of a
recommendation, and the lesson carried out of the phase is about the
benchmark: two of the three panels a careful person would reach for first
cannot tell these estimators apart, because their truth makes the trivial
portfolio optimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .covariance import correlation, ledoit_wolf, sample_covariance
from .linalg import (
    Matrix,
    NotPositiveDefinite,
    Vector,
    check_symmetric,
    cholesky,
    eigh,
    quadratic_form,
    reconstruct,
)
from .series import Panel, ReturnSeries, TooShort

__all__ = [
    "BadSpectrum",
    "Cleaned",
    "OutOfSample",
    "clip_spectrum",
    "compare_estimators",
    "detection_threshold",
    "eigenvector_overlap",
    "implied_spike",
    "marchenko_pastur_density",
    "marchenko_pastur_edges",
    "spike_location",
]


class BadSpectrum(ValueError):
    """The dimensions do not admit the law this module rests on."""


def _ratio(assets: int, observations: int) -> float:
    if assets < 1:
        raise BadSpectrum(f"a covariance needs at least one asset, got {assets}")
    if observations < 2:
        raise BadSpectrum(
            f"the law is about a sample covariance, which needs at least two "
            f"observations; got {observations}"
        )
    return assets / observations


def marchenko_pastur_edges(assets: int, observations: int) -> tuple[float, float]:
    """The noise band for the eigenvalues of a correlation matrix.

    ``((1 - sqrt(q))**2, (1 + sqrt(q))**2)`` with ``q = assets / observations``.
    These are limits, so a finite sample puts a few eigenvalues just outside
    them; what they are not is a rule of thumb — at ``q = 0.5`` the band is
    ``[0.086, 2.914]``, so an eigenvalue of two and a half in a sixty-asset
    estimate from a hundred and twenty observations is what noise does.

    Above ``q = 1`` there are more assets than observations, the sample matrix is
    singular, and the lower edge is the location of the bulk rather than of the
    ``N - T`` exact zeros. That is reported rather than refused, because the
    upper edge — the one cleaning uses — is unaffected.
    """
    ratio = _ratio(assets, observations)
    root = math.sqrt(ratio)
    return (1.0 - root) ** 2, (1.0 + root) ** 2


def marchenko_pastur_density(level: float, assets: int, observations: int) -> float:
    """The limiting density of the sample eigenvalues at ``level``.

    Zero outside the band. Integrates to one over the band for ``q <= 1``, and
    to ``1 / q`` for ``q > 1`` with the rest of the mass at zero — both checked
    against quadrature in the tests rather than taken on trust.
    """
    ratio = _ratio(assets, observations)
    lower, upper = marchenko_pastur_edges(assets, observations)
    if level <= lower or level >= upper or level <= 0.0:
        return 0.0
    return math.sqrt((upper - level) * (level - lower)) / (2.0 * math.pi * ratio * level)


def detection_threshold(assets: int, observations: int) -> float:
    """The smallest factor strength that produces a detectable eigenvalue.

    ``sqrt(q)``. A factor of strength ``s`` makes the truth's largest eigenvalue
    ``1 + s``; below this threshold the *sample's* largest eigenvalue sits at the
    band edge whatever ``s`` is, so the factor leaves no trace in the spectrum at
    all. At sixty assets and two hundred and forty observations the threshold is
    0.5 — a direction carrying half again the variance of every other is the
    smallest one this much data can see.
    """
    return math.sqrt(_ratio(assets, observations))


def spike_location(strength: float, assets: int, observations: int) -> float:
    """Where a factor of strength ``strength`` shows up in the sample spectrum.

    ``(1 + s) (1 + q / s)`` above the threshold, and the band edge at or below
    it. The gap between this and the truth's ``1 + s`` is the upward bias, and
    it is not small: 31% at ``q = 0.25`` and ``s = 0.8``.
    """
    ratio = _ratio(assets, observations)
    if not strength > 0.0:
        raise BadSpectrum(
            f"a factor's strength is the variance it adds and must be positive, "
            f"got {strength!r}"
        )
    if strength <= math.sqrt(ratio):
        return (1.0 + math.sqrt(ratio)) ** 2
    return (1.0 + strength) * (1.0 + ratio / strength)


def implied_spike(observed: float, assets: int, observations: int) -> float | None:
    """Invert :func:`spike_location`: the factor strength an eigenvalue implies.

    ``None`` for an eigenvalue at or inside the band, because there is no answer
    rather than a small one — every strength below the threshold produces the
    same observation, so the observation determines nothing.

    Above the band the inverse is a quadratic with one admissible root:
    ``s**2 - (observed - 1 - q) s + q = 0``, taking the larger root, which is the
    branch on which the map is increasing. Composing the two directions returns
    the input to rounding, which is the test.
    """
    ratio = _ratio(assets, observations)
    _, upper = marchenko_pastur_edges(assets, observations)
    if observed <= upper:
        return None
    middle = observed - 1.0 - ratio
    discriminant = middle * middle - 4.0 * ratio
    if discriminant <= 0.0:  # pragma: no cover - excluded by observed > upper
        return None
    return 0.5 * (middle + math.sqrt(discriminant))


def eigenvector_overlap(strength: float, assets: int, observations: int) -> float:
    """The limiting squared overlap of the sample eigenvector with the true one.

    ``(1 - q / s**2) / (1 + q / s)`` above the threshold and zero at or below
    it. The zero is the statement that matters: an undetectable factor's
    direction is not estimated poorly, it is not estimated. Measured at
    ``q = 0.25`` and ``s = 3.0`` the limit is 0.897 against 0.897 simulated; at
    ``s = 0.4``, below the threshold, the limit is zero and the simulation gives
    0.11 at sixty assets and 0.05 at two hundred, falling as it should.
    """
    ratio = _ratio(assets, observations)
    if not strength > 0.0:
        raise BadSpectrum(
            f"a factor's strength is the variance it adds and must be positive, "
            f"got {strength!r}"
        )
    if strength <= math.sqrt(ratio):
        return 0.0
    return (1.0 - ratio / (strength * strength)) / (1.0 + ratio / strength)


@dataclass(frozen=True)
class Cleaned:
    """A covariance with its noise directions flattened, and what was flattened.

    ``kept`` is how many eigenvalues were above the band. It is the number of
    directions the data can actually distinguish, and when it is zero the
    cleaned correlation matrix is the identity — which is the honest answer, not
    a failure.
    """

    covariance: Matrix
    #: The eigenvalues of the sample *correlation* matrix, descending.
    observed: Vector
    #: The same after flattening the noise band, descending.
    cleaned: Vector
    upper_edge: float
    kept: int
    #: The common level the flattened eigenvalues were set to.
    noise_level: float
    observations: int

    @property
    def assets(self) -> int:
        return len(self.observed)

    @property
    def noise_fraction(self) -> float:
        """The share of the correlation matrix's trace inside the noise band."""
        return sum(self.cleaned[self.kept :]) / sum(self.cleaned)

    @property
    def implied_strengths(self) -> tuple[float, ...]:
        """What each detected eigenvalue says the factor's strength is."""
        return tuple(
            value
            for value in (
                implied_spike(level, self.assets, self.observations)
                for level in self.observed[: self.kept]
            )
            if value is not None
        )


def clip_spectrum(panel: Panel, *, ddof: int = 1) -> Cleaned:
    """Flatten the eigenvalues of ``panel``'s correlation matrix inside the band.

    The eigenvalues at or below the upper Marchenko-Pastur edge are replaced by
    their own average, which keeps the trace exactly: the total variance is
    estimated well even where its distribution across directions is not. The
    diagonal is restored to one after rebuilding, because a matrix rebuilt from a
    modified spectrum does not have unit diagonal, and the sample standard
    deviations are put back at the end.

    Working on the correlation matrix rather than the covariance is not a
    convenience. The law is stated for unit-variance series, so applying it to a
    covariance means fitting a scale to the bulk; the correlation matrix has that
    scale fixed at one by construction.
    """
    covariance = sample_covariance(panel, ddof=ddof)
    assets = panel.assets
    observations = panel.observations
    if observations < 2:  # pragma: no cover - sample_covariance refuses first
        raise TooShort("a correlation matrix needs at least two observations")
    deviations = [math.sqrt(covariance[i][i]) for i in range(assets)]
    matrix = correlation(covariance)
    check_symmetric(matrix)
    values, vectors = eigh(matrix)
    # ``eigh`` returns ascending values with the vectors as rows, so the
    # reordering is a reordering of rows and not a transpose.
    order = sorted(range(assets), key=lambda i: -values[i])
    descending = [values[i] for i in order]
    basis = [vectors[i] for i in order]
    _, upper = marchenko_pastur_edges(assets, observations)
    kept = sum(1 for value in descending if value > upper)
    tail = descending[kept:]
    level = sum(tail) / len(tail) if tail else 0.0
    flattened = descending[:kept] + [level] * len(tail)
    rebuilt = reconstruct(flattened, basis)
    # Rebuilding from a modified spectrum loses the unit diagonal, and a
    # correlation matrix with a diagonal of 0.98 is not a correlation matrix.
    # Rescaling by the square roots of the diagonal restores it and keeps the
    # matrix positive semidefinite, which setting the diagonal to one would not.
    scale = [math.sqrt(rebuilt[i][i]) for i in range(assets)]
    cleaned_covariance = [
        [
            rebuilt[i][j] / (scale[i] * scale[j]) * deviations[i] * deviations[j]
            for j in range(assets)
        ]
        for i in range(assets)
    ]
    return Cleaned(
        covariance=cleaned_covariance,
        observed=descending,
        cleaned=flattened,
        upper_edge=upper,
        kept=kept,
        noise_level=level,
        observations=observations,
    )


def _minimum_variance(matrix: Matrix) -> Vector | None:
    """Weights solving ``matrix w = 1``, normalised to sum to one.

    ``None`` when the matrix is not positive definite, which is what a sample
    covariance from fewer observations than assets is. Returning ``None`` rather
    than raising, because "this estimator has no answer here" is one of the
    comparison's results and not an error in the caller's request.
    """
    size = len(matrix)
    try:
        lower = cholesky(matrix)
    except NotPositiveDefinite:
        return None
    forward = [0.0] * size
    for i in range(size):
        forward[i] = (
            1.0 - sum(lower[i][k] * forward[k] for k in range(i))
        ) / lower[i][i]
    weights = [0.0] * size
    for i in range(size - 1, -1, -1):
        weights[i] = (
            forward[i] - sum(lower[k][i] * weights[k] for k in range(i + 1, size))
        ) / lower[i][i]
    total = sum(weights)
    if total == 0.0:  # pragma: no cover - needs an exactly antisymmetric solve
        return None
    return [value / total for value in weights]


@dataclass(frozen=True)
class OutOfSample:
    """What one covariance estimator's minimum-variance portfolio realised."""

    estimator: str
    #: Realised variance over the held-out half, per period. ``None`` when the
    #: estimator produced no invertible matrix on the training half.
    variance: float | None
    weights: Vector
    #: Sum of squared weights. One over this is the effective number of
    #: positions, and it is reported because the sample estimator's failure mode
    #: is a portfolio of enormous offsetting bets rather than a wrong number.
    concentration: float


def compare_estimators(
    panel: Panel, *, split: float = 0.5, ddof: int = 1
) -> tuple[OutOfSample, ...]:
    """Build a minimum-variance portfolio four ways and measure each out of sample.

    The first ``split`` of the panel estimates the covariance; the rest measures
    what the resulting portfolio actually realised. No simulation and no
    assumption about the truth — which is the only way to compare estimators
    whose ordering depends on the truth, as these do.

    Four ways, because the fourth is the one that catches a rigged comparison:
    the equal-weight portfolio is exactly optimal whenever the truth treats the
    assets alike, and on such a panel no covariance estimator can show anything.
    On a truth with unequal loadings cleaning closes 91% of the gap between the
    sample estimate and the truth against shrinkage's 71%; on an isotropic one
    they tie and equal weighting beats both. Neither dominates, so this returns
    a table rather than a recommendation.
    """
    if not 0.0 < split < 1.0:
        raise BadSpectrum(f"the split must be strictly between zero and one, got {split!r}")
    total = panel.observations
    cut = int(total * split)
    if cut < 2 or total - cut < 2:
        raise TooShort(
            f"splitting {total} observations at {split!r} leaves {cut} and "
            f"{total - cut}; both halves need at least two"
        )
    train = Panel(
        tuple(
            ReturnSeries(one.name, one.values[:cut], one.convention)
            for one in panel.series
        )
    )
    test = Panel(
        tuple(
            ReturnSeries(one.name, one.values[cut:], one.convention)
            for one in panel.series
        )
    )
    realised = sample_covariance(test, ddof=ddof)
    size = panel.assets
    candidates: list[tuple[str, Matrix]] = [
        ("sample", sample_covariance(train, ddof=ddof)),
        ("shrunk", ledoit_wolf(train).matrix),
        ("clipped", clip_spectrum(train, ddof=ddof).covariance),
    ]
    out: list[OutOfSample] = []
    for name, matrix in candidates:
        weights = _minimum_variance(matrix)
        if weights is None:
            out.append(OutOfSample(name, None, [], 0.0))
            continue
        out.append(
            OutOfSample(
                name,
                quadratic_form(weights, realised),
                weights,
                sum(value * value for value in weights),
            )
        )
    equal = [1.0 / size] * size
    out.append(
        OutOfSample(
            "equal weight",
            quadratic_form(equal, realised),
            equal,
            sum(value * value for value in equal),
        )
    )
    return tuple(out)

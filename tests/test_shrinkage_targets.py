"""Ledoit-Wolf shrinkage towards the scaled identity, and the choice between targets.

The identity estimator is checked against scikit-learn's ``ledoit_wolf``, which
is a separate implementation of the same paper written against NumPy. The
reference figures below were produced by scikit-learn 1.9.1 from exactly these
panels; they are a transcript rather than a dependency, so the suite needs no
NumPy to run. Everything else is checked against identities the estimator has
to satisfy whatever the data.
"""

from __future__ import annotations

import random

import pytest

from shortfall import (
    Panel,
    ShrinkageTarget,
    Shrunk,
    eigenvalues,
    ledoit_wolf,
    scaled_identity_target,
)


def factor_panel(
    assets: int,
    periods: int,
    seed: int,
    *,
    loading: float = 0.6,
    scales: list[float] | None = None,
) -> Panel:
    """One common factor plus noise, from the standard library's generator."""
    generator = random.Random(seed)
    size = scales or [1.0] * assets
    market = [generator.gauss(0.0, 0.01) for _ in range(periods)]
    columns = [
        [size[i] * (loading * market[t] + generator.gauss(0.0, 0.01)) for t in range(periods)]
        for i in range(assets)
    ]
    return Panel.from_columns({f"a{i}": column for i, column in enumerate(columns)})


#: ``(assets, periods, seed, loading, scales)``, then scikit-learn's intensity
#: and three entries of its matrix: ``[0][0]``, ``[0][1]`` and the last diagonal.
#: scikit-learn returns the blend on the maximum-likelihood covariance, so its
#: entries are this library's times ``(T - 1) / T``.
SCIKIT_LEARN: tuple[
    tuple[tuple[int, int, int, float, list[float] | None], float, tuple[float, float, float]],
    ...,
] = (
    (
        (4, 60, 1, 0.6, None),
        0.3239713264046774,
        (0.00013817555750588256, 2.636486724503642e-05, 0.00012252086852565383),
    ),
    (
        (6, 25, 2, 0.3, None),
        0.734193272759233,
        (0.00012519638599601846, 2.264506967339182e-07, 0.00011847812207855596),
    ),
    # Fewer observations than assets: the sample is singular and the estimate is not.
    (
        (10, 8, 3, 0.8, None),
        0.6773141437488129,
        (0.00012885190285383055, 1.130545938095132e-05, 0.0002127946664524756),
    ),
    (
        (5, 400, 4, 0.0, [1.0, 2.0, 0.5, 3.0, 1.5]),
        0.020264868958114003,
        (0.00010888490021165718, 8.977689349502454e-06, 0.00020721207835253292),
    ),
)


@pytest.mark.parametrize(("case", "intensity", "entries"), SCIKIT_LEARN)
def test_the_identity_target_agrees_with_scikit_learn(
    case: tuple[int, int, int, float, list[float] | None],
    intensity: float,
    entries: tuple[float, float, float],
) -> None:
    """Measured agreement is within 1.1e-15 on the intensity and 8.1e-15 on the
    matrix; the bounds are 1e-12, far below anything a wrong formula produces."""
    assets, periods, seed, loading, scales = case
    shrunk = ledoit_wolf(
        factor_panel(assets, periods, seed, loading=loading, scales=scales),
        target=ShrinkageTarget.SCALED_IDENTITY,
    )
    assert shrunk.intensity == pytest.approx(intensity, rel=1e-12, abs=0.0)
    ratio = (periods - 1) / periods
    ours = (shrunk.matrix[0][0], shrunk.matrix[0][1], shrunk.matrix[-1][-1])
    for value, expected in zip(ours, entries, strict=True):
        assert value * ratio == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert not shrunk.clamped


def test_the_default_target_is_unchanged() -> None:
    """Constant correlation by default, and to the bit, by enum and by string."""
    panel = factor_panel(6, 80, 11)
    plain = ledoit_wolf(panel)
    for spelling in (ShrinkageTarget.CONSTANT_CORRELATION, "constant_correlation"):
        chosen = ledoit_wolf(panel, target=spelling)
        assert chosen == plain
    assert plain.target_kind is ShrinkageTarget.CONSTANT_CORRELATION


def test_the_identity_target_is_the_average_variance_and_not_the_diagonal() -> None:
    """The trap in reusing the constant-correlation rescaling.

    That one rebuilds the target through its correlations onto the sample's own
    variances. Applied to the identity it keeps the zero correlations and puts
    back every asset's own variance, which is the diagonal target — a different
    estimator — and on assets of different volatility the difference is plain.
    """
    panel = factor_panel(5, 120, 12, scales=[1.0, 2.0, 0.5, 3.0, 1.5])
    shrunk = ledoit_wolf(panel, target="scaled_identity")
    size = len(shrunk.sample)
    average = sum(shrunk.sample[i][i] for i in range(size)) / size
    assert shrunk.target_kind is ShrinkageTarget.SCALED_IDENTITY
    for i in range(size):
        for j in range(size):
            expected = average if i == j else 0.0
            assert shrunk.target[i][j] == pytest.approx(expected, rel=1e-14, abs=0.0)
    assert shrunk.target == scaled_identity_target(shrunk.sample)
    variances = [shrunk.sample[i][i] for i in range(size)]
    assert max(variances) > 10.0 * min(variances)


def test_the_blend_keeps_the_trace_and_lifts_the_smallest_eigenvalue() -> None:
    """The two properties that make the 2004 estimator well conditioned.

    The trace is the sum of the eigenvalues and the target shares it, so the blend
    has the sample's trace at any intensity. And since the sample is positive
    semidefinite, the blend's smallest eigenvalue is at least the intensity times
    the average variance — strictly positive even with fewer observations than
    assets, where the sample's own smallest eigenvalue is zero.
    """
    shrunk = ledoit_wolf(factor_panel(10, 8, 3, loading=0.8), target="scaled_identity")
    size = len(shrunk.sample)
    trace = sum(shrunk.sample[i][i] for i in range(size))
    blended = sum(shrunk.matrix[i][i] for i in range(size))
    assert blended == pytest.approx(trace, rel=1e-14, abs=0.0)
    floor = shrunk.intensity * trace / size
    assert min(eigenvalues(shrunk.matrix)) >= floor * (1.0 - 1e-9)
    assert min(eigenvalues(shrunk.sample)) < floor * 1e-6
    assert not shrunk.diagnostics.numerically_singular


def test_a_sample_that_is_already_the_target_takes_it() -> None:
    """Uncorrelated columns of one variance: the sample is a multiple of the
    identity, the distance to the target is zero, and the intensity is one."""
    panel = Panel.from_columns(
        {"a": [0.01, -0.01, 0.01, -0.01], "b": [0.01, 0.01, -0.01, -0.01]}
    )
    shrunk = ledoit_wolf(panel, target="scaled_identity")
    assert shrunk.intensity == 1.0
    assert shrunk.matrix == shrunk.sample


def test_an_unknown_target_is_refused() -> None:
    panel = factor_panel(3, 40, 13)
    for spelling in ("identity", "single_factor", "diagonal"):
        with pytest.raises(ValueError, match="not a valid ShrinkageTarget"):
            ledoit_wolf(panel, target=spelling)


# -- which target, measured ---------------------------------------------------


def _frobenius_losses(
    loading: float, scales: list[float], seeds: range
) -> tuple[list[float], list[float]]:
    """Squared Frobenius distance to the true matrix, per seed, for each target."""
    assets, periods, sigma = len(scales), 40, 0.01
    truth = [
        [
            scales[i] * scales[j] * sigma**2 * (loading**2 + (1.0 if i == j else 0.0))
            for j in range(assets)
        ]
        for i in range(assets)
    ]

    def loss(estimate: Shrunk) -> float:
        return sum(
            (estimate.matrix[i][j] - truth[i][j]) ** 2
            for i in range(assets)
            for j in range(assets)
        )

    correlated, identity = [], []
    for seed in seeds:
        panel = factor_panel(assets, periods, 1000 + seed, loading=loading, scales=scales)
        correlated.append(loss(ledoit_wolf(panel)))
        identity.append(loss(ledoit_wolf(panel, target="scaled_identity")))
    return correlated, identity


EQUAL = [1.0] * 10
UNEQUAL = [0.5, 1.0, 2.0, 3.0, 0.7, 1.5, 2.5, 0.4, 1.2, 4.0]


@pytest.mark.parametrize(
    ("loading", "scales", "wins", "ratio"),
    [
        # Uncorrelated and equally volatile: the identity is the truth's shape.
        (0.0, EQUAL, (19, 20), (0.08, 0.25)),
        # Uncorrelated with differing volatilities: pulling them together hurts.
        (0.0, UNEQUAL, (0, 3), (1.6, 3.2)),
        # A common factor: constant correlation is the truth's shape.
        (0.6, EQUAL, (0, 3), (1.6, 3.2)),
    ],
)
def test_which_target_wins_is_a_question_about_the_data(
    loading: float,
    scales: list[float],
    wins: tuple[int, int],
    ratio: tuple[float, float],
) -> None:
    """Ten assets, forty observations, twenty seeds.

    Measured: the identity's total loss is 0.139 times constant correlation's when
    the assets are uncorrelated and equally volatile, lower on all twenty seeds;
    2.35 times when they are uncorrelated and their volatilities differ, lower on
    one; and 2.35 times under a common factor, lower on none. Quoted as bands,
    since every figure is a Monte Carlo — over forty seeds the factor case is
    1.77, which is how far one of these ratios moves with the sample.
    """
    correlated, identity = _frobenius_losses(loading, scales, range(20))
    won = sum(a < b for a, b in zip(identity, correlated, strict=True))
    assert wins[0] <= won <= wins[1]
    measured = sum(identity) / sum(correlated)
    assert ratio[0] < measured < ratio[1]

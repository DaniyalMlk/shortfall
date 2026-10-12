"""The noise band of a covariance estimate, and what can and cannot be read off it.

**The law is checked against quadrature, not asserted.** The Marchenko-Pastur
density has to integrate to one over its band and have mean one, since it is the
limiting distribution of eigenvalues of a correlation matrix whose trace is the
number of assets. Simpson agrees to eight figures at three different ratios.
That is the whole of the law's arithmetic, and nothing below it is taken on
trust.

**The spike map is checked by composing it with its inverse, and by simulation.**
``spike_location`` and ``implied_spike`` return each other's input to ten
figures; separately, a simulated panel with a planted factor puts its largest
eigenvalue where the formula says, and the cleaned estimate detects exactly one
direction.

**The detection threshold is tested as a threshold.** Below ``sqrt(q)`` the
formula returns the band edge for *every* strength, which is the statement that
a weak factor leaves no trace in the spectrum at all — not a faint one. The
simulated version of the same thing is the striking number: at ``q = 0.5``, a
factor carrying 40% more variance than every other direction produces a largest
eigenvalue inside the band, indistinguishable from the one a panel with no
factor at all produces.

**The structural properties of the cleaned matrix are checked exactly.** The
trace of the cleaned spectrum equals the trace of the observed one to rounding,
because the flattened eigenvalues are set to their own average; the diagonal of
the cleaned covariance equals the sample diagonal to rounding, because the
rebuild is rescaled rather than overwritten; and the result is positive
semidefinite, which overwriting the diagonal would not guarantee.

**And the comparison is read sceptically.** Two of the three panels a careful
person would reach for cannot distinguish these estimators, because their truth
treats all assets alike and therefore makes the equal-weight portfolio exactly
optimal. The fourth column of the comparison is what says so, and the test for
it asserts that equal weighting *wins* on those panels — the opposite of what a
test written to show cleaning working would assert.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall import (
    BadDimensions,
    Cleaned,
    Panel,
    ReturnSeries,
    TooShort,
    clip_spectrum,
    compare_estimators,
    detection_threshold,
    eigenvector_overlap,
    implied_spike,
    ledoit_wolf,
    marchenko_pastur_density,
    marchenko_pastur_edges,
    sample_covariance,
    spike_location,
)
from shortfall.cleaning import _minimum_variance
from shortfall.linalg import Matrix, Vector, is_positive_semidefinite, quadratic_form

RATIOS = ((60, 120), (30, 300), (100, 200), (40, 160))


# -- the law -------------------------------------------------------------------


@pytest.mark.parametrize(("assets", "observations"), RATIOS)
def test_the_density_integrates_to_one(assets: int, observations: int) -> None:
    """Composite Simpson over the band. The density is a limit and this is the
    only check on it that does not reuse its own algebra."""
    lower, upper = marchenko_pastur_edges(assets, observations)
    panels = 20_000
    step = (upper - lower) / panels
    mass = 0.0
    for index in range(panels + 1):
        level = lower + index * step
        weight = 1.0 if index in (0, panels) else (4.0 if index % 2 else 2.0)
        mass += weight * marchenko_pastur_density(level, assets, observations)
    assert mass * step / 3.0 == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(("assets", "observations"), RATIOS)
def test_the_density_has_mean_one(assets: int, observations: int) -> None:
    """It must: the trace of a correlation matrix is the number of assets, so
    the average eigenvalue is one however the sampling spread them out."""
    lower, upper = marchenko_pastur_edges(assets, observations)
    panels = 20_000
    step = (upper - lower) / panels
    mean = 0.0
    for index in range(panels + 1):
        level = lower + index * step
        weight = 1.0 if index in (0, panels) else (4.0 if index % 2 else 2.0)
        mean += weight * level * marchenko_pastur_density(level, assets, observations)
    assert mean * step / 3.0 == pytest.approx(1.0, abs=1e-6)


def test_the_edges_are_where_the_density_vanishes() -> None:
    lower, upper = marchenko_pastur_edges(60, 120)
    assert lower == pytest.approx((1.0 - math.sqrt(0.5)) ** 2, rel=1e-15)
    assert upper == pytest.approx((1.0 + math.sqrt(0.5)) ** 2, rel=1e-15)
    assert marchenko_pastur_density(lower, 60, 120) == 0.0
    assert marchenko_pastur_density(upper, 60, 120) == 0.0
    assert marchenko_pastur_density(lower * 1.0001, 60, 120) > 0.0
    assert marchenko_pastur_density(upper * 0.9999, 60, 120) > 0.0
    assert marchenko_pastur_density(-1.0, 60, 120) == 0.0
    assert marchenko_pastur_density(upper + 1.0, 60, 120) == 0.0


def test_the_band_widens_as_the_sample_shortens() -> None:
    widths = [
        marchenko_pastur_edges(60, observations)[1]
        - marchenko_pastur_edges(60, observations)[0]
        for observations in (6000, 600, 240, 120)
    ]
    assert widths == sorted(widths)
    # Half again the long-run variance in a direction is inside the band at
    # q = 0.5, which is the figure that makes the phase worth having.
    assert marchenko_pastur_edges(60, 120)[1] > 2.9


# -- the spike -----------------------------------------------------------------


@pytest.mark.parametrize("strength", [0.6, 0.8, 1.5, 3.0, 10.0])
def test_the_spike_map_and_its_inverse_compose_to_the_identity(
    strength: float,
) -> None:
    located = spike_location(strength, 100, 400)
    assert implied_spike(located, 100, 400) == pytest.approx(strength, rel=1e-10)


def test_a_detectable_spike_sits_above_the_band_and_above_the_truth() -> None:
    """The gap between the sample eigenvalue and the truth is the upward bias,
    and at q = 0.25 with a strength of 0.8 it is 31%: the truth is 1.8 and the
    sample reads 2.3625."""
    _, upper = marchenko_pastur_edges(100, 400)
    located = spike_location(0.8, 100, 400)
    assert located > upper
    assert located == pytest.approx(2.3625, rel=1e-12)
    assert located / 1.8 == pytest.approx(1.3125, rel=1e-12)


@pytest.mark.parametrize("strength", [0.01, 0.1, 0.3, 0.49, 0.5])
def test_below_the_threshold_every_strength_gives_the_same_eigenvalue(
    strength: float,
) -> None:
    """A threshold, not a soft transition. The factor is not weakly represented
    in the spectrum; it is absent, and the observation determines nothing about
    it — which is why ``implied_spike`` returns None rather than a small number."""
    _, upper = marchenko_pastur_edges(100, 400)
    assert detection_threshold(100, 400) == pytest.approx(0.5, rel=1e-15)
    assert spike_location(strength, 100, 400) == pytest.approx(upper, rel=1e-15)
    assert eigenvector_overlap(strength, 100, 400) == 0.0
    assert implied_spike(upper, 100, 400) is None
    assert implied_spike(upper * 0.9, 100, 400) is None


def test_the_overlap_rises_with_the_strength_and_never_reaches_one() -> None:
    overlaps = [eigenvector_overlap(s, 100, 400) for s in (0.6, 1.0, 3.0, 10.0, 100.0)]
    assert overlaps == sorted(overlaps)
    assert all(0.0 < value < 1.0 for value in overlaps)
    assert overlaps[2] == pytest.approx(0.8974358974, rel=1e-9)


def test_the_threshold_falls_as_the_sample_lengthens() -> None:
    thresholds = [detection_threshold(60, t) for t in (120, 600, 6000)]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[0] == pytest.approx(math.sqrt(0.5), rel=1e-15)


@pytest.mark.parametrize(
    ("assets", "observations", "message"),
    [(0, 100, "at least one asset"), (10, 1, "at least two"), (10, 0, "at least two")],
)
def test_degenerate_dimensions_are_refused(
    assets: int, observations: int, message: str
) -> None:
    with pytest.raises(BadDimensions, match=message):
        marchenko_pastur_edges(assets, observations)


@pytest.mark.parametrize("strength", [0.0, -1.0])
def test_a_non_positive_strength_is_not_a_factor(strength: float) -> None:
    with pytest.raises(BadDimensions, match="must be positive"):
        spike_location(strength, 60, 240)
    with pytest.raises(BadDimensions, match="must be positive"):
        eigenvector_overlap(strength, 60, 240)


# -- panels --------------------------------------------------------------------


def simulated(
    assets: int,
    observations: int,
    *,
    loadings: list[list[float]] | None = None,
    seed: int = 0,
) -> tuple[Panel, Matrix]:
    """A panel and the covariance it was actually drawn from.

    ``loadings[i][k]`` is asset ``i``'s exposure to factor ``k``; with none the
    truth is a scaled identity. Returning the truth is the point — an estimator
    comparison that only has the data cannot say which estimate is better.
    """
    generator = random.Random(seed)
    width = 0 if loadings is None else len(loadings[0])
    factors = [
        [generator.gauss(0.0, 1.0) for _ in range(observations)] for _ in range(width)
    ]
    rows = []
    for index in range(assets):
        row = []
        for time in range(observations):
            common = (
                0.0
                if loadings is None
                else sum(loadings[index][k] * factors[k][time] for k in range(width))
            )
            row.append(0.01 * (common + generator.gauss(0.0, 1.0)))
        rows.append(tuple(row))
    truth: Matrix = [
        [
            1e-4
            * (
                (0.0 if loadings is None else sum(
                    loadings[i][k] * loadings[j][k] for k in range(width)
                ))
                + (1.0 if i == j else 0.0)
            )
            for j in range(assets)
        ]
        for i in range(assets)
    ]
    panel = Panel(tuple(ReturnSeries(f"a{i}", rows[i]) for i in range(assets)))
    return panel, truth


def test_clipping_keeps_the_trace_and_the_diagonal_exactly() -> None:
    """Both are structural rather than approximate. The trace, because the
    flattened eigenvalues are set to their own average; the diagonal, because
    the rebuild is rescaled by the square roots of its own diagonal rather than
    having ones written into it."""
    panel, _ = simulated(30, 120, loadings=[[0.7] for _ in range(30)], seed=1)
    cleaned = clip_spectrum(panel)
    assert isinstance(cleaned, Cleaned)
    assert sum(cleaned.cleaned) == pytest.approx(sum(cleaned.observed), rel=1e-12)
    assert sum(cleaned.cleaned) == pytest.approx(30.0, rel=1e-12)
    sample = sample_covariance(panel)
    for index in range(30):
        assert cleaned.covariance[index][index] == pytest.approx(
            sample[index][index], rel=1e-12
        )
    assert is_positive_semidefinite(cleaned.covariance)


def test_clipping_a_planted_factor_keeps_exactly_one_direction() -> None:
    panel, _ = simulated(40, 160, loadings=[[0.8] for _ in range(40)], seed=2)
    cleaned = clip_spectrum(panel)
    assert cleaned.kept == 1
    assert cleaned.observed[0] > cleaned.upper_edge
    assert cleaned.observed[1] <= cleaned.upper_edge
    assert len(cleaned.implied_strengths) == 1
    # The cleaned spectrum is the kept eigenvalue and one repeated level.
    assert cleaned.cleaned[0] == cleaned.observed[0]
    assert len(set(cleaned.cleaned[1:])) == 1
    assert cleaned.cleaned[1] == pytest.approx(cleaned.noise_level, rel=1e-15)
    assert 0.0 < cleaned.noise_fraction < 1.0
    assert cleaned.assets == 40
    assert cleaned.observations == 160


def test_a_panel_with_no_factor_is_cleaned_to_a_diagonal_matrix() -> None:
    """Which is the honest answer and not a failure: with nothing above the band
    there are no directions the data can distinguish, and the cleaned
    correlation matrix is the identity."""
    panel, _ = simulated(30, 150, seed=3)
    cleaned = clip_spectrum(panel)
    assert cleaned.kept == 0
    assert cleaned.implied_strengths == ()
    assert cleaned.noise_fraction == pytest.approx(1.0, rel=1e-12)
    for i in range(30):
        for j in range(30):
            if i != j:
                assert abs(cleaned.covariance[i][j]) < 1e-12


def test_the_planted_factor_lands_where_the_formula_says() -> None:
    """A loading of 0.8 on every one of forty assets is a factor of strength
    ``40 * 0.64 / 1.64`` in correlation terms, and the sample's largest
    eigenvalue has to be near the spike location that implies. Checked loosely,
    because one panel is one draw — the point is that it is not near the band
    edge."""
    assets, observations = 40, 160
    panel, _ = simulated(
        assets, observations, loadings=[[0.8] for _ in range(assets)], seed=4
    )
    cleaned = clip_spectrum(panel)
    strength = assets * 0.64 / 1.64
    predicted = spike_location(strength, assets, observations)
    assert cleaned.observed[0] == pytest.approx(predicted, rel=0.1)
    assert cleaned.observed[0] > 5.0 * cleaned.upper_edge


def test_a_weak_factor_is_indistinguishable_from_none_at_all() -> None:
    """The phase's headline, simulated. At q = 0.5 the threshold is 0.707, so a
    factor adding 40% to one direction's variance is below it — and the largest
    eigenvalue of a panel carrying it is inside the band, where a panel with no
    factor at all also puts its largest."""
    assets, observations = 30, 60
    loading = math.sqrt(0.4 / assets)
    weak, _ = simulated(
        assets, observations, loadings=[[loading] for _ in range(assets)], seed=5
    )
    none, _ = simulated(assets, observations, seed=5)
    with_factor = clip_spectrum(weak)
    without = clip_spectrum(none)
    assert detection_threshold(assets, observations) > 0.4
    assert with_factor.kept == 0
    assert without.kept == 0
    assert with_factor.observed[0] < with_factor.upper_edge
    assert with_factor.observed[0] == pytest.approx(without.observed[0], rel=0.2)


# -- the comparison, read sceptically ------------------------------------------


def realised(weights: Vector, truth: Matrix) -> float:
    return quadratic_form(weights, truth)


def averaged(
    kind: str, *, seeds: int = 6, assets: int = 40, observations: int = 160
) -> dict[str, float]:
    """The true variance of each estimator's minimum-variance portfolio."""
    generator = random.Random(99)
    totals: dict[str, list[float]] = {}
    for seed in range(seeds):
        if kind == "isotropic":
            loadings = None
        elif kind == "one factor":
            loadings = [[0.8] for _ in range(assets)]
        else:
            loadings = [
                [generator.gauss(0.0, 0.5) for _ in range(3)] for _ in range(assets)
            ]
        panel, truth = simulated(assets, observations, loadings=loadings, seed=seed)
        for result in compare_estimators(panel):
            if result.variance is None:
                continue
            totals.setdefault(result.estimator, []).append(
                realised(result.weights, truth)
            )
        best = _minimum_variance(truth)
        assert best is not None
        totals.setdefault("truth", []).append(realised(best, truth))
    return {name: sum(values) / len(values) for name, values in totals.items()}


def test_every_estimator_beats_the_sample_covariance() -> None:
    """The one thing that is true on every panel. At q = 0.5 the sample estimate
    roughly doubles the variance the truth would have delivered."""
    for kind in ("isotropic", "three factors"):
        table = averaged(kind)
        assert table["sample"] > table["shrunk"]
        assert table["sample"] > table["clipped"]
        assert table["sample"] > table["truth"]
        assert table["sample"] / table["truth"] > 1.5


def test_on_an_exchangeable_truth_equal_weighting_wins_and_nothing_is_learned() -> None:
    """Written the way a sceptic would write it. If the truth treats all assets
    alike then the equal-weight portfolio is exactly optimal, every covariance
    estimate can only be noise against it, and a comparison on such a panel
    measures nothing about the estimators. Two of the three obvious test panels
    are of this kind, which is why the fourth column is in the output."""
    for kind in ("isotropic", "one factor"):
        table = averaged(kind)
        assert table["equal weight"] == pytest.approx(table["truth"], rel=1e-9)
        assert table["equal weight"] <= table["clipped"]
        assert table["equal weight"] <= table["shrunk"]


def test_cleaning_wins_where_the_truth_distinguishes_the_assets() -> None:
    """The only panel of the three that tests anything: three factors with
    unequal loadings, so the equal-weight portfolio is not optimal and the
    shrinkage target is not the truth. Cleaning closes 92.9% of the gap between
    the sample estimate and the truth, shrinkage 69.3%, and equal weighting
    50.7%."""
    table = averaged("three factors")
    gap = table["sample"] - table["truth"]
    closed = {
        name: (table["sample"] - table[name]) / gap
        for name in ("shrunk", "clipped", "equal weight")
    }
    assert closed["clipped"] > closed["shrunk"] > closed["equal weight"]
    assert closed["clipped"] > 0.8
    assert closed["shrunk"] > 0.55
    assert table["equal weight"] > table["clipped"]


def test_the_sample_estimator_has_no_answer_when_it_is_singular() -> None:
    """More assets than training observations, so the sample covariance is
    singular and its row reports no variance rather than a number from a solve
    that should not have succeeded. The other two still answer."""
    panel, _ = simulated(30, 50, loadings=[[0.6] for _ in range(30)], seed=6)
    results = {one.estimator: one for one in compare_estimators(panel)}
    assert results["sample"].variance is None
    assert results["sample"].weights == []
    assert results["clipped"].variance is not None
    assert results["equal weight"].variance is not None


def test_the_sample_portfolio_is_the_concentrated_one() -> None:
    """The failure mode of the sample estimator is not a slightly wrong number,
    it is a portfolio of large offsetting bets. Reported as the sum of squared
    weights, whose reciprocal is the effective number of positions."""
    panel, _ = simulated(
        40, 160, loadings=[[0.8, 0.2] for _ in range(40)], seed=7
    )
    results = {one.estimator: one for one in compare_estimators(panel)}
    assert results["sample"].concentration > results["clipped"].concentration
    assert results["equal weight"].concentration == pytest.approx(1.0 / 40.0, rel=1e-12)


def test_clipping_and_shrinking_agree_about_the_total_variance() -> None:
    """Both preserve the trace, by construction in one case and by the target's
    scaling in the other, so the two estimates cannot disagree about how much
    risk there is in total — only about where it is."""
    panel, _ = simulated(30, 120, loadings=[[0.7] for _ in range(30)], seed=8)
    clipped = clip_spectrum(panel).covariance
    shrunk = ledoit_wolf(panel).matrix
    sample = sample_covariance(panel)
    traces = [sum(matrix[i][i] for i in range(30)) for matrix in (clipped, shrunk, sample)]
    assert traces[0] == pytest.approx(traces[2], rel=1e-10)
    assert traces[1] == pytest.approx(traces[2], rel=1e-10)


@pytest.mark.parametrize("split", [0.0, 1.0, -0.5, 1.5])
def test_a_degenerate_split_is_refused(split: float) -> None:
    panel, _ = simulated(10, 100, seed=9)
    with pytest.raises(BadDimensions, match="strictly between zero and one"):
        compare_estimators(panel, split=split)


def test_a_split_that_leaves_nothing_to_measure_on_is_refused() -> None:
    panel, _ = simulated(4, 20, seed=10)
    with pytest.raises(TooShort, match="both halves need at least two"):
        compare_estimators(panel, split=0.99)

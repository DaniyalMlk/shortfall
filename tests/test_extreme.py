"""The fitted tail, checked against the closed forms and then measured.

Extreme value theory is unusually easy to implement plausibly and wrongly. Every
expression divides by the shape, so the exponential case is a removable
singularity that a naive implementation walks straight into; the quantile and the
tail mean are affine in each other, so a sign error in one of them still produces
a monotone increasing function of confidence; and the whole point of the
estimator is to report numbers beyond the sample, where there is nothing to check
them against.

So the load-bearing tests here are the exact ones. Grimshaw's reduction has to
agree with the two-parameter log-likelihood it is a reduction *of*, at the point
it returns. The moment estimator has to recover the parameters exactly from a
sample built out of the population's own quantiles. The mean excess of a
generalised Pareto is affine in the threshold with a known slope, and that
identity pins the tail mean against the quantile. Everything is scale
equivariant, which catches a threshold folded in at the wrong point.

What is left is measured rather than asserted, and the measurements are in
``examples/tail_comparison.py`` so they run in continuous integration and cannot
go stale in a docstring. The short version is that the fitted tail is worth most
where there is no data at all, and worth almost nothing at a confidence the
sample still covers.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from shortfall.extreme import (
    MINIMUM_EXCEEDANCES,
    SHAPE_EPSILON,
    ExtremeRisk,
    GeneralisedPareto,
    ImpossibleFit,
    NotEnoughTail,
    OutsideTheFit,
    TailMethod,
    UndefinedTailMean,
    _log_likelihood,
    _probability_weighted_moments,
    _profile,
    excesses_over,
    extreme_risk,
    fit_generalised_pareto,
    fit_tail,
    hill_curve,
    hill_shape,
    mean_excess_curve,
    threshold_for,
)
from shortfall.series import Convention, ReturnSeries, TooShort

# -- generators ---------------------------------------------------------------


def pareto_quantile(probability: float, shape: float, scale: float) -> float:
    """The generalised Pareto quantile, written out independently of the module."""
    if shape == 0.0:
        return -scale * math.log1p(-probability)
    return scale / shape * (math.pow(1.0 - probability, -shape) - 1.0)


def pareto_sample(
    rng: random.Random, shape: float, scale: float, count: int
) -> list[float]:
    return [pareto_quantile(rng.random(), shape, scale) for _ in range(count)]


def exact_pareto(shape: float, scale: float, count: int) -> list[float]:
    """A sample placed on the population quantiles, with no sampling noise.

    Used where an estimator has an exact answer on the population: the moment
    estimator is a function of two sample moments, and on a sample built this way
    those moments are quadrature approximations to the population's.
    """
    return [
        pareto_quantile((index + 0.5) / count, shape, scale) for index in range(count)
    ]


def student_t(rng: random.Random, degrees: float) -> float:
    """One Student-t draw, as a normal over the root of a scaled chi-square.

    The tail index of a Student-t is the reciprocal of its degrees of freedom, so
    this is the one generator here whose *true* shape is known while its
    distribution is not a generalised Pareto — which is what makes it the honest
    test of the limit result rather than of the arithmetic.
    """
    chi_square = 2.0 * rng.gammavariate(degrees / 2.0, 1.0)
    return rng.gauss(0.0, 1.0) / math.sqrt(chi_square / degrees)


def student_t_losses(rng: random.Random, degrees: float, count: int) -> list[float]:
    return [-student_t(rng, degrees) for _ in range(count)]


# -- the reduction agrees with the likelihood it reduces -----------------------


@pytest.mark.parametrize("shape", [-0.4, -0.1, 0.0, 0.15, 0.4, 0.9])
def test_the_profile_is_the_log_likelihood_along_its_own_ridge(shape: float) -> None:
    """Grimshaw's substitution has to be the same function, not a similar one.

    For a fixed ``theta = xi / beta`` the shape maximising the likelihood is the
    mean log excess, and substituting it collapses the two-parameter surface to a
    curve. The test is that the collapsed expression equals the full
    log-likelihood at the point the curve passes through — which it does only if
    the algebra that dropped the sum term is right.
    """
    rng = random.Random(41)
    excesses = pareto_sample(rng, shape, 1.3, 300)
    fit = fit_generalised_pareto(excesses, observations=300)
    theta = fit.shape / fit.scale
    # The profile drops the additive ``-n`` that the substitution makes constant,
    # because a constant cannot move an argmax and leaving it out saves a pass
    # over the data at every candidate. The reported log-likelihood puts it back,
    # so the two differ by exactly the sample size and by nothing else.
    assert _profile(excesses, theta) - len(excesses) == pytest.approx(
        _log_likelihood(excesses, fit.shape, fit.scale), rel=1e-12
    )
    assert fit.log_likelihood == pytest.approx(
        _profile(excesses, theta) - len(excesses), rel=1e-12
    )


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.25, 0.6, 1.1])
def test_the_fit_is_not_beaten_by_a_search_over_the_whole_surface(
    shape: float,
) -> None:
    """A two-dimensional sweep cannot find a better point than the optimiser did.

    This is the test that would have caught the first version of the search. Its
    upper bound on ``theta`` came from doubling until the profile stopped
    improving, which stepped past the maximum on a shape of 0.5 and stopped short
    of it; the fit looked entirely reasonable and sat 0.17 of log-likelihood
    below the best a grid could find.
    """
    rng = random.Random(97)
    excesses = pareto_sample(rng, shape, 2.0, 250)
    fit = fit_generalised_pareto(excesses, observations=250)
    best = -math.inf
    for index in range(200):
        candidate_shape = -0.95 + 2.75 * index / 199.0
        for other in range(200):
            candidate_scale = 0.05 + 6.0 * other / 199.0
            best = max(best, _log_likelihood(excesses, candidate_shape, candidate_scale))
    assert fit.log_likelihood >= best - 1e-9


def test_the_optimiser_finds_a_heavy_tail_whose_scale_is_tiny() -> None:
    """The search bound has to be free of the units of the data.

    A bound expressed as a multiple of ``1 / mean(excess)`` is; one expressed as
    a constant is not, and the failure only appears when the losses are small
    numbers, which for returns expressed as fractions they always are.
    """
    rng = random.Random(3)
    excesses = pareto_sample(rng, 0.7, 0.0004, 400)
    fit = fit_generalised_pareto(excesses, observations=400)
    assert fit.shape == pytest.approx(0.7, abs=0.12)
    assert fit.scale == pytest.approx(0.0004, rel=0.25)


# -- the moment estimator is exact on the population --------------------------


@pytest.mark.parametrize("shape", [-0.2, 0.0, 0.1, 0.3, 0.45])
def test_the_moment_estimator_recovers_the_population_it_was_built_from(
    shape: float,
) -> None:
    """Probability-weighted moments are a closed form, so this is arithmetic.

    On a sample placed at the population quantiles the two sample moments are
    midpoint quadrature for the population integrals, so the estimator returns
    the population parameters to the accuracy of the quadrature. Getting the
    plotting position wrong, or the weight exponent, breaks this at the third
    digit while leaving a random sample looking fine.
    """
    excesses = exact_pareto(shape, 1.5, 4000)
    fit = fit_generalised_pareto(
        excesses,
        observations=4000,
        method=TailMethod.PROBABILITY_WEIGHTED_MOMENTS,
    )
    assert fit.shape == pytest.approx(shape, abs=0.01)
    assert fit.scale == pytest.approx(1.5, rel=0.01)
    assert fit.shape_standard_error is None
    assert fit.scale_standard_error is None


def test_the_moment_estimator_cannot_report_a_shape_of_one_whatever_it_is_given() -> None:
    """A bound of the estimator rather than of the data, and worth knowing about.

    The moment estimator's shape is ``1 - 2 / (R - 2)`` where ``R`` is the ratio
    of the plain mean to the probability-weighted one. Its weights fall as the
    data rise, so ``R`` is above 2 for any sample of distinct positive excesses,
    so the shape is strictly below 1 — always, on any data. A tail with no finite
    mean is therefore reported by this route as a shape just under one, and
    maximum likelihood is the only route that can say what it actually is.

    Checked here over a sweep that includes shapes of 1.4 and 2.5, where the
    population mean does not exist at all.
    """
    for seed in range(60):
        rng = random.Random(seed)
        shape = rng.choice([0.1, 0.5, 1.0, 1.4, 2.5])
        count = rng.choice([12, 40, 200])
        estimated, scale = _probability_weighted_moments(
            pareto_sample(rng, shape, 1.0, count)
        )
        assert estimated < 1.0
        assert scale > 0.0
    heavy = exact_pareto(1.4, 1.0, 400)
    moments = fit_generalised_pareto(
        heavy, observations=400, method=TailMethod.PROBABILITY_WEIGHTED_MOMENTS
    )
    likelihood = fit_generalised_pareto(heavy, observations=400)
    assert moments.shape < 1.0 < likelihood.shape
    with pytest.raises(UndefinedTailMean):
        likelihood.expected_shortfall(0.999)
    assert moments.expected_shortfall(0.999) > 0.0


def test_a_tail_that_ends_below_the_largest_exceedance_is_refused() -> None:
    """The moment estimator's other failure, and it is not rare.

    A negative shape bounds the loss above, and nothing in the closed form stops
    that bound landing below the largest exceedance it was fitted to. Over 40,000
    samples drawn with shapes between -1.2 and 0.4 it happened to about one in six
    of the samples the estimator gave a negative shape, and the bound came in as
    low as 0.58 of the largest exceedance. The seed below is one of them.

    Maximum likelihood cannot do it — the likelihood is negative infinity
    wherever an observation is outside the support — so the same sample fits, and
    the test checks that rather than only the refusal.
    """
    rng = random.Random(13)
    excesses = pareto_sample(rng, -0.786, 1.0, 60)
    with pytest.raises(ImpossibleFit, match="zero probability to a loss that happened"):
        fit_generalised_pareto(
            excesses,
            observations=600,
            method=TailMethod.PROBABILITY_WEIGHTED_MOMENTS,
        )
    likelihood = fit_generalised_pareto(excesses, observations=600)
    endpoint = likelihood.upper_endpoint
    assert endpoint is not None
    assert endpoint > max(excesses)


# -- the identities the estimate has to satisfy -------------------------------


def a_fit(
    *,
    shape: float,
    scale: float = 0.02,
    threshold: float = 0.03,
    exceedances: int = 100,
    observations: int = 2000,
) -> GeneralisedPareto:
    return GeneralisedPareto(
        shape=shape,
        scale=scale,
        threshold=threshold,
        exceedances=exceedances,
        observations=observations,
        method=TailMethod.MAXIMUM_LIKELIHOOD,
        log_likelihood=0.0,
    )


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.2, 0.5, 0.85])
@pytest.mark.parametrize("confidence", [0.95, 0.99, 0.999, 0.99999])
def test_the_quantile_and_the_tail_probability_invert_each_other(
    shape: float, confidence: float
) -> None:
    fit = a_fit(shape=shape)
    quantile = fit.quantile(confidence)
    assert fit.tail_probability(quantile) == pytest.approx(1.0 - confidence, rel=1e-10)


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.2, 0.5, 0.85])
def test_the_quantile_at_the_lowest_confidence_is_the_threshold(shape: float) -> None:
    """The fit is continuous with the empirical exceedance probability.

    At ``1 - Nu/n`` the extrapolation has not started yet, so the answer has to
    be the threshold exactly. A missing or doubled exceedance probability shows
    up here and nowhere else.
    """
    fit = a_fit(shape=shape)
    assert fit.quantile(fit.lowest_confidence) == pytest.approx(fit.threshold, abs=1e-12)


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.2, 0.5, 0.85])
def test_the_tail_mean_at_the_threshold_is_the_mean_excess(shape: float) -> None:
    """``u + beta / (1 - xi)`` is the generalised Pareto's own mean excess.

    The expected shortfall is computed from the quantile by an affine map rather
    than by integrating, so this is the identity that pins the map's intercept.
    """
    fit = a_fit(shape=shape)
    expected = fit.threshold + fit.scale / (1.0 - shape)
    assert fit.expected_shortfall(fit.lowest_confidence) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.2, 0.5, 0.85])
@pytest.mark.parametrize("confidence", [0.99, 0.9995])
def test_the_mean_excess_above_the_quantile_is_affine_in_it(
    shape: float, confidence: float
) -> None:
    """Exceedances of a generalised Pareto are generalised Pareto, same shape.

    So the mean excess above any higher level ``v`` is ``(beta + xi (v - u)) /
    (1 - xi)``: the slope is the shape's and the scale reappears untouched. This
    is the property that makes the closed form legitimate.
    """
    fit = a_fit(shape=shape)
    quantile = fit.quantile(confidence)
    excess = fit.expected_shortfall(confidence) - quantile
    predicted = (fit.scale + shape * (quantile - fit.threshold)) / (1.0 - shape)
    assert excess == pytest.approx(predicted, rel=1e-10)


def test_the_exponential_branch_joins_the_general_one_across_the_cutoff() -> None:
    """The shape appears only as a denominator, so zero is a removable pole.

    The limit forms are used below ``SHAPE_EPSILON``, and what has to be true is
    that nothing steps as a fitted shape crosses it — a quantile that jumps by a
    percent there is invisible in one fit and glaring in the difference of two.

    The comparison is made just either side of the cutoff rather than at some
    convenient small number, because the two expressions genuinely differ by
    about ``xi * log(ratio) ** 2 / 2``: at a shape of 1e-7 and the 99.999%
    quantile that is 4e-6, which is mathematics rather than error and would make
    a tight assertion here fail for the right reason.
    """
    below = a_fit(shape=SHAPE_EPSILON * 0.9)
    above = a_fit(shape=SHAPE_EPSILON * 1.1)
    for confidence in (0.99, 0.999, 0.99999):
        assert below.quantile(confidence) == pytest.approx(
            above.quantile(confidence), rel=1e-6
        )
        assert below.expected_shortfall(confidence) == pytest.approx(
            above.expected_shortfall(confidence), rel=1e-6
        )
        assert below.tail_probability(above.quantile(confidence)) == pytest.approx(
            1.0 - confidence, rel=1e-5
        )


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.2, 0.5, 0.85])
def test_the_estimate_is_coherent_and_monotone(shape: float) -> None:
    fit = a_fit(shape=shape)
    previous_quantile = -math.inf
    previous_probability = math.inf
    for confidence in (0.951, 0.96, 0.99, 0.999, 0.9999):
        quantile = fit.quantile(confidence)
        assert quantile > previous_quantile
        assert fit.expected_shortfall(confidence) >= quantile
        probability = fit.tail_probability(quantile)
        assert probability < previous_probability
        previous_quantile, previous_probability = quantile, probability


def test_a_return_level_is_the_quantile_at_one_over_the_period() -> None:
    fit = a_fit(shape=0.25)
    assert fit.return_level(250.0) == pytest.approx(fit.quantile(1.0 - 1.0 / 250.0))
    with pytest.raises(ValueError, match="more than one period"):
        fit.return_level(1.0)


def test_a_negative_shape_ends_the_tail_and_the_endpoint_is_reachable() -> None:
    """A finite upper endpoint is a claim, so it is reported as one.

    Beyond it the fitted probability is zero rather than a small number, and the
    quantile approaches it from below at every confidence. A shape that comes out
    negative on loss data is nearly always a threshold inside the body, which is
    why the endpoint is exposed rather than buried in the parameters.
    """
    fit = a_fit(shape=-0.25, scale=0.02, threshold=0.03)
    endpoint = fit.upper_endpoint
    assert endpoint is not None
    assert endpoint == pytest.approx(0.03 + 0.02 / 0.25)
    assert fit.tail_probability(endpoint) == pytest.approx(0.0, abs=1e-15)
    assert fit.tail_probability(endpoint * 2.0) == 0.0
    assert fit.quantile(0.9999999) < endpoint
    assert a_fit(shape=0.0).upper_endpoint is None
    assert a_fit(shape=0.3).upper_endpoint is None


@pytest.mark.parametrize("factor", [1e-4, 0.5, 250.0])
def test_the_whole_estimate_is_scale_equivariant(factor: float) -> None:
    """Rescaling the losses rescales the scale, the threshold and every quantile.

    The shape is dimensionless and must not move. This catches a threshold that
    has been added in the wrong place, which is otherwise invisible: both
    versions are monotone in confidence and both look like risk numbers.
    """
    rng = random.Random(11)
    losses = pareto_sample(rng, 0.3, 0.01, 800)
    plain = fit_tail(losses, tail_fraction=0.1)
    scaled = fit_tail([value * factor for value in losses], tail_fraction=0.1)
    # A millionth of a percent, which is the accuracy a derivative-free search
    # can reach on this problem and not a slack tolerance. The profile is
    # quadratic at its maximum, so the values on either side of the optimum
    # differ by less than machine precision once the bracket is within about the
    # square root of it — measured at 5e-8 relative across factors from 1e-4 to
    # 1e3. Tightening the search would not improve it; only a derivative would.
    assert scaled.shape == pytest.approx(plain.shape, rel=1e-6)
    assert scaled.scale == pytest.approx(plain.scale * factor, rel=1e-6)
    assert scaled.threshold == pytest.approx(plain.threshold * factor, rel=1e-12)
    assert scaled.quantile(0.999) == pytest.approx(plain.quantile(0.999) * factor, rel=1e-6)


# -- the threshold ------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.5, 0.1, 0.05, 0.01])
def test_the_threshold_lands_on_an_observation_and_the_count_follows(
    fraction: float,
) -> None:
    """Interpolating here would make the exceedance count depend on the method.

    The count is what connects the fitted tail to an unconditional probability,
    so it is pinned to the order statistic: the ``k``-th largest loss is exceeded
    by exactly ``k - 1`` of them when there are no ties.
    """
    rng = random.Random(23)
    losses = [rng.gauss(0.0, 0.01) for _ in range(1000)]
    threshold = threshold_for(losses, fraction)
    position = round(fraction * 1000)
    assert len(excesses_over(losses, threshold)) == position - 1
    assert threshold == sorted(losses, reverse=True)[position - 1]


def test_losses_equal_to_the_threshold_are_not_exceedances() -> None:
    """A zero excess has no information and biases the scale downwards.

    It contributes ``log(1) = 0`` to the likelihood's sum while still counting in
    the sample size, so including ties pulls the fitted scale towards zero
    without any warning.
    """
    assert excesses_over([1.0, 2.0, 2.0, 3.0], 2.0) == (1.0,)
    assert excesses_over([1.0, 1.0, 1.0], 1.0) == ()


def test_the_mean_excess_of_an_exponential_sample_is_flat() -> None:
    """The exponential is the shape-zero case and it is memoryless.

    So its mean excess does not depend on the threshold at all, and the curve is
    a horizontal line whose height is the scale. A curve that trends on
    exponential data is measuring the estimator, not the data.
    """
    rng = random.Random(19)
    losses = pareto_sample(rng, 0.0, 0.05, 6000)
    curve = mean_excess_curve(losses, points=8, minimum_exceedances=200)
    assert [point.threshold for point in curve] == sorted(
        point.threshold for point in curve
    )
    assert [point.exceedances for point in curve] == sorted(
        (point.exceedances for point in curve), reverse=True
    )
    for point in curve:
        assert point.mean_excess == pytest.approx(0.05, abs=4.0 * point.standard_error)


def test_the_mean_excess_slope_reads_the_shape_back() -> None:
    """Above a generalised Pareto threshold the curve is a line of known slope.

    ``xi / (1 - xi)``, so a least squares fit to the curve is a second estimate
    of the shape that does not use the likelihood at all. It is much noisier than
    the fit — successive points share nearly all their data — so this asserts the
    reading is in the right region rather than that it is accurate.
    """
    rng = random.Random(29)
    losses = pareto_sample(rng, 0.3, 0.04, 20000)
    curve = mean_excess_curve(losses, points=12, minimum_exceedances=400)
    thresholds = [point.threshold for point in curve]
    excesses = [point.mean_excess for point in curve]
    centre = statistics.mean(thresholds)
    height = statistics.mean(excesses)
    variance = math.fsum((value - centre) ** 2 for value in thresholds)
    slope = (
        math.fsum(
            (value - centre) * (excess - height)
            for value, excess in zip(thresholds, excesses, strict=True)
        )
        / variance
    )
    assert slope == pytest.approx(0.3 / 0.7, abs=0.12)


def test_a_curve_refuses_a_sample_it_cannot_cover() -> None:
    with pytest.raises(NotEnoughTail, match="lower `minimum_exceedances`"):
        mean_excess_curve([1.0, 2.0, 3.0], minimum_exceedances=50)
    with pytest.raises(ValueError, match="at least two points"):
        mean_excess_curve([float(index) for index in range(100)], points=1)


# -- the Hill estimator -------------------------------------------------------


def test_the_hill_estimator_is_exact_on_a_geometric_tail() -> None:
    """Constructed so the answer is known without any statistics.

    If the top ``k`` losses are all the same multiple ``exp(xi)`` of the
    ``k+1``-th, the mean log excess is ``xi`` exactly. An off-by-one in which
    order statistic is the pivot shows up immediately, because the pivot would
    then be inside the mean.
    """
    pivot = 3.0
    top = [pivot * math.exp(0.4)] * 20
    estimate = hill_shape([*top, pivot, 1.0, 0.5], 20)
    assert estimate.shape == pytest.approx(0.4, rel=1e-12)
    assert estimate.threshold == pivot
    assert estimate.standard_error == pytest.approx(0.4 / math.sqrt(20))


def test_the_hill_estimator_agrees_with_the_fit_on_a_pareto_tail() -> None:
    """Two estimators of the same quantity, on data where both are right.

    A pure Pareto sample is in the generalised Pareto family with threshold zero,
    so the likelihood fit and the Hill estimator are estimating the same shape
    from the same observations by different routes.
    """
    rng = random.Random(31)
    # Pareto of the first kind: the survivor is exactly ``x ** (-1 / xi)``, so
    # the exceedances over any threshold are exactly generalised Pareto with the
    # same shape and a scale of ``xi * threshold``. Both estimators are then
    # estimating the same parameter of the same correct model, with no
    # approximation left between them.
    losses = [math.pow(1.0 - rng.random(), -0.35) for _ in range(4000)]
    fitted = fit_tail(losses, tail_fraction=0.1)
    hill = hill_shape(losses, 400)
    assert fitted.scale == pytest.approx(0.35 * fitted.threshold, rel=0.2)
    assert hill.shape == pytest.approx(0.35, abs=0.05)
    assert fitted.shape == pytest.approx(0.35, abs=0.05)
    assert hill.shape == pytest.approx(fitted.shape, abs=0.04)


def test_the_hill_estimator_is_biased_upwards_by_a_shifted_tail() -> None:
    """Which is why it is a second opinion rather than a check.

    Hill assumes the tail is Pareto about the origin. A generalised Pareto with a
    positive scale is a Pareto *shifted* by ``beta / xi``, and at a threshold
    where that shift is still a large fraction of the loss the estimator reads
    high: on 4,000 draws with a shape of 0.35 and a scale of 1, the shift is 2.86
    against a 90th percentile of 3.5, and Hill on the worst 400 returns about
    0.55 against the true 0.35. The likelihood fit, which knows about the scale,
    does not have this problem.

    So a Hill curve and a fit that disagree is a statement about the threshold,
    not evidence that one of them is broken.
    """
    rng = random.Random(31)
    losses = pareto_sample(rng, 0.35, 1.0, 4000)
    hill = hill_shape(losses, 400)
    fitted = fit_tail(losses, tail_fraction=0.1)
    assert hill.shape > fitted.shape + 0.1
    assert fitted.shape == pytest.approx(0.35, abs=0.08)
    assert hill.shape == pytest.approx(0.55, abs=0.08)


def test_the_hill_estimator_refuses_signed_returns() -> None:
    """It works in logarithms, so a gain passed in as a loss is not a small number.

    It is a negative one, and ``log`` of it does not exist. The refusal says what
    to do, because the mistake is passing a return series where a loss series was
    wanted and the fix is to cut it rather than to shift it.
    """
    rng = random.Random(37)
    signed = [rng.gauss(0.0, 0.01) for _ in range(200)]
    with pytest.raises(ValueError, match="cut to its losses"):
        hill_shape(signed, 150)


def test_the_hill_estimator_needs_one_more_observation_than_it_uses() -> None:
    losses = [float(index + 1) for index in range(50)]
    with pytest.raises(NotEnoughTail, match="at most 49"):
        hill_shape(losses, 50)
    with pytest.raises(ValueError, match="at least two order statistics"):
        hill_shape(losses, 1)
    assert hill_shape(losses, 49).order_statistics == 49


def test_the_hill_curve_walks_the_order_statistics() -> None:
    rng = random.Random(43)
    losses = [abs(rng.gauss(0.0, 1.0)) for _ in range(500)]
    curve = hill_curve(losses, points=6)
    counts = [point.order_statistics for point in curve]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)
    assert counts[0] == MINIMUM_EXCEEDANCES
    assert counts[-1] <= 499
    with pytest.raises(NotEnoughTail):
        hill_curve([1.0, 2.0, 3.0])


# -- recovery, and the bias that recovery does not have --------------------


@pytest.mark.parametrize(
    ("shape", "exceedances"), [(0.1, 400), (0.25, 400), (0.5, 400), (0.25, 120)]
)
def test_the_fit_recovers_its_own_family_within_its_own_standard_error(
    shape: float, exceedances: int
) -> None:
    """On exact generalised Pareto data the model is right, so this is estimation.

    Three standard errors is the band, which is the assertion the standard error
    itself justifies — and the coverage of that standard error is measured
    separately, because at a hundred exceedances it is optimistic.
    """
    rng = random.Random(53 + exceedances)
    excesses = pareto_sample(rng, shape, 1.0, exceedances)
    fit = fit_generalised_pareto(excesses, observations=exceedances * 20)
    error = fit.shape_standard_error
    assert error is not None
    assert abs(fit.shape - shape) < 3.0 * error
    assert fit.exceedance_probability == pytest.approx(0.05)
    assert fit.lowest_confidence == pytest.approx(0.95)


def test_the_shape_is_biased_down_on_a_student_t_and_the_note_says_so() -> None:
    """The limit result is a limit, and a Student-t approaches it slowly.

    A Student-t on four degrees of freedom has tail index 0.25 exactly. Fitted
    above the worst 5% of 2,000 draws the estimate averages about 0.15 over
    twenty samples — biased towards zero, because at that threshold the body is
    still contributing. Raising the threshold reduces the bias and raises the
    variance faster; ``examples/tail_comparison.py`` measures the trade.

    Asserted as a direction with a generous band rather than as a number,
    because the number is a property of this generator and twenty samples.
    """
    shapes = []
    for seed in range(20):
        rng = random.Random(4000 + seed)
        losses = student_t_losses(rng, 4.0, 2000)
        shapes.append(fit_tail(losses, tail_fraction=0.05).shape)
    mean = statistics.mean(shapes)
    assert 0.05 < mean < 0.25
    assert statistics.mean(
        fit_tail(
            student_t_losses(random.Random(4000 + seed), 4.0, 2000), tail_fraction=0.2
        ).shape
        for seed in range(20)
    ) < mean


# -- refusals -----------------------------------------------------------------


def test_a_confidence_inside_the_body_is_refused_with_the_number_that_is_not() -> None:
    """The fit deliberately knows nothing below its threshold.

    Reading the empirical quantile instead would be a different estimator
    answering under this one's name, so it raises — and names the lowest
    confidence it does cover, because that is the number the caller needs.
    """
    fit = a_fit(shape=0.2, exceedances=100, observations=2000)
    with pytest.raises(OutsideTheFit, match=r"0\.95"):
        fit.quantile(0.9)
    with pytest.raises(OutsideTheFit):
        fit.expected_shortfall(0.5)
    assert fit.quantile(0.95) == pytest.approx(fit.threshold)
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            fit.quantile(bad)


def test_a_shape_of_one_or_more_has_no_tail_mean_and_says_which_shape() -> None:
    """An infinite expectation is not a large number and must not be returned.

    The value at risk is still defined at that shape, so the refusal is on the
    expected shortfall alone and the message says the quantile survives.
    """
    fit = a_fit(shape=1.0)
    with pytest.raises(UndefinedTailMean, match=r"1\.0000"):
        fit.expected_shortfall(0.99)
    assert fit.quantile(0.99) > fit.threshold
    with pytest.raises(UndefinedTailMean, match="Value at risk is still defined"):
        a_fit(shape=1.6).expected_shortfall(0.999)


def test_too_few_exceedances_is_refused_rather_than_fitted() -> None:
    rng = random.Random(59)
    excesses = pareto_sample(rng, 0.2, 1.0, MINIMUM_EXCEEDANCES - 1)
    with pytest.raises(NotEnoughTail, match="largest observation with extra steps"):
        fit_generalised_pareto(excesses, observations=500)
    losses = [rng.gauss(0.0, 0.01) for _ in range(40)]
    with pytest.raises(NotEnoughTail, match=r"below the 10 a fit will accept"):
        fit_tail(losses, tail_fraction=0.05)


def test_the_boundary_conditions_on_the_inputs() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fit_generalised_pareto([1.0, 2.0, 0.0] + [1.5] * 10, observations=100)
    with pytest.raises(ValueError, match="not finite"):
        fit_generalised_pareto([1.0, float("nan")] + [1.5] * 10, observations=100)
    with pytest.raises(TooShort, match="empty"):
        fit_generalised_pareto([], observations=10)
    with pytest.raises(ValueError, match="not possible"):
        fit_generalised_pareto([1.0] * 20, observations=5)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        threshold_for([1.0, 2.0, 3.0], 1.0)
    with pytest.raises(ValueError, match="not finite"):
        excesses_over([1.0, 2.0], float("inf"))


def test_a_fit_cannot_be_constructed_with_impossible_counts() -> None:
    with pytest.raises(ValueError, match="the scale is positive"):
        a_fit(shape=0.2, scale=0.0)
    with pytest.raises(ValueError, match="not possible"):
        a_fit(shape=0.2, exceedances=100, observations=50)
    with pytest.raises(ValueError, match="a fit has exceedances"):
        a_fit(shape=0.2, exceedances=0)


# -- the estimate over a return series ----------------------------------------


def test_returns_are_negated_once_and_at_the_boundary() -> None:
    """The one sign convention in the module, applied in one place.

    A gain is a negative loss and is dropped by the threshold, so nothing has to
    filter the series first — and the quantile field carries the other sign
    convention so a caller working in returns does not negate it back by hand.
    """
    rng = random.Random(61)
    returns = [-value for value in pareto_sample(rng, 0.3, 0.01, 1000)]
    returns = [value + rng.gauss(0.0, 0.002) for value in returns]
    estimate = extreme_risk(returns, confidence=0.999, tail_fraction=0.1)
    direct = fit_tail([-value for value in returns], tail_fraction=0.1)
    assert estimate.fit.shape == pytest.approx(direct.shape)
    assert estimate.value_at_risk == pytest.approx(direct.quantile(0.999))
    assert estimate.quantile == pytest.approx(-estimate.value_at_risk)
    assert estimate.expected_shortfall is not None
    assert estimate.expected_shortfall >= estimate.value_at_risk
    assert estimate.tail_probability == pytest.approx(0.001)


def test_a_return_series_goes_in_where_a_list_does() -> None:
    rng = random.Random(67)
    values = tuple(rng.gauss(0.0005, 0.012) for _ in range(1500))
    series = ReturnSeries(name="fund", values=values, convention=Convention.SIMPLE)
    from_series = extreme_risk(series, confidence=0.995, tail_fraction=0.1)
    from_list = extreme_risk(list(values), confidence=0.995, tail_fraction=0.1)
    assert from_series.value_at_risk == pytest.approx(from_list.value_at_risk)


def test_how_much_of_the_answer_is_extrapolation_is_reported() -> None:
    """The count of observations beyond the estimate, which is often zero.

    That is the estimator working rather than failing, and it is the single most
    useful number for deciding how much weight the figure deserves — so it is a
    field rather than something the caller recomputes.
    """
    rng = random.Random(71)
    returns = [-value for value in pareto_sample(rng, 0.25, 0.01, 2000)]
    inside = extreme_risk(returns, confidence=0.99, tail_fraction=0.1)
    assert inside.observed_beyond == pytest.approx(20, abs=8)
    assert not inside.is_extrapolated
    far = extreme_risk(returns, confidence=0.999999, tail_fraction=0.1)
    assert far.observed_beyond == 0
    assert far.is_extrapolated
    assert far.value_at_risk > max(-value for value in returns)


def test_an_undefined_tail_mean_is_a_missing_field_rather_than_an_exception() -> None:
    """Raising would throw away the value at risk in the same result.

    So the estimate carries ``None`` and the fit itself is what raises, with the
    shape in the message for whoever needs to know why.
    """
    rng = random.Random(73)
    returns = [-value for value in pareto_sample(rng, 1.4, 0.01, 1200)]
    estimate = extreme_risk(returns, confidence=0.999, tail_fraction=0.1)
    assert estimate.fit.shape > 1.0
    assert estimate.expected_shortfall is None
    assert estimate.value_at_risk > 0.0
    with pytest.raises(UndefinedTailMean):
        estimate.fit.expected_shortfall(0.999)


def test_the_estimate_is_not_a_scalable_risk_object() -> None:
    """Deliberately: a fitted tail does not scale by the square root of time.

    The sum of a horizon's heavy-tailed innovations is not a generalised Pareto
    variate with a scaled parameter, and ``Risk.scaled_to`` would apply exactly
    that substitution. The absence is the design, so it is asserted.
    """
    rng = random.Random(79)
    returns = [-value for value in pareto_sample(rng, 0.3, 0.01, 800)]
    estimate = extreme_risk(returns, confidence=0.999, tail_fraction=0.1)
    assert isinstance(estimate, ExtremeRisk)
    assert not hasattr(estimate, "scaled_to")
    assert not hasattr(estimate, "volatility")

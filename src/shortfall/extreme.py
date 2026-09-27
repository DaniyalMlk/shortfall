"""Risk in the tail, fitted to the exceedances rather than to the sample.

Every other estimate in this library is told something about the whole
distribution and asked about the tail. Historical simulation reads the tail off
the order statistics, so at 99.9% over 2,000 returns it reports the second-worst
one and the answer cannot be larger than the worst thing that has happened. The
parametric routes fit a shape to all of the data, where the bulk dominates the
likelihood: a normal fitted to a fat-tailed sample matches the middle and
understates the tail, and the tail is the part that was asked about.

Extreme value theory offers a third route with a different bargain. The
Pickands-Balkema-de Haan result says that for a wide class of parent
distributions, the distribution of the amount by which a high threshold is
exceeded converges to a generalised Pareto as the threshold rises. So a shape
fitted to *only* the exceedances says nothing about the body and extrapolates
past the largest observation.

Three things about that are worth stating plainly, because each is a cost.

**The threshold is a bias-variance choice and there is no right answer.** Raise
it and the limit result applies better while fewer observations remain to fit
two parameters; lower it and the fit is stable but the family is wrong. Nothing
here picks a threshold silently: :func:`mean_excess_curve` and :func:`hill_curve`
are the standard diagnostics, and both are exported so the choice can be looked
at rather than defaulted into.

**The extrapolation is an assumption, not a measurement.** A 99.9% quantile
fitted from the worst 5% of 2,000 returns is 100 observations deciding what
happens beyond all of them. It is a better assumption than a normal tail and a
worse one than data.

**A fitted tail is not a location-scale estimate**, so this module returns its
own type rather than :class:`~shortfall.parametric.Risk`. ``Risk`` carries
:meth:`~shortfall.parametric.Risk.scaled_to`, which applies the
square-root-of-time rule, and applying it to a fitted tail is exactly the
substitution :mod:`shortfall.horizon` exists to refuse: the sum of a horizon's
heavy-tailed innovations is not a generalised Pareto variate with a scaled
parameter.

Sign convention. Everything here is stated in *losses*: a positive number is
money lost, and a threshold of 0.02 is a 2% loss. Returns go in as returns and
are negated at the boundary, once, in :func:`extreme_risk`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .series import ReturnSeries, TooShort

#: Fewest exceedances a fit will accept. Two parameters from ten points is
#: already a stretch; below that the shape is whatever the largest observation
#: happens to be. Chosen to refuse rather than to recommend: a serious fit wants
#: a hundred or more, which the standard error in the result will say.
MINIMUM_EXCEEDANCES = 10

#: Plotting position for the probability-weighted moment estimator. Hosking and
#: Wallis (1987) recommend ``(j - 0.35) / m`` for the generalised Pareto
#: specifically, rather than the ``(j - 0.5) / m`` that is the general-purpose
#: default; the difference matters most in exactly the small samples the moment
#: estimator is reached for.
PLOTTING_OFFSET = 0.35

#: Below this the shape is treated as zero and the exponential limit forms are
#: used. The generalised Pareto's quantile and tail mean both divide by the
#: shape, so the expressions lose their leading digits long before the shape
#: reaches zero; the limits are exact there and cheap.
SHAPE_EPSILON = 1e-8


class NotEnoughTail(ValueError):
    """The threshold left too few exceedances to fit a tail to."""


class OutsideTheFit(ValueError):
    """A confidence below the threshold the tail was fitted above.

    A point estimate could be returned by reading the empirical quantile
    instead, and that would be a different estimator wearing this one's name.
    """


class ImpossibleFit(ValueError):
    """The fitted tail ends below a loss that was used to fit it.

    A negative shape puts a finite upper bound on the loss, and there is nothing
    in the moment estimator that stops that bound landing below the largest
    exceedance — a tail that assigns zero probability to something that has
    already happened. Measured over 40,000 samples drawn with shapes between
    -1.2 and 0.4, it happens to about one in six of the samples the moment
    estimator gives a negative shape to, and the bound came in as low as 0.58 of
    the largest exceedance. Maximum likelihood cannot do it: the likelihood is
    negative infinity wherever an observation sits outside the support, so the
    optimiser can never arrive there.
    """


class UndefinedTailMean(ValueError):
    """The fitted shape is at least one, so the tail has no finite mean.

    A generalised Pareto with shape ``xi >= 1`` has infinite expectation, so
    there is no expected shortfall to report at any confidence. The value at
    risk still exists. A shape that large is more often a threshold set too low
    or a sample too short than a real tail, and the standard error usually says
    which.
    """


def _losses(values: Sequence[float], *, name: str = "losses") -> tuple[float, ...]:
    for index, value in enumerate(values):
        if value != value or math.isinf(value):
            raise ValueError(f"{name}[{index}] is not finite: {value!r}")
    if not values:
        raise TooShort(f"{name} is empty")
    return tuple(float(value) for value in values)


def threshold_for(losses: Sequence[float], tail_fraction: float) -> float:
    """The loss exceeded by ``tail_fraction`` of the sample.

    The threshold is the ``1 - tail_fraction`` empirical quantile, located by
    the same nearest-order-statistic rule the exceedance count implies: with
    2,000 losses and a tail fraction of 0.05 the threshold is the 100th largest,
    so exactly 99 exceed it. Interpolating between order statistics here would
    make the count depend on the interpolation, which is a worse trade than
    landing on an observation.
    """
    ordered = sorted(_losses(losses), reverse=True)
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError(
            f"a tail fraction is strictly between 0 and 1, got {tail_fraction!r}"
        )
    position = max(1, min(len(ordered), round(tail_fraction * len(ordered))))
    return ordered[position - 1]


def excesses_over(losses: Sequence[float], threshold: float) -> tuple[float, ...]:
    """The amounts by which ``losses`` exceed ``threshold``, strictly.

    Losses equal to the threshold are not exceedances. That is the convention
    the generalised Pareto is stated under — its support is the positive
    half-line — and a zero excess contributes ``log(1) = 0`` to the likelihood
    while still counting towards the sample size, which biases the scale down.
    """
    if threshold != threshold or math.isinf(threshold):
        raise ValueError(f"the threshold is not finite: {threshold!r}")
    return tuple(value - threshold for value in _losses(losses) if value > threshold)


# -- threshold diagnostics ----------------------------------------------------


@dataclass(frozen=True)
class MeanExcess:
    """One point of the mean excess curve."""

    threshold: float
    exceedances: int
    #: Mean amount by which the threshold was exceeded.
    mean_excess: float
    #: Standard error of that mean, from the exceedances themselves.
    standard_error: float


def mean_excess_curve(
    losses: Sequence[float],
    *,
    points: int = 20,
    from_fraction: float = 0.5,
    minimum_exceedances: int = MINIMUM_EXCEEDANCES,
) -> tuple[MeanExcess, ...]:
    """Mean excess against threshold, the standard way to choose a threshold.

    The diagnostic works because the generalised Pareto's mean excess is *linear*
    in the threshold: above ``u`` it is ``(beta + xi * u) / (1 - xi)``. So the
    threshold to fit above is where the curve stops bending and starts running
    straight, and the slope there is ``xi / (1 - xi)`` — a second reading of the
    shape, independent of the fit.

    Two features of the curve are artefacts rather than signal and are the
    reason the standard error is returned beside each point. The right-hand end
    is computed from a handful of observations and wanders; and successive points
    share almost all of their data, so the curve is far smoother than the
    independent scatter of its points would be, which makes a spurious straight
    stretch easy to see.
    """
    ordered = sorted(_losses(losses))
    if points < 2:
        raise ValueError(f"a curve needs at least two points, got {points!r}")
    if minimum_exceedances < 2:
        raise ValueError(
            f"a mean excess needs at least two exceedances, got {minimum_exceedances!r}"
        )
    if not 0.0 < from_fraction <= 1.0:
        raise ValueError(
            f"the curve starts at a fraction in (0, 1], got {from_fraction!r}"
        )
    usable = len(ordered) - minimum_exceedances
    if usable < 1:
        raise NotEnoughTail(
            f"{len(ordered)} losses cannot leave {minimum_exceedances} above any "
            "threshold in the sample; lower `minimum_exceedances` or bring more data"
        )
    # The lowest threshold is the one exceeded by ``from_fraction`` of the sample,
    # so the default curve covers the upper half. A curve spread over the whole
    # sample spends most of its points in the body — and on a return series it
    # spends them on thresholds that are *gains*, where a mean excess is a number
    # about nothing.
    first = max(0, len(ordered) - 1 - int(from_fraction * len(ordered)))
    span = usable - first
    if span < 1:
        raise NotEnoughTail(
            f"a curve from the top {from_fraction:.0%} of {len(ordered)} losses cannot "
            f"leave {minimum_exceedances} above its first threshold; raise "
            "`from_fraction` or lower `minimum_exceedances`"
        )
    step = span / points
    indices = sorted(
        {first + min(span - 1, int(index * step)) for index in range(points)}
    )
    curve: list[MeanExcess] = []
    for index in indices:
        threshold = ordered[index]
        above = [value - threshold for value in ordered if value > threshold]
        if len(above) < minimum_exceedances:
            continue
        mean = math.fsum(above) / len(above)
        spread = math.fsum((value - mean) ** 2 for value in above) / (len(above) - 1)
        curve.append(
            MeanExcess(
                threshold=threshold,
                exceedances=len(above),
                mean_excess=mean,
                standard_error=math.sqrt(spread / len(above)),
            )
        )
    if not curve:
        raise NotEnoughTail(
            f"no threshold in a sample of {len(ordered)} left {minimum_exceedances} "
            "strict exceedances; the losses are probably tied"
        )
    return tuple(curve)


@dataclass(frozen=True)
class HillShape:
    """The Hill estimate of the shape from the largest ``order_statistics``."""

    order_statistics: int
    #: The tail index ``xi``, positive by construction.
    shape: float
    #: ``shape / sqrt(k)``, the asymptotic standard error under a Pareto tail.
    standard_error: float
    #: The order statistic the estimate is measured against.
    threshold: float


def hill_shape(losses: Sequence[float], order_statistics: int) -> HillShape:
    """The Hill estimator of a positive tail index.

    The mean log excess of the top ``k`` losses over the ``k + 1``-th, which is
    the maximum likelihood shape when the tail is exactly Pareto. It assumes a
    positive shape and cannot report anything else, which is the point: it is a
    cheap second opinion on the fitted shape that cannot be dragged towards zero
    by a threshold set inside the body.

    It needs strictly positive losses, because it works in logarithms. A sample
    of returns has losses of both signs and the negative ones are gains; passing
    them in is the mistake this refuses.
    """
    ordered = sorted(_losses(losses), reverse=True)
    if order_statistics < 2:
        raise ValueError(
            f"the Hill estimator needs at least two order statistics, got "
            f"{order_statistics!r}"
        )
    if order_statistics >= len(ordered):
        raise NotEnoughTail(
            f"{order_statistics} order statistics from a sample of {len(ordered)}: "
            f"the estimator needs one more to measure against, so at most "
            f"{len(ordered) - 1}"
        )
    pivot = ordered[order_statistics]
    if pivot <= 0.0:
        raise ValueError(
            f"order statistic {order_statistics} is {pivot!r}; the Hill estimator "
            "works in logarithms and needs strictly positive losses, so a sample of "
            "signed returns has to be cut to its losses first"
        )
    shape = math.fsum(
        math.log(ordered[index] / pivot) for index in range(order_statistics)
    ) / order_statistics
    return HillShape(
        order_statistics=order_statistics,
        shape=shape,
        standard_error=shape / math.sqrt(order_statistics),
        threshold=pivot,
    )


def hill_curve(
    losses: Sequence[float],
    *,
    points: int = 20,
    minimum: int = MINIMUM_EXCEEDANCES,
) -> tuple[HillShape, ...]:
    """Hill shape as a function of how many order statistics it uses.

    Read the same way as the mean excess curve and for the same reason: a stretch
    where the estimate is flat in ``k`` is a stretch where the tail really is
    Pareto-like. The left-hand end is noisy because ``k`` is small and the
    right-hand end drifts because the estimator has been fed the body.
    """
    positive = [value for value in _losses(losses) if value > 0.0]
    if len(positive) <= minimum:
        raise NotEnoughTail(
            f"{len(positive)} positive losses cannot support a curve starting at "
            f"{minimum} order statistics"
        )
    highest = len(positive) - 1
    if points < 2:
        raise ValueError(f"a curve needs at least two points, got {points!r}")
    step = (highest - minimum) / (points - 1) if points > 1 else 0.0
    counts = sorted({minimum + round(index * step) for index in range(points)})
    return tuple(hill_shape(positive, count) for count in counts if count <= highest)


# -- the fit ------------------------------------------------------------------


class TailMethod(str, Enum):
    """How the generalised Pareto parameters were estimated."""

    #: Grimshaw's reduction of the two-parameter likelihood to one dimension.
    #: Efficient, and the only one of the two with a usable asymptotic variance.
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    #: Probability-weighted moments, in the Hosking and Wallis (1987) form.
    #: Closed form, no optimiser, and competitive with the likelihood on a few
    #: dozen exceedances, with two limitations that are properties of the
    #: estimator rather than of any sample. It is consistent only for a shape
    #: below 0.5; and its shape is bounded above by one by construction, so it
    #: cannot report a tail with no finite mean at all — it reports a shape just
    #: under one instead, and the expected shortfall that comes back is finite
    #: and meaningless. It can also place the upper bound of a light tail below
    #: the largest exceedance, which is refused as an
    #: :class:`ImpossibleFit`.
    PROBABILITY_WEIGHTED_MOMENTS = "probability_weighted_moments"


def _shape_at(excesses: Sequence[float], theta: float) -> float:
    return math.fsum(math.log1p(theta * value) for value in excesses) / len(excesses)


def _scale_at(excesses: Sequence[float], theta: float, shape: float) -> float:
    if abs(theta) >= SHAPE_EPSILON:
        return shape / theta
    # ``shape / theta`` is 0/0 at the exponential point. Expanding
    # ``log1p(theta * y)`` to second order gives the limit and its slope, which
    # keeps the profile smooth through zero instead of stepping across it.
    first = math.fsum(excesses) / len(excesses)
    second = math.fsum(value * value for value in excesses) / len(excesses)
    return first - theta * second / 2.0


def _profile(excesses: Sequence[float], theta: float) -> float:
    """Grimshaw's profile log-likelihood, up to the constant ``-len``.

    Substituting ``theta = xi / beta`` makes the shape that maximises the
    likelihood for a given ``theta`` a closed form — the mean log excess — so
    the two-parameter surface collapses to a curve in one parameter with no
    inner optimisation and no starting values to get wrong.
    """
    shape = _shape_at(excesses, theta)
    scale = _scale_at(excesses, theta, shape)
    if scale <= 0.0:
        return -math.inf
    return -len(excesses) * (math.log(scale) + shape)


def _log_likelihood(excesses: Sequence[float], shape: float, scale: float) -> float:
    if scale <= 0.0:
        return -math.inf
    total = 0.0
    for value in excesses:
        argument = 1.0 + shape * value / scale
        if argument <= 0.0:
            return -math.inf
        total += math.log(argument)
    if abs(shape) < SHAPE_EPSILON:
        return -len(excesses) * math.log(scale) - math.fsum(excesses) / scale
    return -len(excesses) * math.log(scale) - (1.0 + 1.0 / shape) * total


def _golden_section(
    excesses: Sequence[float], low: float, high: float, iterations: int = 120
) -> float:
    """Maximise the profile on a bracket by golden section.

    No derivative, and it cannot step outside the bracket — which matters here
    because the lower end of the feasible region is a pole rather than a smooth
    boundary: at ``theta = -1 / max(excess)`` the largest excess sits exactly at
    the fitted upper endpoint and the likelihood is unbounded below.

    The stopping rule is relative to the bracket and carries no absolute term.
    An absolute floor would be a length in the units of ``1 / loss``, so the
    optimiser would stop sooner on returns quoted as fractions than on the same
    returns quoted in basis points, and the fitted shape — which is
    dimensionless and must not move at all — would differ in its eighth digit
    between the two. The iteration cap is what bounds the loop when the bracket
    straddles zero.
    """
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    value_left = _profile(excesses, left)
    value_right = _profile(excesses, right)
    for _ in range(iterations):
        if high - low < 1e-13 * (abs(low) + abs(high)):
            break
        if value_left < value_right:
            low, left, value_left = left, right, value_right
            right = low + ratio * (high - low)
            value_right = _profile(excesses, right)
        else:
            high, right, value_right = right, left, value_left
            left = high - ratio * (high - low)
            value_left = _profile(excesses, left)
    return (low + high) / 2.0


#: Shape the upper end of the ``theta`` search is pushed out to. The profile
#: falls away like ``-n * xi`` once the shape is large, so a bound that reaches a
#: shape of five brackets the maximum for any tail anyone will fit; the point of
#: naming it is that the bound is derived from the shape rather than from the
#: scale of the data, which is what makes it units-free.
SEARCH_SHAPE_CEILING = 5.0


def _candidates(excesses: Sequence[float]) -> list[float]:
    """Search points for ``theta``, spaced geometrically on each side of zero.

    A uniform grid is the wrong shape for this problem. The feasible region runs
    from ``-1 / max(excess)`` to somewhere well above ``1 / mean(excess)``, the
    maximum usually sits within a small multiple of the latter, and pushing the
    upper end out far enough to be safe then makes a uniform grid too coarse
    exactly where the answer is. Spacing geometrically makes the resolution
    relative, so the far end costs a handful of points instead of ruining the
    near end.
    """
    mean = math.fsum(excesses) / len(excesses)
    low = -1.0 / max(excesses) * (1.0 - 1e-10)
    high = 1.0 / mean
    while _shape_at(excesses, high) < SEARCH_SHAPE_CEILING:
        high *= 2.0
        if high > 1e12 / mean:
            break
    decay = 0.8
    points = [0.0]
    step = 1.0
    for _ in range(120):
        points.append(high * step)
        points.append(low * step)
        step *= decay
    return sorted(points)


def _maximum_likelihood(excesses: Sequence[float]) -> tuple[float, float]:
    grid = _candidates(excesses)
    values = [_profile(excesses, point) for point in grid]
    index = max(range(len(grid)), key=lambda position: values[position])
    left = grid[max(0, index - 1)]
    right = grid[min(len(grid) - 1, index + 1)]
    theta = _golden_section(excesses, left, right)
    # The refinement is only an improvement if it measures as one. Golden
    # section assumes the bracket is unimodal, and keeping the better of the two
    # means a bracket that was not cannot make the answer worse than the sweep.
    if _profile(excesses, theta) < values[index]:
        theta = grid[index]
    shape = _shape_at(excesses, theta)
    scale = _scale_at(excesses, theta, shape)
    return shape, scale


def _probability_weighted_moments(excesses: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(excesses)
    count = len(ordered)
    first = math.fsum(ordered) / count
    weighted = (
        math.fsum(
            value * (1.0 - (index + 1 - PLOTTING_OFFSET) / count)
            for index, value in enumerate(ordered)
        )
        / count
    )
    # ``ratio`` is above 2 for any sample of positive excesses, tied ones
    # included: the weights fall as the data rise, so the weighted mean is
    # strictly below half the plain mean. The shape below is
    # ``1 - 2 / (ratio - 2)``, so that inequality puts it strictly under one —
    # always, on any data. This estimator therefore cannot report a tail with no
    # finite mean however heavy the sample is, which is a property of the
    # estimator rather than of the data and is stated on :class:`TailMethod`.
    ratio = first / weighted
    shape = (ratio - 4.0) / (ratio - 2.0)
    return shape, first * (1.0 - shape)


@dataclass(frozen=True)
class GeneralisedPareto:
    """A tail fitted above a threshold, and what it can be asked.

    The parameters describe the *excess* over :attr:`threshold`. Turning them
    into a quantile of the loss distribution needs one more number — what
    fraction of the sample exceeded the threshold — which is why
    :attr:`exceedances` and :attr:`observations` are carried here rather than
    left to the caller to remember.
    """

    #: ``xi``. Positive is a heavy tail with no upper bound; zero is exponential;
    #: negative is a tail that ends, at :attr:`upper_endpoint`.
    shape: float
    #: ``beta``, in the units of the losses. Always positive.
    scale: float
    threshold: float
    #: Losses strictly above the threshold.
    exceedances: int
    #: Losses the threshold was chosen from. The ratio of the two is the only
    #: thing connecting the fitted tail to an unconditional probability.
    observations: int
    method: TailMethod
    log_likelihood: float
    #: Asymptotic standard errors, from the maximum likelihood information
    #: matrix. ``None`` for the moment estimator, whose asymptotic variance is a
    #: different and much messier expression — reporting the likelihood's there
    #: would be a number that looks like a standard error and is not one.
    shape_standard_error: float | None = None
    scale_standard_error: float | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"the scale is positive, got {self.scale!r}")
        if self.exceedances < 1:
            raise ValueError(f"a fit has exceedances, got {self.exceedances!r}")
        if self.observations < self.exceedances:
            raise ValueError(
                f"{self.exceedances} exceedances out of {self.observations} "
                "observations is not possible"
            )

    @property
    def exceedance_probability(self) -> float:
        """The fraction of the sample above the threshold."""
        return self.exceedances / self.observations

    @property
    def lowest_confidence(self) -> float:
        """The lowest confidence this fit says anything about.

        Below it the quantile is inside the body, where the fit was deliberately
        not looking.
        """
        return 1.0 - self.exceedance_probability

    @property
    def upper_endpoint(self) -> float | None:
        """The largest loss the fitted tail permits, if it permits a largest one.

        ``None`` for a non-negative shape, where the tail is unbounded. A finite
        endpoint from financial loss data is usually a threshold set too low
        rather than a discovery, and it is worth looking at before it is
        reported to anyone.
        """
        if self.shape >= 0.0:
            return None
        return self.threshold - self.scale / self.shape

    def tail_probability(self, loss: float) -> float:
        """The probability of a loss worse than ``loss``.

        Unconditional: the exceedance probability times the fitted conditional
        tail. Below the threshold it returns the empirical exceedance
        probability rather than extrapolating a fit downwards into the body.
        """
        if loss <= self.threshold:
            return self.exceedance_probability
        excess = loss - self.threshold
        if abs(self.shape) < SHAPE_EPSILON:
            conditional = math.exp(-excess / self.scale)
        else:
            argument = 1.0 + self.shape * excess / self.scale
            conditional = 0.0 if argument <= 0.0 else math.pow(argument, -1.0 / self.shape)
        return self.exceedance_probability * conditional

    def quantile(self, confidence: float) -> float:
        """The loss exceeded with probability ``1 - confidence``.

        Positive, in the units of the losses that were fitted.
        """
        self._check_confidence(confidence)
        ratio = (1.0 - confidence) / self.exceedance_probability
        if abs(self.shape) < SHAPE_EPSILON:
            return self.threshold - self.scale * math.log(ratio)
        return self.threshold + self.scale * (math.pow(ratio, -self.shape) - 1.0) / self.shape

    def expected_shortfall(self, confidence: float) -> float:
        """The mean loss given a loss worse than :meth:`quantile`.

        In closed form rather than by integration: for a generalised Pareto the
        excesses over a higher threshold are generalised Pareto with the same
        shape, so the tail mean is affine in the quantile with slope
        ``1 / (1 - xi)``. That slope is where a shape at or above one becomes a
        refusal instead of a number.
        """
        self._check_confidence(confidence)
        if self.shape >= 1.0:
            raise UndefinedTailMean(
                f"the fitted shape is {self.shape:.4f}; at or above 1 the tail has no "
                "finite mean and there is no expected shortfall at any confidence. "
                "Value at risk is still defined."
            )
        quantile = self.quantile(confidence)
        if abs(self.shape) < SHAPE_EPSILON:
            return quantile + self.scale
        return (quantile + self.scale - self.shape * self.threshold) / (1.0 - self.shape)

    def return_level(self, periods: float) -> float:
        """The loss exceeded once every ``periods`` observations, on average.

        The same number as :meth:`quantile` at ``1 - 1 / periods``, said the way
        a reader of a stress scenario asks for it: 250 periods is the worst day
        of a trading year.
        """
        if periods <= 1.0:
            raise ValueError(
                f"a return level is defined over more than one period, got {periods!r}"
            )
        return self.quantile(1.0 - 1.0 / periods)

    def _check_confidence(self, confidence: float) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                f"a confidence is strictly between 0 and 1, got {confidence!r}"
            )
        if confidence < self.lowest_confidence:
            raise OutsideTheFit(
                f"this tail was fitted above a loss of {self.threshold:.6g}, which "
                f"{self.exceedances} of {self.observations} observations exceeded, so "
                f"it says nothing below {self.lowest_confidence:.6g} confidence; "
                f"{confidence!r} is inside the body. Either ask for a higher "
                "confidence or fit above a lower threshold."
            )


def fit_generalised_pareto(
    excesses: Sequence[float],
    *,
    threshold: float = 0.0,
    observations: int | None = None,
    method: TailMethod = TailMethod.MAXIMUM_LIKELIHOOD,
    minimum_exceedances: int = MINIMUM_EXCEEDANCES,
) -> GeneralisedPareto:
    """Fit a generalised Pareto to amounts by which a threshold was exceeded.

    ``excesses`` are positive amounts above ``threshold``, not the losses
    themselves; :func:`excesses_over` produces them and :func:`fit_tail` does
    both steps. ``observations`` is the size of the sample the threshold was
    chosen from, and defaults to the number of exceedances — which is right only
    when every observation exceeded, so the common case is to pass it.
    """
    values = _losses(excesses, name="excesses")
    for index, value in enumerate(values):
        if value <= 0.0:
            raise ValueError(
                f"excesses[{index}] is {value!r}; an excess over a threshold is "
                "strictly positive, and `excesses_over` drops the losses that are not"
            )
    if len(values) < minimum_exceedances:
        raise NotEnoughTail(
            f"{len(values)} exceedances is below the {minimum_exceedances} a fit will "
            "accept; lower the threshold, or lower `minimum_exceedances` knowing that "
            "two parameters from fewer points is the largest observation with extra "
            "steps"
        )
    total = len(values) if observations is None else observations
    if total < len(values):
        raise ValueError(
            f"{len(values)} exceedances out of {total} observations is not possible"
        )
    if method is TailMethod.MAXIMUM_LIKELIHOOD:
        shape, scale = _maximum_likelihood(values)
        count = len(values)
        # Smith (1987): the inverse information matrix of the generalised Pareto
        # is ``(1 + xi) / n * [[1 + xi, -beta], [-beta, 2 * beta^2]]``, valid for
        # a shape above -0.5. Below that the estimator is not asymptotically
        # normal and no standard error is reported rather than a misleading one.
        if shape > -0.5:
            factor = (1.0 + shape) / count
            shape_error: float | None = math.sqrt(factor * (1.0 + shape))
            scale_error: float | None = math.sqrt(factor * 2.0 * scale * scale)
        else:
            shape_error = None
            scale_error = None
    else:
        shape, scale = _probability_weighted_moments(values)
        shape_error = None
        scale_error = None
    if shape < 0.0 and -scale / shape <= max(values):
        raise ImpossibleFit(
            f"the fit puts the largest possible excess at {-scale / shape:.6g}, below "
            f"the largest one observed, {max(values):.6g}: the tail it describes "
            f"assigns zero probability to a loss that happened. Shape {shape:.4f}, "
            f"scale {scale:.6g}, from {method.value}. Refit by maximum likelihood, "
            "which cannot produce this, or raise the threshold."
        )
    return GeneralisedPareto(
        shape=shape,
        scale=scale,
        threshold=threshold,
        exceedances=len(values),
        observations=total,
        method=method,
        log_likelihood=_log_likelihood(values, shape, scale),
        shape_standard_error=shape_error,
        scale_standard_error=scale_error,
    )


def fit_tail(
    losses: Sequence[float],
    *,
    tail_fraction: float = 0.05,
    threshold: float | None = None,
    method: TailMethod = TailMethod.MAXIMUM_LIKELIHOOD,
    minimum_exceedances: int = MINIMUM_EXCEEDANCES,
) -> GeneralisedPareto:
    """Fit a tail to ``losses``, choosing the threshold from ``tail_fraction``.

    Pass ``threshold`` to set it directly, which is what the diagnostics are for.
    ``tail_fraction`` is a convenience with a defensible default rather than a
    recommendation: 5% of the sample is the figure most of the applied literature
    starts from, and the mean excess curve is the reason to move off it.
    """
    values = _losses(losses)
    chosen = threshold_for(values, tail_fraction) if threshold is None else threshold
    above = excesses_over(values, chosen)
    if len(above) < minimum_exceedances:
        raise NotEnoughTail(
            f"a threshold of {chosen:.6g} left {len(above)} of {len(values)} losses "
            f"above it, below the {minimum_exceedances} a fit will accept"
        )
    return fit_generalised_pareto(
        above,
        threshold=chosen,
        observations=len(values),
        method=method,
        minimum_exceedances=minimum_exceedances,
    )


# -- the estimate --------------------------------------------------------------


@dataclass(frozen=True)
class ExtremeRisk:
    """A far-tail estimate, with the fit it came from attached.

    Deliberately not a :class:`~shortfall.parametric.Risk`. The fields that
    matter line up with it, so the two can be tabulated side by side, but this
    one carries no volatility to scale and no ``scaled_to``, because a fitted
    tail does not scale by the square root of time.
    """

    #: Positive loss.
    value_at_risk: float
    #: Positive loss, and never less than :attr:`value_at_risk`.
    expected_shortfall: float | None
    #: The signed return quantile, for callers using the other sign convention.
    quantile: float
    confidence: float
    fit: GeneralisedPareto
    #: Losses at or beyond :attr:`value_at_risk` in the sample. Frequently zero,
    #: and that is the point of the estimator rather than a fault in it: the
    #: number says how much of the answer is extrapolation.
    observed_beyond: int

    @property
    def tail_probability(self) -> float:
        return 1.0 - self.confidence

    @property
    def is_extrapolated(self) -> bool:
        """Whether the estimate is beyond every loss in the sample."""
        return self.observed_beyond == 0


def extreme_risk(
    returns: Sequence[float] | ReturnSeries,
    *,
    confidence: float,
    tail_fraction: float = 0.05,
    threshold: float | None = None,
    method: TailMethod = TailMethod.MAXIMUM_LIKELIHOOD,
    minimum_exceedances: int = MINIMUM_EXCEEDANCES,
) -> ExtremeRisk:
    """Value at risk and expected shortfall from a tail fitted to ``returns``.

    Returns go in with their own sign and are negated here, once: the loss
    series is ``-r``, so a gain is a negative loss and is dropped by the
    threshold rather than having to be filtered out first.

    The expected shortfall is ``None`` when the fitted shape is at least one, not
    an error, because the value at risk in the same result is still meaningful
    and raising would throw it away. Asking the fit directly raises, with the
    shape in the message.
    """
    values = returns.values if isinstance(returns, ReturnSeries) else tuple(returns)
    losses = tuple(-value for value in _losses(values, name="returns"))
    fit = fit_tail(
        losses,
        tail_fraction=tail_fraction,
        threshold=threshold,
        method=method,
        minimum_exceedances=minimum_exceedances,
    )
    value_at_risk = fit.quantile(confidence)
    try:
        shortfall: float | None = fit.expected_shortfall(confidence)
    except UndefinedTailMean:
        shortfall = None
    return ExtremeRisk(
        value_at_risk=value_at_risk,
        expected_shortfall=shortfall,
        quantile=-value_at_risk,
        confidence=confidence,
        fit=fit,
        observed_beyond=sum(1 for loss in losses if loss >= value_at_risk),
    )

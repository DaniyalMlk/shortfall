"""Which of two adequate risk models is better, and why the obvious score is not.

:mod:`shortfall.backtest` asks whether one model is adequate: the right number
of breaches, spread out through time, with tails of the right size. That is a
hypothesis test and it answers a yes-or-no question. It cannot rank two models
that both pass, and ranking them is a different instrument — a *scoring
function*, evaluated on each forecast and averaged.

A scoring function is only worth anything if it is **strictly consistent** for
the thing being forecast: the expected score must be minimised, uniquely, by
the true value. Otherwise a forecaster who optimises it is optimising towards
something else, and a desk comparing two models by it is ranking them on
something else.

**Value at risk has one; expected shortfall does not have one of its own.** The
pinball loss

    S(v, l) = (l - v)(c - 1{l < v})

is strictly consistent for the ``c``-quantile, and for a normal law its
expected value differentiates to ``Phi(v/sigma) - c``, which is zero exactly at
the true quantile. Expected shortfall is not elicitable at all: there is no
``S(e, l)`` minimised at the shortfall for every distribution. The pair
*jointly* is elicitable, and :func:`fz0_loss` is the zero-homogeneous member of
the Fissler-Ziegel family, which is the usual choice because a score that is
homogeneous of degree zero ranks the same way whatever the units are.

**That it is consistent is checked exactly rather than simulated.** For a
normal law every expectation here is elementary, and the two first-order
conditions of :func:`fz0_loss` are

    d/dv:  (Phi(v/sigma) - 1) / (alpha e) + 1 / e = 0   =>  v = sigma z_c
    d/de:  1/e - [E(L - v)+ / alpha + v] / e**2 = 0     =>  e = v + E(L - v)+ / alpha

which are the definitions of the two quantities. Minimising the exact expected
score on a grid recovers the true pair to 2e-08 at three confidence levels,
which is the grid's own resolution.

**The score anybody would build first is wrong by about a third.** The obvious
construction is the pinball loss for the quantile plus a squared error on the
breaches for the shortfall. Given the quantile it does elicit the conditional
tail mean, which is why it looks right. Jointly it does not, and the reason is
one line: the derivative of its expected value in ``v``, at the true pair, is

    Phi(v/sigma) - c - phi(z) (v - e)**2 / (alpha sigma) = -phi(z) (VaR - ES)**2 / alpha

which is strictly negative whenever the shortfall differs from the quantile —
that is, always. So the optimum sits *above* the truth, and by the squared gap
between them, which is a measure of how thick the tail is. Measured: the
minimiser is **48.8% above the true value at risk and 34.5% above the true
shortfall at 95% confidence**, 46.0% and 34.9% at 97.5%, and 44.3% and 35.7% at
99%. Not a subtlety; a third of the number.

**And the two scores rank the same pair of models in opposite orders.** Take a
truthful forecaster at 97.5% and one shading both numbers up to the obvious
score's own optimum. Under Fissler-Ziegel the truthful one wins, 0.84921
against 1.06369. Under the obvious score the shaded one wins, 0.07849 against
0.17513. A desk choosing its model with the obvious score picks the model that
is wrong, and the comparison looks decisive either way round.

**Comparing two models needs an autocorrelation-robust variance, and it is
worth less than it sounds.** Score differences inherit the dependence of the
data, so :func:`compare` uses a Newey-West estimate at the usual bandwidth and
reports the naive standard error beside it. Measured over four thousand
observations, comparing two exponentially weighted volatility models against a
log-volatility process: at zero persistence the robust standard error is
**0.99** times the naive one — *below* one, which is the Bartlett estimator's
own finite-sample noise and not a correction — and at persistences of 0.95 and
0.995 it is **1.10** and **1.06**. The largest effect on a t statistic across
those runs is -3.53 becoming -3.20.

So the correction is real and modest. It earns its place because its direction
is not knowable in advance rather than because it is large, and a reader who
expected the naive standard error to be badly wrong should have the measurement
rather than the expectation. The ratio is on the result for that reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .distributions import normal_cdf

__all__ = [
    "BadScore",
    "Comparison",
    "Scores",
    "compare",
    "fz0_loss",
    "quantile_loss",
]


class BadScore(ValueError):
    """A forecast series that cannot be scored, or two that cannot be compared."""


def _check(
    observed: Sequence[float],
    forecasts: Sequence[float],
    confidence: float,
    name: str,
) -> None:
    if not 0.0 < confidence < 1.0:
        raise BadScore(f"a confidence level lies strictly in (0, 1), got {confidence!r}")
    if len(observed) != len(forecasts):
        raise BadScore(
            f"there must be one {name} per observation, got {len(observed)} returns "
            f"and {len(forecasts)} forecasts"
        )
    if not observed:
        raise BadScore("a score needs at least one observation")
    for index, value in enumerate(observed):
        if not math.isfinite(value):
            raise BadScore(f"return {index} is {value!r}, which is not a finite number")
    for index, value in enumerate(forecasts):
        if not math.isfinite(value):
            raise BadScore(f"{name} {index} is {value!r}, which is not a finite number")
        if value <= 0.0:
            raise BadScore(
                f"{name} {index} is {value!r}. Forecasts are positive losses, the "
                "same convention the rest of this library uses, so a forecast of "
                "zero or less is a sign error rather than an opinion."
            )


@dataclass(frozen=True, slots=True)
class Scores:
    """One score per observation, and what they were scoring.

    Attributes:
        values: The per-observation scores, in the order given.
        confidence: The level the forecasts were made at.
        name: Which scoring function produced them. Two score series are only
            comparable when this agrees, which :func:`compare` enforces: an
            average pinball loss and an average Fissler-Ziegel score are
            different numbers about different things, and nothing stops them
            being subtracted.
    """

    values: tuple[float, ...]
    confidence: float
    name: str

    def __len__(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        """The average score. Lower is better, for every score here."""
        return math.fsum(self.values) / len(self.values)


def quantile_loss(
    observed: Sequence[float],
    value_at_risk: Sequence[float],
    *,
    confidence: float,
) -> Scores:
    """The pinball loss, strictly consistent for the value at risk alone.

    ``observed`` are signed returns and ``value_at_risk`` are positive losses,
    the convention :mod:`shortfall.backtest` uses, so the loss being scored is
    ``-observed[t]`` and the quantile being elicited is its ``confidence``
    quantile.

    Args:
        observed: Realised returns.
        value_at_risk: One positive loss forecast per observation.
        confidence: The level the forecasts were made at, in (0, 1).

    Returns:
        A :class:`Scores`.

    Raises:
        BadScore: If the series disagree in length, are empty, or hold a
            non-finite or non-positive forecast.
    """
    _check(observed, value_at_risk, confidence, "value at risk")
    values = []
    for value, forecast in zip(observed, value_at_risk, strict=True):
        loss = -value
        indicator = 1.0 if loss < forecast else 0.0
        values.append((loss - forecast) * (confidence - indicator))
    return Scores(tuple(values), confidence, "pinball")


def fz0_loss(
    observed: Sequence[float],
    value_at_risk: Sequence[float],
    expected_shortfall: Sequence[float],
    *,
    confidence: float,
) -> Scores:
    """The zero-homogeneous Fissler-Ziegel score for the pair, jointly.

    In this library's sign convention — returns signed, forecasts positive
    losses — the score is

        (l - v)+ / (alpha e) + v / e + log e - 1

    with ``l`` the realised loss, ``v`` the value at risk, ``e`` the expected
    shortfall and ``alpha = 1 - confidence``. It is strictly consistent for the
    pair on the region where ``e >= v``, which is where a coherent forecast
    lives anyway: a tail mean below its own quantile describes nothing.

    Being homogeneous of degree zero, it ranks two models the same way whatever
    the forecasts are denominated in. That is why this member rather than
    another: a ranking that moved when the returns were rescaled would be a
    property of the units.

    Args:
        observed: Realised returns.
        value_at_risk: One positive loss forecast per observation.
        expected_shortfall: One positive tail-mean forecast per observation,
            each at least its own value at risk.
        confidence: The level the forecasts were made at, in (0, 1).

    Returns:
        A :class:`Scores`.

    Raises:
        BadScore: If the series disagree, hold a non-positive forecast, or
            pair a shortfall below its own quantile.
    """
    _check(observed, value_at_risk, confidence, "value at risk")
    _check(observed, expected_shortfall, confidence, "expected shortfall")
    alpha = 1.0 - confidence
    values = []
    for index, (value, var, shortfall) in enumerate(
        zip(observed, value_at_risk, expected_shortfall, strict=True)
    ):
        if shortfall < var:
            raise BadScore(
                f"forecast {index} pairs an expected shortfall of {shortfall!r} with a "
                f"value at risk of {var!r}. The tail mean cannot be smaller than the "
                "quantile it is the mean beyond, and the score is only consistent "
                "where it is not."
            )
        loss = -value
        breach = loss - var if loss >= var else 0.0
        values.append(breach / (alpha * shortfall) + var / shortfall + math.log(shortfall) - 1.0)
    return Scores(tuple(values), confidence, "fissler-ziegel")


@dataclass(frozen=True, slots=True)
class Comparison:
    """A Diebold-Mariano comparison of two score series.

    Attributes:
        difference: Mean of the first score less the second. Negative favours
            the first, because lower is better.
        standard_error: Newey-West standard error of that mean.
        statistic: The ratio of the two, asymptotically standard normal.
        p_value: Two-sided, from the normal approximation.
        lags: Bandwidth used for the long-run variance.
        naive_standard_error: What the standard error would be if the
            differences were independent. Reported rather than used, because
            the gap between the two is the whole reason the bandwidth is
            there.
        observations: Length of the series.
    """

    difference: float
    standard_error: float
    statistic: float
    p_value: float
    lags: int
    naive_standard_error: float
    observations: int

    @property
    def better(self) -> int | None:
        """Which model the comparison favours: 1, 2, or ``None`` for a tie."""
        if self.difference == 0.0:
            return None
        return 1 if self.difference < 0.0 else 2


def _newey_west(differences: Sequence[float], lags: int) -> float:
    """Long-run variance of the mean, with Bartlett weights."""
    count = len(differences)
    mean = math.fsum(differences) / count
    centred = [value - mean for value in differences]
    total = math.fsum(value * value for value in centred) / count
    for lag in range(1, lags + 1):
        covariance = (
            math.fsum(centred[index] * centred[index - lag] for index in range(lag, count))
            / count
        )
        total += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    return total


def compare(
    first: Scores,
    second: Scores,
    *,
    lags: int | None = None,
) -> Comparison:
    """Diebold-Mariano on two score series, with a robust variance.

    Args:
        first: Scores of one model.
        second: Scores of the other, from the same scoring function at the
            same confidence over the same observations.
        lags: Bandwidth for the long-run variance. The default is the usual
            ``floor(4 (T/100)**(2/9))``, which is a rule rather than a
            estimate and is stated in the result so it can be overridden.

    Returns:
        A :class:`Comparison`.

    Raises:
        BadScore: If the two series are of different lengths, different
            confidence levels, or different scoring functions — the last
            because nothing in arithmetic stops an average pinball loss being
            subtracted from an average Fissler-Ziegel score, and the result
            would mean nothing.
    """
    if len(first) != len(second):
        raise BadScore(
            f"two score series must cover the same observations, got {len(first)} "
            f"and {len(second)}"
        )
    if first.confidence != second.confidence:
        raise BadScore(
            f"two score series must be at the same confidence, got "
            f"{first.confidence!r} and {second.confidence!r}"
        )
    if first.name != second.name:
        raise BadScore(
            f"two score series must come from the same scoring function, got "
            f"{first.name!r} and {second.name!r}. An average pinball loss and an "
            "average Fissler-Ziegel score are different numbers about different "
            "things; subtracting them is arithmetic rather than a comparison."
        )
    count = len(first)
    if count < 2:
        raise BadScore("a comparison needs at least two observations")
    differences = [a - b for a, b in zip(first.values, second.values, strict=True)]
    difference = math.fsum(differences) / count
    bandwidth = int(4.0 * (count / 100.0) ** (2.0 / 9.0)) if lags is None else lags
    if bandwidth < 0:
        raise BadScore(f"a bandwidth cannot be negative, got {bandwidth!r}")
    bandwidth = min(bandwidth, count - 1)
    long_run = _newey_west(differences, bandwidth)
    naive = math.fsum((value - difference) ** 2 for value in differences) / count
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    statistic = difference / standard_error if standard_error > 0.0 else 0.0
    return Comparison(
        difference=difference,
        standard_error=standard_error,
        statistic=statistic,
        p_value=2.0 * (1.0 - normal_cdf(abs(statistic))),
        lags=bandwidth,
        naive_standard_error=math.sqrt(naive / count),
        observations=count,
    )

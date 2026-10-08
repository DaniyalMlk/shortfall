"""Saddlepoint tail probabilities, measured against an exact convolution.

Three references, none of which shares anything with the approximation. The
convolution on the exposure lattice is exact and uses no tilt, no expansion and
no normal distribution function. For a homogeneous portfolio that convolution
can itself be checked against
:func:`shortfall.distributions.binomial_sf`, which was written for something
else entirely. And two identities are exact rather than approximate: the
contributions at a zero level sum to the mean, and the contributions at any
level sum to the numerator of the conditional mean.

The rest is measurement. How much better than a normal approximation, how much
the lattice correction is worth, how wide the neighbourhood of the mean has to
be, and where the approximation stops being a distribution at all.
"""

from __future__ import annotations

import math
import random

import pytest

from shortfall.distributions import binomial_sf, normal_cdf, normal_pdf
from shortfall.saddlepoint import (
    SADDLE_FLOOR,
    SPAN_DENOMINATOR,
    Exceedance,
    LossPortfolio,
    Obligor,
    SaddlepointError,
    exact_distribution,
    exact_shortfall,
    exact_tail,
    exceedance_probability,
    lattice_span,
    normal_tail,
    saddlepoint,
    saddlepoint_shortfall,
    shortfall_contributions,
    tail_probability,
)


def heterogeneous() -> LossPortfolio:
    """A hundred names, the portfolio every figure in the docstring is from."""
    generator = random.Random(20261008)
    return LossPortfolio.detected(
        tuple(
            Obligor(
                round(0.004 + 0.075 * generator.random(), 6),
                float(generator.randint(1, 20)),
            )
            for _ in range(100)
        )
    )


@pytest.fixture(scope="module")
def portfolio() -> LossPortfolio:
    return heterogeneous()


@pytest.fixture(scope="module")
def masses(portfolio: LossPortfolio) -> tuple[float, ...]:
    return exact_distribution(portfolio)


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize("probability", [-0.01, 1.01, 2.0])
def test_a_probability_outside_the_unit_interval_is_rejected(probability: float) -> None:
    with pytest.raises(SaddlepointError, match="not one"):
        Obligor(probability, 1.0)


@pytest.mark.parametrize("exposure", [0.0, -1.0, math.inf, math.nan])
def test_an_exposure_that_is_not_one_is_rejected(exposure: float) -> None:
    with pytest.raises(SaddlepointError, match="not one"):
        Obligor(0.1, exposure)


def test_an_empty_portfolio_is_rejected() -> None:
    with pytest.raises(SaddlepointError, match="at least one"):
        LossPortfolio(())


def test_a_portfolio_that_cannot_lose_anything_is_rejected() -> None:
    """Not an empty portfolio, and it has no tail either.

    Worth its own refusal because the obligors are all perfectly well formed
    and the maximum loss is positive if a zero-probability name is allowed to
    raise it -- which is why :attr:`LossPortfolio.maximum` skips them.
    """
    with pytest.raises(SaddlepointError, match="identically zero"):
        LossPortfolio.detected((Obligor(0.0, 5.0), Obligor(0.0, 3.0)))


def test_the_maximum_ignores_names_that_cannot_default() -> None:
    portfolio = LossPortfolio.detected((Obligor(0.1, 4.0), Obligor(0.0, 1000.0)))
    assert portfolio.maximum == 4.0


# -- the lattice --------------------------------------------------------------


@pytest.mark.parametrize(
    ("exposures", "span"),
    [
        ((1.0, 1.0, 1.0), 1.0),
        ((2.0, 4.0, 6.0), 2.0),
        ((2.0, 3.0), 1.0),
        ((0.25, 0.1), 0.05),
        ((5.0, 7.5), 2.5),
        ((1.0, math.pi), 0.0),
        ((1.0, 1.0 / 2048.0), 0.0),
    ],
)
def test_the_lattice_span_is_found_exactly(exposures: tuple[float, ...], span: float) -> None:
    """Cleared with rational arithmetic, so 0.25 and 0.1 give 0.05 and not nearly it."""
    obligors = tuple(Obligor(0.1, exposure) for exposure in exposures)
    assert lattice_span(obligors) == span


def test_an_empty_sequence_has_no_span() -> None:
    assert lattice_span(()) == 0.0


def test_the_lattice_question_needs_a_resolution_to_be_worth_asking() -> None:
    """Every double is a dyadic rational, so asked exactly the answer is always yes.

    An exposure of pi is a whole multiple of a power of two, and reporting
    that as a lattice span of 1e-16 would be true and would make the midpoint
    correction meaningless. The denominator ceiling is what makes the question
    about the market rather than about floating point.
    """
    assert SPAN_DENOMINATOR == 1024
    assert lattice_span((Obligor(0.1, 1.0), Obligor(0.1, math.pi))) == 0.0
    assert lattice_span((Obligor(0.1, 1.0), Obligor(0.1, 1.0 / 1024.0))) == pytest.approx(
        1.0 / 1024.0
    )
    assert lattice_span((Obligor(0.1, 1.0), Obligor(0.1, 1.0 / 2048.0))) == 0.0


def test_a_sub_portfolio_keeps_the_parent_lattice() -> None:
    """Dropping a name can coarsen the common unit, and that would be wrong.

    The level a contribution asks about is the parent's level less one
    exposure, which sits on the parent's lattice. Re-detecting the span would
    put the midpoint correction half a step of the *wrong* lattice away.
    """
    portfolio = LossPortfolio.detected((Obligor(0.1, 2.0), Obligor(0.1, 3.0)))
    assert portfolio.span == 1.0
    assert portfolio.without(1).span == 1.0
    assert lattice_span(portfolio.without(1).obligors) == 2.0


# -- the cumulant generating function -----------------------------------------


@pytest.mark.parametrize("tilt", [-1.5, -0.3, 0.0, 0.2, 0.8, 2.0])
def test_the_derivatives_are_the_derivatives(tilt: float, portfolio: LossPortfolio) -> None:
    """Written out obligor by obligor, so checked against finite differences."""
    step = 1e-6
    assert portfolio.derivative(tilt) == pytest.approx(
        (portfolio.cumulant(tilt + step) - portfolio.cumulant(tilt - step)) / (2.0 * step),
        rel=1e-6,
    )
    assert portfolio.second_derivative(tilt) == pytest.approx(
        (portfolio.derivative(tilt + step) - portfolio.derivative(tilt - step)) / (2.0 * step),
        rel=1e-6,
    )
    assert portfolio.third_derivative(tilt) == pytest.approx(
        (portfolio.second_derivative(tilt + step) - portfolio.second_derivative(tilt - step))
        / (2.0 * step),
        rel=1e-5,
    )


def test_the_cumulant_at_zero_gives_the_first_two_moments(
    portfolio: LossPortfolio,
) -> None:
    assert portfolio.cumulant(0.0) == 0.0
    assert portfolio.derivative(0.0) == pytest.approx(portfolio.mean, rel=1e-14)
    assert portfolio.second_derivative(0.0) == pytest.approx(portfolio.variance, rel=1e-14)


def test_a_large_tilt_saturates_instead_of_overflowing(
    portfolio: LossPortfolio,
) -> None:
    """The tilted probability is written as a logistic for exactly this.

    ``p e^{te} / (1 + p(e^{te} - 1))`` overflows on the way to its own limit of
    one, which a saddlepoint search deep in the tail reaches routinely.
    """
    for tilt in (50.0, 500.0, 5000.0):
        assert portfolio.derivative(tilt) == pytest.approx(portfolio.maximum)
        assert portfolio.second_derivative(tilt) == pytest.approx(0.0, abs=1e-9)
    for tilt in (-50.0, -500.0, -5000.0):
        assert portfolio.derivative(tilt) == pytest.approx(0.0, abs=1e-9)


# -- the saddlepoint ----------------------------------------------------------


@pytest.mark.parametrize("level", [1.0, 10.0, 35.0, 90.0, 300.0, 900.0])
def test_the_saddlepoint_solves_its_own_equation(level: float, portfolio: LossPortfolio) -> None:
    tilt = saddlepoint(portfolio, level)
    assert portfolio.derivative(tilt) == pytest.approx(level, rel=1e-12)


def test_the_saddlepoint_is_increasing_in_the_level(portfolio: LossPortfolio) -> None:
    """Because K'' is a variance, so K' is increasing and the root is unique."""
    tilts = [saddlepoint(portfolio, level) for level in range(1, 900, 37)]
    assert tilts == sorted(tilts)


def test_the_saddlepoint_is_zero_at_the_mean(portfolio: LossPortfolio) -> None:
    assert abs(saddlepoint(portfolio, portfolio.mean)) < 1e-12


@pytest.mark.parametrize("level", [-1.0, 0.0, 948.0, 1000.0])
def test_a_level_outside_the_support_is_refused(level: float, portfolio: LossPortfolio) -> None:
    with pytest.raises(SaddlepointError, match="outside the support"):
        saddlepoint(portfolio, level)


# -- the exact reference, checked against something else again ----------------


def test_the_convolution_is_a_distribution(portfolio: LossPortfolio) -> None:
    masses = exact_distribution(portfolio)
    assert len(masses) == int(portfolio.maximum) + 1
    assert sum(masses) == pytest.approx(1.0, rel=1e-12)
    assert all(mass >= 0.0 for mass in masses)
    mean = sum(index * mass for index, mass in enumerate(masses))
    assert mean == pytest.approx(portfolio.mean, rel=1e-10)
    variance = sum((index - mean) ** 2 * mass for index, mass in enumerate(masses))
    assert variance == pytest.approx(portfolio.variance, rel=1e-10)


@pytest.mark.parametrize("probability", [0.05, 0.2, 0.5])
def test_the_convolution_reproduces_the_binomial_survival_function(
    probability: float,
) -> None:
    """A reference written for something else entirely, in another module.

    The homogeneous case is binomial, and ``distributions.binomial_sf`` has
    nothing to do with convolution or with saddlepoints. **It is also
    ``P(X >= k)`` where this module's tail is ``P(L > x)``**, so the comparison
    carries a shift; asserted the other way the two disagree by 0.10 at the
    median, which looks like a catastrophic accuracy failure and is a
    convention.
    """
    portfolio = LossPortfolio.homogeneous(60, probability, 1.0)
    masses = exact_distribution(portfolio)
    for successes in (1, 2, 6, 13, 31, 56):
        assert exact_tail(masses, float(successes - 1)) == pytest.approx(
            binomial_sf(successes, 60, probability), rel=1e-11, abs=1e-18
        )


def test_the_convolution_refuses_exposures_off_its_lattice() -> None:
    portfolio = LossPortfolio.detected((Obligor(0.1, 1.0), Obligor(0.1, math.pi)))
    with pytest.raises(SaddlepointError, match="whole units only"):
        exact_distribution(portfolio)


def test_a_non_positive_lattice_spacing_is_refused(portfolio: LossPortfolio) -> None:
    with pytest.raises(SaddlepointError, match="not one"):
        exact_distribution(portfolio, unit=0.0)


def test_the_convolution_works_on_a_fractional_lattice() -> None:
    portfolio = LossPortfolio.detected((Obligor(0.3, 0.25), Obligor(0.4, 0.5)))
    masses = exact_distribution(portfolio, unit=0.25)
    assert sum(masses) == pytest.approx(1.0)
    assert masses[0] == pytest.approx(0.7 * 0.6)
    assert masses[3] == pytest.approx(0.3 * 0.4)


# -- how good it is -----------------------------------------------------------


def test_the_error_shrinks_into_the_tail_where_a_normal_grows_to_everything(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    """The whole distinction between expanding at the point and at the centre.

    If the correction entered with the wrong sign the saddlepoint column would
    read 14% high at the mean rising to 65% in the tail -- still monotone,
    still plausible, and with exactly the shape of failure this method exists
    to avoid. Only the exact reference tells the two apart.
    """
    targets = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    expected = (3.3e-04, 2.5e-04, 2.0e-04, 1.6e-04, 1.1e-04)
    worst_normal = 0.0
    for target, predicted in zip(targets, expected, strict=True):
        level = next(
            value
            for value in range(int(portfolio.mean), int(portfolio.maximum))
            if exact_tail(masses, float(value)) <= target
        )
        exact = exact_tail(masses, float(level))
        approximate = tail_probability(portfolio, float(level)).probability
        relative = (approximate - exact) / exact
        assert relative == pytest.approx(predicted, rel=0.2)
        assert relative > 0.0
        normal = (normal_tail(portfolio, float(level)) - exact) / exact
        assert normal < -0.6
        worst_normal = min(worst_normal, normal)
    assert worst_normal < -0.999
    assert expected[-1] < expected[0]


def test_the_normal_approximation_fails_near_the_mean_not_in_the_tail(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    """So it is not a deep-tail caveat; it is wrong where anybody looks."""
    first = next(
        value
        for value in range(int(portfolio.mean), int(portfolio.maximum))
        if exact_tail(masses, float(value)) > 0.0
        and abs(normal_tail(portfolio, float(value)) - exact_tail(masses, float(value)))
        / exact_tail(masses, float(value))
        > 0.10
    )
    assert exact_tail(masses, float(first)) == pytest.approx(0.46, abs=0.03)


def test_the_approximation_is_worst_in_the_body(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    worst, where = 0.0, 0.0
    for level in range(int(portfolio.maximum)):
        exact = exact_tail(masses, float(level))
        if not 1e-12 < exact < 0.999:
            continue
        error = abs(tail_probability(portfolio, float(level)).probability - exact) / exact
        if error > worst:
            worst, where = error, exact
    assert worst == pytest.approx(5.9e-03, rel=0.2)
    assert where > 0.9


def test_the_lattice_correction_is_worth_two_orders_of_magnitude(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    """And the uncorrected error grows into the tail, which is the giveaway.

    A sum of integer exposures has no density, so the continuous form is not
    merely less accurate here -- it fails in the direction the saddlepoint is
    supposed to fix.
    """
    continuous = LossPortfolio(portfolio.obligors, 0.0)
    uncorrected = []
    corrected = []
    for target in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        level = float(
            next(
                value
                for value in range(int(portfolio.mean), int(portfolio.maximum))
                if exact_tail(masses, float(value)) <= target
            )
        )
        exact = exact_tail(masses, level)
        uncorrected.append((tail_probability(continuous, level).probability - exact) / exact)
        corrected.append((tail_probability(portfolio, level).probability - exact) / exact)
    assert uncorrected[0] == pytest.approx(4.7e-02, rel=0.2)
    assert uncorrected[-1] == pytest.approx(7.2e-02, rel=0.2)
    assert uncorrected == sorted(uncorrected)
    assert corrected == sorted(corrected, reverse=True)
    assert uncorrected[0] / corrected[0] > 100.0
    assert uncorrected[-1] / corrected[-1] > 500.0


def test_the_lattice_form_tends_to_the_continuous_one_as_the_span_shrinks(
    portfolio: LossPortfolio,
) -> None:
    """``(2/d) sinh(td/2) -> t``, so one form is the limit of the other.

    Checked numerically rather than algebraically, because the midpoint shift
    goes to zero at the same time and the two cancellations have to agree.
    """
    level = 90.0
    reference = tail_probability(LossPortfolio(portfolio.obligors, 0.0), level).raw
    gaps = []
    for span in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        lattice = LossPortfolio(portfolio.obligors, span)
        gaps.append(abs(tail_probability(lattice, level).raw - reference))
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-5
    ratios = [gaps[index] / gaps[index + 1] for index in range(len(gaps) - 2)]
    assert all(ratio > 5.0 for ratio in ratios)


# -- the singularity at the mean ----------------------------------------------


def test_the_generic_form_degrades_before_the_saddlepoint_reaches_zero(
    portfolio: LossPortfolio,
) -> None:
    """Which is why the switch is to a neighbourhood and not to a point.

    ``w`` is a difference of two nearly equal quantities under a square root.
    Its cancellation error grows as the tilt falls while the limiting form's
    own error shrinks linearly, so the two cross -- and the measurement finds
    where.
    """
    third = portfolio.third_derivative(0.0)
    variance = portfolio.second_derivative(0.0)
    limit = 0.5 - third / (6.0 * math.sqrt(2.0 * math.pi) * variance**1.5)

    def generic(tilt: float) -> float:
        level = portfolio.derivative(tilt)
        inner = 2.0 * (tilt * level - portfolio.cumulant(tilt))
        w = math.copysign(math.sqrt(max(inner, 0.0)), tilt)
        second = portfolio.second_derivative(tilt)
        span = portfolio.span
        u = (2.0 / span) * math.sinh(0.5 * tilt * span) * math.sqrt(second)
        return 1.0 - normal_cdf(w) - normal_pdf(w) * (1.0 / w - 1.0 / u)

    gaps = {tilt: abs(generic(tilt) - limit) for tilt in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)}
    assert gaps[1e-2] == pytest.approx(8.2e-02, rel=0.2)
    assert gaps[1e-4] == pytest.approx(8.0e-04, rel=0.2)
    assert gaps[1e-6] == pytest.approx(7.6e-06, rel=0.3)
    assert gaps[1e-8] > 100.0 * gaps[1e-6]
    assert gaps[1e-10] > 10.0
    assert SADDLE_FLOOR == 1.0e-6


def test_the_limiting_form_is_used_at_the_mean_and_joins_the_generic_one(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    """The branch fires only where the saddlepoint is genuinely near zero.

    On a lattice that is half a step below the mean rather than at it, because
    the correction solves at ``level + span / 2`` -- and half a step below an
    integer mean is not a lattice point, so the case to check the branch on is
    the continuous treatment. The two forms have to meet at the floor, which
    is a check on the limit that needs no reference distribution at all.
    """
    continuous = LossPortfolio(portfolio.obligors, 0.0)
    at_mean = tail_probability(continuous, continuous.mean)
    assert at_mean.near_mean
    assert abs(at_mean.tilt) < SADDLE_FLOOR
    just_outside = tail_probability(continuous, continuous.derivative(SADDLE_FLOOR * 1.01))
    assert not just_outside.near_mean
    assert just_outside.raw == pytest.approx(at_mean.raw, rel=1e-4)

    # And against the exact lattice law it is off by about half a step's worth
    # of mass, which is the lattice-versus-continuum gap rather than an error
    # in the limiting form.
    exact = exact_tail(masses, portfolio.mean)
    density = masses[int(portfolio.mean)]
    assert abs(at_mean.probability - exact) < 1.0 * density
    assert abs(at_mean.probability - exact) > 0.1 * density


def test_a_lattice_level_off_the_lattice_pays_for_it(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    """Because the midpoint correction is half a step of a stated lattice.

    Asking a lattice portfolio about 35.34 and about 35 is asking the same
    question -- both are ``P(L >= 36)`` -- and only the second gets the
    midpoint right. Measured: 1.3% out against 5.4e-04.
    """
    exact = exact_tail(masses, 35.0)
    assert exact_tail(masses, 35.34) == exact
    on_lattice = tail_probability(portfolio, 35.0).probability
    off_lattice = tail_probability(portfolio, 35.34).probability
    assert abs(on_lattice - exact) / exact < 2e-3
    assert abs(off_lattice - exact) / exact > 1e-2


def test_a_degenerate_second_derivative_falls_back_to_the_limit() -> None:
    """One name with a probability of one has no variance left to tilt."""
    portfolio = LossPortfolio.detected((Obligor(1.0, 1.0), Obligor(0.4, 1.0)))
    result = tail_probability(portfolio, 1.0)
    assert 0.0 <= result.probability <= 1.0


# -- not a distribution -------------------------------------------------------


def test_round_off_is_the_only_thing_wrong_on_a_hundred_names(
    portfolio: LossPortfolio,
) -> None:
    """Nine non-monotone steps out of nine hundred, all below 4e-16.

    Saying "the approximation is not monotone" without this measurement would
    be true and useless. Here it is round-off; the next test is a case where
    it is not.
    """
    raws = [tail_probability(portfolio, float(level)).raw for level in range(948)]
    rises = [
        (raws[index], raws[index + 1])
        for index in range(len(raws) - 1)
        if raws[index + 1] > raws[index] + 1e-18
    ]
    assert len(rises) < 20
    assert all(max(pair) < 1e-15 for pair in rises)
    assert max(raws) < 1.0
    assert min(raws) > -1e-15


def test_two_names_at_four_hundred_to_one_is_not_approximated_at_all(
    portfolio: LossPortfolio,
) -> None:
    """There is no asymptotic regime with two summands, and it shows.

    The raw value leaves ``[0, 1]`` in both directions and the worst relative
    error is 300%, which is why the result carries a flag rather than only a
    number.
    """
    pathological = LossPortfolio.detected((Obligor(0.5, 1.0), Obligor(0.5, 400.0)))
    masses = exact_distribution(pathological)
    raws = [tail_probability(pathological, float(level)).raw for level in range(401)]
    assert max(raws) > 4.0
    assert min(raws) < -3.0
    clamped = [tail_probability(pathological, float(level)) for level in (0.0, 400.0 - 1e-9)]
    assert any(one.clamped for one in clamped)
    assert all(0.0 <= one.probability <= 1.0 for one in clamped)
    worst = max(
        abs(tail_probability(pathological, float(level)).probability - exact) / exact
        for level in range(401)
        if 1e-12 < (exact := exact_tail(masses, float(level))) < 0.999
    )
    assert worst > 1.0


def test_the_exceedance_record_reports_its_own_clamping() -> None:
    inside = Exceedance(level=1.0, probability=0.25, raw=0.25, tilt=0.5, near_mean=False)
    assert not inside.clamped
    outside = Exceedance(level=1.0, probability=1.0, raw=1.3, tilt=0.5, near_mean=False)
    assert outside.clamped


# -- the ends of the support, and the two exact identities --------------------


def test_the_ends_of_the_support_are_exact_not_approximated(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    assert exceedance_probability(portfolio, -5.0) == 1.0
    assert exceedance_probability(portfolio, portfolio.maximum) == 0.0
    assert exceedance_probability(portfolio, portfolio.maximum + 1.0) == 0.0
    survival = 1.0
    for one in portfolio.obligors:
        survival *= 1.0 - one.probability
    assert exceedance_probability(portfolio, 0.0) == pytest.approx(1.0 - survival, rel=1e-14)
    assert exceedance_probability(portfolio, 0.0) == pytest.approx(
        exact_tail(masses, 0.0), rel=1e-12
    )


def test_the_contributions_at_a_zero_level_sum_to_the_mean_exactly(
    portfolio: LossPortfolio,
) -> None:
    """No approximation in it at all, which is what makes it a check.

    ``L`` is non-negative, so ``E[L 1{L>0}] = E[L]``, and every sub-portfolio's
    shifted level is negative -- where the exceedance probability is one by
    definition rather than by expansion. So the decomposition has to return
    the mean to machine precision, and a mistake in the shift or in the
    exposure weighting shows up here rather than inside a tolerance.
    """
    shares = shortfall_contributions(portfolio, 0.0)
    assert len(shares) == len(portfolio.obligors)
    assert sum(shares) == pytest.approx(portfolio.mean, rel=1e-14)
    for share, one in zip(shares, portfolio.obligors, strict=True):
        assert share == pytest.approx(one.mean, rel=1e-14)


def test_the_conditional_mean_is_its_decomposition_over_its_probability(
    portfolio: LossPortfolio,
) -> None:
    for level in (20.0, 40.0, 90.0, 200.0):
        shares = shortfall_contributions(portfolio, level)
        assert saddlepoint_shortfall(portfolio, level) == pytest.approx(
            sum(shares) / exceedance_probability(portfolio, level), rel=1e-14
        )


def test_the_conditional_mean_beats_the_probability_it_divides_by(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    """Because its numerator is an exact decomposition, not a second expansion."""
    for level, predicted in ((40.0, 1.2e-04), (60.0, 1.9e-05), (120.0, 4.3e-05)):
        exact = exact_shortfall(masses, level)
        approximate = saddlepoint_shortfall(portfolio, level)
        assert abs(approximate - exact) / exact == pytest.approx(predicted, rel=0.4)
        assert abs(approximate - exact) / exact < 1e-3


def test_a_contribution_is_never_more_than_the_obligor_can_lose(
    portfolio: LossPortfolio,
) -> None:
    for level in (0.0, 30.0, 100.0):
        for share, one in zip(
            shortfall_contributions(portfolio, level), portfolio.obligors, strict=True
        ):
            assert 0.0 <= share <= one.exposure + 1e-12


def test_a_name_that_cannot_default_contributes_nothing() -> None:
    portfolio = LossPortfolio.detected((Obligor(0.2, 5.0), Obligor(0.0, 5.0), Obligor(0.3, 5.0)))
    assert shortfall_contributions(portfolio, 0.0)[1] == 0.0


def test_a_single_name_portfolio_has_no_sub_portfolio_to_recurse_into() -> None:
    """The branch exists because LossPortfolio refuses to be empty."""
    portfolio = LossPortfolio.detected((Obligor(0.25, 8.0),))
    assert shortfall_contributions(portfolio, 0.0) == (pytest.approx(2.0),)
    assert shortfall_contributions(portfolio, 8.0 - 1e-9)[0] == pytest.approx(2.0)
    assert saddlepoint_shortfall(portfolio, 0.0) == pytest.approx(8.0, rel=1e-12)


@pytest.mark.parametrize("level", [-1.0, 948.0, 1000.0])
def test_contributions_outside_the_support_are_refused(
    level: float, portfolio: LossPortfolio
) -> None:
    with pytest.raises(SaddlepointError, match="nothing above the level"):
        shortfall_contributions(portfolio, level)


def test_a_conditional_mean_with_nothing_above_it_is_refused(
    portfolio: LossPortfolio, masses: tuple[float, ...]
) -> None:
    with pytest.raises(SaddlepointError, match="divides by it"):
        saddlepoint_shortfall(portfolio, portfolio.maximum)
    with pytest.raises(SaddlepointError, match="no mean above it"):
        exact_shortfall(masses, portfolio.maximum)


def test_the_normal_tail_handles_a_portfolio_with_no_variance() -> None:
    portfolio = LossPortfolio.detected((Obligor(1.0, 3.0),))
    assert normal_tail(portfolio, 2.0) == 1.0
    assert normal_tail(portfolio, 4.0) == 0.0

"""The README's known-gaps list defends itself.

A list of shortcomings is the first thing in a document to go stale, because
closing a gap is satisfying and deleting the line that described it is not. So
each line carries a predicate, and the predicate holds only while the gap is
real: implement the feature and its line fails, forcing the line out.

Two rules about what belongs on the list:

**Absences, not refusals.** Most of what this library declines to do it
declines on purpose -- an unconverged integral is refused, a sample too short
for the tail it is asked about is refused, an expectile level is not passed off
as a confidence -- and those are documented as behaviour where they belong. A
list of deliberate refusals relabelled as gaps would read as candour and be
worth less than no list at all.

**Call something.** A predicate that searches for a name cannot see the feature
arriving under a different one. Each predicate below either calls an entry
point and observes what happens, or asserts the exact shape of a public
signature, so that a new parameter under any name breaks it.
"""

from __future__ import annotations

import inspect
import math
import random
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from shortfall import (
    BadMixture,
    Family,
    Panel,
    constant_correlation_target,
    fit_copula,
    fit_garch,
    fit_mixture,
    garch_variances,
    horizon_mixture,
    ledoit_wolf,
    occupation_law,
    stationary,
    tail_dependence_coefficient,
)

README = Path(__file__).resolve().parent.parent / "README.md"

#: The heading the list lives under, and the heading that ends it.
SECTION = "## Known gaps"
NEXT_SECTION = "## Development"


def parameters(function: Callable[..., object]) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def correlated_panel(seed: int, observations: int = 400) -> Panel:
    """Three assets on one common factor, enough for every fit below."""
    rng = random.Random(seed)
    columns: dict[str, list[float]] = {name: [] for name in ("a", "b", "c")}
    for _ in range(observations):
        common = rng.gauss(0.0, 1.0)
        for name in columns:
            columns[name].append(0.01 * (0.7 * common + 0.71 * rng.gauss(0.0, 1.0)))
    return Panel.from_columns(columns)


# --- one predicate per line of the list ----------------------------------- #


def the_variance_recursion_is_symmetric() -> None:
    """Flip the sign of a loss and every later variance holds still, exactly.

    The seed is passed rather than left to default, because the default is the
    sample variance about the sample mean and flipping one return moves the
    mean. With the seed fixed and a zero mean the recursion sees each return
    only through its square, so the two paths agree to the last bit. A leverage
    term would make the loss raise the next variance by more than the gain does.
    """
    rng = random.Random(7)
    returns = [rng.gauss(0.0, 0.01) for _ in range(300)]
    shock = 150
    returns[shock] = -0.05
    flipped = list(returns)
    flipped[shock] = 0.05
    kwargs = {"omega": 2e-6, "alpha": 0.08, "beta": 0.9, "mean": 0.0, "seed": 1e-4}
    down = garch_variances(returns, **kwargs)
    up = garch_variances(flipped, **kwargs)
    assert down[shock + 1] > down[shock], "the shock did not reach the variance"
    assert down == up
    assert parameters(garch_variances) == ("returns", "omega", "alpha", "beta", "mean", "seed")
    assert parameters(fit_garch) == (
        "returns",
        "variance_targeting",
        "mean",
        "strict",
        "max_iterations",
        "innovation",
    )


def the_copulas_are_elliptical() -> None:
    """Two families, refused by name, and one tail-dependence figure for both tails."""
    assert {member.name for member in Family} == {"GAUSSIAN", "STUDENT_T"}
    panel = correlated_panel(3)
    for spelling in ("clayton", "gumbel", "frank"):
        with pytest.raises(ValueError, match="is not a valid Family"):
            fit_copula(panel, family=Family(spelling))
    # Both families that do exist fit the same panel, so the refusal is about
    # the family and not about the data.
    fit_copula(panel, family=Family.GAUSSIAN)
    fit_copula(panel, family=Family.STUDENT_T, degrees_of_freedom=8.0)
    # Upper and lower are one number: there is no argument to ask for either.
    assert parameters(tail_dependence_coefficient) == ("correlation", "degrees_of_freedom")


def the_regime_chain_has_two_states() -> None:
    """A mixture can have three components; the chain over them cannot have three states."""
    rng = random.Random(11)
    sample = [rng.gauss(0.0, 0.01) for _ in range(300)]
    sample += [rng.gauss(-0.02, 0.03) for _ in range(60)]
    sample += [rng.gauss(0.01, 0.005) for _ in range(60)]
    assert len(fit_mixture(sample, 3).mixture.components) == 3
    with pytest.raises(BadMixture, match="two states"):
        occupation_law((0.9, 0.8, 0.7), (0.5, 0.3, 0.2), 5)  # type: ignore[arg-type]
    with pytest.raises(BadMixture, match="two states"):
        stationary((0.9, 0.8, 0.7))  # type: ignore[arg-type]
    assert parameters(horizon_mixture) == ("calm", "stressed", "persistence", "start", "periods")


def shrinkage_has_one_target() -> None:
    """No argument chooses the target, and the target it uses is constant correlation."""
    assert parameters(ledoit_wolf) == ("panel",)
    shrunk = ledoit_wolf(correlated_panel(5))
    # Built from the maximum-likelihood covariance and then rescaled onto the
    # unbiased variances, so it agrees with a direct construction to rounding
    # and not to the bit.
    direct = constant_correlation_target(shrunk.sample)
    for built, expected in zip(shrunk.target, direct, strict=True):
        assert built == pytest.approx(expected, rel=1e-13, abs=0.0)
    # And the target is not the identity or a diagonal: its off-diagonal
    # entries are all one correlation times the two volatilities.
    rho = [
        shrunk.target[i][j] / math.sqrt(shrunk.target[i][i] * shrunk.target[j][j])
        for i in range(3)
        for j in range(3)
        if i < j
    ]
    assert rho[0] > 0.1
    assert max(rho) - min(rho) < 1e-12


GAPS: tuple[tuple[str, Callable[[], None]], ...] = (
    (
        "The variance process has no leverage term.",
        the_variance_recursion_is_symmetric,
    ),
    (
        "The copulas are elliptical, so both tails are the same.",
        the_copulas_are_elliptical,
    ),
    (
        "The regime chain has two states.",
        the_regime_chain_has_two_states,
    ),
    (
        "Covariance shrinkage has one target.",
        shrinkage_has_one_target,
    ),
)


def gaps_section() -> str:
    """The text from the list's heading up to the heading that follows it."""
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.index(NEXT_SECTION, start + len(SECTION))
    return text[start:end]


def rendered_bullets() -> list[str]:
    """The bold lead of each bullet under the heading, in page order."""
    return re.findall(r"^- \*\*(.+?)\*\*", gaps_section(), re.MULTILINE)


@pytest.mark.parametrize("line, predicate", GAPS, ids=[line for line, _ in GAPS])
def test_the_gap_is_still_real(line: str, predicate: Callable[[], None]) -> None:
    predicate()


@pytest.mark.parametrize("line", [line for line, _ in GAPS])
def test_the_line_is_still_on_the_page(line: str) -> None:
    assert line in rendered_bullets(), f"not a bullet under {SECTION!r}: {line!r}"


def test_the_page_lists_nothing_that_has_no_predicate() -> None:
    """Same lines, same order: a bullet added to the page needs a predicate too."""
    assert rendered_bullets() == [line for line, _ in GAPS]


def test_every_predicate_is_distinct() -> None:
    predicates = [predicate for _, predicate in GAPS]
    assert len({p.__name__ for p in predicates}) == len(predicates)


def test_the_section_ends_at_the_heading_that_follows_it() -> None:
    """The list is bounded by the next ``## `` heading, whatever it is called.

    ``NEXT_SECTION not in gaps_section()`` looks like it checks this and cannot
    fail, because the slice is cut at that heading. What can go wrong is a new
    section inserted between the two, whose bullets would then be read as gaps.
    """
    headings = re.findall(r"^## .+$", README.read_text(encoding="utf-8"), re.MULTILINE)
    assert SECTION in headings
    following = headings[headings.index(SECTION) + 1]
    assert following == NEXT_SECTION, (
        f"the list is bounded by {NEXT_SECTION!r} and is followed by {following!r}"
    )

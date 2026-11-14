"""The front page's count of the roadmap is the roadmap's own count.

A number written into prose is a claim that stops being true quietly. The
README said "All seven phases are done" and enumerated seven; the roadmap has
twenty-nine phase sections. The sentence was true when it was written, nothing
recomputed it, and the suite stayed green because nothing in it read either
document — so the front page understated the library by twenty-two phases.

The suite size on the same page had a related problem, fixed in
:mod:`tests.test_readme_evidence`: the guard over it read only the *first*
figure the page states, and the page states two.

Four things are asserted here, each demonstrated to fail by being broken the
way the bug would break it:

* the phase count the README states equals the number of sections the roadmap
  has;
* no two phases share a number;
* the numbering has no gap;
* the headings ascend, so a phase is not filed out of order.

The last three are not what went wrong this time. They are cheap, and a
document whose count is checked while its numbering is not can still be a
document with two phase sevens in it — which is a real way for a roadmap to
drift and one that a count alone cannot see.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ROADMAP = ROOT / "ROADMAP.md"

#: A phase heading: ``## Phase 12 — Something``. The dash is an em dash in
#: every heading, and matching it rather than any dash keeps a hyphen inside a
#: title from being read as the separator.
HEADING = re.compile(r"^## Phase (\d+) — (.+)$", re.MULTILINE)

#: The sentence the count is written in, on the front page.
STATED_COUNT = re.compile(r"All ([a-z]+(?:-[a-z]+)?) phases are done")

#: The count is written as a word, so the words that could plausibly appear are
#: mapped rather than a numeral being required. A count outside this range needs
#: a line added here, which is the point at which somebody notices the document
#: is being counted.
WORDS = {
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-one": 21, "twenty-two": 22, "twenty-three": 23, "twenty-four": 24,
    "twenty-five": 25, "twenty-six": 26, "twenty-seven": 27,
    "twenty-eight": 28, "twenty-nine": 29, "thirty": 30, "thirty-one": 31,
    "thirty-two": 32, "thirty-three": 33, "thirty-four": 34, "thirty-five": 35,
}


def phase_numbers(roadmap: str) -> list[int]:
    """Every phase number in the roadmap, in the order it appears."""
    return [int(number) for number, _ in HEADING.findall(roadmap)]


def stated_phase_count(readme: str) -> int:
    """The phase count the front page claims, read off its own sentence."""
    found = STATED_COUNT.search(readme)
    assert found is not None, (
        "the README no longer states a phase count in the form this gate reads. "
        "Either the sentence moved, in which case fix the pattern, or the claim "
        "was deleted, in which case delete this check with it rather than "
        "leaving a pattern that matches nothing and passes."
    )
    word = found.group(1)
    assert word in WORDS, f"unmapped number word {word!r}; add it to WORDS"
    return WORDS[word]


@pytest.fixture(scope="module")
def roadmap() -> str:
    return ROADMAP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_the_front_page_count_is_the_number_of_phases(readme: str, roadmap: str) -> None:
    """The README's stated count equals the sections the roadmap has.

    This is the check the defect needed: the sentence said seven and the
    roadmap had twenty-nine.
    """
    numbers = phase_numbers(roadmap)
    stated = stated_phase_count(readme)
    assert stated == len(numbers), (
        f"the README claims {stated} phases and the roadmap has {len(numbers)} "
        "sections"
    )


def test_no_two_phases_share_a_number(roadmap: str) -> None:
    """Each phase number appears once."""
    numbers = phase_numbers(roadmap)
    repeated = {n: c for n, c in Counter(numbers).items() if c > 1}
    assert not repeated, f"phase numbers used more than once: {repeated}"


def test_the_numbering_has_no_gap(roadmap: str) -> None:
    """The numbers run 1 to N with nothing missing."""
    numbers = phase_numbers(roadmap)
    expected = set(range(1, len(numbers) + 1))
    assert set(numbers) == expected, (
        f"missing: {sorted(expected - set(numbers))}; "
        f"unexpected: {sorted(set(numbers) - expected)}"
    )


def test_the_headings_ascend(roadmap: str) -> None:
    """A phase is filed in the order its number says."""
    numbers = phase_numbers(roadmap)
    out_of_order = [
        (numbers[i], numbers[i + 1])
        for i in range(len(numbers) - 1)
        if numbers[i + 1] <= numbers[i]
    ]
    assert not out_of_order, f"headings out of order at: {out_of_order}"


def test_the_probe_reads_a_roadmap_worth_counting(roadmap: str) -> None:
    """The checks above are not passing because they found nothing.

    Every assertion here is satisfied by an empty roadmap: a count of zero
    equals a count of zero, and an empty list has no duplicates, no gaps and no
    disorder. So the probe is required to have read something, with a loose
    bound — the claim is that phases were found at all, not that there are
    exactly this many.
    """
    numbers = phase_numbers(roadmap)
    assert len(numbers) >= 20, (
        f"the heading probe found {len(numbers)} phases; the roadmap had 29 when "
        "this was written, and a sharp drop means the pattern stopped matching"
    )
    assert numbers[0] == 1, f"the first phase found is numbered {numbers[0]}"

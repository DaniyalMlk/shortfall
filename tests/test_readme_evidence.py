"""The figures on the front page are the figures the examples produce.

The README opens with a table of results checked against things outside this
repository. A table like that is worth exactly as much as the guarantee that
it still matches, and a number copied into a document once is a number that
stops being true quietly: the code changes, the suite stays green because
nothing in it reads the document, and the front page goes on claiming a
precision the library no longer has.

So the claim is pinned here. Every figure the table quotes has to appear in
the output of the example that produces it, and each of those examples is
already a gate of its own — it exits non-zero if a published figure moves.
The two together mean a reader can trust the front page without running
anything, and a change that moves a figure fails the suite rather than the
reader.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EXAMPLES = ROOT / "examples"

#: Each example, and the figures the README quotes from it. Written out here
#: rather than parsed from the table, so that deleting a row does not silently
#: delete the check defending it.
#:
#: Exact strings are safe for these three. The figures are closed forms and
#: seeded simulations, and they were confirmed identical on 3.11, 3.12 and
#: 3.13 before being quoted. Anything added later needs the same check, or a
#: bound instead of digits: a quantity whose last bits come out of a root
#: finder is not portable, and quoting ten significant figures of one puts a
#: number on the front page that is true only where it was measured.
EVIDENCE: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "basel_equivalence.py",
        ("2.326348", "2.337803", "0.49%", "10.99%"),
    ),
    (
        "tail_comparison.py",
        ("+0.2%", "-21.1%", "-59.6%"),
    ),
    (
        "worst_case_scenario.py",
        ("1.8501e-11",),
    ),
)


def _run(example: str) -> str:
    finished = subprocess.run(
        [sys.executable, str(EXAMPLES / example)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert finished.returncode == 0, (
        f"examples/{example} exited {finished.returncode}:"
        f"\n{finished.stdout}\n{finished.stderr}"
    )
    return finished.stdout


@pytest.fixture(scope="module")
def outputs() -> dict[str, str]:
    """Run each example once and share its output across the tests."""
    return {example: _run(example) for example, _ in EVIDENCE}


@pytest.mark.parametrize(
    ("example", "figure"),
    [(example, figure) for example, figures in EVIDENCE for figure in figures],
    ids=[
        f"{example}:{figure}"
        for example, figures in EVIDENCE
        for figure in figures
    ],
)
def test_a_quoted_figure_is_one_its_example_prints(
    example: str, figure: str, outputs: dict[str, str]
) -> None:
    assert figure in outputs[example], (
        f"the README quotes {figure}, which examples/{example} no longer prints"
    )


#: The row labels of the README's evidence table.
ROWS = (
    "Basel's 99% value at risk under a normal",
    "Basel's 97.5% expected shortfall, the measure that replaced it",
    "The gap between the two, which is the point of the switch",
    "A fitted tail at 99.99%, against a truth the sample cannot see",
    "Historical simulation at the same point",
    "A fitted normal at the same point",
    'What "three sigma" names over fifty factors',
)


@pytest.mark.parametrize("row", ROWS)
def test_the_evidence_table_still_has_the_row(row: str) -> None:
    """A row removed from the table takes its claim with it.

    Only the label, not the figure beside it: the direction that catches drift
    is the other one above, where every quoted figure has to be one its
    example prints. This covers the other way a claim disappears, which is
    somebody deleting it.
    """
    assert row in README.read_text(), f"the evidence table no longer has: {row}"


@pytest.mark.parametrize("example", [example for example, _ in EVIDENCE])
def test_the_readme_tells_the_reader_how_to_run_it(example: str) -> None:
    """A figure a reader cannot reproduce is a figure they have to trust."""
    assert example in README.read_text()


def test_the_normal_is_the_most_precise_and_the_most_wrong() -> None:
    """The claim the table makes in prose, asserted rather than described.

    The table says the fitted normal has the smallest spread of the three
    while being the most biased, which is the whole point of the row. If the
    ordering ever reversed, the prose beside it would be wrong and nothing
    else in the suite would notice.
    """
    output = _run("tail_comparison.py")
    section = output.split("99.99% confidence")[1].split("\n\n")[0]
    spreads = {}
    biases = {}
    for line in section.splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0] in {"fitted", "historical", "normal"}:
            biases[parts[0]] = abs(float(parts[2].rstrip("%")))
            spreads[parts[0]] = float(parts[-1].rstrip("%"))
    assert set(spreads) == {"fitted", "historical", "normal"}, section
    assert spreads["normal"] == min(spreads.values())
    assert biases["normal"] == max(biases.values())


def test_the_suite_size_on_the_front_page_is_not_wildly_stale() -> None:
    """A loose bound, deliberately.

    Pinning the exact count would fail on every commit that adds a test, which
    trains people to update the number without reading it. An order-of-
    magnitude check still catches the README claiming hundreds when there are
    thousands, which is the error that matters.
    """
    import re

    claimed = re.search(r"([\d,]+) tests", README.read_text())
    assert claimed is not None, "the README no longer states a suite size"
    stated = int(claimed.group(1).replace(",", ""))

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
         "-o", "addopts="],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    found = re.search(r"(\d+) tests collected", collected.stdout)
    assert found is not None, collected.stdout[-500:]
    actual = int(found.group(1))
    assert 0.75 * actual <= stated <= 1.25 * actual, (
        f"the README says {stated} tests and there are {actual}"
    )

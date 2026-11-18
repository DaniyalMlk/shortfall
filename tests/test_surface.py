"""The package's public surface is what its modules declare.

``src/shortfall/__init__.py`` re-exports the package's public names and it is a
hand-maintained list, now of 378 entries. A hand-maintained list that long is a
list with something missing from it, and twenty-six names were missing: declared
public by the module that defines them, raising ``AttributeError`` from the
package root.

``linalg`` showed the shape of it. The module declares twenty-four names; the
root carried fourteen, and the ten it dropped were not a coherent subset.
``cholesky`` is typed ``(matrix: Matrix) -> Matrix`` and ``eigh`` returns
``tuple[Vector, Matrix]``. Both functions were importable from the root and
neither of the types they are written in was, so a caller could invoke them and
could not name what they take or give back. ``MAX_SWEEPS``, the constant behind
``eigh``'s own default, was unreachable for the same reason.

Nothing noticed because nothing looked. Every test file in this suite imports
submodules directly — ``from shortfall.linalg import ...`` — which is the right
style for a test and exactly the style that cannot see a missing re-export: the
module is exercised in full and the surface it is absent from is never touched.
Two things, each tested on its own, with the gap between them tested by neither.

So the surface is recomputed here from the modules rather than compared against
a second list that would need the same remembering.

**A module's own ``__all__`` is the authority.** The alternative is to decide
here which names are public, which puts the judgement in the gate instead of
beside the code it is about, and makes the gate wrong whenever a module changes
its mind. What is asserted is a disagreement *inside* the package: a module
declaring a name public while the root does not carry it.

**Equality, and the probe that makes it safe to assert.** The root is
exactly the union of the module declarations plus ``__version__``. That could
only be asserted once every module declared something and every declaration
was complete, because until then the root offered forty-nine names no module
claimed: ``copula``, ``extreme`` and ``ranks`` carried no ``__all__`` and four
other modules had forgotten names of their own. Completeness is read from
source with :mod:`ast`, not from attributes, because most of what had been
forgotten was bare floats and ints, which carry no ``__module__`` to sweep by.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

import shortfall

ROOT = Path(__file__).resolve().parent.parent

#: A front-page example reaching inside the package rather than importing from it.
SUBMODULE_IMPORT = re.compile(r"^from (shortfall\.[a-z0-9_]+) import", re.MULTILINE)

#: The house style, which the check above needs to have found at least once.
ROOT_IMPORT = re.compile(r"^from shortfall import", re.MULTILINE)

#: Names the root carries that no module declares, because they belong to the
#: package rather than to any one module.
ROOT_ONLY = frozenset({"__version__"})

#: A module declaring a name it does not itself define. None does; the set is
#: written down so that the first one is a decision rather than an accident.
RE_EXPORTS: frozenset[tuple[str, str]] = frozenset()

#: The command line is not part of the importable surface: it is an entry point
#: and its names are wired to ``argparse``, not to callers.
NOT_SURFACE = frozenset({"cli"})

def submodules() -> dict[str, ModuleType]:
    """Every importable submodule of the package, bar the entry point."""
    return {
        info.name: importlib.import_module(f"shortfall.{info.name}")
        for info in pkgutil.iter_modules(shortfall.__path__)
        if info.name not in NOT_SURFACE
    }


def public_definitions(module: str) -> frozenset[str]:
    """Every public name a module defines at its top level, read from source.

    Read with :mod:`ast` rather than off the imported module. An attribute
    sweep has to decide which attributes a module *defines* as opposed to
    imports, and the usual test, ``obj.__module__``, does not exist on a plain
    ``int`` or ``float``. All eleven names this was written for were exactly
    that: ``SUPERVISORY_OBSERVATIONS``, ``NEGLIGIBLE``, ``EDGE`` and the rest are
    bare numbers an attribute probe cannot attribute to anybody.
    """
    source = ROOT / "src" / "shortfall" / f"{module}.py"
    return public_names_in(source.read_text(encoding="utf-8"))


def public_names_in(source: str) -> frozenset[str]:
    """The public top-level names a piece of module source defines.

    Separated from the file it usually reads so that it can be tested against
    source written for the purpose: nothing in ``src/shortfall`` declares an
    ``async def`` at the top level or assigns two names in one statement, and
    a branch the package does not exercise is one nobody has run.
    """
    found: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        # ``__all__`` itself starts with an underscore, so this one rule excludes
        # it too; a separate ``!= "__all__"`` test was dead code, which a break
        # that removed it and left the suite green is how it was found.
        found.update(name for name in names if not name.startswith("_"))
    return frozenset(found)


def declared() -> dict[str, frozenset[str]]:
    """Each module's own declaration of what it makes public."""
    return {
        name: frozenset(module.__all__)
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is not None
    }


@pytest.fixture(scope="module")
def root() -> frozenset[str]:
    return frozenset(shortfall.__all__)


def test_every_declared_name_reaches_the_root(root: frozenset[str]) -> None:
    """A name a module declares public is importable from the package.

    This is the check the defect needed. Twenty-six names failed it, ten of
    them from ``linalg`` alone.
    """
    missing: dict[str, list[str]] = {}
    for module, names in declared().items():
        absent = sorted(names - root)
        if absent:
            missing[module] = absent
    assert not missing, (
        f"declared public but not re-exported from shortfall/__init__.py: {missing}"
    )


def test_a_function_at_the_root_can_have_its_types_named_there(
    root: frozenset[str],
) -> None:
    """The specific failure that made this worth fixing, kept as its own case.

    The containment check above covers it, but as one of twenty-six loose names
    rather than as the thing a caller actually trips over: a function reachable
    from the root whose parameter and return types are not. Spelling it out
    means a future partial re-export fails with a message that says what broke
    instead of listing names.
    """
    pairs = [
        ("cholesky", ("Matrix",)),
        ("eigh", ("Matrix", "Vector")),
        ("quadratic_form", ("Matrix", "Vector")),
        ("least_squares", ("Matrix", "Vector")),
    ]
    for function, types in pairs:
        assert function in root, f"{function} is not at the root at all"
        for name in types:
            assert name in root, (
                f"shortfall.{function} is importable from the root but "
                f"shortfall.{name}, which its signature is written in, is not"
            )


def test_no_module_is_absent_from_the_root_entirely(root: frozenset[str]) -> None:
    """A module whose whole surface is missing is reported as the module.

    The containment check would catch it too, as a pile of loose names across
    several modules, which reads as several small omissions rather than as the
    one large one it would be. A module nobody wired in at all is a different
    mistake from a name somebody forgot.
    """
    absent = sorted(
        module for module, names in declared().items() if names and not (names & root)
    )
    assert not absent, (
        "these modules declare public names and none of them is at the root, so "
        f"the module was never wired into shortfall/__init__.py: {absent}"
    )


def test_every_exported_name_is_actually_there(root: frozenset[str]) -> None:
    """Nothing in ``__all__`` is a name the package does not have.

    ``__all__`` is a list of strings and nothing checks it against the module,
    so a rename that updates the import and not the list, or a typo in either,
    leaves an entry that only fails for somebody doing ``from shortfall import
    *``.
    """
    phantom = sorted(name for name in root if not hasattr(shortfall, name))
    assert not phantom, f"in shortfall.__all__ but not attributes of it: {phantom}"


def test_the_root_exports_each_name_once() -> None:
    """``__all__`` does not repeat itself.

    Harmless at import time, and the visible residue of a merge that went
    wrong, so it is worth failing on while it is still one line.
    """
    repeated = {
        name: count for name, count in Counter(shortfall.__all__).items() if count > 1
    }
    assert not repeated, f"repeated in shortfall.__all__: {repeated}"


def test_every_module_declares_something() -> None:
    """No module carries no ``__all__`` at all.

    Every other check here is silent about a module that declares nothing, so
    a module without one is exempt from all of them. ``copula``, ``extreme``
    and ``ranks`` were in exactly that position, between them responsible for
    all forty-nine names the root offered and no module claimed.

    There is no exemption list. A module with nothing public to say can
    declare an empty ``__all__`` and say so.
    """
    silent = sorted(
        name
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is None
    )
    assert not silent, (
        f"modules carrying no __all__, and so exempt from every check here: {silent}"
    )


def test_a_module_declares_every_public_name_it_defines() -> None:
    """``__all__`` is not a curated subset of what the module defines.

    This catches the *next* omission rather than the last one: a function
    added to a module and left out of its ``__all__`` fails here, at the
    module, instead of reaching the root as a name nobody wired in.

    Six names failed it across four modules. Two are the defaults of public
    parameters: ``fit_mixture(floor=DEFAULT_FLOOR)`` and
    ``convolve(negligible=NEGLIGIBLE)`` showed a caller a value they could not
    import by name.
    """
    undeclared: dict[str, list[str]] = {}
    for module, names in declared().items():
        missing = sorted(public_definitions(module) - names)
        if missing:
            undeclared[module] = missing
    assert not undeclared, (
        f"defined at a module's top level and absent from its own __all__: {undeclared}"
    )


def test_a_name_a_module_declares_but_does_not_define_is_a_pinned_re_export() -> None:
    """A module declaring a name from elsewhere is a decision, not a default.

    None does today. A declaration of a name the module only imports is also
    what a bad merge or a half-finished move looks like, so the first one has
    to be written into ``RE_EXPORTS`` by whoever means it.
    """
    found = {
        (module, name)
        for module, names in declared().items()
        for name in names - public_definitions(module)
    }
    assert found == set(RE_EXPORTS), (
        f"re-exports found: {sorted(found)}; pinned: {sorted(RE_EXPORTS)}"
    )


def test_the_root_is_exactly_the_union_of_the_declarations(
    root: frozenset[str],
) -> None:
    """Equality, not containment.

    The containment above catches a name the root is missing. This catches the
    other direction: a name the root offers that no module claims, which is
    what a hand-kept list of nearly four hundred entries accumulates, and what
    this one had accumulated forty-nine of.
    """
    union = frozenset().union(*declared().values())
    assert root == union | ROOT_ONLY, (
        f"only at the root: {sorted(root - union - ROOT_ONLY)}; "
        f"only in a module: {sorted(union - root)}"
    )


def test_no_public_name_is_defined_by_two_modules() -> None:
    """One name, one thing.

    ``contributions`` and ``volatility`` each defined a ``DidNotConverge``,
    unrelated to each other. The root can carry one object per name, so it
    carried the risk-parity class, and ``except shortfall.DidNotConverge``
    around a GARCH fit caught nothing. Every check above is blind to this,
    because a union of sets counts a name once however many modules declare it.
    """
    owners: dict[str, list[str]] = {}
    for module in submodules():
        for name in public_definitions(module):
            owners.setdefault(name, []).append(module)
    shared = {name: sorted(found) for name, found in owners.items() if len(found) > 1}
    assert not shared, f"public names defined by more than one module: {shared}"


def test_no_name_is_declared_by_two_modules() -> None:
    """The declaration-side twin of the check above.

    Both modules listed ``DidNotConverge`` in their ``__all__``. A name two
    modules declare is either two objects, which the check above refuses, or a
    re-export, which ``RE_EXPORTS`` must already name; either way it should not
    pass silently through the union.
    """
    owners: dict[str, list[str]] = {}
    for module, names in declared().items():
        for name in names:
            owners.setdefault(name, []).append(module)
    shared = {name: sorted(found) for name, found in owners.items() if len(found) > 1}
    assert not shared, f"names declared by more than one module: {shared}"


def test_the_front_page_imports_from_the_package_and_not_from_inside_it() -> None:
    """No example on the front page reaches into a submodule.

    Every example does import from the root today. The rule is asserted so
    that a missing re-export cannot be worked around in the document, where a
    ``from shortfall.module import`` line reads as a style choice rather than
    as the gap it would be.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    reaching_in = sorted(set(SUBMODULE_IMPORT.findall(readme)))
    assert not reaching_in, (
        "front-page examples importing from inside the package rather than from "
        f"it: {reaching_in}"
    )
    assert ROOT_IMPORT.search(readme) is not None, (
        "no `from shortfall import` line found on the front page at all, so the "
        "check above passed by reading nothing"
    )


def test_the_probe_reads_a_surface_worth_checking(root: frozenset[str]) -> None:
    """The checks above are not passing because they found nothing.

    Every assertion in this file is satisfied by an empty set, so a probe that
    stops importing modules or stops finding ``__all__`` turns the whole file
    green. The sizes are asserted with loose bounds — the claim is that a
    surface was read at all, not that it is exactly this big — and ``linalg``
    is named, because it is the module the defect centred on and the one a
    future rewrite is most likely to trim again.
    """
    modules = declared()
    assert len(modules) >= 18, f"only {len(modules)} modules declared an __all__"
    assert len(root) >= 300, f"the root surface came back as {len(root)} names"
    assert "linalg" in modules, "the linalg module was not read"
    assert len(modules["linalg"]) >= 20, (
        f"linalg declares {len(modules['linalg'])} names; it declared 24 when this "
        "was written, and a sharp drop means the probe or the module changed shape"
    )


#: Source written to exercise the probe rather than to be a module. Each line
#: is one rule, and the comment says which.
PROBE_SOURCE = '''
"""A docstring, which defines nothing."""

from __future__ import annotations

import math                      # an import is not a definition
from pathlib import Path         # nor is this
from typing import Final

PUBLIC_CONSTANT: Final = 1.0     # an annotated bare float, as most of ours are
PLAIN = 2                        # an unannotated one
PAIR_ONE = PAIR_TWO = 3          # one statement, two names
_PRIVATE = 4                     # underscore: not public
__all__ = ["PUBLIC_CONSTANT"]    # the declaration itself is never a definition


def public_function() -> None:
    """A function."""
    local_name = 5               # a name inside a body is not top level


async def public_coroutine() -> None:
    """An async function, which this package has none of."""


def _private_function() -> None:
    """Underscore again."""


class PublicClass:
    """A class."""

    attribute = 6                # a class attribute is not a module name


class _PrivateClass:
    """And a private one."""
'''


def test_the_probe_reads_each_kind_of_definition_and_skips_each_non_definition() -> None:
    """The probe, against source written for it.

    The exclusions matter as much as the inclusions. A probe that counted
    imports would report every module as failing to declare ``math``; one that
    walked function bodies would report ``local_name``; one that missed
    ``__all__`` itself would have every module failing to declare its own
    declaration.
    """
    found = public_names_in(PROBE_SOURCE)
    assert found == {
        "PUBLIC_CONSTANT",
        "PLAIN",
        "PAIR_ONE",
        "PAIR_TWO",
        "public_function",
        "public_coroutine",
        "PublicClass",
    }, f"the probe read {sorted(found)}"

    # Said the other way round, so that a future widening has to break one of
    # these deliberately rather than by accident.
    for absent in (
        "math",
        "Path",
        "Final",
        "_PRIVATE",
        "__all__",
        "local_name",
        "_private_function",
        "_PrivateClass",
        "attribute",
    ):
        assert absent not in found, f"the probe counted {absent!r} as a definition"

"""The package's public surface is what its modules declare.

``src/shortfall/__init__.py`` re-exports the package's public names and it is a
hand-maintained list, now of 367 entries. A hand-maintained list that long is a
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

**Containment, not equality, and that is a gap rather than an oversight.**
Forty-nine names reach the root without appearing in any module's ``__all__``.
``copula``, ``extreme`` and ``ranks`` declare none at all, and four other
modules have forgotten names of their own. Asserting that the root is *exactly*
the union would mean deciding, here, that each of those is public — the
judgement this file has just refused to make. Closing it means giving those
three modules an ``__all__`` and completing the others, in the modules, after
which the equality can replace the containment.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections import Counter
from types import ModuleType

import pytest

import shortfall

#: The command line is not part of the importable surface: it is an entry point
#: and its names are wired to ``argparse``, not to callers.
NOT_SURFACE = frozenset({"cli"})

#: The modules that declare no ``__all__``. Pinned rather than discovered, so
#: that a new module arriving without one is a decision somebody makes here
#: instead of a silent exemption from every check in this file.
NO_DUNDER_ALL = frozenset({"copula", "extreme", "ranks"})


def submodules() -> dict[str, ModuleType]:
    """Every importable submodule of the package, bar the entry point."""
    return {
        info.name: importlib.import_module(f"shortfall.{info.name}")
        for info in pkgutil.iter_modules(shortfall.__path__)
        if info.name not in NOT_SURFACE
    }


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


def test_the_modules_without_a_declaration_are_the_ones_pinned_here() -> None:
    """Only the three known modules carry no ``__all__``.

    Every check above is silent about a module that declares nothing, so the
    set of modules exempt from them is written down here. A new module without
    an ``__all__`` fails this, which is the point at which somebody decides
    whether that is what was wanted — and the point at which the containment
    above could become an equality.
    """
    silent = {
        name
        for name, module in submodules().items()
        if getattr(module, "__all__", None) is None
    }
    assert silent == set(NO_DUNDER_ALL), (
        f"modules carrying no __all__: {sorted(silent)}; "
        f"pinned: {sorted(NO_DUNDER_ALL)}"
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

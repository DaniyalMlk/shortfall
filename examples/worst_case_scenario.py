"""Measure what searching the whole plausible set buys over one factor at a time.

The case for a joint worst-case stress test is usually made by argument: moving
one factor at a time cannot find a move that needs two factors together, so it
must understate. The argument is right and it does not say by how much. This
script measures it, on random books with a known exposure, and reports every
number whichever way it comes out.

Both searches cover the *same* ellipsoid. That is the only arrangement in which
the comparison means anything: each single-factor move is scaled so that it
lands on the boundary of the set the joint search is working over, so the only
difference left is that one of them searches all of it. The version a risk
report actually contains -- move each factor by three of its own standard
deviations -- is also measured, and it turns out not to be inside the set at all.

Five things, all of them measured:

1. How much the joint worst case exceeds the worst single-factor loss, against
   the number of factors.
2. How much of that gap is the correlation and how much is the curvature.
3. Whether the sum of the single-factor losses is a bound on anything.
4. Whether the usual "n sigma in each factor" move is even inside the ellipsoid.
5. What a 99% ellipsoid's radius actually is, which is not 3.

Exits non-zero if any of them moves, so a change to the solver cannot quietly
rewrite a figure the README quotes.

Run with ``python examples/worst_case_scenario.py``.
"""

from __future__ import annotations

import math
import random
import sys

from shortfall import (
    Exposure,
    chi_square_cdf,
    one_at_a_time,
    radius_for,
    worst_case_scenario,
)

BOOKS = 400


def _spd(generator: random.Random, size: int) -> list[list[float]]:
    noise = [[generator.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    return [
        [
            math.fsum(noise[i][k] * noise[j][k] for k in range(size)) + (0.4 if i == j else 0.0)
            for j in range(size)
        ]
        for i in range(size)
    ]


def _book(generator: random.Random, size: int) -> tuple[Exposure, list[list[float]]]:
    covariance = _spd(generator, size)
    delta = [generator.gauss(0.0, 1.0) for _ in range(size)]
    noise = [[generator.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    gamma = [[0.5 * (noise[i][j] + noise[j][i]) for j in range(size)] for i in range(size)]
    return Exposure(delta=delta, gamma=gamma), covariance


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> dict[str, object]:
    print("A 99% ellipsoid's radius, and what 'three sigma' actually names")
    print(f"{'factors':>8}{'99% radius':>12}{'P[3 sigma set]':>16}")
    radii = {}
    for factors in (1, 2, 3, 5, 10, 50):
        radii[factors] = radius_for(0.99, factors)
        print(f"{factors:>8}{radii[factors]:>12.4f}{chi_square_cdf(9.0, factors):>16.6g}")
    print("the phrase does not distinguish 99.7% of the probability from 2e-11")
    print()

    print(f"joint worst case against one factor at a time, {BOOKS} random books each")
    print(f"{'factors':>8}{'median':>9}{'q1':>7}{'q3':>7}{'min':>7}{'linear':>8}")
    gaps: dict[int, float] = {}
    linear_gaps: dict[int, float] = {}
    for size in (2, 3, 5, 8):
        generator = random.Random(11 + size)
        radius = radius_for(0.99, size)
        ratios: list[float] = []
        flat_ratios: list[float] = []
        for _ in range(BOOKS):
            exposure, covariance = _book(generator, size)
            flat = Exposure(delta=exposure.delta, gamma=[[0.0] * size for _ in range(size)])
            for book, bucket in ((exposure, ratios), (flat, flat_ratios)):
                naive = one_at_a_time(book, covariance, radius)
                if naive.worst > 0.0:
                    bucket.append(worst_case_scenario(book, covariance, radius).loss / naive.worst)
        ratios.sort()
        gaps[size] = _median(ratios)
        linear_gaps[size] = _median(flat_ratios)
        print(
            f"{size:>8}{gaps[size]:>9.2f}{ratios[len(ratios) // 4]:>7.2f}"
            f"{ratios[3 * len(ratios) // 4]:>7.2f}{min(ratios):>7.2f}{linear_gaps[size]:>8.2f}"
        )
    print("the last column is the same deltas with the curvature removed, so about")
    print("half the gap is the correlation and half is the gamma")
    print()

    print("is the sum of the single-factor losses a bound on anything?")
    print(f"{'factors':>8}{'median ratio':>14}{'joint exceeds sum':>20}")
    exceeded: dict[int, float] = {}
    for size in (3, 5, 8):
        generator = random.Random(400 + size)
        radius = radius_for(0.99, size)
        ratios = []
        over = 0
        for _ in range(BOOKS):
            exposure, covariance = _book(generator, size)
            naive = one_at_a_time(exposure, covariance, radius)
            if naive.total <= 0.0:
                continue
            joint = worst_case_scenario(exposure, covariance, radius).loss
            ratios.append(joint / naive.total)
            over += joint > naive.total
        exceeded[size] = over / len(ratios)
        print(f"{size:>8}{_median(ratios):>14.2f}{exceeded[size]:>19.1%}")
    print("no: adding up independent stresses gives a number no single point produces")
    print()

    print("is the usual 'n sigma in each factor' move inside the ellipsoid?")
    outside_all = True
    for size in (2, 3, 5, 8):
        generator = random.Random(13 + size)
        outside = 0
        for _ in range(100):
            exposure, covariance = _book(generator, size)
            if not one_at_a_time(
                exposure, covariance, radius_for(0.99, size), on_the_ellipsoid=False
            ).inside:
                outside += 1
        outside_all = outside_all and outside == 100
        print(f"   {size} factors: outside in {outside} of 100")
    print("never, unless a factor is uncorrelated with every other one")
    print()

    # One concrete book, and the move that breaks it.
    # Monthly covariance: standard deviations of 2%, 3%, 1.5% and 1.2%.
    covariance = [
        [0.000400, 0.000120, 0.000000, 0.000060],
        [0.000120, 0.000900, -0.000100, 0.000150],
        [0.000000, -0.000100, 0.000225, -0.000020],
        [0.000060, 0.000150, -0.000020, 0.000144],
    ]
    names = ("equity", "credit", "gold", "utilities")
    exposure = Exposure(
        delta=[0.5, -0.2, 0.1, 0.3],
        gamma=[
            [-40.0, 0.0, 0.0, 0.0],
            [0.0, -30.0, 0.0, 0.0],
            [0.0, 0.0, -25.0, 0.0],
            [0.0, 0.0, 0.0, -10.0],
        ],
    )
    radius = radius_for(0.99, 4)
    scenario = worst_case_scenario(exposure, covariance, radius)
    naive = one_at_a_time(exposure, covariance, radius)
    print(f"a short-gamma book over four factors, 99% ellipsoid of radius {radius:.4f}")
    print(f"   joint worst case            {scenario.loss:>8.2%}")
    print(f"   worst single factor         {naive.worst:>8.2%}  ({names[naive.worst_factor]})")
    print(f"   sum of the single factors   {naive.total:>8.2%}")
    print("   the move, in each factor's standard deviations:")
    for index, name in enumerate(names):
        print(f"      {name:<10}{scenario.standardised[index]:>+8.2f}")

    results: dict[str, object] = {
        "radii": radii,
        "gaps": gaps,
        "linear_gaps": linear_gaps,
        "exceeded": exceeded,
        "always_outside": outside_all,
        "book_joint": scenario.loss,
        "book_naive": naive.worst,
    }

    failures = []
    if not 1.2 < gaps[2] < 1.6:
        failures.append(f"two-factor gap {gaps[2]:.3f} moved out of [1.2, 1.6]")
    if not 5.5 < gaps[8] < 8.5:
        failures.append(f"eight-factor gap {gaps[8]:.3f} moved out of [5.5, 8.5]")
    if not linear_gaps[8] < gaps[8]:
        failures.append("the curvature stopped contributing to the gap")
    if not exceeded[8] > 0.95:
        failures.append(f"the sum stopped being exceeded at eight factors ({exceeded[8]:.3f})")
    if not outside_all:
        failures.append("an 'n sigma in each factor' move landed inside the ellipsoid")
    if not abs(radii[50] - 8.7266) < 1e-3:
        failures.append(f"the fifty-factor radius moved to {radii[50]:.4f}")
    if failures:
        print()
        for line in failures:
            print(f"FAILED: {line}")
        sys.exit(1)
    print()
    print("All figures reproduced.")
    return results


if __name__ == "__main__":
    main()

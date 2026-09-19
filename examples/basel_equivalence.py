"""Reproduce the published figure behind the Basel move to expected shortfall.

The 2016 Fundamental Review of the Trading Book replaced 99% value at risk with
97.5% expected shortfall as the capital measure. The calibration rests on a
published fact: under a **normal** distribution those two numbers are almost the
same, so the switch changes what the measure *is* — from a quantile that says
nothing about the losses beyond it to an average of them — without changing how
much capital it demands in the ordinary case.

The standard figures are

    normal 99% value at risk       2.3263 sigma
    normal 97.5% expected shortfall 2.3378 sigma

which agree to within half a per cent. This script recomputes both from the
library and checks them against those published values.

It then shows the part that matters more, which is that the equivalence is a
property of the normal distribution and not of returns. Real returns have
heavier tails than normal, and the heavier the tail the further apart the two
measures drift — expected shortfall rising faster, because it is an average over
the tail and therefore sees the tail thicken while a quantile only sees it move.
By three degrees of freedom the gap is more than twenty times the normal one.

Run it with ``python examples/basel_equivalence.py``. It exits non-zero if any
figure fails to reproduce, so continuous integration notices if a change to the
numerics moves a number that has been published since 2016.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

from shortfall import (
    DAILY_TRADING,
    Distribution,
    Panel,
    historical_risk,
    ledoit_wolf,
    normal_risk,
    portfolio_risk,
    student_t_risk,
)

#: Published, and reproduced here to four decimal places.
PUBLISHED_VALUE_AT_RISK = 2.3263
PUBLISHED_EXPECTED_SHORTFALL = 2.3378
TOLERANCE = 5e-5

WEIGHTS = [0.40, 0.25, 0.15, 0.20]


def unit(confidence: float, *, shortfall: bool) -> float:
    """The multiple of volatility a zero-mean normal portfolio risks."""
    risk = normal_risk(mean=0.0, volatility=1.0, confidence=confidence)
    return risk.expected_shortfall if shortfall else risk.value_at_risk


def check(label: str, computed: float, published: float) -> bool:
    gap = abs(computed - published)
    verdict = "ok" if gap <= TOLERANCE else "FAILED"
    print(f"  {label:<38} {computed:.6f}   published {published:.4f}   {verdict}")
    return gap <= TOLERANCE


def the_equivalence() -> bool:
    print("The Basel calibration, under a normal distribution")
    print("-" * 64)
    value_at_risk = unit(0.99, shortfall=False)
    shortfall = unit(0.975, shortfall=True)
    passed = check("99% value at risk", value_at_risk, PUBLISHED_VALUE_AT_RISK)
    passed &= check(
        "97.5% expected shortfall", shortfall, PUBLISHED_EXPECTED_SHORTFALL
    )
    gap = shortfall / value_at_risk - 1.0
    print(f"\n  The two differ by {gap:+.2%}, which is what makes the switch")
    print("  a change of measure rather than a change of capital.\n")
    return passed


def where_it_stops_holding() -> bool:
    """The equivalence under progressively heavier tails."""
    print("The same comparison under a Student-t, scaled to the same volatility")
    print("-" * 64)
    print(f"  {'tail':<10}{'VaR 99%':>12}{'ES 97.5%':>12}{'gap':>10}")
    normal_gap = unit(0.975, shortfall=True) / unit(0.99, shortfall=False) - 1.0
    print(
        f"  {'normal':<10}{unit(0.99, shortfall=False):>12.4f}"
        f"{unit(0.975, shortfall=True):>12.4f}{normal_gap:>9.2%}"
    )
    widening = [normal_gap]
    for degrees in (8.0, 5.0, 4.0, 3.0):
        at_risk = student_t_risk(
            mean=0.0, volatility=1.0, confidence=0.99, degrees=degrees
        ).value_at_risk
        shortfall = student_t_risk(
            mean=0.0, volatility=1.0, confidence=0.975, degrees=degrees
        ).expected_shortfall
        gap = shortfall / at_risk - 1.0
        widening.append(gap)
        label = f"t({int(degrees)})"
        print(f"  {label:<10}{at_risk:>12.4f}{shortfall:>12.4f}{gap:>9.2%}")

    # The gap must widen monotonically as the tail thickens. That is the claim
    # the table is making, so it is asserted rather than left to the reader.
    ordered = all(later > earlier for earlier, later in pairwise(widening))
    print(
        f"\n  The gap widens monotonically as the tail thickens: {ordered}."
        f"\n  At three degrees of freedom it is {widening[-1] / widening[0]:.0f} times"
        " the normal gap,"
        "\n  so the equivalence is a fact about the normal distribution and not"
        "\n  about returns.\n"
    )
    return ordered


def on_real_weights() -> bool:
    """The same two measures on the bundled portfolio, assumed and empirical."""
    path = Path(__file__).with_name("returns.csv")
    rows = [line.split(",") for line in path.read_text().strip().splitlines()]
    names = [cell.strip() for cell in rows[0][1:]]
    panel = Panel.from_columns(
        {
            name: [float(row[position + 1]) for row in rows[1:]]
            for position, name in enumerate(names)
        }
    )
    covariance = ledoit_wolf(panel).matrix

    print("The same pair on the bundled four-asset portfolio")
    print("-" * 64)
    at_risk = portfolio_risk(
        WEIGHTS, covariance, confidence=0.99, distribution=Distribution.NORMAL
    ).value_at_risk
    shortfall = portfolio_risk(
        WEIGHTS, covariance, confidence=0.975, distribution=Distribution.NORMAL
    ).expected_shortfall
    print(f"  assumed normal   VaR 99% {at_risk:.4%}    ES 97.5% {shortfall:.4%}")

    portfolio = panel.portfolio(WEIGHTS)
    empirical_var = historical_risk(portfolio, confidence=0.99).value_at_risk
    empirical_es = historical_risk(portfolio, confidence=0.975).expected_shortfall
    print(
        f"  from the sample  VaR 99% {empirical_var:.4%}    ES 97.5% {empirical_es:.4%}"
    )

    assumed_gap = shortfall / at_risk - 1.0
    sample_gap = empirical_es / empirical_var - 1.0
    print(
        f"\n  Under the normal assumption the two measures sit {assumed_gap:+.2%}"
        f"\n  apart, as they must. Taken from the sample they sit {sample_gap:+.2%}"
        "\n  apart — the same widening the Student-t table shows, arriving here"
        "\n  from data rather than from an assumption.\n"
    )
    print(
        f"  Annualised portfolio volatility: "
        f"{portfolio.annualised_volatility(DAILY_TRADING):.2%}\n"
    )
    # The sample has a heavier tail than a normal, so its gap must be the wider
    # of the two. If this ever failed, the historical estimator and the
    # parametric one would be disagreeing about which direction the tail lies in.
    return sample_gap > assumed_gap


def main() -> int:
    print()
    results = [the_equivalence(), where_it_stops_holding(), on_real_weights()]
    if all(results):
        print("All figures reproduced.\n")
        return 0
    print("A figure did not reproduce.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())

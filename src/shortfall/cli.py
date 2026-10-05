"""Command line interface over a file of returns.

Reads a comma-separated file whose header row names the columns and whose rows
are periodic returns. A first column that is not a number is taken as a date
index and used to label dates in the drawdown output.

The design decision worth stating is about errors. A risk tool reached for by
somebody who is not its author gets pointed at the wrong file, at prices instead
of returns, and at columns of differing length, and the useful response to each
is a sentence saying which — not a traceback through the estimator that happened
to notice. So parsing validates and reports by line number, and the estimators
downstream may assume their input is well formed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from . import __version__
from .backtest import validate
from .contributions import (
    diversification_ratio,
    effective_bets,
    principal_bets,
    risk_contributions,
    risk_parity,
    volatility_contributions,
)
from .copula import Family, Marginal, copula_risk
from .covariance import ledoit_wolf, sample_covariance
from .drawdown import calmar, maximum_drawdown, sortino, ulcer_index
from .expectile import (
    matching_level,
    normal_expectile,
    sample_expectile,
    student_t_expectile,
)
from .extreme import TailMethod, extreme_risk, mean_excess_curve
from .factors import attribute_risk, fit_factor_model
from .historical import historical_risk
from .horizon import Innovations, horizon_risk
from .parametric import Distribution, normal_risk, portfolio_risk, student_t_risk
from .realised import BadBar, Bars, Estimator, realised_volatility, tick_bias_factor
from .scoring import compare, fz0_loss, quantile_loss
from .series import Panel
from .volatility import Innovation, fat_tail_test, fit_garch


class InputError(ValueError):
    """The file could not be read as a table of returns."""


@dataclass(frozen=True)
class Table:
    """A parsed returns file."""

    panel: Panel
    #: Row labels, if the file carried a non-numeric first column.
    index: tuple[str, ...] | None


def read_table(path: Path) -> Table:
    """Parse a returns CSV, reporting the line of anything malformed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise InputError(f"could not read {path}: {error}") from error

    rows = list(csv.reader(text.splitlines()))
    rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise InputError(
            f"{path} has {len(rows)} non-empty rows; a returns file needs a header "
            "of column names and at least one row of numbers"
        )
    header = [cell.strip() for cell in rows[0]]
    body = rows[1:]

    # A first column that is not a number in the first data row is a date index.
    dated = bool(body[0]) and not _is_number(body[0][0])
    names = header[1:] if dated else header
    if not names:
        raise InputError(f"{path} has no return columns after the date column")
    if len(set(names)) != len(names):
        raise InputError(f"{path} has duplicate column names: {names}")

    labels: list[str] = []
    columns: list[list[float]] = [[] for _ in names]
    for offset, row in enumerate(body):
        line = offset + 2
        cells = [cell.strip() for cell in row]
        if len(cells) != len(header):
            raise InputError(
                f"{path} line {line} has {len(cells)} fields against {len(header)} "
                "in the header; every row needs a value for every column"
            )
        if dated:
            labels.append(cells[0])
            cells = cells[1:]
        for position, cell in enumerate(cells):
            if not _is_number(cell):
                raise InputError(
                    f"{path} line {line}, column {names[position]!r}: {cell!r} is not "
                    "a number"
                )
            value = float(cell)
            if math.isnan(value) or math.isinf(value):
                raise InputError(
                    f"{path} line {line}, column {names[position]!r}: {cell!r} is not "
                    "finite"
                )
            if value <= -1.0:
                raise InputError(
                    f"{path} line {line}, column {names[position]!r}: {value!r} is at "
                    "or below -100%. This file may hold prices rather than returns; "
                    "a simple return of -1 is a total loss and nothing below it is a "
                    "return at all."
                )
            columns[position].append(value)
    _refuse_if_not_returns(path, names, columns)
    panel = Panel.from_columns(dict(zip(names, columns, strict=True)))
    return Table(panel=panel, index=tuple(labels) if dated else None)


def _refuse_if_not_returns(
    path: Path, names: Sequence[str], columns: Sequence[Sequence[float]]
) -> None:
    """Catch a file of prices, or of returns written as percentages.

    Both parse as numbers and both produce confident nonsense — a price series
    read as returns gave a one-day value at risk of 164% before this check
    existed, which is not a number anybody would have queried if it had come out
    merely large rather than absurd.

    The test is the *median* absolute value, not the maximum. A single return
    above 100% is rare but real, and refusing a file for one of them would be
    wrong; half the periods moving more than 100% is not a return series under
    any circumstances. The median ignores the first case and catches the second.
    """
    for name, column in zip(names, columns, strict=True):
        ordered = sorted(abs(value) for value in column)
        middle = ordered[len(ordered) // 2]
        if middle > 1.0:
            raise InputError(
                f"{path}: column {name!r} has a median absolute value of "
                f"{middle:.4g}, so half its periods move by more than 100%. That is "
                "not a return series. If the column holds prices, difference it "
                "first — Panel.from_prices does that; if it holds returns written "
                "as percentages, divide by 100."
            )


def _is_number(cell: str) -> bool:
    try:
        float(cell.strip())
    except ValueError:
        return False
    return True


def parse_weights(text: str | None, panel: Panel) -> list[float]:
    """Weights from a comma-separated list, or equal weights when absent.

    Not normalised. A caller who passes weights summing to 0.8 is holding 20% in
    cash and means it, and quietly scaling that up to one would report the risk
    of a portfolio they do not have.
    """
    if text is None:
        return [1.0 / panel.assets] * panel.assets
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(pieces) != panel.assets:
        raise InputError(
            f"{len(pieces)} weights against {panel.assets} columns in the file; "
            f"the columns are {', '.join(panel.names)}"
        )
    out = []
    for piece in pieces:
        if not _is_number(piece):
            raise InputError(f"{piece!r} is not a number in the weight list")
        out.append(float(piece))
    return out


def covariance_of(panel: Panel, *, shrink: bool) -> list[list[float]]:
    if shrink:
        return ledoit_wolf(panel).matrix
    return sample_covariance(panel)


# -- output ------------------------------------------------------------------


def table(rows: Sequence[Sequence[str]], stream: TextIO) -> None:
    """Print rows in aligned columns, first row being the header."""
    if not rows:
        return
    widths = [
        max(len(str(row[column])) for row in rows) for column in range(len(rows[0]))
    ]
    for position, row in enumerate(rows):
        cells = [
            str(cell).ljust(widths[column])
            if column == 0
            else str(cell).rjust(widths[column])
            for column, cell in enumerate(row)
        ]
        print("  ".join(cells).rstrip(), file=stream)
        if position == 0:
            print("  ".join("-" * width for width in widths), file=stream)


def percent(value: float) -> str:
    return f"{value * 100:.3f}%"


# -- commands ----------------------------------------------------------------


def command_risk(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    weights = parse_weights(arguments.weights, panel)
    covariance = covariance_of(panel, shrink=arguments.shrink)
    risk = portfolio_risk(
        weights,
        covariance,
        confidence=arguments.confidence,
        distribution=Distribution(arguments.distribution),
        degrees=arguments.degrees,
    )
    portfolio = panel.portfolio(weights)
    empirical = historical_risk(portfolio, confidence=arguments.confidence)
    payload = {
        "observations": panel.observations,
        "assets": panel.assets,
        "confidence": arguments.confidence,
        "distribution": risk.distribution.value,
        "volatility": risk.volatility,
        "value_at_risk": risk.value_at_risk,
        "expected_shortfall": risk.expected_shortfall,
        "historical_value_at_risk": empirical.value_at_risk,
        "historical_expected_shortfall": empirical.expected_shortfall,
        "effective_tail_sample": empirical.effective_sample,
    }
    if arguments.json:
        return payload
    print(
        f"{panel.assets} assets over {panel.observations} periods, "
        f"{arguments.confidence:.1%} confidence\n",
        file=stream,
    )
    table(
        [
            ["measure", "assumed", "historical"],
            [
                f"value at risk ({risk.distribution.value})",
                percent(risk.value_at_risk),
                percent(empirical.value_at_risk),
            ],
            [
                "expected shortfall",
                percent(risk.expected_shortfall),
                percent(empirical.expected_shortfall),
            ],
            ["period volatility", percent(risk.volatility), ""],
        ],
        stream,
    )
    print(
        f"\nThe historical tail rests on {empirical.effective_sample:.1f} "
        f"observations of {panel.observations}.",
        file=stream,
    )
    if arguments.periods:
        annual = risk.volatility * math.sqrt(arguments.periods)
        print(f"Annualised volatility {percent(annual)}.", file=stream)
    return payload


def command_contributions(
    arguments: argparse.Namespace, stream: TextIO
) -> dict[str, Any]:
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    weights = parse_weights(arguments.weights, panel)
    covariance = covariance_of(panel, shrink=arguments.shrink)
    allocation = (
        volatility_contributions(weights, covariance, names=panel.names)
        if arguments.measure == "volatility"
        else risk_contributions(
            weights,
            covariance,
            confidence=arguments.confidence,
            names=panel.names,
            of_expected_shortfall=arguments.measure == "shortfall",
        )
    )
    payload: dict[str, Any] = {
        "measure": allocation.measure,
        "total": allocation.total,
        "identity_error": allocation.identity_error,
        "positions": [
            {
                "name": name,
                "weight": weight,
                "marginal": marginal,
                "component": component,
                "share": share,
            }
            for name, weight, marginal, component, share in zip(
                allocation.names,
                allocation.weights,
                allocation.marginal,
                allocation.component,
                allocation.percentage,
                strict=True,
            )
        ],
        "diversification_ratio": None,
        "effective_bets": None,
        "principal_bets": principal_bets(weights, covariance).effective_bets,
    }
    if all(weight >= 0.0 for weight in weights):
        payload["diversification_ratio"] = diversification_ratio(weights, covariance)
    if not allocation.has_negative_contribution:
        payload["effective_bets"] = effective_bets(allocation)
    if arguments.json:
        return payload

    rows = [["position", "weight", "marginal", "contribution", "share"]]
    for entry in payload["positions"]:
        rows.append(
            [
                entry["name"],
                f"{entry['weight']:.4f}",
                percent(entry["marginal"]),
                percent(entry["component"]),
                f"{entry['share'] * 100:.1f}%",
            ]
        )
    rows.append(
        ["total", f"{math.fsum(weights):.4f}", "", percent(allocation.total), "100.0%"]
    )
    print(f"Euler allocation of {allocation.measure}\n", file=stream)
    table(rows, stream)
    print(
        f"\nComponents reconcile to the total within {allocation.identity_error:.2e}.",
        file=stream,
    )
    if payload["diversification_ratio"] is not None:
        print(
            f"Diversification ratio {payload['diversification_ratio']:.3f}.",
            file=stream,
        )
    if payload["effective_bets"] is not None:
        print(
            f"Effective bets: {payload['effective_bets']:.2f} over positions, "
            f"{payload['principal_bets']:.2f} over principal components.",
            file=stream,
        )
    else:
        print(
            "A position hedges the rest, so the risk shares are not a distribution "
            "and the bet count over positions is not defined. Over principal "
            f"components it is {payload['principal_bets']:.2f}.",
            file=stream,
        )
    return payload


def command_parity(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    covariance = covariance_of(panel, shrink=arguments.shrink)
    result = risk_parity(covariance, names=panel.names)
    payload = {
        "weights": dict(zip(result.names, result.weights, strict=True)),
        "sweeps": result.sweeps,
        "budget_error": result.budget_error,
        "converged": result.converged,
        "volatility": result.allocation.total,
    }
    if arguments.json:
        return payload
    rows = [["position", "weight", "risk share"]]
    for name, weight, share in zip(
        result.names, result.weights, result.allocation.percentage, strict=True
    ):
        rows.append([name, f"{weight:.4f}", f"{share * 100:.2f}%"])
    print("Equal risk contribution weights\n", file=stream)
    table(rows, stream)
    print(
        f"\nConverged in {result.sweeps} sweeps; the worst risk share is "
        f"{result.budget_error:.2e} from its budget.",
        file=stream,
    )
    print(f"Portfolio volatility {percent(result.allocation.total)}.", file=stream)
    return payload


def command_drawdown(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    weights = parse_weights(arguments.weights, panel)
    portfolio = panel.portfolio(weights)
    # The index labels valuations, of which there is one more than there are
    # returns; the file's labels mark the returns, so the first valuation is
    # named for where the path started rather than for a date that exists.
    labels = ["start", *parsed.index] if parsed.index is not None else None
    worst = maximum_drawdown(portfolio, index=labels)
    payload: dict[str, Any] = {
        "maximum_drawdown": worst.depth,
        "peak": str(worst.peak_label),
        "trough": str(worst.trough_label),
        "recovery": None if not worst.recovered else str(worst.recovery_label),
        "recovery_periods": worst.recovery_periods,
        "underwater_periods": worst.underwater_periods,
        "gain_needed_to_recover": worst.recovery_return,
        "ulcer_index": ulcer_index(portfolio),
        "cumulative_return": portfolio.cumulative(),
    }
    if arguments.periods:
        payload["annualised_return"] = portfolio.annualised_return(arguments.periods)
        payload["annualised_volatility"] = portfolio.annualised_volatility(
            arguments.periods
        )
        if worst.depth > 0.0:
            payload["calmar"] = calmar(portfolio, arguments.periods)
        try:
            payload["sortino"] = sortino(portfolio, arguments.periods)
        except ValueError:
            payload["sortino"] = None
    if arguments.json:
        return payload

    print("Path statistics\n", file=stream)
    rows = [
        ["statistic", "value"],
        ["cumulative return", percent(payload["cumulative_return"])],
        ["maximum drawdown", percent(worst.depth)],
        ["peak", str(worst.peak_label)],
        ["trough", str(worst.trough_label)],
        [
            "recovery",
            str(worst.recovery_label) if worst.recovered else "not within the sample",
        ],
        ["periods under water", str(worst.underwater_periods)],
        ["gain needed to recover", percent(worst.recovery_return)],
        ["ulcer index", percent(payload["ulcer_index"])],
    ]
    if arguments.periods:
        rows.append(["annualised return", percent(payload["annualised_return"])])
        rows.append(["annualised volatility", percent(payload["annualised_volatility"])])
        if "calmar" in payload:
            rows.append(["Calmar", f"{payload['calmar']:.3f}"])
        if payload.get("sortino") is not None:
            rows.append(["Sortino", f"{payload['sortino']:.3f}"])
    table(rows, stream)
    if not worst.recovered:
        print(
            "\nThe worst drawdown had not recovered by the end of the sample, so its "
            "recovery time is unknown rather than zero.",
            file=stream,
        )
    return payload


def command_factors(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    parsed = read_table(arguments.returns)
    factors = read_table(arguments.factors)
    panel = parsed.panel
    if panel.observations != factors.panel.observations:
        raise InputError(
            f"{arguments.returns} has {panel.observations} rows and "
            f"{arguments.factors} has {factors.panel.observations}; a factor model "
            "needs them aligned"
        )
    weights = parse_weights(arguments.weights, panel)
    model = fit_factor_model(panel, factors.panel)
    attribution = attribute_risk(weights, model)
    first, second, worst = model.worst_residual_correlation()
    payload = {
        "total": attribution.total,
        "factor_volatility": attribution.factor_volatility,
        "specific_volatility": attribution.specific_volatility,
        "factor_share": attribution.factor_share,
        "identity_error": attribution.identity_error,
        "exposures": dict(zip(model.factor_names, attribution.exposures, strict=True)),
        "components": dict(zip(model.factor_names, attribution.components, strict=True)),
        "worst_residual_correlation": {
            "assets": [first, second],
            "value": worst,
        },
    }
    if arguments.json:
        return payload

    print("Exposures\n", file=stream)
    rows: list[list[str]] = [["asset", *model.factor_names, "R-squared", "specific"]]
    for name in model.asset_names:
        fit = model.regression(name)
        rows.append(
            [
                name,
                *[f"{value:.3f}" for value in fit.betas],
                f"{fit.r_squared:.3f}",
                percent(fit.specific_volatility),
            ]
        )
    table(rows, stream)

    print("\nRisk attribution\n", file=stream)
    rows = [["source", "exposure", "contribution", "share"]]
    for name, exposure, component in zip(
        model.factor_names, attribution.exposures, attribution.components, strict=True
    ):
        rows.append(
            [
                name,
                f"{exposure:.4f}",
                percent(component),
                f"{component / attribution.total * 100:.1f}%",
            ]
        )
    rows.append(
        [
            "specific",
            "",
            percent(attribution.specific_contribution),
            f"{attribution.specific_contribution / attribution.total * 100:.1f}%",
        ]
    )
    rows.append(["total", "", percent(attribution.total), "100.0%"])
    table(rows, stream)
    print(
        f"\nThe factors explain {attribution.factor_share:.1%} of the variance; the "
        f"parts reconcile within {attribution.identity_error:.2e}.",
        file=stream,
    )
    print(
        f"Largest residual correlation: {first} against {second}, {worst:+.3f}. "
        + (
            "Large enough that the diagonal specific-risk assumption is doing real "
            "work here, and the implied covariance understates how much those two "
            "move together."
            if abs(worst) > 0.3
            else "Small enough for the diagonal specific-risk assumption to be "
            "tenable."
        ),
        file=stream,
    )
    return payload


def command_validate(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    """Score a forecast series against what actually happened.

    The file is a different shape from every other command's: one row per
    period carrying the realised return and the forecast that was made *for*
    that period, rather than a panel of asset returns. A forecast series is
    one-step-ahead, so no offsetting happens here — whoever built the file
    lined it up, and getting that wrong is the single easiest way to make a
    broken model look fine.
    """
    parsed = read_table(arguments.forecasts)
    panel = parsed.panel
    columns = list(panel.names)
    required = [arguments.returns_column, arguments.var_column]
    missing = [name for name in required if name not in columns]
    if missing:
        raise InputError(
            f"{arguments.forecasts} has no column named {missing[0]!r}. It has "
            f"{', '.join(repr(name) for name in columns)}. Name the columns with "
            "--returns-column and --var-column if they are called something else."
        )
    observed = list(panel[arguments.returns_column].values)
    value_at_risk = list(panel[arguments.var_column].values)
    shortfalls: list[float] | None = None
    if arguments.es_column is not None:
        if arguments.es_column not in columns:
            raise InputError(
                f"{arguments.forecasts} has no column named {arguments.es_column!r}"
            )
        shortfalls = list(panel[arguments.es_column].values)

    result = validate(
        observed,
        value_at_risk,
        confidence=arguments.confidence,
        expected_shortfall=shortfalls,
        distribution=Distribution(arguments.distribution),
        degrees=arguments.degrees,
        replications=arguments.replications,
        seed=arguments.seed,
    )
    breaches = result.exceedances
    payload: dict[str, Any] = {
        "observations": breaches.observations,
        "breaches": breaches.count,
        "expected_breaches": breaches.expected,
        "breach_rate": breaches.rate,
        "confidence": arguments.confidence,
        "tests": {
            test.name: {
                "statistic": test.statistic,
                "degrees_of_freedom": test.degrees_of_freedom,
                "p_value": test.p_value,
                "advisory": test.advisory,
            }
            for test in (result.unconditional, result.independence, result.conditional)
        },
        "traffic_light": {
            "zone": result.traffic_light.zone.value,
            "cumulative_probability": result.traffic_light.cumulative_probability,
            "plus_factor": result.traffic_light.plus_factor,
        },
        "rejected_at_5_percent": list(result.rejected_at(0.05)),
        "warnings": list(result.warnings),
    }
    if result.expected_shortfall is not None:
        payload["expected_shortfall"] = {
            "conditional": result.expected_shortfall.conditional,
            "unconditional": result.expected_shortfall.unconditional,
            "conditional_p_value": result.expected_shortfall.conditional_p_value,
            "unconditional_p_value": result.expected_shortfall.unconditional_p_value,
            "replications": result.expected_shortfall.replications,
            "direction": result.expected_shortfall.direction,
        }
    if arguments.json:
        return payload

    print("Risk model validation\n", file=stream)
    rows = [
        ["quantity", "value"],
        ["observations", str(breaches.observations)],
        ["breaches", str(breaches.count)],
        ["expected breaches", f"{breaches.expected:.2f}"],
        ["breach rate", percent(breaches.rate)],
        ["traffic light", result.traffic_light.zone.value],
    ]
    if result.traffic_light.plus_factor is not None:
        rows.append(["capital add-on", f"{result.traffic_light.plus_factor:.2f}"])
    table(rows, stream)

    print("", file=stream)
    test_rows = [["test", "statistic", "p-value", "verdict"]]
    for test in (result.unconditional, result.independence, result.conditional):
        verdict = "rejected" if test.rejects_at(0.05) else "not rejected"
        if test.advisory:
            verdict += " (advisory)"
        test_rows.append([test.name, f"{test.statistic:.4f}", f"{test.p_value:.4f}", verdict])
    table(test_rows, stream)

    for test in (result.unconditional, result.independence):
        print(f"\n{test.name}: {test.interpretation}", file=stream)

    if result.expected_shortfall is not None:
        found = result.expected_shortfall
        print(
            f"\nExpected shortfall: test 1 {found.conditional:+.4f}, "
            f"test 2 {found.unconditional:+.4f}"
            + (
                f" (p = {found.unconditional_p_value:.4f} from "
                f"{found.replications} replications)"
                if found.unconditional_p_value is not None
                else " — pass --replications for a p-value"
            ),
            file=stream,
        )
        print(f"  {found.direction}", file=stream)

    for warning in result.warnings:
        print(f"\nNote: {warning}", file=stream)
    return payload


def command_volatility(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    """Fit a conditional volatility model and report what it implies.

    Takes the same returns file every other command does, and a column to fit.
    The forecast series it prints is the one `validate` scores, which is the
    pairing the model exists for.
    """
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    names = list(panel.names)
    if arguments.column is not None:
        if arguments.column not in names:
            raise InputError(
                f"{arguments.returns} has no column named {arguments.column!r}. It has "
                f"{', '.join(repr(name) for name in names)}."
            )
        values = list(panel[arguments.column].values)
        label = arguments.column
    elif len(names) == 1:
        values = list(panel.column(0).values)
        label = names[0]
    else:
        weights = parse_weights(arguments.weights, panel)
        values = list(panel.portfolio(weights).values)
        label = "portfolio"

    choice = str(arguments.innovation)
    verdict = None
    if choice == "auto":
        # The whole reason to test rather than always fit the fatter tail: a
        # series with thin innovations gets its degrees of freedom estimated
        # anyway, lands somewhere arbitrary and large, and the quantile that
        # comes out is very slightly too wide for no reason anybody asked for.
        verdict = fat_tail_test(values, variance_targeting=arguments.variance_targeting)
        innovation = Innovation.STUDENT_T if verdict.fat else Innovation.NORMAL
    else:
        innovation = Innovation(choice)

    fitted = fit_garch(
        values,
        variance_targeting=arguments.variance_targeting,
        strict=False,
        innovation=innovation,
    )
    conditional = fitted.risk(confidence=arguments.confidence, last_return=values[-1])
    kurtosis = fitted.implied_excess_kurtosis
    horizon = int(arguments.horizon)
    payload: dict[str, Any] = {
        "series": label,
        "observations": fitted.observations,
        "omega": fitted.omega,
        "alpha": fitted.alpha,
        "beta": fitted.beta,
        "persistence": fitted.persistence,
        "halfLife": fitted.half_life,
        "longRunVolatility": fitted.long_run_volatility,
        "currentVolatility": fitted.volatilities[-1],
        "nextVolatility": math.sqrt(fitted.next_variance(values[-1])),
        "logLikelihood": fitted.log_likelihood,
        "iterations": fitted.iterations,
        "converged": fitted.converged,
        "varianceTargeted": fitted.variance_targeted,
        "horizon": horizon,
        "horizonVolatility": math.sqrt(
            fitted.horizon_variance(horizon, last_return=values[-1])
        ),
        "squareRootOfTimeRatio": fitted.scaling_against_square_root_of_time(
            horizon, last_return=values[-1]
        ),
        "innovation": fitted.innovation.value,
        "degreesOfFreedom": fitted.degrees_of_freedom if fitted.degrees_identified else None,
        # None rather than the number when the fourth moment does not exist.
        # `6 / (v - 4)` is genuinely infinite at four degrees of freedom or
        # below, and a fit landing there is not exotic — it happens on any
        # series fat enough to be worth this model. `json.dumps` writes bare
        # `Infinity` for it, which is not JSON, and a strict parser rejects the
        # whole document rather than that one field.
        "impliedExcessKurtosis": (
            kurtosis if fitted.degrees_identified and math.isfinite(kurtosis) else None
        ),
        "confidence": arguments.confidence,
        "valueAtRisk": conditional.value_at_risk,
        "expectedShortfall": conditional.expected_shortfall,
    }
    if arguments.paths:
        simulated = horizon_risk(
            fitted,
            values,
            steps=horizon,
            confidence=arguments.confidence,
            paths=int(arguments.paths),
            innovations=Innovations(arguments.draw),
            seed=arguments.seed,
        )
        payload["horizonRisk"] = {
            "paths": simulated.paths,
            "draw": arguments.draw,
            "seed": arguments.seed,
            "valueAtRisk": simulated.value_at_risk,
            "expectedShortfall": simulated.expected_shortfall,
            "standardError": simulated.standard_error,
            "relativeStandardError": simulated.relative_standard_error,
            "simulatedVolatility": simulated.simulated_volatility,
            "analyticVolatility": simulated.analytic_volatility,
            "volatilityAgainstSquareRootOfTime": (
                simulated.scaling_against_square_root_of_time
            ),
            "quantileAgainstSquareRootOfTime": (
                simulated.quantile_against_square_root_of_time
            ),
        }
    if verdict is not None:
        payload["fatTail"] = {
            "statistic": verdict.statistic,
            "pValue": verdict.p_value,
            "degreesOfFreedom": verdict.degrees_of_freedom if verdict.identified else None,
            "identified": verdict.identified,
            "fat": verdict.fat,
        }
    if arguments.json:
        return payload

    print(f"Conditional volatility — {label}\n", file=stream)
    ratio = payload["squareRootOfTimeRatio"]
    rows = [
        ["parameter", "value"],
        ["observations", str(fitted.observations)],
        ["omega", f"{fitted.omega:.6g}"],
        ["alpha", f"{fitted.alpha:.4f}"],
        ["beta", f"{fitted.beta:.4f}"],
        ["persistence", f"{fitted.persistence:.4f}"],
        ["half-life (periods)", f"{fitted.half_life:.1f}"],
        ["long-run volatility", percent(fitted.long_run_volatility)],
        ["current volatility", percent(fitted.volatilities[-1])],
        ["one step ahead", percent(payload["nextVolatility"])],
        [f"{horizon} periods ahead", percent(payload["horizonVolatility"])],
        ["against square-root-of-time", f"{ratio:.4f}"],
        ["log likelihood", f"{fitted.log_likelihood:.2f}"],
        ["converged", "yes" if fitted.converged else "NO"],
        ["innovation", fitted.innovation.value],
        [
            "degrees of freedom",
            f"{fitted.degrees_of_freedom:.2f}"
            if fitted.degrees_identified
            else "not identified",
        ],
        [f"value at risk ({arguments.confidence:.1%})", percent(conditional.value_at_risk)],
        ["expected shortfall", percent(conditional.expected_shortfall)],
    ]
    table(rows, stream)
    direction = "overstates" if ratio < 1.0 else "understates"
    print(
        f"\nScaling today's volatility by the square root of {horizon} "
        f"{direction} the horizon figure by {abs(1.0 - ratio):.1%}, because the "
        "process mean-reverts towards its long-run level and an exponentially "
        "weighted estimate has no long-run level to revert to.",
        file=stream,
    )
    if verdict is not None:
        if verdict.fat:
            print(
                f"\nThe innovations are fat-tailed: the likelihood ratio against "
                f"normal innovations is {verdict.statistic:.1f} on one degree of "
                f"freedom (p = {verdict.p_value:.2g}), at "
                f"{verdict.degrees_of_freedom:.1f} degrees of freedom. The figures "
                "above use that tail. Read against a normal quantile the same "
                "volatility would have given a smaller number.",
                file=stream,
            )
        else:
            print(
                f"\nNo fat tail was found in the innovations (likelihood ratio "
                f"{verdict.statistic:.2f}, p = {verdict.p_value:.2g}), so the "
                "figures above assume normal ones. The p-value is conservative: "
                "the null sits on the boundary of the parameter space, which makes "
                "it about twice the true probability.",
                file=stream,
            )
    if arguments.paths:
        simulated = horizon_risk(
            fitted,
            values,
            steps=horizon,
            confidence=arguments.confidence,
            paths=int(arguments.paths),
            innovations=Innovations(arguments.draw),
            seed=arguments.seed,
        )
        print(
            f"\nHorizon risk over {horizon} periods, from "
            f"{simulated.paths:,} simulated paths ({arguments.draw} innovations)\n",
            file=stream,
        )
        table(
            [
                ["figure", "value"],
                [
                    f"value at risk ({arguments.confidence:.1%})",
                    f"{percent(simulated.value_at_risk)} "
                    f"+/- {percent(simulated.standard_error)}",
                ],
                ["expected shortfall", percent(simulated.expected_shortfall)],
                ["simulated volatility", percent(simulated.simulated_volatility)],
                ["analytic volatility", percent(simulated.analytic_volatility)],
                [
                    "volatility vs square-root-of-time",
                    f"{simulated.scaling_against_square_root_of_time:.4f}",
                ],
                [
                    "quantile vs square-root-of-time",
                    f"{simulated.quantile_against_square_root_of_time:.4f}",
                ],
            ],
            stream,
        )
        print(
            "\nThe two comparisons above disagree, and the sign of the "
            "disagreement is not a constant — which is the reason to simulate "
            "rather than scale. Two effects pull against each other. The "
            "variance path is stochastic rather than its own expectation, so a "
            "large draw early raises the variance for every remaining step and "
            "the total is leptokurtic even when each innovation is normal; that "
            "pushes the quantile above the scaled figure. And summing the "
            "horizon's innovations pulls the total towards normality, while the "
            "one-step quantile keeps the whole of the innovation's own tail; "
            "that pushes it below. Measured over ten samples at ten steps, the "
            "quantile averaged 1.07 times the scaled figure under normal "
            "innovations and exceeded it on all ten; under a fitted tail near "
            "four and a half degrees of freedom it averaged 0.99 and exceeded it "
            "on four of ten. So the direction is dependable in one case and not "
            "in the other, and no multiplier on a scaled volatility is even "
            "consistently wrong.",
            file=stream,
        )

    if not fitted.converged:
        print(
            "\nThe optimiser did not converge. The parameters above are the best "
            "point it reached and should not be read as a fit; try "
            "--variance-targeting, which removes the worst-determined parameter "
            "from the search.",
            file=stream,
        )
    return payload


def command_tail(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    """Fit a generalised Pareto above a threshold and read the far tail off it.

    The command that answers a confidence the other commands cannot. `risk`
    fits a shape to the whole sample and `validate` scores one-step forecasts;
    neither has anything to say at 99.9% on a few years of daily data, because
    the answer there is two observations under one route and an extrapolation of
    the body under the other.
    """
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    names = list(panel.names)
    if arguments.column is not None:
        if arguments.column not in names:
            raise InputError(
                f"{arguments.returns} has no column named {arguments.column!r}. It has "
                f"{', '.join(repr(name) for name in names)}."
            )
        values = list(panel[arguments.column].values)
        label = arguments.column
    elif len(names) == 1:
        values = list(panel.column(0).values)
        label = names[0]
    else:
        weights = parse_weights(arguments.weights, panel)
        values = list(panel.portfolio(weights).values)
        label = "portfolio"

    estimate = extreme_risk(
        values,
        confidence=arguments.confidence,
        tail_fraction=arguments.tail_fraction,
        threshold=arguments.threshold,
        method=TailMethod(arguments.method),
    )
    fit = estimate.fit
    historical = historical_risk(values, confidence=arguments.confidence)
    payload: dict[str, Any] = {
        "series": label,
        "observations": fit.observations,
        "method": fit.method.value,
        "threshold": fit.threshold,
        "exceedances": fit.exceedances,
        "shape": fit.shape,
        "scale": fit.scale,
        "shapeStandardError": fit.shape_standard_error,
        "scaleStandardError": fit.scale_standard_error,
        "logLikelihood": fit.log_likelihood,
        "upperEndpoint": fit.upper_endpoint,
        "lowestConfidence": fit.lowest_confidence,
        "confidence": arguments.confidence,
        "valueAtRisk": estimate.value_at_risk,
        "expectedShortfall": estimate.expected_shortfall,
        "observedBeyond": estimate.observed_beyond,
        "isExtrapolated": estimate.is_extrapolated,
        "historicalValueAtRisk": historical.value_at_risk,
        "historicalTailObservations": historical.tail_observations,
    }
    if arguments.curve:
        payload["meanExcess"] = [
            {
                "threshold": point.threshold,
                "exceedances": point.exceedances,
                "meanExcess": point.mean_excess,
                "standardError": point.standard_error,
            }
            for point in mean_excess_curve(
                [-value for value in values], points=arguments.curve
            )
        ]
    if arguments.json:
        return payload

    print(f"Fitted tail — {label}\n", file=stream)
    rows = [
        ["parameter", "value"],
        ["observations", str(fit.observations)],
        ["method", fit.method.value.replace("_", " ")],
        ["threshold (loss)", percent(fit.threshold)],
        ["exceedances", f"{fit.exceedances} ({fit.exceedance_probability:.2%})"],
        [
            "shape",
            f"{fit.shape:+.4f}"
            + (
                f" +/- {fit.shape_standard_error:.4f}"
                if fit.shape_standard_error is not None
                else ""
            ),
        ],
        [
            "scale",
            f"{fit.scale:.6g}"
            + (
                f" +/- {fit.scale_standard_error:.6g}"
                if fit.scale_standard_error is not None
                else ""
            ),
        ],
        ["log likelihood", f"{fit.log_likelihood:.2f}"],
        [
            "largest possible loss",
            percent(fit.upper_endpoint) if fit.upper_endpoint is not None else "unbounded",
        ],
        [f"value at risk ({arguments.confidence:.3%})", percent(estimate.value_at_risk)],
        [
            "expected shortfall",
            percent(estimate.expected_shortfall)
            if estimate.expected_shortfall is not None
            else "no finite mean",
        ],
        ["historical value at risk", percent(historical.value_at_risk)],
        ["observations beyond the estimate", str(estimate.observed_beyond)],
    ]
    table(rows, stream)
    if estimate.is_extrapolated:
        print(
            f"\nNothing in the sample is as bad as this figure, which is what the "
            f"estimator is for and also the reason to be careful with it: "
            f"{fit.exceedances} exceedances are deciding what happens beyond all "
            f"{fit.observations} observations. The historical figure beside it "
            f"cannot exceed the worst loss in the file and is reported from "
            f"{historical.tail_observations} observation(s).",
            file=stream,
        )
    if fit.upper_endpoint is not None:
        print(
            "\nThe fitted shape is negative, so the tail ends. A bounded loss "
            "from market data is more often a threshold set inside the body than "
            "a finding; raise --tail-fraction's exceedance count or set "
            "--threshold from the mean excess curve before quoting it.",
            file=stream,
        )
    if arguments.curve:
        print("\nMean excess against threshold\n", file=stream)
        table(
            [["threshold", "exceedances", "mean excess", "standard error"]]
            + [
                [
                    percent(point["threshold"]),
                    str(point["exceedances"]),
                    percent(point["meanExcess"]),
                    percent(point["standardError"]),
                ]
                for point in payload["meanExcess"]
            ],
            stream,
        )
        print(
            "\nA generalised Pareto has a mean excess that is linear in the "
            "threshold, with slope shape/(1 - shape). So the threshold to fit "
            "above is where the curve straightens out, and the standard errors "
            "are there because successive points share nearly all of their data "
            "and the curve is far smoother than its points are independent.",
            file=stream,
        )
    return payload


def command_copula(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    weights = parse_weights(arguments.weights, panel)
    result = copula_risk(
        panel,
        weights,
        confidence=arguments.confidence,
        family=Family(arguments.family),
        marginal=Marginal(arguments.marginal),
        degrees_of_freedom=arguments.degrees,
        paths=arguments.paths,
        seed=arguments.seed,
    )
    fitted = result.copula
    pairs = fitted.tail_dependence()
    payload: dict[str, Any] = {
        "observations": panel.observations,
        "assets": panel.assets,
        "confidence": arguments.confidence,
        "family": fitted.family.value,
        "marginal": result.marginal.value,
        "paths": result.paths,
        "degrees_of_freedom": fitted.degrees_of_freedom,
        "likelihood_ratio": fitted.likelihood_ratio,
        "projected": fitted.projected,
        "value_at_risk": result.value_at_risk,
        "expected_shortfall": result.expected_shortfall,
        "standard_error": result.standard_error,
        "gaussian_value_at_risk": result.gaussian_value_at_risk,
        "gaussian_expected_shortfall": result.gaussian_expected_shortfall,
        "tail_dependence": [
            {
                "first": pair.first,
                "second": pair.second,
                "correlation": pair.correlation,
                "coefficient": pair.coefficient,
            }
            for pair in pairs
        ],
    }
    if fitted.family is Family.STUDENT_T:
        payload["tail_dependence_premium"] = result.tail_dependence_premium
    if arguments.json:
        return payload
    label = (
        "Student-t"
        if fitted.family is Family.STUDENT_T
        else "Gaussian"
    )
    degrees = (
        f" at {fitted.degrees_of_freedom:.1f} degrees of freedom"
        if fitted.degrees_of_freedom is not None
        else ""
    )
    print(
        f"{panel.assets} assets over {panel.observations} periods, "
        f"{label} copula{degrees}, {result.paths:,} paths, "
        f"{arguments.confidence:.1%} confidence\n",
        file=stream,
    )
    table(
        [
            ["measure", "fitted copula", "gaussian copula"],
            [
                "value at risk",
                percent(result.value_at_risk),
                percent(result.gaussian_value_at_risk),
            ],
            [
                "expected shortfall",
                percent(result.expected_shortfall),
                percent(result.gaussian_expected_shortfall),
            ],
        ],
        stream,
    )
    print(
        f"\nMonte Carlo standard error on the value at risk "
        f"{percent(result.standard_error)}, from 20 batches; read it as a lower "
        f"bound.",
        file=stream,
    )
    if fitted.family is Family.STUDENT_T:
        print(
            f"Expected shortfall is {result.tail_dependence_premium:+.1%} against "
            f"the same marginals and the same correlation matrix under a Gaussian "
            f"copula, which has no tail dependence at any correlation below one. "
            f"The two figures share their normal draws, so the difference is the "
            f"dependence assumption and not simulation noise.",
            file=stream,
        )
        print(
            f"Likelihood ratio against the Gaussian special case "
            f"{fitted.likelihood_ratio:.1f}.",
            file=stream,
        )
    if fitted.projected:
        print(
            "The pairwise rank inversion was not a valid correlation matrix and "
            "has been projected onto the nearest one, so the correlations below "
            "are not exactly the ones estimated.",
            file=stream,
        )
    shown = sorted(pairs, key=lambda pair: -pair.coefficient)[: arguments.pairs]
    if shown:
        print("", file=stream)
        table(
            [
                ["pair", "correlation", "tail dependence"],
                *[
                    [
                        f"{pair.first} / {pair.second}",
                        f"{pair.correlation:+.3f}",
                        f"{pair.coefficient:.3f}",
                    ]
                    for pair in shown
                ],
            ],
            stream,
        )
        print(
            "\nTail dependence is the limiting probability that one of a pair is "
            "beyond its own quantile given that the other is. A Gaussian copula "
            "puts it at zero for every correlation below one; that is a statement "
            "about quantiles nobody has data for, which are the ones being "
            "extrapolated to.",
            file=stream,
        )
    return payload


def read_bars(path: Path) -> Bars:
    """Parse an OHLC CSV, reporting the line of anything malformed.

    Four numeric columns in the order open, high, low, close, with an optional
    leading date column and an optional header. The header is detected the
    same way the returns reader detects one: a first row whose cells are not
    all numbers.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise InputError(f"could not read {path}: {error}") from error

    rows = [row for row in csv.reader(text.splitlines()) if row and any(c.strip() for c in row)]
    if not rows:
        raise InputError(f"{path} is empty")
    if not all(_is_number(cell.strip()) for cell in rows[0][-4:]):
        rows = rows[1:]
    if not rows:
        raise InputError(f"{path} has a header and no data")

    values: list[tuple[float, float, float, float]] = []
    for offset, row in enumerate(rows):
        line = offset + 1
        cells = [cell.strip() for cell in row]
        if len(cells) < 4:
            raise InputError(
                f"{path} line {line} has {len(cells)} fields; a bar needs four, "
                "open, high, low and close, after any date column"
            )
        # A leading date column is dropped rather than parsed: nothing here
        # needs the date, and refusing a file for the format of a column that
        # is not used would be unhelpful.
        quartet = cells[-4:]
        for name, cell in zip(("open", "high", "low", "close"), quartet, strict=True):
            if not _is_number(cell):
                raise InputError(f"{path} line {line}, {name}: {cell!r} is not a number")
        numbers = [float(cell) for cell in quartet]
        values.append((numbers[0], numbers[1], numbers[2], numbers[3]))

    try:
        return Bars.from_rows(str(path), values)
    except BadBar as error:
        raise InputError(str(error)) from error


def command_bars(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    """Every estimator on one file, so the disagreement between them is visible.

    The disagreement is the output. Two estimators of the same quantity that
    differ by a third are saying something about the data -- a trend, an
    overnight gap, or too few trades in the bar -- and which one is right
    depends on which of those it is.
    """
    bars = read_bars(arguments.bars)
    periods = arguments.periods
    estimates = {
        estimator.value: realised_volatility(bars, periods, estimator)
        for estimator in Estimator
        if len(bars) >= _MINIMUM_BARS[estimator]
    }
    baseline = estimates.get(Estimator.CLOSE_TO_CLOSE.value)
    payload: dict[str, Any] = {
        "bars": len(bars),
        "periods_per_year": periods,
        "annualised_volatility": estimates,
    }
    if arguments.ticks is not None:
        factor = tick_bias_factor(arguments.ticks)
        payload["tick_bias_factor"] = factor
        payload["corrected"] = {
            name: value / math.sqrt(factor)
            for name, value in estimates.items()
            if name != Estimator.CLOSE_TO_CLOSE.value
        }
    if arguments.json:
        return payload

    print(f"Annualised volatility from {len(bars)} bars\n", file=stream)
    rows = [["estimator", "volatility", "vs close-to-close", "assumes"]]
    for estimator in Estimator:
        name = estimator.value
        if name not in estimates:
            continue
        value = estimates[name]
        ratio = "-" if not baseline else f"{value / baseline:.3f}"
        rows.append([name, percent(value), ratio, _ASSUMES[estimator]])
    table(rows, stream)

    if arguments.ticks is not None:
        factor = payload["tick_bias_factor"]
        print(
            f"\nAt {arguments.ticks} price observations a bar, a range-based variance "
            f"is expected to read {factor:.4f} of the truth, because the observed high "
            f"and low sit inside the continuous ones. Dividing by the square root of "
            f"that:\n",
            file=stream,
        )
        corrected = [["estimator", "corrected volatility"]]
        for name, value in payload["corrected"].items():
            corrected.append([name, percent(value)])
        table(corrected, stream)
    return payload


_MINIMUM_BARS = {
    Estimator.CLOSE_TO_CLOSE: 3,
    Estimator.PARKINSON: 1,
    Estimator.GARMAN_KLASS: 1,
    Estimator.ROGERS_SATCHELL: 1,
    Estimator.GARMAN_KLASS_YANG_ZHANG: 2,
    Estimator.YANG_ZHANG: 4,
}

_ASSUMES = {
    Estimator.CLOSE_TO_CLOSE: "nothing; uses one price in four",
    Estimator.PARKINSON: "no drift, no overnight gap",
    Estimator.GARMAN_KLASS: "no drift, no overnight gap",
    Estimator.ROGERS_SATCHELL: "no overnight gap",
    Estimator.GARMAN_KLASS_YANG_ZHANG: "no drift",
    Estimator.YANG_ZHANG: "nothing; uses all four and the previous close",
}


def command_score(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    """Rank two forecast series by a strictly consistent score.

    ``validate`` asks whether one model is adequate. This asks which of two is
    better, which is a different question and needs a different instrument: a
    score that is minimised, uniquely, at the truth. Expected shortfall has no
    such score of its own, so the pair is scored jointly by the
    Fissler-Ziegel function; dropping the shortfall columns falls back to the
    pinball loss, which is consistent for the quantile alone.

    The comparison prints both standard errors. The autocorrelation-robust one
    is what the test uses, and the naive one is beside it because the gap
    between them is small — measured between 0.99 and 1.10 times — and a
    reader is entitled to see that rather than take the bandwidth on trust.
    """
    parsed = read_table(arguments.forecasts)
    panel = parsed.panel
    columns = list(panel.names)

    def column(name: str) -> list[float]:
        if name not in columns:
            raise InputError(
                f"{arguments.forecasts} has no column named {name!r}. It has "
                f"{', '.join(repr(each) for each in columns)}."
            )
        return list(panel[name].values)

    observed = column(arguments.returns_column)
    joint = arguments.es_columns is not None
    if joint:
        first = fz0_loss(
            observed,
            column(arguments.var_columns[0]),
            column(arguments.es_columns[0]),
            confidence=arguments.confidence,
        )
        second = fz0_loss(
            observed,
            column(arguments.var_columns[1]),
            column(arguments.es_columns[1]),
            confidence=arguments.confidence,
        )
    else:
        first = quantile_loss(
            observed, column(arguments.var_columns[0]), confidence=arguments.confidence
        )
        second = quantile_loss(
            observed, column(arguments.var_columns[1]), confidence=arguments.confidence
        )

    result = compare(first, second, lags=arguments.lags)
    names = (arguments.var_columns[0], arguments.var_columns[1])
    payload: dict[str, Any] = {
        "score": first.name,
        "confidence": arguments.confidence,
        "observations": result.observations,
        "models": [
            {"name": names[0], "mean_score": first.mean},
            {"name": names[1], "mean_score": second.mean},
        ],
        "difference": result.difference,
        "standard_error": result.standard_error,
        "naive_standard_error": result.naive_standard_error,
        "robust_over_naive": (
            result.standard_error / result.naive_standard_error
            if result.naive_standard_error > 0.0
            else float("nan")
        ),
        "statistic": result.statistic,
        "p_value": result.p_value,
        "lags": result.lags,
        "better": None if result.better is None else names[result.better - 1],
    }
    if arguments.json:
        return payload

    print(f"score: {first.name} at {arguments.confidence:.4g}", file=stream)
    print(f"observations: {result.observations}", file=stream)
    print(f"{'model':>24}{'mean score':>16}", file=stream)
    print(f"{names[0]:>24}{first.mean:>16.8f}", file=stream)
    print(f"{names[1]:>24}{second.mean:>16.8f}", file=stream)
    print(
        f"difference: {result.difference:+.8f}, standard error {result.standard_error:.8f} "
        f"at {result.lags} lags ({payload['robust_over_naive']:.4f} times the naive "
        f"{result.naive_standard_error:.8f})",
        file=stream,
    )
    print(
        f"Diebold-Mariano: {result.statistic:+.4f}, p = {result.p_value:.4g}", file=stream
    )
    if result.better is None:
        print("The two score identically.", file=stream)
    else:
        print(f"Lower score, so better: {names[result.better - 1]}", file=stream)
    print(
        "Lower is better for both scores, and two scores from different "
        "functions are not comparable: the ranking is only meaningful within "
        "one of them.",
        file=stream,
    )
    return payload


def command_expectile(arguments: argparse.Namespace, stream: TextIO) -> dict[str, Any]:
    """The expectile of a portfolio, and the level that states it in familiar units.

    An expectile level means nothing on its own, so the command spends most of
    its output on translation: which expected shortfall this expectile equals,
    and which level would reproduce the expected shortfall the caller asked
    for. The two are not inverses of each other across distributions, and the
    table is laid out so that is visible rather than buried — the level matched
    on the fitted normal is carried over to the fitted Student-t, and the gap
    is the number worth looking at.
    """
    parsed = read_table(arguments.returns)
    panel = parsed.panel
    weights = parse_weights(arguments.weights, panel)
    portfolio = panel.portfolio(weights)
    # Losses are positive, which is the package's convention everywhere else.
    losses = [-value for value in portfolio.values]
    volatility = portfolio.stdev()
    mean = math.fsum(losses) / len(losses)

    empirical = sample_expectile(losses, arguments.level)
    assumed = normal_expectile(mean=mean, volatility=volatility, level=arguments.level)
    heavy = student_t_expectile(
        mean=mean,
        volatility=volatility,
        level=arguments.level,
        degrees=arguments.degrees,
    )
    risk = normal_risk(mean=-mean, volatility=volatility, confidence=arguments.confidence)
    fat = student_t_risk(
        mean=-mean,
        volatility=volatility,
        confidence=arguments.confidence,
        degrees=arguments.degrees,
    )
    matched = matching_level(
        risk.expected_shortfall,
        lambda level: normal_expectile(mean=mean, volatility=volatility, level=level).value,
    )
    transplanted = student_t_expectile(
        mean=mean, volatility=volatility, level=matched.level, degrees=arguments.degrees
    )
    payload = {
        "observations": panel.observations,
        "level": arguments.level,
        "confidence": arguments.confidence,
        "degrees": arguments.degrees,
        "sample_expectile": empirical.value,
        "normal_expectile": assumed.value,
        "student_t_expectile": heavy.value,
        "exceedance_ratio": empirical.exceedance_ratio,
        "identity_residual": empirical.identity,
        "normal_expected_shortfall": risk.expected_shortfall,
        "student_t_expected_shortfall": fat.expected_shortfall,
        "matched_level": matched.level,
        "transplanted_expectile": transplanted.value,
        "transplant_overstatement": transplanted.value / fat.expected_shortfall - 1.0,
    }
    if arguments.json:
        return payload
    print(
        f"{panel.observations} periods, expectile level {arguments.level:.5f}, "
        f"confidence {arguments.confidence:.1%}\n",
        file=stream,
    )
    table(
        [
            ["expectile", "value"],
            ["sample", percent(empirical.value)],
            ["normal", percent(assumed.value)],
            [f"Student-t ({arguments.degrees:g} df)", percent(heavy.value)],
        ],
        stream,
    )
    print(
        f"\nThe sample expectile is the point where expected overshoot is "
        f"{empirical.exceedance_ratio:.4f} times expected undershoot, which is "
        f"{(1.0 - arguments.level) / arguments.level:.4f} by construction. "
        f"It is not a quantile and no share of the sample sits beyond it.",
        file=stream,
    )
    print(
        f"\nLevel {matched.level:.6f} reproduces the {arguments.confidence:.1%} "
        f"expected shortfall of {percent(risk.expected_shortfall)} under the fitted "
        f"normal.",
        file=stream,
    )
    print(
        f"Carried over to the fitted Student-t, that level gives "
        f"{percent(transplanted.value)} against a true expected shortfall of "
        f"{percent(fat.expected_shortfall)}: "
        f"{payload['transplant_overstatement']:+.1%}. An expectile level is not "
        f"transferable between distributions the way a confidence level is.",
        file=stream,
    )
    return payload


COMMANDS = {
    "risk": command_risk,
    "contributions": command_contributions,
    "parity": command_parity,
    "drawdown": command_drawdown,
    "factors": command_factors,
    "validate": command_validate,
    "volatility": command_volatility,
    "tail": command_tail,
    "copula": command_copula,
    "bars": command_bars,
    "score": command_score,
    "expectile": command_expectile,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shortfall",
        description=(
            "Portfolio risk over a file of returns. The file has a header row of "
            "column names and one row per period; a leading non-numeric column is "
            "taken as dates."
        ),
    )
    # Worth having for its own sake, and it doubles as the cheapest possible
    # smoke test of an install: it imports the package and prints something.
    parser.add_argument("--version", action="version", version=f"shortfall {__version__}")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable output"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(
        sub: argparse.ArgumentParser, *, weights: bool = True, shrink: bool = True
    ) -> None:
        sub.add_argument("returns", type=Path, help="CSV of periodic returns")
        if weights:
            sub.add_argument(
                "--weights",
                help="comma-separated portfolio weights; equal weights if omitted. "
                "Not normalised: weights summing to 0.8 mean 20% in cash.",
            )
        if shrink:
            sub.add_argument(
                "--shrink",
                action="store_true",
                help="use the Ledoit-Wolf shrinkage covariance instead of the "
                "sample one",
            )

    risk = subparsers.add_parser("risk", help="value at risk and expected shortfall")
    common(risk)
    risk.add_argument("--confidence", type=float, default=0.99)
    risk.add_argument(
        "--distribution",
        default="normal",
        choices=["normal", "student-t", "cornish-fisher"],
    )
    risk.add_argument("--degrees", type=float, default=5.0)
    risk.add_argument(
        "--periods", type=float, default=None, help="periods per year, for annualising"
    )

    contributions = subparsers.add_parser(
        "contributions", help="Euler allocation of risk across positions"
    )
    common(contributions)
    contributions.add_argument("--confidence", type=float, default=0.99)
    contributions.add_argument(
        "--measure", default="volatility", choices=["volatility", "var", "shortfall"]
    )

    parity = subparsers.add_parser("parity", help="equal risk contribution weights")
    common(parity, weights=False)

    path = subparsers.add_parser("drawdown", help="drawdown and path statistics")
    common(path)
    path.add_argument("--periods", type=float, default=None, help="periods per year")

    factors = subparsers.add_parser("factors", help="factor exposures and attribution")
    common(factors)
    factors.add_argument(
        "--factors", type=Path, required=True, help="CSV of factor returns"
    )

    moving = subparsers.add_parser(
        "volatility",
        help="fit a GARCH(1,1) and report what it forecasts",
        description=(
            "Fits a conditional volatility model by maximum likelihood. The "
            "horizon figure is what square-root-of-time approximates, and the "
            "ratio between them says by how much and in which direction."
        ),
    )
    common(moving)
    moving.add_argument(
        "--column", default=None, help="fit this column rather than the portfolio"
    )
    moving.add_argument(
        "--horizon", type=int, default=10, help="periods ahead to aggregate. Defaults to 10."
    )
    moving.add_argument(
        "--innovation",
        choices=["auto", "normal", "student-t"],
        default="auto",
        help="the shape assumed for the standardised residuals. 'auto' tests "
        "whether a fat tail is there by likelihood ratio and uses one only if it "
        "is. Defaults to auto.",
    )
    moving.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="confidence for the conditional value at risk and expected "
        "shortfall. Defaults to 0.99.",
    )
    moving.add_argument(
        "--paths",
        type=int,
        default=0,
        help="simulate the horizon risk from this many paths. Off by default. "
        "The horizon quantile is not analytic — the sum of the innovations is "
        "not a member of the family they came from — so this is the only honest "
        "route to a multi-period figure.",
    )
    moving.add_argument(
        "--draw",
        choices=["bootstrap", "parametric"],
        default="bootstrap",
        help="where simulated innovations come from. 'bootstrap' resamples the "
        "model's own standardised residuals and assumes no tail shape; "
        "'parametric' draws from the fitted distribution. Defaults to bootstrap.",
    )
    moving.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the simulation, so two identical calls agree. Defaults to 0.",
    )
    moving.add_argument(
        "--variance-targeting",
        action="store_true",
        help="fix the long-run variance to the sample variance and estimate only "
        "the two dynamic parameters, which is more robust on a short sample",
    )

    extreme = subparsers.add_parser(
        "tail",
        help="fit a generalised Pareto to the exceedances and read the far tail",
        description=(
            "Fits the tail on its own rather than fitting the whole distribution "
            "and asking about the tail. The figure it returns can exceed every "
            "loss in the file, which is the point; the report says how many "
            "observations the extrapolation rests on."
        ),
    )
    common(extreme)
    extreme.add_argument(
        "--column", default=None, help="fit this column rather than the portfolio"
    )
    extreme.add_argument(
        "--confidence",
        type=float,
        default=0.999,
        help="confidence for the value at risk and expected shortfall. Defaults "
        "to 0.999, which is where this estimator earns its keep.",
    )
    extreme.add_argument(
        "--tail-fraction",
        type=float,
        default=0.05,
        help="fraction of the sample to fit above. Defaults to 0.05, which is "
        "where the applied literature starts rather than a recommendation: use "
        "--curve and then --threshold.",
    )
    extreme.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="fit above this loss instead, as a positive fraction. 0.02 is a 2% loss.",
    )
    extreme.add_argument(
        "--method",
        choices=[method.value for method in TailMethod],
        default=TailMethod.MAXIMUM_LIKELIHOOD.value,
        help="how to estimate the two parameters. Maximum likelihood is the "
        "default and the only one with a usable standard error; the moment "
        "estimator is steadier on a few dozen exceedances and cannot report a "
        "shape at or above one at all.",
    )
    extreme.add_argument(
        "--curve",
        type=int,
        default=0,
        metavar="POINTS",
        help="also print the mean excess curve with this many points, which is "
        "how the threshold gets chosen rather than defaulted into",
    )

    dependence = subparsers.add_parser(
        "copula",
        help="portfolio risk with the dependence fitted separately from the marginals",
        description=(
            "Every other multi-asset route here ties the joint distribution to a "
            "covariance matrix, which forces the probability of two assets being "
            "in their own tails together to zero under a normal and to one number "
            "for every pair under a multivariate t. This fits the dependence to "
            "the ranks and each marginal on its own, then puts them back together "
            "by simulation, and prints the Gaussian-copula figure beside its own "
            "so the assumption can be priced rather than argued about."
        ),
    )
    common(dependence, shrink=False)
    dependence.add_argument("--confidence", type=float, default=0.99)
    dependence.add_argument(
        "--family",
        choices=[family.value for family in Family],
        default=Family.STUDENT_T.value,
        help="which elliptical copula to fit. The Student-t has one extra "
        "parameter, the degrees of freedom, and it is what carries joint tail "
        "arrival; the Gaussian is the special case it is compared against.",
    )
    dependence.add_argument(
        "--marginal",
        choices=[marginal.value for marginal in Marginal],
        default=Marginal.EMPIRICAL.value,
        help="where each asset's own distribution comes from. Empirical assumes "
        "no shape and cannot produce a draw past that asset's worst observation; "
        "extreme_value splices a fitted generalised Pareto onto the loss tail, "
        "which extrapolates deliberately.",
    )
    dependence.add_argument(
        "--degrees",
        type=float,
        default=None,
        help="fix the copula degrees of freedom instead of fitting them, which is "
        "how to ask what a heavier joint tail would cost rather than what this "
        "sample supports",
    )
    dependence.add_argument(
        "--paths",
        type=int,
        default=40_000,
        help="simulated paths. The default takes a few seconds; the reported "
        "standard error is what says whether it was enough.",
    )
    dependence.add_argument(
        "--pairs",
        type=int,
        default=10,
        help="how many pairs of the tail dependence table to print, worst first",
    )
    dependence.add_argument("--seed", type=int, default=0)

    ranked = subparsers.add_parser(
        "score",
        help="rank two forecast series by a strictly consistent score",
        description=(
            "'validate' asks whether one model is adequate. This asks which of "
            "two is better, which needs a scoring function that is minimised at "
            "the truth rather than a hypothesis test. Expected shortfall has no "
            "such score of its own, so the pair is scored jointly by the "
            "Fissler-Ziegel function; with no shortfall columns it falls back "
            "to the pinball loss, which is consistent for the quantile alone. "
            "The obvious alternative -- a pinball loss plus a squared error on "
            "the breaches -- is not consistent, and its optimum sits about a "
            "third above the truth, which is why it is not offered."
        ),
    )
    ranked.add_argument("forecasts", type=Path, help="CSV of returns and forecasts")
    ranked.add_argument(
        "--returns-column", default="return", help="column holding the realised return"
    )
    ranked.add_argument(
        "--var-columns",
        nargs=2,
        default=("var_a", "var_b"),
        metavar=("FIRST", "SECOND"),
        help="the two value-at-risk columns to compare, as positive losses",
    )
    ranked.add_argument(
        "--es-columns",
        nargs=2,
        default=None,
        metavar=("FIRST", "SECOND"),
        help=(
            "the two expected-shortfall columns; omit to score the quantile "
            "alone with the pinball loss"
        ),
    )
    ranked.add_argument("--confidence", type=float, default=0.975)
    ranked.add_argument(
        "--lags",
        type=int,
        default=None,
        help="bandwidth for the long-run variance; the default is the usual rule",
    )

    checked = subparsers.add_parser(
        "validate",
        help="score a value-at-risk forecast series against realised returns",
        description=(
            "Takes a CSV with one row per period holding the realised return and "
            "the forecast made for that period. The rows are used as they stand: "
            "a forecast series is one-step-ahead, and lining it up is the caller's "
            "job because only the caller knows how the file was built."
        ),
    )
    checked.add_argument("forecasts", type=Path, help="CSV of returns and forecasts")
    checked.add_argument(
        "--returns-column", default="return", help="column holding the realised return"
    )
    checked.add_argument(
        "--var-column",
        default="var",
        help="column holding the value-at-risk forecast, as a positive loss",
    )
    checked.add_argument(
        "--es-column",
        default=None,
        help="column holding the expected-shortfall forecast; omit to skip those tests",
    )
    checked.add_argument("--confidence", type=float, default=0.99)
    checked.add_argument(
        "--replications",
        type=int,
        default=0,
        help="simulate the expected-shortfall null this many times for a p-value",
    )
    checked.add_argument(
        "--distribution",
        default="normal",
        choices=["normal", "student-t"],
        help="the predictive distribution the forecasts were built under, "
        "which is what the simulated null draws from",
    )
    checked.add_argument("--degrees", type=float, default=5.0)
    checked.add_argument("--seed", type=int, default=0)
    expectile = subparsers.add_parser(
        "expectile",
        help="the coherent and elicitable risk measure, and what its level means",
        description=(
            "Reports the sample expectile of a portfolio beside the normal and "
            "Student-t closed forms, translates the level into the expected "
            "shortfall it reproduces, and then carries that level over to a "
            "fatter-tailed fit to show what the translation costs. The last "
            "number is the point: a confidence level means the same thing on "
            "every book and an expectile level does not."
        ),
    )
    expectile.add_argument("returns", type=Path, help="CSV of returns")
    expectile.add_argument(
        "--weights", default=None, help="portfolio weights; equal weights if omitted"
    )
    expectile.add_argument(
        "--level", type=float, default=0.99, help="the expectile level, tau"
    )
    expectile.add_argument(
        "--confidence",
        type=float,
        default=0.975,
        help="the expected-shortfall confidence the level is matched against",
    )
    expectile.add_argument("--degrees", type=float, default=5.0)

    bars = subparsers.add_parser(
        "bars",
        help="volatility from open, high, low and close",
        description=(
            "Reads a CSV of open, high, low and close -- with an optional leading "
            "date column and an optional header -- and reports every estimator in "
            "the library beside the close-to-close baseline. The disagreement "
            "between them is the output: a trend inflates the estimators that "
            "assume no drift, an overnight gap deflates the ones that only look "
            "inside the bar, and too few trades deflates all of them."
        ),
    )
    bars.add_argument("--bars", type=Path, required=True, help="CSV of OHLC rows")
    bars.add_argument(
        "--periods", type=float, default=252.0, help="bars per year, for annualising"
    )
    bars.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="price observations within a bar; reports the discretisation "
        "correction alongside",
    )
    bars.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None, stream: TextIO | None = None) -> int:
    out = stream if stream is not None else sys.stdout
    arguments = build_parser().parse_args(argv)
    try:
        payload = COMMANDS[arguments.command](arguments, out)
    except (InputError, ValueError) as error:
        print(f"shortfall: {error}", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(payload, indent=2, default=str), file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

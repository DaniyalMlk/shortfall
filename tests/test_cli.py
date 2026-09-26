"""The command line interface, and the input validation in front of it.

Most of what is tested here is refusal. The estimators are covered elsewhere;
what the command line adds is everything that happens when somebody points it at
the wrong file, and the useful behaviour there is a sentence naming the problem
rather than a traceback through a covariance routine.
"""

from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import pytest

from shortfall.cli import InputError, main, parse_weights, read_table
from shortfall.series import Panel


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def sample_file(path: Path, *, periods: int = 300, dated: bool = True) -> Path:
    rng = random.Random(5)
    market = [rng.gauss(0.0004, 0.011) for _ in range(periods)]
    lines = ["date,alpha,beta,gamma"] if dated else ["alpha,beta,gamma"]
    for t in range(periods):
        row = [
            f"{1.1 * market[t] + rng.gauss(0.0, 0.006):.8f}",
            f"{0.7 * market[t] + rng.gauss(0.0, 0.004):.8f}",
            f"{-0.2 * market[t] + rng.gauss(0.0, 0.008):.8f}",
        ]
        if dated:
            row.insert(0, f"2024-{(t % 12) + 1:02d}-{(t % 28) + 1:02d}")
        lines.append(",".join(row))
    return write(path, "\n".join(lines) + "\n")


def run(*argv: str) -> tuple[int, str]:
    stream = io.StringIO()
    code = main(list(argv), stream=stream)
    return code, stream.getvalue()


# -- parsing -----------------------------------------------------------------


def test_a_dated_file_keeps_its_labels(tmp_path: Path) -> None:
    table = read_table(sample_file(tmp_path / "r.csv"))
    assert table.index is not None
    assert len(table.index) == 300
    assert table.panel.names == ["alpha", "beta", "gamma"]


def test_an_undated_file_has_no_index(tmp_path: Path) -> None:
    table = read_table(sample_file(tmp_path / "r.csv", dated=False))
    assert table.index is None
    assert table.panel.assets == 3


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = write(tmp_path / "r.csv", "a,b\n0.01,0.02\n\n\n0.03,-0.01\n\n")
    assert read_table(path).panel.observations == 2


def test_a_missing_file_is_reported_as_such(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="could not read"):
        read_table(tmp_path / "absent.csv")


def test_a_file_with_no_rows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="needs a header"):
        read_table(write(tmp_path / "r.csv", "a,b\n"))


def test_a_ragged_row_names_its_line(tmp_path: Path) -> None:
    path = write(tmp_path / "r.csv", "a,b\n0.01,0.02\n0.03\n")
    with pytest.raises(InputError, match="line 3 has 1 fields"):
        read_table(path)


def test_a_non_numeric_cell_names_its_line_and_column(tmp_path: Path) -> None:
    path = write(tmp_path / "r.csv", "a,b\n0.01,0.02\n0.03,oops\n")
    with pytest.raises(InputError, match="line 3, column 'b'"):
        read_table(path)


def test_a_non_finite_cell_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "r.csv", "a,b\n0.01,0.02\n0.03,nan\n")
    with pytest.raises(InputError, match="not finite"):
        read_table(path)


def test_duplicate_column_names_are_refused(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="duplicate column names"):
        read_table(write(tmp_path / "r.csv", "a,a\n0.01,0.02\n"))


def test_a_file_of_only_dates_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="no return columns"):
        read_table(write(tmp_path / "r.csv", "date\n2024-01-01\n2024-01-02\n"))


def test_a_return_at_or_below_minus_one_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "r.csv", "a\n0.01\n-1.0\n0.02\n")
    with pytest.raises(InputError, match="at or below -100%"):
        read_table(path)


def test_a_price_series_is_refused_rather_than_silently_misread(
    tmp_path: Path,
) -> None:
    """The case that produced a one-day value at risk of 164% before the guard.

    Prices parse as numbers and the estimators have no way to know they are not
    returns, so the answer comes out confident and absurd — and an absurd number
    is only caught because it is absurd, which is not a guarantee.
    """
    path = write(
        tmp_path / "p.csv",
        "date,a\n2024-01-01,100.0\n2024-01-02,101.5\n2024-01-03,99.8\n",
    )
    with pytest.raises(InputError, match="median absolute value"):
        read_table(path)


def test_returns_written_as_percentages_are_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "p.csv", "a\n1.5\n2.2\n1.8\n3.1\n")
    with pytest.raises(InputError, match="divide by 100"):
        read_table(path)


def test_a_single_extreme_return_does_not_trip_the_guard(tmp_path: Path) -> None:
    """The reason the test is on the median rather than the maximum.

    A return above 100% in one period is rare and real. Refusing a whole file
    for one of them would be wrong, and a maximum-based check would.
    """
    rows = ["a", "3.5"] + [f"{0.001 * k:.4f}" for k in range(1, 40)]
    table = read_table(write(tmp_path / "r.csv", "\n".join(rows) + "\n"))
    assert table.panel.observations == 40


# -- weights -----------------------------------------------------------------


def test_weights_default_to_equal(tmp_path: Path) -> None:
    panel = read_table(sample_file(tmp_path / "r.csv")).panel
    assert parse_weights(None, panel) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_weights_are_not_normalised() -> None:
    """Weights summing to 0.8 mean a fifth in cash, and are taken at face value."""
    panel = Panel.from_columns({"a": [0.01, -0.01], "b": [0.02, 0.0]})
    assert parse_weights("0.5,0.3", panel) == [0.5, 0.3]


def test_the_wrong_number_of_weights_lists_the_columns(tmp_path: Path) -> None:
    panel = read_table(sample_file(tmp_path / "r.csv")).panel
    with pytest.raises(InputError, match="alpha, beta, gamma"):
        parse_weights("0.5,0.5", panel)


def test_a_non_numeric_weight_is_refused(tmp_path: Path) -> None:
    panel = read_table(sample_file(tmp_path / "r.csv")).panel
    with pytest.raises(InputError, match="not a number"):
        parse_weights("0.5,0.3,half", panel)


# -- the commands ------------------------------------------------------------


def test_risk_reports_both_the_assumed_and_the_empirical_figure(
    tmp_path: Path,
) -> None:
    code, output = run("risk", str(sample_file(tmp_path / "r.csv")), "--periods", "252")
    assert code == 0
    assert "value at risk" in output
    assert "expected shortfall" in output
    assert "historical" in output
    assert "Annualised volatility" in output


def test_risk_states_how_thin_the_tail_sample_is(tmp_path: Path) -> None:
    """The number a reader most needs and is least often given."""
    code, output = run("risk", str(sample_file(tmp_path / "r.csv", periods=300)))
    assert code == 0
    assert "3.0 observations of 300" in output


@pytest.mark.parametrize("distribution", ["normal", "student-t", "cornish-fisher"])
def test_every_distribution_runs(tmp_path: Path, distribution: str) -> None:
    code, output = run(
        "risk",
        str(sample_file(tmp_path / "r.csv")),
        "--distribution",
        distribution,
    )
    assert code == 0
    assert distribution in output


def test_contributions_reconcile_and_say_so(tmp_path: Path) -> None:
    code, output = run(
        "contributions", str(sample_file(tmp_path / "r.csv")), "--weights", "0.5,0.3,0.2"
    )
    assert code == 0
    assert "Euler allocation of volatility" in output
    assert "reconcile to the total" in output
    assert "total" in output


@pytest.mark.parametrize("measure", ["volatility", "var", "shortfall"])
def test_every_contribution_measure_runs(tmp_path: Path, measure: str) -> None:
    code, _ = run(
        "contributions",
        str(sample_file(tmp_path / "r.csv")),
        "--measure",
        measure,
    )
    assert code == 0


def test_contributions_explain_a_hedging_position(tmp_path: Path) -> None:
    """``gamma`` is built with a negative loading, so it removes risk.

    The bet count over positions is then undefined, and the output has to say
    that rather than print a number computed from negative shares.
    """
    code, output = run(
        "contributions",
        str(sample_file(tmp_path / "r.csv")),
        "--weights",
        "0.6,0.3,0.4",
    )
    assert code == 0
    assert "hedges the rest" in output
    assert "principal components" in output


def test_parity_reports_its_convergence(tmp_path: Path) -> None:
    code, output = run("parity", str(sample_file(tmp_path / "r.csv")), "--shrink")
    assert code == 0
    assert "Converged in" in output
    assert "25.00%" in output or "33.33%" in output


def test_drawdown_labels_the_dates_from_the_file(tmp_path: Path) -> None:
    code, output = run(
        "drawdown", str(sample_file(tmp_path / "r.csv")), "--periods", "252"
    )
    assert code == 0
    assert "maximum drawdown" in output
    assert "2024-" in output
    assert "gain needed to recover" in output


def test_drawdown_says_when_the_worst_fall_never_recovered(tmp_path: Path) -> None:
    path = write(tmp_path / "r.csv", "date,a\n" + "".join(
        f"2024-01-{day:02d},-0.02\n" for day in range(1, 20)
    ))
    code, output = run("drawdown", str(path))
    assert code == 0
    assert "not within the sample" in output
    assert "unknown rather than zero" in output


def test_factors_attribute_and_report_the_residual_diagnostic(
    tmp_path: Path,
) -> None:
    returns = sample_file(tmp_path / "r.csv", periods=300)
    rng = random.Random(9)
    lines = ["date,market"]
    for _ in range(300):
        lines.append(f"2024-01-01,{rng.gauss(0.0, 0.011):.8f}")
    factors = write(tmp_path / "f.csv", "\n".join(lines) + "\n")
    code, output = run("factors", str(returns), "--factors", str(factors))
    assert code == 0
    assert "Exposures" in output
    assert "Risk attribution" in output
    assert "residual correlation" in output
    assert "reconcile within" in output


def test_factors_refuse_a_misaligned_factor_file(tmp_path: Path) -> None:
    returns = sample_file(tmp_path / "r.csv", periods=300)
    factors = write(tmp_path / "f.csv", "date,market\n2024-01-01,0.01\n")
    code, _ = run("factors", str(returns), "--factors", str(factors))
    assert code == 2


# -- machine-readable output -------------------------------------------------


@pytest.mark.parametrize(
    "command", [["risk"], ["contributions"], ["parity"], ["drawdown"]]
)
def test_json_output_parses(tmp_path: Path, command: list[str]) -> None:
    code, output = run("--json", *command, str(sample_file(tmp_path / "r.csv")))
    assert code == 0
    payload = json.loads(output)
    assert isinstance(payload, dict)
    assert payload


def test_json_risk_carries_the_numbers_not_the_formatting(tmp_path: Path) -> None:
    _, output = run("--json", "risk", str(sample_file(tmp_path / "r.csv")))
    payload = json.loads(output)
    assert payload["observations"] == 300
    assert payload["value_at_risk"] > 0.0
    assert payload["expected_shortfall"] > payload["value_at_risk"]


def test_json_contributions_carry_the_identity_error(tmp_path: Path) -> None:
    _, output = run(
        "--json",
        "contributions",
        str(sample_file(tmp_path / "r.csv")),
        "--weights",
        "0.5,0.3,0.2",
    )
    payload = json.loads(output)
    assert payload["identity_error"] < 1e-15
    assert len(payload["positions"]) == 3
    assert sum(entry["component"] for entry in payload["positions"]) == pytest.approx(
        payload["total"], rel=1e-12
    )


# -- exit codes --------------------------------------------------------------


def test_a_bad_file_exits_two_and_writes_nothing_to_the_output(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    code = main(["risk", str(tmp_path / "absent.csv")], stream=stream)
    assert code == 2
    assert stream.getvalue() == ""


def test_a_good_run_exits_zero(tmp_path: Path) -> None:
    code, _ = run("risk", str(sample_file(tmp_path / "r.csv")))
    assert code == 0


def test_no_command_is_an_argument_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([])


# --------------------------------------------------------------------------
# validate: scoring a forecast series against what happened.
# --------------------------------------------------------------------------


def forecast_file(
    path: Path,
    *,
    periods: int = 250,
    true_volatility: float = 0.012,
    forecast_volatility: float = 0.012,
    seed: int = 5,
) -> Path:
    """Returns beside the forecasts that were made for them.

    Separating the true and forecast volatilities is what lets a test point
    the command at a model that is wrong on purpose.
    """
    from shortfall.distributions import normal_pdf, normal_ppf

    rng = random.Random(seed)
    quantile = -normal_ppf(0.01)
    value_at_risk = forecast_volatility * quantile
    shortfall = forecast_volatility * normal_pdf(quantile) / 0.01
    lines = ["date,return,var,es"]
    for index in range(periods):
        lines.append(
            f"2024-{(index % 12) + 1:02d}-{(index % 28) + 1:02d},"
            f"{rng.gauss(0.0, true_volatility):.8f},{value_at_risk:.8f},{shortfall:.8f}"
        )
    return write(path, "\n".join(lines) + "\n")


def test_validate_reports_a_healthy_model(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(["validate", str(forecast_file(tmp_path / "f.csv"))], stream=stream)
    assert code == 0
    output = stream.getvalue()
    assert "Risk model validation" in output
    assert "green" in output
    assert "not rejected" in output


def test_validate_emits_json_a_caller_can_branch_on(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        ["--json", "validate", str(forecast_file(tmp_path / "f.csv")), "--es-column", "es"],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue().split("\n\n")[-1])
    assert payload["observations"] == 250
    assert payload["tests"]["unconditional coverage"]["degrees_of_freedom"] == 1
    assert payload["tests"]["conditional coverage"]["degrees_of_freedom"] == 2
    assert payload["traffic_light"]["zone"] == "green"
    assert payload["traffic_light"]["plus_factor"] == 0.0
    assert payload["rejected_at_5_percent"] == []
    assert "expected_shortfall" in payload


def test_validate_condemns_a_model_whose_volatility_is_far_too_low(tmp_path: Path) -> None:
    stream = io.StringIO()
    path = forecast_file(tmp_path / "f.csv", true_volatility=0.03, forecast_volatility=0.01)
    code = main(["--json", "validate", str(path)], stream=stream)
    assert code == 0
    payload = json.loads(stream.getvalue())
    assert payload["breaches"] > 20
    assert payload["traffic_light"]["zone"] == "red"
    assert payload["traffic_light"]["plus_factor"] == 1.0
    assert "unconditional coverage" in payload["rejected_at_5_percent"]


def test_validate_attaches_a_p_value_when_asked_to_simulate(tmp_path: Path) -> None:
    stream = io.StringIO()
    path = forecast_file(tmp_path / "f.csv", true_volatility=0.02, forecast_volatility=0.01)
    code = main(
        [
            "--json",
            "validate",
            str(path),
            "--es-column",
            "es",
            "--replications",
            "200",
            "--seed",
            "3",
        ],
        stream=stream,
    )
    assert code == 0
    found = json.loads(stream.getvalue())["expected_shortfall"]
    assert found["replications"] == 200
    assert found["unconditional_p_value"] is not None
    assert found["unconditional_p_value"] < 0.05
    assert "understated" in found["direction"]


def test_validate_says_which_column_it_could_not_find(tmp_path: Path) -> None:
    path = write(tmp_path / "f.csv", "date,ret,forecast\n2024-01-01,-0.01,0.02\n")
    stream = io.StringIO()
    code = main(["validate", str(path)], stream=stream)
    assert code == 2


def test_validate_names_the_columns_the_file_actually_has(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path / "f.csv", "date,ret,forecast\n2024-01-01,-0.01,0.02\n")
    main(["validate", str(path)], stream=io.StringIO())
    message = capsys.readouterr().err
    assert "'return'" in message
    assert "'ret'" in message and "'forecast'" in message


def test_validate_refuses_a_forecast_column_of_negative_losses(tmp_path: Path) -> None:
    """The commonest way to hold this wrong, and it must not score quietly."""
    path = write(
        tmp_path / "f.csv",
        "date,return,var\n2024-01-01,-0.01,-0.02\n2024-01-02,0.01,-0.02\n",
    )
    stream = io.StringIO()
    assert main(["validate", str(path)], stream=stream) == 2


def test_validate_accepts_forecasts_that_change_every_day(tmp_path: Path) -> None:
    rng = random.Random(9)
    lines = ["date,return,var,es"]
    for index in range(300):
        volatility = 0.005 + 0.02 * rng.random()
        lines.append(
            f"2024-{(index % 12) + 1:02d}-{(index % 28) + 1:02d},"
            f"{rng.gauss(0.0, volatility):.8f},{2.3263 * volatility:.8f},"
            f"{2.6652 * volatility:.8f}"
        )
    path = write(tmp_path / "f.csv", "\n".join(lines) + "\n")
    stream = io.StringIO()
    assert main(["--json", "validate", str(path), "--es-column", "es"], stream=stream) == 0
    payload = json.loads(stream.getvalue())
    assert payload["rejected_at_5_percent"] == []


# --------------------------------------------------------------------------
# volatility: fitting the conditional model.
# --------------------------------------------------------------------------


def garch_file(path: Path, *, periods: int = 900, seed: int = 4) -> Path:
    """A simulated GARCH path, so the fit has something real to recover."""
    rng = random.Random(seed)
    variance = 2e-6
    lines = ["date,fund"]
    for index in range(periods):
        value = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        variance = 2e-6 + 0.09 * value * value + 0.89 * variance
        lines.append(f"d{index},{value:.8f}")
    return write(path, "\n".join(lines) + "\n")


def test_volatility_recovers_the_simulated_parameters(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(["--json", "volatility", str(garch_file(tmp_path / "v.csv"))], stream=stream) == 0
    payload = json.loads(stream.getvalue())
    assert payload["converged"] is True
    assert payload["alpha"] == pytest.approx(0.09, abs=0.05)
    assert payload["beta"] == pytest.approx(0.89, abs=0.05)
    assert 0.9 < payload["persistence"] < 1.0
    assert payload["halfLife"] > 0.0


def test_volatility_prints_the_square_root_of_time_comparison(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(
        ["volatility", str(garch_file(tmp_path / "v.csv")), "--horizon", "250"],
        stream=stream,
    ) == 0
    output = stream.getvalue()
    assert "Conditional volatility" in output
    assert "against square-root-of-time" in output
    assert "square root of 250" in output
    assert "overstates" in output or "understates" in output


def test_the_horizon_figure_differs_from_square_root_of_time(tmp_path: Path) -> None:
    """And by more at a longer horizon, because there is more time to revert."""
    path = garch_file(tmp_path / "v.csv")
    ratios = []
    for horizon in ("10", "250"):
        stream = io.StringIO()
        assert main(["--json", "volatility", str(path), "--horizon", horizon], stream=stream) == 0
        ratios.append(json.loads(stream.getvalue())["squareRootOfTimeRatio"])
    near, far = ratios
    assert abs(far - 1.0) > abs(near - 1.0)


def test_variance_targeting_is_available_from_the_command_line(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(
        ["--json", "volatility", str(garch_file(tmp_path / "v.csv")), "--variance-targeting"],
        stream=stream,
    ) == 0
    payload = json.loads(stream.getvalue())
    assert payload["varianceTargeted"] is True
    assert payload["converged"] is True


def test_volatility_fits_a_named_column_of_a_multi_asset_file(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(
        ["--json", "volatility", str(sample_file(tmp_path / "r.csv")), "--column", "beta"],
        stream=stream,
    ) == 0
    assert json.loads(stream.getvalue())["series"] == "beta"


def test_volatility_falls_back_to_the_portfolio_on_a_multi_asset_file(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    assert main(
        ["--json", "volatility", str(sample_file(tmp_path / "r.csv"))], stream=stream
    ) == 0
    assert json.loads(stream.getvalue())["series"] == "portfolio"


def test_volatility_names_the_columns_when_asked_for_one_that_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(
        ["volatility", str(sample_file(tmp_path / "r.csv")), "--column", "delta"],
        stream=io.StringIO(),
    ) == 2
    message = capsys.readouterr().err
    assert "'delta'" in message
    assert "'alpha'" in message


def test_volatility_refuses_a_sample_too_short_to_fit(tmp_path: Path) -> None:
    path = garch_file(tmp_path / "v.csv", periods=40)
    assert main(["volatility", str(path)], stream=io.StringIO()) == 2


def fat_garch_file(
    path: Path, *, periods: int = 1200, seed: int = 8, degrees: float = 4.5
) -> Path:
    """A GARCH path whose innovations are a standardised Student-t.

    The same variance parameters as :func:`garch_file`, so the two files differ
    only in the shape of the innovations. That is what makes the pair of tests
    below a comparison rather than two anecdotes.
    """
    rng = random.Random(seed)
    scale = math.sqrt(degrees / (degrees - 2.0))
    variance = 2e-6
    lines = ["date,fund"]
    for index in range(periods):
        chi_square = 2.0 * rng.gammavariate(degrees / 2.0, 1.0)
        innovation = rng.gauss(0.0, 1.0) / math.sqrt(chi_square / degrees) / scale
        value = math.sqrt(variance) * innovation
        variance = 2e-6 + 0.09 * value * value + 0.89 * variance
        lines.append(f"d{index},{value:.8f}")
    return write(path, "\n".join(lines) + "\n")


def test_volatility_finds_the_fat_tail_and_widens_the_forecast(tmp_path: Path) -> None:
    """The default path: test for a fat tail, and use one when it is there."""
    stream = io.StringIO()
    assert main(
        ["--json", "volatility", str(fat_garch_file(tmp_path / "f.csv"))], stream=stream
    ) == 0
    payload = json.loads(stream.getvalue())
    assert payload["innovation"] == "student-t"
    assert payload["fatTail"]["fat"] is True
    assert payload["fatTail"]["pValue"] < 0.01
    assert 2.0 < payload["degreesOfFreedom"] < 10.0
    assert payload["expectedShortfall"] > payload["valueAtRisk"] > 0.0


def test_volatility_leaves_a_thin_tailed_series_alone(tmp_path: Path) -> None:
    """Gaussian innovations: reported as not identified rather than as a number."""
    stream = io.StringIO()
    assert main(
        ["--json", "volatility", str(garch_file(tmp_path / "v.csv"))], stream=stream
    ) == 0
    payload = json.loads(stream.getvalue())
    assert payload["innovation"] == "normal"
    assert payload["fatTail"]["fat"] is False
    assert payload["degreesOfFreedom"] is None
    assert payload["impliedExcessKurtosis"] is None


def test_the_chosen_innovation_changes_the_reported_risk(tmp_path: Path) -> None:
    """Same data, same volatility, two quantiles — and the gap is the point."""
    path = fat_garch_file(tmp_path / "f.csv")
    figures = {}
    for innovation in ("normal", "student-t"):
        stream = io.StringIO()
        assert main(
            ["--json", "volatility", str(path), "--innovation", innovation], stream=stream
        ) == 0
        payload = json.loads(stream.getvalue())
        assert payload["innovation"] == innovation
        assert "fatTail" not in payload, "the test is skipped when the shape is given"
        figures[innovation] = payload
    assert figures["student-t"]["valueAtRisk"] > figures["normal"]["valueAtRisk"]
    assert (
        figures["student-t"]["expectedShortfall"] > figures["normal"]["expectedShortfall"]
    )
    # The expected shortfall gap is the larger one, at the same fitted level.
    var_ratio = figures["student-t"]["valueAtRisk"] / figures["normal"]["valueAtRisk"]
    es_ratio = (
        figures["student-t"]["expectedShortfall"] / figures["normal"]["expectedShortfall"]
    )
    assert es_ratio > var_ratio > 1.0


def test_the_volatility_payload_is_strict_json_on_a_very_fat_tail(tmp_path: Path) -> None:
    """The implied excess kurtosis is infinite below four degrees of freedom.

    ``json.dumps`` writes bare ``Infinity`` for it, which is not JSON and which a
    strict parser rejects for the whole document rather than for the one field.
    The round trip through ``json.loads`` would not catch it, since that accepts
    the token — so the assertion has to be on dumping with ``allow_nan`` off.
    """
    stream = io.StringIO()
    assert main(
        [
            "--json",
            "volatility",
            str(fat_garch_file(tmp_path / "f.csv", degrees=3.0, seed=13)),
        ],
        stream=stream,
    ) == 0
    payload = json.loads(stream.getvalue())
    assert json.dumps(payload, allow_nan=False)
    if payload["degreesOfFreedom"] is not None and payload["degreesOfFreedom"] <= 4.0:
        assert payload["impliedExcessKurtosis"] is None


def test_volatility_explains_a_fat_tail_in_words(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(["volatility", str(fat_garch_file(tmp_path / "f.csv"))], stream=stream) == 0
    output = stream.getvalue()
    assert "innovations are fat-tailed" in output
    assert "degrees of freedom" in output
    assert "value at risk" in output


def test_volatility_says_so_when_it_finds_no_fat_tail(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(["volatility", str(garch_file(tmp_path / "v.csv"))], stream=stream) == 0
    output = stream.getvalue()
    assert "No fat tail was found" in output
    assert "conservative" in output
    assert "not identified" in output


def test_volatility_simulates_a_horizon_when_asked(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(
        [
            "--json",
            "volatility",
            str(garch_file(tmp_path / "v.csv")),
            "--paths",
            "4000",
            "--horizon",
            "10",
        ],
        stream=stream,
    ) == 0
    payload = json.loads(stream.getvalue())
    simulated = payload["horizonRisk"]
    assert simulated["paths"] == 4000
    assert simulated["draw"] == "bootstrap"
    assert simulated["expectedShortfall"] > simulated["valueAtRisk"] > 0.0
    assert 0.0 < simulated["standardError"] < simulated["valueAtRisk"]
    # The simulated and analytic horizon volatilities are the same quantity
    # measured two ways, so they have to agree.
    assert simulated["simulatedVolatility"] == pytest.approx(
        simulated["analyticVolatility"], rel=0.05
    )
    json.dumps(payload, allow_nan=False)


def test_the_horizon_simulation_is_off_unless_asked_for(tmp_path: Path) -> None:
    """It costs paths times steps of work, so it is not on the default path."""
    stream = io.StringIO()
    assert main(["--json", "volatility", str(garch_file(tmp_path / "v.csv"))], stream=stream) == 0
    assert "horizonRisk" not in json.loads(stream.getvalue())


def test_the_horizon_simulation_explains_why_it_is_not_a_scaling(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert main(
        ["volatility", str(garch_file(tmp_path / "v.csv")), "--paths", "4000"], stream=stream
    ) == 0
    output = stream.getvalue()
    assert "Horizon risk over" in output
    assert "quantile vs square-root-of-time" in output
    assert "consistently wrong" in output


def test_the_two_draws_give_different_horizon_figures(tmp_path: Path) -> None:
    path = fat_garch_file(tmp_path / "f.csv")
    figures = {}
    for draw in ("bootstrap", "parametric"):
        stream = io.StringIO()
        assert main(
            ["--json", "volatility", str(path), "--paths", "4000", "--draw", draw],
            stream=stream,
        ) == 0
        figures[draw] = json.loads(stream.getvalue())["horizonRisk"]
        assert figures[draw]["draw"] == draw
    assert figures["bootstrap"]["valueAtRisk"] != figures["parametric"]["valueAtRisk"]


def test_a_path_count_the_simulation_refuses_is_reported_not_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Below the floor the tail of the simulation is a handful of paths.

    Reported as a sentence naming the problem rather than as a traceback, which is
    what the rest of this interface does with every other refusal.
    """
    assert (
        main(
            ["volatility", str(garch_file(tmp_path / "v.csv")), "--paths", "50"],
            stream=io.StringIO(),
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "paths must be between" in error
    assert "quantile of anything" in error

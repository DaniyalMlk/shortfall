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
from statistics import NormalDist

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
    code, output = run("drawdown", str(sample_file(tmp_path / "r.csv")), "--periods", "252")
    assert code == 0
    assert "maximum drawdown" in output
    assert "2024-" in output
    assert "gain needed to recover" in output


def test_drawdown_says_when_the_worst_fall_never_recovered(tmp_path: Path) -> None:
    path = write(
        tmp_path / "r.csv",
        "date,a\n" + "".join(f"2024-01-{day:02d},-0.02\n" for day in range(1, 20)),
    )
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


@pytest.mark.parametrize("command", [["risk"], ["contributions"], ["parity"], ["drawdown"]])
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
    assert (
        main(
            ["volatility", str(garch_file(tmp_path / "v.csv")), "--horizon", "250"],
            stream=stream,
        )
        == 0
    )
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
    assert (
        main(
            ["--json", "volatility", str(garch_file(tmp_path / "v.csv")), "--variance-targeting"],
            stream=stream,
        )
        == 0
    )
    payload = json.loads(stream.getvalue())
    assert payload["varianceTargeted"] is True
    assert payload["converged"] is True


def test_volatility_fits_a_named_column_of_a_multi_asset_file(tmp_path: Path) -> None:
    stream = io.StringIO()
    assert (
        main(
            ["--json", "volatility", str(sample_file(tmp_path / "r.csv")), "--column", "beta"],
            stream=stream,
        )
        == 0
    )
    assert json.loads(stream.getvalue())["series"] == "beta"


def test_volatility_falls_back_to_the_portfolio_on_a_multi_asset_file(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    assert main(["--json", "volatility", str(sample_file(tmp_path / "r.csv"))], stream=stream) == 0
    assert json.loads(stream.getvalue())["series"] == "portfolio"


def test_volatility_names_the_columns_when_asked_for_one_that_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            ["volatility", str(sample_file(tmp_path / "r.csv")), "--column", "delta"],
            stream=io.StringIO(),
        )
        == 2
    )
    message = capsys.readouterr().err
    assert "'delta'" in message
    assert "'alpha'" in message


def test_volatility_refuses_a_sample_too_short_to_fit(tmp_path: Path) -> None:
    path = garch_file(tmp_path / "v.csv", periods=40)
    assert main(["volatility", str(path)], stream=io.StringIO()) == 2


def fat_garch_file(path: Path, *, periods: int = 1200, seed: int = 8, degrees: float = 4.5) -> Path:
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
    assert (
        main(["--json", "volatility", str(fat_garch_file(tmp_path / "f.csv"))], stream=stream) == 0
    )
    payload = json.loads(stream.getvalue())
    assert payload["innovation"] == "student-t"
    assert payload["fatTail"]["fat"] is True
    assert payload["fatTail"]["pValue"] < 0.01
    assert 2.0 < payload["degreesOfFreedom"] < 10.0
    assert payload["expectedShortfall"] > payload["valueAtRisk"] > 0.0


def test_volatility_leaves_a_thin_tailed_series_alone(tmp_path: Path) -> None:
    """Gaussian innovations: reported as not identified rather than as a number."""
    stream = io.StringIO()
    assert main(["--json", "volatility", str(garch_file(tmp_path / "v.csv"))], stream=stream) == 0
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
        assert (
            main(["--json", "volatility", str(path), "--innovation", innovation], stream=stream)
            == 0
        )
        payload = json.loads(stream.getvalue())
        assert payload["innovation"] == innovation
        assert "fatTail" not in payload, "the test is skipped when the shape is given"
        figures[innovation] = payload
    assert figures["student-t"]["valueAtRisk"] > figures["normal"]["valueAtRisk"]
    assert figures["student-t"]["expectedShortfall"] > figures["normal"]["expectedShortfall"]
    # The expected shortfall gap is the larger one, at the same fitted level.
    var_ratio = figures["student-t"]["valueAtRisk"] / figures["normal"]["valueAtRisk"]
    es_ratio = figures["student-t"]["expectedShortfall"] / figures["normal"]["expectedShortfall"]
    assert es_ratio > var_ratio > 1.0


def test_the_volatility_payload_is_strict_json_on_a_very_fat_tail(tmp_path: Path) -> None:
    """The implied excess kurtosis is infinite below four degrees of freedom.

    ``json.dumps`` writes bare ``Infinity`` for it, which is not JSON and which a
    strict parser rejects for the whole document rather than for the one field.
    The round trip through ``json.loads`` would not catch it, since that accepts
    the token — so the assertion has to be on dumping with ``allow_nan`` off.
    """
    stream = io.StringIO()
    assert (
        main(
            [
                "--json",
                "volatility",
                str(fat_garch_file(tmp_path / "f.csv", degrees=3.0, seed=13)),
            ],
            stream=stream,
        )
        == 0
    )
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
    assert (
        main(
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
        )
        == 0
    )
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
    assert (
        main(["volatility", str(garch_file(tmp_path / "v.csv")), "--paths", "4000"], stream=stream)
        == 0
    )
    output = stream.getvalue()
    assert "Horizon risk over" in output
    assert "quantile vs square-root-of-time" in output
    assert "consistently wrong" in output


def test_the_two_draws_give_different_horizon_figures(tmp_path: Path) -> None:
    path = fat_garch_file(tmp_path / "f.csv")
    figures = {}
    for draw in ("bootstrap", "parametric"):
        stream = io.StringIO()
        assert (
            main(
                ["--json", "volatility", str(path), "--paths", "4000", "--draw", draw],
                stream=stream,
            )
            == 0
        )
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


# -- the fitted tail ----------------------------------------------------------


def test_tail_prints_the_fit_the_extrapolation_and_the_comparison(
    tmp_path: Path,
) -> None:
    """Three things a fitted far-tail figure is useless without.

    The parameters, so the shape can be sanity-checked against the standard error
    beside it; how many observations lie beyond the answer, which is the measure
    of how much of it is extrapolation; and the historical figure, so the reader
    can see the number this estimator exists to improve on.
    """
    code, output = run(
        "tail", str(sample_file(tmp_path / "r.csv", periods=1200)), "--confidence", "0.999"
    )
    assert code == 0
    assert "Fitted tail" in output
    assert "maximum likelihood" in output
    assert "shape" in output
    assert "exceedances" in output
    assert "historical value at risk" in output
    assert "observations beyond the estimate" in output


def test_tail_says_when_nothing_in_the_sample_is_that_bad(tmp_path: Path) -> None:
    code, output = run(
        "tail",
        str(sample_file(tmp_path / "r.csv", periods=1200)),
        "--confidence",
        "0.99999",
    )
    assert code == 0
    assert "Nothing in the sample is as bad as this figure" in output


def test_tail_refuses_a_confidence_the_fit_does_not_cover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A confidence inside the body, which the fit deliberately knows nothing about.

    The message has to name the lowest confidence that is legal, because that is
    the number the caller needs in order to fix the call.
    """
    stream = io.StringIO()
    code = main(
        ["tail", str(sample_file(tmp_path / "r.csv", periods=600)), "--confidence", "0.5"],
        stream=stream,
    )
    assert code == 2
    message = capsys.readouterr().err
    assert "says nothing below 0.95" in message
    assert "inside the body" in message


def test_tail_reports_the_curve_the_threshold_should_come_from(tmp_path: Path) -> None:
    code, output = run("tail", str(sample_file(tmp_path / "r.csv", periods=1200)), "--curve", "5")
    assert code == 0
    assert "Mean excess against threshold" in output
    lines = [line for line in output.splitlines() if line.startswith(("0.", "1.", "2."))]
    assert len(lines) >= 4


def test_tail_takes_a_threshold_and_a_method(tmp_path: Path) -> None:
    path = sample_file(tmp_path / "r.csv", periods=1200)
    stream = io.StringIO()
    code = main(
        [
            "--json",
            "tail",
            str(path),
            "--threshold",
            "0.008",
            "--method",
            "probability_weighted_moments",
        ],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue())
    assert payload["threshold"] == pytest.approx(0.008)
    assert payload["method"] == "probability_weighted_moments"
    # The moment estimator has no asymptotic standard error here, and the field is
    # null rather than absent so a consumer does not have to tell the two apart.
    assert payload["shapeStandardError"] is None
    assert payload["shape"] < 1.0


def test_the_tail_payload_is_json_a_strict_parser_accepts(tmp_path: Path) -> None:
    """Every numeric field finite, including the ones that can be infinite.

    ``json.dumps`` writes bare ``Infinity`` for a non-finite float, which is not
    JSON at all — and ``json.loads`` accepts it, so a round trip does not catch
    it. The upper endpoint is the field at risk: it does not exist for a
    non-negative shape and is null rather than infinity.
    """
    stream = io.StringIO()
    code = main(
        ["--json", "tail", str(sample_file(tmp_path / "r.csv", periods=1200))],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue())
    assert json.dumps(payload, allow_nan=False)
    assert payload["upperEndpoint"] is None or payload["upperEndpoint"] > 0.0
    assert payload["observations"] == 1200
    assert 0.0 < payload["lowestConfidence"] < 1.0


def test_tail_fits_one_named_column(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stream = io.StringIO()
    code = main(
        ["--json", "tail", str(sample_file(tmp_path / "r.csv", periods=900)), "--column", "beta"],
        stream=stream,
    )
    assert code == 0
    assert json.loads(stream.getvalue())["series"] == "beta"
    assert (
        main(
            ["tail", str(sample_file(tmp_path / "r.csv")), "--column", "absent"],
            stream=io.StringIO(),
        )
        == 2
    )
    assert "no column named 'absent'" in capsys.readouterr().err


# -- copula ------------------------------------------------------------------


def copula_file(path: Path, *, periods: int = 400, degrees: float | None = 4.0) -> Path:
    """A four-asset file with a known copula, small enough to run in a test."""
    rng = random.Random(17)
    rho = 0.5
    lines = ["date,alpha,beta,gamma,delta"]
    for t in range(periods):
        common = rng.gauss(0.0, 1.0)
        mixing = (
            1.0 if degrees is None else math.sqrt(degrees / rng.gammavariate(degrees / 2.0, 2.0))
        )
        row = [f"2024-{(t % 12) + 1:02d}-{(t % 28) + 1:02d}"]
        for _ in range(4):
            z = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * rng.gauss(0.0, 1.0)
            row.append(f"{0.012 * z * mixing:.8f}")
        lines.append(",".join(row))
    return write(path, "\n".join(lines) + "\n")


def test_copula_reports_both_figures_and_the_pairs(tmp_path: Path) -> None:
    code, output = run(
        "copula", str(copula_file(tmp_path / "c.csv")), "--paths", "2000", "--seed", "1"
    )
    assert code == 0
    assert "Student-t copula at" in output
    assert "gaussian copula" in output
    assert "tail dependence" in output
    assert "alpha / beta" in output


def test_copula_json_carries_every_number(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        [
            "--json",
            "copula",
            str(copula_file(tmp_path / "c.csv")),
            "--paths",
            "2000",
            "--seed",
            "2",
        ],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue())
    assert payload["family"] == "student_t"
    assert payload["marginal"] == "empirical"
    assert payload["paths"] == 2000
    assert payload["degrees_of_freedom"] > 0.0
    assert payload["value_at_risk"] > 0.0
    assert payload["gaussian_value_at_risk"] > 0.0
    assert payload["standard_error"] > 0.0
    assert len(payload["tail_dependence"]) == 6
    assert payload["tail_dependence"][0]["coefficient"] > 0.0
    # The whole payload has to survive a strict encoder: a non-finite number is
    # not JSON and a strict reader rejects the document, not the field.
    json.dumps(payload, allow_nan=False)


def test_copula_gaussian_family_omits_the_premium(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        [
            "--json",
            "copula",
            str(copula_file(tmp_path / "c.csv")),
            "--family",
            "gaussian",
            "--paths",
            "2000",
            "--seed",
            "3",
        ],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue())
    assert payload["degrees_of_freedom"] is None
    assert "tail_dependence_premium" not in payload
    assert all(pair["coefficient"] == 0.0 for pair in payload["tail_dependence"])


def test_copula_extreme_value_marginals_are_selectable(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        [
            "--json",
            "copula",
            str(copula_file(tmp_path / "c.csv")),
            "--marginal",
            "extreme_value",
            "--paths",
            "2000",
            "--seed",
            "4",
        ],
        stream=stream,
    )
    assert code == 0
    assert json.loads(stream.getvalue())["marginal"] == "extreme_value"


def test_copula_fixed_degrees_are_reported_back(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        [
            "--json",
            "copula",
            str(copula_file(tmp_path / "c.csv")),
            "--degrees",
            "3",
            "--paths",
            "2000",
            "--seed",
            "5",
        ],
        stream=stream,
    )
    assert code == 0
    assert json.loads(stream.getvalue())["degrees_of_freedom"] == 3.0


def test_copula_reports_a_bad_path_count_as_a_sentence(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        ["copula", str(copula_file(tmp_path / "c.csv")), "--paths", "10"],
        stream=stream,
    )
    assert code == 2
    assert stream.getvalue() == ""


def test_copula_refuses_a_single_column_file(tmp_path: Path) -> None:
    path = write(
        tmp_path / "one.csv",
        "alpha\n" + "\n".join(f"{0.001 * (i % 7 - 3):.6f}" for i in range(300)) + "\n",
    )
    stream = io.StringIO()
    code = main(["copula", str(path), "--paths", "2000"], stream=stream)
    assert code == 2


def ohlc_file(path: Path, *, drift: float = 0.0, header: bool = True, dated: bool = True) -> Path:
    """A synthetic OHLC file: 120 bars of 200 ticks at 1.1% a bar."""
    rng = random.Random(5)
    price = 100.0
    lines = ["date,open,high,low,close"] if header else []
    for day in range(120):
        opening = price
        high = price
        low = price
        for _ in range(200):
            price *= math.exp(drift / 200.0 + 0.011 * rng.gauss(0.0, 1.0) / math.sqrt(200.0))
            high = max(high, price)
            low = min(low, price)
        cells = [f"{opening:.4f}", f"{high:.4f}", f"{low:.4f}", f"{price:.4f}"]
        if dated:
            cells.insert(0, f"2026-{1 + day // 30:02d}-{1 + day % 30:02d}")
        lines.append(",".join(cells))
    return write(path, "\n".join(lines) + "\n")


def test_bars_reports_every_estimator(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(["bars", "--bars", str(ohlc_file(tmp_path / "b.csv"))], stream=stream)
    assert code == 0
    out = stream.getvalue()
    for name in (
        "close-to-close",
        "parkinson",
        "garman-klass",
        "rogers-satchell",
        "garman-klass-yang-zhang",
        "yang-zhang",
    ):
        assert name in out
    assert "vs close-to-close" in out


def test_bars_estimators_agree_on_data_that_meets_their_assumptions(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    code = main(["bars", "--bars", str(ohlc_file(tmp_path / "b.csv")), "--json"], stream=stream)
    assert code == 0
    values = json.loads(stream.getvalue())["annualised_volatility"]
    # No gap and a negligible drift, so the only systematic difference left is
    # the tick bias, which moves the range estimators a few per cent below the
    # close-to-close one and no further.
    baseline = values["close-to-close"]
    for name, value in values.items():
        if name != "close-to-close":
            assert 0.90 < value / baseline < 1.00


def test_bars_shows_a_trend_as_a_disagreement(tmp_path: Path) -> None:
    # Two per cent of drift a day against 1.1% of volatility. Parkinson reads
    # it as volatility, Rogers-Satchell does not, and the gap between them is
    # the diagnostic the command exists to show.
    stream = io.StringIO()
    code = main(
        ["bars", "--bars", str(ohlc_file(tmp_path / "t.csv", drift=0.02)), "--json"],
        stream=stream,
    )
    assert code == 0
    values = json.loads(stream.getvalue())["annualised_volatility"]
    assert values["parkinson"] > 1.25 * values["rogers-satchell"]
    assert values["yang-zhang"] < 1.05 * values["rogers-satchell"]


def test_bars_reports_the_tick_correction_when_asked(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        ["bars", "--bars", str(ohlc_file(tmp_path / "b.csv")), "--ticks", "200", "--json"],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue())
    assert payload["tick_bias_factor"] == pytest.approx(0.8994, abs=1e-4)
    raw = payload["annualised_volatility"]["parkinson"]
    corrected = payload["corrected"]["parkinson"]
    assert corrected == pytest.approx(raw / math.sqrt(0.8994), rel=1e-3)
    # Close-to-close is not corrected, because it is not biased.
    assert "close-to-close" not in payload["corrected"]


def test_bars_reads_a_file_with_no_header_and_no_dates(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        [
            "bars",
            "--bars",
            str(ohlc_file(tmp_path / "plain.csv", header=False, dated=False)),
            "--json",
        ],
        stream=stream,
    )
    assert code == 0
    assert json.loads(stream.getvalue())["bars"] == 120


def test_bars_names_the_line_of_an_impossible_bar(tmp_path: Path) -> None:
    path = write(
        tmp_path / "bad.csv",
        "open,high,low,close\n100,101,99,100\n100,99,98,100\n100,101,99,100\n",
    )
    stream = io.StringIO()
    assert main(["bars", "--bars", str(path)], stream=stream) == 2
    assert stream.getvalue() == ""


def test_bars_names_the_line_of_a_non_numeric_cell(tmp_path: Path) -> None:
    path = write(tmp_path / "bad.csv", "open,high,low,close\n100,101,99,100\n100,x,98,100\n")
    stream = io.StringIO()
    assert main(["bars", "--bars", str(path)], stream=stream) == 2


def test_bars_refuses_a_short_row(tmp_path: Path) -> None:
    path = write(tmp_path / "bad.csv", "open,high,low,close\n100,101,99\n")
    stream = io.StringIO()
    assert main(["bars", "--bars", str(path)], stream=stream) == 2


def test_bars_refuses_an_empty_file(tmp_path: Path) -> None:
    path = write(tmp_path / "empty.csv", "\n")
    stream = io.StringIO()
    assert main(["bars", "--bars", str(path)], stream=stream) == 2


def two_model_file(path: Path, *, periods: int = 1500) -> Path:
    """Returns from a persistent volatility process, with two models' forecasts.

    One model smooths at 0.94 and the other at 0.995, so they disagree about
    the state and their score differences are persistent — which is the case
    the robust variance exists for.
    """
    normal = NormalDist()
    confidence = 0.975
    alpha = 1.0 - confidence
    z = normal.inv_cdf(confidence)
    rng = random.Random(21)
    returns: list[float] = []
    volatilities: list[float] = []
    state = 0.0
    for _ in range(periods):
        state = 0.97 * state + math.sqrt(1.0 - 0.97**2) * rng.gauss(0.0, 0.5)
        sigma = 0.01 * math.exp(state)
        volatilities.append(sigma)
        returns.append(rng.gauss(0.0, sigma))

    def ewma(decay: float) -> list[float]:
        out: list[float] = []
        variance = volatilities[0] ** 2
        for value in returns:
            out.append(math.sqrt(variance))
            variance = decay * variance + (1.0 - decay) * value * value
        return out

    fast, slow = ewma(0.94), ewma(0.995)
    lines = ["return,var_a,es_a,var_b,es_b"]
    for value, a, b in zip(returns, fast, slow, strict=True):
        lines.append(
            f"{value:.10f},{a * z:.10f},{a * normal.pdf(z) / alpha:.10f},"
            f"{b * z:.10f},{b * normal.pdf(z) / alpha:.10f}"
        )
    return write(path, "\n".join(lines) + "\n")


def test_score_ranks_two_models_jointly(tmp_path: Path) -> None:
    code, text = run(
        "score",
        str(two_model_file(tmp_path / "two.csv")),
        "--es-columns",
        "es_a",
        "es_b",
    )
    assert code == 0
    assert "fissler-ziegel" in text
    assert "Lower score, so better:" in text
    # Both standard errors are printed, because the gap between them is the
    # whole reason the bandwidth is there.
    assert "times the naive" in text


def test_score_in_json_carries_both_standard_errors(tmp_path: Path) -> None:
    code, text = run(
        "--json",
        "score",
        str(two_model_file(tmp_path / "two.csv")),
        "--es-columns",
        "es_a",
        "es_b",
    )
    assert code == 0
    payload = json.loads(text)
    assert payload["score"] == "fissler-ziegel"
    assert payload["observations"] == 1500
    assert payload["better"] in {"var_a", "var_b"}
    assert payload["robust_over_naive"] > 0.9
    assert payload["standard_error"] > 0.0
    assert payload["naive_standard_error"] > 0.0
    assert len(payload["models"]) == 2
    # The winner is the one with the lower mean score, by construction.
    scores = {model["name"]: model["mean_score"] for model in payload["models"]}
    assert payload["better"] == min(scores, key=lambda name: scores[name])


def test_score_falls_back_to_the_pinball_loss(tmp_path: Path) -> None:
    code, text = run("--json", "score", str(two_model_file(tmp_path / "two.csv")))
    assert code == 0
    payload = json.loads(text)
    assert payload["score"] == "pinball"
    # Both scores are positive here, where the joint one is negative: the two
    # are different numbers about different things and the module says so.
    assert all(model["mean_score"] > 0.0 for model in payload["models"])


def test_score_takes_an_explicit_bandwidth(tmp_path: Path) -> None:
    path = two_model_file(tmp_path / "two.csv")
    code, text = run("--json", "score", str(path), "--lags", "0")
    assert code == 0
    payload = json.loads(text)
    assert payload["lags"] == 0
    # No lags means the robust estimate is the naive one.
    assert payload["standard_error"] == pytest.approx(payload["naive_standard_error"], rel=1e-12)


def test_score_names_a_missing_column_and_lists_what_there_is(tmp_path: Path) -> None:
    path = two_model_file(tmp_path / "two.csv")
    stream = io.StringIO()
    code = main(["score", str(path), "--var-columns", "var_a", "nope"], stream=stream)
    assert code == 2


def test_score_refuses_an_impossible_confidence(tmp_path: Path) -> None:
    path = two_model_file(tmp_path / "two.csv")
    assert main(["score", str(path), "--confidence", "1.0"], stream=io.StringIO()) == 2


def test_expectile_reports_the_three_estimates_and_the_translation(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        ["expectile", str(sample_file(tmp_path / "r.csv")), "--level", "0.99"],
        stream=stream,
    )
    assert code == 0
    output = stream.getvalue()
    assert "sample" in output
    assert "normal" in output
    assert "Student-t" in output
    assert "by construction" in output
    assert "not transferable" in output


def test_expectile_emits_json_with_the_transplant_cost_in_it(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        ["--json", "expectile", str(sample_file(tmp_path / "r.csv")), "--degrees", "5"],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue().split("\n\n")[-1])
    assert payload["observations"] == 300
    # The defining condition, carried on the result rather than recomputed.
    assert abs(payload["identity_residual"]) < 1e-15
    assert payload["exceedance_ratio"] == pytest.approx(
        (1.0 - payload["level"]) / payload["level"], rel=1e-9
    )
    # The level matched on the normal overstates the fatter-tailed shortfall,
    # which is the whole reason the command prints it.
    assert payload["transplant_overstatement"] > 0.1
    assert payload["matched_level"] > payload["level"]
    # json.dumps writes bare Infinity and NaN, and json.loads reads them back,
    # so nothing upstream notices a non-finite number reaching a payload.
    assert json.dumps(payload, allow_nan=False)


def test_expectile_refuses_a_level_outside_the_open_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run("expectile", str(sample_file(tmp_path / "r.csv")), "--level", "1.0")
    assert code == 2
    assert "strictly inside" in capsys.readouterr().err


# -- spectrum ----------------------------------------------------------------


def test_spectrum_matches_every_row_to_one_charge(tmp_path: Path) -> None:
    """The matching is the report: the headline figure is identical in each row."""
    stream = io.StringIO()
    code = main(
        ["--json", "spectrum", str(sample_file(tmp_path / "r.csv", periods=2000))],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue().split("\n\n")[-1])
    charges = [row["charge"] for row in payload["spectra"]]
    assert len(charges) == 4
    for charge in charges[1:]:
        assert charge == pytest.approx(charges[0], rel=1e-6)
    # And they do not all agree about where it came from.
    shares = [row["deep_tail_share"] for row in payload["spectra"]]
    assert max(shares) > 1.3 * min(shares)
    assert all(row["is_coherent"] for row in payload["spectra"])
    assert all(row["subadditive"] for row in payload["spectra"])
    assert all(abs(row["comonotonic_gap"]) < 1e-12 for row in payload["spectra"])


def test_spectrum_says_where_the_spectra_stop_agreeing(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(["spectrum", str(sample_file(tmp_path / "r.csv", periods=2000))], stream=stream)
    assert code == 0
    output = stream.getvalue()
    assert "matched to the" in output
    assert "Wang transform" in output
    assert "proportional hazards" in output
    assert "same number in every row" in output
    assert "comonotonically additive" in output


def test_spectrum_reports_a_non_coherent_spectrum_as_one(tmp_path: Path) -> None:
    """Asked for an increasing weight function, it prices it and says it fails."""
    stream = io.StringIO()
    code = main(
        [
            "spectrum",
            str(sample_file(tmp_path / "r.csv", periods=2000)),
            "--exponent",
            "0.4",
        ],
        stream=stream,
    )
    assert code == 0
    output = stream.getvalue()
    assert "is not coherent" in output
    assert "fails subadditivity" in output
    assert "NO" in output


def test_spectrum_json_carries_the_coherence_verdict(tmp_path: Path) -> None:
    stream = io.StringIO()
    code = main(
        [
            "--json",
            "spectrum",
            str(sample_file(tmp_path / "r.csv", periods=2000)),
            "--exponent",
            "0.4",
        ],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue().split("\n\n")[-1])
    offender = payload["spectra"][-1]
    assert offender["is_coherent"] is False
    assert offender["subadditive"] is False
    assert offender["subadditivity_gap"] > 0.0
    # The coherent rows are unaffected by its presence.
    assert all(row["subadditive"] for row in payload["spectra"][:-1])


def test_spectrum_reports_the_wang_shift_matching_the_normal(tmp_path: Path) -> None:
    """So the spectrum can be stated in units a desk already uses."""
    stream = io.StringIO()
    code = main(
        [
            "--json",
            "spectrum",
            str(sample_file(tmp_path / "r.csv", periods=2000)),
            "--confidence",
            "0.975",
        ],
        stream=stream,
    )
    assert code == 0
    payload = json.loads(stream.getvalue().split("\n\n")[-1])
    assert 1.5 < payload["wang_shift_matching_normal"] < 3.5


# -- the portfolio loss tail --------------------------------------------------


def obligor_file(path: Path, *, names: int = 100) -> Path:
    rng = random.Random(20261008)
    lines = ["probability,exposure"]
    for _ in range(names):
        lines.append(f"{round(0.004 + 0.075 * rng.random(), 6)},{rng.randint(1, 20)}")
    return write(path, "\n".join(lines) + "\n")


def test_portfolio_measures_the_error_against_the_exact_distribution(
    tmp_path: Path,
) -> None:
    """The exact column is what makes the saddlepoint column a measurement."""
    path = obligor_file(tmp_path / "book.csv")
    stream = io.StringIO()
    assert main(["--json", "portfolio", str(path)], stream) == 0
    payload = json.loads(stream.getvalue())
    assert payload["obligors"] == 100
    assert payload["lattice_span"] == 1.0
    assert payload["exact_available"] is True
    for entry in payload["levels"]:
        assert abs(entry["saddlepoint_error"]) < 2e-3
        assert entry["exact"] > 0.0
    deepest = payload["levels"][-1]
    assert abs(deepest["normal_error"]) > 0.9
    assert abs(deepest["saddlepoint_error"]) < 1e-3


def test_portfolio_contributions_sum_to_the_shortfall_numerator(
    tmp_path: Path,
) -> None:
    path = obligor_file(tmp_path / "book.csv")
    stream = io.StringIO()
    assert main(["--json", "portfolio", str(path), "--top", "100"], stream) == 0
    payload = json.loads(stream.getvalue())
    first = payload["levels"][0]
    total = math.fsum(entry["contribution"] for entry in payload["contributions"])
    assert total / first["saddlepoint"] == pytest.approx(first["conditional_mean"], rel=1e-10)
    shares = [entry["contribution"] for entry in payload["contributions"]]
    assert shares == sorted(shares, reverse=True)


def test_portfolio_treats_incommensurate_exposures_as_continuous(
    tmp_path: Path,
) -> None:
    """No common unit means no exact column, and the report says so."""
    path = write(
        tmp_path / "odd.csv",
        "probability,exposure\n0.02,1.0\n0.03,3.14159265358979\n0.04,2.5\n",
    )
    stream = io.StringIO()
    assert main(["portfolio", str(path)], stream) == 0
    assert "no common exposure unit" in stream.getvalue()


def test_portfolio_refuses_probabilities_in_per_cent(tmp_path: Path) -> None:
    """A file in the other unit parses perfectly and is wrong by a hundred."""
    path = write(tmp_path / "pc.csv", "probability,exposure\n2.0,100\n3.0,200\n")
    stream = io.StringIO()
    assert main(["portfolio", str(path)], stream) == 2


def test_portfolio_names_a_bad_line_and_an_empty_file(tmp_path: Path) -> None:
    path = write(tmp_path / "bad.csv", "probability,exposure\n0.02,100\nnonsense,x\n")
    stream = io.StringIO()
    assert main(["portfolio", str(path)], stream) == 2
    empty = write(tmp_path / "empty.csv", "# only a comment\n")
    assert main(["portfolio", str(empty)], stream) == 2
    missing = tmp_path / "nope.csv"
    assert main(["portfolio", str(missing)], stream) == 2


def test_portfolio_at_one_requested_level(tmp_path: Path) -> None:
    path = obligor_file(tmp_path / "book.csv")
    stream = io.StringIO()
    assert main(["--json", "portfolio", str(path), "--level", "143"], stream) == 0
    payload = json.loads(stream.getvalue())
    assert len(payload["levels"]) == 1
    assert payload["levels"][0]["level"] == 143.0
    assert payload["levels"][0]["exact"] == pytest.approx(4.426e-05, rel=0.01)


def test_portfolio_reports_a_clamped_level_rather_than_hiding_it(
    tmp_path: Path,
) -> None:
    """Two names at four hundred to one has no asymptotic regime to appeal to.

    At a loss of 400 the raw value is 4.15, so clamping is the only thing
    keeping the report inside a probability. The sequence is not monotone
    either: the raw value at a loss of 2 is above the one at a loss of 1.
    """
    path = write(tmp_path / "two.csv", "probability,exposure\n0.5,1\n0.5,400\n")
    stream = io.StringIO()
    assert main(["portfolio", str(path), "--level", "400"], stream) == 0
    text = stream.getvalue()
    assert "left [0, 1] before clamping" in text
    stream = io.StringIO()
    assert main(["--json", "portfolio", str(path), "--level", "400"], stream) == 0
    payload = json.loads(stream.getvalue())
    assert payload["levels"][0]["raw"] > 4.0
    assert payload["levels"][0]["saddlepoint"] == 1.0


# -- stress ------------------------------------------------------------------


def test_stress_reports_the_risk_before_and_after(tmp_path: Path) -> None:
    path = sample_file(tmp_path / "r.csv")
    code, out = run("stress", str(path), "--on", "alpha", "--mean", "-0.01")
    assert code == 0
    assert "view on alpha" in out
    assert "expected shortfall" in out
    assert "effective scenarios" in out
    assert "Nothing was discarded" in out


def test_stress_moves_the_mean_by_the_regression_slope(tmp_path: Path) -> None:
    """The report's own arithmetic, checked against itself through the payload.

    The predicted move is the slope times the realised move in the driver, and
    the mean of the portfolio moves by exactly that. It is the expected
    shortfall that does not, which is the point of the command.
    """
    path = sample_file(tmp_path / "r.csv")
    code, out = run(
        "--json", "stress", str(path), "--on", "alpha", "--mean", "-0.01"
    )
    assert code == 0
    payload = json.loads(out)
    predicted_mean = payload["mean_before"] + payload["regression_slope"] * payload[
        "driver_move"
    ]
    # To 0.7% here rather than the 0.1% of the thousand-scenario sample in
    # test_entropy: the agreement is exact only for a jointly normal sample and
    # this file has three hundred rows of a mixture.
    assert payload["mean_after"] == pytest.approx(predicted_mean, rel=1e-2)
    assert payload["captured_fraction"] != pytest.approx(1.0, rel=0.05)
    assert 0.0 < payload["concentration"] < 1.0
    assert payload["relative_entropy"] > 0.0


def test_stress_takes_a_tail_probability_view(tmp_path: Path) -> None:
    path = sample_file(tmp_path / "r.csv")
    code, out = run(
        "--json",
        "stress",
        str(path),
        "--on",
        "alpha",
        "--below",
        "-0.015",
        "--multiple",
        "3",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["views"] == ["P(alpha < -0.015)"]
    assert payload["expected_shortfall_after"] > payload["expected_shortfall_before"]


def test_stress_blends_back_towards_the_prior(tmp_path: Path) -> None:
    path = sample_file(tmp_path / "r.csv")
    full = json.loads(
        run("--json", "stress", str(path), "--on", "alpha", "--mean", "-0.01")[1]
    )
    half = json.loads(
        run(
            "--json",
            "stress",
            str(path),
            "--on",
            "alpha",
            "--mean",
            "-0.01",
            "--confidence-in-view",
            "0.5",
        )[1]
    )
    assert half["relative_entropy"] < full["relative_entropy"]
    assert half["effective_scenarios"] > full["effective_scenarios"]
    assert abs(half["driver_move"]) < abs(full["driver_move"])
    _, text = run(
        "stress",
        str(path),
        "--on",
        "alpha",
        "--mean",
        "-0.01",
        "--confidence-in-view",
        "0.5",
    )
    assert "holds only in part" in text


def test_stress_refuses_a_view_it_cannot_place(tmp_path: Path) -> None:
    path = sample_file(tmp_path / "r.csv")
    assert run("stress", str(path))[0] == 2
    assert run("stress", str(path), "--on", "absent", "--mean", "0.0")[0] == 2
    # A level no scenario reaches cannot be given any probability by reweighting
    # a sample, however the weights are chosen.
    assert run("stress", str(path), "--below", "-5.0", "--multiple", "2")[0] == 2
    # And a mean outside the range the scenarios span is refused by the solver.
    assert run("stress", str(path), "--on", "alpha", "--mean", "5.0")[0] == 2

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data_loader import build_dataset_spec_from_data
from core.experiment_runner import (
    ExperimentConfig,
    ModelResult,
    evaluate_model,
    train_nit,
    train_sy,
    train_wm,
)
from core.scenarios import (
    ScenarioConfig,
    apply_training_scenario,
    prepare_numeric_training_data,
    print_scenario_summary,
)


DEFAULT_PERIOD_COUNT = 3
DEFAULT_TRAIN_SAMPLE_SIZE = 450
DEFAULT_TEST_SAMPLE_SIZE = 450
DEFAULT_X_MIN = 0.0
DEFAULT_X_MAX = DEFAULT_PERIOD_COUNT * 2.0 * np.pi
OUT_OF_RANGE_EXTENSION_RATIO = 0.5

RULE_CONFIGS = [
    (5, 2),
    (17, 5),
    (29, 10),
    (51, 22),
    (153, 153),
]
PLOT_CONFIGS = [(5, 2), (17, 5), (29, 10), (51, 22), (153, 153)]
SCENARIO_RULE_CONFIGS = [(29, 10), (51, 22)]
TIMING_TABLE_CONFIG = (153, 153)
Y_LABELS = ["NB", "NS", "ZE", "PS", "PB"]


@dataclass(frozen=True)
class SineExperimentSpec:
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    scenario: ScenarioConfig


@dataclass(frozen=True)
class ExperimentRun:
    label_count: int
    sy_rule_count: int
    wm: ModelResult
    nit: ModelResult
    sy: ModelResult


def _generate_labels(count: int) -> list[str]:
    if count % 2 == 0:
        raise ValueError("Liczba etykiet musi byc nieparzysta.")

    middle = count // 2
    return [f"N{i}" for i in range(middle, 0, -1)] + ["ZE"] + [
        f"P{i}" for i in range(1, middle + 1)
    ]


def _true_process(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(x) + 1.0


def _build_base_dataset(sample_size: int, x_min: float, x_max: float) -> pd.DataFrame:
    x_values = np.linspace(x_min, x_max, sample_size)
    return pd.DataFrame({"x": x_values, "y": _true_process(x_values)})


def _build_out_of_range_test_data(reference_data: pd.DataFrame) -> pd.DataFrame:
    x_min = float(reference_data["x"].min())
    x_max = float(reference_data["x"].max())
    span = x_max - x_min
    return _build_base_dataset(
        len(reference_data),
        x_min - OUT_OF_RANGE_EXTENSION_RATIO * span,
        x_max + OUT_OF_RANGE_EXTENSION_RATIO * span,
    )


def _noise_std_from_y_percent(data: pd.DataFrame, percent: float) -> float:
    y_range = float(data["y"].max() - data["y"].min())
    return y_range * percent / 100.0


def _format_pi_tick(value: float, _position: int) -> str:
    multiplier = int(round(value / np.pi))
    if not np.isclose(value, multiplier * np.pi, atol=1e-8):
        return f"{value / np.pi:.1f}Pi"
    if multiplier == 0:
        return "0"
    if multiplier == 1:
        return "Pi"
    if multiplier == -1:
        return "-Pi"
    return f"{multiplier}Pi"


def _apply_angle_x_axis(*axes: Any) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    for axis in axes:
        axis.xaxis.set_major_locator(MultipleLocator(np.pi))
        axis.xaxis.set_major_formatter(FuncFormatter(_format_pi_tick))
        axis.set_xlabel("x")


def build_experiment_spec(
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
    test_sample_size: int = DEFAULT_TEST_SAMPLE_SIZE,
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    scenario: ScenarioConfig | None = None,
    seed: int = 42,
) -> SineExperimentSpec:
    scenario = scenario or ScenarioConfig()
    clean_train_data = _build_base_dataset(train_sample_size, x_min, x_max)
    train_data = apply_training_scenario(
        data=clean_train_data,
        scenario=scenario,
        seed=seed,
        gaussian_noise_columns=["y"],
        missing_columns=["x", "y"],
        outlier_columns=["y"],
    )
    train_data = prepare_numeric_training_data(train_data, sort_by="x", columns=["x", "y"])

    return SineExperimentSpec(
        train_data=train_data,
        test_data=_build_base_dataset(test_sample_size, x_min, x_max),
        scenario=scenario,
    )


def _build_config(
    train_data: pd.DataFrame,
    label_count: int,
    sy_rule_count: int,
    train_sample_size: int,
) -> ExperimentConfig:
    dataset_spec = build_dataset_spec_from_data(
        data=train_data,
        inputs=["x"],
        outputs=["y"],
        labels_by_variable={"x": _generate_labels(label_count), "y": Y_LABELS},
    )
    return ExperimentConfig(
        inputs=dataset_spec.inputs,
        outputs=dataset_spec.outputs,
        fuzzy_sets=dataset_spec.fuzzy_sets,
        universes=dataset_spec.universes,
        sample_size=train_sample_size,
        nit_params={"alpha": 1.0},
        sy_params={"n_rules": sy_rule_count, "eps_sigma": 0.4},
    )


def _train_and_evaluate(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_count: int,
    sy_rule_count: int,
    train_sample_size: int,
) -> ExperimentRun:
    config = _build_config(train_data, label_count, sy_rule_count, train_sample_size)
    wm_model, wm_time, wm_rule_time, wm_structure_time, wm_learning_time = train_wm(
        train_data, config
    )
    nit_model, nit_time, nit_rule_time, nit_structure_time, nit_learning_time = train_nit(
        train_data, config
    )
    sy_model, sy_time, sy_rule_time, sy_structure_time, sy_learning_time = train_sy(
        train_data, config
    )

    return ExperimentRun(
        label_count=label_count,
        sy_rule_count=sy_rule_count,
        wm=evaluate_model(
            wm_model,
            "wm",
            test_data,
            config,
            wm_time,
            wm_rule_time,
            wm_structure_time,
            wm_learning_time,
        ),
        nit=evaluate_model(
            nit_model,
            "nit",
            test_data,
            config,
            nit_time,
            nit_rule_time,
            nit_structure_time,
            nit_learning_time,
        ),
        sy=evaluate_model(
            sy_model,
            "sy",
            test_data,
            config,
            sy_time,
            sy_rule_time,
            sy_structure_time,
            sy_learning_time,
        ),
    )


def _print_scenario_summary(spec: SineExperimentSpec) -> None:
    print_scenario_summary(
        title="EKSPERYMENT: y = 2sin(x) + 1",
        scenario=spec.scenario,
        sample_size=len(spec.train_data),
        missing_columns=["x", "y"],
        outlier_columns=["y"],
    )


def _metric_rows(run_result: ExperimentRun) -> list[dict[str, Any]]:
    rows = []
    for method in ["wm", "nit", "sy"]:
        result = getattr(run_result, method)
        rows.append(
            {
                "wm_nit_label_count": run_result.label_count if method != "sy" else "N/A",
                "sy_requested_rules": run_result.sy_rule_count if method == "sy" else "N/A",
                "method": method.upper(),
                "rule_count": result.rule_count,
                "mse": result.mse,
                "mae": result.mae,
                "rmse": result.rmse,
                "r_squared": result.r_squared,
                "training_time_seconds": result.training_time_seconds,
                "rule_creation_time_seconds": result.rule_creation_time_seconds,
                "structure_time_seconds": result.structure_time_seconds,
                "learning_time_seconds": result.learning_time_seconds,
            }
        )
    return rows


def _print_metrics_table(title: str, rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"{'WM/NIT':<8} {'SY':<8} {'Metoda':<8} {'Reguly':<8} "
        f"{'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['wm_nit_label_count']!s:<8} {row['sy_requested_rules']!s:<8} "
            f"{row['method']:<8} {row['rule_count']:<8} {row['mse']:<12.6f} "
            f"{row['mae']:<12.6f} {row['rmse']:<12.6f} {row['r_squared']:<12.6f}"
        )


def _print_timing_table(run_result: ExperimentRun) -> None:
    print("\n" + "=" * 88)
    print(
        "TABELA CZASOW: "
        f"WM/NIT={run_result.label_count}, SY={run_result.sy_rule_count}"
    )
    print("=" * 88)
    print(
        f"{'Metoda':<8} {'Reguly':<8} {'Trening [s]':<14} "
        f"{'Tworzenie [s]':<15} {'Struktura [s]':<15} {'Uczenie [s]':<12}"
    )
    print("-" * 88)
    for method in ["wm", "nit", "sy"]:
        result = getattr(run_result, method)
        print(
            f"{method.upper():<8} {result.rule_count:<8} "
            f"{result.training_time_seconds:<14.6f} "
            f"{result.rule_creation_time_seconds:<15.6f} "
            f"{result.structure_time_seconds:<15.6f} "
            f"{result.learning_time_seconds:<12.6f}"
        )


def _save_summary(rows: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nZapisano wyniki do: {path}")


def _configure_matplotlib() -> None:
    matplotlib_config_dir = os.path.join(
        os.getcwd(),
        "results",
        ".matplotlib",
        f"{os.getpid()}_{uuid.uuid4().hex}",
    )
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = matplotlib_config_dir


def _plot_result_set(
    test_data: pd.DataFrame,
    run_result: ExperimentRun,
    output_path: str,
    title_prefix: str = "Sinus",
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    x_test = test_data["x"].to_numpy(dtype=float)
    y_true = test_data["y"].to_numpy(dtype=float)
    sort_idx = np.argsort(x_test)
    x_sorted = x_test[sort_idx]
    y_sorted = y_true[sort_idx]

    styles = {
        "wm": ("Wang-Mendel", "r--"),
        "nit": ("Nozaki-Ishibuchi-Tanaka", "g--"),
        "sy": ("Sugeno-Yasukawa", "m--"),
    }
    fig, axes = plt.subplots(3, 1, figsize=(7, 12))
    for axis, method in zip(axes, ["wm", "nit", "sy"]):
        result = getattr(run_result, method)
        displayed_rule_count = (
            run_result.sy_rule_count if method == "sy" else run_result.label_count
        )
        pred = np.asarray(result.predictions["y"], dtype=float).ravel()[sort_idx]
        method_name, line_style = styles[method]
        axis.plot(x_sorted, y_sorted, "b-", label="y_true", linewidth=2)
        axis.plot(x_sorted, pred, line_style, label="y_pred", linewidth=1.5, alpha=0.85)
        axis.set_ylabel("y")
        axis.set_title(
            f"{method_name}\nR^2={result.r_squared:.4f}, reguly={displayed_rule_count}"
        )
        axis.legend()
        axis.grid(True, alpha=0.3)

    _apply_angle_x_axis(*axes)
    fig.suptitle(
        f"{title_prefix}: WM/NIT={run_result.label_count}, SY={run_result.sy_rule_count}",
        fontsize=14,
    )
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano wykres: {output}")


def run_rule_variation_experiments(
    spec: SineExperimentSpec,
    train_sample_size: int,
) -> list[ExperimentRun]:
    results = []
    for label_count, sy_rule_count in RULE_CONFIGS:
        print(
            "\n"
            + "=" * 70
            + f"\nREGULY: WM/NIT={label_count}, SY={sy_rule_count}\n"
            + "=" * 70
        )
        results.append(
            _train_and_evaluate(
                spec.train_data,
                spec.test_data,
                label_count,
                sy_rule_count,
                train_sample_size,
            )
        )
    return results


def plot_rule_variations(
    spec: SineExperimentSpec,
    rule_results: list[ExperimentRun],
) -> None:
    print("\n" + "=" * 70)
    print("GENEROWANIE WYKRESOW DLA ZESTAWOW REGUL")
    print("=" * 70)
    results_by_config = {
        (result.label_count, result.sy_rule_count): result for result in rule_results
    }
    for config in PLOT_CONFIGS:
        run_result = results_by_config.get(config)
        if run_result is None:
            print(
                "Pominieto wykres dla brakujacego zestawu: "
                f"WM/NIT={config[0]}, SY={config[1]}"
            )
            continue
        output_path = (
            "results/plots/"
            f"sinus_comparison_wmnit_{run_result.label_count}_sy_{run_result.sy_rule_count}.png"
        )
        _plot_result_set(spec.test_data, run_result, output_path)


def _scenario_definitions(spec: SineExperimentSpec) -> list[tuple[str, ScenarioConfig]]:
    return [
        ("baseline", ScenarioConfig()),
        ("outliers_5pct", ScenarioConfig(outlier_ratio=0.05)),
        ("outliers_10pct", ScenarioConfig(outlier_ratio=0.10)),
        (
            "gaussian_1pct",
            ScenarioConfig(gaussian_noise_std=_noise_std_from_y_percent(spec.train_data, 1.0)),
        ),
        (
            "gaussian_2pct",
            ScenarioConfig(gaussian_noise_std=_noise_std_from_y_percent(spec.train_data, 2.0)),
        ),
        ("missing_5pct", ScenarioConfig(missing_ratio=0.05)),
        ("missing_10pct", ScenarioConfig(missing_ratio=0.10)),
        ("out_of_range", ScenarioConfig()),
    ]


def run_scenario_variation_experiments(
    spec: SineExperimentSpec,
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
    rule_configs: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    rule_configs = rule_configs or SCENARIO_RULE_CONFIGS
    print("\n" + "=" * 70)
    print("SCENARIUSZE")
    print("=" * 70)

    scenario_results = []
    for label_count, sy_rule_count in rule_configs:
        print(
            "\n"
            + "=" * 70
            + f"\nSCENARIUSZE: WM/NIT={label_count}, SY={sy_rule_count}\n"
            + "=" * 70
        )
        for scenario_name, scenario in _scenario_definitions(spec):
            print(f"\nScenariusz: {scenario_name}")
            if scenario_name == "out_of_range":
                train_data = spec.train_data.copy()
                test_data = _build_out_of_range_test_data(spec.test_data)
            else:
                train_data = apply_training_scenario(
                    data=spec.train_data,
                    scenario=scenario,
                    seed=42,
                    gaussian_noise_columns=["y"],
                    missing_columns=["x", "y"],
                    outlier_columns=["y"],
                )
                train_data = prepare_numeric_training_data(
                    train_data,
                    sort_by="x",
                    columns=["x", "y"],
                )
                test_data = spec.test_data

            run_result = _train_and_evaluate(
                train_data,
                test_data,
                label_count,
                sy_rule_count,
                train_sample_size,
            )
            scenario_results.append(
                {
                    "scenario": scenario_name,
                    "test_data": test_data,
                    "result": run_result,
                }
            )
            _print_metrics_table(
                f"SCENARIUSZ {scenario_name} | WM/NIT={label_count}, SY={sy_rule_count}",
                _metric_rows(run_result),
            )

    rows = []
    for entry in scenario_results:
        for row in _metric_rows(entry["result"]):
            rows.append({"scenario": entry["scenario"], **row})
    _save_summary(rows, "results/summaries/sinus_scenario_variation.csv")
    return scenario_results


def plot_scenario_comparison(
    scenario_results: list[dict[str, Any]],
    output_dir: str = "results/plots",
) -> None:
    print("\n" + "=" * 70)
    print("GENEROWANIE WYKRESOW DLA SCENARIUSZY")
    print("=" * 70)
    for entry in scenario_results:
        run_result = entry["result"]
        output_path = (
            f"{output_dir}/sinus_scenario_{entry['scenario']}_"
            f"wmnit_{run_result.label_count}_sy_{run_result.sy_rule_count}.png"
        )
        _plot_result_set(
            entry["test_data"],
            run_result,
            output_path,
            title_prefix=f"Scenariusz {entry['scenario']}",
        )


def run(
    scenario: ScenarioConfig | None = None,
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
    test_sample_size: int = DEFAULT_TEST_SAMPLE_SIZE,
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    seed: int = 42,
) -> None:
    spec = build_experiment_spec(
        train_sample_size=train_sample_size,
        test_sample_size=test_sample_size,
        x_min=x_min,
        x_max=x_max,
        scenario=scenario,
        seed=seed,
    )
    _print_scenario_summary(spec)

    rule_results = run_rule_variation_experiments(spec, train_sample_size)
    summary_rows = [row for result in rule_results for row in _metric_rows(result)]
    _print_metrics_table("WYNIKI DLA ZESTAWOW REGUL", summary_rows)
    _save_summary(summary_rows, "results/summaries/sinus_rule_variation.csv")

    timing_result = next(
        result
        for result in rule_results
        if (result.label_count, result.sy_rule_count) == TIMING_TABLE_CONFIG
    )
    _print_timing_table(timing_result)

    plot_rule_variations(spec, rule_results)
    scenario_results = run_scenario_variation_experiments(spec, train_sample_size)
    plot_scenario_comparison(scenario_results)


if __name__ == "__main__":
    run()

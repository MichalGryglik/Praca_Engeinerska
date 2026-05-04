from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.data_loader import build_dataset_spec_from_data, load_tep_test, load_tep_train
from core.experiment_runner import (
    ExperimentConfig,
    ModelResult,
    evaluate_model,
    plot_predictions_vs_true,
    print_summary,
    train_nit,
    train_sy,
    train_wm,
)
from core.results_writer import save_metrics_summary
from core.scenarios import (
    ScenarioConfig,
    apply_training_scenario,
    prepare_numeric_training_data,
    print_scenario_summary,
)


BASE_INTERVAL_LABELS = ["S2", "S1", "CE", "B1", "B2"]
BASE_RULE_CONFIGS = [
    (10, 10),
]
STRIPPER_RULE_VARIATION_CONFIGS = [
    (3, 3),
    (5, 3),
    (20, 50),
]
AUTOREGRESSION_CONFIG = (10, 10)
STRIPPER_EXPERIMENT_NAME = "stripper_level"
TEP_EXPERIMENT_SPECS = [
    {
        "name": "valve_to_flow",
        "inputs": ["xmv_9"],
        "outputs": ["xmeas_19"],
    },
    {
        "name": "stripper_level",
        "inputs": ["xmv_8", "xmeas_17", "xmv_9", "xmeas_19"],
        "outputs": ["xmeas_15"],
    },
    {
        "name": "stripper_underflow",
        "inputs": ["xmv_8"],
        "outputs": ["xmeas_17"],
    },
]


@dataclass(frozen=True)
class TepExperimentRun:
    interval_count: int
    sy_rule_count: int
    wm: ModelResult
    nit: ModelResult
    sy: ModelResult


def _generate_interval_labels(count: int) -> list[str]:
    if count == len(BASE_INTERVAL_LABELS):
        return BASE_INTERVAL_LABELS.copy()
    if count <= 0:
        raise ValueError("Liczba przedzialow musi byc dodatnia.")
    return [f"I{idx:02d}" for idx in range(1, count + 1)]


def _unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _single_output_experiment_specs() -> list[dict[str, list[str] | str]]:
    expanded_specs = []
    for experiment_spec in TEP_EXPERIMENT_SPECS:
        outputs = experiment_spec["outputs"]
        if len(outputs) == 1:
            expanded_specs.append(experiment_spec)
            continue

        for output_name in outputs:
            expanded_specs.append(
                {
                    "name": f"{experiment_spec['name']}_{output_name}",
                    "inputs": experiment_spec["inputs"],
                    "outputs": [output_name],
                }
            )
    return expanded_specs


def _build_sigma_floor_by_input(
    train_data: pd.DataFrame,
    inputs: list[str],
    range_ratio: float = 0.05,
) -> dict[str, float]:
    sigma_floor = {}
    for input_name in inputs:
        input_range = float(train_data[input_name].max() - train_data[input_name].min())
        sigma_floor[input_name] = max(input_range * range_ratio, 1e-6)
    return sigma_floor


def _build_config(
    train_data: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    sample_size: int,
    interval_count: int,
    sy_rule_count: int,
) -> ExperimentConfig:
    labels = _generate_interval_labels(interval_count)
    variable_names = _unique_columns(inputs + outputs)
    labels_by_variable = {
        variable_name: labels
        for variable_name in variable_names
    }
    tep_spec = build_dataset_spec_from_data(
        data=train_data[variable_names],
        inputs=inputs,
        outputs=outputs,
        labels_by_variable=labels_by_variable,
    )

    return ExperimentConfig(
        inputs=tep_spec.inputs,
        outputs=tep_spec.outputs,
        fuzzy_sets=tep_spec.fuzzy_sets,
        universes=tep_spec.universes,
        sample_size=sample_size,
        nit_params={"alpha": 1.0},
        sy_params={
            "n_rules": sy_rule_count,
            "eps_sigma": _build_sigma_floor_by_input(train_data, inputs),
        },
    )


def _train_and_evaluate(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    sample_size: int,
    interval_count: int,
    sy_rule_count: int,
) -> TepExperimentRun:
    config = _build_config(
        train_data=train_data,
        inputs=inputs,
        outputs=outputs,
        sample_size=sample_size,
        interval_count=interval_count,
        sy_rule_count=sy_rule_count,
    )

    (
        wm_model,
        wm_training_time,
        wm_rule_creation_time,
        wm_structure_time,
        wm_learning_time,
    ) = train_wm(train_data, config)
    (
        nit_model,
        nit_training_time,
        nit_rule_creation_time,
        nit_structure_time,
        nit_learning_time,
    ) = train_nit(train_data, config)
    (
        sy_model,
        sy_training_time,
        sy_rule_creation_time,
        sy_structure_time,
        sy_learning_time,
    ) = train_sy(train_data, config)

    return TepExperimentRun(
        interval_count=interval_count,
        sy_rule_count=sy_rule_count,
        wm=evaluate_model(
            wm_model,
            "wm",
            test_data,
            config,
            training_time_seconds=wm_training_time,
            rule_creation_time_seconds=wm_rule_creation_time,
            structure_time_seconds=wm_structure_time,
            learning_time_seconds=wm_learning_time,
        ),
        nit=evaluate_model(
            nit_model,
            "nit",
            test_data,
            config,
            training_time_seconds=nit_training_time,
            rule_creation_time_seconds=nit_rule_creation_time,
            structure_time_seconds=nit_structure_time,
            learning_time_seconds=nit_learning_time,
        ),
        sy=evaluate_model(
            sy_model,
            "sy",
            test_data,
            config,
            training_time_seconds=sy_training_time,
            rule_creation_time_seconds=sy_rule_creation_time,
            structure_time_seconds=sy_structure_time,
            learning_time_seconds=sy_learning_time,
        ),
    )


def _metric_rows(run_result: TepExperimentRun) -> list[dict[str, Any]]:
    rows = []
    for method in ["wm", "nit", "sy"]:
        result = getattr(run_result, method)
        rows.append(
            {
                "interval_count": run_result.interval_count,
                "sy_requested_rules": (
                    run_result.sy_rule_count if method == "sy" else "N/A"
                ),
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


def _print_rule_variation_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 82)
    print("WYNIKI TEP DLA LICZBY PRZEDZIALOW I REGUL SY")
    print("=" * 82)
    print(
        f"{'Przedz.':<9} {'SY':<8} {'Metoda':<8} {'Reguly':<8} "
        f"{'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{row['interval_count']:<9} {row['sy_requested_rules']!s:<8} "
            f"{row['method']:<8} {row['rule_count']:<8} {row['mse']:<12.6f} "
            f"{row['mae']:<12.6f} {row['rmse']:<12.6f} {row['r_squared']:<12.6f}"
        )


def _print_single_run_table(title: str, run_result: TepExperimentRun) -> None:
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)
    print(
        f"{'Metoda':<8} {'Reguly':<8} {'MSE':<12} {'MAE':<12} "
        f"{'RMSE':<12} {'R^2':<12}"
    )
    print("-" * 82)
    for method in ["wm", "nit", "sy"]:
        result = getattr(run_result, method)
        print(
            f"{method.upper():<8} {result.rule_count:<8} {result.mse:<12.6f} "
            f"{result.mae:<12.6f} {result.rmse:<12.6f} "
            f"{result.r_squared:<12.6f}"
        )


def _save_rule_variation_summary(rows: list[dict[str, Any]]) -> Path:
    output_path = Path("results/summaries/tep_rule_variation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _run_experiment_spec(
    experiment_name: str,
    tep_train: pd.DataFrame,
    tep_test: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    sample_size: int,
    rule_configs: list[tuple[int, int]],
) -> tuple[list[TepExperimentRun], list[dict[str, Any]]]:
    print("\n" + "#" * 82)
    print(
        f"TEP: {experiment_name} | inputs={', '.join(inputs)} "
        f"-> outputs={', '.join(outputs)}"
    )
    print("#" * 82)

    autoregression_inputs = outputs.copy()
    autoregression_interval_count, autoregression_sy_rule_count = AUTOREGRESSION_CONFIG
    autoregression_result = _train_and_evaluate(
        train_data=tep_train,
        test_data=tep_test,
        inputs=autoregression_inputs,
        outputs=outputs,
        sample_size=sample_size,
        interval_count=autoregression_interval_count,
        sy_rule_count=autoregression_sy_rule_count,
    )
    _print_single_run_table(
        (
            "AUTOREGRESJA WYJSCIA TEP "
            f"(input=output={outputs[0]}, przedzialy={autoregression_interval_count}, "
            f"SY={autoregression_sy_rule_count})"
        ),
        autoregression_result,
    )

    run_results = []
    for interval_count, sy_rule_count in rule_configs:
        print(
            "\n"
            + "=" * 70
            + f"\nTEP: przedzialy={interval_count}, reguly SY={sy_rule_count}\n"
            + "=" * 70
        )
        result = _train_and_evaluate(
            train_data=tep_train,
            test_data=tep_test,
            inputs=inputs,
            outputs=outputs,
            sample_size=sample_size,
            interval_count=interval_count,
            sy_rule_count=sy_rule_count,
        )
        run_results.append(result)
        print_summary(result.wm, result.nit, result.sy)
        plot_predictions_vs_true(
            result.wm,
            result.nit,
            result.sy,
            title=(
                f"Tennessee Eastman Process: {experiment_name}, "
                f"przedzialy={interval_count}, SY={sy_rule_count}"
            ),
            output_path=(
                "results/plots/"
                f"tep_{experiment_name}_intervals_{interval_count}_sy_{sy_rule_count}.png"
            ),
        )

    summary_rows = [
        {"experiment": experiment_name, **row}
        for result in run_results
        for row in _metric_rows(result)
    ]
    _print_rule_variation_table(summary_rows)
    return run_results, summary_rows


def run(scenario: ScenarioConfig | None = None, seed: int = 42):
    scenario = scenario or ScenarioConfig()
    run_config = {
        "sample_size": 1000,
        "test_sample_size": 1000,
    }
    experiment_specs = _single_output_experiment_specs()
    all_inputs = [
        input_name
        for experiment_spec in experiment_specs
        for input_name in experiment_spec["inputs"]
    ]
    all_outputs = [
        output_name
        for experiment_spec in experiment_specs
        for output_name in experiment_spec["outputs"]
    ]

    tep_train = load_tep_train(n_samples=run_config["sample_size"])
    tep_test = load_tep_test(n_samples=run_config["test_sample_size"])
    scenario_columns = _unique_columns(all_inputs + all_outputs)
    scenario_outputs = _unique_columns(all_outputs)
    tep_train = tep_train.copy()
    tep_train[scenario_columns] = apply_training_scenario(
        data=tep_train[scenario_columns],
        scenario=scenario,
        seed=seed,
        gaussian_noise_columns=scenario_outputs,
        missing_columns=scenario_columns,
        outlier_columns=scenario_outputs,
    )
    tep_train[scenario_columns] = prepare_numeric_training_data(
        tep_train[scenario_columns],
        columns=scenario_columns,
    )

    print_scenario_summary(
        title="EKSPERYMENT: Tennessee Eastman Process",
        scenario=scenario,
        sample_size=run_config["sample_size"],
        missing_columns=scenario_columns,
        outlier_columns=scenario_outputs,
    )

    all_run_results = []
    all_summary_rows = []
    for experiment_spec in experiment_specs:
        rule_configs = BASE_RULE_CONFIGS.copy()
        if experiment_spec["name"] == STRIPPER_EXPERIMENT_NAME:
            rule_configs.extend(STRIPPER_RULE_VARIATION_CONFIGS)

        run_results, summary_rows = _run_experiment_spec(
            experiment_name=experiment_spec["name"],
            tep_train=tep_train,
            tep_test=tep_test,
            inputs=experiment_spec["inputs"],
            outputs=experiment_spec["outputs"],
            sample_size=run_config["sample_size"],
            rule_configs=rule_configs,
        )
        all_run_results.extend(run_results)
        all_summary_rows.extend(summary_rows)

    rule_variation_path = _save_rule_variation_summary(all_summary_rows)
    print(f"Zapisano zbiorcze wyniki wariantow: {rule_variation_path}")

    metrics_path = save_metrics_summary(
        "tep",
        [
            model_result
            for result in all_run_results
            for model_result in [result.wm, result.nit, result.sy]
        ],
    )
    print(f"Zapisano metryki: {metrics_path}")

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.data_loader import build_dataset_spec_from_data
from core.experiment_runner import (
    ExperimentConfig,
    evaluate_model,
    print_sample_preview,
    print_summary,
    train_nit,
    train_sy,
    train_wm,
)
from core.results_writer import save_metrics_summary, save_predictions
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


@dataclass(frozen=True)
class SineExperimentSpec:
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    scenario: ScenarioConfig


def _true_process(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(x) + 1.0


def _build_base_dataset(
    sample_size: int,
    x_min: float,
    x_max: float,
) -> pd.DataFrame:
    x_values = np.linspace(x_min, x_max, sample_size)
    return pd.DataFrame({"x": x_values, "y": _true_process(x_values)})


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
    test_data = _build_base_dataset(test_sample_size, x_min, x_max)

    scenario_train_data = apply_training_scenario(
        data=clean_train_data,
        scenario=scenario,
        seed=seed,
        gaussian_noise_columns=["y"],
        missing_columns=["x", "y"],
        outlier_columns=["y"],
    )
    prepared_train_data = prepare_numeric_training_data(
        scenario_train_data,
        sort_by="x",
        columns=["x", "y"],
    )
    return SineExperimentSpec(
        train_data=prepared_train_data,
        test_data=test_data,
        scenario=scenario,
    )


def _print_scenario_summary(spec: SineExperimentSpec) -> None:
    print_scenario_summary(
        title="EKSPERYMENT: y = 2sin(x) + 1",
        scenario=spec.scenario,
        sample_size=len(spec.train_data),
        missing_columns=["x", "y"],
        outlier_columns=["y"],
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

    labels_by_variable = {
        "x": ["N8", "N7", "N6", "N5", "N4", "N3", "N2", "N1", "ZE", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
        "y": ["NB", "NS", "ZE", "PS", "PB"],
    }
    dataset_spec = build_dataset_spec_from_data(
        data=spec.train_data,
        inputs=["x"],
        outputs=["y"],
        labels_by_variable=labels_by_variable,
    )
    config = ExperimentConfig(
        inputs=dataset_spec.inputs,
        outputs=dataset_spec.outputs,
        fuzzy_sets=dataset_spec.fuzzy_sets,
        universes=dataset_spec.universes,
        sample_size=train_sample_size,
        sy_params={"n_rules": 6, "eps_sigma": 0.4},
    )

    print("\nPodglad danych testowych:")
    print_sample_preview(spec.test_data.head(5), config)

    (
        wm_model,
        wm_training_time,
        wm_rule_creation_time,
        wm_structure_time,
        wm_learning_time,
    ) = train_wm(spec.train_data, config)
    (
        nit_model,
        nit_training_time,
        nit_rule_creation_time,
        nit_structure_time,
        nit_learning_time,
    ) = train_nit(spec.train_data, config)
    (
        sy_model,
        sy_training_time,
        sy_rule_creation_time,
        sy_structure_time,
        sy_learning_time,
    ) = train_sy(spec.train_data, config)

    wm_results = evaluate_model(
        wm_model,
        "wm",
        spec.test_data,
        config,
        training_time_seconds=wm_training_time,
        rule_creation_time_seconds=wm_rule_creation_time,
        structure_time_seconds=wm_structure_time,
        learning_time_seconds=wm_learning_time,
    )
    nit_results = evaluate_model(
        nit_model,
        "nit",
        spec.test_data,
        config,
        training_time_seconds=nit_training_time,
        rule_creation_time_seconds=nit_rule_creation_time,
        structure_time_seconds=nit_structure_time,
        learning_time_seconds=nit_learning_time,
    )
    sy_results = evaluate_model(
        sy_model,
        "sy",
        spec.test_data,
        config,
        training_time_seconds=sy_training_time,
        rule_creation_time_seconds=sy_rule_creation_time,
        structure_time_seconds=sy_structure_time,
        learning_time_seconds=sy_learning_time,
    )

    print_summary(wm_results, nit_results, sy_results)
    metrics_path = save_metrics_summary(
        "sinus_baseline",
        [wm_results, nit_results, sy_results],
    )
    predictions_path = save_predictions(
        [wm_results, nit_results, sy_results],
        "results/predictions/sinus_baseline.csv",
    )
    print(f"Zapisano metryki: {metrics_path}")
    print(f"Zapisano predykcje: {predictions_path}")


if __name__ == "__main__":
    run()

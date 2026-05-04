"""Uniwersalne funkcje pomocnicze do uruchamiania eksperymentów."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from core.evaluation.metrics import (
    compute_mae,
    compute_mse,
    compute_r_squared,
    compute_rmse,
)
from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from core.rule_generators import sugeno_yasukawa as sy
from core.rule_generators import wang_mendel as wm


DEFAULT_SY_PARAMS = {
    "n_rules": 3,
    "m": 2.0,
    "eps_sigma": 1.0,
}

DEFAULT_NIT_PARAMS = {
    "alpha": 1.0,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Konfiguracja wspólna dla różnych eksperymentów."""

    inputs: list[str]
    outputs: list[str]
    fuzzy_sets: dict[str, Any]
    universes: dict[str, Any]
    sample_size: int | None = None
    fault: int | None = None
    nit_params: dict[str, Any] | None = None
    sy_params: dict[str, Any] | None = None

    def merged_nit_params(self) -> dict[str, Any]:
        params = DEFAULT_NIT_PARAMS.copy()
        if self.nit_params:
            params.update(self.nit_params)
        return params

    def merged_sy_params(self) -> dict[str, Any]:
        params = DEFAULT_SY_PARAMS.copy()
        if self.sy_params:
            params.update(self.sy_params)
        return params


@dataclass
class ModelResult:
    """Wynik ewaluacji jednego modelu."""

    name: str
    predictions: dict[str, np.ndarray]
    y_true: dict[str, np.ndarray]
    rule_count: int
    mse: float
    mae: float
    rmse: float
    r_squared: float
    training_time_seconds: float
    rule_creation_time_seconds: float
    structure_time_seconds: float
    learning_time_seconds: float


def _measure_training_time(training_callback) -> tuple[dict[str, Any], float]:
    """Uruchamia trening i zwraca model wraz z czasem trenowania w sekundach."""
    start_time = perf_counter()
    model = training_callback()
    training_time_seconds = perf_counter() - start_time
    return model, training_time_seconds


def train_wm(
    train_data: pd.DataFrame, config: ExperimentConfig
) -> tuple[dict[str, Any], float, float, float, float]:
    """Trenuje model Wang-Mendel."""
    model, training_time_seconds = _measure_training_time(
        lambda: wm.generate_rules(
            data=train_data,
            inputs=config.inputs,
            outputs=config.outputs,
            fuzzy_sets=config.fuzzy_sets,
            universes=config.universes,
        )
    )
    return (
        model,
        training_time_seconds,
        training_time_seconds,
        training_time_seconds,
        0.0,
    )


def train_nit(
    train_data: pd.DataFrame, config: ExperimentConfig
) -> tuple[dict[str, Any], float, float, float, float]:
    """Trenuje model Nozaki-Ishibuchi-Tanaka."""
    nit_params = config.merged_nit_params()
    model, training_time_seconds = _measure_training_time(
        lambda: nit.generate_rules(
            data=train_data,
            inputs=config.inputs,
            outputs=config.outputs,
            fuzzy_sets=config.fuzzy_sets,
            universes=config.universes,
            alpha=nit_params["alpha"],
        )
    )
    return (
        model,
        training_time_seconds,
        training_time_seconds,
        training_time_seconds,
        0.0,
    )


def train_sy(
    train_data: pd.DataFrame, config: ExperimentConfig
) -> tuple[dict[str, Any], float, float, float, float]:
    """Trenuje model Sugeno-Yasukawa."""
    def _train() -> dict[str, Any]:
        sy_params = config.merged_sy_params()
        structure_start_time = perf_counter()

        centers, membership_matrix = sy.initialize_clusters_with_cmeans(
            data=train_data,
            inputs=config.inputs,
            n_rules=sy_params["n_rules"],
            m=sy_params["m"],
        )
        sigmas = sy.estimate_cluster_sigmas(
            data=train_data,
            inputs=config.inputs,
            membership_matrix=membership_matrix,
            m=sy_params["m"],
            eps_sigma=sy_params["eps_sigma"],
        )

        rules_dict = sy.build_initial_rules_from_clusters(
            centers=centers,
            inputs=config.inputs,
            outputs=config.outputs,
            eps_sigma=sy_params["eps_sigma"],
            sigmas=sigmas,
        )
        nonlocal rule_creation_time_seconds, structure_time_seconds
        rule_creation_time_seconds = perf_counter() - structure_start_time
        structure_time_seconds = rule_creation_time_seconds

        normalized_strengths = sy.compute_normalized_firing_strengths(
            data=train_data,
            inputs=config.inputs,
            rules_dict=rules_dict,
            fuzzy_sets=config.fuzzy_sets,
            universes=config.universes,
        )

        learning_start_time = perf_counter()
        sy.update_consequents_ls_wls(
            data=train_data,
            inputs=config.inputs,
            outputs=config.outputs,
            rules_dict=rules_dict,
            normalized_strengths=normalized_strengths,
        )
        sy.update_antecedents(
            data=train_data,
            inputs=config.inputs,
            rules_dict=rules_dict,
            normalized_strengths=normalized_strengths,
            eps_sigma=sy_params["eps_sigma"],
        )
        nonlocal learning_time_seconds
        learning_time_seconds = perf_counter() - learning_start_time
        return rules_dict

    rule_creation_time_seconds = 0.0
    structure_time_seconds = 0.0
    learning_time_seconds = 0.0
    model, training_time_seconds = _measure_training_time(_train)
    return (
        model,
        training_time_seconds,
        rule_creation_time_seconds,
        structure_time_seconds,
        learning_time_seconds,
    )


def evaluate_model(
    model: dict[str, Any],
    model_type: str,
    test_data: pd.DataFrame,
    config: ExperimentConfig,
    training_time_seconds: float = 0.0,
    rule_creation_time_seconds: float = 0.0,
    structure_time_seconds: float = 0.0,
    learning_time_seconds: float = 0.0,
) -> ModelResult:
    """Uruchamia predykcję i liczy podstawowe metryki dla modelu."""
    predictors = {
        "wm": lambda: wm.predict(
            data=test_data,
            inputs=config.inputs,
            outputs=config.outputs,
            rules_dict=model,
            fuzzy_sets=config.fuzzy_sets,
            universes=config.universes,
        ),
        "nit": lambda: nit.predict(
            data=test_data,
            inputs=config.inputs,
            outputs=config.outputs,
            rules_dict=model,
            fuzzy_sets=config.fuzzy_sets,
            universes=config.universes,
        ),
        "sy": lambda: sy.predict(
            data=test_data,
            inputs=config.inputs,
            outputs=config.outputs,
            rules_dict=model,
        ),
    }

    if model_type not in predictors:
        raise ValueError(f"Unsupported model type: {model_type}")

    predictions = predictors[model_type]()
    y_true = {
        output_name: test_data[output_name].to_numpy(dtype=float)
        for output_name in config.outputs
    }

    return ModelResult(
        name=model_type,
        predictions=predictions,
        y_true=y_true,
        rule_count=len(model),
        mse=compute_mse(y_true, predictions),
        mae=compute_mae(y_true, predictions),
        rmse=compute_rmse(y_true, predictions),
        r_squared=compute_r_squared(y_true, predictions),
        training_time_seconds=training_time_seconds,
        rule_creation_time_seconds=rule_creation_time_seconds,
        structure_time_seconds=structure_time_seconds,
        learning_time_seconds=learning_time_seconds,
    )


def print_sample_preview(test_data: pd.DataFrame, config: ExperimentConfig) -> None:
    """Wypisuje krótki podgląd próbek testowych."""
    print("\nProbki testowe:")
    for idx, (_, row) in enumerate(test_data.iterrows(), start=1):
        inputs_text = ", ".join(
            f"{input_name}={row[input_name]:.3f}" for input_name in config.inputs
        )
        outputs_text = ", ".join(
            f"{output_name}={row[output_name]:.3f}" for output_name in config.outputs
        )
        print(f"  Probka {idx}: {inputs_text} -> oczekiwane {outputs_text}")


def _print_results_table(*results: ModelResult) -> None:
    """Wypisuje wyniki w formie tabeli, gdzie kolumnami są metody."""
    if not results:
        return

    method_order = {"wm": 0, "nit": 1, "sy": 2}
    ordered_results = sorted(
        results,
        key=lambda result: (method_order.get(result.name, 999), result.name),
    )
    metric_rows = [
        ("Liczba regul", lambda result: str(result.rule_count)),
        ("MSE", lambda result: f"{result.mse:.6f}"),
        ("MAE", lambda result: f"{result.mae:.6f}"),
        ("RMSE", lambda result: f"{result.rmse:.6f}"),
        ("R^2", lambda result: f"{result.r_squared:.6f}"),
        ("Czas trenowania [s]", lambda result: f"{result.training_time_seconds:.6f}"),
        (
            "Czas tworzenia regul [s]",
            lambda result: f"{result.rule_creation_time_seconds:.6f}",
        ),
        ("Czas struktury [s]", lambda result: f"{result.structure_time_seconds:.6f}"),
        ("Czas uczenia [s]", lambda result: f"{result.learning_time_seconds:.6f}"),
    ]

    method_labels = [result.name.upper() for result in ordered_results]
    first_col_width = max(
        len("Metryka"),
        max(len(metric_name) for metric_name, _ in metric_rows),
    )
    method_col_width = 20

    header = f"{'Metryka':<{first_col_width}} | " + " | ".join(
        f"{method_label:^{method_col_width}}" for method_label in method_labels
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)
    for metric_name, formatter in metric_rows:
        row_values = [formatter(result) for result in ordered_results]
        row = f"{metric_name:<{first_col_width}} | " + " | ".join(
            f"{value:>{method_col_width}}" for value in row_values
        )
        print(row)
    print(separator)


def print_model_results(result: ModelResult, config: ExperimentConfig) -> None:
    """Wypisuje wyniki pojedynczego modelu dla wszystkich próbek testowych."""
    output_name = config.outputs[0]
    y_true = result.y_true[output_name]
    y_pred = result.predictions[output_name]

    print(f"\n{result.name.upper()}:")
    for idx, (y_true_value, y_pred_value) in enumerate(zip(y_true, y_pred), start=1):
        print(
            f"  Probka {idx}: oczekiwane={y_true_value:.3f}, "
            f"predykcja={y_pred_value:.3f}, "
            f"blad={abs(y_true_value - y_pred_value):.3f}"
        )
    print("  Podsumowanie:")
    print(f"    Liczba regul:        {result.rule_count}")
    print(f"    MSE na testach:      {result.mse:.6f}")
    print(f"    MAE na testach:      {result.mae:.6f}")
    print(f"    RMSE na testach:     {result.rmse:.6f}")
    print(f"    R^2 na testach:      {result.r_squared:.6f}")
    print(f"    Czas trenowania:     {result.training_time_seconds:.6f} s")
    print(f"    Czas tworzenia regul:{result.rule_creation_time_seconds:.6f} s")
    print(f"    Czas struktury:      {result.structure_time_seconds:.6f} s")
    print(f"    Czas uczenia:        {result.learning_time_seconds:.6f} s")


def print_summary(*results: ModelResult) -> None:
    """Wypisuje krótkie podsumowanie porównawcze modeli."""
    print("\n" + "=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)
    print("\nMetryki dla kazdej metody:")
    _print_results_table(*results)
    print("\n" + "=" * 70 + "\n")


def plot_predictions_vs_true(
    *results: ModelResult,
    title: str = "Predykcje modeli: y_pred vs y_true",
    output_path: str | Path | None = None,
) -> None:
    """Wyswietla i opcjonalnie zapisuje wykres y_pred wzgledem y_true."""
    if not results:
        return

    matplotlib_config_dir = os.path.join(
        os.getcwd(),
        "results",
        ".matplotlib",
        str(os.getpid()),
    )
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_config_dir)
    import matplotlib.pyplot as plt

    model_display_names = {
        "wm": "Metoda Wang-Mendel",
        "nit": "Metoda Nozaki-Ishibuchi-Tanaka",
        "sy": "Metoda Sugeno-Yasukawa",
    }
    output_name = next(iter(results[0].y_true.keys()))
    n_results = len(results)
    fig, axes = plt.subplots(
        n_results,
        1,
        figsize=(6, 4.5 * n_results),
        squeeze=False,
    )

    all_true = np.concatenate(
        [np.asarray(result.y_true[output_name], dtype=float).ravel() for result in results]
    )
    all_pred = np.concatenate(
        [
            np.asarray(result.predictions[output_name], dtype=float).ravel()
            for result in results
        ]
    )
    data_min = float(np.nanmin([np.nanmin(all_true), np.nanmin(all_pred)]))
    data_max = float(np.nanmax([np.nanmax(all_true), np.nanmax(all_pred)]))
    margin = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
    axis_min = data_min - margin
    axis_max = data_max + margin

    for axis, result in zip(axes[:, 0], results):
        y_true = np.asarray(result.y_true[output_name], dtype=float).ravel()
        y_pred = np.asarray(result.predictions[output_name], dtype=float).ravel()
        axis.scatter(y_true, y_pred, alpha=0.75, edgecolors="none")
        axis.plot([data_min, data_max], [data_min, data_max], "--", linewidth=1)
        axis.set_title(model_display_names.get(result.name, result.name.upper()))
        axis.set_xlabel("Wartosci rzeczywiste")
        axis.set_ylabel("Wartosci przewidywane")
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Zapisano wykres: {output_path}")

    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()

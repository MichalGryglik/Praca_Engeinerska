"""Uniwersalne funkcje pomocnicze do uruchamiania eksperymentów."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from core.evaluation.metrics import compute_mse
from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from core.rule_generators import sugeno_yasukawa as sy
from core.rule_generators import wang_mendel as wm


DEFAULT_SY_PARAMS = {
    "n_rules": 3,
    "eps_sigma": 1.0,
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
    sy_params: dict[str, Any] | None = None

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
    mse: float


def train_wm(train_data: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    """Trenuje model Wang-Mendel."""
    return wm.generate_rules(
        data=train_data,
        inputs=config.inputs,
        outputs=config.outputs,
        fuzzy_sets=config.fuzzy_sets,
        universes=config.universes,
    )


def train_nit(train_data: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    """Trenuje model Nozaki-Ishibuchi-Tanaka."""
    return nit.generate_rules(
        data=train_data,
        inputs=config.inputs,
        outputs=config.outputs,
        fuzzy_sets=config.fuzzy_sets,
        universes=config.universes,
    )


def train_sy(train_data: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    """Trenuje model Sugeno-Yasukawa."""
    sy_params = config.merged_sy_params()
    centers, _ = sy.initialize_clusters_with_cmeans(
        data=train_data,
        inputs=config.inputs,
        n_rules=sy_params["n_rules"],
    )

    rules_dict = sy.build_initial_rules_from_clusters(
        centers=centers,
        inputs=config.inputs,
        outputs=config.outputs,
        eps_sigma=sy_params["eps_sigma"],
    )

    normalized_strengths = sy.compute_normalized_firing_strengths(
        data=train_data,
        inputs=config.inputs,
        rules_dict=rules_dict,
        fuzzy_sets=config.fuzzy_sets,
        universes=config.universes,
    )

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
    return rules_dict


def evaluate_model(
    model: dict[str, Any],
    model_type: str,
    test_data: pd.DataFrame,
    config: ExperimentConfig,
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
        mse=compute_mse(y_true, predictions),
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
    print(f"  MSE na testach: {result.mse:.6f}")


def print_summary(*results: ModelResult) -> None:
    """Wypisuje krótkie podsumowanie porównawcze modeli."""
    print("\n" + "=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)
    print("\nMSE dla kazdej metody:")
    for result in results:
        print(f"  {result.name.upper():<30} {result.mse:.6f}")
    print("\n" + "=" * 70 + "\n")

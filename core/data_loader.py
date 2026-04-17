"""Ładowanie danych i budowa konfiguracji zbiorów dla eksperymentów."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.membership_functions import create_sets


DEFAULT_TEP_TRAIN_PATH = os.path.join(
    os.path.dirname(__file__), "../data/TEP_FaultFree_Training.csv"
)
DEFAULT_TEP_TEST_PATH = os.path.join(
    os.path.dirname(__file__), "../data/TEP_FaultFree_Testing.csv"
)
DEFAULT_SET_LABELS = ["S2", "S1", "CE", "B1", "B2"]


@dataclass(frozen=True)
class DatasetSpec:
    """Specyfikacja danych i parametrów rozmytych dla eksperymentu."""

    inputs: list[str]
    outputs: list[str]
    fuzzy_sets: dict[str, dict[str, np.ndarray]]
    universes: dict[str, np.ndarray]
    set_params: dict[str, dict[str, list[float]]]


def load_csv_dataset(path: str, n_samples: int | None = None, **read_csv_kwargs) -> pd.DataFrame:
    """Wczytuje dane z CSV i opcjonalnie ogranicza liczbę próbek."""
    try:
        data = pd.read_csv(path, **read_csv_kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found: {path}. See README.md -> Dane (dataset)"
        ) from exc

    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("n_samples musi być dodatnią liczbą całkowitą.")
        return data.head(n_samples).copy()
    return data.copy()


def load_tep_train(path: str | None = None, n_samples: int | None = None) -> pd.DataFrame:
    """Wczytuje zbiór treningowy TEP."""
    return load_csv_dataset(path or DEFAULT_TEP_TRAIN_PATH, n_samples=n_samples)


def load_tep_test(path: str | None = None, n_samples: int | None = None) -> pd.DataFrame:
    """Wczytuje zbiór testowy TEP."""
    return load_csv_dataset(path or DEFAULT_TEP_TEST_PATH, n_samples=n_samples)


def build_triangular_partition(
    min_value: float,
    max_value: float,
    labels: list[str],
) -> dict[str, list[float]]:
    """Buduje równomierny podział zakresu na trójkątne zbiory rozmyte."""
    if not labels:
        raise ValueError("Lista etykiet zbiorów rozmytych nie może być pusta.")

    if max_value < min_value:
        raise ValueError("max_value nie może być mniejsze od min_value.")

    if len(labels) == 1:
        return {labels[0]: [min_value, min_value, max_value]}

    span = max_value - min_value
    if span == 0:
        span = 1e-9

    anchors = np.linspace(min_value, max_value, len(labels) + 1)
    params: dict[str, list[float]] = {}

    for idx, label in enumerate(labels):
        if idx == 0:
            params[label] = [anchors[0], anchors[0], anchors[1]]
        elif idx == len(labels) - 1:
            params[label] = [anchors[-2], anchors[-1], anchors[-1]]
        else:
            params[label] = [anchors[idx], anchors[idx + 1], anchors[idx + 2]]

    return params


def build_dataset_spec_from_data(
    data: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    labels_by_variable: dict[str, list[str]] | None = None,
    universe_resolution: int = 100,
) -> DatasetSpec:
    """Buduje pełną specyfikację eksperymentu na podstawie wybranych kolumn danych."""
    if data.empty:
        raise ValueError("Nie można zbudować specyfikacji dla pustego zbioru danych.")

    variable_names = inputs + outputs
    missing_columns = [column for column in variable_names if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Brak wymaganych kolumn w danych: {missing_columns}")

    labels_by_variable = labels_by_variable or {}
    universes: dict[str, np.ndarray] = {}
    fuzzy_sets: dict[str, dict[str, np.ndarray]] = {}
    set_params: dict[str, dict[str, list[float]]] = {}

    for variable_name in variable_names:
        labels = labels_by_variable.get(variable_name, DEFAULT_SET_LABELS)
        min_value = float(data[variable_name].min())
        max_value = float(data[variable_name].max())

        if min_value == max_value:
            min_value -= 0.5
            max_value += 0.5

        universe = np.linspace(min_value, max_value, universe_resolution)
        variable_params = build_triangular_partition(min_value, max_value, labels)

        universes[variable_name] = universe
        set_params[variable_name] = variable_params
        fuzzy_sets[variable_name] = create_sets(universe, variable_params)

    return DatasetSpec(
        inputs=inputs.copy(),
        outputs=outputs.copy(),
        fuzzy_sets=fuzzy_sets,
        universes=universes,
        set_params=set_params,
    )

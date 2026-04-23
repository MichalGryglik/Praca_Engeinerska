from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioConfig:
    gaussian_noise_std: float = 0.0
    missing_ratio: float = 0.0
    outlier_ratio: float = 0.0
    outlier_magnitude: float = 3.0

    def has_modifications(self) -> bool:
        return (
            self.gaussian_noise_std > 0.0
            or self.missing_ratio > 0.0
            or self.outlier_ratio > 0.0
        )


def apply_training_scenario(
    data: pd.DataFrame,
    scenario: ScenarioConfig | None = None,
    seed: int = 42,
    gaussian_noise_columns: Iterable[str] | None = None,
    missing_columns: Iterable[str] | None = None,
    outlier_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    scenario = scenario or ScenarioConfig()
    rng = np.random.default_rng(seed)

    scenario_data = data.copy()
    scenario_data = inject_gaussian_noise(
        scenario_data,
        std=scenario.gaussian_noise_std,
        columns=gaussian_noise_columns,
        rng=rng,
    )
    scenario_data = inject_missing_values(
        scenario_data,
        missing_ratio=scenario.missing_ratio,
        columns=missing_columns,
        rng=rng,
    )
    scenario_data = inject_outliers(
        scenario_data,
        outlier_ratio=scenario.outlier_ratio,
        outlier_magnitude=scenario.outlier_magnitude,
        columns=outlier_columns,
        rng=rng,
    )
    return scenario_data


def inject_gaussian_noise(
    data: pd.DataFrame,
    std: float,
    columns: Iterable[str] | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if std <= 0.0:
        return data

    noisy_data = data.copy()
    for column in _resolve_numeric_columns(noisy_data, columns):
        noisy_data[column] = noisy_data[column] + rng.normal(
            0.0,
            std,
            size=len(noisy_data),
        )
    return noisy_data


def inject_missing_values(
    data: pd.DataFrame,
    missing_ratio: float,
    columns: Iterable[str] | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if missing_ratio <= 0.0:
        return data

    incomplete_data = data.copy()
    missing_count = _count_affected_rows(len(incomplete_data), missing_ratio)

    for column in _resolve_numeric_columns(incomplete_data, columns):
        missing_idx = rng.choice(
            incomplete_data.index.to_numpy(),
            size=missing_count,
            replace=False,
        )
        incomplete_data.loc[missing_idx, column] = np.nan
    return incomplete_data


def inject_outliers(
    data: pd.DataFrame,
    outlier_ratio: float,
    outlier_magnitude: float,
    columns: Iterable[str] | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if outlier_ratio <= 0.0:
        return data

    outlier_data = data.copy()
    outlier_count = _count_affected_rows(len(outlier_data), outlier_ratio)
    outlier_idx = rng.choice(
        outlier_data.index.to_numpy(),
        size=outlier_count,
        replace=False,
    )

    for column in _resolve_numeric_columns(outlier_data, columns):
        direction = rng.choice([-1.0, 1.0], size=outlier_count)
        outlier_data.loc[outlier_idx, column] = (
            outlier_data.loc[outlier_idx, column].to_numpy(dtype=float)
            + direction * outlier_magnitude
        )
    return outlier_data


def prepare_numeric_training_data(
    data: pd.DataFrame,
    sort_by: str | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    prepared = data.copy()
    if sort_by:
        prepared = prepared.sort_values(sort_by, na_position="last").reset_index(drop=True)

    for column in _resolve_numeric_columns(prepared, columns):
        prepared[column] = prepared[column].interpolate(
            method="linear",
            limit_direction="both",
        )
        prepared[column] = prepared[column].fillna(prepared[column].median())

    if sort_by:
        prepared = prepared.sort_values(sort_by).reset_index(drop=True)
    return prepared


def print_scenario_summary(
    title: str,
    scenario: ScenarioConfig,
    sample_size: int,
    missing_columns: Iterable[str] | None = None,
    outlier_columns: Iterable[str] | None = None,
) -> None:
    missing_column_text = _format_columns(missing_columns)
    outlier_column_text = _format_columns(outlier_columns)

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("Scenariusz danych uczacych:")
    print(f"  Szum gaussowski sigma: {scenario.gaussian_noise_std:.3f}")
    print(
        f"  Braki danych:          {scenario.missing_ratio:.1%} "
        f"(ok. {_count_affected_rows(sample_size, scenario.missing_ratio)} brakow "
        f"w kolumnach: {missing_column_text})"
    )
    print(
        f"  Dane odstajace:        {scenario.outlier_ratio:.1%} "
        f"(modyfikacja o +/-{scenario.outlier_magnitude:.3f} "
        f"w kolumnach: {outlier_column_text})"
    )
    print("  Uwaga: braki sa uzupelniane interpolacja liniowa przed treningiem.")


def _count_affected_rows(sample_size: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    return max(1, int(round(sample_size * ratio)))


def _resolve_numeric_columns(
    data: pd.DataFrame,
    columns: Iterable[str] | None,
) -> list[str]:
    selected_columns = list(columns) if columns is not None else list(data.columns)
    return [
        column
        for column in selected_columns
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column])
    ]


def _format_columns(columns: Iterable[str] | None) -> str:
    if columns is None:
        return "wszystkie numeryczne"
    column_list = list(columns)
    if not column_list:
        return "brak"
    return ", ".join(column_list)

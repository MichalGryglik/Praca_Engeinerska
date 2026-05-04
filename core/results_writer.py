from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_METRICS_PATH = DEFAULT_RESULTS_DIR / "summaries" / "metrics_summary.csv"
SIGNIFICANT_DIGITS_FORMAT = "%.4g"


def _as_1d_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).flatten()


def _resolve_output_names(results: Iterable, output_names: list[str] | None) -> list[str]:
    if output_names is not None:
        return output_names

    results = list(results)
    if not results:
        return []

    return list(results[0].y_true.keys())


def save_metrics_summary(
    experiment_name: str,
    results: Iterable,
    output_path: str | Path = DEFAULT_METRICS_PATH,
    replace_experiment: bool = True,
) -> Path:
    """Save model metrics in a shared CSV summary file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "experiment": experiment_name,
            "model": result.name.upper(),
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
        for result in results
    ]
    summary = pd.DataFrame(rows)

    if replace_experiment and output_path.exists():
        existing_summary = pd.read_csv(output_path)
        existing_summary = existing_summary[
            existing_summary["experiment"] != experiment_name
        ]
        summary = pd.concat([existing_summary, summary], ignore_index=True)

    summary.to_csv(output_path, index=False, float_format=SIGNIFICANT_DIGITS_FORMAT)
    return output_path


def save_predictions(
    results: Iterable,
    output_path: str | Path,
    output_names: list[str] | None = None,
) -> Path:
    """Save per-sample predictions for one or more model results."""
    results = list(results)
    if not results:
        raise ValueError("At least one model result is required to save predictions.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_names = _resolve_output_names(results, output_names)
    data: dict[str, np.ndarray] = {}

    for output_name in output_names:
        actual = _as_1d_array(results[0].y_true[output_name])

        for result in results:
            prediction = _as_1d_array(result.predictions[output_name])
            if len(prediction) != len(actual):
                raise ValueError(
                    f"Prediction length mismatch for {result.name}/{output_name}: "
                    f"{len(prediction)} vs {len(actual)}"
                )

            prediction_column = (
                f"prediction_{result.name.lower()}"
                if len(output_names) == 1
                else f"prediction_{result.name.lower()}_{output_name}"
            )
            data[prediction_column] = prediction

        actual_column = "actual" if len(output_names) == 1 else f"actual_{output_name}"
        data[actual_column] = actual

    predictions = pd.DataFrame(data)
    predictions.to_csv(
        output_path,
        index=False,
        float_format=SIGNIFICANT_DIGITS_FORMAT,
    )
    return output_path

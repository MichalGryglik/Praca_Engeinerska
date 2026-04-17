"""Moduł z uniwersalnymi metrykami oceny dla wszystkich generatorów reguł rozmytych.

Zawiera funkcje do obliczania błędów (MSE, MAE, RMSE, R-squared), dokładności i innych miar
niezależnie od struktury reguł (Wang–Mendel, NIT, Sugeno–Yasukawa).
"""

import numpy as np


def _prepare_metric_values(y_true, y_pred):
    """Normalizuje dane wejściowe do wspólnego formatu dla metryk regresyjnych."""
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        output_names = sorted(y_true.keys())
        prepared_pairs = []

        for output_name in output_names:
            y_t = np.asarray(y_true[output_name], dtype=float).flatten()
            y_p = np.asarray(y_pred[output_name], dtype=float).flatten()

            if len(y_t) != len(y_p):
                raise ValueError(
                    f"Niezgodne długości dla wyjścia '{output_name}': "
                    f"{len(y_t)} vs {len(y_p)}"
                )

            prepared_pairs.append((y_t, y_p))

        return prepared_pairs

    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Niezgodne długości: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    return [(y_true, y_pred)]


def compute_mse(y_true, y_pred):
    """Oblicza Mean Squared Error między rzeczywistymi a przewidywanymi wartościami."""
    prepared_pairs = _prepare_metric_values(y_true, y_pred)
    mse_per_output = [
        float(np.mean(np.square(y_t - y_p))) for y_t, y_p in prepared_pairs
    ]
    return float(np.mean(mse_per_output))


def compute_mae(y_true, y_pred):
    """Oblicza Mean Absolute Error między rzeczywistymi a przewidywanymi wartościami."""
    prepared_pairs = _prepare_metric_values(y_true, y_pred)
    mae_per_output = [
        float(np.mean(np.abs(y_t - y_p))) for y_t, y_p in prepared_pairs
    ]
    return float(np.mean(mae_per_output))


def compute_rmse(y_true, y_pred):
    """Oblicza Root Mean Squared Error między rzeczywistymi a przewidywanymi wartościami."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_r_squared(y_true, y_pred):
    """Oblicza współczynnik determinacji R-squared."""
    prepared_pairs = _prepare_metric_values(y_true, y_pred)
    r2_per_output = []

    for y_t, y_p in prepared_pairs:
        ss_res = float(np.sum(np.square(y_t - y_p)))
        y_mean = float(np.mean(y_t))
        ss_tot = float(np.sum(np.square(y_t - y_mean)))

        if ss_tot == 0.0:
            r2_value = 1.0 if ss_res == 0.0 else 0.0
        else:
            r2_value = 1.0 - (ss_res / ss_tot)

        r2_per_output.append(r2_value)

    return float(np.mean(r2_per_output))

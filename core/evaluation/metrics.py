"""Common regression metrics used by all fuzzy rule generators."""

import numpy as np


def _drop_nan_pairs(y_true, y_pred):
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[valid_mask], y_pred[valid_mask]


def _prepare_metric_values(y_true, y_pred):
    """Convert metric inputs to aligned arrays and drop pairs containing NaN."""
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        output_names = sorted(y_true.keys())
        prepared_pairs = []

        for output_name in output_names:
            y_t = np.asarray(y_true[output_name], dtype=float).flatten()
            y_p = np.asarray(y_pred[output_name], dtype=float).flatten()

            if len(y_t) != len(y_p):
                raise ValueError(
                    f"Niezgodne dlugosci dla wyjscia '{output_name}': "
                    f"{len(y_t)} vs {len(y_p)}"
                )

            prepared_pairs.append(_drop_nan_pairs(y_t, y_p))

        return prepared_pairs

    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Niezgodne dlugosci: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    return [_drop_nan_pairs(y_true, y_pred)]


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def compute_mse(y_true, y_pred):
    """Compute mean squared error, ignoring pairs containing NaN."""
    prepared_pairs = _prepare_metric_values(y_true, y_pred)
    mse_per_output = [
        float(np.mean(np.square(y_t - y_p))) if len(y_t) > 0 else np.nan
        for y_t, y_p in prepared_pairs
    ]
    return _nanmean(mse_per_output)


def compute_mae(y_true, y_pred):
    """Compute mean absolute error, ignoring pairs containing NaN."""
    prepared_pairs = _prepare_metric_values(y_true, y_pred)
    mae_per_output = [
        float(np.mean(np.abs(y_t - y_p))) if len(y_t) > 0 else np.nan
        for y_t, y_p in prepared_pairs
    ]
    return _nanmean(mae_per_output)


def compute_rmse(y_true, y_pred):
    """Compute root mean squared error, ignoring pairs containing NaN."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_r_squared(y_true, y_pred):
    """Compute R-squared, ignoring pairs containing NaN."""
    prepared_pairs = _prepare_metric_values(y_true, y_pred)
    r2_per_output = []

    for y_t, y_p in prepared_pairs:
        if len(y_t) == 0:
            r2_per_output.append(np.nan)
            continue

        ss_res = float(np.sum(np.square(y_t - y_p)))
        y_mean = float(np.mean(y_t))
        ss_tot = float(np.sum(np.square(y_t - y_mean)))

        if ss_tot == 0.0:
            r2_value = 1.0 if ss_res == 0.0 else 0.0
        else:
            r2_value = 1.0 - (ss_res / ss_tot)

        r2_per_output.append(r2_value)

    return _nanmean(r2_per_output)

"""Moduł z uniwersalnymi metrykami oceny dla wszystkich generatorów reguł rozmytych.

Zawiera funkcje do obliczania błędów (MSE), dokładności i innych miar
niezależnie od struktury reguł (Wang–Mendel, NIT, Sugeno–Yasukawa).
"""

import numpy as np


def compute_mse(y_true, y_pred):
    """Oblicza Mean Squared Error między rzeczywistymi a przewidywanymi wartościami.

    Funkcja jest uniwersalna dla wszystkich typów generatorów reguł.

    Parameters
    ----------
    y_true : np.ndarray or dict
        Rzeczywiste wartości wyjściowe.
        - Jeśli np.ndarray: shape (n_samples,) lub (n_samples, n_outputs)
        - Jeśli dict: {output_name: np.ndarray}
    y_pred : np.ndarray or dict
        Przewidywane wartości wyjściowe (taki sam format jak y_true).

    Returns
    -------
    float
        Wartość Mean Squared Error.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.0, 2.9])
    >>> mse = compute_mse(y_true, y_pred)

    >>> y_true = {"y": np.array([1.0, 2.0, 3.0])}
    >>> y_pred = {"y": np.array([1.1, 2.0, 2.9])}
    >>> mse = compute_mse(y_true, y_pred)
    """
    # Obsługa słowników (wiele wyjść)
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        output_names = sorted(y_true.keys())
        mse_per_output = []

        for output_name in output_names:
            y_t = np.asarray(y_true[output_name], dtype=float).flatten()
            y_p = np.asarray(y_pred[output_name], dtype=float).flatten()

            if len(y_t) != len(y_p):
                raise ValueError(
                    f"Niezgodne długości dla wyjścia '{output_name}': "
                    f"{len(y_t)} vs {len(y_p)}"
                )

            mse_per_output.append(float(np.mean(np.square(y_t - y_p))))

        return float(np.mean(mse_per_output))

    # Obsługa tablic numpy
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Niezgodne długości: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    return float(np.mean(np.square(y_true - y_pred)))

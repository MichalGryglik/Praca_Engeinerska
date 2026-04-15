# experiments/example_config.py
# Konfiguracja zestawu 1
import numpy as np
from core.membership_functions import create_sets

# --- Lista zmiennych wejściowych i wyjściowych ---
inputs = ["x1", "x2"]
outputs = ["y"]

# --- Uniwersa ---
x1_universe = np.linspace(0, 10, 100)
x2_universe = np.linspace(0, 10, 100)
y_universe  = np.linspace(0, 20, 100)

# --- Parametry zbiorów rozmytych ---
x1_params = {
    "S1": [0, 0, 5],
    "CE": [0, 5, 10],
    "B1": [5, 10, 10]
}

x2_params = {
    "S2": [0, 0, 2.5],
    "S1": [0, 2.5, 5],
    "CE": [2.5, 5, 7.5],
    "B1": [5, 7.5, 10],
    "B2": [7.5, 10, 10]
}

y_params = {
    "S2": [0, 0, 5],
    "S1": [0, 5, 10],
    "CE": [5, 10, 15],
    "B1": [10, 15, 20],
    "B2": [15, 20, 20]
}

# --- Generacja zbiorów rozmytych ---
x1_sets = create_sets(x1_universe, x1_params)
x2_sets = create_sets(x2_universe, x2_params)
y_sets  = create_sets(y_universe, y_params)

# --- Zbiory i uniwersa w jednym miejscu do łatwego przekazania ---
fuzzy_sets = {
    "x1": x1_sets,
    "x2": x2_sets,
    "y": y_sets
}

universes = {
    "x1": x1_universe,
    "x2": x2_universe,
    "y": y_universe
}

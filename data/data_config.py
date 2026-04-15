# data/data_config.py
# Konfiguracja zestawu 2
import numpy as np
import pandas as pd
import os
from core.membership_functions import create_sets

# Wczytanie danych do obliczenia min/max
data_path = os.path.join(os.path.dirname(__file__), "TEP_FaultFree_Testing.csv")
tep_data = pd.read_csv(data_path)

# Obliczenie min/max dla każdej zmiennej
min_xmeas_1 = tep_data['xmeas_1'].min()
max_xmeas_1 = tep_data['xmeas_1'].max()
min_xmeas_38 = tep_data['xmeas_38'].min()
max_xmeas_38 = tep_data['xmeas_38'].max()
min_xmeas_33 = tep_data['xmeas_33'].min()
max_xmeas_33 = tep_data['xmeas_33'].max()
min_xmeas_27 = tep_data['xmeas_27'].min()
max_xmeas_27 = tep_data['xmeas_27'].max()

# --- Lista zmiennych wejściowych i wyjściowych ---
inputs = ["xmeas_38", "xmeas_33", "xmeas_27"]
outputs = ["xmeas_1"]

# --- Uniwersa ---
xmeas_1_universe = np.linspace(min_xmeas_1, max_xmeas_1, 100)
xmeas_38_universe = np.linspace(min_xmeas_38, max_xmeas_38, 100)
xmeas_33_universe = np.linspace(min_xmeas_33, max_xmeas_33, 100)
xmeas_27_universe = np.linspace(min_xmeas_27, max_xmeas_27, 100)

# --- Parametry zbiorów rozmytych ---
# Dla każdej zmiennej dzielimy zakres na 5 równych części: S2, S1, CE, B1, B2

# xmeas_1 (output)
width_1 = (max_xmeas_1 - min_xmeas_1) / 5
xmeas_1_params = {
    "S2": [min_xmeas_1, min_xmeas_1, min_xmeas_1 + width_1],
    "S1": [min_xmeas_1 + width_1, min_xmeas_1 + 2*width_1, min_xmeas_1 + 3*width_1],
    "CE": [min_xmeas_1 + 2*width_1, min_xmeas_1 + 3*width_1, min_xmeas_1 + 4*width_1],
    "B1": [min_xmeas_1 + 3*width_1, min_xmeas_1 + 4*width_1, min_xmeas_1 + 5*width_1],
    "B2": [min_xmeas_1 + 4*width_1, min_xmeas_1 + 5*width_1, min_xmeas_1 + 5*width_1]
}

# xmeas_38 (input)
width_38 = (max_xmeas_38 - min_xmeas_38) / 5
xmeas_38_params = {
    "S2": [min_xmeas_38, min_xmeas_38, min_xmeas_38 + width_38],
    "S1": [min_xmeas_38 + width_38, min_xmeas_38 + 2*width_38, min_xmeas_38 + 3*width_38],
    "CE": [min_xmeas_38 + 2*width_38, min_xmeas_38 + 3*width_38, min_xmeas_38 + 4*width_38],
    "B1": [min_xmeas_38 + 3*width_38, min_xmeas_38 + 4*width_38, min_xmeas_38 + 5*width_38],
    "B2": [min_xmeas_38 + 4*width_38, min_xmeas_38 + 5*width_38, min_xmeas_38 + 5*width_38]
}

# xmeas_33 (input)
width_33 = (max_xmeas_33 - min_xmeas_33) / 5
xmeas_33_params = {
    "S2": [min_xmeas_33, min_xmeas_33, min_xmeas_33 + width_33],
    "S1": [min_xmeas_33 + width_33, min_xmeas_33 + 2*width_33, min_xmeas_33 + 3*width_33],
    "CE": [min_xmeas_33 + 2*width_33, min_xmeas_33 + 3*width_33, min_xmeas_33 + 4*width_33],
    "B1": [min_xmeas_33 + 3*width_33, min_xmeas_33 + 4*width_33, min_xmeas_33 + 5*width_33],
    "B2": [min_xmeas_33 + 4*width_33, min_xmeas_33 + 5*width_33, min_xmeas_33 + 5*width_33]
}

# xmeas_27 (input)
width_27 = (max_xmeas_27 - min_xmeas_27) / 5
xmeas_27_params = {
    "S2": [min_xmeas_27, min_xmeas_27, min_xmeas_27 + width_27],
    "S1": [min_xmeas_27 + width_27, min_xmeas_27 + 2*width_27, min_xmeas_27 + 3*width_27],
    "CE": [min_xmeas_27 + 2*width_27, min_xmeas_27 + 3*width_27, min_xmeas_27 + 4*width_27],
    "B1": [min_xmeas_27 + 3*width_27, min_xmeas_27 + 4*width_27, min_xmeas_27 + 5*width_27],
    "B2": [min_xmeas_27 + 4*width_27, min_xmeas_27 + 5*width_27, min_xmeas_27 + 5*width_27]
}

# --- Generacja zbiorów rozmytych ---
xmeas_1_sets = create_sets(xmeas_1_universe, xmeas_1_params)
xmeas_38_sets = create_sets(xmeas_38_universe, xmeas_38_params)
xmeas_33_sets = create_sets(xmeas_33_universe, xmeas_33_params)
xmeas_27_sets = create_sets(xmeas_27_universe, xmeas_27_params)

# --- Zbiory i uniwersa w jednym miejscu do łatwego przekazania ---
fuzzy_sets = {
    "xmeas_1": xmeas_1_sets,
    "xmeas_38": xmeas_38_sets,
    "xmeas_33": xmeas_33_sets,
    "xmeas_27": xmeas_27_sets
}

universes = {
    "xmeas_1": xmeas_1_universe,
    "xmeas_38": xmeas_38_universe,
    "xmeas_33": xmeas_33_universe,
    "xmeas_27": xmeas_27_universe
}

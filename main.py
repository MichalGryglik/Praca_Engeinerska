# main.py – punkt startowy programu
from core import membership_functions
from core.rule_generators import wang_mendel as wm
from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from examples import example1_config

print("Framework loaded and modules detected.")

import os
import pandas as pd

# Wczytanie danych
data = pd.read_csv("data/example1_data/data.csv", sep=";")

# Generacja reguł Wang–Mendel
rules_dict = wm.generate_rules(
    data=data,
    inputs=example1_config.inputs,
    outputs=example1_config.outputs,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes
)

# Testowe wartości wejściowe
inputs_values = {"x1": 6.0, "x2": 6.0}

# Obliczenie wyniku na podstawie wygenerowanych reguł
y_pred, activated = wm.apply_rules(
    inputs=inputs_values,
    rules_dict=rules_dict,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes
)

print(f"Przewidywana wartość y: {y_pred:.3f}")
print("Aktywne reguły:")
for y_label, w in activated:
    print(f"THEN y is {y_label} [siła aktywacji = {w:.3f}]")
    
wm.pretty_print_rules(rules_dict, example1_config.inputs)

from core.rule_generators import nozaki_ishibuchi_tanaka as nit

# Generacja reguł NIT
nit_rules = nit.generate_rules(
    data=data,
    inputs=example1_config.inputs,
    outputs=example1_config.outputs,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes
)

# Wnioskowanie NIT
y_pred_nit, activated_nit = nit.apply_rules(
    inputs=inputs_values,
    rules_dict=nit_rules,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes
)

print(f"NIT: przewidywana wartość y = {y_pred_nit:.3f}")
print(f"Liczba aktywnych reguł NIT: {len(activated_nit)}")
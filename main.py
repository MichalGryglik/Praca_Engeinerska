# main.py – punkt startowy programu
import importlib.util
from pathlib import Path

import pandas as pd

from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from core.rule_generators import wang_mendel as wm
from examples import example1_config


print("Framework loaded and modules detected.")


# Wczytanie danych uczących z przykładowego zbioru
# Dane będą użyte zarówno przez istniejące metody, jak i przez szkic Sugeno–Yasukawa.
data = pd.read_csv("data/example1_data/data.csv", sep=";")

# ---------------------------------------------------------------------
# CZĘŚĆ 1: Wang–Mendel
# ---------------------------------------------------------------------
# Generujemy bazę reguł metodą Wang–Mendel.
rules_dict = wm.generate_rules(
    data=data,
    inputs=example1_config.inputs,
    outputs=example1_config.outputs,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes,
)

# Definiujemy przykładowy wektor wejściowy do testowego wnioskowania.
inputs_values = {"x1": 6.0, "x2": 6.0}

# Uruchamiamy wnioskowanie na wygenerowanej bazie reguł.
y_pred, activated = wm.apply_rules(
    inputs=inputs_values,
    rules_dict=rules_dict,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes,
)

print(f"Przewidywana wartość y: {y_pred:.3f}")
print("Aktywne reguły:")
for y_label, w in activated:
    print(f"THEN y is {y_label} [siła aktywacji = {w:.3f}]")

wm.pretty_print_rules(rules_dict, example1_config.inputs)

# ---------------------------------------------------------------------
# CZĘŚĆ 2: Nozaki–Ishibuchi–Tanaka
# ---------------------------------------------------------------------
# Generujemy reguły metodą NIT na tym samym zbiorze danych.
nit_rules = nit.generate_rules(
    data=data,
    inputs=example1_config.inputs,
    outputs=example1_config.outputs,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes,
)

# Sprawdzamy wynik wnioskowania dla tej samej próbki testowej.
y_pred_nit, activated_nit = nit.apply_rules(
    inputs=inputs_values,
    rules_dict=nit_rules,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes,
)

print(f"NIT: przewidywana wartość y = {y_pred_nit:.3f}")
print(f"Liczba aktywnych reguł NIT: {len(activated_nit)}")

# ---------------------------------------------------------------------
# CZĘŚĆ 3: Sugeno–Yasukawa – uruchomienie przygotowanych funkcji
# ---------------------------------------------------------------------
# Plik ma w nazwie myślnik, więc ładujemy go dynamicznie przez importlib.
metoda_3_path = Path(__file__).with_name("metoda-3.py")
spec = importlib.util.spec_from_file_location("metoda_3_module", metoda_3_path)
metoda_3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metoda_3)

# Ustalamy podstawowe parametry testowego uruchomienia algorytmu.
n_rules = 3
eps_sigma = 0.1

# Inicjalizujemy klastry metodą c-means, aby uzyskać środki reguł startowych.
centers, membership_matrix = metoda_3.initialize_clusters_with_cmeans(
    data=data,
    inputs=example1_config.inputs,
    n_rules=n_rules,
)

# Budujemy początkową bazę reguł na podstawie środków klastrów.
sugeno_rules = metoda_3.build_initial_rules_from_clusters(
    centers=centers,
    inputs=example1_config.inputs,
    outputs=example1_config.outputs,
    eps_sigma=eps_sigma,
)

# Obliczamy surowe i znormalizowane siły aktywacji dla wszystkich próbek.
normalized_strengths_result = metoda_3.compute_normalized_firing_strengths(
    data=data,
    inputs=example1_config.inputs,
    rules_dict=sugeno_rules,
    fuzzy_sets=example1_config.fuzzy_sets,
    universes=example1_config.universes,
)

# Rozpakowujemy dane do zmiennych pomocniczych, aby ich wydruk był czytelniejszy.
rule_ids = normalized_strengths_result["rule_ids"]
normalized_strengths = normalized_strengths_result["normalized"]
w_norm = normalized_strengths_result["normalized"]
raw_strengths = normalized_strengths_result["raw"]

print("\n=== SUGENO–YASUKAWA: DEBUG START ===")
print("Centers:")
print(centers)
print("\nMembership matrix z c-means:")
print(membership_matrix)
print("\nInitial rules:")
print(sugeno_rules)
print("\nRaw strengths:")
print(raw_strengths)
print("\nZnormalizowane siły aktywacji:")
print(normalized_strengths)
print("\nIdentyfikatory reguł:")
print(rule_ids)
print("\nMacierz w_norm:")
print(w_norm)
print("\nSuma wag w każdym wierszu (powinna być bliska 1):")
print(w_norm.sum(axis=1))
print("\nLiczba reguł Sugeno:")
print(len(sugeno_rules))
print("Liczba próbek w danych:")
print(len(data))
print("=== SUGENO–YASUKAWA: DEBUG END ===")

# main.py – punkt startowy programu
import pandas as pd
import numpy as np

from core.rule_generators import Sugeno_Yasukawa as sy
from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from core.rule_generators import wang_mendel as wm
from core.evaluation.metrics import compute_mse
from experiments import example_config
from data import data_config


print("Framework loaded and modules detected.")


# Wczytanie danych uczących z przykładowego zbioru
# Dane będą użyte zarówno przez istniejące metody, jak i przez szkic Sugeno–Yasukawa.
data = pd.read_csv("experiments/data.csv", sep=";")

# Definiujemy przykładowy wektor wejściowy do testowego wnioskowania.
inputs_values = {"x1": 6.0, "x2": 6.0}

# ---------------------------------------------------------------------
# METODA 1: Wang–Mendel
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("METODA 1: WANG–MENDEL")
print("="*70)

rules_dict = wm.generate_rules(
    data=data,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes,
)

print("\nReguły Wang–Mendel:")
wm.pretty_print_rules(rules_dict, example_config.inputs)

# Przewidywanie dla przykładowego wektora wejścia
print("\nWyniki Wang–Mendel dla pojedynczej próbki:")
y_pred_wm_single, activated_wm = wm.apply_rules(
    inputs=inputs_values,
    rules_dict=rules_dict,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes,
    outputs=example_config.outputs,
)
print(f"  Przewidywana wartość y = {y_pred_wm_single:.3f} dla x1={inputs_values['x1']}, x2={inputs_values['x2']}")
print("  Aktywne reguły:")
for y_label, w in activated_wm:
    print(f"    THEN y is {y_label} [siła aktywacji = {w:.3f}]")

# ---------------------------------------------------------------------
# METODA 2: Nozaki–Ishibuchi–Tanaka
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("METODA 2: NOZAKI–ISHIBUCHI–TANAKA")
print("="*70)

nit_rules = nit.generate_rules(
    data=data,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes,
)

print("\nReguły Nozaki–Ishibuchi–Tanaka:")
nit.pretty_print_rules(nit_rules, example_config.inputs)

print("\nWyniki NIT dla pojedynczej próbki:")
y_pred_nit_single, activated_nit = nit.apply_rules(
    inputs=inputs_values,
    rules_dict=nit_rules,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes,
    outputs=example_config.outputs,
)
print(f"  Przewidywana wartość y = {y_pred_nit_single:.3f} dla x1={inputs_values['x1']}, x2={inputs_values['x2']}")
print(f"  Liczba aktywnych reguł: {len(activated_nit)}")

# ---------------------------------------------------------------------
# METODA 3: Sugeno–Yasukawa
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("METODA 3: SUGENO–YASUKAWA")
print("="*70)

n_rules = 3
eps_sigma = 1.0

centers, membership_matrix = sy.initialize_clusters_with_cmeans(
    data=data,
    inputs=example_config.inputs,
    n_rules=n_rules,
)

sugeno_rules = sy.build_initial_rules_from_clusters(
    centers=centers,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    eps_sigma=eps_sigma,
)

print("\nReguły początkowe Sugeno–Yasukawa:")
sy.print_rules(sugeno_rules)

normalized_strengths_result = sy.compute_normalized_firing_strengths(
    data=data,
    inputs=example_config.inputs,
    rules_dict=sugeno_rules,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes,
)

print("\nAktualizacja reguł Sugeno–Yasukawa:")
sy.update_consequents_ls_wls(
    data=data,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=sugeno_rules,
    normalized_strengths=normalized_strengths_result,
)
sy.update_antecedents(
    data=data,
    inputs=example_config.inputs,
    rules_dict=sugeno_rules,
    normalized_strengths=normalized_strengths_result,
    eps_sigma=eps_sigma,
)

print("\nReguły końcowe Sugeno–Yasukawa:")
sy.print_rules(sugeno_rules)

print("\nSugeno–Yasukawa: przewidywanie dla pojedynczej próbki:")
single_sample = pd.DataFrame([inputs_values])
y_pred_sy_single = sy.predict(
    data=single_sample,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=sugeno_rules,
)
y_pred_sy_single_value = y_pred_sy_single[example_config.outputs[0]][0]
print(f"  Przewidywana wartość y = {y_pred_sy_single_value:.3f} dla x1={inputs_values['x1']}, x2={inputs_values['x2']}")

# =====================================================================
# CZĘŚĆ 4: Obliczenie predykcji i MSE dla wszystkich generatorów
# =====================================================================
print("\n\n" + "="*70)
print("CZĘŚĆ 4: PORÓWNANIE PREDYKCJI I MSE")
print("="*70)

# Pobierz prawdziwe wartości wyjścia z całego zbioru danych
y_true = data[example_config.outputs[0]].to_numpy(dtype=float)

# --- Wang–Mendel ---
print("\n1. WANG–MENDEL")
print("-" * 50)
y_pred_wm = wm.predict(
    data=data,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=rules_dict,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes
)
y_pred_wm_values = y_pred_wm[example_config.outputs[0]]
mse_wm = compute_mse(y_true, y_pred_wm_values)
print(f"MSE: {mse_wm:.6f}")

# Dla inputs_values
y_pred_wm_single, _ = wm.apply_rules(
    inputs=inputs_values,
    rules_dict=rules_dict,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes,
    outputs=example_config.outputs,
)
print(f"Predykcja dla x1={inputs_values['x1']}, x2={inputs_values['x2']}: y={y_pred_wm_single:.3f}")

# --- Nozaki–Ishibuchi–Tanaka ---
print("\n2. NOZAKI–ISHIBUCHI–TANAKA")
print("-" * 50)
y_pred_nit = nit.predict(
    data=data,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=nit_rules,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes
)
y_pred_nit_values = y_pred_nit[example_config.outputs[0]]
mse_nit = compute_mse(y_true, y_pred_nit_values)
print(f"MSE: {mse_nit:.6f}")

# Dla inputs_values
y_pred_nit_single, _ = nit.apply_rules(
    inputs=inputs_values,
    rules_dict=nit_rules,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes,
    outputs=example_config.outputs,
)
print(f"Predykcja dla x1={inputs_values['x1']}, x2={inputs_values['x2']}: y={y_pred_nit_single:.3f}")

# --- Sugeno–Yasukawa ---
print("\n3. SUGENO–YASUKAWA")
print("-" * 50)
y_pred_sy = sy.predict(
    data=data,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=sugeno_rules
)
y_pred_sy_values = y_pred_sy[example_config.outputs[0]]
mse_sy = compute_mse(y_true, y_pred_sy_values)
print(f"MSE: {mse_sy:.6f}")

# Dla inputs_values (zamiast pojedynczej próbki, tworzymy mini-DataFrame)
single_sample = pd.DataFrame([inputs_values])
y_pred_sy_single = sy.predict(
    data=single_sample,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=sugeno_rules
)
y_pred_sy_single_value = y_pred_sy_single[example_config.outputs[0]][0]
print(f"Predykcja dla x1={inputs_values['x1']}, x2={inputs_values['x2']}: y={y_pred_sy_single_value:.3f}")

# =====================================================================
# CZĘŚĆ 5: Testowanie na nowych próbkach (poza zbiorem treningowym)
# =====================================================================
print("\n\n" + "="*70)
print("CZĘŚĆ 5: TESTOWANIE NA NOWYCH PRÓBKACH")
print("="*70)

# Tworzymy 3 próbki testowe (x1, x2, y=x1+x2)
test_samples = pd.DataFrame([
    {"x1": 2.0, "x2": 3.0},     # y=5.0
    {"x1": 7.5, "x2": 1.5},     # y=9.0
    {"x1": 4.0, "x2": 8.0},     # y=12.0
])

test_y_true = test_samples[["x1", "x2"]].sum(axis=1).to_numpy()
test_y_true_dict = {example_config.outputs[0]: test_y_true}

print("\nPróbki testowe:")
for idx, row in test_samples.iterrows():
    expected_y = row['x1'] + row['x2']
    print(f"  Próbka {idx+1}: x1={row['x1']:.1f}, x2={row['x2']:.1f} -> oczekiwane y={expected_y:.1f}")

# Predykcje dla każdej metody
print("\n" + "-"*70)
print("PREDYKCJE NA NOWYCH PRÓBKACH:")
print("-"*70)

# Wang-Mendel
y_pred_wm_test = wm.predict(
    data=test_samples,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=rules_dict,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes
)
y_pred_wm_test_values = y_pred_wm_test[example_config.outputs[0]]
mse_wm_test = compute_mse(test_y_true, y_pred_wm_test_values)

print("\nWang-Mendel:")
for idx, (y_true, y_pred) in enumerate(zip(test_y_true, y_pred_wm_test_values)):
    print(f"  Próbka {idx+1}: oczekiwane={y_true:.1f}, predykcja={y_pred:.3f}, błąd={abs(y_true-y_pred):.3f}")
print(f"  MSE na testach: {mse_wm_test:.6f}")

# Nozaki-Ishibuchi-Tanaka
y_pred_nit_test = nit.predict(
    data=test_samples,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=nit_rules,
    fuzzy_sets=example_config.fuzzy_sets,
    universes=example_config.universes
)
y_pred_nit_test_values = y_pred_nit_test[example_config.outputs[0]]
mse_nit_test = compute_mse(test_y_true, y_pred_nit_test_values)

print("\nNozaki-Ishibuchi-Tanaka:")
for idx, (y_true, y_pred) in enumerate(zip(test_y_true, y_pred_nit_test_values)):
    print(f"  Próbka {idx+1}: oczekiwane={y_true:.1f}, predykcja={y_pred:.3f}, błąd={abs(y_true-y_pred):.3f}")
print(f"  MSE na testach: {mse_nit_test:.6f}")

# Sugeno-Yasukawa
y_pred_sy_test = sy.predict(
    data=test_samples,
    inputs=example_config.inputs,
    outputs=example_config.outputs,
    rules_dict=sugeno_rules
)
y_pred_sy_test_values = y_pred_sy_test[example_config.outputs[0]]
mse_sy_test = compute_mse(test_y_true, y_pred_sy_test_values)

print("\nSugeno-Yasukawa:")
for idx, (y_true, y_pred) in enumerate(zip(test_y_true, y_pred_sy_test_values)):
    print(f"  Próbka {idx+1}: oczekiwane={y_true:.1f}, predykcja={y_pred:.3f}, błąd={abs(y_true-y_pred):.3f}")
print(f"  MSE na testach: {mse_sy_test:.6f}")

# --- Podsumowanie ---
print("\n" + "="*70)
print("PODSUMOWANIE")
print("="*70)
print(f"\nMSE dla każdej metody na całym zbiorze danych:")
print(f"  Wang–Mendel:                   {mse_wm:.6f}")
print(f"  Nozaki–Ishibuchi–Tanaka:       {mse_nit:.6f}")
print(f"  Sugeno–Yasukawa:               {mse_sy:.6f}")

print(f"\nPredykcje dla wektora wejścia x1={inputs_values['x1']}, x2={inputs_values['x2']}:")
print("  Oczekiwane: y ~ 12.0")
print(f"  Wang–Mendel:                   y = {y_pred_wm_single:.3f}")
print(f"  Nozaki–Ishibuchi–Tanaka:       y = {y_pred_nit_single:.3f}")
print(f"  Sugeno–Yasukawa:               y = {y_pred_sy_single_value:.3f}")

print("\n" + "="*70 + "\n")

# =====================================================================
# EXAMPLE 2: TEP DATA
# =====================================================================
print("\n" + "="*70)
print("EXAMPLE 2: TEP DATA")
print("="*70)

# Wczytanie danych treningowych: pierwsze 1000 próbek z TEP_FaultFree_Training.csv
tep_train_path = "data/TEP_FaultFree_Training.csv"
tep_test_path = "data/TEP_FaultFree_Testing.csv"

try:
    tep_train = pd.read_csv(tep_train_path).head(1000)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Dataset not found: {tep_train_path}. See README.md -> Dane (dataset)"
    ) from exc

# Generowanie reguł dla każdej metody

# Wang-Mendel
rules_dict_tep = wm.generate_rules(
    data=tep_train,
    inputs=data_config.inputs,
    outputs=data_config.outputs,
    fuzzy_sets=data_config.fuzzy_sets,
    universes=data_config.universes,
)

# Nozaki-Ishibuchi-Tanaka
nit_rules_tep = nit.generate_rules(
    data=tep_train,
    inputs=data_config.inputs,
    outputs=data_config.outputs,
    fuzzy_sets=data_config.fuzzy_sets,
    universes=data_config.universes,
)

# Sugeno-Yasukawa
n_rules = 3
eps_sigma = 1.0

centers, membership_matrix = sy.initialize_clusters_with_cmeans(
    data=tep_train,
    inputs=data_config.inputs,
    n_rules=n_rules,
)

sugeno_rules_tep = sy.build_initial_rules_from_clusters(
    centers=centers,
    inputs=data_config.inputs,
    outputs=data_config.outputs,
    eps_sigma=eps_sigma,
)

normalized_strengths_result = sy.compute_normalized_firing_strengths(
    data=tep_train,
    inputs=data_config.inputs,
    rules_dict=sugeno_rules_tep,
    fuzzy_sets=data_config.fuzzy_sets,
    universes=data_config.universes,
)

sy.update_consequents_ls_wls(
    data=tep_train,
    inputs=data_config.inputs,
    outputs=data_config.outputs,
    rules_dict=sugeno_rules_tep,
    normalized_strengths=normalized_strengths_result,
)

sy.update_antecedents(
    data=tep_train,
    inputs=data_config.inputs,
    rules_dict=sugeno_rules_tep,
    normalized_strengths=normalized_strengths_result,
    eps_sigma=eps_sigma,
)

# Teraz predykcje na nowych próbkach: pierwsze 3 z TEP_FaultFree_Testing.csv
try:
    test_samples_tep = pd.read_csv(tep_test_path).head(3)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Dataset not found: {tep_test_path}. See README.md -> Dane (dataset)"
    ) from exc
test_y_true_tep = test_samples_tep[data_config.outputs[0]].to_numpy(dtype=float)

print("\nPróbki testowe TEP:")
for idx, row in test_samples_tep.iterrows():
    print(f"  Próbka {idx+1}: xmeas_38={row['xmeas_38']:.3f}, xmeas_33={row['xmeas_33']:.3f}, xmeas_27={row['xmeas_27']:.3f} -> oczekiwane xmeas_1={row['xmeas_1']:.3f}")

print("\n" + "-"*70)
print("PREDYKCJE NA NOWYCH PRÓBKACH TEP:")
print("-"*70)

# Wang-Mendel
y_pred_wm_tep = wm.predict(
    data=test_samples_tep,
    inputs=data_config.inputs,
    outputs=data_config.outputs,
    rules_dict=rules_dict_tep,
    fuzzy_sets=data_config.fuzzy_sets,
    universes=data_config.universes
)
y_pred_wm_tep_values = y_pred_wm_tep[data_config.outputs[0]]
mse_wm_tep = compute_mse(test_y_true_tep, y_pred_wm_tep_values)

print("\nWang-Mendel:")
for idx, (y_true, y_pred) in enumerate(zip(test_y_true_tep, y_pred_wm_tep_values)):
    print(f"  Próbka {idx+1}: oczekiwane={y_true:.3f}, predykcja={y_pred:.3f}, błąd={abs(y_true-y_pred):.3f}")
print(f"  MSE na testach: {mse_wm_tep:.6f}")

# Nozaki-Ishibuchi-Tanaka
y_pred_nit_tep = nit.predict(
    data=test_samples_tep,
    inputs=data_config.inputs,
    outputs=data_config.outputs,
    rules_dict=nit_rules_tep,
    fuzzy_sets=data_config.fuzzy_sets,
    universes=data_config.universes
)
y_pred_nit_tep_values = y_pred_nit_tep[data_config.outputs[0]]
mse_nit_tep = compute_mse(test_y_true_tep, y_pred_nit_tep_values)

print("\nNozaki-Ishibuchi-Tanaka:")
for idx, (y_true, y_pred) in enumerate(zip(test_y_true_tep, y_pred_nit_tep_values)):
    print(f"  Próbka {idx+1}: oczekiwane={y_true:.3f}, predykcja={y_pred:.3f}, błąd={abs(y_true-y_pred):.3f}")
print(f"  MSE na testach: {mse_nit_tep:.6f}")

# Sugeno-Yasukawa
y_pred_sy_tep = sy.predict(
    data=test_samples_tep,
    inputs=data_config.inputs,
    outputs=data_config.outputs,
    rules_dict=sugeno_rules_tep
)
y_pred_sy_tep_values = y_pred_sy_tep[data_config.outputs[0]]
mse_sy_tep = compute_mse(test_y_true_tep, y_pred_sy_tep_values)

print("\nSugeno-Yasukawa:")
for idx, (y_true, y_pred) in enumerate(zip(test_y_true_tep, y_pred_sy_tep_values)):
    print(f"  Próbka {idx+1}: oczekiwane={y_true:.3f}, predykcja={y_pred:.3f}, błąd={abs(y_true-y_pred):.3f}")
print(f"  MSE na testach: {mse_sy_tep:.6f}")

# --- Podsumowanie TEP ---
print("\n" + "="*70)
print("PODSUMOWANIE TEP")
print("="*70)
print(f"\nMSE dla każdej metody na próbkach testowych TEP:")
print(f"  Wang–Mendel:                   {mse_wm_tep:.6f}")
print(f"  Nozaki–Ishibuchi–Tanaka:       {mse_nit_tep:.6f}")
print(f"  Sugeno–Yasukawa:               {mse_sy_tep:.6f}")

print("\n" + "="*70 + "\n")

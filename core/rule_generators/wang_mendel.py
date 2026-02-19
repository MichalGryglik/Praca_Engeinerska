# core/rule_generators/wang_mendel.py
# Implementacja algorytmu Wang–Mendel
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from core.membership_functions import find_best_membership

def generate_rules(data, inputs, outputs, fuzzy_sets, universes):
    rules_dict = {}

    for _, row in data.iterrows():
        # Oblicz przynależności wszystkich wejść
        labels = {}
        strengths = []
        for inp in inputs:
            lbl, mu = find_best_membership(row[inp], universes[inp], fuzzy_sets[inp])
            labels[inp] = lbl
            strengths.append(mu)

        firing_strength = np.prod(strengths)  # mnożymy wszystkie siły wejść

        # Przypisujemy wyjścia (dla prostoty zakładamy 1 wyjście)
        out = outputs[0]
        out_lbl, _ = find_best_membership(row[out], universes[out], fuzzy_sets[out])

        key = tuple(labels[inp] for inp in inputs)

        if key not in rules_dict or firing_strength > rules_dict[key][1]:
            rules_dict[key] = (out_lbl, firing_strength)

    return rules_dict


def apply_rules(inputs, rules_dict, fuzzy_sets, universes):
    """
    inputs: dict, np. {"x1": 7.0, "x2": 6.0}
    rules_dict: wygenerowane reguły Wang-Mendel
    fuzzy_sets: wszystkie zbiory rozmyte w formie {'x1': {...}, 'x2': {...}, 'y': {...}}
    universes: uniwersa dla wszystkich zmiennych {'x1': ..., 'x2': ..., 'y': ...}

    Zwraca:
        y_pred: przewidywana wartość wyjścia
        activated: lista aktywnych reguł (y_label, siła)
    """

    # --- Przynależności wejść ---
    memberships = {}
    for inp_name, inp_val in inputs.items():
        memberships[inp_name] = {name: fuzz.interp_membership(universes[inp_name], mf, inp_val)
                                 for name, mf in fuzzy_sets[inp_name].items()}

    # --- Aktywacja reguł ---
    activated = []
    for condition, (y_label, rule_strength) in rules_dict.items():
        # condition to krotka etykiet wszystkich wejść
        mu_list = []
        for idx, inp_name in enumerate(inputs.keys()):
            label_in_condition = condition[idx]
            mu_list.append(memberships[inp_name].get(label_in_condition, 0))
        
        firing_strength = np.prod(mu_list)
        if firing_strength > 0:
            activated.append((y_label, firing_strength))

    if not activated:
        y_pred = np.nan
    else:
        # --- Agregacja wyników ---
        numerator = 0
        denominator = 0
        for y_label, w in activated:
            centroid = fuzz.defuzz(universes['y'], fuzzy_sets['y'][y_label], 'centroid')
            numerator += w * centroid
            denominator += w
        y_pred = numerator / denominator

    return y_pred, activated

def pretty_print_rules(rules, inputs):
    """
    Czytelny wypis reguł Wang–Mendel.
    Parametr:
        rules – dict w formacie:
            {
                (x1_label, x2_label, ...): y_label,
                ...
            }
    """
    print("\n=== REGUŁY WANG–MENDEL ===\n")

    if not rules:
        print("Brak reguł do wyświetlenia.")
        return

    for i, (input_labels, output_info) in enumerate(rules.items(), start=1):
        # output_info ma postać (label, weight)
        if isinstance(output_info, tuple):
            y_label, weight = output_info
        else:
            # fallback — gdyby algorytm zwrócił tylko label
            y_label, weight = output_info, None

        inputs_text = " AND ".join([
            f"{inp} IS {lbl}" for inp, lbl in zip(inputs, input_labels)
        ])

        if weight is not None:
            weight_str = f"{float(weight):.6f}"
            print(f"Rule {i}: IF {inputs_text} THEN y IS {y_label} [waga = {weight_str}]")
        else:
            print(f"Rule {i}: IF {inputs_text} THEN y IS {y_label}")

    print("\n=== KONIEC LISTY REGUŁ ===\n")
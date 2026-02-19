# nozaki_ishibuchi_tanaka

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from core.membership_functions import find_best_membership

def generate_rules(data, inputs, outputs, fuzzy_sets, universes):
    """
    Generuje bazę reguł rozmytych metodą Nozaki–Ishibuchi–Tanaka.

    Parameters
    ----------
    data : pandas.DataFrame
        Dane uczące
    inputs : list[str]
        Lista nazw zmiennych wejściowych
    outputs : list[str]
        Lista nazw zmiennych wyjściowych (zakładamy 1)
    fuzzy_sets : dict
        {'x1': {...}, 'x2': {...}, 'y': {...}}
    universes : dict
        {'x1': np.array, 'x2': np.array, 'y': np.array}

    Returns
    -------
    dict
        Baza reguł:
        key   -> krotka etykiet wejść
        value -> {
            'consequent': dict[str, float],
            'weight': float
        }
    """

    # --- Inicjalizacja pustej bazy reguł ---
    rule_base = {}

    output_var = outputs[0]  # zakładamy jedno wyjście

    # ==========================================================
    # ITERACJA PO WSZYSTKICH PUNKTACH DANYCH
    # ==========================================================
    for _, row in data.iterrows():

        antecedent_labels = []
        antecedent_memberships = []

        # ------------------------------------------------------
        # 1. WYZNACZENIE WARUNKU REGUŁY
        # ------------------------------------------------------
        for inp in inputs:
            value = row[inp]

            label, mu = find_best_membership(
                value,
                universes[inp],
                fuzzy_sets[inp]
            )

            antecedent_labels.append(label)
            antecedent_memberships.append(mu)

        # ------------------------------------------------------
        # 2. SIŁA AKTYWACJI REGUŁY (iloczyn przynależności)
        # ------------------------------------------------------
        firing_strength = np.prod(antecedent_memberships)

        # Jeżeli reguła w ogóle się nie aktywuje — pomijamy
        if firing_strength == 0:
            continue

        # ------------------------------------------------------
        # 3. KONSEKWENT ROZMYTY (RÓWNANIE 7)
        # ------------------------------------------------------
        y_value = row[output_var]

        consequent_vector = {}

        for label, mf in fuzzy_sets[output_var].items():
            mu_y = fuzz.interp_membership(
                universes[output_var],
                mf,
                y_value
            )
            consequent_vector[label] = mu_y

        # ------------------------------------------------------
        # 4. KLUCZ REGUŁY (warunek)
        # ------------------------------------------------------
        rule_key = tuple(antecedent_labels)

        # ------------------------------------------------------
        # 5. AGREGACJA REGUŁ O TYM SAMYM WARUNKU
        # ------------------------------------------------------
        if rule_key not in rule_base:
            # pierwsze wystąpienie tej reguły
            rule_base[rule_key] = {
                "sum_vector": {
                    lbl: firing_strength * mu
                    for lbl, mu in consequent_vector.items()
                },
                "sum_weight": firing_strength
            }
        else:
            # kolejne wystąpienie – sumowanie
            for lbl in consequent_vector:
                rule_base[rule_key]["sum_vector"][lbl] += (
                    firing_strength * consequent_vector[lbl]
                )
            rule_base[rule_key]["sum_weight"] += firing_strength

    # ==========================================================
    # 6. NORMALIZACJA KONSEKWENTÓW (ŚREDNIA WAŻONA)
    # ==========================================================
    final_rules = {}

    for rule_key, data_rule in rule_base.items():
        sum_vector = data_rule["sum_vector"]
        sum_weight = data_rule["sum_weight"]

        normalized_consequent = {
            lbl: sum_vector[lbl] / sum_weight
            for lbl in sum_vector
        }

        final_rules[rule_key] = {
            "consequent": normalized_consequent,
            "weight": sum_weight
        }

    return final_rules

 


def pretty_print_rules(rules, inputs, threshold=0.0):
    print("\n=== BAZA REGUŁ (Nozaki–Ishibuchi–Tanaka) ===")
    
    if not rules:
        print("Brak reguł do wyświetlenia.")
        return

    for i, (input_labels, rule_data) in enumerate(rules.items(), start=1):
        inputs_text = " AND ".join(
        f"{inp} IS {lbl}"
        for inp, lbl in zip(inputs, input_labels)
        )


        consequent_text = ", ".join(
        f"{lbl}:{mu:.3f}"
        for lbl, mu in rule_data["consequent"].items()
        if mu > threshold
        )


        print(
        f"Rule {i}: IF {inputs_text} "
        f"THEN y IS {{{consequent_text}}} "
        f"[waga = {rule_data['weight']:.6f}]"
        )

    print("=== KONIEC LISTY REGUŁ ===\n")




def apply_rules(inputs, rules_dict, fuzzy_sets, universes, output_var="y"):
    # 1. Fuzyfikacja wejść
    memberships = {}
    for inp_name, inp_val in inputs.items():
        memberships[inp_name] = {
            name: fuzz.interp_membership(
                universes[inp_name], mf, inp_val
            )
            for name, mf in fuzzy_sets[inp_name].items()
        }

    # 2. Inicjalizacja agregacji wyjścia
    aggregated_output = np.zeros_like(universes[output_var])
    activated_rules = []

    # 3. Przetwarzanie reguł
    for condition, rule_data in rules_dict.items():
        # firing strength
        mu_list = []
        for idx, inp_name in enumerate(inputs.keys()):
            mu_list.append(
                memberships[inp_name].get(condition[idx], 0.0)
            )

        firing_strength = np.prod(mu_list)
        if firing_strength == 0:
            continue

        activated_rules.append((condition, firing_strength))

        # skalowanie i agregacja konsekwentu
        for y_label, coeff in rule_data["consequent"].items():
            aggregated_output += (
                firing_strength
                * coeff
                * fuzzy_sets[output_var][y_label]
            )

    if not np.any(aggregated_output):
        return np.nan, []

    # 4. Defuzyfikacja
    y_pred = fuzz.defuzz(
        universes[output_var],
        aggregated_output,
        "centroid"
    )

    return y_pred, activated_rules
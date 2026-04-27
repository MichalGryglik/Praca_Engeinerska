# nozaki_ishibuchi_tanaka

import numpy as np
import skfuzzy as fuzz

from core.membership_functions import clip_to_universe, find_best_membership

DEFAULT_ALPHA = 1.0


def _compute_input_memberships(row, inputs, fuzzy_sets, universes):
    antecedent_labels = []
    antecedent_memberships = []

    for inp in inputs:
        value = row[inp]
        label, mu = find_best_membership(
            value,
            universes[inp],
            fuzzy_sets[inp],
        )
        antecedent_labels.append(label)
        antecedent_memberships.append(mu)

    return tuple(antecedent_labels), np.asarray(antecedent_memberships, dtype=float)


def _compute_output_memberships(value, output_var, fuzzy_sets, universes):
    bounded_value = clip_to_universe(value, universes[output_var])
    memberships = {}
    for label, mf in fuzzy_sets[output_var].items():
        memberships[label] = float(
            fuzz.interp_membership(universes[output_var], mf, bounded_value)
        )
    return memberships


def _derive_mamdani_consequents(tsk_value, output_var, fuzzy_sets, universes):
    output_memberships = _compute_output_memberships(
        tsk_value,
        output_var,
        fuzzy_sets,
        universes,
    )
    ranked_labels = sorted(
        output_memberships.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    main_label, main_certainty = ranked_labels[0]
    secondary = None

    if len(ranked_labels) > 1:
        secondary_label, secondary_certainty = ranked_labels[1]
        if secondary_certainty > 0.0:
            secondary = {
                "label": secondary_label,
                "certainty": secondary_certainty,
            }

    return {
        "main": {
            "label": main_label,
            "certainty": main_certainty,
        },
        "secondary": secondary,
        "memberships": output_memberships,
    }


def _compute_output_label_centroids(output_var, fuzzy_sets, universes):
    return {
        label: float(fuzz.defuzz(universes[output_var], mf, "centroid"))
        for label, mf in fuzzy_sets[output_var].items()
    }


def generate_rules(
    data,
    inputs,
    outputs,
    fuzzy_sets,
    universes,
    alpha=DEFAULT_ALPHA,
):
    """
    Generuje bazę reguł metodą Nozaki–Ishibuchi–Tanaka bliższą oryginalnemu
    artykułowi:
    1. buduje reguły TSK-0 ze średnią ważoną wyjścia,
    2. odwzorowuje każdy konsekwent TSK na główną i pomocniczą regułę
       Mamdaniego wraz ze współczynnikami pewności.

    Parameters
    ----------
    data : pandas.DataFrame
        Dane uczące.
    inputs : list[str]
        Lista nazw zmiennych wejściowych.
    outputs : list[str]
        Lista nazw zmiennych wyjściowych (obsługiwane jedno wyjście).
    fuzzy_sets : dict
        {'x1': {...}, 'x2': {...}, 'y': {...}}
    universes : dict
        {'x1': np.array, 'x2': np.array, 'y': np.array}
    alpha : float, default=1.0
        Parametr nieliniowego skalowania wag z fazy TSK.

    Returns
    -------
    dict
        Baza reguł:
        key   -> krotka etykiet wejść
        value -> {
            "tsk_consequent": float,
            "weight": float,
            "main": {"label": str, "certainty": float},
            "secondary": {"label": str, "certainty": float} | None,
            "memberships": dict[str, float],
        }
    """
    if alpha <= 0:
        raise ValueError("Parametr alpha musi być dodatni.")

    output_var = outputs[0]
    tsk_rule_base = {}

    # Faza 1: reguły TSK-0 z ważoną średnią wyjścia.
    for _, row in data.iterrows():
        rule_key, antecedent_memberships = _compute_input_memberships(
            row,
            inputs,
            fuzzy_sets,
            universes,
        )
        sample_weight = float(np.prod(np.power(antecedent_memberships, alpha)))
        if sample_weight == 0.0:
            continue

        y_value = float(row[output_var])
        if rule_key not in tsk_rule_base:
            tsk_rule_base[rule_key] = {
                "weighted_output_sum": sample_weight * y_value,
                "sum_weight": sample_weight,
            }
        else:
            tsk_rule_base[rule_key]["weighted_output_sum"] += sample_weight * y_value
            tsk_rule_base[rule_key]["sum_weight"] += sample_weight

    # Faza 2: odwzorowanie reguł TSK na główną i pomocniczą bazę Mamdaniego.
    final_rules = {}
    for rule_key, rule_data in tsk_rule_base.items():
        sum_weight = rule_data["sum_weight"]
        if sum_weight == 0.0:
            continue

        tsk_consequent = rule_data["weighted_output_sum"] / sum_weight
        mamdani_consequents = _derive_mamdani_consequents(
            tsk_consequent,
            output_var,
            fuzzy_sets,
            universes,
        )
        final_rules[rule_key] = {
            "tsk_consequent": float(tsk_consequent),
            "weight": float(sum_weight),
            "main": mamdani_consequents["main"],
            "secondary": mamdani_consequents["secondary"],
            "memberships": mamdani_consequents["memberships"],
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

        main_rule = rule_data["main"]
        main_text = (
            f"{main_rule['label']} (CF={main_rule['certainty']:.3f})"
        )

        secondary_rule = rule_data.get("secondary")
        secondary_text = ""
        if secondary_rule and secondary_rule["certainty"] > threshold:
            secondary_text = (
                f", secondary={secondary_rule['label']}"
                f" (CF={secondary_rule['certainty']:.3f})"
            )

        print(
            f"Rule {i}: IF {inputs_text} "
            f"THEN y IS {main_text}{secondary_text} "
            f"[TSK={rule_data['tsk_consequent']:.6f}, "
            f"waga = {rule_data['weight']:.6f}]"
        )

    print("=== KONIEC LISTY REGUŁ ===\n")


def apply_rules(inputs, rules_dict, fuzzy_sets, universes, outputs):
    output_var = outputs[0]
    memberships = {}
    for inp_name, inp_val in inputs.items():
        memberships[inp_name] = {
            name: fuzz.interp_membership(universes[inp_name], mf, inp_val)
            for name, mf in fuzzy_sets[inp_name].items()
        }

    output_centroids = _compute_output_label_centroids(
        output_var,
        fuzzy_sets,
        universes,
    )

    numerator = 0.0
    denominator = 0.0
    activated_rules = []

    for condition, rule_data in rules_dict.items():
        mu_list = []
        for idx, inp_name in enumerate(inputs.keys()):
            mu_list.append(memberships[inp_name].get(condition[idx], 0.0))

        firing_strength = float(np.prod(mu_list))
        if firing_strength == 0.0:
            continue

        main_rule = rule_data["main"]
        main_activation = firing_strength * main_rule["certainty"]
        numerator += main_activation * output_centroids[main_rule["label"]]
        denominator += main_activation

        activated_rule = {
            "condition": condition,
            "firing_strength": firing_strength,
            "main": {
                "label": main_rule["label"],
                "certainty": main_rule["certainty"],
                "activation": main_activation,
            },
        }

        secondary_rule = rule_data.get("secondary")
        if secondary_rule is not None:
            secondary_activation = (
                firing_strength * secondary_rule["certainty"]
            )
            numerator += secondary_activation * output_centroids[secondary_rule["label"]]
            denominator += secondary_activation
            activated_rule["secondary"] = {
                "label": secondary_rule["label"],
                "certainty": secondary_rule["certainty"],
                "activation": secondary_activation,
            }

        activated_rules.append(activated_rule)

    if denominator == 0.0:
        return np.nan, []

    y_pred = numerator / denominator
    return y_pred, activated_rules


def predict(data, inputs, outputs, rules_dict, fuzzy_sets, universes):
    """Oblicza predykcje modelu Nozaki–Ishibuchi–Tanaka dla wszystkich próbek.

    Parameters
    ----------
    data : pandas.DataFrame
        Dane wejściowe.
    inputs : list[str]
        Nazwy zmiennych wejściowych.
    outputs : list[str]
        Nazwy zmiennych wyjściowych (obsługiwane jedno wyjście).
    rules_dict : dict
        Słownik reguł NIT.
    fuzzy_sets : dict
        Zbiory rozmyte.
    universes : dict
        Uniwersa dla wszystkich zmiennych.

    Returns
    -------
    dict[str, np.ndarray]
        Słownik przewidywanych wartości: {output_name: np.ndarray}
    """
    output_name = outputs[0]
    n_samples = len(data)
    y_predictions = np.zeros(n_samples, dtype=float)

    for idx, (_, row) in enumerate(data.iterrows()):
        input_dict = {inp: row[inp] for inp in inputs}
        y_pred, _ = apply_rules(input_dict, rules_dict, fuzzy_sets, universes, outputs)
        y_predictions[idx] = y_pred if not np.isnan(y_pred) else 0.0

    return {output_name: y_predictions}

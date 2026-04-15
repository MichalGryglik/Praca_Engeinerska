import pandas as pd

from core.evaluation.metrics import compute_mse
from core.rule_generators import sugeno_yasukawa as sy
from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from core.rule_generators import wang_mendel as wm
from sandbox import example_config


def run():
    data = pd.read_csv("sandbox/data.csv", sep=";")
    inputs_values = {"x1": 6.0, "x2": 6.0}

    print("\n" + "=" * 70)
    print("METODA 1: WANG-MENDEL")
    print("=" * 70)

    rules_dict = wm.generate_rules(
        data=data,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
    )

    print("\nReguly Wang-Mendel:")
    wm.pretty_print_rules(rules_dict, example_config.inputs)

    print("\nWyniki Wang-Mendel dla pojedynczej probki:")
    y_pred_wm_single, activated_wm = wm.apply_rules(
        inputs=inputs_values,
        rules_dict=rules_dict,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
        outputs=example_config.outputs,
    )
    print(
        f"  Przewidywana wartosc y = {y_pred_wm_single:.3f} "
        f"dla x1={inputs_values['x1']}, x2={inputs_values['x2']}"
    )
    print("  Aktywne reguly:")
    for y_label, w in activated_wm:
        print(f"    THEN y is {y_label} [sila aktywacji = {w:.3f}]")

    print("\n" + "=" * 70)
    print("METODA 2: NOZAKI-ISHIBUCHI-TANAKA")
    print("=" * 70)

    nit_rules = nit.generate_rules(
        data=data,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
    )

    print("\nReguly Nozaki-Ishibuchi-Tanaka:")
    nit.pretty_print_rules(nit_rules, example_config.inputs)

    print("\nWyniki NIT dla pojedynczej probki:")
    y_pred_nit_single, activated_nit = nit.apply_rules(
        inputs=inputs_values,
        rules_dict=nit_rules,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
        outputs=example_config.outputs,
    )
    print(
        f"  Przewidywana wartosc y = {y_pred_nit_single:.3f} "
        f"dla x1={inputs_values['x1']}, x2={inputs_values['x2']}"
    )
    print(f"  Liczba aktywnych regul: {len(activated_nit)}")

    print("\n" + "=" * 70)
    print("METODA 3: SUGENO-YASUKAWA")
    print("=" * 70)

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

    print("\nReguly poczatkowe Sugeno-Yasukawa:")
    sy.print_rules(sugeno_rules)

    normalized_strengths_result = sy.compute_normalized_firing_strengths(
        data=data,
        inputs=example_config.inputs,
        rules_dict=sugeno_rules,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
    )

    print("\nAktualizacja regul Sugeno-Yasukawa:")
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

    print("\nReguly koncowe Sugeno-Yasukawa:")
    sy.print_rules(sugeno_rules)

    print("\nSugeno-Yasukawa: przewidywanie dla pojedynczej probki:")
    single_sample = pd.DataFrame([inputs_values])
    y_pred_sy_single = sy.predict(
        data=single_sample,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=sugeno_rules,
    )
    y_pred_sy_single_value = y_pred_sy_single[example_config.outputs[0]][0]
    print(
        f"  Przewidywana wartosc y = {y_pred_sy_single_value:.3f} "
        f"dla x1={inputs_values['x1']}, x2={inputs_values['x2']}"
    )

    print("\n\n" + "=" * 70)
    print("CZESC 4: POROWNANIE PREDYKCJI I MSE")
    print("=" * 70)

    y_true = data[example_config.outputs[0]].to_numpy(dtype=float)

    print("\n1. WANG-MENDEL")
    print("-" * 50)
    y_pred_wm = wm.predict(
        data=data,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=rules_dict,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
    )
    y_pred_wm_values = y_pred_wm[example_config.outputs[0]]
    mse_wm = compute_mse(y_true, y_pred_wm_values)
    print(f"MSE: {mse_wm:.6f}")

    y_pred_wm_single, _ = wm.apply_rules(
        inputs=inputs_values,
        rules_dict=rules_dict,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
        outputs=example_config.outputs,
    )
    print(
        f"Predykcja dla x1={inputs_values['x1']}, "
        f"x2={inputs_values['x2']}: y={y_pred_wm_single:.3f}"
    )

    print("\n2. NOZAKI-ISHIBUCHI-TANAKA")
    print("-" * 50)
    y_pred_nit = nit.predict(
        data=data,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=nit_rules,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
    )
    y_pred_nit_values = y_pred_nit[example_config.outputs[0]]
    mse_nit = compute_mse(y_true, y_pred_nit_values)
    print(f"MSE: {mse_nit:.6f}")

    y_pred_nit_single, _ = nit.apply_rules(
        inputs=inputs_values,
        rules_dict=nit_rules,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
        outputs=example_config.outputs,
    )
    print(
        f"Predykcja dla x1={inputs_values['x1']}, "
        f"x2={inputs_values['x2']}: y={y_pred_nit_single:.3f}"
    )

    print("\n3. SUGENO-YASUKAWA")
    print("-" * 50)
    y_pred_sy = sy.predict(
        data=data,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=sugeno_rules,
    )
    y_pred_sy_values = y_pred_sy[example_config.outputs[0]]
    mse_sy = compute_mse(y_true, y_pred_sy_values)
    print(f"MSE: {mse_sy:.6f}")

    single_sample = pd.DataFrame([inputs_values])
    y_pred_sy_single = sy.predict(
        data=single_sample,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=sugeno_rules,
    )
    y_pred_sy_single_value = y_pred_sy_single[example_config.outputs[0]][0]
    print(
        f"Predykcja dla x1={inputs_values['x1']}, "
        f"x2={inputs_values['x2']}: y={y_pred_sy_single_value:.3f}"
    )

    print("\n\n" + "=" * 70)
    print("CZESC 5: TESTOWANIE NA NOWYCH PROBKACH")
    print("=" * 70)

    test_samples = pd.DataFrame(
        [
            {"x1": 2.0, "x2": 3.0},
            {"x1": 7.5, "x2": 1.5},
            {"x1": 4.0, "x2": 8.0},
        ]
    )

    test_y_true = test_samples[["x1", "x2"]].sum(axis=1).to_numpy()

    print("\nProbki testowe:")
    for idx, row in test_samples.iterrows():
        expected_y = row["x1"] + row["x2"]
        print(
            f"  Probka {idx + 1}: x1={row['x1']:.1f}, "
            f"x2={row['x2']:.1f} -> oczekiwane y={expected_y:.1f}"
        )

    print("\n" + "-" * 70)
    print("PREDYKCJE NA NOWYCH PROBKACH:")
    print("-" * 70)

    y_pred_wm_test = wm.predict(
        data=test_samples,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=rules_dict,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
    )
    y_pred_wm_test_values = y_pred_wm_test[example_config.outputs[0]]
    mse_wm_test = compute_mse(test_y_true, y_pred_wm_test_values)

    print("\nWang-Mendel:")
    for idx, (y_true_value, y_pred_value) in enumerate(
        zip(test_y_true, y_pred_wm_test_values)
    ):
        print(
            f"  Probka {idx + 1}: oczekiwane={y_true_value:.1f}, "
            f"predykcja={y_pred_value:.3f}, "
            f"blad={abs(y_true_value - y_pred_value):.3f}"
        )
    print(f"  MSE na testach: {mse_wm_test:.6f}")

    y_pred_nit_test = nit.predict(
        data=test_samples,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=nit_rules,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
    )
    y_pred_nit_test_values = y_pred_nit_test[example_config.outputs[0]]
    mse_nit_test = compute_mse(test_y_true, y_pred_nit_test_values)

    print("\nNozaki-Ishibuchi-Tanaka:")
    for idx, (y_true_value, y_pred_value) in enumerate(
        zip(test_y_true, y_pred_nit_test_values)
    ):
        print(
            f"  Probka {idx + 1}: oczekiwane={y_true_value:.1f}, "
            f"predykcja={y_pred_value:.3f}, "
            f"blad={abs(y_true_value - y_pred_value):.3f}"
        )
    print(f"  MSE na testach: {mse_nit_test:.6f}")

    y_pred_sy_test = sy.predict(
        data=test_samples,
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        rules_dict=sugeno_rules,
    )
    y_pred_sy_test_values = y_pred_sy_test[example_config.outputs[0]]
    mse_sy_test = compute_mse(test_y_true, y_pred_sy_test_values)

    print("\nSugeno-Yasukawa:")
    for idx, (y_true_value, y_pred_value) in enumerate(
        zip(test_y_true, y_pred_sy_test_values)
    ):
        print(
            f"  Probka {idx + 1}: oczekiwane={y_true_value:.1f}, "
            f"predykcja={y_pred_value:.3f}, "
            f"blad={abs(y_true_value - y_pred_value):.3f}"
        )
    print(f"  MSE na testach: {mse_sy_test:.6f}")

    print("\n" + "=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)
    print("\nMSE dla kazdej metody na calym zbiorze danych:")
    print(f"  Wang-Mendel:                   {mse_wm:.6f}")
    print(f"  Nozaki-Ishibuchi-Tanaka:       {mse_nit:.6f}")
    print(f"  Sugeno-Yasukawa:               {mse_sy:.6f}")

    print(
        f"\nPredykcje dla wektora wejscia "
        f"x1={inputs_values['x1']}, x2={inputs_values['x2']}:"
    )
    print("  Oczekiwane: y ~ 12.0")
    print(f"  Wang-Mendel:                   y = {y_pred_wm_single:.3f}")
    print(f"  Nozaki-Ishibuchi-Tanaka:       y = {y_pred_nit_single:.3f}")
    print(f"  Sugeno-Yasukawa:               y = {y_pred_sy_single_value:.3f}")

    print("\n" + "=" * 70 + "\n")

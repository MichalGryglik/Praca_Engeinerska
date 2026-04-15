import pandas as pd

from core.evaluation.metrics import compute_mse
from core.rule_generators import sugeno_yasukawa as sy
from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from core.rule_generators import wang_mendel as wm
from core import data_loader


def run():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: TEP DATA")
    print("=" * 70)

    tep_train_path = "data/TEP_FaultFree_Training.csv"
    tep_test_path = "data/TEP_FaultFree_Testing.csv"

    try:
        tep_train = pd.read_csv(tep_train_path).head(1000)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found: {tep_train_path}. See README.md -> Dane (dataset)"
        ) from exc

    rules_dict_tep = wm.generate_rules(
        data=tep_train,
        inputs=data_loader.inputs,
        outputs=data_loader.outputs,
        fuzzy_sets=data_loader.fuzzy_sets,
        universes=data_loader.universes,
    )

    nit_rules_tep = nit.generate_rules(
        data=tep_train,
        inputs=data_loader.inputs,
        outputs=data_loader.outputs,
        fuzzy_sets=data_loader.fuzzy_sets,
        universes=data_loader.universes,
    )

    n_rules = 3
    eps_sigma = 1.0

    centers, membership_matrix = sy.initialize_clusters_with_cmeans(
        data=tep_train,
        inputs=data_loader.inputs,
        n_rules=n_rules,
    )

    sugeno_rules_tep = sy.build_initial_rules_from_clusters(
        centers=centers,
        inputs=data_loader.inputs,
        outputs=data_loader.outputs,
        eps_sigma=eps_sigma,
    )

    normalized_strengths_result = sy.compute_normalized_firing_strengths(
        data=tep_train,
        inputs=data_loader.inputs,
        rules_dict=sugeno_rules_tep,
        fuzzy_sets=data_loader.fuzzy_sets,
        universes=data_loader.universes,
    )

    sy.update_consequents_ls_wls(
        data=tep_train,
        inputs=data_loader.inputs,
        outputs=data_loader.outputs,
        rules_dict=sugeno_rules_tep,
        normalized_strengths=normalized_strengths_result,
    )

    sy.update_antecedents(
        data=tep_train,
        inputs=data_loader.inputs,
        rules_dict=sugeno_rules_tep,
        normalized_strengths=normalized_strengths_result,
        eps_sigma=eps_sigma,
    )

    try:
        test_samples_tep = pd.read_csv(tep_test_path).head(3)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found: {tep_test_path}. See README.md -> Dane (dataset)"
        ) from exc

    test_y_true_tep = test_samples_tep[data_loader.outputs[0]].to_numpy(dtype=float)

    print("\nProbki testowe TEP:")
    for idx, row in test_samples_tep.iterrows():
        print(
            f"  Probka {idx + 1}: xmeas_38={row['xmeas_38']:.3f}, "
            f"xmeas_33={row['xmeas_33']:.3f}, xmeas_27={row['xmeas_27']:.3f} "
            f"-> oczekiwane xmeas_1={row['xmeas_1']:.3f}"
        )

    print("\n" + "-" * 70)
    print("PREDYKCJE NA NOWYCH PROBKACH TEP:")
    print("-" * 70)

    y_pred_wm_tep = wm.predict(
        data=test_samples_tep,
        inputs=data_loader.inputs,
        outputs=data_loader.outputs,
        rules_dict=rules_dict_tep,
        fuzzy_sets=data_loader.fuzzy_sets,
        universes=data_loader.universes,
    )
    y_pred_wm_tep_values = y_pred_wm_tep[data_loader.outputs[0]]
    mse_wm_tep = compute_mse(test_y_true_tep, y_pred_wm_tep_values)

    print("\nWang-Mendel:")
    for idx, (y_true_value, y_pred_value) in enumerate(
        zip(test_y_true_tep, y_pred_wm_tep_values)
    ):
        print(
            f"  Probka {idx + 1}: oczekiwane={y_true_value:.3f}, "
            f"predykcja={y_pred_value:.3f}, "
            f"blad={abs(y_true_value - y_pred_value):.3f}"
        )
    print(f"  MSE na testach: {mse_wm_tep:.6f}")

    y_pred_nit_tep = nit.predict(
        data=test_samples_tep,
        inputs=data_loader.inputs,
        outputs=data_loader.outputs,
        rules_dict=nit_rules_tep,
        fuzzy_sets=data_loader.fuzzy_sets,
        universes=data_loader.universes,
    )
    y_pred_nit_tep_values = y_pred_nit_tep[data_loader.outputs[0]]
    mse_nit_tep = compute_mse(test_y_true_tep, y_pred_nit_tep_values)

    print("\nNozaki-Ishibuchi-Tanaka:")
    for idx, (y_true_value, y_pred_value) in enumerate(
        zip(test_y_true_tep, y_pred_nit_tep_values)
    ):
        print(
            f"  Probka {idx + 1}: oczekiwane={y_true_value:.3f}, "
            f"predykcja={y_pred_value:.3f}, "
            f"blad={abs(y_true_value - y_pred_value):.3f}"
        )
    print(f"  MSE na testach: {mse_nit_tep:.6f}")

    y_pred_sy_tep = sy.predict(
        data=test_samples_tep,
        inputs=data_loader.inputs,
        outputs=data_loader.outputs,
        rules_dict=sugeno_rules_tep,
    )
    y_pred_sy_tep_values = y_pred_sy_tep[data_loader.outputs[0]]
    mse_sy_tep = compute_mse(test_y_true_tep, y_pred_sy_tep_values)

    print("\nSugeno-Yasukawa:")
    for idx, (y_true_value, y_pred_value) in enumerate(
        zip(test_y_true_tep, y_pred_sy_tep_values)
    ):
        print(
            f"  Probka {idx + 1}: oczekiwane={y_true_value:.3f}, "
            f"predykcja={y_pred_value:.3f}, "
            f"blad={abs(y_true_value - y_pred_value):.3f}"
        )
    print(f"  MSE na testach: {mse_sy_tep:.6f}")

    print("\n" + "=" * 70)
    print("PODSUMOWANIE TEP")
    print("=" * 70)
    print("\nMSE dla kazdej metody na probkach testowych TEP:")
    print(f"  Wang-Mendel:                   {mse_wm_tep:.6f}")
    print(f"  Nozaki-Ishibuchi-Tanaka:       {mse_nit_tep:.6f}")
    print(f"  Sugeno-Yasukawa:               {mse_sy_tep:.6f}")

    print("\n" + "=" * 70 + "\n")

from core.data_loader import build_dataset_spec_from_data, load_tep_test, load_tep_train
from core.experiment_runner import (
    ExperimentConfig,
    evaluate_model,
    print_summary,
    train_nit,
    train_sy,
    train_wm,
)


def run():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: TEP DATA")
    print("=" * 70)

    run_config = {
        "sample_size": 1000,
        "test_sample_size": 1000,
        "inputs": ["xmeas_38", "xmeas_33", "xmeas_27"],
        "outputs": ["xmeas_1"],
    }
    inputs = run_config["inputs"]
    outputs = run_config["outputs"]
    labels_by_variable = {
        variable_name: ["S2", "S1", "CE", "B1", "B2"]
        for variable_name in inputs + outputs
    }

    tep_train = load_tep_train(n_samples=run_config["sample_size"])
    tep_test = load_tep_test(n_samples=run_config["test_sample_size"])
    spec_data = tep_train[inputs + outputs]
    tep_spec = build_dataset_spec_from_data(
        data=spec_data,
        inputs=inputs,
        outputs=outputs,
        labels_by_variable=labels_by_variable,
    )

    config = ExperimentConfig(
        inputs=tep_spec.inputs,
        outputs=tep_spec.outputs,
        fuzzy_sets=tep_spec.fuzzy_sets,
        universes=tep_spec.universes,
        sample_size=run_config["sample_size"],
        sy_params={"n_rules": 3, "eps_sigma": 1.0},
    )

    wm_model = train_wm(tep_train, config)
    nit_model = train_nit(tep_train, config)
    sy_model = train_sy(tep_train, config)

    wm_results = evaluate_model(wm_model, "wm", tep_test, config)
    nit_results = evaluate_model(nit_model, "nit", tep_test, config)
    sy_results = evaluate_model(sy_model, "sy", tep_test, config)

    print_summary(wm_results, nit_results, sy_results)

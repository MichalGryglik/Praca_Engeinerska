from core.data_loader import build_dataset_spec_from_data, load_tep_test, load_tep_train
from core.experiment_runner import (
    ExperimentConfig,
    evaluate_model,
    print_summary,
    train_nit,
    train_sy,
    train_wm,
)
from core.results_writer import save_metrics_summary
from core.scenarios import (
    ScenarioConfig,
    apply_training_scenario,
    prepare_numeric_training_data,
    print_scenario_summary,
)


def run(scenario: ScenarioConfig | None = None, seed: int = 42):
    scenario = scenario or ScenarioConfig()
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
    scenario_columns = inputs + outputs
    tep_train = tep_train.copy()
    tep_train[scenario_columns] = apply_training_scenario(
        data=tep_train[scenario_columns],
        scenario=scenario,
        seed=seed,
        gaussian_noise_columns=outputs,
        missing_columns=scenario_columns,
        outlier_columns=outputs,
    )
    tep_train[scenario_columns] = prepare_numeric_training_data(
        tep_train[scenario_columns],
        columns=scenario_columns,
    )
    spec_data = tep_train[scenario_columns]
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

    print_scenario_summary(
        title="EKSPERYMENT: Tennessee Eastman Process",
        scenario=scenario,
        sample_size=run_config["sample_size"],
        missing_columns=scenario_columns,
        outlier_columns=outputs,
    )

    (
        wm_model,
        wm_training_time,
        wm_rule_creation_time,
        wm_structure_time,
        wm_learning_time,
    ) = train_wm(tep_train, config)
    (
        nit_model,
        nit_training_time,
        nit_rule_creation_time,
        nit_structure_time,
        nit_learning_time,
    ) = train_nit(tep_train, config)
    (
        sy_model,
        sy_training_time,
        sy_rule_creation_time,
        sy_structure_time,
        sy_learning_time,
    ) = train_sy(tep_train, config)

    wm_results = evaluate_model(
        wm_model,
        "wm",
        tep_test,
        config,
        training_time_seconds=wm_training_time,
        rule_creation_time_seconds=wm_rule_creation_time,
        structure_time_seconds=wm_structure_time,
        learning_time_seconds=wm_learning_time,
    )
    nit_results = evaluate_model(
        nit_model,
        "nit",
        tep_test,
        config,
        training_time_seconds=nit_training_time,
        rule_creation_time_seconds=nit_rule_creation_time,
        structure_time_seconds=nit_structure_time,
        learning_time_seconds=nit_learning_time,
    )
    sy_results = evaluate_model(
        sy_model,
        "sy",
        tep_test,
        config,
        training_time_seconds=sy_training_time,
        rule_creation_time_seconds=sy_rule_creation_time,
        structure_time_seconds=sy_structure_time,
        learning_time_seconds=sy_learning_time,
    )

    print_summary(wm_results, nit_results, sy_results)
    metrics_path = save_metrics_summary(
        "tep",
        [wm_results, nit_results, sy_results],
    )
    print(f"Zapisano metryki: {metrics_path}")

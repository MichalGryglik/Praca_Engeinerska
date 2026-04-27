from dataclasses import dataclass, replace

import pandas as pd

from core.data_loader import load_csv_dataset
from core.experiment_runner import (
    ExperimentConfig,
    ModelResult,
    _print_results_table,
    evaluate_model,
    print_model_results,
    plot_predictions_vs_true,
    print_sample_preview,
    print_summary,
    train_nit,
    train_sy,
    train_wm,
)
from core.scenarios import (
    ScenarioConfig,
    apply_training_scenario,
    prepare_numeric_training_data,
    print_scenario_summary,
)
from core.rule_generators import nozaki_ishibuchi_tanaka as nit
from core.rule_generators import sugeno_yasukawa as sy
from core.rule_generators import wang_mendel as wm
from sandbox import example_config


@dataclass(frozen=True)
class ExampleExperimentSpec:
    train_path: str
    inputs_values: dict[str, float]
    test_samples: pd.DataFrame
    expected_output_name: str


def build_example_spec() -> ExampleExperimentSpec:
    test_samples = pd.DataFrame(
        [
            {"x1": 2.0, "x2": 3.0, "y": 5.0},
            {"x1": 7.5, "x2": 1.5, "y": 9.0},
            {"x1": 4.0, "x2": 8.0, "y": 12.0},
        ]
    )
    return ExampleExperimentSpec(
        train_path="sandbox/data.csv",
        inputs_values={"x1": 6.0, "x2": 6.0},
        test_samples=test_samples,
        expected_output_name=example_config.outputs[0],
    )


def print_single_prediction_wm(model, config: ExperimentConfig, inputs_values: dict[str, float]) -> None:
    y_pred, activated_rules = wm.apply_rules(
        inputs=inputs_values,
        rules_dict=model,
        fuzzy_sets=config.fuzzy_sets,
        universes=config.universes,
        outputs=config.outputs,
    )
    print("\nWyniki Wang-Mendel dla pojedynczej probki:")
    print(
        f"  Przewidywana wartosc y = {y_pred:.3f} "
        f"dla x1={inputs_values['x1']}, x2={inputs_values['x2']}"
    )
    print("  Aktywne reguly:")
    for y_label, weight in activated_rules:
        print(f"    THEN y is {y_label} [sila aktywacji = {weight:.3f}]")


def print_single_prediction_nit(model, config: ExperimentConfig, inputs_values: dict[str, float]) -> None:
    y_pred, activated_rules = nit.apply_rules(
        inputs=inputs_values,
        rules_dict=model,
        fuzzy_sets=config.fuzzy_sets,
        universes=config.universes,
        outputs=config.outputs,
    )
    print("\nWyniki NIT dla pojedynczej probki:")
    print(
        f"  Przewidywana wartosc y = {y_pred:.3f} "
        f"dla x1={inputs_values['x1']}, x2={inputs_values['x2']}"
    )
    print(f"  Liczba aktywnych regul: {len(activated_rules)}")


def print_single_prediction_sy(model, config: ExperimentConfig, inputs_values: dict[str, float]) -> None:
    single_sample = pd.DataFrame([inputs_values])
    prediction = sy.predict(
        data=single_sample,
        inputs=config.inputs,
        outputs=config.outputs,
        rules_dict=model,
    )
    y_pred = prediction[config.outputs[0]][0]
    print("\nSugeno-Yasukawa: przewidywanie dla pojedynczej probki:")
    print(
        f"  Przewidywana wartosc y = {y_pred:.3f} "
        f"dla x1={inputs_values['x1']}, x2={inputs_values['x2']}"
    )


def print_train_metrics(*results) -> None:
    print("\nMetryki dla kazdej metody na calym zbiorze danych:")
    _print_results_table(*results)


def _print_parameter_sensitivity_table(
    parameter_name: str,
    rows: list[tuple[float, ModelResult]],
) -> None:
    print(
        f"{parameter_name:<10} | {'Liczba regul':>12} | {'MSE':>12} | "
        f"{'MAE':>12} | {'RMSE':>12} | {'R^2':>12} | {'Czas [s]':>12}"
    )
    print("-" * 93)
    for parameter_value, result in rows:
        print(
            f"{parameter_value:<10.3g} | {result.rule_count:>12} | "
            f"{result.mse:>12.6f} | {result.mae:>12.6f} | "
            f"{result.rmse:>12.6f} | {result.r_squared:>12.6f} | "
            f"{result.training_time_seconds:>12.6f}"
        )


def analyze_nit_alpha_sensitivity(
    train_data: pd.DataFrame,
    config: ExperimentConfig,
    alpha_values: list[float],
) -> None:
    print("\n\n" + "=" * 70)
    print("ANALIZA PARAMETRU ALPHA DLA METODY NIT")
    print("=" * 70)

    rows = []
    for alpha in alpha_values:
        alpha_config = replace(
            config,
            nit_params={**(config.nit_params or {}), "alpha": alpha},
        )
        (
            model,
            training_time,
            rule_creation_time,
            structure_time,
            learning_time,
        ) = train_nit(train_data, alpha_config)
        result = evaluate_model(
            model,
            "nit",
            train_data,
            alpha_config,
            training_time_seconds=training_time,
            rule_creation_time_seconds=rule_creation_time,
            structure_time_seconds=structure_time,
            learning_time_seconds=learning_time,
        )
        rows.append((alpha, result))

    _print_parameter_sensitivity_table("alpha", rows)


def analyze_sy_m_sensitivity(
    train_data: pd.DataFrame,
    config: ExperimentConfig,
    m_values: list[float],
) -> None:
    print("\n\n" + "=" * 70)
    print("ANALIZA PARAMETRU M DLA METODY SUGENO-YASUKAWA")
    print("=" * 70)

    rows = []
    for m_value in m_values:
        sy_params = {**(config.sy_params or {}), "m": m_value}
        m_config = replace(config, sy_params=sy_params)
        (
            model,
            training_time,
            rule_creation_time,
            structure_time,
            learning_time,
        ) = train_sy(train_data, m_config)
        result = evaluate_model(
            model,
            "sy",
            train_data,
            m_config,
            training_time_seconds=training_time,
            rule_creation_time_seconds=rule_creation_time,
            structure_time_seconds=structure_time,
            learning_time_seconds=learning_time,
        )
        rows.append((m_value, result))

    _print_parameter_sensitivity_table("m", rows)


def run(scenario: ScenarioConfig | None = None, seed: int = 42):
    scenario = scenario or ScenarioConfig()
    spec = build_example_spec()
    train_data = load_csv_dataset(spec.train_path, sep=";")
    scenario_columns = example_config.inputs + example_config.outputs
    train_data = train_data.copy()
    train_data[scenario_columns] = apply_training_scenario(
        data=train_data[scenario_columns],
        scenario=scenario,
        seed=seed,
        gaussian_noise_columns=example_config.outputs,
        missing_columns=scenario_columns,
        outlier_columns=example_config.outputs,
    )
    train_data[scenario_columns] = prepare_numeric_training_data(
        train_data[scenario_columns],
        columns=scenario_columns,
    )
    config = ExperimentConfig(
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
        nit_params={"alpha": 1.0},
        sy_params={"n_rules": 3, "eps_sigma": 1.0},
    )

    print_scenario_summary(
        title="EKSPERYMENT: Przykladowy zbior danych",
        scenario=scenario,
        sample_size=len(train_data),
        missing_columns=scenario_columns,
        outlier_columns=example_config.outputs,
    )

    print("\n" + "=" * 70)
    print("METODA 1: WANG-MENDEL")
    print("=" * 70)

    (
        wm_model,
        wm_training_time,
        wm_rule_creation_time,
        wm_structure_time,
        wm_learning_time,
    ) = train_wm(train_data, config)

    print("\nReguly Wang-Mendel:")
    wm.pretty_print_rules(wm_model, config.inputs)
    print_single_prediction_wm(wm_model, config, spec.inputs_values)

    print("\n" + "=" * 70)
    print("METODA 2: NOZAKI-ISHIBUCHI-TANAKA")
    print("=" * 70)

    (
        nit_model,
        nit_training_time,
        nit_rule_creation_time,
        nit_structure_time,
        nit_learning_time,
    ) = train_nit(train_data, config)

    print("\nReguly Nozaki-Ishibuchi-Tanaka:")
    nit.pretty_print_rules(nit_model, config.inputs)
    print_single_prediction_nit(nit_model, config, spec.inputs_values)

    print("\n" + "=" * 70)
    print("METODA 3: SUGENO-YASUKAWA")
    print("=" * 70)

    (
        sy_model,
        sy_training_time,
        sy_rule_creation_time,
        sy_structure_time,
        sy_learning_time,
    ) = train_sy(train_data, config)

    print("\nReguly poczatkowe Sugeno-Yasukawa:")
    sy.print_rules(sy_model)
    print_single_prediction_sy(sy_model, config, spec.inputs_values)

    print("\n\n" + "=" * 70)
    print("CZESC 4: POROWNANIE PREDYKCJI I METRYK")
    print("=" * 70)

    wm_train_results = evaluate_model(
        wm_model,
        "wm",
        train_data,
        config,
        training_time_seconds=wm_training_time,
        rule_creation_time_seconds=wm_rule_creation_time,
        structure_time_seconds=wm_structure_time,
        learning_time_seconds=wm_learning_time,
    )
    nit_train_results = evaluate_model(
        nit_model,
        "nit",
        train_data,
        config,
        training_time_seconds=nit_training_time,
        rule_creation_time_seconds=nit_rule_creation_time,
        structure_time_seconds=nit_structure_time,
        learning_time_seconds=nit_learning_time,
    )
    sy_train_results = evaluate_model(
        sy_model,
        "sy",
        train_data,
        config,
        training_time_seconds=sy_training_time,
        rule_creation_time_seconds=sy_rule_creation_time,
        structure_time_seconds=sy_structure_time,
        learning_time_seconds=sy_learning_time,
    )
    print_train_metrics(wm_train_results, nit_train_results, sy_train_results)

    analyze_nit_alpha_sensitivity(
        train_data=train_data,
        config=config,
        alpha_values=[0.5, 0.8, 1.0, 2.0, 3.0],
    )
    analyze_sy_m_sensitivity(
        train_data=train_data,
        config=config,
        m_values=[1.5, 2.0, 2.5, 3.0, 4.0],
    )

    print("\n\n" + "=" * 70)
    print("CZESC 5: TESTOWANIE NA NOWYCH PROBKACH")
    print("=" * 70)

    test_data = spec.test_samples
    print_sample_preview(test_data, config)

    print("\n" + "-" * 70)
    print("PREDYKCJE NA NOWYCH PROBKACH:")
    print("-" * 70)

    wm_test_results = evaluate_model(
        wm_model,
        "wm",
        test_data,
        config,
        training_time_seconds=wm_training_time,
        rule_creation_time_seconds=wm_rule_creation_time,
        structure_time_seconds=wm_structure_time,
        learning_time_seconds=wm_learning_time,
    )
    nit_test_results = evaluate_model(
        nit_model,
        "nit",
        test_data,
        config,
        training_time_seconds=nit_training_time,
        rule_creation_time_seconds=nit_rule_creation_time,
        structure_time_seconds=nit_structure_time,
        learning_time_seconds=nit_learning_time,
    )
    sy_test_results = evaluate_model(
        sy_model,
        "sy",
        test_data,
        config,
        training_time_seconds=sy_training_time,
        rule_creation_time_seconds=sy_rule_creation_time,
        structure_time_seconds=sy_structure_time,
        learning_time_seconds=sy_learning_time,
    )
    print_model_results(wm_test_results, config)
    print_model_results(nit_test_results, config)
    print_model_results(sy_test_results, config)
    print_summary(wm_test_results, nit_test_results, sy_test_results)
    plot_predictions_vs_true(
        wm_test_results,
        nit_test_results,
        sy_test_results,
        title="Przykladowy zbior danych: y_pred vs y_true",
        output_path="results/example_experiment_y_pred_vs_y_true.png",
    )

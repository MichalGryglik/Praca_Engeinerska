from dataclasses import dataclass

import pandas as pd

from core.data_loader import load_csv_dataset
from core.experiment_runner import (
    ExperimentConfig,
    evaluate_model,
    print_model_results,
    print_sample_preview,
    print_summary,
    train_nit,
    train_sy,
    train_wm,
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
    print("\nMSE dla kazdej metody na calym zbiorze danych:")
    for result in results:
        print(f"  {result.name.upper():<30} {result.mse:.6f}")


def run():
    spec = build_example_spec()
    train_data = load_csv_dataset(spec.train_path, sep=";")
    config = ExperimentConfig(
        inputs=example_config.inputs,
        outputs=example_config.outputs,
        fuzzy_sets=example_config.fuzzy_sets,
        universes=example_config.universes,
        sy_params={"n_rules": 3, "eps_sigma": 1.0},
    )

    print("\n" + "=" * 70)
    print("METODA 1: WANG-MENDEL")
    print("=" * 70)

    wm_model = train_wm(train_data, config)

    print("\nReguly Wang-Mendel:")
    wm.pretty_print_rules(wm_model, config.inputs)
    print_single_prediction_wm(wm_model, config, spec.inputs_values)

    print("\n" + "=" * 70)
    print("METODA 2: NOZAKI-ISHIBUCHI-TANAKA")
    print("=" * 70)

    nit_model = train_nit(train_data, config)

    print("\nReguly Nozaki-Ishibuchi-Tanaka:")
    nit.pretty_print_rules(nit_model, config.inputs)
    print_single_prediction_nit(nit_model, config, spec.inputs_values)

    print("\n" + "=" * 70)
    print("METODA 3: SUGENO-YASUKAWA")
    print("=" * 70)

    sy_model = train_sy(train_data, config)

    print("\nReguly poczatkowe Sugeno-Yasukawa:")
    sy.print_rules(sy_model)
    print_single_prediction_sy(sy_model, config, spec.inputs_values)

    print("\n\n" + "=" * 70)
    print("CZESC 4: POROWNANIE PREDYKCJI I MSE")
    print("=" * 70)

    wm_train_results = evaluate_model(wm_model, "wm", train_data, config)
    nit_train_results = evaluate_model(nit_model, "nit", train_data, config)
    sy_train_results = evaluate_model(sy_model, "sy", train_data, config)
    print_train_metrics(wm_train_results, nit_train_results, sy_train_results)

    print("\n\n" + "=" * 70)
    print("CZESC 5: TESTOWANIE NA NOWYCH PROBKACH")
    print("=" * 70)

    test_data = spec.test_samples
    print_sample_preview(test_data, config)

    print("\n" + "-" * 70)
    print("PREDYKCJE NA NOWYCH PROBKACH:")
    print("-" * 70)

    wm_test_results = evaluate_model(wm_model, "wm", test_data, config)
    nit_test_results = evaluate_model(nit_model, "nit", test_data, config)
    sy_test_results = evaluate_model(sy_model, "sy", test_data, config)
    print_model_results(wm_test_results, config)
    print_model_results(nit_test_results, config)
    print_model_results(sy_test_results, config)
    print_summary(wm_test_results, nit_test_results, sy_test_results)

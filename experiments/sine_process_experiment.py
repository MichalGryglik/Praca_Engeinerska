from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from core.data_loader import build_dataset_spec_from_data
from core.experiment_runner import (
    ExperimentConfig,
    evaluate_model,
    plot_predictions_vs_true,
    print_sample_preview,
    print_summary,
    train_nit,
    train_sy,
    train_wm,
)
from core.results_writer import save_metrics_summary, save_predictions
from core.scenarios import (
    ScenarioConfig,
    apply_training_scenario,
    prepare_numeric_training_data,
    print_scenario_summary,
)


DEFAULT_PERIOD_COUNT = 3
DEFAULT_TRAIN_SAMPLE_SIZE = 450
DEFAULT_TEST_SAMPLE_SIZE = 450
DEFAULT_X_MIN = 0.0
DEFAULT_X_MAX = DEFAULT_PERIOD_COUNT * 2.0 * np.pi
OUT_OF_RANGE_EXTENSION_RATIO = 0.5


def _generate_labels(count: int) -> list[str]:
    """Generuje listę etykiet dla danej liczby."""
    if count % 2 == 0:
        raise ValueError("Liczba etykiet musi być nieparzysta (potrzebuje ZE w środku).")
    
    mid = count // 2
    labels = []
    for i in range(mid, 0, -1):
        if i <= 3:
            labels.append(f"N{i}")
        else:
            labels.append(f"N{i}")
    labels.append("ZE")
    for i in range(1, mid + 1):
        if i <= 3:
            labels.append(f"P{i}")
        else:
            labels.append(f"P{i}")
    return labels


@dataclass(frozen=True)
class SineExperimentSpec:
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    scenario: ScenarioConfig


def _true_process(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(x) + 1.0


def _build_base_dataset(
    sample_size: int,
    x_min: float,
    x_max: float,
) -> pd.DataFrame:
    x_values = np.linspace(x_min, x_max, sample_size)
    return pd.DataFrame({"x": x_values, "y": _true_process(x_values)})


def _build_out_of_range_test_data(reference_data: pd.DataFrame) -> pd.DataFrame:
    x_min = float(reference_data["x"].min())
    x_max = float(reference_data["x"].max())
    span = x_max - x_min
    extended_min = x_min - OUT_OF_RANGE_EXTENSION_RATIO * span
    extended_max = x_max + OUT_OF_RANGE_EXTENSION_RATIO * span
    return _build_base_dataset(len(reference_data), extended_min, extended_max)


def _noise_std_from_y_percent(data: pd.DataFrame, percent: float) -> float:
    y_range = float(data["y"].max() - data["y"].min())
    return y_range * percent / 100.0


def _format_pi_tick(value: float, _position: int) -> str:
    multiplier = int(round(value / np.pi))
    if not np.isclose(value, multiplier * np.pi, atol=1e-8):
        return f"{value / np.pi:.1f}Pi"
    if multiplier == 0:
        return "0"
    if multiplier == 1:
        return "Pi"
    if multiplier == -1:
        return "-Pi"
    return f"{multiplier}Pi"


def _apply_angle_x_axis(*axes: Any) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    for axis in axes:
        axis.xaxis.set_major_locator(MultipleLocator(np.pi))
        axis.xaxis.set_major_formatter(FuncFormatter(_format_pi_tick))
        axis.set_xlabel("x")


def build_experiment_spec(
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
    test_sample_size: int = DEFAULT_TEST_SAMPLE_SIZE,
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    scenario: ScenarioConfig | None = None,
    seed: int = 42,
) -> SineExperimentSpec:
    scenario = scenario or ScenarioConfig()

    clean_train_data = _build_base_dataset(train_sample_size, x_min, x_max)
    test_data = _build_base_dataset(test_sample_size, x_min, x_max)

    scenario_train_data = apply_training_scenario(
        data=clean_train_data,
        scenario=scenario,
        seed=seed,
        gaussian_noise_columns=["y"],
        missing_columns=["x", "y"],
        outlier_columns=["y"],
    )
    prepared_train_data = prepare_numeric_training_data(
        scenario_train_data,
        sort_by="x",
        columns=["x", "y"],
    )
    return SineExperimentSpec(
        train_data=prepared_train_data,
        test_data=test_data,
        scenario=scenario,
    )


def _print_scenario_summary(spec: SineExperimentSpec) -> None:
    print_scenario_summary(
        title="EKSPERYMENT: y = 2sin(x) + 1",
        scenario=spec.scenario,
        sample_size=len(spec.train_data),
        missing_columns=["x", "y"],
        outlier_columns=["y"],
    )


def run_single_experiment(
    spec: SineExperimentSpec,
    label_counts: list[int],
    sy_rule_counts: list[int],
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Uruchamia eksperymenty dla różnych liczb etykiet/reguł."""
    all_results = []
    
    y_labels = ["NB", "NS", "ZE", "PS", "PB"]
    
    # Najpierw WM i NIT dla wszystkich label_counts
    wm_nit_results = {}
    for label_count in label_counts:
        x_labels = _generate_labels(label_count)
        
        print(f"\n{'='*60}")
        print(f"LICZBA ETYKIET: {label_count}")
        print(f"{'='*60}")
        
        labels_by_variable = {
            "x": x_labels,
            "y": y_labels,
        }
        dataset_spec = build_dataset_spec_from_data(
            data=spec.train_data,
            inputs=["x"],
            outputs=["y"],
            labels_by_variable=labels_by_variable,
        )
        config = ExperimentConfig(
            inputs=dataset_spec.inputs,
            outputs=dataset_spec.outputs,
            fuzzy_sets=dataset_spec.fuzzy_sets,
            universes=dataset_spec.universes,
            sample_size=train_sample_size,
            nit_params={"alpha": 1.0},
            sy_params={"n_rules": 6, "eps_sigma": 0.4},
        )
        
        # Trening WM i NIT
        (
            wm_model,
            wm_training_time,
            wm_rule_creation_time,
            wm_structure_time,
            wm_learning_time,
        ) = train_wm(spec.train_data, config)
        (
            nit_model,
            nit_training_time,
            nit_rule_creation_time,
            nit_structure_time,
            nit_learning_time,
        ) = train_nit(spec.train_data, config)
        
        # Ewaluacja WM
        wm_results = evaluate_model(
            wm_model,
            "wm",
            spec.test_data,
            config,
            training_time_seconds=wm_training_time,
            rule_creation_time_seconds=wm_rule_creation_time,
            structure_time_seconds=wm_structure_time,
            learning_time_seconds=wm_learning_time,
        )
        
        # Ewaluacja NIT
        nit_results = evaluate_model(
            nit_model,
            "nit",
            spec.test_data,
            config,
            training_time_seconds=nit_training_time,
            rule_creation_time_seconds=nit_rule_creation_time,
            structure_time_seconds=nit_structure_time,
            learning_time_seconds=nit_learning_time,
        )
        
        wm_nit_results[label_count] = {
            "wm": wm_results,
            "nit": nit_results,
        }
        
        all_results.append({
            "method": "wm",
            "label_count": label_count,
            "result": wm_results,
        })
        all_results.append({
            "method": "nit",
            "label_count": label_count,
            "result": nit_results,
        })
    
    # Teraz SY z różnymi liczbami reguł (klastrów)
    sy_results_by_clusters = {}
    for sy_rule_count in sy_rule_counts:
        print(f"\n{'='*60}")
        print(f"LICZBA KLASTRÓW SY: {sy_rule_count}")
        print(f"{'='*60}")
        
        x_labels = _generate_labels(17)  # używamy 17 etykiet dla SY
        
        labels_by_variable = {
            "x": x_labels,
            "y": y_labels,
        }
        dataset_spec = build_dataset_spec_from_data(
            data=spec.train_data,
            inputs=["x"],
            outputs=["y"],
            labels_by_variable=labels_by_variable,
        )
        config = ExperimentConfig(
            inputs=dataset_spec.inputs,
            outputs=dataset_spec.outputs,
            fuzzy_sets=dataset_spec.fuzzy_sets,
            universes=dataset_spec.universes,
            sample_size=train_sample_size,
            nit_params={"alpha": 1.0},
            sy_params={"n_rules": sy_rule_count, "eps_sigma": 0.4},
        )
        
        (
            sy_model,
            sy_training_time,
            sy_rule_creation_time,
            sy_structure_time,
            sy_learning_time,
        ) = train_sy(spec.train_data, config)
        
        sy_results = evaluate_model(
            sy_model,
            "sy",
            spec.test_data,
            config,
            training_time_seconds=sy_training_time,
            rule_creation_time_seconds=sy_rule_creation_time,
            structure_time_seconds=sy_structure_time,
            learning_time_seconds=sy_learning_time,
        )
        
        sy_results_by_clusters[sy_rule_count] = sy_results
        
        all_results.append({
            "method": "sy",
            "rule_count": sy_rule_count,
            "result": sy_results,
        })
    
    # Wyświetl 5 tabel z wynikami
    print("\n" + "="*80)
    print("TABELE WYNIKÓW DLA RÓŻNYCH LICZB REGUŁ")
    print("="*80)
    
    # Tabela 1: WM i NIT dla label_count=5
    print("\n" + "="*60)
    print("TABELA 1: WM i NIT (5 etykiet = 5 reguł)")
    print("="*60)
    lc = 5
    print(f"{'Metoda':<8} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    wm_r = wm_nit_results[lc]["wm"]
    nit_r = wm_nit_results[lc]["nit"]
    print(f"{'WM':<8} {wm_r.rule_count:<8} {wm_r.mse:<12.6f} {wm_r.mae:<12.6f} {wm_r.rmse:<12.6f} {wm_r.r_squared:<12.6f}")
    print(f"{'NIT':<8} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")
    
    # Tabela 2: WM i NIT dla label_count=11
    print("\n" + "="*60)
    print("TABELA 2: WM i NIT (11 etykiet = 11 reguł)")
    print("="*60)
    lc = 11
    print(f"{'Metoda':<8} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    wm_r = wm_nit_results[lc]["wm"]
    nit_r = wm_nit_results[lc]["nit"]
    print(f"{'WM':<8} {wm_r.rule_count:<8} {wm_r.mse:<12.6f} {wm_r.mae:<12.6f} {wm_r.rmse:<12.6f} {wm_r.r_squared:<12.6f}")
    print(f"{'NIT':<8} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")
    
    # Tabela 3: WM i NIT dla label_count=17
    print("\n" + "="*60)
    print("TABELA 3: WM i NIT (17 etykiet = 17 reguł)")
    print("="*60)
    lc = 17
    print(f"{'Metoda':<8} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    wm_r = wm_nit_results[lc]["wm"]
    nit_r = wm_nit_results[lc]["nit"]
    print(f"{'WM':<8} {wm_r.rule_count:<8} {wm_r.mse:<12.6f} {wm_r.mae:<12.6f} {wm_r.rmse:<12.6f} {wm_r.r_squared:<12.6f}")
    print(f"{'NIT':<8} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")
    
    # Tabela 4: WM i NIT dla label_count=23
    print("\n" + "="*60)
    print("TABELA 4: WM i NIT (23 etykiet = 23 reguły)")
    print("="*60)
    lc = 23
    print(f"{'Metoda':<8} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    wm_r = wm_nit_results[lc]["wm"]
    nit_r = wm_nit_results[lc]["nit"]
    print(f"{'WM':<8} {wm_r.rule_count:<8} {wm_r.mse:<12.6f} {wm_r.mae:<12.6f} {wm_r.rmse:<12.6f} {wm_r.r_squared:<12.6f}")
    print(f"{'NIT':<8} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")
    
    # Tabela 5: WM i NIT dla label_count=29
    print("\n" + "="*60)
    print("TABELA 5: WM i NIT (29 etykiet = 29 reguł)")
    print("="*60)
    lc = 29
    print(f"{'Metoda':<8} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    wm_r = wm_nit_results[lc]["wm"]
    nit_r = wm_nit_results[lc]["nit"]
    print(f"{'WM':<8} {wm_r.rule_count:<8} {wm_r.mse:<12.6f} {wm_r.mae:<12.6f} {wm_r.rmse:<12.6f} {wm_r.r_squared:<12.6f}")
    print(f"{'NIT':<8} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")
    
    # Tabela 6: SY dla różnych klastrów
    print("\n" + "="*60)
    print("TABELA 6: SY dla różnych liczb klastrów")
    print("="*60)
    print(f"{'Klastry':<10} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    for nc in sy_rule_counts:
        sy_r = sy_results_by_clusters[nc]
        print(f"{nc:<10} {sy_r.rule_count:<8} {sy_r.mse:<12.6f} {sy_r.mae:<12.6f} {sy_r.rmse:<12.6f} {sy_r.r_squared:<12.6f}")
    
    # Tabela 7: WM i NIT dla label_count=51 (overfitting test)
    print("\n" + "="*60)
    print("TABELA 7: WM i NIT (51 etykiet = 51 reguł) - TEST OVERFITTING")
    print("="*60)
    lc = 51
    print(f"{'Metoda':<8} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    wm_r = wm_nit_results[lc]["wm"]
    nit_r = wm_nit_results[lc]["nit"]
    print(f"{'WM':<8} {wm_r.rule_count:<8} {wm_r.mse:<12.6f} {wm_r.mae:<12.6f} {wm_r.rmse:<12.6f} {wm_r.r_squared:<12.6f}")
    print(f"{'NIT':<8} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")
    
    # Tabela 8: SY dla 22 klastrów (overfitting test)
    print("\n" + "="*60)
    print("TABELA 8: SY (22 klastry) - TEST OVERFITTING")
    print("="*60)
    print(f"{'Klastry':<10} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    nc = 22
    sy_r = sy_results_by_clusters[nc]
    print(f"{nc:<10} {sy_r.rule_count:<8} {sy_r.mse:<12.6f} {sy_r.mae:<12.6f} {sy_r.rmse:<12.6f} {sy_r.r_squared:<12.6f}")

    # Tabela 9: WM i NIT dla label_count=153 (mocny test overfittingu)
    print("\n" + "="*60)
    print("TABELA 9: WM i NIT (153 etykiety = 153 reguły) - TEST OVERFITTING 3x")
    print("="*60)
    lc = 153
    print(f"{'Metoda':<8} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    wm_r = wm_nit_results[lc]["wm"]
    nit_r = wm_nit_results[lc]["nit"]
    print(f"{'WM':<8} {wm_r.rule_count:<8} {wm_r.mse:<12.6f} {wm_r.mae:<12.6f} {wm_r.rmse:<12.6f} {wm_r.r_squared:<12.6f}")
    print(f"{'NIT':<8} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")

    # Tabela 10: SY dla 66 klastrów (mocny test overfittingu)
    print("\n" + "="*60)
    print("TABELA 10: SY (66 klastrów) - TEST OVERFITTING 3x")
    print("="*60)
    print(f"{'Klastry':<10} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    nc = 66
    sy_r = sy_results_by_clusters[nc]
    print(f"{nc:<10} {sy_r.rule_count:<8} {sy_r.mse:<12.6f} {sy_r.mae:<12.6f} {sy_r.rmse:<12.6f} {sy_r.r_squared:<12.6f}")
    
    return all_results


def plot_comparison_x_y(
    spec: SineExperimentSpec,
    wm_result: Any,
    nit_result: Any,
    sy_result: Any,
    output_path: str = "results/plots/sinus_comparison.png",
) -> None:
    """Tworzy 3 wykresy obok siebie: x vs y_true i y_pred dla każdej metody."""
    import os
    import matplotlib.pyplot as plt
    
    # Konfiguracja matplotlib
    matplotlib_config_dir = os.path.join(os.getcwd(), "results", ".matplotlib", str(os.getpid()))
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_config_dir)
    
    # Pobierz dane testowe
    x_test = spec.test_data["x"].values
    y_true = spec.test_data["y"].values
    
    # Pobierz predykcje
    wm_pred = np.asarray(wm_result.predictions["y"], dtype=float).ravel()
    nit_pred = np.asarray(nit_result.predictions["y"], dtype=float).ravel()
    sy_pred = np.asarray(sy_result.predictions["y"], dtype=float).ravel()
    
    # Sortuj według x
    sort_idx = np.argsort(x_test)
    x_sorted = x_test[sort_idx]
    y_true_sorted = y_true[sort_idx]
    wm_pred_sorted = wm_pred[sort_idx]
    nit_pred_sorted = nit_pred[sort_idx]
    sy_pred_sorted = sy_pred[sort_idx]
    
    # Tworzenie wykresu
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Wang-Mendel
    axes[0].plot(x_sorted, y_true_sorted, 'b-', label='y_true (2sin(x)+1)', linewidth=2)
    axes[0].plot(x_sorted, wm_pred_sorted, 'r--', label='y_pred', linewidth=1.5, alpha=0.8)
    axes[0].scatter(x_sorted, wm_pred_sorted, c='red', s=10, alpha=0.5, label='predykcje')
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Wang-Mendel (R^2={wm_result.r_squared:.4f}, {wm_result.rule_count} reguł)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Nozaki-Ishibuchi-Tanaka
    axes[1].plot(x_sorted, y_true_sorted, 'b-', label='y_true (2sin(x)+1)', linewidth=2)
    axes[1].plot(x_sorted, nit_pred_sorted, 'g--', label='y_pred', linewidth=1.5, alpha=0.8)
    axes[1].scatter(x_sorted, nit_pred_sorted, c='green', s=10, alpha=0.5, label='predykcje')
    axes[1].set_ylabel("y")
    axes[1].set_title(f"Nozaki-Ishibuchi-Tanaka (R^2={nit_result.r_squared:.4f}, {nit_result.rule_count} reguł)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Sugeno-Yasukawa
    axes[2].plot(x_sorted, y_true_sorted, 'b-', label='y_true (2sin(x)+1)', linewidth=2)
    axes[2].plot(x_sorted, sy_pred_sorted, 'm--', label='y_pred', linewidth=1.5, alpha=0.8)
    axes[2].scatter(x_sorted, sy_pred_sorted, c='purple', s=10, alpha=0.5, label='predykcje')
    axes[2].set_ylabel("y")
    axes[2].set_title(f"Sugeno-Yasukawa (R^2={sy_result.r_squared:.4f}, {sy_result.rule_count} reguł)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    _apply_angle_x_axis(*axes)
    
    plt.tight_layout()
    
    # Zapisz wykres
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nZapisano wykres do: {output_path}")
    plt.close()


def run_single_model_comparison(
    spec: SineExperimentSpec,
    label_count: int,
    sy_rule_count: int,
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
) -> tuple[Any, Any, Any]:
    """Trenuje i ewaluuje modele dla określonej liczby reguł."""
    y_labels = ["NB", "NS", "ZE", "PS", "PB"]
    x_labels = _generate_labels(label_count)
    
    labels_by_variable = {
        "x": x_labels,
        "y": y_labels,
    }
    dataset_spec = build_dataset_spec_from_data(
        data=spec.train_data,
        inputs=["x"],
        outputs=["y"],
        labels_by_variable=labels_by_variable,
    )
    config = ExperimentConfig(
        inputs=dataset_spec.inputs,
        outputs=dataset_spec.outputs,
        fuzzy_sets=dataset_spec.fuzzy_sets,
        universes=dataset_spec.universes,
        sample_size=train_sample_size,
        nit_params={"alpha": 1.0},
        sy_params={"n_rules": sy_rule_count, "eps_sigma": 0.4},
    )
    
    # Trening modeli
    (wm_model, wm_time, _, _, _) = train_wm(spec.train_data, config)
    (nit_model, nit_time, _, _, _) = train_nit(spec.train_data, config)
    (sy_model, sy_time, _, _, _) = train_sy(spec.train_data, config)
    
    # Ewaluacja
    wm_result = evaluate_model(wm_model, "wm", spec.test_data, config, training_time_seconds=wm_time)
    nit_result = evaluate_model(nit_model, "nit", spec.test_data, config, training_time_seconds=nit_time)
    sy_result = evaluate_model(sy_model, "sy", spec.test_data, config, training_time_seconds=sy_time)
    
    return wm_result, nit_result, sy_result


def analyze_nit_alpha_sensitivity(
    spec: SineExperimentSpec,
    alpha_values: list[float],
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Analizuje wpływ parametru alpha na metodę NIT."""
    print("\n" + "="*70)
    print("ANALIZA PARAMETRU ALPHA DLA METODY NIT")
    print("="*70)
    
    y_labels = ["NB", "NS", "ZE", "PS", "PB"]
    x_labels = _generate_labels(17)  # 17 etykiet = 17 reguł
    
    labels_by_variable = {
        "x": x_labels,
        "y": y_labels,
    }
    dataset_spec = build_dataset_spec_from_data(
        data=spec.train_data,
        inputs=["x"],
        outputs=["y"],
        labels_by_variable=labels_by_variable,
    )
    
    results = []
    for alpha in alpha_values:
        print(f"\nAlpha = {alpha}")
        
        config = ExperimentConfig(
            inputs=dataset_spec.inputs,
            outputs=dataset_spec.outputs,
            fuzzy_sets=dataset_spec.fuzzy_sets,
            universes=dataset_spec.universes,
            sample_size=train_sample_size,
            nit_params={"alpha": alpha},
            sy_params={"n_rules": 5, "eps_sigma": 0.4},
        )
        
        (
            nit_model,
            nit_training_time,
            nit_rule_creation_time,
            nit_structure_time,
            nit_learning_time,
        ) = train_nit(spec.train_data, config)
        
        nit_results = evaluate_model(
            nit_model,
            "nit",
            spec.test_data,
            config,
            training_time_seconds=nit_training_time,
            rule_creation_time_seconds=nit_rule_creation_time,
            structure_time_seconds=nit_structure_time,
            learning_time_seconds=nit_learning_time,
        )
        
        results.append({
            "alpha": alpha,
            "result": nit_results,
        })
        
        print(f"  Liczba reguł: {nit_results.rule_count}")
        print(f"  MSE: {nit_results.mse:.6f}, MAE: {nit_results.mae:.6f}, R^2: {nit_results.r_squared:.6f}")
    
    # Tabela wyników
    print("\n" + "="*60)
    print("TABELA: WPŁYW ALPHA NA NIT (17 reguł)")
    print("="*60)
    print(f"{'Alpha':<10} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    for r in results:
        nit_r = r["result"]
        print(f"{r['alpha']:<10} {nit_r.rule_count:<8} {nit_r.mse:<12.6f} {nit_r.mae:<12.6f} {nit_r.rmse:<12.6f} {nit_r.r_squared:<12.6f}")
    
    return results


def analyze_sy_m_sensitivity(
    spec: SineExperimentSpec,
    m_values: list[float],
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Analizuje wpływ parametru m na metodę SY."""
    print("\n" + "="*70)
    print("ANALIZA PARAMETRU M DLA METODY SUGENO-YASUKAWA")
    print("="*70)
    
    y_labels = ["NB", "NS", "ZE", "PS", "PB"]
    x_labels = _generate_labels(17)  # 17 etykiet
    
    labels_by_variable = {
        "x": x_labels,
        "y": y_labels,
    }
    dataset_spec = build_dataset_spec_from_data(
        data=spec.train_data,
        inputs=["x"],
        outputs=["y"],
        labels_by_variable=labels_by_variable,
    )
    
    results = []
    for m_value in m_values:
        print(f"\nm = {m_value}")
        
        config = ExperimentConfig(
            inputs=dataset_spec.inputs,
            outputs=dataset_spec.outputs,
            fuzzy_sets=dataset_spec.fuzzy_sets,
            universes=dataset_spec.universes,
            sample_size=train_sample_size,
            nit_params={"alpha": 1.0},
            sy_params={"n_rules": 5, "eps_sigma": 0.4, "m": m_value},
        )
        
        (
            sy_model,
            sy_training_time,
            sy_rule_creation_time,
            sy_structure_time,
            sy_learning_time,
        ) = train_sy(spec.train_data, config)
        
        sy_results = evaluate_model(
            sy_model,
            "sy",
            spec.test_data,
            config,
            training_time_seconds=sy_training_time,
            rule_creation_time_seconds=sy_rule_creation_time,
            structure_time_seconds=sy_structure_time,
            learning_time_seconds=sy_learning_time,
        )
        
        results.append({
            "m": m_value,
            "result": sy_results,
        })
        
        print(f"  Liczba reguł: {sy_results.rule_count}")
        print(f"  MSE: {sy_results.mse:.6f}, MAE: {sy_results.mae:.6f}, R^2: {sy_results.r_squared:.6f}")
    
    # Tabela wyników
    print("\n" + "="*60)
    print("TABELA: WPŁYW M NA SY (5 reguł)")
    print("="*60)
    print(f"{'m':<10} {'Reguły':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R^2':<12}")
    print("-"*60)
    for r in results:
        sy_r = r["result"]
        print(f"{r['m']:<10} {sy_r.rule_count:<8} {sy_r.mse:<12.6f} {sy_r.mae:<12.6f} {sy_r.rmse:<12.6f} {sy_r.r_squared:<12.6f}")
    
    return results


def run(
    scenario: ScenarioConfig | None = None,
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
    test_sample_size: int = DEFAULT_TEST_SAMPLE_SIZE,
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    seed: int = 42,
) -> None:
    spec = build_experiment_spec(
        train_sample_size=train_sample_size,
        test_sample_size=test_sample_size,
        x_min=x_min,
        x_max=x_max,
        scenario=scenario,
        seed=seed,
    )
    _print_scenario_summary(spec)
    
    # Konfiguracja eksperymentów
    wm_nit_label_counts = [5, 11, 17, 23, 29, 51, 153]
    sy_rule_counts = [2, 3, 5, 8, 10, 22, 66]
    
    all_results = run_single_experiment(
        spec=spec,
        label_counts=wm_nit_label_counts,
        sy_rule_counts=sy_rule_counts,
        train_sample_size=train_sample_size,
    )
    
    # Zapisz wyniki do pliku CSV
    rows = []
    for res in all_results:
        result = res["result"]
        method = res["method"]
        if method in ["wm", "nit"]:
            label_count = res["label_count"]
            rows.append({
                "method": method,
                "label_count": label_count,
                "rule_count": result.rule_count,
                "mse": result.mse,
                "mae": result.mae,
                "rmse": result.rmse,
                "r_squared": result.r_squared,
                "training_time_seconds": result.training_time_seconds,
            })
        else:  # sy
            rule_count = res["rule_count"]
            rows.append({
                "method": method,
                "label_count": "N/A",
                "rule_count": result.rule_count,
                "mse": result.mse,
                "mae": result.mae,
                "rmse": result.rmse,
                "r_squared": result.r_squared,
                "training_time_seconds": result.training_time_seconds,
            })
    
    results_df = pd.DataFrame(rows)
    output_path = "results/summaries/sinus_rule_variation.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nZapisano wyniki do: {output_path}")
    
    # Analiza wrażliwości parametrów
    alpha_values = [0.5, 0.8, 1.0, 2.0, 3.0]
    nit_alpha_results = analyze_nit_alpha_sensitivity(
        spec=spec,
        alpha_values=alpha_values,
        train_sample_size=train_sample_size,
    )
    
    m_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    sy_m_results = analyze_sy_m_sensitivity(
        spec=spec,
        m_values=m_values,
        train_sample_size=train_sample_size,
    )
    
    # Wygeneruj wykresy porównawcze dla różnych ilości reguł
    print("\n" + "="*70)
    print("GENEROWANIE WYKRESÓW PORÓWNAWCZYCH DLA RÓŻNYCH ILOŚCI REGUŁ")
    print("="*70)
    
    # Konfiguracje: (label_count dla WM/NIT, sy_rule_count)
    # - graniczne: 5 i 51 reguł dla WM/NIT, 2 i 22 reguły dla SY
    # - overfitting 3x: 153 reguły dla WM/NIT, 66 reguł dla SY
    # - środkowe: 17 reguł dla WM/NIT, 5 reguł dla SY
    configs = [
        (5, 2),    # minimalne
        (17, 5),   # środkowe
        (51, 22),  # maksymalne
        (153, 66), # overfitting 3x
    ]
    
    config_names = ["min", "srednie", "max", "overfit_3x"]
    
    for i, (label_count, sy_rule_count) in enumerate(configs):
        print(f"\nGenerowanie wykresu {i+1}: {label_count} reguł (WM/NIT), {sy_rule_count} reguł (SY)")
        
        wm_result, nit_result, sy_result = run_single_model_comparison(
            spec=spec,
            label_count=label_count,
            sy_rule_count=sy_rule_count,
            train_sample_size=train_sample_size,
        )
        
        output_path = f"results/plots/sinus_comparison_{config_names[i]}.png"
        plot_comparison_x_y(spec, wm_result, nit_result, sy_result, output_path)
    
    # Uruchom eksperymenty ze zmianami scenariuszy
    print("\n" + "="*70)
    print("URUCHAMIANIE EKSPERYMENTÓW Z ZMIANAMI SCENARIUSZY")
    print("="*70)
    
    scenario_results = run_scenario_variation_experiments(
        spec=spec,
        label_count=17,
        sy_rule_count=5,
        train_sample_size=train_sample_size,
    )
    
    # Generuj wykresy dla każdego scenariusza
    plot_scenario_comparison(spec, scenario_results)


def run_scenario_variation_experiments(
    spec: SineExperimentSpec,
    label_count: int = 17,
    sy_rule_count: int = 5,
    train_sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
    rule_counts: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Uruchamia eksperymenty z pojedynczymi zmianami scenariuszy."""
    # Domyślne konfiguracje: (label_count dla WM/NIT, sy_rule_count)
    rule_counts = rule_counts or [(label_count, sy_rule_count), (29, 10), (51, 22), (153, 66)]
    
    print("\n" + "="*70)
    print("EKSPERYMENTY Z POJEDYNCZYMI ZMIANAMI SCENARIUSZY")
    print("="*70)
    
    all_scenario_results = []

    # Definicje scenariuszy do przetestowania.
    # ScenarioConfig przyjmuje sigma w jednostkach y, dlatego nazwy procentowe
    # przeliczamy względem rozpiętości wartości y dla czystego procesu.
    scenarios = [
        ("baseline", ScenarioConfig()),
        ("outliers_5pct", ScenarioConfig(outlier_ratio=0.05)),
        ("outliers_10pct", ScenarioConfig(outlier_ratio=0.10)),
        ("gaussian_1pct", ScenarioConfig(gaussian_noise_std=_noise_std_from_y_percent(spec.train_data, 1.0))),
        ("gaussian_2pct", ScenarioConfig(gaussian_noise_std=_noise_std_from_y_percent(spec.train_data, 2.0))),
        ("missing_5pct", ScenarioConfig(missing_ratio=0.05)),
        ("missing_10pct", ScenarioConfig(missing_ratio=0.10)),
        ("out_of_range", ScenarioConfig()),
    ]

    for label_count, sy_rule_count in rule_counts:
        print(f"\n{'='*60}")
        print(f"LICZBA REGUŁ: WM/NIT={label_count}, SY={sy_rule_count}")
        print(f"{'='*60}")
        
        y_labels = ["NB", "NS", "ZE", "PS", "PB"]
        x_labels = _generate_labels(label_count)

        labels_by_variable = {
            "x": x_labels,
            "y": y_labels,
        }
        dataset_spec = build_dataset_spec_from_data(
            data=spec.train_data,
            inputs=["x"],
            outputs=["y"],
            labels_by_variable=labels_by_variable,
        )

        results_for_current_rules = []

        for scenario_name, scenario in scenarios:
            print(f"\n--- Scenariusz: {scenario_name} ---")

            if scenario_name == "out_of_range":
                modified_train_data = spec.train_data.copy()
                scenario_test_data = _build_out_of_range_test_data(spec.test_data)
                print(
                    "  Test x poza zakresem uczenia: "
                    f"{scenario_test_data['x'].min():.3f} .. {scenario_test_data['x'].max():.3f} "
                    f"(uczenie: {spec.train_data['x'].min():.3f} .. {spec.train_data['x'].max():.3f})"
                )
            else:
                modified_train_data = apply_training_scenario(
                    data=spec.train_data,
                    scenario=scenario,
                    seed=42,
                    gaussian_noise_columns=["y"],
                    missing_columns=["x", "y"],
                    outlier_columns=["y"],
                )
                scenario_test_data = spec.test_data

            prepared_train_data = prepare_numeric_training_data(
                modified_train_data,
                sort_by="x",
                columns=["x", "y"],
            )

            config = ExperimentConfig(
                inputs=dataset_spec.inputs,
                outputs=dataset_spec.outputs,
                fuzzy_sets=dataset_spec.fuzzy_sets,
                universes=dataset_spec.universes,
                sample_size=train_sample_size,
                nit_params={"alpha": 1.0},
                sy_params={"n_rules": sy_rule_count, "eps_sigma": 0.4},
            )

            (wm_model, wm_time, _, _, _) = train_wm(prepared_train_data, config)
            (nit_model, nit_time, _, _, _) = train_nit(prepared_train_data, config)
            (sy_model, sy_time, _, _, _) = train_sy(prepared_train_data, config)

            wm_result = evaluate_model(wm_model, "wm", scenario_test_data, config, training_time_seconds=wm_time)
            nit_result = evaluate_model(nit_model, "nit", scenario_test_data, config, training_time_seconds=nit_time)
            sy_result = evaluate_model(sy_model, "sy", scenario_test_data, config, training_time_seconds=sy_time)

            result_entry = {
                "label_count": label_count,
                "sy_rule_count": sy_rule_count,
                "scenario": scenario_name,
                "test_data": scenario_test_data,
                "wm": wm_result,
                "nit": nit_result,
                "sy": sy_result,
            }
            results_for_current_rules.append(result_entry)
            all_scenario_results.append(result_entry)

            print(f"  WM: MSE={wm_result.mse:.6f}, R^2={wm_result.r_squared:.4f}, reguł={wm_result.rule_count}")
            print(f"  NIT: MSE={nit_result.mse:.6f}, R^2={nit_result.r_squared:.4f}, reguł={nit_result.rule_count}")
            print(f"  SY: MSE={sy_result.mse:.6f}, R^2={sy_result.r_squared:.4f}, reguł={sy_result.rule_count}")

        print("\n" + "="*70)
        print(f"TABELA PORÓWNAWCZA: SCENARIUSZE (WM/NIT={label_count}, SY={sy_rule_count})")
        print("="*70)
        print(f"{'Scenariusz':<20} {'Metoda':<8} {'MSE':<12} {'MAE':<12} {'R^2':<12}")
        print("-"*70)

        for r in results_for_current_rules:
            sc = r["scenario"]
            print(f"{sc:<20} {'WM':<8} {r['wm'].mse:<12.6f} {r['wm'].mae:<12.6f} {r['wm'].r_squared:<12.6f}")
            print(f"{'':<20} {'NIT':<8} {r['nit'].mse:<12.6f} {r['nit'].mae:<12.6f} {r['nit'].r_squared:<12.6f}")
            print(f"{'':<20} {'SY':<8} {r['sy'].mse:<12.6f} {r['sy'].mae:<12.6f} {r['sy'].r_squared:<12.6f}")

    print("\n" + "="*70)
    print("TABELA PORÓWNAWCZA: WPŁYW ZMIAN SCENARIUSZY")
    print("="*70)
    print(f"{'Reguły':<18} {'Scenariusz':<20} {'Metoda':<8} {'MSE':<12} {'MAE':<12} {'R^2':<12}")
    print("-"*70)

    for r in all_scenario_results:
        rules_text = f"WM/NIT={r['label_count']}, SY={r['sy_rule_count']}"
        sc = r["scenario"]
        print(f"{rules_text:<18} {sc:<20} {'WM':<8} {r['wm'].mse:<12.6f} {r['wm'].mae:<12.6f} {r['wm'].r_squared:<12.6f}")
        print(f"{'':<18} {'':<20} {'NIT':<8} {r['nit'].mse:<12.6f} {r['nit'].mae:<12.6f} {r['nit'].r_squared:<12.6f}")
        print(f"{'':<18} {'':<20} {'SY':<8} {r['sy'].mse:<12.6f} {r['sy'].mae:<12.6f} {r['sy'].r_squared:<12.6f}")

    scenario_rows = []
    for r in all_scenario_results:
        for method in ["wm", "nit", "sy"]:
            result = r[method]
            scenario_rows.append({
                "label_count": r["label_count"] if method in ["wm", "nit"] else "N/A",
                "sy_rule_count": r["sy_rule_count"] if method == "sy" else "N/A",
                "scenario": r["scenario"],
                "method": method,
                "rule_count": result.rule_count,
                "mse": result.mse,
                "mae": result.mae,
                "rmse": result.rmse,
                "r_squared": result.r_squared,
                "training_time_seconds": result.training_time_seconds,
            })

    scenario_output_path = "results/summaries/sinus_scenario_variation.csv"
    pd.DataFrame(scenario_rows).to_csv(scenario_output_path, index=False)
    print(f"\nZapisano wyniki scenariuszy do: {scenario_output_path}")

    return all_scenario_results


def plot_scenario_comparison(
    spec: SineExperimentSpec,
    results: list[dict[str, Any]],
    output_dir: str = "results/plots",
) -> None:
    """Generuje wykresy porównawcze dla każdego scenariusza."""
    import os
    import matplotlib.pyplot as plt
    
    matplotlib_config_dir = os.path.join(os.getcwd(), "results", ".matplotlib", str(os.getpid()))
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_config_dir)

    train_x_min = float(spec.train_data["x"].min())
    train_x_max = float(spec.train_data["x"].max())
    
    for r in results:
        scenario_name = r["scenario"]
        label_count = r["label_count"]
        sy_rule_count = r["sy_rule_count"]
        test_data = r.get("test_data", spec.test_data)

        x_test = test_data["x"].values
        y_true = test_data["y"].values
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_true_sorted = y_true[sort_idx]
        
        wm_pred = np.asarray(r["wm"].predictions["y"], dtype=float).ravel()[sort_idx]
        nit_pred = np.asarray(r["nit"].predictions["y"], dtype=float).ravel()[sort_idx]
        sy_pred = np.asarray(r["sy"].predictions["y"], dtype=float).ravel()[sort_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # WM
        axes[0].plot(x_sorted, y_true_sorted, 'b-', label='y_true', linewidth=2)
        axes[0].plot(x_sorted, wm_pred, 'r--', label='y_pred', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Wang-Mendel (R^2={r['wm'].r_squared:.4f})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # NIT
        axes[1].plot(x_sorted, y_true_sorted, 'b-', label='y_true', linewidth=2)
        axes[1].plot(x_sorted, nit_pred, 'g--', label='y_pred', linewidth=1.5, alpha=0.8)
        axes[1].set_ylabel("y")
        axes[1].set_title(f"Nozaki-Ishibuchi-Tanaka (R^2={r['nit'].r_squared:.4f})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # SY
        axes[2].plot(x_sorted, y_true_sorted, 'b-', label='y_true', linewidth=2)
        axes[2].plot(x_sorted, sy_pred, 'm--', label='y_pred', linewidth=1.5, alpha=0.8)
        axes[2].set_ylabel("y")
        axes[2].set_title(f"Sugeno-Yasukawa (R^2={r['sy'].r_squared:.4f})")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        _apply_angle_x_axis(*axes)

        if scenario_name == "out_of_range":
            for axis in axes:
                axis.axvspan(train_x_min, train_x_max, color="gray", alpha=0.08, label="zakres uczenia")
        
        plt.suptitle(
            f"Scenariusz: {scenario_name} | WM/NIT={label_count}, SY={sy_rule_count}",
            fontsize=14,
        )
        plt.tight_layout()
        
        output_path = (
            f"{output_dir}/sinus_scenario_{scenario_name}_"
            f"wmnit_{label_count}_sy_{sy_rule_count}.png"
        )
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano wykres: {output_path}")
        plt.close()


if __name__ == "__main__":
    run()

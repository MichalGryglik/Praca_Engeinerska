# Praca_Inzynierska

Lekki framework badawczy do generowania, trenowania i porównywania baz reguł rozmytych.

Projekt porównuje trzy podejścia:
- Wang-Mendel (WM),
- Nozaki-Ishibuchi-Tanaka (NIT),
- Sugeno-Yasukawa (SY).

Eksperymenty liczą metryki predykcji, czas trenowania, czas tworzenia reguł, czas budowy struktury oraz czas uczenia. Wyniki są wypisywane w konsoli i zapisywane w katalogu `results/`.

## Szybki Start

1. Utwórz i aktywuj środowisko wirtualne.
2. Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

3. Przygotuj dane TEP w katalogu `data/` zgodnie z sekcją "Dane TEP".
4. Uruchom domyślny eksperyment:

```bash
python main.py
```

Aktualnie `main.py` uruchamia eksperyment TEP:

```python
run_tep_experiment()
```

Pozostałe eksperymenty można uruchomić bezpośrednio:

```bash
python -c "from experiments.example_experiment import run; run()"
python -m experiments.sine_process_experiment
python -c "from experiments.tep_experiment import run; run()"
```

## Eksperymenty

### Przykładowy Zbiór Danych

Plik: `experiments/example_experiment.py`

Eksperyment korzysta z `sandbox/data.csv` i konfiguracji z `sandbox/example_config.py`. Trenuje wszystkie trzy metody, pokazuje pojedyncze predykcje, liczy metryki na danych treningowych i testowych oraz wykonuje analizę wrażliwości parametrów:
- `alpha` dla NIT,
- `m` dla SY.

Wykres porównawczy jest zapisywany do:
- `results/example_experiment_y_pred_vs_y_true.png`

### Proces Sinusoidalny

Plik: `experiments/sine_process_experiment.py`

Eksperyment syntetyczny dla procesu:

```text
y = 2sin(x) + 1
```

Sprawdza różne liczby etykiet/reguł, porównuje metody WM, NIT i SY oraz testuje scenariusze odporności:
- dane bazowe,
- dane odstające,
- szum gaussowski,
- braki danych,
- test poza zakresem treningowym.

Najważniejsze wyniki:
- `results/summaries/sinus_rule_variation.csv`
- `results/summaries/sinus_scenario_variation.csv`
- `results/plots/sinus_*.png`

### Tennessee Eastman Process

Plik: `experiments/tep_experiment.py`

Eksperyment na danych Tennessee Eastman Process (TEP). Obecnie sprawdzane są relacje:
- `xmv_9 -> xmeas_19` (`valve_to_flow`),
- `xmv_8, xmeas_17, xmv_9, xmeas_19 -> xmeas_15` (`stripper_level`),
- `xmv_8 -> xmeas_17` (`stripper_underflow`).

Dodatkowo wykonywana jest autoregresja wyjścia dla każdego eksperymentu. Dla `stripper_level` testowane są dodatkowe warianty liczby przedziałów i reguł SY.

Najważniejsze wyniki:
- `results/summaries/tep_rule_variation.csv`
- `results/summaries/metrics_summary.csv`
- `results/plots/tep_*.png`

## Scenariusze Danych

Moduł `core/scenarios.py` pozwala modyfikować dane treningowe przez:
- dodanie szumu gaussowskiego,
- wstawienie braków danych,
- dodanie danych odstających.

Braki danych są uzupełniane interpolacją liniową przed treningiem. Scenariusze są używane w eksperymentach sinusoidalnym, przykładowym i TEP.

## Najważniejsze Moduły

- `main.py` - główny punkt wejścia projektu.
- `core/data_loader.py` - wczytywanie CSV oraz budowa specyfikacji danych i zbiorów rozmytych.
- `core/membership_functions.py` - funkcje przynależności i wybór najlepiej dopasowanej etykiety.
- `core/experiment_runner.py` - wspólne trenowanie, ewaluacja, tabele wyników i wykresy.
- `core/scenarios.py` - scenariusze zakłóceń danych treningowych.
- `core/results_writer.py` - zapis metryk i predykcji do CSV.
- `core/evaluation/metrics.py` - MSE, MAE, RMSE i R^2.
- `core/rule_generators/wang_mendel.py` - metoda Wang-Mendel.
- `core/rule_generators/nozaki_ishibuchi_tanaka.py` - metoda Nozaki-Ishibuchi-Tanaka.
- `core/rule_generators/sugeno_yasukawa.py` - metoda Sugeno-Yasukawa.
- `experiments/` - gotowe scenariusze eksperymentalne.
- `sandbox/` - dane i konfiguracja prostego eksperymentu demonstracyjnego.

## Dane TEP

Projekt wykorzystuje dane Tennessee Eastman Process.

Źródło:
- https://www.kaggle.com/datasets/afrniomelo/tep-csv

Wymagane pliki:
- `data/TEP_FaultFree_Training.csv`
- `data/TEP_FaultFree_Testing.csv`

Pliki datasetów nie są przechowywane w repozytorium ze względu na rozmiar. Katalog `data/` jest ignorowany przez Git.

## Wyniki

Katalog `results/` zawiera artefakty generowane przez eksperymenty:
- `results/summaries/` - zbiorcze metryki CSV,
- `results/plots/` - wykresy PNG,
- `results/predictions/` - zapisane predykcje, jeżeli dany eksperyment je generuje.

Część wyników i plików CSV jest ignorowana przez `.gitignore`, dlatego przed porównywaniem rezultatów najlepiej uruchomić eksperymenty lokalnie.

## Zależności

Lista zależności znajduje się w `requirements.txt`:
- `numpy`,
- `pandas`,
- `scikit-fuzzy`,
- `scipy`,
- `matplotlib`.

# Praca_Engeinerska

Lekki framework badawczy do budowy i testowania baz reguł rozmytych.

## Szybki obraz projektu
- `main.py` uruchamia pełny przepływ dla przykładowego zestawu danych: wczytuje CSV, buduje reguły Wang–Mendel i Nozaki–Ishibuchi–Tanaka, a następnie wykonuje wnioskowanie.
- `core/membership_functions.py` zawiera podstawy fuzyfikacji (tworzenie funkcji przynależności i wybór najlepszego zbioru dla wartości).
- `core/rule_generators/` zawiera implementacje metod generowania i użycia reguł.
- `examples/example1_config.py` definiuje uniwersa, zbiory i konfigurację wejść/wyjść dla przykładu 1.

## Co warto zrozumieć na początku
1. **Przepływ danych**: dane wejściowe -> fuzyfikacja -> aktywacja reguł -> agregacja -> defuzyfikacja.
2. **Rola konfiguracji**: jakość reguł i predykcji zależy głównie od definicji zbiorów rozmytych i uniwersów.
3. **Różnice między metodami**:
   - Wang–Mendel buduje pojedynczy konsekwent z wagą dla danego antecedentu.
   - Nozaki–Ishibuchi–Tanaka tworzy rozmyty konsekwent (wektor udziałów etykiet wyjścia).
   - Sugeno–Yasukawa wykorzystuje funkcje Sugeno w konsekwentach, co pozwala na modelowanie zależności liniowych lub wielomianowych w przestrzeni wyjść.

## Obecny stan repozytorium
- `core/inference.py`, `core/utils.py`, `examples/example3_config.py`, `examples/example4_config.py` są placeholderami (`pass`).
- `Struktura.txt` to pomocniczy szkic katalogów; aktualny stan plików sprawdzaj przez `rg --files`.

## Dane (dataset)
Projekt wykorzystuje dane Tennessee Eastman Process (TEP).

Źródło:
- https://www.kaggle.com/datasets/afrniomelo/tep-csv

Wymagane pliki:
- `TEP_FaultFree_Training.csv`
- `TEP_FaultFree_Testing.csv`

### Instrukcja przygotowania
1. Pobierz dane z Kaggle przy pomocy powyższego linku.
2. Rozpakuj pliki `.csv`.
3. Umieść je w katalogu:
   - `data/example2_data/`

> Uwaga: pliki datasetów nie są przechowywane w repozytorium ze względu na ich rozmiar.


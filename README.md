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

## Obecny stan repozytorium
- `core/inference.py`, `core/utils.py`, `examples/example2_config.py`, `example3_config.py`, `example4_config.py`, `core/rule_generators/metoda_3.py` są placeholders (`pass`).
- `Struktura.txt` to pomocniczy szkic katalogów; aktualny stan plików sprawdzaj przez `rg --files`.

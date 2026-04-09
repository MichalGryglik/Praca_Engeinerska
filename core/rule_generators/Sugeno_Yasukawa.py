"""Szkielet metody Sugeno–Yasukawa z inicjalizacją klastrów przez scikit-fuzzy.

Plik celowo zawiera funkcje-szkielety (TODO), które będą uzupełniane
w kolejnych krokach implementacji.
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz


def initialize_clusters_with_cmeans(data, inputs, n_rules, m=2.0, error=1e-5, maxiter=1000, seed=42):
    """Inicjalizacja centrów klastrów algorytmem FCM (cmeans).

    Parameters
    ----------
    data : pandas.DataFrame
        Dane uczące.
    inputs : list[str]
        Nazwy kolumn wejściowych.
    n_rules : int
        Docelowa liczba klastrów/reguł.
    m : float
        Parametr rozmycia FCM.
    error : float
        Tolerancja zatrzymania cmeans.
    maxiter : int
        Maks. liczba iteracji cmeans.
    seed : int
        Ziarno losowe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (centers, membership_matrix)
        centers: shape (n_rules, n_inputs)
        membership_matrix: shape (n_rules, n_samples)
    """
    x = data[inputs].to_numpy(dtype=float).T  # cmeans wymaga (features, samples)
    centers, membership_matrix, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=x,
        c=n_rules,
        m=m,
        error=error,
        maxiter=maxiter,
        seed=seed,
    )
    return centers, membership_matrix


def build_initial_rules_from_clusters(centers, inputs, outputs, eps_sigma):
    """Buduje początkową bazę reguł Sugeno na podstawie centrów klastrów.

    Parameters
    ----------
    centers : np.ndarray
        Centra klastrów o kształcie (n_rules, n_inputs).
    inputs : list[str]
        Nazwy zmiennych wejściowych.
    outputs : list[str]
        Nazwy zmiennych wyjściowych (obsługiwane także >1 wyjście).
    eps_sigma : float
        Minimalna wartość odchylenia standardowego dla poprzedników.

    Returns
    -------
    dict
        Słownik reguł, gdzie kluczem jest indeks reguły, a wartością jej parametry.
    """
    # Walidujemy, czy centra mają oczekiwany format 2D: (liczba_reguł, liczba_wejść)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2:
        raise ValueError("Parametr 'centers' musi być tablicą 2D o kształcie (n_rules, n_inputs).")

    # Sprawdzamy zgodność liczby wymiarów centrów z liczbą wejść z konfiguracji
    n_rules, n_inputs = centers.shape
    if n_inputs != len(inputs):
        raise ValueError(
            "Liczba kolumn w 'centers' musi odpowiadać liczbie elementów w 'inputs'."
        )

    # Normalizujemy minimalną szerokość funkcji przynależności,
    # aby uniknąć zerowych lub ujemnych szerokości zbiorów rozmytych
    sigma_value = max(float(eps_sigma), 1e-6)

    # Inicjalizujemy pusty słownik, który będzie przechowywał wszystkie reguły startowe
    rules_dict = {}

    # Dla każdego centrum klastra budujemy jedną regułę Sugeno typu IF-THEN
    for rule_idx in range(n_rules):
        # Pobieramy współrzędne środka aktualnego klastra
        center_vector = centers[rule_idx]

        # Tworzymy parametry poprzedników (część IF):
        # dla każdej zmiennej wejściowej zapisujemy środek i szerokość sigma
        antecedent_params = {}
        for input_idx, input_name in enumerate(inputs):
            antecedent_params[input_name] = {
                "center": float(center_vector[input_idx]),
                "sigma": sigma_value,
            }

        # Tworzymy parametry konsekwentów (część THEN) dla każdej zmiennej wyjściowej:
        # startowo ustawiamy liniowy model Sugeno o zerowych współczynnikach
        # y = a0 + a1*x1 + ... + an*xn
        consequent_params = {}
        for output_name in outputs:
            consequent_params[output_name] = {
                "type": "linear",
                "coefficients": np.zeros(len(inputs) + 1, dtype=float),
            }

        # Składamy pełny opis reguły i zapisujemy ją pod numerycznym identyfikatorem
        rules_dict[rule_idx] = {
            "antecedent": antecedent_params,
            "consequent": consequent_params,
            "weight": 1.0,
            "active": True,
        }

    # Zwracamy kompletną bazę reguł startowych do dalszego strojenia
    return rules_dict


def compute_normalized_firing_strengths(data, inputs, rules_dict, fuzzy_sets, universes):
    """Oblicza i normalizuje siły aktywacji reguł dla każdej próbki.

    W aktualnym szkielecie aktywacja poprzedników jest liczona na podstawie
    parametrów gaussowskich zapisanych w każdej regule (`center`, `sigma`).
    Parametry `fuzzy_sets` i `universes` zostają w sygnaturze dla spójności
    interfejsu i przyszłej rozbudowy.

    Returns
    -------
    dict
        {
            "rule_ids": list[int],
            "raw": np.ndarray shape (n_samples, n_rules),
            "normalized": np.ndarray shape (n_samples, n_rules)
        }
    """
    # Zachowujemy parametry dla kompatybilności z dalszą implementacją pipeline'u
    _ = (fuzzy_sets, universes)

    # Jeżeli nie ma reguł, zwracamy puste macierze o odpowiednim kształcie
    if not rules_dict:
        n_samples = len(data)
        empty = np.zeros((n_samples, 0), dtype=float)
        return {"rule_ids": [], "raw": empty, "normalized": empty}

    # Pobieramy dane wejściowe w postaci macierzy numerycznej: (n_samples, n_inputs)
    x = data[inputs].to_numpy(dtype=float)
    n_samples = x.shape[0]

    # Ustalamy kolejność reguł, aby mapowanie kolumn macierzy było jednoznaczne
    rule_ids = sorted(rules_dict.keys())
    n_rules = len(rule_ids)

    # Inicjalizujemy macierz surowych sił aktywacji reguł
    raw_strengths = np.zeros((n_samples, n_rules), dtype=float)

    # Liczymy aktywację każdej reguły dla każdej próbki (T-norma jako iloczyn)
    for rule_col, rule_id in enumerate(rule_ids):
        rule_data = rules_dict[rule_id]
        antecedent = rule_data.get("antecedent", {})

        # Startujemy od jedynek, a następnie mnożymy przynależności gaussowskie
        rule_firing = np.ones(n_samples, dtype=float)

        for input_idx, input_name in enumerate(inputs):
            # Pobieramy parametry poprzednika; gdy brakuje, używamy bezpiecznych domyślnych
            params = antecedent.get(input_name, {})
            center = float(params.get("center", 0.0))
            sigma = max(float(params.get("sigma", 1.0)), 1e-6)

            # Obliczamy wartość funkcji gaussowskiej dla całej kolumny próbek
            # mu(x) = exp(-0.5 * ((x - c) / sigma)^2)
            z = (x[:, input_idx] - center) / sigma
            mu = np.exp(-0.5 * np.square(z))

            # Aktualizujemy siłę odpalenia reguły (iloczyn po wszystkich wejściach)
            rule_firing *= mu

        raw_strengths[:, rule_col] = rule_firing

    # Normalizujemy siły reguł per próbka: w_bar_i = w_i / sum_j(w_j)
    normalized_strengths = np.zeros_like(raw_strengths)
    row_sums = raw_strengths.sum(axis=1, keepdims=True)

    # Dla próbek z dodatnią sumą wykonujemy standardową normalizację
    valid_rows = row_sums.squeeze(axis=1) > 0.0
    normalized_strengths[valid_rows] = raw_strengths[valid_rows] / row_sums[valid_rows]

    # Dla przypadków degenerate (suma = 0) rozkładamy wagę równomiernie między reguły
    if np.any(~valid_rows) and n_rules > 0:
        normalized_strengths[~valid_rows] = 1.0 / n_rules

    # Zwracamy komplet informacji (surowe i znormalizowane aktywacje)
    return {
        "rule_ids": rule_ids,
        "raw": raw_strengths,
        "normalized": normalized_strengths,
    }


def update_consequents_ls_wls(data, inputs, outputs, rules_dict, normalized_strengths):
    """Aktualizuje parametry części THEN metodą ważonych najmniejszych kwadratów.

    Dla każdej reguły i każdego wyjścia dopasowywany jest liniowy model Sugeno:
    y = a0 + a1*x1 + ... + an*xn,
    gdzie wagi próbek pochodzą z `normalized_strengths`.

    Funkcja aktualizuje `rules_dict` in-place i dodatkowo zwraca ten sam obiekt
    dla wygody łańcuchowania.
    """
    # Walidujemy format wejścia z aktywacjami znormalizowanymi
    if "rule_ids" not in normalized_strengths or "normalized" not in normalized_strengths:
        raise ValueError("normalized_strengths musi zawierać klucze 'rule_ids' i 'normalized'.")

    rule_ids = normalized_strengths["rule_ids"]
    w_norm = np.asarray(normalized_strengths["normalized"], dtype=float)

    # Gdy nie ma reguł albo próbek, nie ma czego aktualizować
    if len(rule_ids) == 0 or len(data) == 0:
        return rules_dict

    # Budujemy macierz regresji z wyrazem wolnym: [1, x1, ..., xn]
    x_raw = data[inputs].to_numpy(dtype=float)
    n_samples = x_raw.shape[0]
    x_design = np.hstack([np.ones((n_samples, 1), dtype=float), x_raw])

    # Dla każdego wyjścia wykonujemy osobne dopasowanie współczynników
    for output_name in outputs:
        y = data[output_name].to_numpy(dtype=float)

        # Iterujemy po regułach zgodnie z kolejnością kolumn w macierzy wag
        for rule_col, rule_id in enumerate(rule_ids):
            # Jeżeli reguła zniknęła (np. po adaptacji), pomijamy ją bez błędu
            if rule_id not in rules_dict:
                continue

            weights = w_norm[:, rule_col]

            # Jeżeli łączna waga jest praktycznie zerowa, zostawiamy stare współczynniki
            if np.sum(weights) <= 1e-12:
                continue

            # Tworzymy postać WLS przez przemnożenie przez sqrt(w):
            # (sqrt(W)X) beta = sqrt(W)y
            sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
            x_weighted = x_design * sqrt_w[:, None]
            y_weighted = y * sqrt_w

            # Rozwiązujemy układ najmniejszych kwadratów;
            # lstsq jest stabilne numerycznie także dla układów osobliwych
            coeffs, _, _, _ = np.linalg.lstsq(x_weighted, y_weighted, rcond=None)

            # Upewniamy się, że istnieje struktura konsekwentu dla danego wyjścia
            consequent = rules_dict[rule_id].setdefault("consequent", {})
            consequent_entry = consequent.setdefault(
                output_name,
                {"type": "linear", "coefficients": np.zeros(len(inputs) + 1, dtype=float)},
            )

            # Aktualizujemy typ i współczynniki modelu liniowego Sugeno
            consequent_entry["type"] = "linear"
            consequent_entry["coefficients"] = coeffs.astype(float)

    # Zwracamy zaktualizowany słownik reguł
    return rules_dict


def update_antecedents(data, inputs, rules_dict, normalized_strengths, eps_sigma):
    """Aktualizuje parametry części IF (center, sigma) na podstawie wag reguł.

    Implementacja używa znormalizowanych sił aktywacji jako wag i dla każdej
    reguły wyznacza ważoną średnią (nowe center) oraz ważone odchylenie
    standardowe (nowe sigma) dla każdej zmiennej wejściowej.

    Parametr `eps_sigma` wymusza dolne ograniczenie na szerokość gaussów,
    aby uniknąć zapadania się funkcji przynależności.
    """
    # Sprawdzamy, czy struktura wag ma wymagane pola z poprzedniego kroku pipeline'u
    if "rule_ids" not in normalized_strengths or "normalized" not in normalized_strengths:
        raise ValueError("normalized_strengths musi zawierać klucze 'rule_ids' i 'normalized'.")

    # Pobieramy mapowanie reguł do kolumn oraz samą macierz wag (n_samples, n_rules)
    rule_ids = normalized_strengths["rule_ids"]
    w_norm = np.asarray(normalized_strengths["normalized"], dtype=float)

    # Jeżeli nie ma reguł albo próbek, kończymy bez zmian
    if len(rule_ids) == 0 or len(data) == 0:
        return rules_dict

    # Przygotowujemy macierz wejść do obliczeń wektorowych
    x = data[inputs].to_numpy(dtype=float)

    # Minimalna dopuszczalna sigma (zabezpieczenie przed zerem/ujemną wartością)
    sigma_floor = max(float(eps_sigma), 1e-6)

    # Dla każdej reguły aktualizujemy parametry antecedentu osobno dla każdego wejścia
    for rule_col, rule_id in enumerate(rule_ids):
        # Jeżeli reguła została wcześniej usunięta, pomijamy ją
        if rule_id not in rules_dict:
            continue

        # Pobieramy wektor wag przypisanych do tej reguły dla wszystkich próbek
        weights = np.clip(w_norm[:, rule_col], 0.0, None)
        weight_sum = float(np.sum(weights))

        # Jeżeli suma wag jest zbyt mała, nie aktualizujemy tej reguły
        if weight_sum <= 1e-12:
            continue

        # Upewniamy się, że słownik antecedentu istnieje
        antecedent = rules_dict[rule_id].setdefault("antecedent", {})

        for input_idx, input_name in enumerate(inputs):
            # Pobieramy wszystkie wartości danej zmiennej wejściowej
            x_col = x[:, input_idx]

            # Liczymy ważoną średnią -> nowe center
            center_new = float(np.sum(weights * x_col) / weight_sum)

            # Liczymy ważoną wariancję i sigma -> nowe rozmycie gaussa
            variance_new = float(np.sum(weights * np.square(x_col - center_new)) / weight_sum)
            sigma_new = float(np.sqrt(max(variance_new, 0.0)))

            # Wymuszamy dolne ograniczenie sigma, by zachować stabilność obliczeń
            sigma_new = max(sigma_new, sigma_floor)

            # Aktualizujemy strukturę parametru antecedentu dla tej zmiennej
            antecedent_entry = antecedent.setdefault(
                input_name,
                {"center": center_new, "sigma": sigma_new},
            )
            antecedent_entry["center"] = center_new
            antecedent_entry["sigma"] = sigma_new

    # Zwracamy zaktualizowane reguły (aktualizacja wykonana in-place)
    return rules_dict


def compute_objective_mse(data, inputs, outputs, rules_dict, fuzzy_sets, universes):
    """TODO: Oblicz funkcję celu (MSE)."""
    _ = (data, inputs, outputs, rules_dict, fuzzy_sets, universes)
    pass


def adapt_rule_structure(rules_dict, normalized_strengths, local_errors):
    """TODO: Adaptacja struktury reguł (usuń/dodaj/scal)."""
    _ = (rules_dict, normalized_strengths, local_errors)
    pass


def estimate_local_errors(data, inputs, outputs, rules_dict, fuzzy_sets, universes):
    """TODO: Wyznacz błędy lokalne na potrzeby adaptacji struktury."""
    _ = (data, inputs, outputs, rules_dict, fuzzy_sets, universes)
    pass


def print_rules(rules_dict):
    for rule_id, rule in rules_dict.items():
        print(f"\nRule {rule_id}")

        for var, params in rule["antecedent"].items():
            print(f"  IF {var} ≈ {params['center']:.2f} (σ={params['sigma']:.2f})")

        for out, cons in rule["consequent"].items():
            coeffs = cons["coefficients"]
            print(f"  THEN {out} = {coeffs}")

def sugeno_yasukawa_pseudocode(
    data,
    inputs,
    outputs,
    fuzzy_sets,
    universes,
    n_rules,
    eps_j,
    eps_sigma,
    max_iter,
):
    """Szkielet pętli uczenia metody Sugeno–Yasukawa."""
    centers, _ = initialize_clusters_with_cmeans(data, inputs, n_rules)
    rules_dict = build_initial_rules_from_clusters(centers, inputs, outputs, eps_sigma)

    previous_error = float("inf")

    for _iteration in range(1, max_iter + 1):
        normalized_strengths = compute_normalized_firing_strengths(
            data, inputs, rules_dict, fuzzy_sets, universes
        )

        update_consequents_ls_wls(data, inputs, outputs, rules_dict, normalized_strengths)
        update_antecedents(data, inputs, rules_dict, normalized_strengths, eps_sigma)

        current_error = compute_objective_mse(
            data, inputs, outputs, rules_dict, fuzzy_sets, universes
        )

        local_errors = estimate_local_errors(
            data, inputs, outputs, rules_dict, fuzzy_sets, universes
        )
        adapt_rule_structure(rules_dict, normalized_strengths, local_errors)

        if abs(previous_error - current_error) < eps_j:
            break

        previous_error = current_error

    return rules_dict

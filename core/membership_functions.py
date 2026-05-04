# membership_functions.py
# Tu będą funkcje generacji MF
# core/membership_functions.py
import numpy as np
import skfuzzy as fuzz


def clip_to_universe(value, universe):
    """Ogranicza wartosc do zakresu zdefiniowanego uniwersum."""
    universe = np.asarray(universe, dtype=float)
    return float(np.clip(value, np.nanmin(universe), np.nanmax(universe)))


def create_sets(universe, set_params):
    """
    Tworzy zbiory rozmyte (trójkątne) dla podanego uniwersum.

    Parameters
    ----------
    universe : np.array
        Uniwersum wartości wejścia/wyjścia
    set_params : dict
        Słownik parametrów zbiorów, np. {"S1": [0,0,5], "CE": [0,5,10]}

    Returns
    -------
    dict
        Słownik nazw zbiorów -> funkcje przynależności
    """
    return {name: fuzz.trimf(universe, params) for name, params in set_params.items()}

def find_best_membership(value, universe, fuzzy_sets):
    """
    Znajduje zbiór rozmyty o największej przynależności dla podanej wartości.

    Returns
    -------
    tuple
        (nazwa zbioru, wartość przynależności)
    """
    bounded_value = clip_to_universe(value, universe)
    memberships = {name: fuzz.interp_membership(universe, mf, bounded_value)
                   for name, mf in fuzzy_sets.items()}
    best_name = max(memberships, key=memberships.get)
    best_value = memberships[best_name]
    return best_name, best_value

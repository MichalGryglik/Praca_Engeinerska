# membership_functions.py
# Tu będą funkcje generacji MF
# core/membership_functions.py
import numpy as np
import skfuzzy as fuzz

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
    memberships = {name: fuzz.interp_membership(universe, mf, value)
                   for name, mf in fuzzy_sets.items()}
    best_name = max(memberships, key=memberships.get)
    best_value = memberships[best_name]
    return best_name, best_value
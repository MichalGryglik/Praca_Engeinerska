import pandas as pd

# === 1. Wczytanie danych ===
tep_data = pd.read_csv("data/example2_data/TEP_FaultFree_Testing.csv")

print("Pierwsze 4 wiersze:")
print(tep_data.head(4).to_string())

# === 2. Usuwamy kolumny techniczne ===
drop_cols = ['faultNumber', 'simulationRun', 'sample']
numeric_cols = [col for col in tep_data.columns if col not in drop_cols]

data = tep_data[numeric_cols]

# === 3. Korelacja ===
corr = data.corr()

# === 4. Wybierz potencjalne wyjście ===
target = numeric_cols[0]  # tymczasowo pierwsza kolumna

print(f"\nAnaliza korelacji dla: {target}")
print(corr[target].sort_values(ascending=False))

# === 5. TOP powiązane zmienne ===
top_features = corr[target].abs().sort_values(ascending=False)

# usuwamy samą siebie
top_features = top_features.drop(target)

print("\nNajbardziej powiązane zmienne:")
print(top_features.head(5))

# === 6. Propozycja input/output ===
inputs = list(top_features.head(2).index)
output = target

print("\nPropozycja do modelu:")
print("inputs =", inputs)
print("output =", output)

print("\nKorelacje z xmeas_2:")
print(corr["xmeas_2"].sort_values(ascending=False))
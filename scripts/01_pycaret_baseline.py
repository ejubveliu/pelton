import pandas as pd
from pycaret.regression import setup, compare_models, pull

# 1. Daten laden
df = pd.read_csv("data_anon.csv")
print("Dataset shape:", df.shape)

# 2. Setup
s = setup(
    data=df,
    target="y",
    session_id=42,
    train_size=0.8,
    verbose=False
)

# 3. AutoML
best = compare_models(sort="MAE")

# 4. Ergebnisse
results = pull()
print(results.head())
print("\nBest model:", best)

# 5. Kleine Summary-Datei schreiben
best_row = results.iloc[0]  # erste Zeile = bestes Modell
summary = pd.DataFrame([{
    "tool": "pycaret",
    "best_model": best_row["Model"],
    "MAE": best_row["MAE"],
}])

summary.to_csv("summary_pycaret.csv", index=False)
print("\n✔ summary_pycaret.csv geschrieben")
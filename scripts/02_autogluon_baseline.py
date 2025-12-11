import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

# 1. Daten laden
df = pd.read_csv("data_anon.csv")
print("Dataset shape:", df.shape)

label = "y"

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

# 2. Nur „sichere“ Modelle benutzen – NN_TORCH deaktivieren
hyperparameters = {
    "NN_TORCH": None,   # Neural Nets komplett ausschalten
    # Rest lässt AutoGluon standardmäßig zu (GBM, RF, XGBoost* falls vorhanden, etc.)
}

predictor = TabularPredictor(
    label=label,
    eval_metric="mae"
).fit(
    train,
    hyperparameters=hyperparameters,
    presets="medium_quality_faster_train",
    num_cpus=4
)

# 3. Evaluate
perf = predictor.evaluate(test)
print("\nAutoGluon performance:", perf)

lb = predictor.leaderboard(test, silent=True)
print(lb.head())

# 4. Summary-Datei
best_row = lb.iloc[0]
score_col = "score_val" if "score_val" in best_row.index else "score"
summary = pd.DataFrame([{
    "tool": "autogluon",
    "best_model": best_row["model"],
    "MAE": best_row[score_col],
}])

summary.to_csv("summary_autogluon.csv", index=False)
print("\n✔ summary_autogluon.csv geschrieben")
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Daten laden
df = pd.read_csv("data_anon.csv")
print("Dataset shape:", df.shape)

X = df.drop(columns=["y"])
y = df["y"]

# 80/20 Split wie bei den anderen Tools
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. AutoML konfigurieren
automl = AutoML()

automl_settings = {
    "time_budget": 60,           # Sekunden Gesamtzeit (kannst du erhöhen)
    "metric": "mae",             # gleiche Metrik wie bei PyCaret
    "task": "regression",
    "log_file_name": "flaml.log",
    "seed": 42,
}

# 3. Fit
automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

print("\nBest FLAML model:", automl.best_estimator)
print("Best config:", automl.best_config)
print("Best MAE (val):", automl.best_loss)

# 4. Test-Performance
y_pred = automl.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred)
print("Test MAE:", mae_test)

# 5. Summary-CSV im gleichen Format wie PyCaret/H2O
summary = pd.DataFrame([{
    "tool": "flaml",
    "best_model": automl.best_estimator,
    "MAE": mae_test,
}])
summary.to_csv("summary_flaml.csv", index=False)
print("\n✔ summary_flaml.csv geschrieben")
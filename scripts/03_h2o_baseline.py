import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# 1. Daten laden
pdf = pd.read_csv("data_anon.csv")
print("Dataset shape:", pdf.shape)

h2o.init()
df = h2o.H2OFrame(pdf)

y = "y"
x = [c for c in df.columns if c != y]

# gleicher Split wie bei den anderen Tools
train, test = df.split_frame(ratios=[0.8], seed=42)

# 2. AutoML
aml = H2OAutoML(
    max_models=20,
    seed=42,
    sort_metric="MAE"
)
aml.train(x=x, y=y, training_frame=train)

lb = aml.leaderboard
print(lb.head(rows=10))

# 3. MAE auf Testset
perf = aml.leader.model_performance(test)
mae = perf.mae()
print("H2O MAE:", mae)

# 4. Summary-Datei
# im H2O-Leaderboard heißt das Modell id meist "model_id"
best_model_id = lb[0, "model_id"]

summary = pd.DataFrame([{
    "tool": "h2o",
    "best_model": best_model_id,
    "MAE": mae,
}])

summary.to_csv("summary_h2o.csv", index=False)
print("\n✔ summary_h2o.csv geschrieben")

h2o.shutdown(prompt=False)
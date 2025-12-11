import pandas as pd

files = [
    "summary_pycaret.csv",
    "summary_autogluon.csv",
    "summary_h2o.csv",
]

dfs = [pd.read_csv(f) for f in files]
summary = pd.concat(dfs, ignore_index=True)

print("\n=== Final Tool Comparison ===")
print(summary)

summary.to_csv("results_summary.csv", index=False)
print("\n✔ results_summary.csv geschrieben")
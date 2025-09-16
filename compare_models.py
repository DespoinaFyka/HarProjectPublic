import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

base_path = os.path.join("results")

models = {
    "LogReg": "logistic_regression/logreg_metrics_all_folds.csv",
    "HMM": "hmm/hmm_metrics_all_folds.csv",
    "CNN": "cnn/cnn_metrics_all_folds.csv",
    "LSTM": "lstm/lstm_metrics_all_folds.csv",
    "CNN+LSTM": "cnn_lstm/cnn_lstm_metrics_all_folds.csv"
}

all_results = []

for model_name, rel_path in models.items():
    full_path = os.path.join(base_path, rel_path)
    df = pd.read_csv(full_path)
    df['Model'] = model_name
    all_results.append(df)

df_all = pd.concat(all_results)

# === Summary Statistics ===
summary = df_all.groupby("Model").agg({
    "Accuracy": "mean",
    "Balanced Accuracy": "mean",
    "Precision": "mean",
    "Recall": "mean",
    "F1-Score": "mean"
}).round(3)

summary = summary.reset_index()
summary.to_csv(os.path.join(base_path, "comparison_summary.csv"), index=False)

print(summary)

# === Bar Plots ===
df = pd.read_csv("results/comparison_summary.csv")

plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["F1-Score"], color='skyblue')
plt.title("F1-Score ανά μοντέλο", fontsize=14)
plt.xlabel("Μοντέλο", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(df["F1-Score"]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
plt.tight_layout()

# Save the plot
plt.savefig("results/f1_score_comparison.png", dpi=300)

# === All Metrics Comparison Bar ===
metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(df["Model"]))
width = 0.15

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, df[metric], width, label=metric)

plt.xticks(x + width*2, df["Model"])
plt.ylabel("Τιμή Μετρικής")
plt.title("Σύγκριση Μετρικών ανά Μοντέλο")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("results/all_metrics_comparison.png", dpi=300)
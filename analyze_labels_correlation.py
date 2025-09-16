import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.labels_utils import new_label_name

# === CONFIG ===
WORKING_DIR = os.getcwd()
MODEL = "lstm"
RESULTS_PATH = os.path.join(WORKING_DIR, "results", MODEL)
LABELS = ['FIX_walking', 'SITTING', 'LOC_home', 'OR_outside', 'OR_indoors']
pretty_label_names = [new_label_name(l) for l in LABELS]

df_true = []
df_pred = []

print("Loading predictions and ground truth per label...")
for label, pretty_name in zip(LABELS, pretty_label_names):
    label_folder = os.path.join(RESULTS_PATH, pretty_name)

    pred_path = os.path.join(label_folder, f"{pretty_name}_Y_pred_all.npy")
    true_path = os.path.join(label_folder, f"{pretty_name}_Y_true_all.npy")

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"[SKIP] {pretty_name}: missing prediction or true files.")
        continue

    y_pred = np.load(pred_path).astype(int)
    y_true = np.load(true_path).astype(int)

    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    idx = np.arange(min_len)
    df_true.append(pd.Series(y_true, index=idx, name=pretty_name))
    df_pred.append(pd.Series(y_pred, index=idx, name=pretty_name))

# === Create DataFrames
df_true = pd.concat(df_true, axis=1).fillna(0).astype(int)
df_pred = pd.concat(df_pred, axis=1).fillna(0).astype(int)

# === Compute Jaccard similarity for each matrix
def compute_jaccard_matrix(df):
    A = df.values.T  # shape [L, N]
    intersection = A @ A.T
    row_sums = A.sum(axis=1).reshape(-1, 1)
    col_sums = A.sum(axis=1).reshape(1, -1)
    union = row_sums + col_sums - intersection
    return np.where(union > 0, intersection / union, 0)

jaccard_true = compute_jaccard_matrix(df_true)
jaccard_pred = compute_jaccard_matrix(df_pred)

# === Plot ground truth heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(jaccard_true, cmap="Greens", xticklabels=df_true.columns, yticklabels=df_true.columns, annot=True, fmt=".2f")
plt.title("Jaccard Similarity Between TRUE Labels")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, f"{MODEL}_jaccard_similarity_true_labels.png"), dpi=300)

# === Plot predicted heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(jaccard_pred, cmap="Blues", xticklabels=df_pred.columns, yticklabels=df_pred.columns, annot=True, fmt=".2f")
plt.title("Jaccard Similarity Between PREDICTED Labels")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, f"{MODEL}_jaccard_similarity_pred_labels.png"), dpi=300)

print("Saved both true and predicted heatmaps.")

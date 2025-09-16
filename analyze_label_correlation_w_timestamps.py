import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.labels_utils import new_label_name

# === CONFIG ===
WORKING_DIR = os.getcwd()
RESULTS_PATH = os.path.join(WORKING_DIR, "results")
LABELS = ['FIX_walking', 'SITTING', 'LOC_home', 'OR_outside', 'OR_indoors']
pretty_label_names = [new_label_name(l) for l in LABELS]

label_series_list = []

print("Loading predictions per label...")
for label, pretty_name in zip(LABELS, pretty_label_names):
    label_folder = os.path.join(RESULTS_PATH, pretty_name)

    pred_path = os.path.join(label_folder, f"{pretty_name}_Y_pred_all.npy")
    time_path = os.path.join(label_folder, f"{pretty_name}_timestamps_all.npy")

    if not os.path.exists(pred_path) or not os.path.exists(time_path):
        print(f"[SKIP] {pretty_name}: missing predictions or timestamps.")
        continue

    y_pred = np.load(pred_path).astype(int)
    timestamps = np.load(time_path)

    if len(y_pred) != len(timestamps):
        print(f"[SKIP] {pretty_name}: length mismatch.")
        continue

    # Use index instead of timestamp (no duplicates)
    index = np.arange(len(y_pred))
    s = pd.Series(y_pred, index=index, name=pretty_name)
    label_series_list.append(s)

if len(label_series_list) < 2:
    print("Not enough valid labels to compare.")
    exit(1)

# Aligned based on the index
print("Aligning predictions...")
df = pd.concat(label_series_list, axis=1).fillna(0).astype(int)

# Compute Jaccard Similarity
print("Computing Jaccard similarity matrix...")
A = df.values.T  # shape [L, N]
intersection = A @ A.T

row_sums = A.sum(axis=1).reshape(-1, 1)  # [L, 1]
col_sums = A.sum(axis=1).reshape(1, -1)  # [1, L]
union = row_sums + col_sums - intersection

jaccard = np.where(union > 0, intersection / union, 0)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(jaccard, cmap="YlGnBu", xticklabels=df.columns, yticklabels=df.columns, annot=True, fmt=".2f")
plt.title("Jaccard Similarity Between Predicted Labels (Aligned by Position)")
plt.tight_layout()

output_path = os.path.join(RESULTS_PATH, "jaccard_similarity_heatmap.png")
plt.savefig(output_path, dpi=300)
print(f"Saved heatmap to: {output_path}")

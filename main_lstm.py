# main_lstm.py

import os
import numpy as np
import pandas as pd
from collections import namedtuple

from models.lstm_model import test_lstm_model, train_lstm_model
from utils.data_loading import read_user_data
from utils.load_train_test_uuids import load_cv_folds
from utils.labels_utils import get_sensor_names_from_features, new_label_name

# --- Configuration ---
WORKING_DIR = os.getcwd()
DATA_FOLDER = "ExtraSensory.per_uuid_features_labels"
FOLDS_FOLDER = "cv_5_folds"
RESULTS_FOLDER = os.path.join(WORKING_DIR, "results", "lstm")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# TARGET_LABEL = 'FIX_walking'
SENSORS_TO_USE = ['Acc', 'WAcc']
LABELS = [
    'FIX_walking', 'SITTING', 'LOC_home', 'OR_outside', 'OR_indoors',
]

UserData = namedtuple("UserData", ["uuid", "X", "Y", "M", "timestamps", "feature_names", "label_names"])

# --- Load all user data ---
USERS_DATA = []
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".csv.gz"):
        uuid = filename.split(".")[0]
        X, Y, M, timestamps, feature_names, label_names = read_user_data(DATA_FOLDER, filename)
        USERS_DATA.append(UserData(uuid, X, Y, M, timestamps, feature_names, label_names))

# --- Load CV folds ---
folds = load_cv_folds(FOLDS_FOLDER)


for TARGET_LABEL in LABELS:
    print(f"Label: {TARGET_LABEL}")
    OUTPUT_LABEL = new_label_name(TARGET_LABEL)
    # --- Per-fold evaluation ---
    fold_results = []
    all_preds = []
    all_trues = []
    all_timestamps = []

    for fold_idx in range(5):
        print(f"\n===== Fold {fold_idx} =====")

        train_uuids = folds[f"fold_{fold_idx}_train_android_uuids.txt"] + folds[f"fold_{fold_idx}_train_iphone_uuids.txt"]
        test_uuids  = folds[f"fold_{fold_idx}_test_android_uuids.txt"]  + folds[f"fold_{fold_idx}_test_iphone_uuids.txt"]

        train_users = [u for u in USERS_DATA if u.uuid in train_uuids]
        test_users  = [u for u in USERS_DATA if u.uuid in test_uuids]

        # Concatenate training data
        X_train = np.vstack([u.X for u in train_users])
        Y_train = np.vstack([u.Y for u in train_users])
        M_train = np.vstack([u.M for u in train_users])
        timestamps_train = np.hstack([u.timestamps for u in train_users])
        feat_sensor_names_train = get_sensor_names_from_features(train_users[0].feature_names)

        model = train_lstm_model(
            X_train, Y_train, M_train, timestamps_train,
            feat_sensor_names_train, train_users[0].label_names,
            SENSORS_TO_USE, TARGET_LABEL
        )

        # Evaluate on test set
        X_test = np.vstack([u.X for u in test_users])
        Y_test = np.vstack([u.Y for u in test_users])
        M_test = np.vstack([u.M for u in test_users])
        timestamps_test = np.hstack([u.timestamps for u in test_users])
        feat_sensor_names_test = get_sensor_names_from_features(test_users[0].feature_names)

        metrics = test_lstm_model(
            X_test, Y_test, M_test, timestamps_test,
            feat_sensor_names_test, test_users[0].label_names,
            model
        )

        # Collect for per-fold metrics
        fold_results.append({
            'Fold': fold_idx,
            'Label': OUTPUT_LABEL,
            **{k: v for k, v in metrics.items() if k not in ['y_true', 'y_pred', 'y_timestamps']}
        })

        # Collect for correlation analysis
        all_trues.append(metrics['y_true'])
        all_preds.append(metrics['y_pred'])
        all_timestamps.append(metrics['y_timestamps'])

    # --- Save fold metrics ---
    label_results_folder = os.path.join(RESULTS_FOLDER, f"{OUTPUT_LABEL}")
    os.makedirs(label_results_folder, exist_ok=True)
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(label_results_folder, f"{OUTPUT_LABEL}_metrics_folds.csv"), index=False, float_format="%.4f")

    # --- Save all predictions and ground truth (binary) ---
    Y_true_all = np.concatenate(all_trues)
    Y_pred_all = np.concatenate(all_preds)
    timestamps_all = np.concatenate(all_timestamps)

    np.save(os.path.join(label_results_folder, f"{OUTPUT_LABEL}_Y_true_all.npy"), Y_true_all)
    np.save(os.path.join(label_results_folder, f"{OUTPUT_LABEL}_Y_pred_all.npy"), Y_pred_all)
    np.save(os.path.join(label_results_folder, f"{OUTPUT_LABEL}_timestamps_all.npy"), timestamps_all)

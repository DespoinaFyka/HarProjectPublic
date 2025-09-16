import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from utils.data_processing import extract_sliding_windows, project_features_to_selected_sensors, estimate_standardization_params, standardize_features
from utils.labels_utils import get_label_pretty_name
from utils.evaluation_metrics import compute_extended_metrics


def train_cnn_lstm_model(X, Y, M, timestamps, feat_sensor_names, label_names, sensors_to_use, target_label,
                          window_size=20, step_size=10):
    # Choose sensors to use
    X = project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use)
    mean_vec, std_vec = estimate_standardization_params(X)
    X = standardize_features(X, mean_vec, std_vec)

    # Find index of label
    label_ind = label_names.index(target_label)

    # Extract sliding windows with at least one positive label
    X_win, y_win, _ = extract_sliding_windows(X, Y, M, timestamps, label_ind=label_ind,
                                              window_size=window_size, step_size=step_size,
                                              label_names=label_names, rule='at_least_one')

    # Handle NaN
    X_win[np.isnan(X_win)] = 0.

    # Imbalance weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_win)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    print(f"== Training CNN+LSTM with {X_win.shape[0]} windows for label '{get_label_pretty_name(target_label)}'")
    print("Class distribution:", dict(zip(*np.unique(y_win, return_counts=True))))
    
    model = Sequential([
        Input(shape=(X_win.shape[1], X_win.shape[2])),
        Conv1D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Conv1D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_win, y_win,
              epochs=30,
              batch_size=64,
              verbose=2,
              validation_split=0.2,
              class_weight=class_weights_dict,
              callbacks=[early_stop])

    return {
        'model': model,
        'mean_vec': mean_vec,
        'std_vec': std_vec,
        'sensors_to_use': sensors_to_use,
        'target_label': target_label
    }


def test_cnn_lstm_model(X, Y, M, timestamps, feat_sensor_names, label_names, model,
                         window_size=20, step_size=10):
    X = project_features_to_selected_sensors(X, feat_sensor_names, model['sensors_to_use'])
    X = standardize_features(X, model['mean_vec'], model['std_vec'])

    label_ind = label_names.index(model['target_label'])

    X_win, y_win, t_win = extract_sliding_windows(X, Y, M, timestamps, label_ind=label_ind,
                                              window_size=window_size, step_size=step_size,
                                              label_names=label_names, rule='at_least_one')

    X_win[np.isnan(X_win)] = 0.

    y_pred_prob = model['model'].predict(X_win, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # acc = accuracy_score(y_win, y_pred)
    prec = precision_score(y_win, y_pred, zero_division=0)
    rec = recall_score(y_win, y_pred, zero_division=0)
    f1 = f1_score(y_win, y_pred, zero_division=0)
    cm = confusion_matrix(y_win, y_pred)

    train_ba = (rec + (cm[0][0] / np.sum(cm[0]))) / 2
    
    metrics = compute_extended_metrics(y_win, y_pred, train_ba=train_ba)
    accuracy  = metrics['accuracy']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']
    balanced_accuracy = metrics['balanced_accuracy']
    train_BA = metrics['train_ba']
    BA_Gap = metrics['ba_gap']

    print("-" * 10)
    print(f"Accuracy:          {accuracy:.2f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}")
    print(f'Recall:           {rec:.2f}')
    print(f'Precision:        {prec:.2f}')
    print(f'F1-Score:         {f1:.2f}')
    print("Confusion Matrix:")
    print(cm)
    print(f"Sensitivity:       {sensitivity:.2f}")
    print(f"Specificity:       {specificity:.2f}")
    print(f"Train BA:          {train_BA:.2f}")
    print(f"BA Gap:            {BA_Gap:.2f}")
    print("-" * 10)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'train_BA': train_BA,
        'BA_gap': BA_Gap,

        'y_true': y_win.astype(int).tolist(),
        'y_pred': y_pred.tolist(),
        'y_timestamps': t_win.tolist()
    }
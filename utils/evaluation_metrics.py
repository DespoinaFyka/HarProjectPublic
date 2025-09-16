from sklearn.metrics import confusion_matrix

def compute_extended_metrics(y_true, y_pred, train_ba=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ba = (sensitivity + specificity) / 2

    ba_gap = train_ba - ba if train_ba is not None else None

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": ba,
        "train_ba": train_ba,
        "ba_gap": ba_gap
    }

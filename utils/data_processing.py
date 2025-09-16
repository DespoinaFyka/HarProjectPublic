import numpy as np


def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool)
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor)
        use_feature = np.logical_or(use_feature,is_from_sensor)
        pass
    X = X[:,use_feature]
    return X

def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train,axis=0)
    std_vec = np.nanstd(X_train,axis=0)
    return (mean_vec,std_vec)

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1))
    # Divide by the standard deviation, to get unit-variance for all features:
    # Avoided dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1))
    X_standard = X_centralized / normalizers
    return X_standard

MUTUALLY_EXCLUSIVE_GROUP = [
    'FIX_walking',
    'SITTING',
    'LYING_DOWN',
    'OR_standing',
    'FIX_running',
    'BICYCLING'
]

def extract_sliding_windows(X, Y, M, timestamps, label_ind,
                            window_size=10, step_size=5, rule='at_least_one',
                            label_names=None,
                            mutually_exclusive_group=MUTUALLY_EXCLUSIVE_GROUP):
    """
    Supports selecting a target label + automatically filters out conflicting labels from a mutually exclusive group
    Params:
    - X, Y, M: inputs
    - label_ind: index of the target label
    - label_names: list of all labels (used for name lookup)
    - mutually_exclusive_group: list of label names that should not co-occur in the same window
    """
    windows_X, windows_y, windows_timestamps = [], [], []
    num_samples = X.shape[0]

    # Preprocess index for mutually exclusive group
    if mutually_exclusive_group and label_names:
        exclusive_inds = [label_names.index(lab) for lab in mutually_exclusive_group]
    else:
        exclusive_inds = []

    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        mid_idx = start_idx + window_size // 2

        window_X = X[start_idx:end_idx]
        window_y_vals = Y[start_idx:end_idx, label_ind]
        window_m_vals = M[start_idx:end_idx, label_ind]

        # Skip if the label is missing in window
        if np.any(window_m_vals):
            continue

        # Check mutually exclusive conflict
        if exclusive_inds:
            group_slice = Y[start_idx:end_idx, exclusive_inds]
            group_mask = M[start_idx:end_idx, exclusive_inds]
            visible = group_slice * (1 - group_mask)
            active_per_label = np.sum(visible, axis=0)
            if np.sum(active_per_label > 0) > 1:
                continue  # More than one labels active in group → conflict

        # Label assignment
        if rule == 'strict':
            label = 1 if np.all(window_y_vals == 1) else 0
        elif rule == 'majority':
            label = 1 if np.mean(window_y_vals) >= 0.5 else 0
        elif rule == 'at_least_one':
            label = int(np.any(window_y_vals))
        else:
            raise ValueError(f"Unknown rule: {rule}")

        windows_X.append(window_X)
        windows_y.append(label)
        ts = timestamps[mid_idx] if mid_idx < len(timestamps) else timestamps[-1]
        windows_timestamps.append(ts)

    return np.array(windows_X), np.array(windows_y), np.array(windows_timestamps)

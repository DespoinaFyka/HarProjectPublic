import os


def load_cv_folds(folds_folder):
    """
    Reads all the .txt files inside the given folder and returns a dictionary {filename: [uuids]}.
    """
    folds = {}
    for filename in sorted(os.listdir(folds_folder)):
        if filename.endswith(".txt"):
            fold_path = os.path.join(folds_folder, filename)
            with open(fold_path, 'r') as f:
                uuids = [line.strip() for line in f.readlines() if line.strip()]
            folds[filename] = uuids
    return folds
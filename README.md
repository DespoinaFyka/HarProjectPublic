# Human Activity Recognition Project (HARProjectPublic)

This repository contains the full pipeline for Human Activity Recognition (HAR) based on the ExtraSensory dataset.  
It includes data preprocessing, feature extraction, label grouping, model training (Logistic Regression, HMM, CNN, LSTM, CNN-LSTM), and comparative evaluation across unseen users.

## Overview

The project explores multiple classical and deep learning models to recognize human activities from multimodal sensor data collected via smartphones and smartwatches.

The workflow includes:
- Data preprocessing and cleaning
- Feature and label alignment per user
- Cross-validation across user folds
- Model training and evaluation (per-fold and per-user)
- Comparative analysis of model performance (accuracy, F1-score, balanced accuracy)

## Repository Structure

<ul>
  <li>HarProjectPublic/
    <ul>
      <li>main_logistic_regression.py # Train/test logistic regression model</li>
      <li>main_hmm.py # Hidden Markov Model pipeline</li>
      <li>main_cnn.py # CNN-based activity recognition</li>
      <li>main_lstm.py # LSTM-based activity recognition</li>
      <li>main_cnn_lstm.py # Hybrid CNN-LSTM model</li>
    </ul>
  </li>
  <li>Fourth item</li>
</ul>

HarProjectPublic/
│
├── main_logistic_regression.py     # Train/test logistic regression model
├── main_hmm.py                     # Hidden Markov Model pipeline
├── main_cnn.py                     # CNN-based activity recognition
├── main_lstm.py                    # LSTM-based activity recognition
├── main_cnn_lstm.py                # Hybrid CNN-LSTM model
│
├── utils/                          # Helper functions (data loading, feature projection, etc.)
│   ├── load_train_test_uuids.py
│   ├── standardization_utils.py
│   ├── feature_extraction.py
│   └── ...
│
├── results/                        # Model evaluation results and metrics per fold/user
├── cv_5_folds/                     # User UUID splits for 5-fold cross-validation
├── ExtraSensory.per_uuid_features_labels/   # Per-user data files
│
├── compare_models.py               # Comparison of all trained models
├── analyze_labels_correlation.py   # Label co-occurrence analysis
├── analyze_label_correlation_w_timestamps.py
├── open_npy_files.py               # Utility for loading stored NumPy results
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip and virtual environment recommended

### Installation
Clone the repository:
```bash
git clone https://github.com/DespoinaFyka/HarProjectPublic.git
cd HarProjectPublic


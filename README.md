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
HarProjectPublic/
    <ul>
      <li>main_logistic_regression.py # Train/test logistic regression model</li>
      <li>main_cnn.py                 # CNN-based activity recognition</li>
      <li>main_lstm.py                # LSTM-based activity recognition</li>
      <li>main_cnn_lstm.py            # Hybrid CNN-LSTM model</li>
      <li>main_data_analysis.py       </li>
      <li>compare_models.py           # Comparison of all trained models</li>
      <li>analyze_labels_correlation.py   # Label co-occurrence analysis</li>
      <li>analyze_label_correlation_w_timestamps.py</li>
      <li>utils/                      # Helper functions (data loading, feature projection, etc.)
          <ul>
            <li>data_loading.py</li>
            <li>data_processing.py</li>
            <li>evaluation_metrics.py</li>
            <li>labels_utils.py</li>
            <li>load_train_test_uuids.py</li>
            <li>plotting.py</li>
          </ul>
      </li>
      <li>results/       # Model evaluation results and metrics per fold/user</li>
      <li>cv_5_folds/    # User UUID splits for 5-fold cross-validation</li>
      <li>ExtraSensory.per_uuid_features_labels/   # Per-user data files from ExtraSensory dataset</li>
      <li>open_npy_files.py               # Utility for loading stored NumPy results</li>
      <li>requirements.txt                # Python dependencies</li>
      <li>README.md                       # This file</li>
    </ul>

## Installation & Setup

### Prerequisites
- Python 3.13.3 / 3.11.8
- pip and virtual environment recommended

### Installation
Clone the repository:
```bash
git clone https://github.com/DespoinaFyka/HarProjectPublic.git
cd HarProjectPublic






# Human Activity Recognition Project (HARProjectPublic)

This repository contains the full pipeline for Human Activity Recognition (HAR) based on the ExtraSensory dataset.  
It includes data preprocessing, feature extraction, label grouping, model training (Logistic Regression, CNN, LSTM, CNN-LSTM), and comparative evaluation across unseen users.

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
      <li>main_logistic_regression.py </li>
          Basic logistic regression model
      <li>main_cnn.py                 </li>
          CNN-based activity recognition
      <li>main_lstm.py                </li>
          LSTM-based activity recognition
      <li>main_cnn_lstm.py            </li>
          Hybrid CNN-LSTM model
      <li>main_data_analysis.py       </li>
          Data analysis methods for the dataset 
      <li>compare_models.py           </li>
          Comparison of all trained models
      <li>analyze_labels_correlation.py   </li>
          Label co-occurrence analysis
      <li>analyze_label_correlation_w_timestamps.py</li>
          Label co-occurrence analysis with timestamps
      <li>utils/                      
          (Helper functions (data loading, feature projection, etc.))
          <ul>
            <li>data_loading.py</li>
            <li>data_processing.py</li>
            <li>evaluation_metrics.py</li>
            <li>labels_utils.py</li>
            <li>load_train_test_uuids.py</li>
            <li>plotting.py</li>
          </ul>
      </li>
      <li>results/       
          Model evaluation results and metrics per fold/user</li>
      <li>cv_5_folds/    </li>
          User UUID splits for 5-fold cross-validation
      <li>ExtraSensory.per_uuid_features_labels/   </li>
          Per-user data files from ExtraSensory dataset
      <li>open_npy_files.py               </li>
          Utility for loading stored NumPy results
      <li>requirements.txt                </li>
          Python dependencies
      <li>README.md                       </li>
          This file
    </ul>


## Prerequisites
- Python 3.13.3 and 3.11.8
- pip and virtual environment recommended

  
## Installation:
1. Clone the repository:
   git clone https://github.com/DespoinaFyka/HarProjectPublic.git
   cd HarProjectPublic

2. Install dependencies:
   pip install -r requirements.txt


## Usage

Each main file represents a specific model pipeline:

| Model | Script | Description |
|-------|---------|-------------|
| Logistic Regression | main_logistic_regression.py | Baseline model for user-independent evaluation |
| CNN | main_cnn.py | Convolutional model for sensor-based feature windows |
| LSTM | main_lstm.py | Recurrent model for sequential time-series learning |
| CNN-LSTM | main_cnn_lstm.py | Hybrid deep learning model combining spatial and temporal features |

Example commands:
- Train and evaluate a CNN model:
  python main_cnn.py

- Train the LSTM model:
  python main_lstm.py

- Compare all models’ results:
  python compare_models.py

- Analyze label correlations:
  python analyze_labels_correlation.py


## Evaluation

Each model is trained on 4/5 of the users and tested on the remaining unseen 1/5.
Metrics include:
- Accuracy
- Balanced Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix
- Sensitivity / Specificity

Results per fold and per user are stored in the "results/" folder.

## Dataset

This project uses the ExtraSensory Dataset (Vaizman et al., 2017).
It includes multimodal data from 60 participants, each with:
- Smartphone and smartwatch sensor features (Acc, Gyro, WAcc, etc.)
- Self-reported activity labels (e.g., Sitting, Walking, At home)

The data are grouped into:
- Main activity labels (mutually exclusive postures)
- Secondary contextual labels (multi-label environments and activities)


## Results

Example findings:
- CNN-LSTM achieved the highest performance among all tested models.
- Logistic Regression provided strong baseline generalization.
- Temporal models (CNN, LSTM) improved recognition of continuous activities (e.g., walking).


## Future Work

- Fine-tune deep models with attention mechanisms
- Extend to real-time inference
- Improve label balancing and augmentation techniques


## Contributing

Contributions are welcome.
Feel free to open an issue, suggest improvements, or submit a pull request.

Steps:
1. Fork the repository
2. Create your feature branch: git checkout -b feature-name
3. Commit your changes
4. Push to the branch and open a Pull Request


## License

Distributed under the MIT License.
See the LICENSE file for details.


## Contact

Despoina Fyka - despoinafyka@gmail.com
GitHub: https://github.com/DespoinaFyka


## References

- Vaizman, Y., Ellis, K., & Lanckriet, G. (2017). Recognizing detailed human context in the wild from smartphones and smartwatches data.
- ExtraSensory Dataset: https://github.com/ExtraSensoryDataset/ExtraSensory




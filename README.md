# Predicting Hit Songs Using Repeated Chorus

## Problem Statement
This mini-project predicts whether a song is popular or not using audio features extracted from a 15-second repeated chorus or hook. The project focuses only on audio information and does not use artist, album, social-media, or release-date metadata.

## Dataset
The project follows a supervised binary-classification setup:
- `1` - popular / hit song
- `0` - unpopular / non-hit song

For each song, `pychorus` is used to locate a repeated chorus and `librosa` is used to extract MFCC, chroma, RMS energy, spectral features, tonal centroid, and zero-crossing rate. Seven summary statistics are calculated for the time-varying features, giving 518 audio features per song.

The input metadata CSV must contain:

```csv
track_id,audio_path,label
song_001,data/audio/song_001.wav,1
song_002,data/audio/song_002.wav,0
```

## Project Structure
```text
UE24CS352A-Predicting-Hit-Songs/
├── src/
│   ├── extract_features.py
│   ├── train_models.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

## Run
### 1. Extract repeated-chorus audio features
```bash
python src/extract_features.py --input data/audio_metadata.csv --output data/features.csv
```

If a repeated chorus cannot be detected for a particular song, the script prints a warning and uses the middle 15 seconds as a fallback.

### 2. Train and compare models
```bash
python src/train_models.py --features data/features.csv --out results
```

The models are compared using 5-fold cross-validation. The saved `best_model.joblib` is selected using cross-validated F1-score, while the held-out test set is kept for final evaluation.

### 3. Predict using the saved model
```bash
python src/predict.py --model results/best_model.joblib --features data/features.csv --output results/predictions.csv
```

## Models Compared
- Logistic Regression
- Linear Discriminant Analysis
- Linear SVM
- RBF SVM
- Polynomial SVM
- Random Forest
- Gradient Boosting
- Neural Network / MLP

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## Note
Audio files and generated outputs are not stored in the repository. Keep the project audio files locally under `data/audio/`, create `data/audio_metadata.csv` using the format shown above, and then run the commands in order.

The mini-project write-up PDF and presentation are maintained separately as submission/review deliverables.

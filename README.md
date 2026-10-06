# Predicting Hit Songs Using Machine Learning

## Problem Statement
This mini-project predicts whether a song belongs to the hit or non-hit class using audio-related features and supervised machine learning.

The output is binary:

- `1` = hit song
- `0` = non-hit song

## Project Overview
The project contains two workflows.

### 1. Raw-Audio Feature Extraction
This workflow processes local WAV audio files.

```text
Audio file -> repeated 15-second chorus -> Librosa features -> summary statistics -> 518 numerical features
```

It uses:

- `pychorus` to identify a repeated chorus
- `librosa` to extract audio features
- the middle 15 seconds as a fallback if a repeated chorus cannot be detected

### 2. Real-Data Machine-Learning Evaluation
The main machine-learning evaluation uses a real labelled music dataset.

Dataset summary:

- Total songs: `6,398`
- Hit songs: `3,199`
- Non-hit songs: `3,199`
- The dataset is balanced
- The dataset contains precomputed music/audio features

The reported `84.69%` test accuracy comes from this 6,398-song real-data evaluation. The raw-audio feature-extraction workflow is separate.

## Dataset and Reproduction Steps

Dataset source:

- [The Spotify Hit Predictor Dataset](https://github.com/fortyTwo102/The-Spotify-Hit-Predictor-Dataset)
- File used: `dataset-of-10s.csv`

The source dataset contains Spotify audio features and a binary `target` column. In the source documentation, `target = 1` means the song appeared in a Billboard Hot 100 weekly list for that decade at least once, while `target = 0` represents the source author's non-hit class.

The dataset files are kept locally and are not stored in this repository.

### Prepare the dataset

1. Download `dataset-of-10s.csv` from the dataset source.
2. Place it inside the local `data/` folder and rename it to:

```text
data/real_billboard.csv
```

3. Prepare the feature file:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/real_billboard.csv")
df = df.rename(columns={
    "target": "label",
    "track": "track_id"
})
df.to_csv("data/real_features.csv", index=False)

print("Songs:", len(df))
print(df["label"].value_counts())
PY
```

For `dataset-of-10s.csv`, this produces 6,398 songs: 3,199 hits and 3,199 non-hits.

## Model Training

The following models are compared:

- Logistic Regression
- Linear Discriminant Analysis
- Linear SVM
- RBF SVM
- Polynomial SVM
- Random Forest
- Gradient Boosting
- Neural Network / MLP

The dataset is split into 75% training data and 25% test data. Five-fold stratified cross-validation is used on the training data. The final model is selected using cross-validation F1-score.

Train the models with:

```bash
python src/train_models.py --features data/real_features.csv --out results_real
```

## Final Result

Best model:

```text
Random Forest
Cross-validation F1: 84.66%
Test accuracy:       84.69%
Test F1-score:       85.25%
Test precision:      82.23%
Test recall:         88.50%
```

## Prediction

Run prediction using the saved best model:

```bash
python src/predict.py --model results_real/best_model.joblib --features data/real_features.csv --output results_real/predictions.csv
```

Show the generated predictions:

```bash
head results_real/predictions.csv
```

The predicted labels use:

- `1` = hit
- `0` = non-hit

## Raw-Audio Feature Extraction

For local WAV files, prepare a CSV containing:

```text
track_id,audio_path,label
```

Run:

```bash
python src/extract_features.py --input data/audio_metadata.csv --output data/features.csv
```

The generated file contains `track_id`, `label`, and 518 numerical audio features for each processed song.

The extracted audio feature groups include:

- Chroma STFT
- Chroma CQT
- Chroma CENS
- MFCC
- RMS
- Spectral centroid
- Spectral bandwidth
- Spectral contrast
- Spectral rolloff
- Tonnetz
- Zero-crossing rate

Each feature dimension is summarized using minimum, mean, median, maximum, standard deviation, skewness, and kurtosis.

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

Dataset files, audio files, trained models, and generated result files are kept locally and are excluded from the repository.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score

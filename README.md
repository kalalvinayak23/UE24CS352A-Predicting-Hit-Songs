# Predicting Hit Songs Using Machine Learning

## Problem Statement
This mini-project predicts whether a song belongs to the hit or non-hit class using audio-related features and supervised machine learning.

The output is binary:

- `1` = hit song
- `0` = non-hit song

## Final Project Explanation
The project has two parts:

### Part 1: Raw-audio pipeline
This part verifies that the program can process local WAV audio files.

Pipeline:

```text
Audio file -> repeated 15-second chorus -> Librosa features -> summary statistics -> 518 numeric features
```

The raw-audio pipeline uses:

- `pychorus` for repeated chorus detection
- `librosa` for audio feature extraction
- fallback to the middle 15 seconds if a chorus cannot be detected

### Part 2: Real-data machine-learning experiment
For the final ML result, a real labelled music dataset was used locally.

Dataset summary:

- Total songs: `6,398`
- Hit songs: `3,199`
- Non-hit songs: `3,199`
- Dataset is balanced
- The real dataset contains precomputed music/audio features

Important: the final `84.69%` test accuracy comes from the real-data experiment, not from the artificial WAV test files.

## Real Dataset and Reproducing the Demo

Dataset source:

- [The Spotify Hit Predictor Dataset](https://github.com/fortyTwo102/The-Spotify-Hit-Predictor-Dataset)
- File used: `dataset-of-10s.csv`

The source dataset contains Spotify audio features and a binary `target` column. In the source documentation, `target = 1` means the song appeared in a Billboard Hot 100 weekly list for that decade at least once; `target = 0` is the source author's non-hit class.

The dataset itself is not stored in this repository. To reproduce the real-data demo:

1. Download `dataset-of-10s.csv` from the source repository.
2. Place it inside the local `data/` folder and rename it to:

```text
data/real_billboard.csv
```

3. Prepare the file for this project:

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

For the downloaded `dataset-of-10s.csv`, this gives 6,398 songs: 3,199 hits and 3,199 non-hits.

4. Train the models:

```bash
python src/train_models.py --features data/real_features.csv --out results_real
```

5. Run prediction using the saved best model:

```bash
python src/predict.py --model results_real/best_model.joblib --features data/real_features.csv --output results_real/predictions.csv
```

6. Show a few predictions:

```bash
head results_real/predictions.csv
```

## Final Result
The following models were compared:

- Logistic Regression
- Linear Discriminant Analysis
- Linear SVM
- RBF SVM
- Polynomial SVM
- Random Forest
- Gradient Boosting
- Neural Network / MLP

The final model was selected using cross-validation F1-score.

Best model:

```text
Random Forest
Cross-validation F1: 84.66%
Test accuracy:       84.69%
Test F1-score:       85.25%
Test precision:      82.23%
Test recall:         88.50%
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

The dataset files and generated result files are intentionally not stored in the GitHub repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

## Raw-Audio Feature Extraction Command
Use this when you have local WAV audio files and a CSV with `track_id,audio_path,label`.

```bash
python src/extract_features.py --input data/audio_metadata.csv --output data/features.csv
```

## Train Models
For raw-audio extracted features:

```bash
python src/train_models.py --features data/features.csv --out results
```

For the local real-data experiment used in the final demo:

```bash
python src/train_models.py --features data/real_features.csv --out results_real
```

## Predict
For the final real-data demo:

```bash
python src/predict.py --model results_real/best_model.joblib --features data/real_features.csv --output results_real/predictions.csv
```

Show predictions:

```bash
head results_real/predictions.csv
```

## Evaluation Metrics
The project uses:

- Accuracy
- Precision
- Recall
- F1-score

## Demo Note
During the live demo, use the saved model in `results_real/best_model.joblib` for fast prediction. Retrain the models only if the evaluator specifically asks to see training.

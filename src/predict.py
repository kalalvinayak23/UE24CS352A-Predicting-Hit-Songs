"""Predict hit/not-hit labels for a feature CSV using a trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(model_path: Path, features_path: Path, output_path: Path) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(features_path)
    ids = df["track_id"] if "track_id" in df.columns else pd.Series(range(len(df)), name="track_id")
    drop_cols = [c for c in ["label", "track_id", "song_name", "artist"] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include="number").fillna(0)
    predictions = model.predict(X)
    out = pd.DataFrame({"track_id": ids, "predicted_label": predictions})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict song popularity")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/predictions.csv"))
    args = parser.parse_args()
    predict(args.model, args.features, args.output)

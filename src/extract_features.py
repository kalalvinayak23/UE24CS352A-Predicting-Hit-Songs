"""Extract repeated-chorus audio features from user-provided songs.

Expected input CSV columns:
    track_id,audio_path,label

A 15-second repeated chorus is detected with pychorus. If pychorus cannot
find a chorus in a particular file, the middle 15 seconds are used as a
fallback and a warning is shown.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install the dependencies: pip install -r requirements.txt") from exc


def _summary_stats(name: str, values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values)
    if flat.ndim == 1:
        flat = flat.reshape(1, -1)

    result: Dict[str, float] = {}
    for row_index, row in enumerate(flat):
        row = np.nan_to_num(row)
        prefix = f"{name}_{row_index + 1:02d}"
        result[f"{prefix}_min"] = float(np.min(row))
        result[f"{prefix}_mean"] = float(np.mean(row))
        result[f"{prefix}_median"] = float(np.median(row))
        result[f"{prefix}_max"] = float(np.max(row))
        result[f"{prefix}_std"] = float(np.std(row))

        centered = row - np.mean(row)
        std = np.std(row) or 1.0
        standardized = centered / std
        result[f"{prefix}_skew"] = float(np.mean(standardized**3))
        result[f"{prefix}_kurtosis"] = float(np.mean(standardized**4))

    return result


def _load_chorus_segment(
    audio_path: Path,
    duration: float = 15.0,
    sr: int = 22050,
) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)
    fallback_start = max(0.0, (total_duration - duration) / 2.0)
    start_time = fallback_start

    try:
        from pychorus import find_and_output_chorus

        detected = find_and_output_chorus(str(audio_path), None, duration)
        if detected is not None and float(detected) >= 0:
            start_time = float(detected)
        else:
            warnings.warn(
                f"No repeated chorus detected for {audio_path.name}; "
                "using the middle 15 seconds instead.",
                RuntimeWarning,
            )
    except ImportError:
        warnings.warn(
            "pychorus is not installed; using the middle 15 seconds. "
            "Install dependencies with: pip install -r requirements.txt",
            RuntimeWarning,
        )
    except Exception as exc:
        warnings.warn(
            f"Chorus detection failed for {audio_path.name} ({exc}); "
            "using the middle 15 seconds instead.",
            RuntimeWarning,
        )

    start_sample = int(start_time * sr)
    end_sample = int(min(len(y), start_sample + duration * sr))
    segment = y[start_sample:end_sample]

    target_length = int(duration * sr)
    if len(segment) < target_length:
        segment = np.pad(segment, (0, target_length - len(segment)))

    return segment, sr


def extract_one(audio_path: Path) -> Dict[str, float]:
    y, sr = _load_chorus_segment(audio_path)
    features: Dict[str, float] = {}
    features.update(_summary_stats("chroma_stft", librosa.feature.chroma_stft(y=y, sr=sr)))
    features.update(_summary_stats("chroma_cqt", librosa.feature.chroma_cqt(y=y, sr=sr)))
    features.update(_summary_stats("chroma_cens", librosa.feature.chroma_cens(y=y, sr=sr)))
    features.update(_summary_stats("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)))
    features.update(_summary_stats("rms", librosa.feature.rms(y=y)))
    features.update(_summary_stats("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.update(_summary_stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.update(_summary_stats("spectral_contrast", librosa.feature.spectral_contrast(y=y, sr=sr)))
    features.update(_summary_stats("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.update(_summary_stats("tonnetz", librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)))
    features.update(_summary_stats("zero_crossing_rate", librosa.feature.zero_crossing_rate(y)))
    return features


def extract_dataset(input_csv: Path, output_csv: Path) -> None:
    metadata = pd.read_csv(input_csv)
    required = {"track_id", "audio_path", "label"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    if metadata.empty:
        raise ValueError("Input CSV does not contain any song rows.")

    labels = set(pd.to_numeric(metadata["label"], errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError("The label column must contain only 0 (non-hit) and 1 (hit).")

    rows = []
    for _, row in metadata.iterrows():
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        feature_row = {
            "track_id": row["track_id"],
            "label": int(row["label"]),
        }
        feature_row.update(extract_one(audio_path))
        rows.append(feature_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Extracted features saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract repeated-chorus audio features")
    parser.add_argument("--input", type=Path, required=True, help="CSV with track_id,audio_path,label")
    parser.add_argument("--output", type=Path, default=Path("data/features.csv"))
    args = parser.parse_args()
    extract_dataset(args.input, args.output)

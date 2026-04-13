"""
Training script for SpectralBandPowerModel.

Loads ECoG data from ./data/ (parquet format used by the BrainStorm pipeline)
and trains the spectral + EMA feature logistic regression classifier.

Usage:
    uv run python scripts/train_spectral.py
    uv run python scripts/train_spectral.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on the path when running as a script
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from brainstorm.constants import FEATURES_FILE_NAME, LABELS_FILE_NAME
from brainstorm.ml.spectral_band_power import SpectralBandPowerModel


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from the standard BrainStorm data directory.

    Expects:
        <data_dir>/features.parquet  — shape (n_samples, n_channels)
        <data_dir>/labels.parquet    — shape (n_samples,)

    Falls back to .npy files if parquet files are not present.

    Args:
        data_dir: Path to the directory containing data files.

    Returns:
        (X, y) where X has shape (n_samples, n_channels) and y shape (n_samples,).
    """
    # Try parquet first (primary format)
    features_parquet = data_dir / FEATURES_FILE_NAME
    labels_parquet = data_dir / LABELS_FILE_NAME

    if features_parquet.exists() and labels_parquet.exists():
        try:
            import pandas as pd

            print(f"Loading data from {data_dir} (parquet format)...")
            X = pd.read_parquet(features_parquet).to_numpy(dtype=np.float32)
            y = pd.read_parquet(labels_parquet).to_numpy().ravel()
            return X, y
        except ImportError:
            print("pandas not available, falling back to numpy format.")

    # Try numpy .npy files as fallback
    features_npy = data_dir / "features.npy"
    labels_npy = data_dir / "labels.npy"
    if features_npy.exists() and labels_npy.exists():
        print(f"Loading data from {data_dir} (numpy format)...")
        X = np.load(features_npy).astype(np.float32)
        y = np.load(labels_npy).ravel()
        return X, y

    raise FileNotFoundError(
        f"No data found in {data_dir}. "
        f"Expected {FEATURES_FILE_NAME} + {LABELS_FILE_NAME} (parquet) "
        "or features.npy + labels.npy (numpy)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SpectralBandPowerModel on BrainStorm ECoG data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing features and labels (default: ./data/).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100,
        help="Circular buffer size in samples for FFT (default: 100 = 100 ms at 1 kHz).",
    )
    args = parser.parse_args()

    # Load data
    X, y = load_data(args.data_dir)
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    print(f"Classes: {np.unique(y)}")

    # Instantiate and train model
    model = SpectralBandPowerModel(buffer_size=args.buffer_size)
    model.fit(X, y)

    print("Training complete. Model and metadata saved.")
    print(f"Model path: {REPO_ROOT / 'model.pkl'}")


if __name__ == "__main__":
    main()

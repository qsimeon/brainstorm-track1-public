from __future__ import annotations

import os
from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path

# Download specific files (not test files!)
DATA_PATH = Path("./data")
DATA_PATH.mkdir(exist_ok=True)

OPEN_DATASET_ID = "PrecisionNeuroscience/BrainStorm2026-Track1"

def _download_file(
    repo_id: str, filename: str, token: str | None = None
) -> pd.DataFrame:
    """Download and load a parquet file from HuggingFace."""
    return pd.read_parquet(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=DATA_PATH,
            token=token,  # Pass token for private repos
        )
    )


def download_train_validation_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Download publicly available train and validation data."""
    train_features = _download_file(OPEN_DATASET_ID, "train_features.parquet")
    train_labels = _download_file(OPEN_DATASET_ID, "train_labels.parquet")
    val_features = _download_file(OPEN_DATASET_ID, "validation_features.parquet")
    val_labels = _download_file(OPEN_DATASET_ID, "validation_labels.parquet")
    return train_features, train_labels, val_features, val_labels


def download_test_data(token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download test data from private repo. Requires authentication token."""
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Token required. Pass as argument or set HF_TOKEN environment variable."
            )

    test_features = _download_file(
        CLOSED_DATASET_ID, "test_features.parquet", token=token
    )
    test_labels = _download_file(CLOSED_DATASET_ID, "test_labels.parquet", token=token)
    return test_features, test_labels

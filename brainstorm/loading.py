from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from typing import Literal
from pydantic import ConfigDict, validate_call

from brainstorm.constants import FEATURES_FILE_NAME, LABELS_FILE_NAME, REPO_ROOT


@validate_call(config=ConfigDict(arbitrary_types_allowed=True), validate_return=True)
def load_raw_data(
    source_directory: Path,
    step: Literal[
        "train",
        "validation",
        "test",
    ] = "train",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw ECoG data and trials labels from a source directory.

    Args:
        source_directory: Path to the source directory containing the ECoG data and trials labels.

    Returns:
        features: DataFrame with columns ['time_s', 'value', 'channel_1', 'channel_2', ..., 'channel_N']
        labels: DataFrame with columns ['time_s', 'label', 'prev_label']
    """
    assert source_directory.exists(), (
        f"Source directory {source_directory} does not exist"
    )
    assert source_directory.is_dir(), (
        f"Source directory {source_directory} is not a directory"
    )

    logger.info(f"Loading data from: {source_directory}")

    features_source = source_directory / f"{step}_{FEATURES_FILE_NAME}"
    assert features_source.exists(), f"Features file {features_source} does not exist"

    labels_source = source_directory / f"{step}_{LABELS_FILE_NAME}"
    assert labels_source.exists(), f"Labels file {labels_source} does not exist"

    # Load from parquet files
    features = pd.read_parquet(features_source)
    labels = pd.read_parquet(labels_source)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")

    start, end = features.index[0], features.index[-1]
    logger.info(f"Features time range: {start} to {end}")

    start, end = labels.index[0], labels.index[-1]
    logger.info(f"Labels time range: {start} to {end}")

    return features, labels


@validate_call(config=ConfigDict(arbitrary_types_allowed=True), validate_return=True)
def load_channel_coordinates() -> np.ndarray:
    """
    Load channel coordinates from a source file.

    Returns:
        Array2D of shape (N_CHANNELS, 2) containing (x, y) coordinates for each channel.
    """
    channels_coords_source = REPO_ROOT / "channels_coords.npy"
    assert channels_coords_source.exists(), (
        f"Channels coords file {channels_coords_source} does not exist"
    )

    return np.load(channels_coords_source)

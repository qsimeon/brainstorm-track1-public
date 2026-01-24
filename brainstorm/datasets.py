"""
PyTorch datasets for streaming neural decoding with growing sequence context.

The key design is UNCONVENTIONAL: rather than treating timesteps as independent
samples, we treat them as a continuous stream where the model sees growing
history. This forces the model to handle variable sequence lengths, critical
for streaming inference.

Dataset Architecture:
- Sample 0: (1024, 1, F_bins) - all channels, first timestep only
- Sample 1: (1024, 2, F_bins) - all channels, first 2 timesteps
- Sample t: (1024, t+1, F_bins) - all channels, timesteps [0:t+1]
- After max_seq_len: sliding window of size max_seq_len

This ensures the model trains on all sequence lengths from 1 to max_seq_len,
making it robust for streaming inference.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import pandas as pd


class GrowingSequenceDataset(Dataset):
    """
    UNCONVENTIONAL dataset for streaming neural decoding.

    Key insight: Treat channels (1024) as the batch dimension and timesteps as
    the sequence dimension. Each sample returns all channels at a single
    timestep with growing history.

    This trains the model to handle ALL sequence lengths from 1 to max_seq_len,
    which is critical for streaming inference where context grows over time.
    """

    def __init__(
        self,
        spectrograms: torch.Tensor,
        labels: torch.Tensor,
        max_seq_len: int = 1000,
        channel_coords: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the GrowingSequenceDataset.

        Args:
            spectrograms: Precomputed spectrograms of shape (1024, T_total, F_bins)
            labels: Class labels for each timestep of shape (T_total,)
            max_seq_len: Maximum history length (sliding window size after reaching this)
            channel_coords: Optional (1024, 2) array of electrode (x, y) coordinates
                           Used to compute spatial positional encodings
        """
        assert spectrograms.shape[0] == 1024, (
            f"Expected 1024 channels in first dimension, got {spectrograms.shape[0]}"
        )
        assert spectrograms.shape[1] == labels.shape[0], (
            f"Temporal dimension mismatch: spectrograms have {spectrograms.shape[1]} "
            f"timesteps but labels have {labels.shape[0]}"
        )

        self.spectrograms = spectrograms  # (1024, T_total, F_bins)
        self.labels = labels  # (T_total,)
        self.max_seq_len = max_seq_len
        self.T_total = spectrograms.shape[1]
        self.n_channels = spectrograms.shape[0]
        self.F_bins = spectrograms.shape[2]

        # Channel coordinates for spatial positional encoding
        self.channel_coords = channel_coords
        self.spatial_encodings: Optional[torch.Tensor] = None

        if channel_coords is not None:
            self.spatial_encodings = self._compute_spatial_encodings(channel_coords)
            logger.info(f"Computed spatial encodings of shape {self.spatial_encodings.shape}")

        logger.info(
            f"Initialized GrowingSequenceDataset with {self.T_total} timesteps, "
            f"{self.n_channels} channels, {self.F_bins} frequency bins, "
            f"max_seq_len={max_seq_len}"
        )

    def _compute_spatial_encodings(self, coords: np.ndarray) -> torch.Tensor:
        """
        Create sinusoidal positional encodings from (x, y) electrode coordinates.

        Similar to transformer positional encodings but based on spatial position
        rather than temporal position. This allows the model to learn the
        spatial layout of the electrode grid.

        Args:
            coords: (1024, 2) array of (x, y) electrode positions

        Returns:
            encodings: (1024, F_bins) sinusoidal encodings matching feature dimension
        """
        assert coords.shape == (1024, 2), f"Expected shape (1024, 2), got {coords.shape}"

        # Normalize coordinates to [0, 1]
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)

        # Create sinusoidal encodings with different frequencies for x and y
        encoding_dim = self.F_bins
        encodings = []

        # Interleave sin/cos for x and y positions with increasing frequencies
        for i in range(encoding_dim // 2):
            freq = 1.0 / (10000 ** (2 * i / encoding_dim))
            encodings.append(np.sin(freq * coords_norm[:, 0]))  # x position
            if len(encodings) < encoding_dim:
                encodings.append(np.cos(freq * coords_norm[:, 1]))  # y position

        # Handle odd dimension
        if len(encodings) < encoding_dim:
            encodings.append(np.sin(coords_norm[:, 0]))

        encodings_array = np.stack(encodings[:encoding_dim], axis=1)  # (1024, F_bins)
        return torch.from_numpy(encodings_array).float()

    def __len__(self) -> int:
        """
        Return the number of timesteps in the dataset.

        We can make a prediction at each timestep from t=0 to t=T_total-1.
        """
        return self.T_total

    def __getitem__(self, t: int) -> Dict[str, Any]:
        """
        Get data for timestep t with all history.

        GROWING PHASE (t < max_seq_len):
            - Use all data from start: history = [0:t+1]
            - Sequence length = t + 1

        SLIDING WINDOW PHASE (t >= max_seq_len):
            - Use only last max_seq_len timesteps
            - Sequence length = max_seq_len
            - Start index = t - max_seq_len + 1

        This ensures the model trains on all sequence lengths [1, max_seq_len].

        Args:
            t: Timestep index (0 to T_total-1)

        Returns:
            Dictionary with:
                'spectrogram': (1024, seq_len_t, F_bins) - all channels with history
                'label': scalar tensor - class label at timestep t
                'seq_len': int - actual sequence length used
                'timestep': int - original timestep index t
                'spatial_encoding': (1024, F_bins) or None - positional encodings
        """
        # Determine sequence length and start index
        if t < self.max_seq_len:
            # Growing phase: use all history from beginning
            seq_len = t + 1
            start_idx = 0
        else:
            # Sliding window phase: use last max_seq_len timesteps
            seq_len = self.max_seq_len
            start_idx = t - self.max_seq_len + 1

        # Extract spectrogram window: (1024, seq_len, F_bins)
        spec_window = self.spectrograms[:, start_idx : start_idx + seq_len, :]

        # Get label for this timestep
        label = self.labels[t]

        return {
            "spectrogram": spec_window,
            "label": label,
            "seq_len": seq_len,
            "timestep": t,
            "spatial_encoding": self.spatial_encodings,
        }


def identity_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple collate function that returns the single item without batching.

    Since each sample already contains all 1024 channels, we treat it as a
    complete "batch". This collate function is used with batch_size=1.

    Args:
        batch: List with single item from GrowingSequenceDataset

    Returns:
        The single item from the dataset (no additional batching)
    """
    assert len(batch) == 1, (
        f"identity_collate expects batch_size=1, got {len(batch)}. "
        "Set batch_size=1 in DataLoader when using this collate function."
    )

    item = batch[0]

    # Ensure tensors are on correct device and dtype
    return {
        "spectrogram": item["spectrogram"].float(),
        "label": item["label"].long(),
        "seq_len": item["seq_len"],
        "timestep": item["timestep"],
        "spatial_encoding": item["spatial_encoding"].float() if item["spatial_encoding"] is not None else None,
    }




def create_dataloader(
    spectrograms: torch.Tensor,
    labels: torch.Tensor,
    max_seq_len: int = 1000,
    channel_coords: Optional[np.ndarray] = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for growing-sequence streaming neural decoding.

    Each batch contains a single timestep with all channels and growing history.
    - Sample 0: (1024, 1, F_bins)
    - Sample 1: (1024, 2, F_bins)
    - Sample t: (1024, seq_len_t, F_bins) where seq_len_t grows from 1 to max_seq_len

    Args:
        spectrograms: Precomputed spectrograms (1024, T_total, F_bins)
        labels: Labels for each timestep (T_total,)
        max_seq_len: Maximum sequence length (sliding window size)
        channel_coords: Optional (1024, 2) electrode coordinates
        shuffle: Whether to shuffle timesteps (usually False for sequential)
        num_workers: Number of data loading workers

    Returns:
        DataLoader yielding batches of growing-context spectrograms
    """
    # Validate input shapes
    assert spectrograms.shape[0] == 1024, f"Expected 1024 channels, got {spectrograms.shape[0]}"
    assert spectrograms.shape[1] == labels.shape[0], (
        f"Temporal dimension mismatch: {spectrograms.shape[1]} vs {labels.shape[0]}"
    )

    # Create dataset
    dataset = GrowingSequenceDataset(
        spectrograms=spectrograms,
        labels=labels,
        max_seq_len=max_seq_len,
        channel_coords=channel_coords,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=identity_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

"""
Spectrogram computation and preprocessing for ECoG signals.

This module handles transformation of raw 1024-channel ECoG voltage data into
spectrogram representations suitable for neural decoding with streaming/causal
constraints (no future data leakage).

Key Design:
- 50ms windows with 1ms steps → 1:1 mapping to labels
- Zero-padding at start for cold start (causality)
- Parallel processing across channels
- Efficient storage in PyTorch .pt format
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy import signal
from loguru import logger
from joblib import Parallel, delayed
from tqdm import tqdm


def compute_channel_spectrogram(
    signal_data: np.ndarray,
    fs: int = 1000,
    window_ms: int = 50,
    step_ms: int = 1,
    freq_range: Tuple[float, float] = (0, 500),
    pad_mode: str = "constant",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a causal spectrogram for a single ECoG channel.

    CAUSALITY: Each window at time t only uses samples [0:t], preventing future
    data leakage. Zero-padding is applied at the beginning to handle cold start.

    Args:
        signal_data: 1D array of voltage samples (T_samples,)
        fs: Sampling rate in Hz (default 1000)
        window_ms: Window duration in milliseconds (default 50)
        step_ms: Step duration in milliseconds (default 1)
        freq_range: Frequency range to keep as (min_freq, max_freq) in Hz
        pad_mode: Padding mode for causality ('constant' for zero-padding)

    Returns:
        times: Time values for each spectrogram window (T_windows,)
        freqs: Frequency bin values (F_bins,)
        Sxx: Power spectral density magnitude (F_bins, T_windows)
    """
    assert signal_data.ndim == 1, f"Expected 1D signal, got shape {signal_data.shape}"
    assert fs > 0, "Sampling rate must be positive"
    assert window_ms > 0, "Window size must be positive"
    assert step_ms > 0, "Step size must be positive"

    # Convert time durations (ms) to sample counts
    nperseg = int(window_ms * fs / 1000)
    noverlap = nperseg - int(step_ms * fs / 1000)

    # Zero-pad beginning to ensure causality at start
    # Pad with nperseg-1 zeros to handle the first window properly
    padded_signal = np.pad(signal_data, (nperseg - 1, 0), mode=pad_mode)

    # Compute spectrogram using scipy
    # mode='magnitude' gives absolute values; 'psd' gives power spectral density
    freqs, times, Sxx = signal.spectrogram(
        padded_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",  # Use magnitude; can experiment with 'psd'
        scaling="spectrum",
    )

    # Filter frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    Sxx_filtered = Sxx[freq_mask, :]
    freqs_filtered = freqs[freq_mask]

    return freqs_filtered, times, Sxx_filtered


def compute_spectrograms(
    features: np.ndarray,
    fs: int = 1000,
    window_ms: int = 50,
    step_ms: int = 1,
    freq_range: Tuple[float, float] = (0, 500),
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute spectrograms for all 1024 channels in parallel.

    This function efficiently processes all channels by parallelizing across
    the channel dimension using joblib.

    Args:
        features: Raw ECoG features of shape (T_samples, 1024) or (1024, T_samples)
        fs: Sampling rate in Hz
        window_ms: Window duration in milliseconds
        step_ms: Step duration in milliseconds
        freq_range: Frequency range (min_freq, max_freq)
        n_jobs: Number of parallel jobs (-1 = use all CPUs)

    Returns:
        spectrograms: Array of shape (1024, T_windows, F_bins) ready for PyTorch
    """
    # Ensure shape is (T_samples, 1024) - time on rows, channels on columns
    if features.shape[0] == 1024:
        # If shape is (1024, T), transpose to (T, 1024)
        features = features.T
        logger.info(f"Transposed features from (1024, T) to (T, 1024)")

    n_samples, n_channels = features.shape
    assert n_channels == 1024, f"Expected 1024 channels, got {n_channels}"

    logger.info(
        f"Computing spectrograms for {n_channels} channels "
        f"({n_samples} samples, {n_samples/fs:.1f} seconds of data)"
    )

    # Compute spectrogram for each channel in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_channel_spectrogram)(
            features[:, ch],
            fs=fs,
            window_ms=window_ms,
            step_ms=step_ms,
            freq_range=freq_range,
        )
        for ch in tqdm(range(n_channels), desc="Computing spectrograms")
    )

    # Extract results
    freqs, times, spectrograms_list = zip(*results)

    # All channels should have same frequency bins and times
    freqs_common = freqs[0]
    times_common = times[0]

    # Stack all spectrograms: (1024, F_bins, T_windows) → transpose to (1024, T_windows, F_bins)
    spectrograms_stacked = np.stack(spectrograms_list, axis=0)  # (1024, F_bins, T_windows)
    spectrograms_final = np.transpose(spectrograms_stacked, (0, 2, 1))  # (1024, T_windows, F_bins)

    logger.info(
        f"✓ Computed spectrograms with shape {spectrograms_final.shape} "
        f"(channels, time_windows, freq_bins)"
    )
    logger.info(f"  - Frequency range: {freqs_common[0]:.1f} - {freqs_common[-1]:.1f} Hz")
    logger.info(f"  - Time bins: {len(times_common)} windows at {1000/step_ms:.0f} Hz effective rate")
    logger.info(f"  - Frequency bins: {len(freqs_common)} bins")

    return spectrograms_final


def save_spectrograms(spectrograms: np.ndarray, save_path: Path) -> None:
    """
    Save spectrograms as a PyTorch .pt file for efficient loading.

    Args:
        spectrograms: Numpy array of shape (1024, T_windows, F_bins)
        save_path: Path to save the .pt file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PyTorch tensor and save
    tensor = torch.from_numpy(spectrograms).float()
    torch.save(tensor, save_path)

    # Report file size
    file_size_gb = save_path.stat().st_size / (1024**3)
    logger.info(f"✓ Saved spectrograms to {save_path} ({file_size_gb:.2f} GB)")


def load_spectrograms(load_path: Path) -> torch.Tensor:
    """
    Load precomputed spectrograms from a .pt file.

    Args:
        load_path: Path to the .pt file

    Returns:
        PyTorch tensor of shape (1024, T_windows, F_bins)
    """
    load_path = Path(load_path)
    assert load_path.exists(), f"Spectrograms file not found: {load_path}"

    tensor = torch.load(load_path, weights_only=True)
    logger.info(
        f"✓ Loaded spectrograms from {load_path} with shape {tensor.shape}"
    )

    return tensor

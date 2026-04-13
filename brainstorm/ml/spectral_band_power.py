"""
SpectralBandPowerModel — causal spectral + EMA feature classifier.

Extracts 7 band-power features (delta through ultra-high gamma) from spatially
averaged 1024-channel ECoG, plus 3 EMA values at different time constants,
then classifies with sklearn LogisticRegression.

The model is purely causal: it uses only past/present samples, never future
data (no forward-looking filters or look-ahead windows).
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.linear_model import LogisticRegression

from brainstorm.constants import REPO_ROOT, SAMPLING_RATE
from brainstorm.ml.base import BaseModel

# ---------------------------------------------------------------------------
# Model persistence path (within the repo, <500 KB pickle)
# ---------------------------------------------------------------------------
MODEL_PATH = REPO_ROOT / "model.pkl"

# ---------------------------------------------------------------------------
# Spectral band definitions (Hz) — all bands capped at Nyquist = 500 Hz.
# Labels correspond to auditory cortex LFP frequency organisation.
# ---------------------------------------------------------------------------
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
    "high_gamma": (100, 300),
    "ultra_high": (300, 500),
}

# EMA decay constants (ms → alpha = 1 - exp(-1/tau) at 1 kHz)
EMA_TAUS_MS = [5, 20, 100]  # milliseconds


def _tau_to_alpha(tau_ms: float, fs: float = SAMPLING_RATE) -> float:
    """Convert time-constant in ms to per-sample EMA alpha."""
    tau_samples = tau_ms * fs / 1000.0  # τ in samples
    return float(1.0 - np.exp(-1.0 / tau_samples))


def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """
    Compute mean power in [low, high] Hz from a 1-D signal via real FFT.

    Causal: operates on a fixed buffer of past samples only.

    Args:
        signal: 1-D array of shape (N,).
        fs:     Sampling rate in Hz.
        low:    Lower band edge in Hz.
        high:   Upper band edge in Hz (inclusive).

    Returns:
        Mean squared magnitude of FFT coefficients in the band.
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal * np.hanning(N))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    power = np.mean(np.abs(fft_vals[mask]) ** 2)
    return float(power)


class SpectralBandPowerModel(BaseModel):
    """
    Causal streaming classifier using spectral band power + EMA features.

    Feature vector (10 total):
        - 7 band-power features: delta, theta, alpha, beta, gamma,
          high_gamma, ultra_high (each computed over the last ``buffer_size``
          samples of the spatially averaged ECoG signal).
        - 3 EMA features: exponential moving averages of the spatial average
          with time constants τ = 5, 20, 100 ms.

    Classifier: sklearn LogisticRegression (balanced class weights).

    Inference is sample-by-sample (causal); each predict() call:
        1. Spatially averages the 1024-channel input to one scalar.
        2. Updates the circular buffer and the three EMA accumulators.
        3. Computes the 10-d feature vector.
        4. Returns the LogisticRegression prediction.

    Attributes:
        classes_:    Numpy array of stimulus labels seen during training.
        buffer_size: Number of past samples kept for spectral analysis.
        ema_alphas:  Per-sample EMA decay coefficients derived from EMA_TAUS_MS.
    """

    # Expected stimulus class labels
    STIMULUS_CLASSES = np.array([0, 120, 224, 421, 789, 1479, 2772, 5195, 9736])

    def __init__(self, buffer_size: int = 100) -> None:
        """
        Args:
            buffer_size: Number of past samples to keep for FFT (default 100 = 100 ms at 1 kHz).
        """
        # nn.Module.__init__ — required because BaseModel inherits nn.Module.
        super().__init__()

        self.buffer_size = buffer_size
        self.ema_alphas: list[float] = [_tau_to_alpha(tau) for tau in EMA_TAUS_MS]

        # Dummy parameter so nn.Module is non-empty (some downstream utilities
        # iterate over parameters).  It plays no role in inference.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Sklearn classifier — fitted during fit_model()
        self._clf: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

        # Causal streaming state (initialised lazily on first predict() call)
        self._buffer: deque | None = None   # deque of floats, maxlen=buffer_size
        self._ema_states: list[float] | None = None  # one float per tau

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_streaming_state(self) -> None:
        """Reset buffer and EMA accumulators to zero (e.g., start of new trial)."""
        self._buffer = deque([0.0] * self.buffer_size, maxlen=self.buffer_size)
        self._ema_states = [0.0] * len(self.ema_alphas)

    def _ensure_streaming_state(self) -> None:
        if self._buffer is None or self._ema_states is None:
            self._reset_streaming_state()

    def _update_and_extract(self, spatial_avg: float) -> np.ndarray:
        """
        Push one scalar sample into the buffer + EMA accumulators,
        then return the 10-d feature vector.

        Args:
            spatial_avg: Spatial average across 1024 channels for this timestep.

        Returns:
            Feature vector of shape (10,).
        """
        self._ensure_streaming_state()

        # Update causal buffer (oldest sample is dropped automatically)
        self._buffer.append(spatial_avg)  # type: ignore[union-attr]

        # Update EMA accumulators
        for i, alpha in enumerate(self.ema_alphas):
            self._ema_states[i] = (  # type: ignore[index]
                alpha * spatial_avg + (1.0 - alpha) * self._ema_states[i]  # type: ignore[index]
            )

        # Compute spectral features from the buffer snapshot
        buf_arr = np.array(self._buffer, dtype=np.float64)
        band_feats = [
            _band_power(buf_arr, SAMPLING_RATE, low, high)
            for low, high in FREQ_BANDS.values()
        ]

        # Concatenate band power + EMA features
        features = np.array(band_feats + list(self._ema_states), dtype=np.float32)  # type: ignore[arg-type]
        return features

    def _extract_features_rolling(
        self, spatial_signal: np.ndarray
    ) -> np.ndarray:
        """
        Build the full feature matrix for a training sequence.

        Uses a rolling window of ``buffer_size`` samples to compute spectral
        features for each timestep.  EMA accumulators are reset to zero at the
        start so there is no look-ahead.

        Args:
            spatial_signal: 1-D array of shape (n_samples,) — spatially
                averaged ECoG across all channels.

        Returns:
            Feature matrix of shape (n_samples, n_features).
        """
        self._reset_streaming_state()
        features = np.zeros(
            (len(spatial_signal), len(FREQ_BANDS) + len(self.ema_alphas)),
            dtype=np.float32,
        )
        for i, s in enumerate(spatial_signal):
            features[i] = self._update_and_extract(float(s))
        return features

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Train the spectral + EMA feature LogisticRegression classifier.

        Args:
            X: ECoG data of shape (n_samples, n_channels) — n_channels = 1024.
            y: Integer labels of shape (n_samples,).
            **kwargs: Unused.
        """
        logger.info(
            f"Fitting SpectralBandPowerModel on {X.shape[0]} samples, "
            f"{X.shape[1]} channels..."
        )

        # Step 1: Spatial average → (n_samples,)
        spatial_signal = X.mean(axis=1)

        # Step 2: Build feature matrix using rolling causal window
        logger.info(
            f"Extracting {len(FREQ_BANDS)} band-power + {len(self.ema_alphas)} EMA features..."
        )
        feat_matrix = self._extract_features_rolling(spatial_signal)

        # Step 3: Store class labels
        self.classes_ = np.unique(y)
        logger.info(f"Classes: {self.classes_}")

        # Step 4: Fit LogisticRegression
        logger.info("Fitting LogisticRegression classifier...")
        self._clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="lbfgs",
            multi_class="multinomial",
        )
        self._clf.fit(feat_matrix, y)
        logger.info("LogisticRegression fitted successfully.")

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the stimulus label for a single sample.

        Maintains causal state between calls: the circular buffer and EMA
        accumulators accumulate history across successive predict() calls
        within a streaming session.

        Args:
            X: Single ECoG sample of shape (n_channels,) = (1024,).

        Returns:
            Predicted stimulus label as int (one of STIMULUS_CLASSES).
        """
        if self._clf is None or self.classes_ is None:
            raise RuntimeError(
                "Model not trained or loaded. Call fit() or SpectralBandPowerModel.load()."
            )

        # Spatial average of the 1024-channel input
        spatial_avg = float(np.mean(X))

        # Update buffer + EMAs, extract features
        features = self._update_and_extract(spatial_avg)

        # Predict with sklearn — returns array of shape (1,)
        label = int(self._clf.predict(features.reshape(1, -1))[0])
        return label

    def save(self) -> Path:
        """
        Pickle the sklearn model, classes, and EMA alphas to MODEL_PATH.

        Returns:
            Absolute Path to the saved pickle file.
        """
        if self._clf is None or self.classes_ is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        payload = {
            "clf": self._clf,
            "classes": self.classes_,
            "ema_alphas": self.ema_alphas,
            "buffer_size": self.buffer_size,
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Model saved to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
        return MODEL_PATH.resolve()

    @classmethod
    def load(cls) -> Self:
        """
        Load a SpectralBandPowerModel from MODEL_PATH.

        Returns:
            A new instance with loaded weights.

        Raises:
            FileNotFoundError: If MODEL_PATH does not exist.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model found at {MODEL_PATH}. "
                "Train and save a model first with fit()."
            )

        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        model = cls(buffer_size=payload["buffer_size"])
        model._clf = payload["clf"]
        model.classes_ = payload["classes"]
        model.ema_alphas = payload["ema_alphas"]
        logger.info(f"Model loaded from {MODEL_PATH}")
        return model  # type: ignore[return-value]

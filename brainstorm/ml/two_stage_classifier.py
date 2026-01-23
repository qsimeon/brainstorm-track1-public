"""
Two-Stage Classifier for continuous classification of ECoG signals.

This module provides a two-stage classification approach:
- Stage 1: Binary classifier to detect stimulus presence (silence vs any stimulus)
- Stage 2: Multi-class classifier to identify the specific frequency (only when stimulus detected)

Designed for low-latency, causal inference with a sliding history buffer.
"""

from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel


# Fixed model path within the repository
_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pt"


class HistoryBuffer:
    """
    Causal sliding window buffer for tracking history.

    Maintains a fixed-size buffer of past samples for feature extraction.
    Only uses past data (causal) - no future lookahead.
    """

    def __init__(self, buffer_size: int, n_channels: int):
        """
        Initialize the history buffer.

        Args:
            buffer_size: Number of past samples to track.
            n_channels: Number of channels per sample.
        """
        self.buffer_size = buffer_size
        self.n_channels = n_channels
        self.buffer = np.zeros((buffer_size, n_channels), dtype=np.float32)
        self.count = 0  # Number of samples seen

    def push(self, sample: np.ndarray) -> None:
        """
        Add a new sample to the buffer (FIFO).

        Args:
            sample: New sample of shape (n_channels,).
        """
        # Shift buffer and add new sample at the end
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = sample
        self.count += 1

    def get_features(self) -> np.ndarray:
        """
        Extract features from the current buffer state.

        Returns aggregated statistics over the history window:
        - Mean per channel
        - Std per channel
        - Current sample (most recent)

        Returns:
            Feature vector of shape (n_channels * 3,).
        """
        # Aggregated features to reduce dimensionality
        mean_features = self.buffer.mean(axis=0)
        std_features = self.buffer.std(axis=0)
        current_sample = self.buffer[-1]

        return np.concatenate([current_sample, mean_features, std_features])

    def reset(self) -> None:
        """Reset the buffer to zeros."""
        self.buffer.fill(0)
        self.count = 0


class Stage1Network(nn.Module):
    """Binary classifier: silence (0) vs stimulus (1)."""

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 2)  # Binary: silence vs stimulus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Stage2Network(nn.Module):
    """Multi-class classifier for frequency identification."""

    def __init__(self, input_size: int, hidden_size: int = 128, n_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TwoStageClassifier(BaseModel):
    """
    Two-Stage Classifier for low-latency ECoG classification.

    Architecture:
        - History Buffer: Tracks past 50ms (50 samples) of data
        - Stage 1: Binary MLP (silence vs stimulus)
        - Stage 2: Multi-class MLP (frequency classification, only when Stage 1 detects stimulus)

    This approach is designed to:
        1. Handle class imbalance (67% silence) efficiently
        2. Reduce latency by quickly filtering silence
        3. Keep model size small for edge deployment

    Attributes:
        history_size: Number of past samples to track (default: 50 = 50ms).
        hidden_size: Hidden layer size for both stages.
        stage1_classes_: Binary class labels [0, 1].
        stage2_classes_: Frequency class labels (non-zero frequencies).

    Example:
        >>> model = TwoStageClassifier(history_size=50, hidden_size=128)
        >>> model.fit(train_features, train_labels)
        >>>
        >>> # Load for inference
        >>> model = TwoStageClassifier.load()
        >>> prediction = model.predict(sample)  # Called per timestep
    """

    def __init__(
        self,
        history_size: int = 50,
        hidden_size: int = 128,
        dropout: float = 0.3,
        n_channels: int = N_CHANNELS,
    ) -> None:
        """
        Initialize the Two-Stage Classifier.

        Args:
            history_size: Number of past samples to track (ms at 1kHz). Default: 50.
            hidden_size: Hidden layer size for both MLPs. Default: 128.
            dropout: Dropout rate for regularization. Default: 0.3.
            n_channels: Number of input channels. Default: 1024.
        """
        super().__init__()

        self.history_size = history_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.n_channels = n_channels

        # Feature size: current + mean + std over history
        self.feature_size = n_channels * 3

        # Class mappings (set during training)
        self.stage1_classes_: np.ndarray | None = None  # [0, 1]
        self.stage2_classes_: np.ndarray | None = None  # Non-zero frequencies
        self.all_classes_: np.ndarray | None = None  # All original classes

        # Networks (initialized during training)
        self.stage1: Stage1Network | None = None
        self.stage2: Stage2Network | None = None

        # History buffer for inference
        self._buffer: HistoryBuffer | None = None

        # Normalization parameters
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _build_networks(self, n_stage2_classes: int) -> None:
        """Build the stage networks."""
        self.stage1 = Stage1Network(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout_rate
        )
        self.stage2 = Stage2Network(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            n_classes=n_stage2_classes,
            dropout=self.dropout_rate
        )

    def _extract_history_features(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract features with history context for training.

        Simulates the streaming inference by building history-aware features.

        Args:
            X: Raw features of shape (n_samples, n_channels).
            y: Labels of shape (n_samples,).

        Returns:
            Tuple of (features, labels) with history context.
        """
        n_samples = len(X)
        features = np.zeros((n_samples, self.feature_size), dtype=np.float32)

        # Use a buffer to simulate streaming
        buffer = HistoryBuffer(self.history_size, self.n_channels)

        for i in range(n_samples):
            buffer.push(X[i])
            features[i] = buffer.get_features()

        return features, y

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Train both stages of the classifier.

        Args:
            X: Feature array of shape (n_samples, n_channels).
            y: Label array of shape (n_samples,).
            epochs: Number of training epochs per stage. Default: 30.
            batch_size: Mini-batch size. Default: 256.
            learning_rate: Learning rate for Adam optimizer. Default: 1e-3.
            verbose: Whether to show training progress. Default: True.
        """
        # Store all classes for reference
        self.all_classes_ = np.unique(y)

        # Separate classes for each stage
        self.stage1_classes_ = np.array([0, 1])  # Binary
        self.stage2_classes_ = np.array([c for c in self.all_classes_ if c != 0])

        n_stage2_classes = len(self.stage2_classes_)
        logger.info(f"Stage 1: Binary classification (silence vs stimulus)")
        logger.info(f"Stage 2: {n_stage2_classes} frequency classes: {self.stage2_classes_.tolist()}")

        # Build networks
        self._build_networks(n_stage2_classes)

        # Compute normalization parameters from raw data
        self._mean = X.mean(axis=0).astype(np.float32)
        self._std = X.std(axis=0).astype(np.float32)
        self._std[self._std < 1e-6] = 1.0  # Avoid division by zero

        # Normalize raw data
        X_normalized = (X - self._mean) / self._std

        # Extract history features
        logger.info("Extracting history features...")
        features, labels = self._extract_history_features(X_normalized, y)

        # Create binary labels for Stage 1
        binary_labels = (labels != 0).astype(np.int64)

        # Train Stage 1 (binary: silence vs stimulus)
        logger.info("Training Stage 1 (silence vs stimulus)...")
        self._train_stage(
            self.stage1,
            features,
            binary_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
            stage_name="Stage 1"
        )

        # Train Stage 2 (frequency classification on stimulus samples only)
        logger.info("Training Stage 2 (frequency classification)...")

        # Filter to only stimulus samples
        stimulus_mask = labels != 0
        stimulus_features = features[stimulus_mask]
        stimulus_labels = labels[stimulus_mask]

        # Map frequency labels to indices
        stage2_class_to_idx = {c: i for i, c in enumerate(self.stage2_classes_)}
        stimulus_indices = np.array([stage2_class_to_idx[l] for l in stimulus_labels])

        self._train_stage(
            self.stage2,
            stimulus_features,
            stimulus_indices,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
            stage_name="Stage 2"
        )

        # Initialize inference buffer
        self._buffer = HistoryBuffer(self.history_size, self.n_channels)

        # Set to eval mode
        self.stage1.eval()
        self.stage2.eval()

        logger.info("Training complete.")

    def _train_stage(
        self,
        network: nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        verbose: bool,
        stage_name: str,
    ) -> None:
        """Train a single stage network."""
        # Convert to tensors
        X_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Setup training
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        avg_loss = 0.0
        epoch_iterator = tqdm(range(epochs), desc=stage_name, disable=not verbose)
        for epoch in epoch_iterator:
            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = network(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")

        network.eval()
        logger.info(f"{stage_name} training complete. Final loss: {avg_loss:.4f}")

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the label for a single sample (streaming inference).

        This method is called once per timestep. It:
        1. Adds the sample to the history buffer
        2. Extracts features from the buffer
        3. Runs Stage 1 to detect stimulus
        4. If stimulus detected, runs Stage 2 for frequency

        Args:
            X: Feature array of shape (n_channels,) for a single timestep.

        Returns:
            Predicted label as an integer (frequency in Hz, or 0 for silence).
        """
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("Model not trained. Call fit() first or load a trained model.")

        if self._buffer is None:
            self._buffer = HistoryBuffer(self.history_size, self.n_channels)

        # Normalize the input
        X_normalized = (X - self._mean) / self._std

        # Update history buffer
        self._buffer.push(X_normalized)

        # Extract features
        features = self._buffer.get_features()

        # Convert to tensor
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Stage 1: Detect stimulus presence
            stage1_logits = self.stage1(x_tensor)
            is_stimulus = torch.argmax(stage1_logits, dim=1).item()

            if is_stimulus == 0:
                # Silence detected
                return 0

            # Stage 2: Classify frequency
            stage2_logits = self.stage2(x_tensor)
            freq_idx = torch.argmax(stage2_logits, dim=1).item()

            return int(self.stage2_classes_[freq_idx])

    def reset_buffer(self) -> None:
        """Reset the history buffer (call between sequences if needed)."""
        if self._buffer is not None:
            self._buffer.reset()

    def save(self) -> Path:
        """
        Save the model to model.pt.

        Returns:
            Path to the saved model file.
        """
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        checkpoint = {
            "config": {
                "history_size": self.history_size,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout_rate,
                "n_channels": self.n_channels,
                "feature_size": self.feature_size,
            },
            "stage1_classes": self.stage1_classes_,
            "stage2_classes": self.stage2_classes_,
            "all_classes": self.all_classes_,
            "stage1_state_dict": self.stage1.state_dict(),
            "stage2_state_dict": self.stage2.state_dict(),
            "normalization": {
                "mean": self._mean,
                "std": self._std,
            },
        }

        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"Model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """
        Load a model from model.pt.

        Returns:
            A new instance of TwoStageClassifier with loaded weights.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using TwoStageClassifier.fit()."
            )

        checkpoint = torch.load(MODEL_PATH, weights_only=False)

        # Reconstruct the model
        config = checkpoint["config"]
        model = cls(
            history_size=config["history_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            n_channels=config["n_channels"],
        )

        # Restore class mappings
        model.stage1_classes_ = checkpoint["stage1_classes"]
        model.stage2_classes_ = checkpoint["stage2_classes"]
        model.all_classes_ = checkpoint["all_classes"]

        # Restore normalization parameters
        model._mean = checkpoint["normalization"]["mean"]
        model._std = checkpoint["normalization"]["std"]

        # Rebuild and load networks
        n_stage2_classes = len(model.stage2_classes_)
        model._build_networks(n_stage2_classes)
        model.stage1.load_state_dict(checkpoint["stage1_state_dict"])
        model.stage2.load_state_dict(checkpoint["stage2_state_dict"])

        # Set to eval mode
        model.stage1.eval()
        model.stage2.eval()

        # Initialize inference buffer
        model._buffer = HistoryBuffer(model.history_size, model.n_channels)

        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model

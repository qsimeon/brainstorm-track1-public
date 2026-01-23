"""
EEGNet-based model for continuous classification of ECoG signals.

This module provides a compact convolutional neural network architecture
based on the EEGNet design, adapted for high-density ECoG recordings.
The architecture uses depthwise separable convolutions for efficiency.

Reference:
    Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural Network
    for EEG-based Brain-Computer Interfaces"
"""

from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from brainstorm.constants import N_CHANNELS, SAMPLING_RATE
from brainstorm.ml.base import BaseModel
from brainstorm.ml.channel_projection import PCAProjection


_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pt"


class EEGNetCore(nn.Module):
    """
    Core EEGNet architecture with depthwise separable convolutions.

    This implements a streamlined version of EEGNet suitable for
    single-timestep predictions with temporal context from a sliding window.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 10,
        window_samples: int = 128,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.25,
    ) -> None:
        """
        Initialize EEGNet core architecture.

        Args:
            n_channels: Number of input channels (after projection).
            n_classes: Number of output classes.
            window_samples: Number of time samples in input window.
            F1: Number of temporal filters in first conv layer.
            D: Depth multiplier for depthwise convolution.
            F2: Number of pointwise filters.
            kernel_length: Length of temporal convolution kernel.
            dropout_rate: Dropout probability.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.window_samples = window_samples
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise spatial convolution
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 3: Separable convolution
        self.separable1 = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.separable2 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, window_samples)
            dummy = self._forward_features(dummy)
            self.flat_size = dummy.numel()

        # Classifier
        self.fc = nn.Linear(self.flat_size, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through convolutional blocks."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)

        # Block 2
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 3
        x = self.separable1(x)
        x = self.separable2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EEGNet.

        Args:
            x: Input tensor of shape (batch, 1, channels, time_samples).

        Returns:
            Logits of shape (batch, n_classes).
        """
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class EEGNet(BaseModel):
    """
    EEGNet model adapted for high-density ECoG recordings.

    Uses PCA for channel reduction from 1024 to a smaller number of channels,
    followed by a compact EEGNet architecture. Maintains a sliding window
    buffer for temporal context during streaming inference.

    Attributes:
        projected_channels: Number of channels after PCA projection.
        window_size: Number of samples in the temporal window.
        classes_: Array of unique class labels learned during fit().

    Example:
        >>> model = EEGNet(projected_channels=64, window_size=128)
        >>> model.fit(train_features, train_labels)
        >>>
        >>> # Load for inference
        >>> model = EEGNet.load()
        >>> prediction = model.predict(sample)
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        projected_channels: int = 64,
        window_size: int = 128,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.25,
    ) -> None:
        """
        Initialize EEGNet model.

        Args:
            input_size: Number of input channels from ECoG array.
            projected_channels: Number of channels after PCA projection.
            window_size: Number of time samples for temporal context.
            F1: Number of temporal filters.
            D: Depthwise multiplier.
            dropout: Dropout rate.
        """
        super().__init__()

        self.input_size = input_size
        self.projected_channels = projected_channels
        self.window_size = window_size
        self.F1 = F1
        self.D = D
        self.dropout_rate = dropout

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None

        # PCA projection (fitted during training)
        self.pca: PCAProjection | None = None
        self.pca_layer: nn.Linear | None = None

        # EEGNet core (built after knowing n_classes)
        self.eegnet: EEGNetCore | None = None

        # Sliding window buffer for inference
        self._window_buffer: np.ndarray | None = None

    def _build_network(self, n_classes: int) -> None:
        """Build the network after knowing n_classes."""
        self._n_classes = n_classes
        self.eegnet = EEGNetCore(
            n_channels=self.projected_channels,
            n_classes=n_classes,
            window_samples=self.window_size,
            F1=self.F1,
            D=self.D,
            F2=self.F1 * self.D,
            dropout_rate=self.dropout_rate,
        )

    def _init_window_buffer(self) -> None:
        """Initialize the sliding window buffer."""
        self._window_buffer = np.zeros(
            (self.window_size, self.projected_channels), dtype=np.float32
        )

    def _update_buffer(self, projected_sample: np.ndarray) -> np.ndarray:
        """
        Update sliding window buffer with new sample.

        Args:
            projected_sample: New sample of shape (projected_channels,).

        Returns:
            Current window of shape (window_size, projected_channels).
        """
        if self._window_buffer is None:
            self._init_window_buffer()

        # Shift buffer and add new sample
        self._window_buffer = np.roll(self._window_buffer, -1, axis=0)
        self._window_buffer[-1] = projected_sample

        return self._window_buffer.copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, 1, projected_channels, window_size).

        Returns:
            Logits of shape (batch, n_classes).
        """
        if self.eegnet is None:
            raise RuntimeError("Network not built. Call fit() or load a trained model.")
        return self.eegnet(x)

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """
        Train the EEGNet model.

        Args:
            X: Feature array of shape (n_samples, n_channels).
            y: Label array of shape (n_samples,).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Learning rate for AdamW optimizer.
            weight_decay: Weight decay for regularization.
            verbose: Whether to show training progress.
            X_val: Optional validation features of shape (n_samples, n_channels).
            y_val: Optional validation labels of shape (n_samples,).
        """
        # Determine classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training EEGNet with {n_classes} classes: {self.classes_.tolist()}")
        logger.info(f"Fitting PCA projection: {self.input_size} -> {self.projected_channels} channels")

        # Fit PCA on training data
        self.pca = PCAProjection(n_components=self.projected_channels)
        X_projected = self.pca.fit_transform(X)
        self.pca_layer = self.pca.get_torch_projection()

        # Build EEGNet
        self._build_network(n_classes)

        # Create windowed samples for training
        logger.info(f"Creating windowed training data with window_size={self.window_size}")
        X_windows, y_windows = self._create_windowed_data(X_projected, y)

        logger.info(f"Training data: {X_windows.shape[0]} windows")

        # Convert to tensors
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        # Reshape to (batch, 1, channels, time)
        X_tensor = X_tensor.permute(0, 2, 1).unsqueeze(1)

        y_indices = np.array([class_to_idx[label] for label in y_windows])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Prepare validation data if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            logger.info("Preparing validation data...")
            X_val_projected = self.pca.transform(X_val)
            X_val_windows, y_val_windows = self._create_windowed_data(X_val_projected, y_val)
            logger.info(f"Validation data: {X_val_windows.shape[0]} windows")

            X_val_tensor = torch.tensor(X_val_windows, dtype=torch.float32)
            X_val_tensor = X_val_tensor.permute(0, 2, 1).unsqueeze(1)
            y_val_indices = np.array([class_to_idx[label] for label in y_val_windows])
            y_val_tensor = torch.tensor(y_val_indices, dtype=torch.long)

            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Setup device
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Training on device: {device}")

        # Move model to device
        self.to(device)

        # Compute class weights for imbalanced data
        class_counts = np.bincount(y_indices)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * n_classes  # Normalize
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        logger.info(f"Class weights: {class_weights.round(2).tolist()}")

        # Training setup
        self.train()
        optimizer = torch.optim.AdamW(self.eegnet.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Best model tracking
        best_val_acc = 0.0
        best_checkpoint_path = Path("/media/M2SSD/mind_meld_checkpoints/eegnet_best.pt")

        # Training loop
        avg_loss = 0.0

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            # Progress bar for batches within epoch
            batch_iterator = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not verbose,
                leave=True,
            )

            for X_batch, y_batch in batch_iterator:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.eegnet.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                batch_iterator.set_postfix(loss=f"{total_loss/n_batches:.4f}")

            scheduler.step()
            avg_loss = total_loss / n_batches

            # Evaluate on validation set if provided
            if val_loader is not None:
                self.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch = X_val_batch.to(device)
                        logits = self.forward(X_val_batch)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(y_val_batch.numpy())
                self.train()

                val_bal_acc = balanced_accuracy_score(all_labels, all_preds)

                # Save best model
                if val_bal_acc > best_val_acc:
                    best_val_acc = val_bal_acc
                    checkpoint = {
                        "config": {
                            "input_size": self.input_size,
                            "projected_channels": self.projected_channels,
                            "window_size": self.window_size,
                            "n_classes": self._n_classes,
                            "F1": self.F1,
                            "D": self.D,
                            "dropout": self.dropout_rate,
                        },
                        "classes": self.classes_,
                        "pca_mean": self.pca.mean_,
                        "pca_components": self.pca.components_,
                        "eegnet_state_dict": self.eegnet.state_dict(),
                        "epoch": epoch + 1,
                        "val_bal_acc": val_bal_acc,
                    }
                    torch.save(checkpoint, best_checkpoint_path)
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Balanced Acc: {val_bal_acc:.4f} [BEST - saved to {best_checkpoint_path}]")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Balanced Acc: {val_bal_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Move back to CPU for inference
        self.to("cpu")
        self.eval()
        self._init_window_buffer()
        logger.info(f"Training complete. Final loss: {avg_loss:.4f}")
        if best_val_acc > 0:
            logger.info(f"Best model saved to {best_checkpoint_path} with Val Balanced Acc: {best_val_acc:.4f}")

    def _create_windowed_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create windowed training samples.

        Args:
            X: Projected features of shape (n_samples, projected_channels).
            y: Labels of shape (n_samples,).

        Returns:
            Tuple of (windowed_X, windowed_y) where:
                windowed_X: shape (n_windows, window_size, projected_channels)
                windowed_y: shape (n_windows,)
        """
        n_samples = X.shape[0]
        windows = []
        labels = []

        for i in range(self.window_size, n_samples):
            window = X[i - self.window_size : i]
            windows.append(window)
            labels.append(y[i - 1])  # Label for the last sample in window

        return np.array(windows), np.array(labels)

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the label for a single sample.

        Args:
            X: Feature array of shape (n_channels,) for a single timestep.

        Returns:
            Predicted label as an integer.
        """
        if self.classes_ is None or self.pca_layer is None or self.eegnet is None:
            raise RuntimeError("Model not trained. Call fit() or load a trained model.")

        self.eval()
        with torch.no_grad():
            # Project input channels
            x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            x_projected = self.pca_layer(x_tensor).squeeze(0).numpy()

            # Update sliding window buffer
            window = self._update_buffer(x_projected)

            # Prepare for network: (1, 1, channels, time)
            window_tensor = torch.tensor(window, dtype=torch.float32)
            window_tensor = window_tensor.T.unsqueeze(0).unsqueeze(0)

            # Get prediction
            logits = self.forward(window_tensor)
            predicted_idx = int(torch.argmax(logits, dim=1).item())

        return int(self.classes_[predicted_idx])

    def save(self) -> Path:
        """
        Save the model weights and configuration.

        Returns:
            Path to the saved model file.
        """
        if self.classes_ is None or self.pca is None or self.eegnet is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        checkpoint = {
            "config": {
                "input_size": self.input_size,
                "projected_channels": self.projected_channels,
                "window_size": self.window_size,
                "n_classes": self._n_classes,
                "F1": self.F1,
                "D": self.D,
                "dropout": self.dropout_rate,
            },
            "classes": self.classes_,
            "pca_mean": self.pca.mean_,
            "pca_components": self.pca.components_,
            "eegnet_state_dict": self.eegnet.state_dict(),
        }

        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"EEGNet model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """
        Load a model from the saved checkpoint.

        Returns:
            A new instance of EEGNet with loaded weights.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using EEGNet.fit()"
            )

        checkpoint = torch.load(MODEL_PATH, weights_only=False)

        config = checkpoint["config"]
        model = cls(
            input_size=config["input_size"],
            projected_channels=config["projected_channels"],
            window_size=config["window_size"],
            F1=config["F1"],
            D=config["D"],
            dropout=config["dropout"],
        )

        # Restore PCA projection
        model.pca = PCAProjection(n_components=config["projected_channels"])
        model.pca.mean_ = checkpoint["pca_mean"]
        model.pca.components_ = checkpoint["pca_components"]
        model.pca.pca = True  # Mark as fitted
        model.pca_layer = model.pca.get_torch_projection()

        # Restore classes and network
        model.classes_ = checkpoint["classes"]
        model._build_network(config["n_classes"])
        model.eegnet.load_state_dict(checkpoint["eegnet_state_dict"])
        model.eval()

        # Initialize inference buffer
        model._init_window_buffer()

        logger.debug(f"EEGNet model loaded from {MODEL_PATH}")
        return model

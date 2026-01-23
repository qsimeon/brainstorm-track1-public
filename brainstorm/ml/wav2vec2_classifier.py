"""
Wav2Vec2-based classifier using pretrained weights for ECoG signals.

This module adapts the tiny Wav2Vec2 model for ECoG signal classification.
Wav2Vec2 was designed for audio waveforms but can process any 1D signal.

The pretrained encoder learns general temporal patterns that can transfer
to neural signal classification.

Key features:
- Pretrained weights from wav2vec2_tiny_random (1.2MB)
- Very fast inference (~0.5ms)
- Requires minimum ~400 samples due to convolutional architecture
"""

from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Config

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel
from brainstorm.ml.channel_projection import PCAProjection


_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pt"

# Minimum samples required by wav2vec2 due to conv kernel sizes
# wav2vec2_tiny_random needs ~1600 samples minimum
MIN_SAMPLES = 1600


class Wav2Vec2Classifier(BaseModel):
    """
    Wav2Vec2-based classifier with pretrained encoder.

    Uses the tiny wav2vec2 model (~1.2MB) pretrained on audio data.
    The encoder learns temporal patterns that transfer to ECoG signals.

    Since wav2vec2 requires minimum ~400 samples, we use a larger window
    or pad shorter windows.

    Attributes:
        projected_channels: Number of channels after PCA (processed independently).
        window_size: Number of samples in temporal window (min 400).
        freeze_encoder: Whether to freeze pretrained weights.
        classes_: Array of unique class labels learned during fit().

    Example:
        >>> model = Wav2Vec2Classifier(window_size=512, freeze_encoder=True)
        >>> model.fit(train_features, train_labels)
        >>> prediction = model.predict(sample)
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        projected_channels: int = 8,  # Process fewer channels for speed
        window_size: int = 512,  # Larger window for wav2vec2
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        model_name: str = "patrickvonplaten/wav2vec2_tiny_random",
    ) -> None:
        """
        Initialize Wav2Vec2Classifier.

        Args:
            input_size: Number of input channels from ECoG array.
            projected_channels: Number of channels after PCA (each processed separately).
            window_size: Number of time samples (minimum 400 for wav2vec2).
            freeze_encoder: Whether to freeze encoder during training.
            dropout: Dropout rate for classification head.
            model_name: HuggingFace model name for wav2vec2.
        """
        super().__init__()

        if window_size < MIN_SAMPLES:
            logger.warning(
                f"window_size {window_size} < {MIN_SAMPLES}. "
                f"Will pad inputs to {MIN_SAMPLES} samples."
            )

        self.input_size = input_size
        self.projected_channels = projected_channels
        self.window_size = window_size
        self.effective_window = max(window_size, MIN_SAMPLES)
        self.freeze_encoder = freeze_encoder
        self.dropout_rate = dropout
        self.model_name = model_name

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None

        # PCA projection
        self.pca: PCAProjection | None = None
        self.pca_layer: nn.Linear | None = None

        # Wav2Vec2 encoder
        self.encoder: Wav2Vec2Model | None = None
        self._hidden_dim: int | None = None

        # Classification head
        self.classifier: nn.Module | None = None

        # Sliding window buffer
        self._window_buffer: np.ndarray | None = None

    def _build_network(self, n_classes: int) -> None:
        """Build the network after knowing n_classes."""
        self._n_classes = n_classes

        logger.info(f"Loading pretrained Wav2Vec2: {self.model_name}")

        # Load pretrained wav2vec2
        self.encoder = Wav2Vec2Model.from_pretrained(self.model_name)

        # Disable masking - it causes issues with short sequences
        self.encoder.config.mask_time_prob = 0.0
        self.encoder.config.mask_feature_prob = 0.0

        self._hidden_dim = self.encoder.config.hidden_size

        logger.info(f"Wav2Vec2 hidden dim: {self._hidden_dim}")

        # Freeze encoder if requested
        if self.freeze_encoder:
            logger.info("Freezing Wav2Vec2 encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        # Input: (batch, projected_channels, hidden_dim) after pooling over time
        self.classifier = nn.Sequential(
            nn.Linear(self.projected_channels * self._hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, n_classes),
        )

        # Log parameter counts
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_encoder = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        logger.info(f"Encoder params: {encoder_params:,} (trainable: {trainable_encoder:,})")
        logger.info(f"Classifier params: {classifier_params:,}")

    def _init_window_buffer(self) -> None:
        """Initialize the sliding window buffer."""
        self._window_buffer = np.zeros(
            (self.effective_window, self.projected_channels), dtype=np.float32
        )

    def _update_buffer(self, projected_sample: np.ndarray) -> np.ndarray:
        """Update sliding window buffer with new sample."""
        if self._window_buffer is None:
            self._init_window_buffer()

        self._window_buffer = np.roll(self._window_buffer, -1, axis=0)
        self._window_buffer[-1] = projected_sample
        return self._window_buffer.copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Logits of shape (batch, n_classes).
        """
        if self.encoder is None or self.classifier is None:
            raise RuntimeError("Network not built. Call fit() or load a trained model.")

        batch_size, n_channels, seq_len = x.shape

        # Process each channel through wav2vec2 encoder
        # Reshape to (batch * channels, seq_len)
        x_flat = x.reshape(batch_size * n_channels, seq_len)

        # Encode - disable masking by setting mask_time_prob=0 or using output_hidden_states
        # We pass mask_time_indices=None and set the model to not apply masking
        encoder_output = self.encoder(
            x_flat,
            mask_time_indices=None,
            output_hidden_states=False,
        )
        hidden_states = encoder_output.last_hidden_state  # (batch*channels, time', hidden)

        # Pool over time dimension
        pooled = hidden_states.mean(dim=1)  # (batch*channels, hidden)

        # Reshape back to (batch, channels * hidden)
        pooled = pooled.reshape(batch_size, n_channels * self._hidden_dim)

        # Classify
        return self.classifier(pooled)

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        encoder_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """
        Train the Wav2Vec2Classifier model.

        Args:
            X: Feature array of shape (n_samples, n_channels).
            y: Label array of shape (n_samples,).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Learning rate for classification head.
            encoder_lr: Learning rate for encoder (if not frozen).
            weight_decay: Weight decay for regularization.
            verbose: Whether to show training progress.
            X_val: Optional validation features of shape (n_samples, n_channels).
            y_val: Optional validation labels of shape (n_samples,).
        """
        # Determine classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training Wav2Vec2Classifier with {n_classes} classes")
        logger.info(f"Fitting PCA projection: {self.input_size} -> {self.projected_channels} channels")

        # Fit PCA
        self.pca = PCAProjection(n_components=self.projected_channels)
        X_projected = self.pca.fit_transform(X)
        self.pca_layer = self.pca.get_torch_projection()

        # Build network
        self._build_network(n_classes)

        # Create windowed samples
        logger.info(f"Creating windowed data with effective_window={self.effective_window}")
        X_windows, y_windows = self._create_windowed_data(X_projected, y)
        logger.info(f"Training data: {X_windows.shape[0]} windows")

        # Convert to tensors: (batch, time, channels) -> (batch, channels, time)
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        X_tensor = X_tensor.permute(0, 2, 1)  # (batch, channels, time)

        y_indices = np.array([class_to_idx[label] for label in y_windows])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Data loader
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
            X_val_tensor = X_val_tensor.permute(0, 2, 1)  # (batch, channels, time)
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

        # Optimizer
        if self.freeze_encoder:
            optimizer = torch.optim.AdamW(
                self.classifier.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW([
                {"params": self.encoder.parameters(), "lr": encoder_lr},
                {"params": self.classifier.parameters(), "lr": learning_rate},
            ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Best model tracking
        best_val_acc = 0.0
        best_checkpoint_path = Path("/media/M2SSD/mind_meld_checkpoints/wav2vec2_best.pt")

        # Training loop
        self.train()
        avg_loss = 0.0

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

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

                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                if not self.freeze_encoder:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)

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
                            "freeze_encoder": self.freeze_encoder,
                            "dropout": self.dropout_rate,
                            "model_name": self.model_name,
                            "n_classes": self._n_classes,
                            "hidden_dim": self._hidden_dim,
                        },
                        "classes": self.classes_,
                        "pca_mean": self.pca.mean_,
                        "pca_components": self.pca.components_,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "classifier_state_dict": self.classifier.state_dict(),
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
        """Create windowed training samples with padding if needed."""
        n_samples = X.shape[0]
        windows = []
        labels = []

        for i in range(self.effective_window, n_samples):
            window = X[i - self.effective_window : i]
            windows.append(window)
            labels.append(y[i - 1])

        return np.array(windows), np.array(labels)

    def predict(self, X: np.ndarray) -> int:
        """Predict the label for a single sample."""
        if self.classes_ is None or self.pca_layer is None or self.encoder is None:
            raise RuntimeError("Model not trained. Call fit() or load a trained model.")

        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            x_projected = self.pca_layer(x_tensor).squeeze(0).numpy()

            window = self._update_buffer(x_projected)

            # (1, channels, time)
            window_tensor = torch.tensor(window, dtype=torch.float32)
            window_tensor = window_tensor.T.unsqueeze(0)

            logits = self.forward(window_tensor)
            predicted_idx = int(torch.argmax(logits, dim=1).item())

        return int(self.classes_[predicted_idx])

    def save(self) -> Path:
        """Save the model weights and configuration."""
        if self.classes_ is None or self.pca is None or self.encoder is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        checkpoint = {
            "config": {
                "input_size": self.input_size,
                "projected_channels": self.projected_channels,
                "window_size": self.window_size,
                "freeze_encoder": self.freeze_encoder,
                "dropout": self.dropout_rate,
                "model_name": self.model_name,
                "n_classes": self._n_classes,
                "hidden_dim": self._hidden_dim,
            },
            "classes": self.classes_,
            "pca_mean": self.pca.mean_,
            "pca_components": self.pca.components_,
            "encoder_state_dict": self.encoder.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
        }

        torch.save(checkpoint, MODEL_PATH)
        logger.info(f"Wav2Vec2Classifier saved to {MODEL_PATH}")

        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB")

        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """Load a model from the saved checkpoint."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using Wav2Vec2Classifier.fit()"
            )

        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        config = checkpoint["config"]

        model = cls(
            input_size=config["input_size"],
            projected_channels=config["projected_channels"],
            window_size=config["window_size"],
            freeze_encoder=config["freeze_encoder"],
            dropout=config["dropout"],
            model_name=config["model_name"],
        )

        # Restore PCA
        model.pca = PCAProjection(n_components=config["projected_channels"])
        model.pca.mean_ = checkpoint["pca_mean"]
        model.pca.components_ = checkpoint["pca_components"]
        model.pca.pca = True
        model.pca_layer = model.pca.get_torch_projection()

        # Restore network
        model.classes_ = checkpoint["classes"]
        model._build_network(config["n_classes"])

        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        model.eval()

        model._init_window_buffer()
        logger.info(f"Wav2Vec2Classifier loaded from {MODEL_PATH}")

        return model

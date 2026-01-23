"""
Chronos-based classifier for continuous classification of ECoG signals.

This module adapts Amazon's Chronos time series foundation model for
classification tasks. Chronos is a pretrained T5-based model that learns
general time series representations through tokenization and language modeling.

We use the Chronos encoder as a feature extractor and add a classification head
on top. The encoder can be frozen (transfer learning) or fine-tuned.

Reference:
    Ansari et al. (2024) "Chronos: Learning the Language of Time Series"
    https://arxiv.org/abs/2403.07815
"""

from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel
from brainstorm.ml.channel_projection import PCAProjection


_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pt"


class ChronosClassificationHead(nn.Module):
    """Classification head for Chronos encoder outputs."""

    def __init__(
        self,
        hidden_dim: int,
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, hidden_dim)
        Returns:
            logits: (batch, n_classes)
        """
        # Pool over sequence dimension (mean pooling)
        pooled = encoder_output.mean(dim=1)
        return self.classifier(pooled)


class ChronosClassifier(BaseModel):
    """
    Chronos-based classifier for ECoG signal classification.

    Uses Amazon's Chronos T5-Tiny foundation model as a pretrained encoder,
    with a classification head added on top. The encoder learns general
    time series patterns through pretraining on diverse time series data.

    The model:
    1. Projects 1024 ECoG channels to fewer channels via PCA
    2. Tokenizes the time series using Chronos's learned tokenizer
    3. Encodes tokens using the pretrained T5 encoder
    4. Classifies using a learned classification head

    Attributes:
        projected_channels: Number of channels after PCA projection.
        window_size: Number of samples in the temporal window.
        freeze_encoder: Whether to freeze encoder weights during training.
        classes_: Array of unique class labels learned during fit().

    Example:
        >>> model = ChronosClassifier(projected_channels=64, window_size=128)
        >>> model.fit(train_features, train_labels)
        >>>
        >>> model = ChronosClassifier.load()
        >>> prediction = model.predict(sample)
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        projected_channels: int = 32,
        window_size: int = 128,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        model_name: str = "amazon/chronos-t5-tiny",
    ) -> None:
        """
        Initialize ChronosClassifier.

        Args:
            input_size: Number of input channels from ECoG array.
            projected_channels: Number of channels after PCA projection.
            window_size: Number of time samples for temporal context.
            freeze_encoder: Whether to freeze encoder during training.
            dropout: Dropout rate for classification head.
            model_name: HuggingFace model name for Chronos.
        """
        super().__init__()

        self.input_size = input_size
        self.projected_channels = projected_channels
        self.window_size = window_size
        self.freeze_encoder = freeze_encoder
        self.dropout_rate = dropout
        self.model_name = model_name

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None

        # PCA projection (fitted during training)
        self.pca: PCAProjection | None = None
        self.pca_layer: nn.Linear | None = None

        # Chronos components (loaded during build)
        self.chronos_pipeline = None
        self.tokenizer = None
        self.encoder: nn.Module | None = None
        self.classifier: ChronosClassificationHead | None = None

        # Hidden dimension from Chronos
        self._hidden_dim: int | None = None

        # Sliding window buffer for inference
        self._window_buffer: np.ndarray | None = None

    def _build_network(self, n_classes: int) -> None:
        """Build the network after knowing n_classes."""
        from chronos import ChronosPipeline

        self._n_classes = n_classes

        logger.info(f"Loading Chronos foundation model: {self.model_name}")

        # Load Chronos
        self.chronos_pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map="cpu",
            dtype=torch.float32,
        )

        # Extract components
        self.tokenizer = self.chronos_pipeline.tokenizer
        self.encoder = self.chronos_pipeline.model.model.encoder

        # Get hidden dimension
        self._hidden_dim = self.chronos_pipeline.model.model.config.d_model
        logger.info(f"Chronos hidden dim: {self._hidden_dim}")

        # Freeze encoder if requested
        if self.freeze_encoder:
            logger.info("Freezing Chronos encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Build classification head
        self.classifier = ChronosClassificationHead(
            hidden_dim=self._hidden_dim,
            n_classes=n_classes,
            dropout=self.dropout_rate,
        )

        # Log parameter counts
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        logger.info(f"Encoder params: {encoder_params:,} (trainable: {trainable_encoder:,})")
        logger.info(f"Classifier params: {classifier_params:,}")

    def _init_window_buffer(self) -> None:
        """Initialize the sliding window buffer."""
        self._window_buffer = np.zeros(
            (self.window_size, self.projected_channels), dtype=np.float32
        )

    def _update_buffer(self, projected_sample: np.ndarray) -> np.ndarray:
        """Update sliding window buffer with new sample."""
        if self._window_buffer is None:
            self._init_window_buffer()

        self._window_buffer = np.roll(self._window_buffer, -1, axis=0)
        self._window_buffer[-1] = projected_sample
        return self._window_buffer.copy()

    def _tokenize_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of time series using Chronos tokenizer.

        Args:
            x: (batch, channels, time) - multiple channels treated as batch

        Returns:
            tokens: (batch * channels, seq_len)
            attention_mask: (batch * channels, seq_len)
        """
        batch_size, n_channels, seq_len = x.shape

        # Reshape to (batch * channels, seq_len) for tokenization
        x_flat = x.reshape(batch_size * n_channels, seq_len)

        # Tokenize each channel independently
        # Chronos tokenizer expects (batch, time)
        tokens, attention_mask, _ = self.tokenizer.context_input_transform(x_flat)

        return tokens, attention_mask

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

        # Tokenize
        tokens, attention_mask = self._tokenize_batch(x)

        # Encode with Chronos encoder
        encoder_output = self.encoder(
            input_ids=tokens,
            attention_mask=attention_mask,
        )

        # Get hidden states: (batch * channels, seq_len, hidden_dim)
        hidden_states = encoder_output.last_hidden_state

        # Reshape to (batch, channels, seq_len, hidden_dim)
        hidden_states = hidden_states.reshape(
            batch_size, n_channels, -1, self._hidden_dim
        )

        # Pool over channels and sequence: (batch, hidden_dim)
        # First pool over sequence, then over channels
        pooled = hidden_states.mean(dim=2).mean(dim=1)

        # Classify
        return self.classifier.classifier(pooled)

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
        **kwargs,
    ) -> None:
        """
        Train the ChronosClassifier model.

        Args:
            X: Feature array of shape (n_samples, n_channels).
            y: Label array of shape (n_samples,).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Learning rate for classification head.
            encoder_lr: Learning rate for encoder (if not frozen).
            weight_decay: Weight decay for regularization.
            verbose: Whether to show training progress.
        """
        # Determine classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training ChronosClassifier with {n_classes} classes")
        logger.info(f"Fitting PCA projection: {self.input_size} -> {self.projected_channels} channels")

        # Fit PCA on training data
        self.pca = PCAProjection(n_components=self.projected_channels)
        X_projected = self.pca.fit_transform(X)
        self.pca_layer = self.pca.get_torch_projection()

        # Build network
        self._build_network(n_classes)

        # Create windowed samples
        logger.info(f"Creating windowed training data with window_size={self.window_size}")
        X_windows, y_windows = self._create_windowed_data(X_projected, y)
        logger.info(f"Training data: {X_windows.shape[0]} windows")

        # Convert to tensors: (batch, time, channels) -> (batch, channels, time)
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        X_tensor = X_tensor.permute(0, 2, 1)

        y_indices = np.array([class_to_idx[label] for label in y_windows])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Setup optimizer with different learning rates
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

        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.train()
        avg_loss = 0.0

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            # Progress bar for each epoch
            batch_iterator = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not verbose,
                leave=True,
            )

            for X_batch, y_batch in batch_iterator:
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

                # Update progress bar with current loss
                batch_iterator.set_postfix(loss=f"{total_loss/n_batches:.4f}")

            scheduler.step()
            avg_loss = total_loss / n_batches
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        self.eval()
        self._init_window_buffer()
        logger.info(f"Training complete. Final loss: {avg_loss:.4f}")

    def _create_windowed_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create windowed training samples."""
        n_samples = X.shape[0]
        windows = []
        labels = []

        for i in range(self.window_size, n_samples):
            window = X[i - self.window_size : i]
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
            window_tensor = torch.tensor(window, dtype=torch.float32)
            window_tensor = window_tensor.T.unsqueeze(0)  # (1, channels, time)

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
        logger.info(f"ChronosClassifier saved to {MODEL_PATH}")

        # Log model size
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB")

        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """Load a model from the saved checkpoint."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using ChronosClassifier.fit()"
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

        # Load saved weights
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        model.eval()

        model._init_window_buffer()
        logger.info(f"ChronosClassifier loaded from {MODEL_PATH}")

        return model

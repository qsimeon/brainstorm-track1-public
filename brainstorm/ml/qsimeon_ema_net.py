"""
EMA Network for streaming BCI classification.

Based on Quilee Simeon's EMA architecture with Gumbel-Softmax channel mixing.
Adapted for sample-by-sample inference with PCA dimensionality reduction.
"""

from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
from loguru import logger
from tqdm import tqdm

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel
from brainstorm.ml.channel_projection import PCAProjection
from brainstorm.config import get_checkpoint_dir

_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pt"


class EMALayer(nn.Module):
    """
    Exponential Moving Average layer with Gumbel-Softmax channel mixing.

    Processes sequences and maintains temporal state via EMA recurrence.
    Each EMA node learns to select input channels and maintain an exponential
    moving average, enabling adaptive feature extraction for time series data.

    The core recurrence relation is:
        h[k](t) = alpha[k] * selected_input[k](t) + (1 - alpha[k]) * h[k](t-1)

    Where:
    - h[k] = EMA state for channel k
    - alpha[k] = learnable weight between 0 and 1 (via sigmoid(theta[k]))
    - selected_input[k] = input channel chosen by Gumbel-Softmax sampling
    - Channel selection depends on both current input and previous state (h[k](t-1))
    """

    def __init__(
        self,
        input_dim: int,
        ema_nodes: int | None = None,
        readout_dim: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize EMA layer.

        Args:
            input_dim: Number of input channels
            ema_nodes: Number of EMA nodes (hidden dimension). Defaults to input_dim
            readout_dim: Output dimension after readout. If None, uses ema_nodes
            temperature: Initial Gumbel-Softmax temperature for channel sampling
        """
        super().__init__()
        self.input_dim = input_dim
        self.ema_nodes = ema_nodes if ema_nodes is not None else input_dim
        self.readout_dim = readout_dim
        self.temperature = temperature

        # Layer normalization for stability
        self.layer_norm_input = nn.LayerNorm(self.input_dim)
        self.layer_norm_h = nn.LayerNorm(self.ema_nodes)

        # Learnable EMA alphas (via sigmoid of thetas)
        # These control the balance between current input and previous state
        self.thetas = nn.Parameter(torch.zeros(self.ema_nodes))

        # Input selection logits (function of input + state)
        # Takes concatenated input and previous state
        # Outputs logits for channel selection for each EMA node
        self.input_logits = nn.Linear(
            self.input_dim + self.ema_nodes,
            self.ema_nodes * self.input_dim
        )

        # Readout layer to transform EMA states to desired output dimension
        self.readout = (
            nn.Linear(self.ema_nodes, readout_dim)
            if readout_dim is not None
            else nn.Identity()
        )

    def sample_gumbel_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution for differentiable channel selection.

        Args:
            logits: Logits of shape (..., input_dim)

        Returns:
            Sampled soft one-hot vectors approximating hard selection
        """
        dist = RelaxedOneHotCategorical(temperature=self.temperature, logits=logits)
        return dist.rsample()

    def anneal_temperature(self, decay_rate: float = 0.99, min_temp: float = 0.5) -> None:
        """
        Anneal temperature: transition from exploration to exploitation.

        As temperature decreases, sampling becomes more deterministic,
        allowing the network to commit to learned channel selections.

        Args:
            decay_rate: Multiplicative factor per step
            min_temp: Minimum temperature threshold
        """
        self.temperature = max(self.temperature * decay_rate, min_temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through EMA layer.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, seq_len, readout_dim or ema_nodes)
        """
        B, L, C_in = x.shape
        alphas = torch.sigmoid(self.thetas)  # (ema_nodes,)

        h_states = []
        h_prev = torch.zeros((B, self.ema_nodes), device=x.device)

        for t in range(L):
            # Normalize input at current timestep
            norm_x_t = self.layer_norm_input(x[:, t])  # (B, input_dim)

            # Channel selection depends on input + state (adaptive mixing)
            concat = torch.cat((norm_x_t, h_prev), dim=-1)
            logits = self.input_logits(concat).view(-1, self.ema_nodes, self.input_dim)
            input_mask = self.sample_gumbel_softmax(logits)  # (B, ema_nodes, input_dim)

            # Select input channels via mask
            selected_input = torch.bmm(input_mask, norm_x_t.unsqueeze(-1)).squeeze(-1)

            # EMA recurrence: h[k] = alpha[k] * x_selected + (1 - alpha[k]) * h_prev[k]
            h_current = self.layer_norm_h(
                alphas.unsqueeze(0) * selected_input +
                (1 - alphas.unsqueeze(0)) * h_prev
            )

            h_prev = h_current
            h_states.append(h_current)

        # Stack all timesteps: (B, L, ema_nodes)
        h = torch.stack(h_states, dim=1)
        return self.readout(h)


class QSimeonEMANet(BaseModel):
    """
    EMA Network adapted for streaming BCI classification.

    Architecture:
    1. PCA projection: 1024 channels → 64 channels (dimensionality reduction)
    2. EMA layer: Learns channel mixing and temporal patterns
    3. Readout: Output class logits

    For streaming inference, maintains a sliding window buffer of projected samples
    and processes them through the EMA layer for prediction.

    Attributes:
        classes_: Unique class labels learned during training
        pca: Fitted PCA projection object
        pca_layer: PyTorch linear layer implementing PCA transform
        ema_layer: Trained EMA layer

    Example:
        >>> model = QSimeonEMANet()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Load and predict
        >>> model = QSimeonEMANet.load()
        >>> pred = model.predict(single_sample)  # Shape: (1024,)
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        projected_channels: int = 64,
        ema_nodes: int = 64,
        window_size: int = 1600,
        temperature: float = 1.0,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize QSimeonEMANet model.

        Args:
            input_size: Number of input channels (1024 for ECoG)
            projected_channels: Channels after PCA projection
            ema_nodes: Number of EMA nodes in hidden layer
            window_size: Size of sliding window for temporal context (samples)
            temperature: Initial Gumbel-Softmax temperature
            dropout: Dropout rate (reserved for future use)
        """
        super().__init__()

        self.input_size = input_size
        self.projected_channels = projected_channels
        self.ema_nodes = ema_nodes
        self.window_size = window_size
        self.temperature = temperature
        self.dropout_rate = dropout

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None

        # PCA projection (fitted during training)
        self.pca: PCAProjection | None = None
        self.pca_layer: nn.Linear | None = None

        # EMA layer (built after knowing n_classes)
        self.ema_layer: EMALayer | None = None

        # State for streaming inference
        self._window_buffer: np.ndarray | None = None

    def _build_network(self, n_classes: int) -> None:
        """Build EMA network once n_classes is known."""
        self._n_classes = n_classes
        self.ema_layer = EMALayer(
            input_dim=self.projected_channels,
            ema_nodes=self.ema_nodes,
            readout_dim=n_classes,
            temperature=self.temperature,
        )

    def _init_buffers(self) -> None:
        """Initialize state and window buffer for streaming inference."""
        self._window_buffer = np.zeros(
            (self.window_size, self.projected_channels),
            dtype=np.float32
        )

    def _create_windowed_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping windows from sequential data for training.

        Each window covers `window_size` consecutive samples.
        Label is assigned as the last sample's label in each window.

        Args:
            X: Projected features of shape (n_samples, projected_channels)
            y: Labels of shape (n_samples,)

        Returns:
            Tuple of (windowed_X, windowed_y):
            - windowed_X: shape (n_windows, window_size, projected_channels)
            - windowed_y: shape (n_windows,) - label of last sample in window
        """
        windows = []
        labels = []

        for i in range(self.window_size, len(X)):
            window = X[i - self.window_size:i]
            windows.append(window)
            labels.append(y[i - 1])

        return np.array(windows), np.array(labels)

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs
    ) -> None:
        """
        Train EMA network on sequential data.

        Training process:
        1. Fit PCA projection on training data
        2. Create overlapping windows for temporal learning
        3. Train EMA layer with temperature annealing
        4. Validate on held-out data if provided

        Args:
            X: Feature array of shape (n_samples, n_channels)
            y: Label array of shape (n_samples,)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate for AdamW optimizer
            X_val: Optional validation features
            y_val: Optional validation labels
            **kwargs: Additional arguments (unused)
        """
        # Determine classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training EMANet with {n_classes} classes: {self.classes_.tolist()}")
        logger.info(f"Fitting PCA: {self.input_size} → {self.projected_channels} channels")

        # Fit PCA on training data
        self.pca = PCAProjection(n_components=self.projected_channels)
        X_proj = self.pca.fit_transform(X)
        self.pca_layer = self.pca.get_torch_projection()

        # Build network
        self._build_network(n_classes)

        # Create windowed data for temporal learning
        logger.info(f"Creating windowed data (window_size={self.window_size})")
        X_windows, y_windows = self._create_windowed_data(X_proj, y)
        logger.info(f"Created {len(X_windows)} training windows")

        # Convert to tensors
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        y_indices = np.array([class_to_idx[label] for label in y_windows])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Setup device - prioritize CUDA (cluster GPU) > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Training on CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     logger.info("Training on Apple MPS device (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            logger.info("Training on CPU (GPU not available)")

        self.to(device)

        # Training setup
        optimizer = torch.optim.AdamW(
            self.ema_layer.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        # Best model tracking
        best_val_acc = 0.0
        checkpoint_path = get_checkpoint_dir() / "ema_net_best.pt"

        # Training loop
        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()

                # Forward: EMA processes entire sequence
                # Output: (batch, window_size, n_classes)
                outputs = self.ema_layer(X_batch)

                # Use last timestep for classification
                logits = outputs[:, -1, :]  # (batch, n_classes)

                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            # Anneal temperature
            self.ema_layer.anneal_temperature()

            avg_loss = total_loss / len(loader)
            logger.info(
                f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                f"temp={self.ema_layer.temperature:.3f}"
            )

            # Validation if provided
            if X_val is not None and y_val is not None:
                val_acc = self._evaluate(X_val, y_val, class_to_idx, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(checkpoint_path)
                    logger.info(f"  Val acc: {val_acc:.3f} [BEST]")
                else:
                    logger.info(f"  Val acc: {val_acc:.3f}")

        # Move to CPU and initialize buffers for inference
        self.to("cpu")
        self.eval()
        self._init_buffers()
        logger.info("Training complete")

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_to_idx: dict,
        device: torch.device
    ) -> float:
        """Evaluate model on validation set and return accuracy."""
        from sklearn.metrics import balanced_accuracy_score

        # Project validation data
        X_proj = self.pca.transform(X)
        X_windows, y_windows = self._create_windowed_data(X_proj, y)

        X_tensor = torch.tensor(X_windows, dtype=torch.float32).to(device)
        y_indices = np.array([class_to_idx[label] for label in y_windows])
        y_tensor = torch.tensor(y_indices, dtype=torch.long).to(device)

        self.eval()
        with torch.no_grad():
            outputs = self.ema_layer(X_tensor)
            logits = outputs[:, -1, :]
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        val_acc = balanced_accuracy_score(y_indices, preds)
        return float(val_acc)

    def _save_checkpoint(self, path: Path) -> None:
        """Save best model checkpoint."""
        checkpoint = {
            "config": {
                "input_size": self.input_size,
                "projected_channels": self.projected_channels,
                "ema_nodes": self.ema_nodes,
                "window_size": self.window_size,
                "n_classes": self._n_classes,
                "temperature": self.temperature,
                "dropout": self.dropout_rate,
            },
            "classes": self.classes_,
            "pca_mean": self.pca.mean_,
            "pca_components": self.pca.components_,
            "ema_state_dict": self.ema_layer.state_dict(),
        }
        torch.save(checkpoint, path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, 1, projected_channels, window_size)

        Returns:
            Logits of shape (batch, n_classes)
        """
        if self.ema_layer is None:
            raise RuntimeError("Network not built. Call fit() or load a trained model.")
        return self.ema_layer(x)

    def predict(self, X: np.ndarray) -> int:
        """
        Streaming single-sample prediction with state management.

        Maintains a sliding window buffer across predict() calls.
        As new samples arrive, the buffer is updated and fed through
        the EMA layer for prediction.

        Args:
            X: Feature array of shape (n_channels,) for a single sample

        Returns:
            Predicted label as an integer
        """
        if self.classes_ is None or self.pca_layer is None or self.ema_layer is None:
            raise RuntimeError("Model not trained. Call fit() or load a trained model.")

        self.eval()
        with torch.no_grad():
            # Project input channels
            x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            x_proj = self.pca_layer(x_tensor).squeeze(0).numpy()

            # Update sliding window buffer
            self._window_buffer = np.roll(self._window_buffer, -1, axis=0)
            self._window_buffer[-1] = x_proj

            # Process window through EMA layer
            window = torch.tensor(self._window_buffer, dtype=torch.float32).unsqueeze(0)
            outputs = self.ema_layer(window)  # (1, window_size, n_classes)
            logits = outputs[0, -1, :]  # Last timestep

            predicted_idx = int(torch.argmax(logits).item())

        return int(self.classes_[predicted_idx])

    def save(self) -> Path:
        """
        Save model checkpoint.

        Returns:
            Path to the saved model file within the repository
        """
        if self.classes_ is None or self.pca is None or self.ema_layer is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        checkpoint = {
            "config": {
                "input_size": self.input_size,
                "projected_channels": self.projected_channels,
                "ema_nodes": self.ema_nodes,
                "window_size": self.window_size,
                "n_classes": self._n_classes,
                "temperature": self.temperature,
                "dropout": self.dropout_rate,
            },
            "classes": self.classes_,
            "pca_mean": self.pca.mean_,
            "pca_components": self.pca.components_,
            "ema_state_dict": self.ema_layer.state_dict(),
        }
        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"QSimeonEMANet model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """
        Load model from saved checkpoint.

        Returns:
            A new instance of QSimeonEMANet with loaded weights

        Raises:
            FileNotFoundError: If model file does not exist
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using QSimeonEMANet.fit()"
            )

        checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
        config = checkpoint["config"]

        model = cls(
            input_size=config["input_size"],
            projected_channels=config["projected_channels"],
            ema_nodes=config["ema_nodes"],
            window_size=config["window_size"],
            temperature=config["temperature"],
            dropout=config["dropout"],
        )

        # Restore PCA
        model.pca = PCAProjection(n_components=config["projected_channels"])
        model.pca.mean_ = checkpoint["pca_mean"]
        model.pca.components_ = checkpoint["pca_components"]
        model.pca_layer = model.pca.get_torch_projection()

        # Restore network
        model.classes_ = checkpoint["classes"]
        model._build_network(config["n_classes"])
        model.ema_layer.load_state_dict(checkpoint["ema_state_dict"])
        model.eval()

        # Initialize inference buffers
        model._init_buffers()

        logger.debug(f"QSimeonEMANet model loaded from {MODEL_PATH}")
        return model

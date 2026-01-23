"""
Stateful LSTM Model for streaming BCI classification.

This module provides a stateful LSTM model that maintains hidden state across
sequential sample-by-sample predictions. The model is designed to capture temporal
dependencies in ECoG signals during continuous online inference.

Key Features:
    - Stateful: Hidden and cell states persist across predict() calls
    - Sequential Training: Trained one sample at a time to match evaluation pattern
    - Temporal Awareness: Leverages temporal structure in ECoG signals
    - Memory Efficient: Detaches gradients after each sample to prevent BPTT explosion
"""

from pathlib import Path

import numpy as np

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
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


class StatefulLSTM(BaseModel):
    """
    Stateful LSTM for continuous classification of ECoG signals.

    This model maintains hidden and cell states across sequential predictions,
    allowing it to leverage temporal patterns in the input signal. During training,
    samples are processed sequentially (one at a time) to match the evaluation pattern
    used by ModelEvaluator.

    Architecture:
        Input (1024) -> LSTM (hidden_size=256) -> Dropout -> Dense -> Output (n_classes)

    The key innovation is state persistence: when predict() is called in a loop by
    ModelEvaluator, the hidden state from the previous prediction influences the
    current one, enabling temporal awareness.

    Attributes:
        input_size: Number of input features (ECoG channels). Default: 1024.
        hidden_size: Number of LSTM hidden units. Default: 256.
        num_layers: Number of LSTM layers. Default: 1.
        dropout: Dropout rate for regularization. Default: 0.3.
        classes_: Array of unique class labels learned during fit().
        hidden_state: Current LSTM hidden state (maintained across predict calls).
        cell_state: Current LSTM cell state (maintained across predict calls).

    Example:
        >>> model = StatefulLSTM(input_size=1024, hidden_size=256)
        >>> model.fit(train_features, train_labels, epochs=10)
        >>>
        >>> # Load for stateful inference
        >>> model = StatefulLSTM.load()
        >>> for sample in test_stream:
        ...     prediction = model.predict(sample)  # State persists!
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the StatefulLSTM model.

        Args:
            input_size: Number of input features (ECoG channels). Default: 1024.
            hidden_size: Number of hidden units in LSTM layer. Default: 256.
            num_layers: Number of LSTM layers. Default: 1.
            dropout: Dropout rate for regularization. Default: 0.3.

        Note:
            The number of output classes is determined automatically during fit()
            based on the unique labels in the training data.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.classes_: np.ndarray | None = None

        # State for stateful inference (maintained across predict calls)
        self.hidden_state: torch.Tensor | None = None
        self.cell_state: torch.Tensor | None = None

        # Layers will be initialized in _build_layers() after we know n_classes
        self.lstm: nn.LSTM | None = None
        self.dropout: nn.Dropout | None = None
        self.fc1: nn.Linear | None = None
        self.fc2: nn.Linear | None = None

    def _build_layers(self, n_classes: int) -> None:
        """
        Build the network layers once n_classes is known.

        Args:
            n_classes: Number of output classes.
        """
        self._n_classes = n_classes

        # LSTM: processes single timestep at a time with maintained state
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=False,  # We handle batching explicitly
            dropout=0.0 if self.num_layers == 1 else self.dropout_rate,
        )

        # Classification head: LSTM output -> dense layers -> class logits
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def reset_state(self) -> None:
        """
        Reset the hidden and cell states to None.

        Call this at the beginning of each epoch during training, or when starting
        a new inference session. States are automatically initialized on the first
        forward pass after reset.
        """
        self.hidden_state = None
        self.cell_state = None

    def _detach_hidden_state(self) -> None:
        """
        Detach hidden and cell states from the computation graph.

        This is called after each training sample to prevent gradients from
        flowing back through the entire sequence (Truncated BPTT). It ensures
        memory usage stays bounded and makes training tractable.
        """
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        if self.cell_state is not None:
            self.cell_state = self.cell_state.detach()

    def _initialize_state_if_needed(self, batch_size: int = 1) -> None:
        """
        Initialize hidden and cell states if they haven't been initialized yet.

        Args:
            batch_size: Batch size for state initialization. Default: 1 (for inference).
        """
        if self.hidden_state is None or self.cell_state is None:
            device = next(self.parameters()).device
            self.hidden_state = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
            self.cell_state = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM and classification head.

        This method updates self.hidden_state and self.cell_state internally,
        maintaining them for the next forward pass. This is what enables
        stateful inference across sequential predict() calls.

        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size) for LSTM
               or (batch_size, input_size) which is reshaped to (1, batch_size, input_size).

        Returns:
            Logits tensor of shape (batch_size, n_classes).
        """
        if (
            self.lstm is None
            or self.fc1 is None
            or self.fc2 is None
            or self.dropout is None
        ):
            raise RuntimeError(
                "Model layers not initialized. Call fit() first or load a trained model."
            )

        # Handle 2D input (batch_size, input_size) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (batch_size, input_size) -> (1, batch_size, input_size)

        batch_size = x.shape[1]

        # Initialize state on first forward pass
        self._initialize_state_if_needed(batch_size)

        # LSTM forward pass with state update
        # Input: (seq_len=1, batch_size, input_size)
        # Output: (seq_len=1, batch_size, hidden_size)
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state)
        )

        # Take the output of the last timestep
        # lstm_out: (seq_len, batch_size, hidden_size) -> (batch_size, hidden_size)
        lstm_out = lstm_out.squeeze(0)

        # Classification head
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        verbose: bool = True,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """
        Train the model on the provided features and labels.

        This is called by the base class fit() method, which handles saving
        and validation.

        **Key Training Strategy**: Samples are processed sequentially (one at a time)
        to match the evaluation pattern. This forces the model to learn to use
        temporal context effectively. State is reset at epoch boundaries (simulating
        new recording sessions) but maintained within each epoch.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Label array of shape (n_samples,) with integer class labels.
            epochs: Number of training epochs. Default: 10.
            learning_rate: Learning rate for Adam optimizer. Default: 1e-3.
            verbose: Whether to show training progress. Default: True.
            X_val: Optional validation features for computing validation metrics.
            y_val: Optional validation labels for computing validation metrics.
        """
        from sklearn.metrics import balanced_accuracy_score

        # Determine unique classes and create mapping
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training StatefulLSTM with {n_classes} classes: {self.classes_.tolist()}")
        logger.info(
            f"Training strategy: Sequential sample-by-sample (matches evaluation pattern)"
        )

        # Build layers now that we know n_classes
        self._build_layers(n_classes)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_indices = np.array([class_to_idx[label] for label in y])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Setup training
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        epoch_iterator = tqdm(range(epochs), desc="Training epochs", disable=not verbose)
        for epoch in epoch_iterator:
            # Reset state at epoch boundary (new recording session)
            self.reset_state()

            total_loss = 0.0
            n_samples = 0
            train_predictions = []
            train_targets = []

            # Sequential processing: one sample at a time
            sample_iterator = tqdm(
                range(len(X)),
                desc=f"  Epoch {epoch + 1}/{epochs}",
                disable=not verbose,
                leave=False,
            )
            for t in sample_iterator:
                # Get single sample
                x_t = X_tensor[t].unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
                y_t = y_tensor[t:t + 1]  # (1,)

                # Forward pass
                optimizer.zero_grad()
                logits = self.forward(x_t)
                loss = criterion(logits, y_t)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Detach state to prevent BPTT through entire sequence
                self._detach_hidden_state()

                total_loss += loss.item()
                n_samples += 1

                # Track predictions for accuracy
                pred_idx = int(torch.argmax(logits, dim=1).item())
                train_predictions.append(self.classes_[pred_idx])
                train_targets.append(y[t])

            # Compute training metrics
            train_loss = total_loss / n_samples
            train_accuracy = balanced_accuracy_score(train_targets, train_predictions)

            # Compute validation metrics if provided
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                self.eval()
                self.reset_state()
                val_total_loss = 0.0
                val_predictions = []
                val_targets = []

                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                    y_val_indices = np.array([class_to_idx[label] for label in y_val])
                    y_val_tensor = torch.tensor(y_val_indices, dtype=torch.long)

                    for t in range(len(X_val)):
                        x_t = X_val_tensor[t].unsqueeze(0).unsqueeze(0)
                        y_t = y_val_tensor[t : t + 1]

                        logits = self.forward(x_t)
                        loss = criterion(logits, y_t)
                        val_total_loss += loss.item()

                        pred_idx = int(torch.argmax(logits, dim=1).item())
                        val_predictions.append(self.classes_[pred_idx])
                        val_targets.append(y_val[t])
                        self._detach_hidden_state()

                val_loss = val_total_loss / len(X_val)
                val_accuracy = balanced_accuracy_score(val_targets, val_predictions)

                self.train()

            # Update progress bar with metrics
            postfix = f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.3f}"
            if val_loss is not None:
                postfix += f", val_loss={val_loss:.4f}, val_acc={val_accuracy:.3f}"
            epoch_iterator.set_postfix_str(postfix)

        self.eval()
        logger.info(f"Training complete. Final train loss: {train_loss:.4f}, accuracy: {train_accuracy:.3f}")

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the label for a single sample with stateful inference.

        **Key Feature**: The hidden and cell states are maintained across calls
        to this method. When predict() is called in a loop by ModelEvaluator,
        the state from the previous sample influences the current prediction,
        enabling temporal awareness.

        Args:
            X: Feature array of shape (n_features,) for a single timestep.
               For ECoG data: shape (1024,).

        Returns:
            Predicted label as an integer (original class value, not index).

        Raises:
            RuntimeError: If model is not trained or loaded.
        """
        if self.classes_ is None:
            raise RuntimeError(
                "Model not trained. Call fit() first or load a trained model."
            )

        self.eval()
        with torch.no_grad():
            # Convert to tensor: (n_features,) -> (1, 1, n_features)
            x_tensor = torch.tensor(X, dtype=torch.float32)
            x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)

            # Forward pass: updates self.hidden_state and self.cell_state
            logits = self.forward(x_tensor)

            # Get prediction
            predicted_idx = int(torch.argmax(logits, dim=1).item())

        return int(self.classes_[predicted_idx])

    def save(self) -> Path:
        """
        Save the model weights and configuration to model.pt.

        **Note**: We do NOT save the hidden and cell states. When the model
        is loaded for evaluation, states are reset to None and will be
        initialized on the first predict() call. This gives clean starting
        conditions for each evaluation session.

        Returns:
            Path to the saved model file.
        """
        if self.classes_ is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        checkpoint = {
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "n_classes": self._n_classes,
                "dropout": self.dropout_rate,
            },
            "classes": self.classes_,
            "state_dict": self.state_dict(),
        }

        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"Model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """
        Load a model from model.pt.

        Returns:
            A new instance of StatefulLSTM with loaded weights.

        Raises:
            FileNotFoundError: If model.pt does not exist.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using StatefulLSTM.fit() which saves to this location."
            )

        checkpoint = torch.load(MODEL_PATH, weights_only=False)

        # Reconstruct the model
        config = checkpoint["config"]
        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )

        # Rebuild layers with the saved n_classes
        model._build_layers(config["n_classes"])

        model.classes_ = checkpoint["classes"]
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model

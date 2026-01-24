"""
Base model interface for continuous classification.

This module defines the abstract base class that all models must implement.
Continuous classification predicts a discrete label (e.g., stimulus frequency in Hz)
at every timestep.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
from loguru import logger
from torch import nn

from brainstorm.constants import REPO_ROOT
from brainstorm.ml.utils import validate_model_loadable


# Fixed metadata path within the repository
METADATA_PATH = REPO_ROOT / "model_metadata.json"


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for continuous classification models.

    Subclasses must implement:
        - fit_model(): Training logic (called by fit())
        - predict(): Make a prediction for a single sample
        - save(): Save model weights and config, return the file path
        - load(): Class method to load model from saved file

    The base class provides a fit() method that:
        1. Calls your fit_model() implementation
        2. Calls save() and validates the saved model
        3. Saves metadata (model path and import string) for evaluation

    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model on the provided features and labels.

        This method:
        1. Calls fit_model() for your training logic
        2. Saves the model via save()
        3. Validates the model file is within the repository
        4. Validates the saved model can be loaded
        5. Saves metadata for programmatic loading during evaluation

        Args:
            X: Feature array of shape (n_timesteps, n_channels).
            y: Label array of shape (n_samples,).
               Integer labels representing stimulus frequency (0 = no stimulus).
            **kwargs: Additional arguments passed to fit_model().

        Returns:
            None. The trained model and metadata are saved.

        Raises:
            RuntimeError: If the model file is saved outside the repository
                         or if validation fails.
        """
        logger.info(f"Training {self.__class__.__name__}...")

        # Call subclass training logic
        self.fit_model(X, y, **kwargs)

        # Save the model
        model_path = self.save()
        assert isinstance(model_path, Path), (
            f"model_path returned by `self.save()` must be a Path, got {type(model_path)}: {model_path}"
        )

        # Validate model is within repository (required for remote evaluation)
        model_path_resolved = model_path.resolve()
        repo_root_resolved = REPO_ROOT.resolve()
        try:
            model_path_resolved.relative_to(repo_root_resolved)
        except ValueError:
            raise RuntimeError(
                f"Model file MUST be saved within the repository for remote evaluation!"
            )

        # Generate import string for this class
        import_string = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # Validate that the model can be loaded programmatically
        logger.info("Validating model is loadable...")
        try:
            # Validate with 25MB size limit for evaluation system
            validate_model_loadable(import_string, model_path, max_size_mb=25.0)
        except Exception as e:
            raise RuntimeError(
                f"Model validation failed! The saved model cannot be loaded. Error: {e}"
            ) from e
        logger.info(f"✓ Model saved to: {model_path}")

        # Save metadata for evaluation
        # Store path relative to repo root for portability
        relative_model_path = model_path_resolved.relative_to(repo_root_resolved)
        metadata = {
            "model_path": str(relative_model_path),
            "import_string": import_string,
        }

        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("✓ Model validation successful")
        logger.info(f"✓ Metadata saved to: {METADATA_PATH}")

    @abstractmethod
    def fit_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model on the provided features and labels.

        Implement your training logic here. This method is called by fit(),
        which handles saving and validation.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            **kwargs: Additional training parameters.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> int:
        """
        Predict the label for a single sample.

        Args:
            X: Feature array of shape (n_features,) for a single timestep.
               For ECoG data: (n_channels,) = (1024,).

        Returns:
            Predicted label as an integer (stimulus frequency in Hz, or 0 for no stimulus).
        """
        pass

    @abstractmethod
    def save(self) -> Path:
        """
        Save the model weights and configuration.

        The saved file should contain everything needed to reconstruct
        the model, including:
            - Model architecture parameters (input_size, hidden_size, etc.)
            - Trained weights (state_dict or sklearn model)
            - Class mapping if applicable

        **IMPORTANT:** The model file MUST be saved within the repository
        directory for remote evaluation. Files outside the repository cannot
        be uploaded for evaluation.

        Returns:
            Path to the saved model file (absolute path within the repository).

        Example:
            >>> def save(self) -> Path:
            ...     model_path = Path(__file__).parent.parent.parent / "model.pt"
            ...     torch.save({"state": ...}, model_path)
            ...     return model_path.resolve()
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls) -> Self:
        """
        Load a model from a saved file.

        Returns:
            A new instance of the model with loaded weights.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the file is corrupted or incompatible.
        """
        pass

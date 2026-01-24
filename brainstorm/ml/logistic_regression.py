"""
Logistic Regression Model for continuous classification of ECoG signals.

This module provides a sklearn-based Logistic Regression model that predicts
discrete frequency labels from neural recordings at each timestep.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Self

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel


# Fixed model path within the repository
_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pkl"


class LogisticRegression(BaseModel):
    """
    Logistic Regression for continuous classification with optimizations.

    A sklearn LogisticRegression model that maps ECoG channel readings
    to stimulus frequency predictions. Optimized for large datasets with:
    - Automatic feature standardization for faster convergence
    - Optional PCA dimensionality reduction

    The model is always saved to and loaded from a fixed location: `model.pkl`
    in the repository root. This ensures consistent model management.

    Attributes:
        input_size: Number of input features (default: 1024 channels).
        max_iter: Maximum iterations for solver convergence. Default: 1000.
        use_pca: Apply PCA dimensionality reduction. Default: False.
        n_components: PCA components (None = 95% variance). Default: None.
        classes_: Array of unique class labels learned during fit().
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        max_iter: int = 1000,
        use_pca: bool = False,
        n_components: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Logistic Regression model.

        Args:
            input_size: Number of input features (ECoG channels). Default: 1024.
            max_iter: Maximum iterations for solver. Default: 1000.
            C: Inverse regularization strength (smaller = stronger). Default: 1.0.
            solver: Solver algorithm. Default: 'saga' (fast for large datasets).
                   Options: 'saga', 'sag', 'lbfgs', 'liblinear'.
            n_jobs: Number of CPU cores to use (-1 = all cores). Default: -1.
            tol: Tolerance for stopping criteria. Higher = faster. Default: 1e-3.
            use_pca: Whether to apply PCA dimensionality reduction. Default: False.
            n_components: Number of PCA components (if use_pca=True).
                         If None, keeps 95% variance. Default: None.
            **kwargs: Additional sklearn LogisticRegression parameters.

        Note:
            The number of output classes is determined automatically during fit()
            based on the unique labels in the training data.
        """
        super().__init__()
        self.input_size = input_size
        self.max_iter = max_iter
        self.use_pca = use_pca
        self.n_components = n_components
        self.kwargs = kwargs
        self.classes_: np.ndarray | None = None
        self._model: SklearnLogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Train the model on the provided features and labels.

        This is called by the base class fit() method, which handles
        saving and validation.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Label array of shape (n_samples,) with integer class labels.
            verbose: Whether to show training progress. Default: True.
            **kwargs: Ignored (for compatibility with other models).
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        logger.info(
            f"Fitting LogisticRegression with {n_classes} classes: {self.classes_.tolist()}"
        )
        logger.info(f"Original feature shape: {X.shape}")

        # Step 1: Standardize features (improves convergence)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        logger.info("Features standardized (zero mean, unit variance)")

        # Step 2: Optional PCA dimensionality reduction
        if self.use_pca:
            self._pca = PCA(
                n_components=self.n_components or 0.95,  # Keep 95% variance by default
                random_state=42,
            )
            X_scaled = self._pca.fit_transform(X_scaled)
            variance_ratio = self._pca.explained_variance_ratio_.sum()
            logger.info(
                f"PCA applied: {X_scaled.shape[1]} components "
                f"(explaining {variance_ratio:.1%} variance)"
            )

        # Step 3: Create and train sklearn model with optimized parameters
        self._model = SklearnLogisticRegression(
            max_iter=self.max_iter,
            verbose=1 if verbose else 0,
            **self.kwargs,
        )

        self._model.fit(X_scaled, y)
        logger.info("Training complete.")

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the label for a single sample.

        Args:
            X: Feature array of shape (n_features,) for a single timestep.

        Returns:
            Predicted label as an integer (original class value).

        Raises:
            RuntimeError: If model is not trained or loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not trained. Call fit() first or load a trained model."
            )

        # Ensure X is 2D for sklearn (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Apply same preprocessing as during training
        if self._scaler is not None:
            X = self._scaler.transform(X)  # type: ignore[assignment]
        if self._pca is not None:
            X = self._pca.transform(X)  # type: ignore[assignment]

        prediction = self._model.predict(X)[0]
        return int(prediction)

    def save(self) -> Path:
        """
        Save the model to model.pkl using pickle.

        Returns:
            Path to the saved model file.
        """
        if self._model is None or self.classes_ is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        checkpoint = {
            "config": {
                "input_size": self.input_size,
                "max_iter": self.max_iter,
                "use_pca": self.use_pca,
                "n_components": self.n_components,
                "kwargs": self.kwargs,
            },
            "classes": self.classes_,
            "model": self._model,
            "scaler": self._scaler,
            "pca": self._pca,
        }

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(checkpoint, f)

        logger.debug(f"Model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """
        Load a model from model.pkl.

        Returns:
            A new instance of LogisticRegression with loaded weights.

        Raises:
            FileNotFoundError: If model.pkl does not exist.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using LogisticRegression.fit() which saves to this location."
            )

        with open(MODEL_PATH, "rb") as f:
            checkpoint = pickle.load(f)

        # Reconstruct the model
        config = checkpoint["config"]
        model = cls(
            input_size=config["input_size"],
            max_iter=config["max_iter"],
            use_pca=config.get("use_pca", False),
            n_components=config.get("n_components"),
            **config.get("kwargs", {}),
        )

        model.classes_ = checkpoint["classes"]  # type: ignore[assignment]
        model._model = checkpoint["model"]
        model._scaler = checkpoint.get("scaler")
        model._pca = checkpoint.get("pca")

        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model

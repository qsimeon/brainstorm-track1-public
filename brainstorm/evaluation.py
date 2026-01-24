"""
Model evaluation for continuous classification.

This module provides the ModelEvaluator class that:
    1. Loads a trained model from metadata
    2. Runs inference on test data
    3. Computes evaluation metrics (balanced accuracy + model size)
    4. Displays a simple summary with the final score
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from brainstorm.ml.base import METADATA_PATH
from brainstorm.ml.metrics import compute_score, MetricsResults
from brainstorm.ml.utils import import_model_class, validate_model_file


@dataclass
class ModelEvaluator:
    """
    Evaluator for continuous classification models.

    The evaluator loads a trained model using the metadata saved during
    training and runs inference on provided features. It computes balanced
    accuracy and considers model size for the final score.

    Attributes:
        test_features: DataFrame of test features with time index.
        test_labels: DataFrame of test labels with time index.

    Example:
        >>> # After training a model with model.fit()
        >>> evaluator = ModelEvaluator(
        ...     test_features=test_features,
        ...     test_labels=test_labels,
        ... )
        >>> results = evaluator.evaluate()
        >>> evaluator.print_summary(results)
    """

    test_features: pd.DataFrame
    test_labels: pd.DataFrame

    # Internal state
    _model: object = field(default=None, init=False, repr=False)
    _predictions: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _runtime_seconds: float = field(default=0.0, init=False, repr=False)
    _model_path: Path | None = field(default=None, init=False, repr=False)

    def _load_model(self) -> tuple[object, Path]:
        """
        Load the model using metadata.

        Returns:
            Tuple of (model instance, model path).

        Raises:
            FileNotFoundError: If metadata file doesn't exist.
            ImportError: If the model class can't be imported.
        """
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {METADATA_PATH}\n"
                "Train a model first using model.fit()"
            )

        # Load metadata
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

        model_path = Path(metadata["model_path"])
        import_string = metadata["import_string"]

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using class: {import_string}")

        # Validate model file exists
        file_size_mb = validate_model_file(model_path)

        # Import and load the model
        model_class = import_model_class(import_string)
        model = model_class.load()

        logger.info(
            f"✓ Model loaded successfully from {model_path} | {file_size_mb:.2f}MB"
        )
        return model, model_path

    def _get_model_size_bytes(self) -> int:
        """Get the size of the model file in bytes."""
        if self._model_path and self._model_path.exists():
            return self._model_path.stat().st_size
        return 0

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"

    def run(self) -> pd.DataFrame:
        """
        Run inference on the test features.

        Loads the model (if not already loaded) and generates predictions
        for all samples in test_features.

        Returns:
            DataFrame with predictions, indexed by time.
        """
        # Load model if needed
        if self._model is None:
            self._model, self._model_path = self._load_model()

        logger.info(f"Running inference with {self._model.__class__.__name__}")
        logger.debug(f"Features shape: {self.test_features.shape}")

        start_time = time.perf_counter()

        # Run prediction for each sample
        predictions = []
        for sample in tqdm(self.test_features.values, desc="Predicting", leave=False):
            prediction = self._model.predict(sample)  # type: ignore[attr-defined]
            predictions.append(prediction)

        self._runtime_seconds = time.perf_counter() - start_time

        # Create predictions DataFrame
        self._predictions = pd.DataFrame(
            {"prediction": predictions},
            index=self.test_features.index,
        )

        logger.info(f"Inference completed in {self._runtime_seconds:.3f}s")
        return self._predictions

    def evaluate(self) -> MetricsResults:
        """
        Run inference and compute evaluation metrics.

        Returns:
            MetricsResults containing total_score, accuracy_score, lag_score, size_score, accuracy, avg_lag_samples

        Raises:
            ValueError: If test_labels is not provided.
        """
        if self.test_labels is None:
            raise ValueError("test_labels is required for evaluation.")

        # Run inference if not already done
        if self._predictions is None:
            self.run()

        logger.info("Computing evaluation metrics...")

        # Get arrays for metrics computation
        y_true = self.test_labels.values.ravel()
        y_pred = np.array(self._predictions["prediction"].values)  # type: ignore[union-attr]

        # Compute metrics
        model_size_bytes = self._get_model_size_bytes()
        metrics = compute_score(
            y_true=y_true,
            y_pred=y_pred,
            model_size_bytes=model_size_bytes,
        )
        return metrics

    def print_summary(
        self,
        metrics: MetricsResults,
    ) -> None:
        console = Console()

        # Create metrics table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold", width=25)
        table.add_column("Value", justify="right", width=20)
        table.add_column("Score", justify="right", width=20)

        # Final Score (prominently displayed first)
        table.add_row(
            "[bold green]FINAL SCORE[/]",
            f"[bold green]{metrics.total_score:.1f}/100[/]",
            f"{metrics.total_score:.1f}",
        )
        table.add_row("", "", "")

        # Balanced Accuracy (50% of score)
        table.add_row(
            "Balanced Accuracy",
            f"{metrics.accuracy * 100:.1f}%",
            f"{metrics.accuracy_score:.1f}",
        )

        # Lag
        table.add_row(
            "Lag",
            f"{metrics.avg_lag_samples:.1f} ms (max 500ms)",
            f"{metrics.lag_score:.1f}",
        )

        # Model Size (50% of score)
        table.add_row(
            "Model Size",
            self._format_size(metrics.model_size_bytes),
            f"{metrics.size_score:.1f}",
        )

        table.add_row("", "", "")

        # Runtime info
        table.add_row(
            "Inference Time",
            f"{self._runtime_seconds:.3f} s",
            "---",
        )
        console.print(table)

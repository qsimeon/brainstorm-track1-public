#!/usr/bin/env python3
"""
Comprehensive overnight hyperparameter search for EEGNet.

This script systematically explores the hyperparameter space:
- Window sizes: [64, 128, 256, 384, 512, 768, 1024, 1280, 1600, 2000]
- Projected channels: [16, 32, 48, 64, 96, 128]
- F1 (temporal filters): [4, 8, 16, 32]
- D (depthwise multiplier): [1, 2, 4]
- Dropout: [0.1, 0.25, 0.4, 0.5]
- Learning rates: [5e-4, 1e-3, 2e-3]

Results are saved incrementally to JSON and CSV files for analysis.

Usage:
    python examples/overnight_hypersearch.py
    python examples/overnight_hypersearch.py --quick  # Quick test run with fewer configs
    python examples/overnight_hypersearch.py --resume  # Resume from last checkpoint
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.ml.eegnet import EEGNet
from brainstorm.ml.metrics import compute_score, MetricsResults


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")
RESULTS_DIR = Path("./hypersearch_results")
RESULTS_JSON = RESULTS_DIR / "results.json"
RESULTS_CSV = RESULTS_DIR / "results.csv"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
BEST_MODEL_DIR = RESULTS_DIR / "best_models"

# Full exhaustive grid (use --exhaustive flag) - 8640 configs, ~12 days
EXHAUSTIVE_GRID = {
    "window_size": [64, 128, 256, 384, 512, 768, 1024, 1280, 1600, 2000],
    "projected_channels": [16, 32, 48, 64, 96, 128],
    "F1": [4, 8, 16, 32],
    "D": [1, 2, 4],
    "dropout": [0.1, 0.25, 0.4, 0.5],
    "learning_rate": [5e-4, 1e-3, 2e-3],
}

# Default overnight grid - ~8-12 hours with focus on key parameters
# Window size is the most important, so we test many values
# Other params have narrower ranges based on prior knowledge
FULL_GRID = {
    "window_size": [64, 128, 256, 384, 512, 768, 1024, 1280, 1600, 2000],
    "projected_channels": [32, 64, 96],
    "F1": [8, 16],
    "D": [2],
    "dropout": [0.25, 0.4],
    "learning_rate": [1e-3],
}
# 10 * 3 * 2 * 1 * 2 * 1 = 120 configs @ ~2min = 4 hours

# Quick test grid (for debugging) - ~12 min
QUICK_GRID = {
    "window_size": [128, 512, 1024],
    "projected_channels": [32, 64],
    "F1": [8],
    "D": [2],
    "dropout": [0.25],
    "learning_rate": [1e-3],
}

# Training constants
EPOCHS = 30
BATCH_SIZE = 64


@dataclass
class ExperimentResult:
    """Result from a single hyperparameter configuration."""

    # Hyperparameters
    window_size: int
    projected_channels: int
    F1: int
    D: int
    dropout: float
    learning_rate: float

    # Metrics
    balanced_accuracy: float
    avg_lag_samples: float
    model_size_bytes: int

    # Scores
    accuracy_score: float
    lag_score: float
    size_score: float
    total_score: float

    # Meta
    training_time_seconds: float
    inference_time_seconds: float
    timestamp: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def setup_logging():
    """Configure logging to file and console."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RESULTS_DIR / f"hypersearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add(log_file, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    return log_file


def load_data():
    """Load training and validation data."""
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        logger.info("Downloading data from Hugging Face...")
        download_train_validation_data()

    logger.info(f"Loading data from {DATA_PATH}")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")

    logger.info(f"Train: {train_features.shape}, Val: {val_features.shape}")
    return train_features, train_labels, val_features, val_labels


def generate_configs(grid: dict) -> list[dict]:
    """Generate all hyperparameter configurations from grid."""
    keys = list(grid.keys())
    values = list(grid.values())

    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def get_completed_configs(results: list[dict]) -> set[str]:
    """Get set of completed configuration hashes."""
    completed = set()
    for r in results:
        key = f"{r['window_size']}_{r['projected_channels']}_{r['F1']}_{r['D']}_{r['dropout']}_{r['learning_rate']}"
        completed.add(key)
    return completed


def config_to_key(config: dict) -> str:
    """Convert config to unique string key."""
    return f"{config['window_size']}_{config['projected_channels']}_{config['F1']}_{config['D']}_{config['dropout']}_{config['learning_rate']}"


def run_single_experiment(
    config: dict,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    n_channels: int,
) -> ExperimentResult:
    """Run a single hyperparameter configuration experiment."""

    timestamp = datetime.now().isoformat()

    try:
        # Create model
        model = EEGNet(
            input_size=n_channels,
            projected_channels=config["projected_channels"],
            window_size=config["window_size"],
            F1=config["F1"],
            D=config["D"],
            dropout=config["dropout"],
        )

        # Train
        train_start = time.perf_counter()
        model.fit(
            X=train_features,
            y=train_labels,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=config["learning_rate"],
            verbose=False,  # Suppress per-batch output
            X_val=val_features,
            y_val=val_labels,
        )
        training_time = time.perf_counter() - train_start

        # Save model temporarily to get size
        model_path = model.save()
        model_size_bytes = model_path.stat().st_size

        # Run inference on validation set
        inference_start = time.perf_counter()
        predictions = []
        for sample in val_features:
            pred = model.predict(sample)
            predictions.append(pred)
        inference_time = time.perf_counter() - inference_start

        # Compute metrics
        y_true = val_labels
        y_pred = np.array(predictions)

        metrics = compute_score(
            y_true=y_true,
            y_pred=y_pred,
            model_size_bytes=model_size_bytes,
        )

        result = ExperimentResult(
            window_size=config["window_size"],
            projected_channels=config["projected_channels"],
            F1=config["F1"],
            D=config["D"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            balanced_accuracy=metrics.accuracy,
            avg_lag_samples=metrics.avg_lag_samples,
            model_size_bytes=model_size_bytes,
            accuracy_score=metrics.accuracy_score,
            lag_score=metrics.lag_score,
            size_score=metrics.size_score,
            total_score=metrics.total_score,
            training_time_seconds=training_time,
            inference_time_seconds=inference_time,
            timestamp=timestamp,
            error=None,
        )

        # Clear GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        traceback.print_exc()

        return ExperimentResult(
            window_size=config["window_size"],
            projected_channels=config["projected_channels"],
            F1=config["F1"],
            D=config["D"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            balanced_accuracy=0.0,
            avg_lag_samples=500.0,
            model_size_bytes=0,
            accuracy_score=0.0,
            lag_score=0.0,
            size_score=0.0,
            total_score=0.0,
            training_time_seconds=0.0,
            inference_time_seconds=0.0,
            timestamp=timestamp,
            error=str(e),
        )


def save_results(results: list[dict], best_score: float):
    """Save results to JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump({
            "results": results,
            "best_score": best_score,
            "total_experiments": len(results),
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)

    # Save CSV for easy analysis
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("total_score", ascending=False)
        df.to_csv(RESULTS_CSV, index=False)


def load_checkpoint() -> tuple[list[dict], float]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        return data.get("results", []), data.get("best_score", 0.0)
    return [], 0.0


def save_checkpoint(results: list[dict], best_score: float):
    """Save checkpoint for resume capability."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "results": results,
            "best_score": best_score,
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)


def save_best_model(result: ExperimentResult, model_path: Path):
    """Save the best model configuration."""
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import shutil
    best_path = BEST_MODEL_DIR / f"best_model_score{result.total_score:.2f}.pt"
    shutil.copy(model_path, best_path)

    # Save config
    config_path = BEST_MODEL_DIR / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Saved best model to {best_path}")


def print_progress(completed: int, total: int, result: ExperimentResult, best_score: float):
    """Print progress update."""
    pct = (completed / total) * 100

    print(f"\n{'='*70}")
    print(f"Progress: {completed}/{total} ({pct:.1f}%)")
    print(f"{'='*70}")
    print(f"Config: window={result.window_size}, channels={result.projected_channels}, "
          f"F1={result.F1}, D={result.D}, dropout={result.dropout}, lr={result.learning_rate}")
    print(f"Score: {result.total_score:.2f} (acc={result.accuracy_score:.1f}, "
          f"lag={result.lag_score:.1f}, size={result.size_score:.1f})")
    print(f"Accuracy: {result.balanced_accuracy*100:.2f}%, Lag: {result.avg_lag_samples:.1f}ms, "
          f"Size: {result.model_size_bytes/1024:.1f}KB")
    print(f"Best score so far: {best_score:.2f}")
    print(f"Training time: {result.training_time_seconds:.1f}s")
    if result.error:
        print(f"ERROR: {result.error}")
    print(f"{'='*70}\n")


def estimate_total_time(configs: list[dict], avg_time_per_config: float = 120.0) -> str:
    """Estimate total runtime."""
    total_seconds = len(configs) * avg_time_per_config
    hours = total_seconds / 3600
    return f"{hours:.1f} hours"


def main():
    parser = argparse.ArgumentParser(description="Overnight EEGNet hyperparameter search")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced grid (~12 min)")
    parser.add_argument("--exhaustive", action="store_true", help="Full exhaustive grid (~12 days)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup
    log_file = setup_logging()
    logger.info(f"Logging to {log_file}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Select grid
    if args.quick:
        grid = QUICK_GRID
    elif args.exhaustive:
        grid = EXHAUSTIVE_GRID
    else:
        grid = FULL_GRID
    configs = generate_configs(grid)

    logger.info(f"Total configurations to test: {len(configs)}")
    logger.info(f"Estimated runtime: {estimate_total_time(configs)}")

    # Load data
    train_features, train_labels, val_features, val_labels = load_data()
    n_channels = train_features.shape[1]

    # Convert to numpy
    train_X = train_features.values
    train_y = train_labels["label"].values
    val_X = val_features.values
    val_y = val_labels["label"].values

    # Load checkpoint if resuming
    results = []
    best_score = 0.0

    if args.resume:
        results, best_score = load_checkpoint()
        logger.info(f"Resumed from checkpoint: {len(results)} completed, best score: {best_score:.2f}")

    completed_keys = get_completed_configs(results)

    # Run experiments
    start_time = time.time()

    for i, config in enumerate(configs):
        key = config_to_key(config)

        # Skip if already completed
        if key in completed_keys:
            logger.info(f"Skipping already completed: {key}")
            continue

        logger.info(f"\n[{i+1}/{len(configs)}] Testing config: {config}")

        # Run experiment
        result = run_single_experiment(
            config=config,
            train_features=train_X,
            train_labels=train_y,
            val_features=val_X,
            val_labels=val_y,
            n_channels=n_channels,
        )

        results.append(result.to_dict())

        # Update best
        if result.total_score > best_score:
            best_score = result.total_score
            logger.info(f"NEW BEST SCORE: {best_score:.2f}")
            # Save best model
            model_path = Path("./model.pt")
            if model_path.exists():
                save_best_model(result, model_path)

        # Progress update
        print_progress(len(results), len(configs), result, best_score)

        # Save checkpoint after each experiment
        save_checkpoint(results, best_score)
        save_results(results, best_score)

    # Final summary
    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"

    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*70)
    print(f"Total experiments: {len(results)}")
    print(f"Total time: {elapsed_str}")
    print(f"Best score: {best_score:.2f}")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Best model saved to: {BEST_MODEL_DIR}")
    print("="*70)

    # Print top 10 configurations
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("total_score", ascending=False)

        print("\nTOP 10 CONFIGURATIONS:")
        print("-"*70)
        for i, row in df.head(10).iterrows():
            print(f"Score: {row['total_score']:.2f} | "
                  f"Acc: {row['balanced_accuracy']*100:.1f}% | "
                  f"Lag: {row['avg_lag_samples']:.0f}ms | "
                  f"win={row['window_size']}, ch={row['projected_channels']}, "
                  f"F1={row['F1']}, D={row['D']}")

        # Analysis
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)

        # Best by window size
        print("\nBest score by window size:")
        for ws in sorted(df["window_size"].unique()):
            ws_best = df[df["window_size"] == ws]["total_score"].max()
            print(f"  window_size={ws}: {ws_best:.2f}")

        # Best by projected channels
        print("\nBest score by projected channels:")
        for pc in sorted(df["projected_channels"].unique()):
            pc_best = df[df["projected_channels"] == pc]["total_score"].max()
            print(f"  projected_channels={pc}: {pc_best:.2f}")

        # Correlation analysis
        print("\nCorrelation with total_score:")
        numeric_cols = ["window_size", "projected_channels", "F1", "D", "dropout", "learning_rate"]
        for col in numeric_cols:
            corr = df[col].corr(df["total_score"])
            print(f"  {col}: {corr:.3f}")


if __name__ == "__main__":
    main()

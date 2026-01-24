#!/usr/bin/env python3
"""
Single configuration trainer for SLURM array jobs.

This script runs ONE hyperparameter configuration based on SLURM_ARRAY_TASK_ID.
Use with submit_hypersearch.sbatch to run many configs in parallel.

Usage:
    # Direct run with specific config index
    python examples/slurm_hypersearch.py --config-id 0

    # Via SLURM (automatic from array job)
    sbatch examples/submit_hypersearch.sbatch
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.ml.eegnet import EEGNet
from brainstorm.ml.metrics import compute_score


# =============================================================================
# Configuration Grid
# =============================================================================

HYPERPARAM_GRID = {
    "window_size": [64, 128, 256, 384, 512, 768, 1024, 1280, 1600, 2000],
    "projected_channels": [32, 64, 96],
    "F1": [8, 16],
    "D": [2],
    "dropout": [0.25, 0.4],
    "learning_rate": [1e-3],
}

DATA_PATH = Path("./data")
RESULTS_DIR = Path("./hypersearch_results")
EPOCHS = 30
BATCH_SIZE = 64


def generate_all_configs() -> list[dict]:
    """Generate all hyperparameter configurations."""
    keys = list(HYPERPARAM_GRID.keys())
    values = list(HYPERPARAM_GRID.values())

    configs = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def get_config_by_id(config_id: int) -> dict:
    """Get a specific configuration by index."""
    configs = generate_all_configs()
    if config_id < 0 or config_id >= len(configs):
        raise ValueError(f"config_id {config_id} out of range [0, {len(configs)})")
    return configs[config_id]


def load_data():
    """Load training and validation data."""
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        logger.info("Downloading data...")
        download_train_validation_data()

    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")

    return (
        train_features.values,
        train_labels["label"].values,
        val_features.values,
        val_labels["label"].values,
        train_features.shape[1],
    )


def run_experiment(config: dict, config_id: int) -> dict:
    """Run a single experiment and return results."""

    timestamp = datetime.now().isoformat()
    result = {
        "config_id": config_id,
        **config,
        "timestamp": timestamp,
    }

    try:
        # Load data
        train_X, train_y, val_X, val_y, n_channels = load_data()

        # Create and train model
        model = EEGNet(
            input_size=n_channels,
            projected_channels=config["projected_channels"],
            window_size=config["window_size"],
            F1=config["F1"],
            D=config["D"],
            dropout=config["dropout"],
        )

        train_start = time.perf_counter()
        model.fit(
            X=train_X,
            y=train_y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=config["learning_rate"],
            verbose=True,
            X_val=val_X,
            y_val=val_y,
        )
        training_time = time.perf_counter() - train_start

        # Get model size
        model_path = Path("./model.pt")
        model_size_bytes = model_path.stat().st_size

        # Run inference
        inference_start = time.perf_counter()
        predictions = [model.predict(sample) for sample in val_X]
        inference_time = time.perf_counter() - inference_start

        # Compute metrics
        metrics = compute_score(
            y_true=val_y,
            y_pred=np.array(predictions),
            model_size_bytes=model_size_bytes,
        )

        result.update({
            "balanced_accuracy": metrics.accuracy,
            "avg_lag_samples": metrics.avg_lag_samples,
            "model_size_bytes": model_size_bytes,
            "accuracy_score": metrics.accuracy_score,
            "lag_score": metrics.lag_score,
            "size_score": metrics.size_score,
            "total_score": metrics.total_score,
            "training_time_seconds": training_time,
            "inference_time_seconds": inference_time,
            "error": None,
        })

        # Save model if good score
        if metrics.total_score > 65:
            save_dir = RESULTS_DIR / "good_models"
            save_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(model_path, save_dir / f"model_config{config_id}_score{metrics.total_score:.2f}.pt")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        traceback.print_exc()
        result.update({
            "balanced_accuracy": 0.0,
            "avg_lag_samples": 500.0,
            "model_size_bytes": 0,
            "accuracy_score": 0.0,
            "lag_score": 0.0,
            "size_score": 0.0,
            "total_score": 0.0,
            "training_time_seconds": 0.0,
            "inference_time_seconds": 0.0,
            "error": str(e),
        })

    return result


def save_result(result: dict, config_id: int):
    """Save individual result to file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_file = RESULTS_DIR / f"result_{config_id:04d}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Result saved to {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Run single hyperparam config")
    parser.add_argument("--config-id", type=int, default=None,
                        help="Config index (default: from SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--list-configs", action="store_true",
                        help="List all configurations and exit")
    args = parser.parse_args()

    # List mode
    if args.list_configs:
        configs = generate_all_configs()
        print(f"Total configurations: {len(configs)}")
        for i, cfg in enumerate(configs):
            print(f"  [{i:3d}] {cfg}")
        return

    # Get config ID
    config_id = args.config_id
    if config_id is None:
        config_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # Setup logging
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RESULTS_DIR / f"config_{config_id:04d}.log"
    logger.add(log_file, level="DEBUG")

    # Get config
    configs = generate_all_configs()
    logger.info(f"Running config {config_id}/{len(configs)-1}")

    config = get_config_by_id(config_id)
    logger.info(f"Config: {config}")

    # Run experiment
    result = run_experiment(config, config_id)

    # Save result
    save_result(result, config_id)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Config {config_id}: {config}")
    print(f"Total Score: {result['total_score']:.2f}")
    print(f"  Accuracy: {result['balanced_accuracy']*100:.1f}% (score: {result['accuracy_score']:.1f})")
    print(f"  Lag: {result['avg_lag_samples']:.1f}ms (score: {result['lag_score']:.1f})")
    print(f"  Size: {result['model_size_bytes']/1024:.1f}KB (score: {result['size_score']:.1f})")
    if result['error']:
        print(f"  ERROR: {result['error']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

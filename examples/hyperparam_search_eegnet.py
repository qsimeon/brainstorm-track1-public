#!/usr/bin/env python3
"""
Hyperparameter search for EEGNet.

This script performs a grid search over key hyperparameters while:
1. Using seeds for reproducibility
2. Evaluating on the full validation set
3. Preserving the working baseline configuration

Current baseline (val_bal_acc=0.6460):
- projected_channels=64, window_size=128, F1=8, D=2
- dropout=0.25, lr=1e-3, batch_size=64, epochs=30

Usage:
    python examples/hyperparam_search_eegnet.py
"""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from brainstorm.loading import load_raw_data
from brainstorm.ml.eegnet import EEGNet


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    train_features,
    train_labels,
    val_features,
    val_labels,
    config: dict,
    checkpoint_dir: Path,
    seed: int = 42,
) -> dict:
    """Run a single experiment with given config."""
    set_seed(seed)

    logger.info(f"Running experiment with config: {config}")
    logger.info(f"Seed: {seed}")

    # Create model
    model = EEGNet(
        input_size=train_features.shape[1],
        projected_channels=config["projected_channels"],
        window_size=config["window_size"],
        F1=config["F1"],
        D=config["D"],
        dropout=config["dropout"],
    )

    # Train model and track best val accuracy manually
    # (since checkpoint path is hardcoded, we track during training)
    model.fit_model(
        X=train_features.values,
        y=train_labels["label"].values,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
        verbose=True,
        X_val=val_features.values,
        y_val=val_labels["label"].values,
    )

    # Load the best checkpoint (saved to hardcoded path)
    checkpoint_path = Path("/media/M2SSD/mind_meld_checkpoints/eegnet_best.pt")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    best_val_acc = checkpoint.get("val_bal_acc", 0.0)
    best_epoch = checkpoint.get("epoch", 0)

    # Copy checkpoint to experiment directory
    experiment_checkpoint_path = checkpoint_dir / "best_model.pt"
    torch.save(checkpoint, experiment_checkpoint_path)

    result = {
        "config": config,
        "seed": seed,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    with open(checkpoint_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    # ==========================================================================
    # Configuration
    # ==========================================================================

    DATA_PATH = Path("./data")
    RESULTS_DIR = Path("/media/M2SSD/mind_meld_checkpoints/hypersearch_v2")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Base seed for reproducibility
    BASE_SEED = 42

    # Baseline config (known to work well: 0.6460 val_bal_acc)
    BASELINE = {
        "projected_channels": 64,
        "window_size": 128,
        "F1": 8,
        "D": 2,
        "dropout": 0.25,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 30,
    }

    # Hyperparameters to search (one at a time from baseline)
    # This is a more careful approach than full grid search
    SEARCH_SPACE = {
        # Vary window_size (temporal context)
        "window_size": [64, 128, 256],
        # Vary PCA channels
        "projected_channels": [32, 64, 128],
        # Vary learning rate
        "learning_rate": [5e-4, 1e-3, 2e-3],
        # Vary dropout
        "dropout": [0.25, 0.4, 0.5],
    }

    # ==========================================================================
    # Load data once
    # ==========================================================================

    logger.info("Loading data...")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")

    logger.info(f"Train: {train_features.shape}, Val: {val_features.shape}")
    logger.info(f"Val samples: {len(val_features)} (full validation set)")

    # ==========================================================================
    # Run experiments
    # ==========================================================================

    all_results = []

    # First, run baseline to confirm reproducibility
    logger.info("=" * 60)
    logger.info("Running BASELINE experiment")
    logger.info("=" * 60)

    baseline_dir = RESULTS_DIR / "baseline"
    baseline_dir.mkdir(exist_ok=True)

    baseline_result = run_experiment(
        train_features, train_labels,
        val_features, val_labels,
        config=BASELINE,
        checkpoint_dir=baseline_dir,
        seed=BASE_SEED,
    )
    all_results.append(baseline_result)
    logger.info(f"Baseline result: {baseline_result['best_val_acc']:.4f} at epoch {baseline_result['best_epoch']}")

    # Then, vary each hyperparameter one at a time
    for param_name, param_values in SEARCH_SPACE.items():
        for value in param_values:
            # Skip if this is the baseline value
            if value == BASELINE[param_name]:
                continue

            # Create config with this variation
            config = BASELINE.copy()
            config[param_name] = value

            exp_name = f"{param_name}_{value}"
            exp_dir = RESULTS_DIR / exp_name
            exp_dir.mkdir(exist_ok=True)

            logger.info("=" * 60)
            logger.info(f"Running experiment: {exp_name}")
            logger.info("=" * 60)

            result = run_experiment(
                train_features, train_labels,
                val_features, val_labels,
                config=config,
                checkpoint_dir=exp_dir,
                seed=BASE_SEED,
            )
            all_results.append(result)

            logger.info(f"Result: {result['best_val_acc']:.4f} at epoch {result['best_epoch']}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Sort by val accuracy
    all_results.sort(key=lambda x: x["best_val_acc"], reverse=True)

    print(f"\n{'Experiment':<30} {'Val Acc':>10} {'Epoch':>8}")
    print("-" * 50)
    for r in all_results:
        # Create experiment name from config diff vs baseline
        config = r["config"]
        diff_parts = []
        for k, v in config.items():
            if k in BASELINE and v != BASELINE[k]:
                diff_parts.append(f"{k}={v}")
        exp_name = ", ".join(diff_parts) if diff_parts else "baseline"
        print(f"{exp_name:<30} {r['best_val_acc']:>10.4f} {r['best_epoch']:>8}")

    # Save all results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {RESULTS_DIR}")
    logger.info(f"Best: {all_results[0]['config']} with {all_results[0]['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()

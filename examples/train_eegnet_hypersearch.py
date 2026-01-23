#!/usr/bin/env python3
"""
EEGNet training script for hyperparameter search.

This script is similar to train_eegnet.py but allows specifying a custom
checkpoint directory for saving the best model during hyperparameter search.

Usage:
    python examples/train_eegnet_hypersearch.py \
        --window-size 64 \
        --projected-channels 32 \
        --checkpoint-dir /path/to/save
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.ml.eegnet import EEGNet, SlidingWindowDataset
from brainstorm.ml.channel_projection import PCAProjection


DATA_PATH = Path("./data")

# Default parameters
WINDOW_SIZE = 128
PROJECTED_CHANNELS = 64
F1 = 8
D = 2
DROPOUT = 0.25
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEGNet with custom checkpoint directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--window-size", type=int, default=WINDOW_SIZE,
        help=f"Window size in samples (default: {WINDOW_SIZE})"
    )
    parser.add_argument(
        "--projected-channels", type=int, default=PROJECTED_CHANNELS,
        help=f"Number of PCA channels (default: {PROJECTED_CHANNELS})"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--F1", type=int, default=F1,
        help=f"Number of temporal filters (default: {F1})"
    )
    parser.add_argument(
        "--D", type=int, default=D,
        help=f"Depthwise multiplier (default: {D})"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Directory to save checkpoints"
    )
    return parser.parse_args()


def train_with_checkpoint_dir(
    train_features,
    train_labels,
    val_features,
    val_labels,
    window_size: int,
    projected_channels: int,
    F1: int,
    D: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: Path,
) -> dict:
    """
    Train EEGNet and save best checkpoint to specified directory.

    Returns dict with best metrics.
    """
    input_size = train_features.shape[1]

    # Determine classes
    y = train_labels["label"].values
    y_val = val_labels["label"].values
    classes = np.unique(y)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    logger.info(f"Training EEGNet with {n_classes} classes")
    logger.info(f"Window size: {window_size}, PCA channels: {projected_channels}")

    # Fit PCA
    logger.info(f"Fitting PCA projection: {input_size} -> {projected_channels} channels")
    pca = PCAProjection(n_components=projected_channels)
    X_projected = pca.fit_transform(train_features.values)
    X_val_projected = pca.transform(val_features.values)

    # Build EEGNet
    from brainstorm.ml.eegnet import EEGNetCore
    eegnet = EEGNetCore(
        n_channels=projected_channels,
        n_classes=n_classes,
        window_samples=window_size,
        F1=F1,
        D=D,
        F2=F1 * D,
        dropout_rate=dropout,
    )

    # Create datasets
    logger.info(f"Creating datasets with window_size={window_size}")
    train_dataset = SlidingWindowDataset(X_projected, y, window_size, class_to_idx)
    val_dataset = SlidingWindowDataset(X_val_projected, y_val, window_size, class_to_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"Training data: {len(train_dataset)} windows")
    logger.info(f"Validation data: {len(val_dataset)} windows")

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Training on device: {device}")
    eegnet = eegnet.to(device)

    # Compute class weights
    y_indices = np.array([class_to_idx[label] for label in y[window_size:]])
    class_counts = np.bincount(y_indices, minlength=n_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(eegnet.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = checkpoint_dir / "best_model.pt"

    # Training loop
    for epoch in range(epochs):
        eegnet.train()
        total_loss = 0.0
        n_batches = 0

        batch_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
        )

        for X_batch, y_batch in batch_iterator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = eegnet(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(eegnet.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_iterator.set_postfix(loss=f"{total_loss/n_batches:.4f}")

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Validation
        eegnet.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                logits = eegnet(X_val_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_val_batch.numpy())

        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)

        # Save best model
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            best_epoch = epoch + 1

            checkpoint = {
                "config": {
                    "input_size": input_size,
                    "projected_channels": projected_channels,
                    "window_size": window_size,
                    "n_classes": n_classes,
                    "F1": F1,
                    "D": D,
                    "dropout": dropout,
                    "use_pca": True,
                },
                "classes": classes,
                "pca_mean": pca.mean_,
                "pca_components": pca.components_,
                "eegnet_state_dict": eegnet.state_dict(),
                "epoch": epoch + 1,
                "val_bal_acc": val_bal_acc,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_bal_acc:.4f} [BEST]")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_bal_acc:.4f}")

    # Save results summary
    results = {
        "window_size": window_size,
        "projected_channels": projected_channels,
        "F1": F1,
        "D": D,
        "epochs": epochs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_loss": avg_loss,
    }

    with open(checkpoint_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rprint(f"\n[bold cyan]EEGNet Hyperparameter Search[/]\n")
    rprint(f"  Window Size: {args.window_size}")
    rprint(f"  PCA Channels: {args.projected_channels}")
    rprint(f"  Checkpoint Dir: {checkpoint_dir}")
    print()

    # Download data if needed
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()

    # Load data
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")

    # Train
    results = train_with_checkpoint_dir(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        window_size=args.window_size,
        projected_channels=args.projected_channels,
        F1=args.F1,
        D=args.D,
        dropout=DROPOUT,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=checkpoint_dir,
    )

    # Print summary
    console = Console()
    table = Table(title="Results Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Best Val Balanced Acc", f"{results['best_val_acc']:.4f}")
    table.add_row("Best Epoch", str(results['best_epoch']))
    table.add_row("Window Size", str(results['window_size']))
    table.add_row("PCA Channels", str(results['projected_channels']))
    table.add_row("Checkpoint", str(checkpoint_dir / "best_model.pt"))

    console.print(table)

    rprint(f"\n[bold green]Training complete![/]")
    rprint(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train and evaluate the EEGNet model for continuous classification.

This script demonstrates training the EEGNet-based model with:
    1. PCA channel projection (1024 -> 64 channels)
    2. Temporal windowing for context
    3. Compact convolutional architecture

Usage:
    python examples/train_eegnet.py
    python examples/train_eegnet.py --window-size 1600  # Match Wav2Vec2
    python examples/train_eegnet.py --projected-channels 32 --epochs 50

The trained model and metadata are saved to the repository root.
"""

import argparse
from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.eegnet import EEGNet


# =============================================================================
# Default Configuration
# =============================================================================

DATA_PATH = Path("./data")

# EEGNet parameters (defaults)
DEFAULT_PROJECTED_CHANNELS = 64  # Number of channels after PCA
DEFAULT_WINDOW_SIZE = 128  # Temporal context window (128ms at 1000Hz)
DEFAULT_F1 = 8  # Number of temporal filters
DEFAULT_D = 2  # Depthwise multiplier
DEFAULT_DROPOUT = 0.25

# Training parameters (defaults)
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet model for ECoG classification")

    # Model architecture
    parser.add_argument("--projected-channels", type=int, default=DEFAULT_PROJECTED_CHANNELS,
                        help=f"Number of channels after PCA projection (default: {DEFAULT_PROJECTED_CHANNELS})")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                        help=f"Temporal context window in samples (default: {DEFAULT_WINDOW_SIZE})")
    parser.add_argument("--F1", type=int, default=DEFAULT_F1,
                        help=f"Number of temporal filters (default: {DEFAULT_F1})")
    parser.add_argument("--D", type=int, default=DEFAULT_D,
                        help=f"Depthwise multiplier (default: {DEFAULT_D})")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT,
                        help=f"Dropout rate (default: {DEFAULT_DROPOUT})")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rprint("\n[bold cyan]EEGNet Training Script[/]\n")

    # Download data if needed
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]Data downloaded.[/]\n")

    rprint(f"[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    validation_features, validation_labels = load_raw_data(DATA_PATH, step="validation")

    # Display dataset info
    console = Console()
    table = Table(
        title="Dataset Overview", show_header=True, header_style="bold magenta"
    )

    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Time Range (s)", style="yellow")
    table.add_column("Unique Labels", style="blue")

    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} -> {train_features.index[-1]:.2f}",
        str(sorted(train_labels["label"].unique().tolist())),
    )

    table.add_row(
        "Validation",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} -> {validation_features.index[-1]:.2f}",
        str(sorted(validation_labels["label"].unique().tolist())),
    )

    console.print(table)
    print()

    # Create and train model
    rprint("\n[bold green]Training EEGNet model...[/]\n")
    rprint(f"  Projected channels: {args.projected_channels}")
    rprint(f"  Window size: {args.window_size} samples")
    rprint(f"  Temporal filters (F1): {args.F1}")
    rprint(f"  Depthwise multiplier (D): {args.D}")
    rprint(f"  Dropout: {args.dropout}")
    rprint(f"  Epochs: {args.epochs}")
    rprint(f"  Batch size: {args.batch_size}")
    rprint(f"  Learning rate: {args.learning_rate}")
    print()

    model = EEGNet(
        input_size=train_features.shape[1],
        projected_channels=args.projected_channels,
        window_size=args.window_size,
        F1=args.F1,
        D=args.D,
        dropout=args.dropout,
    )

    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=True,
        X_val=validation_features.values,
        y_val=validation_labels["label"].values,
    )

    # Evaluate on validation set
    rprint("\n[bold green]Evaluating on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]Training and evaluation complete.[/]\n")


if __name__ == "__main__":
    main()

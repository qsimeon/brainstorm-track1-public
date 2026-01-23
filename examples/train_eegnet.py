#!/usr/bin/env python3
"""
Train and evaluate the EEGNet model for continuous classification.

This script demonstrates training the EEGNet-based model with:
    1. PCA channel projection (1024 -> 64 channels)
    2. Temporal windowing for context
    3. Compact convolutional architecture

Usage:
    python examples/train_eegnet.py

The trained model and metadata are saved to the repository root.
"""

from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.eegnet import EEGNet


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")

# EEGNet parameters
PROJECTED_CHANNELS = 64  # Number of channels after PCA
WINDOW_SIZE = 128  # Temporal context window (128ms at 1000Hz)
F1 = 8  # Number of temporal filters
D = 2  # Depthwise multiplier
DROPOUT = 0.25

# Training parameters
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def main() -> None:
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
    rprint(f"  Projected channels: {PROJECTED_CHANNELS}")
    rprint(f"  Window size: {WINDOW_SIZE} samples")
    rprint(f"  Temporal filters (F1): {F1}")
    rprint(f"  Depthwise multiplier (D): {D}")
    rprint(f"  Epochs: {EPOCHS}")
    print()

    model = EEGNet(
        input_size=train_features.shape[1],
        projected_channels=PROJECTED_CHANNELS,
        window_size=WINDOW_SIZE,
        F1=F1,
        D=D,
        dropout=DROPOUT,
    )

    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
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

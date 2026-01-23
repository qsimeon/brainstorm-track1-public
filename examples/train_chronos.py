#!/usr/bin/env python3
"""
Train and evaluate the Chronos-based classifier for continuous classification.

This script fine-tunes Amazon's Chronos time series foundation model for
ECoG signal classification. Chronos is pretrained on diverse time series data
and learns general temporal patterns that transfer well to new domains.

Features:
    1. PCA channel projection (1024 -> 32 channels)
    2. Pretrained Chronos T5-Tiny encoder (frozen or fine-tuned)
    3. Custom classification head
    4. Temporal windowing for context

Usage:
    python examples/train_chronos.py

    # With encoder fine-tuning (slower but potentially better)
    python examples/train_chronos.py --unfreeze-encoder

    # Custom parameters
    python examples/train_chronos.py --epochs 30 --window-size 64

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
from brainstorm.ml.chronos_classifier import ChronosClassifier


# =============================================================================
# Default Configuration
# =============================================================================

DATA_PATH = Path("./data")

# Model parameters
PROJECTED_CHANNELS = 32  # Fewer channels for Chronos (processes each independently)
WINDOW_SIZE = 128  # Temporal context window (128ms at 1000Hz)
FREEZE_ENCODER = True  # Freeze pretrained encoder by default
DROPOUT = 0.1

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32  # Smaller batch due to Chronos memory usage
LEARNING_RATE = 1e-3  # For classification head
ENCODER_LR = 1e-5  # For encoder (if not frozen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Chronos-based classifier for ECoG signals"
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
        help=f"Learning rate for classifier (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--unfreeze-encoder", action="store_true",
        help="Fine-tune the Chronos encoder (default: frozen)"
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=ENCODER_LR,
        help=f"Learning rate for encoder if unfrozen (default: {ENCODER_LR})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rprint("\n[bold cyan]Chronos Foundation Model Training Script[/]\n")
    rprint("[bold yellow]Using Amazon Chronos-T5-Tiny as pretrained encoder[/]\n")

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

    # Display model configuration
    freeze_encoder = not args.unfreeze_encoder

    config_table = Table(
        title="Model Configuration", show_header=True, header_style="bold magenta"
    )
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Foundation Model", "amazon/chronos-t5-tiny")
    config_table.add_row("Projected Channels", str(args.projected_channels))
    config_table.add_row("Window Size", f"{args.window_size} samples ({args.window_size}ms)")
    config_table.add_row("Encoder Frozen", str(freeze_encoder))
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Classifier LR", str(args.learning_rate))
    if not freeze_encoder:
        config_table.add_row("Encoder LR", str(args.encoder_lr))

    console.print(config_table)
    print()

    # Create and train model
    rprint("\n[bold green]Training Chronos-based classifier...[/]\n")

    model = ChronosClassifier(
        input_size=train_features.shape[1],
        projected_channels=args.projected_channels,
        window_size=args.window_size,
        freeze_encoder=freeze_encoder,
        dropout=DROPOUT,
    )

    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_lr=args.encoder_lr,
        verbose=True,
    )

    # Evaluate on validation set
    rprint("\n[bold green]Evaluating on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    # Print model size info
    model_path = Path("model.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        rprint(f"\n[bold cyan]Model size:[/] {size_mb:.2f} MB")
        if size_mb <= 5:
            rprint("[bold green]Optimal size score (<5MB)[/]")
        elif size_mb <= 25:
            rprint("[bold yellow]Acceptable size (<25MB)[/]")
        else:
            rprint("[bold red]Warning: Model exceeds 25MB limit![/]")

    rprint("\n[bold green]Training and evaluation complete.[/]\n")


if __name__ == "__main__":
    main()

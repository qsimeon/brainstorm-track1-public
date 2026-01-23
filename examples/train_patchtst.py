#!/usr/bin/env python3
"""
Train and evaluate PatchTST for continuous classification.

PatchTST is a transformer architecture designed for time series that:
- Divides input into patches (like Vision Transformer)
- Processes patches with self-attention
- Is efficient and has good inductive bias for time series

This trains from scratch (no pretrained weights) but is fast to train
and has excellent inference latency (~5-10ms).

Usage:
    python examples/train_patchtst.py

    # Custom parameters
    python examples/train_patchtst.py --epochs 30 --d-model 64

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
from brainstorm.ml.patchtst import PatchTST


# =============================================================================
# Default Configuration
# =============================================================================

DATA_PATH = Path("./data")

# Model parameters
PROJECTED_CHANNELS = 64  # Channels after PCA
WINDOW_SIZE = 128  # Temporal context window (128ms at 1000Hz)
PATCH_LENGTH = 16  # Length of each patch
STRIDE = 8  # Stride between patches
D_MODEL = 64  # Transformer dimension
NUM_HEADS = 4  # Attention heads
NUM_LAYERS = 3  # Transformer layers
DROPOUT = 0.1

# Training parameters
EPOCHS = 30
BATCH_SIZE = 32  # Reduced for GPU memory
LEARNING_RATE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PatchTST for ECoG signal classification"
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
        "--d-model", type=int, default=D_MODEL,
        help=f"Transformer dimension (default: {D_MODEL})"
    )
    parser.add_argument(
        "--num-layers", type=int, default=NUM_LAYERS,
        help=f"Number of transformer layers (default: {NUM_LAYERS})"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rprint("\n[bold cyan]PatchTST Training Script[/]\n")
    rprint("[bold yellow]Training from scratch (fast inference ~5-10ms)[/]\n")

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
    config_table = Table(
        title="Model Configuration", show_header=True, header_style="bold magenta"
    )
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Architecture", "PatchTST (from scratch)")
    config_table.add_row("Projected Channels", str(args.projected_channels))
    config_table.add_row("Window Size", f"{args.window_size} samples ({args.window_size}ms)")
    config_table.add_row("Patch Length", str(PATCH_LENGTH))
    config_table.add_row("Stride", str(STRIDE))
    config_table.add_row("d_model", str(args.d_model))
    config_table.add_row("Num Layers", str(args.num_layers))
    config_table.add_row("Num Heads", str(NUM_HEADS))
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Learning Rate", str(args.learning_rate))

    console.print(config_table)
    print()

    # Create and train model
    rprint("\n[bold green]Training PatchTST model...[/]\n")

    model = PatchTST(
        input_size=train_features.shape[1],
        projected_channels=args.projected_channels,
        window_size=args.window_size,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=args.d_model,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=args.num_layers,
        encoder_ffn_dim=args.d_model * 2,
        dropout=DROPOUT,
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

    # Test inference latency
    rprint("\n[bold cyan]Testing inference latency...[/]")
    import torch
    import time

    model.eval()
    sample = torch.randn(1, train_features.shape[1])

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.predict(sample.numpy().flatten())

    # Measure
    start = time.perf_counter()
    n_runs = 100
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model.predict(sample.numpy().flatten())
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    rprint(f"[bold cyan]Inference latency:[/] {elapsed:.2f} ms per sample")
    if elapsed < 10:
        rprint("[bold green]Excellent latency (<10ms)[/]")
    elif elapsed < 50:
        rprint("[bold yellow]Good latency (<50ms)[/]")
    else:
        rprint("[bold red]Warning: High latency (>50ms)[/]")

    rprint("\n[bold green]Training and evaluation complete.[/]\n")


if __name__ == "__main__":
    main()

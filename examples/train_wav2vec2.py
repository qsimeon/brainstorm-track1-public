#!/usr/bin/env python3
"""
Train and evaluate Wav2Vec2-based classifier with pretrained weights.

This script fine-tunes the tiny Wav2Vec2 model (1.2MB pretrained) for
ECoG signal classification. Wav2Vec2 was designed for audio waveforms
but can process any 1D signal.

Features:
    1. Pretrained encoder from wav2vec2_tiny_random
    2. PCA channel reduction (fewer channels for speed)
    3. Custom classification head
    4. ~0.5ms inference time

Note: Wav2Vec2 requires minimum ~400 samples due to convolutional kernels,
so we use a larger window size (512 samples = 512ms at 1000Hz).

Usage:
    python examples/train_wav2vec2.py

    # Unfreeze encoder for full fine-tuning
    python examples/train_wav2vec2.py --unfreeze-encoder

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
from brainstorm.ml.wav2vec2_classifier import Wav2Vec2Classifier


# =============================================================================
# Default Configuration
# =============================================================================

DATA_PATH = Path("./data")

# Model parameters
PROJECTED_CHANNELS = 8  # Fewer channels for speed (each goes through encoder)
WINDOW_SIZE = 1600  # Large window for wav2vec2 (needs min ~1600 samples)
FREEZE_ENCODER = True
DROPOUT = 0.1

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32  # Smaller batch due to encoder memory
LEARNING_RATE = 1e-3
ENCODER_LR = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Wav2Vec2-based classifier for ECoG signals"
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
        help=f"Window size in samples (default: {WINDOW_SIZE}, min 1600)"
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
        help="Fine-tune the Wav2Vec2 encoder (default: frozen)"
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=ENCODER_LR,
        help=f"Learning rate for encoder if unfrozen (default: {ENCODER_LR})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rprint("\n[bold cyan]Wav2Vec2 Pretrained Model Training Script[/]\n")
    rprint("[bold yellow]Using pretrained wav2vec2_tiny_random (1.2MB)[/]\n")

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

    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} -> {train_features.index[-1]:.2f}",
    )

    table.add_row(
        "Validation",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} -> {validation_features.index[-1]:.2f}",
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

    config_table.add_row("Foundation Model", "wav2vec2_tiny_random (1.2MB)")
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
    rprint("\n[bold green]Training Wav2Vec2-based classifier...[/]\n")

    model = Wav2Vec2Classifier(
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

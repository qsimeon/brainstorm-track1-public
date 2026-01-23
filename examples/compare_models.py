#!/usr/bin/env python3
"""
Compare EEGNet and Wav2Vec2 models on validation set.

Evaluates:
1. Balanced accuracy (50% of score)
2. Inference latency (25% of score)
3. Model size (25% of score)

Uses the official scoring formulas from the competition.

Usage:
    python examples/compare_models.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.loading import load_raw_data
from brainstorm.ml.eegnet import EEGNet
from brainstorm.ml.wav2vec2_classifier import Wav2Vec2Classifier


# Scoring formulas from evaluation.md
def accuracy_score(balanced_acc: float) -> float:
    """Accuracy Score = balanced_accuracy × 50"""
    return balanced_acc * 50


def lag_score(lag_ms: float) -> float:
    """Lag Score = exp(-6 × lag_ms / 500) × 25"""
    return np.exp(-6 * lag_ms / 500) * 25


def size_score(size_mb: float) -> float:
    """Size Score = exp(-4 × size_mb / 5) × 25"""
    return np.exp(-4 * size_mb / 5) * 25


def measure_latency(model, sample: np.ndarray, n_warmup: int = 10, n_runs: int = 100) -> float:
    """Measure inference latency in milliseconds."""
    model.eval()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.predict(sample)

    # Measure
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

    return elapsed_ms


def load_eegnet_from_checkpoint(checkpoint_path: Path) -> EEGNet:
    """Load EEGNet from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["config"]

    model = EEGNet(
        input_size=config["input_size"],
        projected_channels=config["projected_channels"],
        window_size=config["window_size"],
        F1=config["F1"],
        D=config["D"],
        dropout=config["dropout"],
    )

    # Restore PCA
    from brainstorm.ml.channel_projection import PCAProjection
    model.pca = PCAProjection(n_components=config["projected_channels"])
    model.pca.mean_ = checkpoint["pca_mean"]
    model.pca.components_ = checkpoint["pca_components"]
    model.pca.pca = True
    model.pca_layer = model.pca.get_torch_projection()

    # Restore network
    model.classes_ = checkpoint["classes"]
    model._build_network(config["n_classes"])
    model.eegnet.load_state_dict(checkpoint["eegnet_state_dict"])
    model.eval()
    model._init_window_buffer()

    return model


def load_wav2vec2_from_checkpoint(checkpoint_path: Path) -> Wav2Vec2Classifier:
    """Load Wav2Vec2 from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["config"]

    model = Wav2Vec2Classifier(
        input_size=config["input_size"],
        projected_channels=config["projected_channels"],
        window_size=config["window_size"],
        freeze_encoder=config["freeze_encoder"],
        dropout=config["dropout"],
        model_name=config["model_name"],
    )

    # Restore PCA
    from brainstorm.ml.channel_projection import PCAProjection
    model.pca = PCAProjection(n_components=config["projected_channels"])
    model.pca.mean_ = checkpoint["pca_mean"]
    model.pca.components_ = checkpoint["pca_components"]
    model.pca.pca = True
    model.pca_layer = model.pca.get_torch_projection()

    # Restore network
    model.classes_ = checkpoint["classes"]
    model._build_network(config["n_classes"])
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval()
    model._init_window_buffer()

    return model


def main():
    console = Console()

    # Paths
    eegnet_checkpoint = Path("/media/M2SSD/mind_meld_checkpoints/hypersearch_v2/baseline/best_model.pt")
    wav2vec2_checkpoint = Path("/media/M2SSD/mind_meld_checkpoints/wav2vec2_best.pt")
    data_path = Path("./data")

    # Check files exist
    if not eegnet_checkpoint.exists():
        rprint(f"[red]EEGNet checkpoint not found: {eegnet_checkpoint}[/]")
        return
    if not wav2vec2_checkpoint.exists():
        rprint(f"[red]Wav2Vec2 checkpoint not found: {wav2vec2_checkpoint}[/]")
        return

    rprint("\n[bold cyan]Model Comparison: EEGNet vs Wav2Vec2[/]\n")

    # Get file sizes
    eegnet_size_mb = eegnet_checkpoint.stat().st_size / (1024 * 1024)
    wav2vec2_size_mb = wav2vec2_checkpoint.stat().st_size / (1024 * 1024)

    rprint(f"[cyan]EEGNet checkpoint:[/] {eegnet_checkpoint}")
    rprint(f"[cyan]Wav2Vec2 checkpoint:[/] {wav2vec2_checkpoint}")

    # Load validation accuracy from checkpoint metadata
    eegnet_ckpt = torch.load(eegnet_checkpoint, weights_only=False)
    wav2vec2_ckpt = torch.load(wav2vec2_checkpoint, weights_only=False)

    eegnet_val_acc = eegnet_ckpt.get("val_bal_acc", 0.0)
    wav2vec2_val_acc = wav2vec2_ckpt.get("val_bal_acc", 0.0)

    rprint(f"\n[cyan]Loading models...[/]")

    # Load models
    eegnet_model = load_eegnet_from_checkpoint(eegnet_checkpoint)
    wav2vec2_model = load_wav2vec2_from_checkpoint(wav2vec2_checkpoint)

    # Load sample for latency test
    rprint("[cyan]Loading validation data for latency test...[/]")
    val_features, _ = load_raw_data(data_path, step="validation")
    sample = val_features.values[0]

    # Measure latency
    rprint("[cyan]Measuring inference latency...[/]")
    eegnet_latency = measure_latency(eegnet_model, sample)
    wav2vec2_latency = measure_latency(wav2vec2_model, sample)

    # Calculate scores
    eegnet_acc_score = accuracy_score(eegnet_val_acc)
    eegnet_lag_score = lag_score(eegnet_latency)
    eegnet_size_score = size_score(eegnet_size_mb)
    eegnet_total = eegnet_acc_score + eegnet_lag_score + eegnet_size_score

    wav2vec2_acc_score = accuracy_score(wav2vec2_val_acc)
    wav2vec2_lag_score = lag_score(wav2vec2_latency)
    wav2vec2_size_score = size_score(wav2vec2_size_mb)
    wav2vec2_total = wav2vec2_acc_score + wav2vec2_lag_score + wav2vec2_size_score

    # Display results
    print()

    # Raw metrics table
    metrics_table = Table(title="Raw Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("EEGNet", style="green", justify="right")
    metrics_table.add_column("Wav2Vec2", style="yellow", justify="right")
    metrics_table.add_column("Winner", style="bold", justify="center")

    # Balanced accuracy (higher is better)
    acc_winner = "EEGNet" if eegnet_val_acc > wav2vec2_val_acc else "Wav2Vec2"
    metrics_table.add_row(
        "Balanced Accuracy",
        f"{eegnet_val_acc:.4f}",
        f"{wav2vec2_val_acc:.4f}",
        acc_winner
    )

    # Latency (lower is better)
    lat_winner = "EEGNet" if eegnet_latency < wav2vec2_latency else "Wav2Vec2"
    metrics_table.add_row(
        "Inference Latency",
        f"{eegnet_latency:.2f} ms",
        f"{wav2vec2_latency:.2f} ms",
        lat_winner
    )

    # Size (lower is better)
    size_winner = "EEGNet" if eegnet_size_mb < wav2vec2_size_mb else "Wav2Vec2"
    metrics_table.add_row(
        "Model Size",
        f"{eegnet_size_mb:.3f} MB",
        f"{wav2vec2_size_mb:.3f} MB",
        size_winner
    )

    console.print(metrics_table)
    print()

    # Scores table
    scores_table = Table(title="Competition Scores", show_header=True, header_style="bold magenta")
    scores_table.add_column("Component", style="cyan")
    scores_table.add_column("Weight", style="white", justify="center")
    scores_table.add_column("EEGNet", style="green", justify="right")
    scores_table.add_column("Wav2Vec2", style="yellow", justify="right")

    scores_table.add_row(
        "Accuracy Score",
        "50%",
        f"{eegnet_acc_score:.2f}",
        f"{wav2vec2_acc_score:.2f}"
    )
    scores_table.add_row(
        "Lag Score",
        "25%",
        f"{eegnet_lag_score:.2f}",
        f"{wav2vec2_lag_score:.2f}"
    )
    scores_table.add_row(
        "Size Score",
        "25%",
        f"{eegnet_size_score:.2f}",
        f"{wav2vec2_size_score:.2f}"
    )
    scores_table.add_row(
        "[bold]TOTAL SCORE[/]",
        "[bold]100%[/]",
        f"[bold]{eegnet_total:.2f}[/]",
        f"[bold]{wav2vec2_total:.2f}[/]"
    )

    console.print(scores_table)
    print()

    # Winner
    if eegnet_total > wav2vec2_total:
        rprint(f"[bold green]Winner: EEGNet[/] ({eegnet_total:.2f} vs {wav2vec2_total:.2f})")
    elif wav2vec2_total > eegnet_total:
        rprint(f"[bold yellow]Winner: Wav2Vec2[/] ({wav2vec2_total:.2f} vs {eegnet_total:.2f})")
    else:
        rprint(f"[bold]Tie![/] ({eegnet_total:.2f})")

    print()

    # Configuration summary
    config_table = Table(title="Model Configurations", show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("EEGNet", style="green", justify="right")
    config_table.add_column("Wav2Vec2", style="yellow", justify="right")

    eegnet_cfg = eegnet_ckpt["config"]
    wav2vec2_cfg = wav2vec2_ckpt["config"]

    config_table.add_row("Window Size", str(eegnet_cfg["window_size"]), str(wav2vec2_cfg["window_size"]))
    config_table.add_row("PCA Channels", str(eegnet_cfg["projected_channels"]), str(wav2vec2_cfg["projected_channels"]))
    config_table.add_row("Dropout", str(eegnet_cfg["dropout"]), str(wav2vec2_cfg["dropout"]))
    config_table.add_row("Best Epoch", str(eegnet_ckpt.get("epoch", "?")), str(wav2vec2_ckpt.get("epoch", "?")))

    console.print(config_table)


if __name__ == "__main__":
    main()

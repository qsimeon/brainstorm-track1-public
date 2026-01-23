#!/usr/bin/env python3
"""
Benchmark EEGNet models with different configurations.

Compares the old (window=128, pca=64) vs new (window=1600, pca=32) EEGNet models
on CPU to measure latency across different hardware.

Usage:
    python examples/benchmark_eegnet.py

Scoring formulas (from docs/evaluation.md):
    Accuracy Score = balanced_accuracy × 50
    Lag Score      = exp(-6 × lag_ms / 500) × 25
    Size Score     = exp(-4 × size_mb / 5) × 25
    Total          = Accuracy Score + Lag Score + Size Score
"""

import platform
import time
from pathlib import Path

import numpy as np
import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.loading import load_raw_data
from brainstorm.ml.channel_projection import PCAProjection
from brainstorm.ml.eegnet import EEGNet


# =============================================================================
# Scoring Formulas (from docs/evaluation.md)
# =============================================================================


def accuracy_score(balanced_acc: float) -> float:
    """Accuracy Score = balanced_accuracy × 50"""
    return balanced_acc * 50


def lag_score(lag_ms: float) -> float:
    """Lag Score = exp(-6 × lag_ms / 500) × 25"""
    return np.exp(-6 * lag_ms / 500) * 25


def size_score(size_mb: float) -> float:
    """Size Score = exp(-4 × size_mb / 5) × 25"""
    return np.exp(-4 * size_mb / 5) * 25


# =============================================================================
# Helper Functions
# =============================================================================


def load_eegnet_from_checkpoint(checkpoint_path: Path) -> tuple[EEGNet, dict, dict]:
    """Load EEGNet from a checkpoint file (CPU only)."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
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
    model.pca = PCAProjection(n_components=config["projected_channels"])
    model.pca.mean_ = checkpoint["pca_mean"]
    model.pca.components_ = checkpoint["pca_components"]
    model.pca.pca = True
    model.pca_layer = model.pca.get_torch_projection()

    # Restore network
    model.classes_ = checkpoint["classes"]
    model._build_network(config["n_classes"])
    model.eegnet.load_state_dict(checkpoint["eegnet_state_dict"])
    model.to("cpu")
    model.eval()
    model._init_window_buffer()

    return model, config, checkpoint


def measure_latency_and_accuracy(
    model: EEGNet, X: np.ndarray, y: np.ndarray | None = None
) -> tuple[float, float | None]:
    """
    Measure inference latency and optionally balanced accuracy.

    Args:
        model: EEGNet model
        X: Input features (n_samples, n_channels)
        y: Optional labels for accuracy calculation

    Returns:
        Tuple of (latency_ms, balanced_accuracy or None)
    """
    from sklearn.metrics import balanced_accuracy_score

    model.eval()
    model._init_window_buffer()  # Reset buffer

    n_samples = len(X)
    predictions = []

    start = time.perf_counter()
    for i in range(n_samples):
        with torch.no_grad():
            pred = model.predict(X[i])
            predictions.append(pred)
    elapsed_ms = (time.perf_counter() - start) / n_samples * 1000

    # Calculate balanced accuracy if labels provided
    bal_acc = None
    if y is not None:
        bal_acc = balanced_accuracy_score(y, predictions)

    return elapsed_ms, bal_acc


def get_system_info() -> str:
    """Get system information string."""
    return f"{platform.system()} {platform.machine()} - {platform.processor() or 'Unknown CPU'}"


# =============================================================================
# Main
# =============================================================================


def main():
    console = Console()

    # Paths relative to repo root
    repo_root = Path(__file__).parent.parent
    old_checkpoint = repo_root / "checkpoints" / "archive" / "eegnet_window128_pca64.pt"
    new_checkpoint = repo_root / "checkpoints" / "eegnet_window1600_pca32.pt"
    data_path = repo_root / "data"

    # Check files exist
    if not old_checkpoint.exists():
        rprint(f"[red]Old checkpoint not found: {old_checkpoint}[/]")
        rprint("[yellow]Run from the repo root or ensure checkpoints are present.[/]")
        return

    if not new_checkpoint.exists():
        rprint(f"[red]New checkpoint not found: {new_checkpoint}[/]")
        rprint("[yellow]Run from the repo root or ensure checkpoints are present.[/]")
        return

    # Header
    rprint("\n[bold cyan]EEGNet Benchmark: Window Size Comparison[/]")
    rprint(f"[dim]System: {get_system_info()}[/]")
    rprint(f"[dim]Device: CPU (forced for cross-platform comparison)[/]\n")

    # Get file sizes
    old_size_mb = old_checkpoint.stat().st_size / (1024 * 1024)
    new_size_mb = new_checkpoint.stat().st_size / (1024 * 1024)

    # Load models
    rprint("[cyan]Loading models...[/]")
    old_model, old_cfg, old_ckpt = load_eegnet_from_checkpoint(old_checkpoint)
    new_model, new_cfg, new_ckpt = load_eegnet_from_checkpoint(new_checkpoint)

    old_val_acc = old_ckpt.get("val_bal_acc", 0.0)
    new_val_acc = new_ckpt.get("val_bal_acc", 0.0)

    # Load data for latency and accuracy test (download if needed)
    if not data_path.exists() or not any(data_path.glob("*.parquet")):
        rprint("[yellow]Data not found, downloading from Hugging Face...[/]")
        from brainstorm.download import download_train_validation_data
        download_train_validation_data()
        rprint("[green]Data downloaded.[/]\n")

    rprint("[cyan]Loading validation data...[/]")
    val_features, val_labels = load_raw_data(data_path, step="validation")
    X = val_features.values
    y = val_labels["label"].values
    n_samples = len(X)
    rprint(f"[dim]Running on {n_samples} samples for latency + accuracy measurement[/]")

    # Measure latency and accuracy on full dataset
    rprint(f"[cyan]Measuring OLD model ({n_samples} samples)...[/]")
    old_latency, old_measured_acc = measure_latency_and_accuracy(old_model, X, y)
    rprint(f"[cyan]Measuring NEW model ({n_samples} samples)...[/]\n")
    new_latency, new_measured_acc = measure_latency_and_accuracy(new_model, X, y)

    # Use measured accuracy if available, otherwise fall back to checkpoint values
    if old_measured_acc is not None:
        old_val_acc = old_measured_acc
        rprint(f"[dim]OLD model measured accuracy: {old_val_acc:.4f}[/]")
    if new_measured_acc is not None:
        new_val_acc = new_measured_acc
        rprint(f"[dim]NEW model measured accuracy: {new_val_acc:.4f}[/]")
    print()

    # Calculate scores
    old_acc_score = accuracy_score(old_val_acc)
    new_acc_score = accuracy_score(new_val_acc)
    old_lag_score = lag_score(old_latency)
    new_lag_score = lag_score(new_latency)
    old_size_score = size_score(old_size_mb)
    new_size_score = size_score(new_size_mb)
    old_total = old_acc_score + old_lag_score + old_size_score
    new_total = new_acc_score + new_lag_score + new_size_score

    # Configuration table
    config_table = Table(
        title="Model Configurations", show_header=True, header_style="bold magenta"
    )
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Old (window=128)", style="yellow", justify="right")
    config_table.add_column("New (window=1600)", style="green", justify="right")

    config_table.add_row(
        "Window Size", str(old_cfg["window_size"]), str(new_cfg["window_size"])
    )
    config_table.add_row(
        "PCA Channels",
        str(old_cfg["projected_channels"]),
        str(new_cfg["projected_channels"]),
    )
    config_table.add_row("F1 (temporal filters)", str(old_cfg["F1"]), str(new_cfg["F1"]))
    config_table.add_row("D (depthwise mult)", str(old_cfg["D"]), str(new_cfg["D"]))
    config_table.add_row("Dropout", str(old_cfg["dropout"]), str(new_cfg["dropout"]))
    config_table.add_row(
        "Best Epoch",
        str(old_ckpt.get("epoch", "?")),
        str(new_ckpt.get("epoch", "?")),
    )

    console.print(config_table)
    print()

    # Raw metrics table
    metrics_table = Table(
        title="Raw Metrics", show_header=True, header_style="bold magenta"
    )
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Old (window=128)", style="yellow", justify="right")
    metrics_table.add_column("New (window=1600)", style="green", justify="right")
    metrics_table.add_column("Change", style="bold", justify="center")

    # Balanced accuracy
    acc_change = new_val_acc - old_val_acc
    metrics_table.add_row(
        "Balanced Accuracy",
        f"{old_val_acc:.4f}",
        f"{new_val_acc:.4f}",
        f"[green]+{acc_change:.2%}[/]" if acc_change > 0 else f"[red]{acc_change:.2%}[/]",
    )

    # Latency
    lat_change = new_latency - old_latency
    metrics_table.add_row(
        "Inference Latency",
        f"{old_latency:.3f} ms",
        f"{new_latency:.3f} ms",
        f"[red]+{lat_change:.2f} ms[/]"
        if lat_change > 0
        else f"[green]{lat_change:.2f} ms[/]",
    )

    # Size
    size_change = new_size_mb - old_size_mb
    metrics_table.add_row(
        "Model Size",
        f"{old_size_mb:.3f} MB",
        f"{new_size_mb:.3f} MB",
        f"[red]+{size_change:.3f} MB[/]"
        if size_change > 0
        else f"[green]{size_change:.3f} MB[/]",
    )

    console.print(metrics_table)
    print()

    # Scores table
    scores_table = Table(
        title="Competition Scores", show_header=True, header_style="bold magenta"
    )
    scores_table.add_column("Component", style="cyan")
    scores_table.add_column("Formula", style="dim")
    scores_table.add_column("Old", style="yellow", justify="right")
    scores_table.add_column("New", style="green", justify="right")
    scores_table.add_column("Δ", style="bold", justify="right")

    scores_table.add_row(
        "Accuracy (50pt)",
        "acc × 50",
        f"{old_acc_score:.2f}",
        f"{new_acc_score:.2f}",
        f"{new_acc_score - old_acc_score:+.2f}",
    )
    scores_table.add_row(
        "Lag (25pt)",
        "exp(-6×ms/500)×25",
        f"{old_lag_score:.2f}",
        f"{new_lag_score:.2f}",
        f"{new_lag_score - old_lag_score:+.2f}",
    )
    scores_table.add_row(
        "Size (25pt)",
        "exp(-4×MB/5)×25",
        f"{old_size_score:.2f}",
        f"{new_size_score:.2f}",
        f"{new_size_score - old_size_score:+.2f}",
    )
    scores_table.add_row(
        "[bold]TOTAL[/]",
        "",
        f"[bold]{old_total:.2f}[/]",
        f"[bold]{new_total:.2f}[/]",
        f"[bold]{new_total - old_total:+.2f}[/]",
    )

    console.print(scores_table)
    print()

    # Summary
    total_change = new_total - old_total
    if total_change > 0:
        rprint(
            f"[bold green]New model wins![/] +{total_change:.2f} points ({old_total:.2f} → {new_total:.2f})"
        )
    else:
        rprint(
            f"[bold yellow]Old model wins.[/] {total_change:.2f} points ({old_total:.2f} → {new_total:.2f})"
        )

    print()
    rprint("[dim]Note: Latency measured on CPU for cross-platform comparison.[/]")
    print()


if __name__ == "__main__":
    main()

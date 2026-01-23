#!/usr/bin/env python3
"""
Compare EEGNet models with different window sizes.

Evaluates the old (window=128) vs new (window=1600) EEGNet models.

Usage:
    python examples/compare_eegnet_windows.py
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
from brainstorm.ml.channel_projection import PCAProjection


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

    # Warmup - fill the buffer
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

    return model, config, checkpoint


def main():
    console = Console()

    # Paths - old (window=128) and new (window=1600)
    old_checkpoint = Path("/media/M2SSD/mind_meld_checkpoints/hypersearch_v2/baseline/best_model.pt")
    new_checkpoint = Path("/media/M2SSD/mind_meld_checkpoints/eegnet_best.pt")
    data_path = Path("./data")

    # Check files exist
    if not old_checkpoint.exists():
        rprint(f"[red]Old EEGNet checkpoint not found: {old_checkpoint}[/]")
        return
    if not new_checkpoint.exists():
        rprint(f"[red]New EEGNet checkpoint not found: {new_checkpoint}[/]")
        return

    rprint("\n[bold cyan]EEGNet Comparison: Window Size 128 vs 1600[/]\n")

    # Get file sizes
    old_size_mb = old_checkpoint.stat().st_size / (1024 * 1024)
    new_size_mb = new_checkpoint.stat().st_size / (1024 * 1024)

    rprint(f"[cyan]Old checkpoint (window=128):[/] {old_checkpoint}")
    rprint(f"[cyan]New checkpoint (window=1600):[/] {new_checkpoint}")

    # Load models
    rprint(f"\n[cyan]Loading models...[/]")
    old_model, old_cfg, old_ckpt = load_eegnet_from_checkpoint(old_checkpoint)
    new_model, new_cfg, new_ckpt = load_eegnet_from_checkpoint(new_checkpoint)

    old_val_acc = old_ckpt.get("val_bal_acc", 0.0)
    new_val_acc = new_ckpt.get("val_bal_acc", 0.0)

    # Load sample for latency test
    rprint("[cyan]Loading validation data for latency test...[/]")
    val_features, _ = load_raw_data(data_path, step="validation")
    sample = val_features.values[0]

    # Measure latency
    rprint("[cyan]Measuring inference latency...[/]")
    old_latency = measure_latency(old_model, sample)
    new_latency = measure_latency(new_model, sample)

    # Calculate scores
    old_acc_score = accuracy_score(old_val_acc)
    old_lag_score = lag_score(old_latency)
    old_size_score = size_score(old_size_mb)
    old_total = old_acc_score + old_lag_score + old_size_score

    new_acc_score = accuracy_score(new_val_acc)
    new_lag_score = lag_score(new_latency)
    new_size_score = size_score(new_size_mb)
    new_total = new_acc_score + new_lag_score + new_size_score

    # Display results
    print()

    # Configuration table
    config_table = Table(title="Model Configurations", show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Old (window=128)", style="yellow", justify="right")
    config_table.add_column("New (window=1600)", style="green", justify="right")

    config_table.add_row("Window Size", str(old_cfg["window_size"]), str(new_cfg["window_size"]))
    config_table.add_row("PCA Channels", str(old_cfg["projected_channels"]), str(new_cfg["projected_channels"]))
    config_table.add_row("F1 (temporal filters)", str(old_cfg["F1"]), str(new_cfg["F1"]))
    config_table.add_row("D (depthwise mult)", str(old_cfg["D"]), str(new_cfg["D"]))
    config_table.add_row("Dropout", str(old_cfg["dropout"]), str(new_cfg["dropout"]))
    config_table.add_row("Best Epoch", str(old_ckpt.get("epoch", "?")), str(new_ckpt.get("epoch", "?")))

    console.print(config_table)
    print()

    # Raw metrics table
    metrics_table = Table(title="Raw Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Old (window=128)", style="yellow", justify="right")
    metrics_table.add_column("New (window=1600)", style="green", justify="right")
    metrics_table.add_column("Change", style="bold", justify="center")

    # Balanced accuracy (higher is better)
    acc_change = new_val_acc - old_val_acc
    acc_change_str = f"+{acc_change:.4f}" if acc_change > 0 else f"{acc_change:.4f}"
    metrics_table.add_row(
        "Balanced Accuracy",
        f"{old_val_acc:.4f}",
        f"{new_val_acc:.4f}",
        f"[green]{acc_change_str}[/]" if acc_change > 0 else f"[red]{acc_change_str}[/]"
    )

    # Latency (lower is better)
    lat_change = new_latency - old_latency
    lat_change_str = f"+{lat_change:.2f} ms" if lat_change > 0 else f"{lat_change:.2f} ms"
    metrics_table.add_row(
        "Inference Latency",
        f"{old_latency:.2f} ms",
        f"{new_latency:.2f} ms",
        f"[red]{lat_change_str}[/]" if lat_change > 0 else f"[green]{lat_change_str}[/]"
    )

    # Size (lower is better)
    size_change = new_size_mb - old_size_mb
    size_change_str = f"+{size_change:.3f} MB" if size_change > 0 else f"{size_change:.3f} MB"
    metrics_table.add_row(
        "Model Size",
        f"{old_size_mb:.3f} MB",
        f"{new_size_mb:.3f} MB",
        f"[red]{size_change_str}[/]" if size_change > 0 else f"[green]{size_change_str}[/]"
    )

    console.print(metrics_table)
    print()

    # Scores table
    scores_table = Table(title="Competition Scores", show_header=True, header_style="bold magenta")
    scores_table.add_column("Component", style="cyan")
    scores_table.add_column("Weight", style="white", justify="center")
    scores_table.add_column("Old (window=128)", style="yellow", justify="right")
    scores_table.add_column("New (window=1600)", style="green", justify="right")

    scores_table.add_row(
        "Accuracy Score",
        "50%",
        f"{old_acc_score:.2f}",
        f"{new_acc_score:.2f}"
    )
    scores_table.add_row(
        "Lag Score",
        "25%",
        f"{old_lag_score:.2f}",
        f"{new_lag_score:.2f}"
    )
    scores_table.add_row(
        "Size Score",
        "25%",
        f"{old_size_score:.2f}",
        f"{new_size_score:.2f}"
    )
    scores_table.add_row(
        "[bold]TOTAL SCORE[/]",
        "[bold]100%[/]",
        f"[bold]{old_total:.2f}[/]",
        f"[bold]{new_total:.2f}[/]"
    )

    console.print(scores_table)
    print()

    # Summary
    total_change = new_total - old_total
    if total_change > 0:
        rprint(f"[bold green]New model is better![/] Score improved by {total_change:.2f} points")
        rprint(f"  Old: {old_total:.2f} → New: {new_total:.2f}")
    else:
        rprint(f"[bold yellow]Old model is better.[/] Score decreased by {abs(total_change):.2f} points")
        rprint(f"  Old: {old_total:.2f} → New: {new_total:.2f}")

    print()


if __name__ == "__main__":
    main()

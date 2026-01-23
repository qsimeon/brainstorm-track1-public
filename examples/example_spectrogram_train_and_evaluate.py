#!/usr/bin/env python3
"""
Example script: Train and Evaluate with Spectrogram Dataset

This script demonstrates the complete workflow for:
    1. Computing spectrograms from raw ECoG data
    2. Creating streaming-ready PyTorch dataset
    3. Training a simple neural decoder
    4. Evaluating on validation set

The spectrogram dataset uses a "growing sequence" design where each sample
contains all channels with growing history (seq_len: 1 → max_seq_len),
training the model to handle variable context lengths essential for
streaming inference.

Usage:
    python examples/example_spectrogram_train_and_evaluate.py

Output:
    - Trained model saved to model.pt
    - Metadata saved to model_metadata.json
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from loguru import logger

from brainstorm.loading import load_raw_data, load_channel_coordinates
from brainstorm.transforms import compute_spectrograms
from brainstorm.datasets import create_dataloader


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")
SPECTROGRAMS_DIR = Path("./data/spectrograms")

EPOCHS = 1
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 500
DEMO_BATCHES = 1000  # Demo with 1000 batches for speed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model
# =============================================================================

class SimpleSpectrogramDecoder(nn.Module):
    """Minimal decoder for spectrogram-based classification.

    Input: (1024 channels, seq_len timesteps, 26 freq bins)
    Output: (n_classes,)
    """

    def __init__(self, n_freq_bins: int = 26, n_channels: int = 1024, n_classes: int = 10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(n_freq_bins, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor, spatial_encoding: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (1024, seq_len, 26) spectrogram
            spatial_encoding: (1024, 26) optional positional encoding

        Returns:
            logits: (n_classes,)
        """
        # Pool over channels: (1024, seq_len, 26) → (seq_len, 26)
        x_pooled = x.mean(dim=0)

        # Temporal processing: apply MLP to each timestep
        x_temporal = self.temporal_mlp(x_pooled)  # (seq_len, 32)

        # Pool over time: (seq_len, 32) → (32,)
        x_final = x_temporal.mean(dim=0)

        # Classify: (32,) → (n_classes,)
        logits = self.classifier(x_final)

        return logits


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        specs = batch["spectrogram"].to(device)  # (1024, seq_len, 26)
        labels = batch["label"].to(device)  # scalar
        spatial_enc = batch["spatial_encoding"]
        if spatial_enc is not None:
            spatial_enc = spatial_enc.to(device)

        # Forward
        logits = model(specs, spatial_enc)  # (n_classes,)
        loss = criterion(logits.unsqueeze(0), labels.unsqueeze(0))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    """Evaluate model and return loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        specs = batch["spectrogram"].to(device)  # (1024, seq_len, 26)
        labels = batch["label"].to(device)  # scalar
        spatial_enc = batch["spatial_encoding"]
        if spatial_enc is not None:
            spatial_enc = spatial_enc.to(device)

        # Forward
        logits = model(specs, spatial_enc)  # (n_classes,)
        loss = criterion(logits.unsqueeze(0), labels.unsqueeze(0))

        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).item()
        total += 1
        num_batches += 1

    return total_loss / num_batches, 100 * correct / total


def main() -> None:
    rprint("\n[bold cyan]Spectrogram-based Neural Decoder[/]\n")

    # =========================================================================
    # Load Data
    # =========================================================================
    rprint("[bold green]Loading raw ECoG data...[/]")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")

    # Display dataset info
    console = Console()
    table = Table(title="Dataset Overview", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", width=10)
    table.add_column("Samples", style="green")
    table.add_column("Duration (s)", style="yellow")
    table.add_column("Channels", style="blue")

    train_dur = train_features.index[-1] - train_features.index[0]
    val_dur = val_features.index[-1] - val_features.index[0]

    table.add_row("Train", str(len(train_features)), f"{train_dur:.1f}", "1024")
    table.add_row("Validation", str(len(val_features)), f"{val_dur:.1f}", "1024")
    console.print(table)

    # =========================================================================
    # Compute Spectrograms
    # =========================================================================
    rprint("\n[bold green]Computing spectrograms (50ms window, 1ms step)...[/]")

    train_specs = compute_spectrograms(
        train_features.values,
        fs=1000,
        window_ms=50,
        step_ms=1,
        freq_range=(0, 500),
        n_jobs=-1,
    )

    val_specs = compute_spectrograms(
        val_features.values,
        fs=1000,
        window_ms=50,
        step_ms=1,
        freq_range=(0, 500),
        n_jobs=-1,
    )

    # Convert to tensors and map labels to class indices
    train_specs = torch.from_numpy(train_specs).float()
    val_specs = torch.from_numpy(val_specs).float()

    # Map frequency labels to class indices (0-9)
    unique_classes = sorted(train_labels["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    train_labels_tensor = torch.from_numpy(
        train_labels["label"].map(class_to_idx).values
    ).long()
    val_labels_tensor = torch.from_numpy(
        val_labels["label"].map(class_to_idx).values
    ).long()

    rprint(f"[green]✓[/] Train spectrograms: {train_specs.shape}")
    rprint(f"[green]✓[/] Validation spectrograms: {val_specs.shape}")

    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    rprint("\n[bold green]Creating DataLoaders (growing sequence dataset)...[/]")

    try:
        channel_coords = load_channel_coordinates()
    except Exception:
        channel_coords = None

    train_loader = create_dataloader(
        spectrograms=train_specs,
        labels=train_labels_tensor,
        max_seq_len=MAX_SEQ_LEN,
        channel_coords=channel_coords,
        shuffle=False,
    )

    val_loader = create_dataloader(
        spectrograms=val_specs,
        labels=val_labels_tensor,
        max_seq_len=MAX_SEQ_LEN,
        channel_coords=channel_coords,
        shuffle=False,
    )

    rprint(f"[green]✓[/] DataLoader created: {len(train_loader)} timesteps")

    # =========================================================================
    # Train Model
    # =========================================================================
    rprint("\n[bold green]Training model...[/]\n")

    model = SimpleSpectrogramDecoder(
        n_freq_bins=train_specs.shape[2],
        n_classes=len(train_labels_tensor.unique()),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, max_batches=DEMO_BATCHES)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, max_batches=100)

        rprint(
            f"[cyan]Epoch {epoch+1}/{EPOCHS}[/] "
            f"[yellow]loss: {train_loss:.4f}[/] "
            f"[blue]val_loss: {val_loss:.4f}[/] "
            f"[green]val_acc: {val_acc:.1f}%[/]"
        )

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    rprint("\n[bold green]Final Evaluation[/]\n")

    test_loss, test_acc = evaluate(model, val_loader, criterion, DEVICE)

    table = Table(title="Final Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Validation Loss", f"{test_loss:.4f}")
    table.add_row("Validation Accuracy", f"{test_acc:.2f}%")
    table.add_row("Model Device", str(DEVICE))
    table.add_row("Input Shape", f"(1024, seq_len, {train_specs.shape[2]})")
    table.add_row("Output Classes", str(len(train_labels_tensor.unique())))

    console.print(table)

    # Save model
    rprint("\n[bold green]Saving model...[/]")
    torch.save(model.state_dict(), Path("model.pt"))
    rprint("[green]✓ Model saved to model.pt[/]\n")


if __name__ == "__main__":
    main()

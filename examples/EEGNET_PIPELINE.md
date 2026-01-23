# EEGNet Pipeline

This document explains the full pipeline for the EEGNet model used for ECoG signal classification.

## Overview

EEGNet is a compact convolutional neural network designed for EEG-based brain-computer interfaces. We've adapted it for high-density ECoG recordings (1024 channels) with PCA-based channel reduction.

## Pipeline Stages

### 1. Input Data
- **Raw ECoG signals**: `(n_samples, 1024)` - 1024 electrode channels at 1000Hz
- **Labels**: 9 frequency classes (0, 120, 224, 421, 789, 1479, 2772, 5195, 9736)

### 2. PCA Channel Projection

```
(n_samples, 1024) → (n_samples, 64)
```

- Reduces 1024 channels to 64 principal components
- Fitted on training data only (no data leakage)
- Captures ~99.7% of variance
- Converted to a PyTorch linear layer for fast inference

**Implementation**: `brainstorm/ml/channel_projection.py:117-205`

### 3. Sliding Window

```
(n_samples, 64) → (n_windows, 128, 64)
```

- Creates overlapping windows of 128 timesteps (128ms at 1000Hz)
- Each window's label = label of the last sample in that window
- During inference, maintains a rolling buffer that shifts by 1 sample per prediction

**Implementation**: `brainstorm/ml/eegnet.py:229-246`

### 4. Reshape for Conv2D

```
(batch, 128, 64) → (batch, 1, 64, 128)
      ↑                  ↑   ↑    ↑
 time, channels         C   H    W
```

- Treats channels as "height" and time as "width"
- Single input channel (like a grayscale image)

### 5. EEGNet Architecture

```
Input: (batch, 1, 64 channels, 128 time)
                    │
                    ▼
┌─────────────────────────────────────┐
│ Block 1: Temporal Convolution       │
│   Conv2d(1→8, kernel=(1,64))        │  Learns 8 temporal filters
│   BatchNorm2d                       │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Block 2: Depthwise Spatial Conv     │
│   Conv2d(8→16, kernel=(64,1))       │  Spatial filter per temporal filter
│   groups=8 (depthwise)              │  D=2 multiplier → 8×2=16 filters
│   BatchNorm2d → ELU                 │
│   AvgPool2d(1,4) → Dropout(0.25)    │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Block 3: Separable Convolution      │
│   Depthwise: Conv2d(16→16, (1,16))  │  Temporal smoothing
│   Pointwise: Conv2d(16→16, (1,1))   │  Channel mixing
│   BatchNorm2d → ELU                 │
│   AvgPool2d(1,8) → Dropout(0.25)    │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Classifier                          │
│   Flatten → Linear(flat_size → 9)   │  9 frequency classes
└─────────────────────────────────────┘
                    │
                    ▼
Output: (batch, 9) logits
```

**Implementation**: `brainstorm/ml/eegnet.py:32-143`

### 6. Training Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Class weighting** | Inverse frequency weights to handle imbalance (class 0 is 67% of data) |
| **AdamW optimizer** | With weight decay for regularization |
| **Cosine annealing LR** | Learning rate scheduler for smooth convergence |
| **Gradient clipping** | Max norm of 1.0 for training stability |
| **Best checkpoint saving** | Saves model with highest validation balanced accuracy |

### 7. Inference

```python
# For each new sample:
sample = (1024,)                    # Raw ECoG reading
    │
    ▼
projected = pca_layer(sample)       # → (64,)
    │
    ▼
window = update_buffer(projected)   # → (128, 64) sliding window
    │
    ▼
tensor = reshape(window)            # → (1, 1, 64, 128)
    │
    ▼
logits = eegnet(tensor)             # → (1, 9)
    │
    ▼
prediction = argmax(logits)         # → class index
    │
    ▼
label = classes_[prediction]        # → actual label (0, 120, 224, ...)
```

**Implementation**: `brainstorm/ml/eegnet.py:374-404`

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| **Depthwise separable convolutions** | Much fewer parameters than standard convolutions |
| **Small model size** | ~10-50KB total, well under 25MB limit |
| **Fast inference** | <1ms per sample for real-time BCI (not sure about this one - will need to confirm!)|
| **128ms context window** | Balances temporal context vs latency |
| **PCA projection** | Efficient dimensionality reduction that preserves signal variance (this is actually kind of slow right now) |

## Usage

```bash
# Train EEGNet
python examples/train_eegnet.py

# With custom parameters
python examples/train_eegnet.py --epochs 50 --batch-size 128
```

## Reference

Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"

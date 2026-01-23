# Wav2Vec2 Pipeline

This document explains the full pipeline for the Wav2Vec2-based classifier used for ECoG signal classification.

## Overview

Wav2Vec2 is a self-supervised speech representation model from Facebook AI. We adapt the **tiny** variant (~1.2MB pretrained) for ECoG classification. While designed for audio waveforms, it can process any 1D signal and learns general temporal patterns that transfer well to neural signals.

## Model Size

| Component | Size |
|-----------|------|
| Pretrained encoder (wav2vec2_tiny_random) | ~1.2 MB |
| PCA projection | ~0.5 KB |
| Classification head | ~70 KB |
| **Total (frozen encoder)** | **~1.3 MB** |
| **Total (fine-tuned encoder)** | **~1.5 MB** |

Well under the 25MB limit, and in the optimal <5MB range for scoring.

## Pipeline Stages

### 1. Input Data
- **Raw ECoG signals**: `(n_samples, 1024)` - 1024 electrode channels at 1000Hz
- **Labels**: 9 frequency classes (0, 120, 224, 421, 789, 1479, 2772, 5195, 9736)

### 2. PCA Channel Projection

```
(n_samples, 1024) → (n_samples, 8)
```

- Reduces 1024 channels to 8 principal components (fewer than EEGNet's 64)
- Each channel is processed independently through the encoder
- Fewer channels = faster inference (8 encoder passes vs 64)

**Implementation**: `brainstorm/ml/channel_projection.py`

### 3. Sliding Window

```
(n_samples, 8) → (n_windows, 1600, 8)
```

- Creates windows of 1600 timesteps (1600ms at 1000Hz)
- Wav2Vec2 requires minimum ~1600 samples due to convolutional kernel sizes
- Each window's label = label of the last sample in that window

**Implementation**: `brainstorm/ml/wav2vec2_classifier.py:419-432`

### 4. Reshape for Encoder

```
(batch, 1600, 8) → (batch, 8, 1600)
     ↑                  ↑    ↑
time, channels      channels, time
```

- Wav2Vec2 expects `(batch, sequence_length)` per channel
- We process each channel independently through the encoder

### 5. Wav2Vec2 Architecture

```
Input: (batch, 8 channels, 1600 time)
                    │
                    ▼
┌─────────────────────────────────────┐
│ Reshape to process channels         │
│   (batch*8, 1600)                   │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Wav2Vec2 Feature Encoder            │
│   7 temporal conv layers            │
│   Extracts local features           │
│   Output: (batch*8, time', 32)      │  32 = hidden_dim for tiny model
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Wav2Vec2 Transformer Encoder        │
│   2 transformer layers              │
│   Self-attention over time          │
│   Output: (batch*8, time', 32)      │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Mean Pooling over Time              │
│   (batch*8, time', 32) → (batch*8, 32)
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Reshape back to batch               │
│   (batch, 8*32) = (batch, 256)      │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Classification Head                 │
│   Linear(256 → 128)                 │
│   LayerNorm → GELU → Dropout(0.1)   │
│   Linear(128 → 9)                   │
└─────────────────────────────────────┘
                    │
                    ▼
Output: (batch, 9) logits
```

**Implementation**: `brainstorm/ml/wav2vec2_classifier.py:172-207`

### 6. Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| **Encoder frozen** | Yes | Only trains classifier head (fast) |
| **Classifier LR** | 1e-3 | Higher LR for unfrozen head |
| **Encoder LR** | 1e-5 | Lower LR if fine-tuning encoder |
| **Batch size** | 32 | Smaller due to encoder memory |
| **Epochs** | 20 | Converges quickly with frozen encoder |
| **Window size** | 1600 | Minimum for Wav2Vec2 architecture |
| **PCA channels** | 8 | Trade-off: accuracy vs speed |

### 7. Training Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Class weighting** | Inverse frequency weights to handle imbalance |
| **AdamW optimizer** | With weight decay for regularization |
| **Cosine annealing LR** | Learning rate scheduler |
| **Gradient clipping** | Max norm of 1.0 for stability |
| **Best checkpoint saving** | Saves model with highest validation balanced accuracy |
| **Separate LR for encoder** | Lower LR for pretrained weights (if unfrozen) |

### 8. Inference

```python
# For each new sample:
sample = (1024,)                    # Raw ECoG reading
    │
    ▼
projected = pca_layer(sample)       # → (8,)
    │
    ▼
window = update_buffer(projected)   # → (1600, 8) sliding window
    │
    ▼
tensor = reshape(window)            # → (1, 8, 1600)
    │
    ▼
# Process each channel through encoder
features = wav2vec2(tensor)         # → (1, 8, 32) after pooling
    │
    ▼
flat = flatten(features)            # → (1, 256)
    │
    ▼
logits = classifier(flat)           # → (1, 9)
    │
    ▼
prediction = argmax(logits)         # → class index
    │
    ▼
label = classes_[prediction]        # → actual label (0, 120, 224, ...)
```

**Implementation**: `brainstorm/ml/wav2vec2_classifier.py:434-453`

## Comparison with EEGNet

| Aspect | EEGNet | Wav2Vec2 |
|--------|--------|----------|
| **Architecture** | Custom CNN | Pretrained Transformer |
| **Model size** | ~400 KB | ~1.3 MB |
| **Window size** | 128 samples | 1600 samples |
| **PCA channels** | 64 | 8 |
| **Training** | From scratch | Transfer learning |
| **Inference latency** | <1ms | ~0.5-2ms |
| **Parameters** | ~50K | ~1.2M (mostly frozen) |

## When to Use Wav2Vec2

**Pros:**
- Pretrained on massive audio data (temporal pattern knowledge)
- Fast convergence with frozen encoder
- Potentially better generalization

**Cons:**
- Larger window size (more latency)
- Larger model size
- Requires HuggingFace transformers library

## Usage

```bash
# Default (frozen encoder, fast training)
python examples/train_wav2vec2.py

# Fine-tune the encoder (slower, potentially better)
python examples/train_wav2vec2.py --unfreeze-encoder

# Custom settings
python examples/train_wav2vec2.py --epochs 30 --batch-size 16 --projected-channels 16
```

## Reference

Baevski et al. (2020) "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"

Model: `patrickvonplaten/wav2vec2_tiny_random` from HuggingFace

# Detailed Tensor Flow: Shapes Through the Network

A reference guide showing exact tensor shapes at each layer during training and inference.

---

## Training Phase: Detailed Tensor Shapes

### Stage 1: Raw Data Loading

```
┌───────────────────────────────────────────────────────────┐
│ Initial Data from Disk                                    │
├───────────────────────────────────────────────────────────┤
│                                                           │
│ features.parquet:                                         │
│   Shape: (10000, 1024)                                    │
│   Type: float32                                           │
│   Interpretation: 10000 timesteps × 1024 channels        │
│   Values: Raw voltages in microvolts (±100 µV range)     │
│                                                           │
│ labels.parquet:                                           │
│   Shape: (10000,)                                         │
│   Type: int64                                             │
│   Values: {0, 120, 224, 421, 789, 1479, 2772, 5195, 9736}│
│   Meaning: Stimulus frequency in Hz (0 = silence)        │
│                                                           │
└───────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│ Loaded into Memory as NumPy Arrays                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ X = np.array([...])  # Shape: (10000, 1024)                   │
│ y = np.array([...])  # Shape: (10000,)                         │
│                                                                 │
│ Example sample at t=500:                                       │
│   X[500] = [12.5, -8.3, 3.1, ..., 15.2]  # 1024 floats       │
│   y[500] = 224                             # Class label       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 2: Train/Validation Split

```
Split 80/20 randomly (with seed for reproducibility):

Before:
  X: (10000, 1024)
  y: (10000,)

After split:
  X_train: (8000, 1024)    ├─ Used for PCA fitting
  y_train: (8000,)         ├─ Used for training

  X_val:   (2000, 1024)    ├─ Used for validation
  y_val:   (2000,)         ├─ (not for optimization)

  X_test:  (held out, not loaded here)


Class distribution in y_train:
  class 0:   5360 samples (67.0%)
  class 120: 160 samples  (2.0%)
  class 224: 140 samples  (1.75%)
  ...
  class 9736: 155 samples (1.94%)

  Highly imbalanced! → Need class weighting
```

### Stage 3: PCA Fitting and Projection

```
┌──────────────────────────────────────────────────────────────────┐
│ FIT PCA on Training Data                                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Input:  X_train shape (8000, 1024)                              │
│         All raw ECoG data                                        │
│                                                                  │
│ Process:                                                         │
│   1. Center data: X_centered = X - mean(X)                      │
│      mean vector shape: (1024,)                                 │
│      X_centered shape: (8000, 1024)                             │
│                                                                  │
│   2. Compute covariance: Cov = (1/n) X_centered.T @ X_centered │
│      Cov shape: (1024, 1024)                                    │
│      Computation: O(1024² × 8000) = expensive!                  │
│                                                                  │
│   3. Eigendecomposition: Cov = U @ Λ @ U.T                      │
│      U: (1024, 1024) eigenvectors                               │
│      Λ: (1024,) eigenvalues (sorted descending)                │
│                                                                  │
│   4. Select top 64 components:                                  │
│      components_ = U[:, :64]  # shape (1024, 64)              │
│      Selects dimensions with largest variance                   │
│                                                                  │
│ Learned Parameters:                                             │
│   pca.mean_: (1024,)                                           │
│   pca.components_: (64, 1024)                                  │
│   Variance explained: 0.997 (99.7% of total)                   │
│                                                                  │
│ Create PyTorch layer for inference:                             │
│   pca_layer = nn.Linear(1024, 64)  # Learnable but frozen      │
│   pca_layer.weight = components_.T  # (64, 1024)              │
│   pca_layer.bias = -mean @ components_.T  # (64,)             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│ PROJECT Training Data via PCA                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Input:  X_train shape (8000, 1024)                              │
│         pca.components_ shape (64, 1024)                        │
│                                                                  │
│ Transform:                                                       │
│   X_train_proj = (X_train - pca.mean_) @ pca.components_.T     │
│                                                                  │
│ Dimensions:                                                      │
│   (8000, 1024) @ (1024, 64) = (8000, 64)                       │
│                                                                  │
│ Output: X_train_proj shape (8000, 64)                           │
│         Type: float32                                            │
│         Interpretation: Projected feature space                 │
│         Memory: 8000 × 64 × 4 bytes = 2.048 MB                 │
│                                                                  │
│ Effect:                                                          │
│   Original: each sample = [v₀, v₁, ..., v₁₀₂₃]  (1024 floats) │
│   Projected: each sample = [p₀, p₁, ..., p₆₃]    (64 floats)   │
│   16× dimension reduction!                                       │
│                                                                  │
│ Repeat for validation:                                           │
│   X_val_proj = (X_val - pca.mean_) @ pca.components_.T         │
│   X_val_proj shape: (2000, 64)                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Stage 4: Create Windowed Training Data

```
┌──────────────────────────────────────────────────────────────────┐
│ Convert Sequential Data to Windowed Samples                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Goal: Convert (8000, 64) into (n_windows, 128, 64)             │
│       where each window has temporal context                     │
│                                                                  │
│ Window size: 128 timesteps (128 ms at 1000 Hz)                 │
│                                                                  │
│ Algorithm:                                                       │
│   windows = []                                                   │
│   labels = []                                                    │
│                                                                  │
│   for t in range(128, 8000):                           [loop]   │
│     window = X_train_proj[t-128:t]  # 128 recent samples      │
│     windows.append(window)                                      │
│     labels.append(y_train[t-1])      # Label at window end    │
│                                                                  │
│   X_windows = np.stack(windows)  # shape (7872, 128, 64)      │
│   y_windows = np.array(labels)    # shape (7872,)              │
│                                                                  │
│ First window (t=128):                                            │
│   X_windows[0] = X_train_proj[0:128]   # timesteps 0-127      │
│   y_windows[0] = y_train[127]           # label at t=127       │
│                                                                  │
│ Second window (t=129):                                           │
│   X_windows[1] = X_train_proj[1:129]   # timesteps 1-128      │
│   y_windows[1] = y_train[128]           # label at t=128       │
│                                                                  │
│ Pattern: 1-sample shift (overlap of 127 samples!)              │
│   This creates ~7,872 training windows                          │
│   (8000 - 128 = 7,872)                                          │
│                                                                  │
│ Result:                                                          │
│   X_windows: (7872, 128, 64)                                    │
│              128 = temporal context window                      │
│              64 = PCA-projected channels                        │
│   y_windows: (7872,)                                            │
│              9 unique classes in this subset                    │
│                                                                  │
│ Memory footprint:                                                │
│   7872 × 128 × 64 × 4 bytes = 256 MB (on disk)                │
│   (loaded in batches during training)                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


VISUALIZATION OF WINDOWING:

Original sequence X_train_proj:
  t:    0     1     2   ...   127    128    129  ...  7999
       [──────────────────────────────────────────────────────]

Window 0 (t=128):
       ├──────────────────────────────────────────┤
       t=0                                      t=127 (label: y[127])

Window 1 (t=129):
             ├──────────────────────────────────────────┤
             t=1                                      t=128 (label: y[128])

Window 2 (t=130):
                   ├──────────────────────────────────────────┤
                   t=2                                      t=129 (label: y[129])

Each window: 128 consecutive timesteps
Each shifted: by exactly 1 timestep
Result: 7872 heavily overlapping training samples
```

### Stage 5: Convert to Tensors and Reshape

```
┌──────────────────────────────────────────────────────────────────┐
│ Convert NumPy Arrays to PyTorch Tensors                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ X_windows: (7872, 128, 64)  numpy float32                      │
│         ↓ torch.tensor()                                         │
│ X_tensor: (7872, 128, 64)  torch.float32                       │
│                                                                  │
│ y_windows: (7872,)  numpy int64 (class indices 0-8)            │
│         ↓ torch.tensor()                                         │
│ y_tensor: (7872,)  torch.long                                  │
│                                                                  │
│ Reshape for Conv2d (expects 4D: B×C×H×W):                       │
│                                                                  │
│   X_tensor: (7872, 128, 64)                                     │
│         ↓ permute(0, 2, 1)                                       │
│   (7872, 64, 128)                                               │
│          ↑    ↑                                                  │
│       channels time                                              │
│         ↓ unsqueeze(1)                                           │
│   (7872, 1, 64, 128)                                            │
│    ↑     ↑  ↑  ↑                                                 │
│   batch channel height width                                    │
│                                                                  │
│ Now compatible with Conv2d!                                      │
│ Interpretation: 7872 grayscale "images"                        │
│   - 1 channel (like grayscale)                                  │
│   - 64 height (64 electrode channels)                           │
│   - 128 width (128 timesteps)                                   │
│                                                                  │
│ y_tensor remains: (7872,) class indices                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


SHAPE TRANSFORMATION SUMMARY:

  (7872, 128, 64)          Original: [batch, time, channels]
         │
         permute(0, 2, 1)  Rearrange dimensions
         │
  (7872, 64, 128)          → [batch, channels, time]
         │
         unsqueeze(1)      Add input channel dimension
         │
  (7872, 1, 64, 128)       → [batch, input_channels, height, width]
         │
      For Conv2d!
      - Input channels: 1 (like grayscale image)
      - Height: 64 (spatial: electrode positions)
      - Width: 128 (temporal: time samples)
```

### Stage 6: Training Loop - Batch Processing

```
┌──────────────────────────────────────────────────────────────────┐
│ Create DataLoader for Batching                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ TensorDataset combines X and y:                                 │
│   dataset = TensorDataset(X_tensor, y_tensor)                   │
│   len(dataset) = 7872                                            │
│                                                                  │
│ DataLoader:                                                      │
│   loader = DataLoader(dataset, batch_size=64, shuffle=True)     │
│   len(loader) = 7872 / 64 = 123 batches per epoch              │
│                                                                  │
│ Batch iteration example:                                         │
│   for X_batch, y_batch in loader:                               │
│                                                                  │
│     X_batch shape: (64, 1, 64, 128)                             │
│     y_batch shape: (64,)                                         │
│                                                                  │
│     Interpretation:                                              │
│       64 training examples (batch)                               │
│       1 input channel (grayscale)                                │
│       64 spatial dimensions (electrodes)                         │
│       128 temporal points (time)                                 │
│                                                                  │
│     y_batch values: [0, 5, 2, 0, 0, 8, 1, 0, ...]              │
│     (class indices, not labels; 0→class0, 5→class5 Hz, etc.)    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


EPOCH ITERATION:

Epoch 1:
  Shuffle dataset (different order each epoch)
  Batch 1: X (64, 1, 64, 128), y (64,)
  Batch 2: X (64, 1, 64, 128), y (64,)
  ...
  Batch 123: X (16, 1, 64, 128), y (16,)  [last batch, fewer samples]

  Total samples trained: 123 × 64 - (64-16) = 7872 ✓

Epoch 2:
  Shuffle again (new order)
  Batch 1: X (64, 1, 64, 128), y (64,)
  ...
```

### Stage 7: Through EEGNetCore

```
┌────────────────────────────────────────────────────────────────────────────┐
│ EEGNET FORWARD PASS - Detailed Tensor Shapes                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│ Input: X_batch shape (64, 1, 64, 128)                                     │
│        y_batch shape (64,)                                                │
│                                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│ BLOCK 1: Temporal Convolution                                             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│                                                                            │
│   Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), ...)        │
│                                                                            │
│   Operation:                                                               │
│   ┌─ Slide 1×64 kernel across spatial×temporal dimensions                │
│   │  (spatial=64, temporal=128)                                           │
│   │                                                                        │
│   │  Kernels per output channel: 1 per 64 channels                       │
│   │  Total: 8 kernels, each size 1×64 = 8 temporal filters              │
│   └─ Learn 8 different "temporal pattern detectors"                       │
│                                                                            │
│   Shape after Conv2d:                                                     │
│   (64, 8, 64, 128)                                                        │
│    ↑  ↑  ↑   ↑                                                            │
│    batch channels height time                                             │
│                                                                            │
│   Interpretation:                                                         │
│   - 64 samples (batch)                                                    │
│   - 8 temporal filters applied                                            │
│   - 64 spatial dimensions unchanged (1×1 kernel on spatial)              │
│   - 128 temporal preserved (padding=32 on each side, same size)          │
│                                                                            │
│   BatchNorm2d(8):                                                         │
│   (64, 8, 64, 128) → (64, 8, 64, 128)                                     │
│   [normalizes each of 8 channels independently]                           │
│                                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│ BLOCK 2: Depthwise Spatial Convolution + Pooling                          │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│                                                                            │
│   Input: (64, 8, 64, 128)                                                 │
│                                                                            │
│   Depthwise Conv2d:                                                       │
│   Conv2d(in=8, out=16, kernel=(64,1), groups=8)                          │
│   ┌─ groups=8: one 64×1 kernel per input channel                         │
│   │  Channel 0: Conv2d(kernel=64×1) → output channel 0,1                 │
│   │  Channel 1: Conv2d(kernel=64×1) → output channel 2,3                 │
│   │  ...                                                                   │
│   │  Channel 7: Conv2d(kernel=64×1) → output channel 14,15                │
│   │                                                                        │
│   │  depth_multiplier D=2: each input → 2 outputs                         │
│   │  8 inputs × 2 = 16 output channels                                    │
│   │                                                                        │
│   └─ Goal: Learn spatial filtering (which electrode regions matter)      │
│                                                                            │
│   Shape after Depthwise:                                                  │
│   (64, 16, 1, 128)                                                        │
│    ↑  ↑  ↑  ↑                                                             │
│    batch channels height time                                             │
│    [spatial dimension collapsed: 64 → 1 via kernel height]               │
│                                                                            │
│   BatchNorm2d(16) → (64, 16, 1, 128)                                      │
│   ELU activation → (64, 16, 1, 128)                                       │
│   [nonlinearity: max(0, x) + alpha×(exp(x)-1))                          │
│                                                                            │
│   AvgPool2d((1, 4)):                                                      │
│   (64, 16, 1, 128) → (64, 16, 1, 32)                                      │
│   [temporal pooling: 128 → 32, downsampling by 4×]                       │
│   Each of 4 consecutive temporal values → 1 average value                 │
│                                                                            │
│   Dropout(0.25):                                                          │
│   (64, 16, 1, 32) → (64, 16, 1, 32)                                       │
│   [25% of activations randomly zeroed during training]                    │
│                                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│ BLOCK 3: Separable Temporal Convolution + Pooling                         │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│                                                                            │
│   Input: (64, 16, 1, 32)                                                  │
│                                                                            │
│   Depthwise Conv2d(16, 16, kernel=(1,16), groups=16):                    │
│   ┌─ One 1×16 temporal kernel per channel                                 │
│   │  Temporal smoothing: combine adjacent time points                     │
│   └─ Output shape: (64, 16, 1, 32)                                        │
│                                                                            │
│   Pointwise Conv2d(16, 16, kernel=(1,1)):                                │
│   ┌─ Mix information across 16 channels                                   │
│   │  Create optimal combination of filters                                │
│   └─ Output shape: (64, 16, 1, 32)                                        │
│                                                                            │
│   BatchNorm2d(16) → (64, 16, 1, 32)                                       │
│   ELU activation → (64, 16, 1, 32)                                        │
│                                                                            │
│   AvgPool2d((1, 8)):                                                      │
│   (64, 16, 1, 32) → (64, 16, 1, 4)                                        │
│   [temporal pooling: 32 → 4, downsampling by 8×]                         │
│                                                                            │
│   Dropout(0.25):                                                          │
│   (64, 16, 1, 4) → (64, 16, 1, 4)                                         │
│                                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│ CLASSIFIER                                                                │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│                                                                            │
│   Flatten:                                                                │
│   (64, 16, 1, 4) → (64, 64)                                               │
│   [64 samples × (16 channels × 1 × 4 values) = 64 samples × 64 values]   │
│                                                                            │
│   Linear(64, 9):                                                          │
│   (64, 64) @ (64, 9) = (64, 9)                                            │
│                                                                            │
│   Output: (64, 9)                                                         │
│    ↑  ↑                                                                    │
│    batch logits                                                           │
│                                                                            │
│   Interpretation:                                                         │
│   64 samples, each with 9 logit scores                                    │
│   logits[i, :] = [score_class0, score_class1, ..., score_class8]         │
│                                                                            │
│   Example logits for first sample:                                        │
│   logits[0] = [-2.1, 0.3, 1.5, -0.8, 2.2, 0.1, -1.0, 0.5, -0.2]         │
│   Highest: class 4 with score 2.2                                         │
│   → Prediction: class 4 (frequency 789 Hz)                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘


SUMMARY OF TENSOR SHAPES THROUGH EEGNET:

Layer                           Shape Transformation
─────────────────────────────────────────────────────
Input                           (64, 1, 64, 128)
Conv2d (F1=8 filters)           (64, 8, 64, 128)
BatchNorm → Depthwise Conv      (64, 16, 1, 128)
ELU → AvgPool4 → Dropout        (64, 16, 1, 32)
─────────────────────────────────────────────────────
Depthwise Conv(16) → Pointwise  (64, 16, 1, 32)
Conv (F2=16)
ELU → AvgPool8 → Dropout        (64, 16, 1, 4)
─────────────────────────────────────────────────────
Flatten                         (64, 64)
Linear(64, 9)                   (64, 9)  ← Final logits
─────────────────────────────────────────────────────

Total parameters: ~50K-100K
```

### Stage 8: Loss Computation and Backprop

```
┌──────────────────────────────────────────────────────────────────┐
│ Loss Computation (Per Batch)                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Logits from network: (64, 9)                                     │
│ Labels: (64,)  values in {0, 1, 2, ..., 8}  (class indices)     │
│                                                                  │
│ Class weights (computed from training distribution):            │
│   weight[0] = 1 / count[0] = 1 / 5360 = 0.000187              │
│   weight[1] = 1 / count[1] = 1 / 160 = 0.00625                │
│   weight[2] = 1 / count[2] = 1 / 140 = 0.00714                │
│   ...                                                            │
│   weight[8] = 1 / count[8] = 1 / 155 = 0.00645                │
│                                                                  │
│   These are normalized: sum(weights) ≈ 1                        │
│                                                                  │
│ CrossEntropyLoss with weight:                                   │
│                                                                  │
│   for i in range(64):  # For each sample in batch              │
│     logit = logits[i]         # shape (9,)                      │
│     label = labels[i]          # shape ()  scalar               │
│     w = weight[label]          # shape ()  scalar               │
│                                                                  │
│     # Cross entropy: -log(softmax(logit)[label])               │
│     softmax = exp(logit) / sum(exp(logit))                      │
│     ce_loss = -log(softmax[label])                              │
│                                                                  │
│     # Weighted by class weight                                  │
│     weighted_loss = w * ce_loss                                 │
│     loss[i] = weighted_loss                                     │
│                                                                  │
│   batch_loss = mean(loss)  # Average over 64 samples           │
│                                                                  │
│ Example for sample 0:                                            │
│   logits[0] = [-2.1, 0.3, 1.5, -0.8, 2.2, 0.1, -1.0, 0.5, -0.2]
│   label[0] = 5  (class 5)                                       │
│   weight[5] = 0.00625                                           │
│                                                                  │
│   softmax[0] = exp(-2.1) / sum(...)  ≈ 0.0001                 │
│   ...                                                            │
│   softmax[5] = exp(2.2) / sum(...)  ≈ 0.85                     │
│   ce_loss = -log(0.85) ≈ 0.163                                 │
│   weighted_loss = 0.00625 × 0.163 = 0.00102                   │
│                                                                  │
│ Example for sample 1 (wrong class):                             │
│   logits[1] = [0.5, 0.2, 0.1, 0.4, -1.0, 0.3, 0.2, 0.6, 0.1]  │
│   label[1] = 4  (minority class)                                │
│   weight[4] = 1 / count[4] = 0.00625 (but actually higher)     │
│                                                                  │
│   softmax[4] = exp(-1.0) / sum(...)  ≈ 0.02  (wrong! low)     │
│   ce_loss = -log(0.02) ≈ 3.9  (high error!)                    │
│   weighted_loss = 0.00625 × 3.9 = 0.024                        │
│                                                                  │
│   This sample contributes 24× more to gradients than correctly  │
│   classified majority class samples! ← Forces learning          │
│                                                                  │
│ Final batch loss: mean(weighted_loss for all 64)                │
│ Example: batch_loss ≈ 0.5                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


BACKWARD PASS (Automatic Differentiation):

  batch_loss.backward()

  PyTorch autograd computes:
    ∂loss/∂logits (64, 9)
    ∂logits/∂hidden (propagates through network)
    ...
    ∂hidden/∂conv_weights
    ∂hidden/∂bias

  All gradients stored in .grad attributes of tensors

  Example gradient magnitudes:
    eegnet.fc.weight.grad      shape (9, 64)    magnitude ≈ 0.01-0.1
    eegnet.fc.bias.grad        shape (9,)       magnitude ≈ 0.001-0.01
    eegnet.separable2.weight.grad  shape (16, 16, 1, 1)
    ...
    eegnet.conv1.weight.grad   shape (8, 1, 1, 64)


GRADIENT CLIPPING:

  torch.nn.utils.clip_grad_norm_(eegnet.parameters(), max_norm=1.0)

  Computes total gradient norm:
    g_total_norm = sqrt(sum(g.grad^2 for all parameters))

  If g_total_norm > 1.0:
    for param in eegnet.parameters():
      param.grad *= (1.0 / g_total_norm)

  Effect: scale down all gradients uniformly if too large
```

### Stage 9: Optimizer Step

```
┌──────────────────────────────────────────────────────────────────┐
│ Parameter Update (AdamW Optimizer)                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Before update:                                                   │
│   param = original_value                                         │
│   param.grad = ∂loss/∂param                                      │
│                                                                  │
│ AdamW maintains two moments (per parameter):                     │
│   m = exponential moving average of gradients (first moment)    │
│   v = exponential moving average of squared gradients (second)  │
│                                                                  │
│   β₁ = 0.9 (decay for m)                                        │
│   β₂ = 0.999 (decay for v)                                      │
│   lr = learning_rate = 0.001                                    │
│   wd = weight_decay = 0.0001                                    │
│                                                                  │
│ Update rule (simplified):                                        │
│   m = β₁ × m + (1 - β₁) × grad                                  │
│   v = β₂ × v + (1 - β₂) × grad²                                 │
│   m_hat = m / (1 - β₁^t)  [bias correction, t = step number]   │
│   v_hat = v / (1 - β₂^t)                                        │
│   param = param - lr × m_hat / (sqrt(v_hat) + eps)              │
│   param = param - wd × lr × param  [weight decay]               │
│                                                                  │
│ Example (single parameter):                                      │
│   Initial: weight = 0.5                                          │
│   Gradient: grad = 0.02                                          │
│   Learning rate: lr = 0.001                                      │
│                                                                  │
│   Moment update:                                                 │
│   m = 0.9 × 0 + 0.1 × 0.02 = 0.002                              │
│   v = 0.999 × 0 + 0.001 × 0.0004 = 0.0000004                    │
│                                                                  │
│   Bias-corrected moments (step 1):                               │
│   m_hat = 0.002 / (1 - 0.9^1) = 0.002 / 0.1 = 0.02             │
│   v_hat = 0.0000004 / (1 - 0.999^1) = 0.0000004 / 0.001 = 0.0004
│                                                                  │
│   Parameter update:                                              │
│   param -= 0.001 × 0.02 / sqrt(0.0004 + 1e-8)                   │
│   param -= 0.001 × 0.02 / 0.02 = 0.001                          │
│   param = 0.5 - 0.001 = 0.499                                   │
│                                                                  │
│   Weight decay:                                                  │
│   param -= 0.0001 × 0.001 × 0.499 ≈ negligible                  │
│                                                                  │
│   Final: weight = 0.499  (moved slightly in gradient direction)│
│                                                                  │
│ Effect across full batch:                                        │
│   All 50K parameters updated similarly                           │
│   Weighted by their individual gradients                         │
│   Learning rate automatically adapts per parameter (adaptive)   │
│                                                                  │
│ Learning rate schedule (cosine annealing):                       │
│   At epoch 1:  lr = 0.001 (full)                                │
│   At epoch 15: lr ≈ 0.0001 (0.1× original)                      │
│   At epoch 30: lr ≈ 0.00001 (0.01× original)                    │
│   Formula: lr(e) = 0.00001 + 0.0009 × cos(π × e / 30) / 2       │
│                                                                  │
│   Effect: rough optimization early, fine-tuning late           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Inference Phase: Detailed Tensor Shapes

### Single Sample Prediction

```
┌────────────────────────────────────────────────────────────────────────┐
│ STREAMING INFERENCE - One Sample at a Time                            │
└────────────────────────────────────────────────────────────────────────┘

INITIAL STATE (before any data):
  model._window_buffer = np.zeros((128, 64))  dtype=float32
  └─ 128 timesteps, 64 channels
  └─ All zeros initially


SAMPLE 0 ARRIVES (t=0, first millisecond):
┌──────────────────────────────────────────────────────────────────────┐
│ Input: raw_ecog_0                                                    │
│   Shape: (1024,)                                                     │
│   Values: [v₀, v₁, ..., v₁₀₂₃]                                       │
│   Example: [-12.5, 8.3, -3.1, ..., 15.2]                            │
│   Range: ±100 µV (microvolts)                                        │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: Convert to PyTorch Tensor                                    │
│                                                                      │
│   x = torch.tensor(raw_ecog_0, dtype=torch.float32)                  │
│   x.shape = (1024,)                                                  │
│                                                                      │
│   Add batch dimension:                                               │
│   x = x.unsqueeze(0)                                                 │
│   x.shape = (1, 1024)                                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: Apply PCA Projection                                         │
│                                                                      │
│   pca_layer = nn.Linear(1024, 64)                                    │
│   weight.shape = (64, 1024)                                          │
│   bias.shape = (64,)                                                 │
│                                                                      │
│   x_proj = pca_layer(x)                                              │
│   x_proj.shape = (1, 64)                                             │
│                                                                      │
│   Computation:                                                       │
│   x_proj = x @ weight.T + bias                                       │
│   (1, 1024) @ (1024, 64) = (1, 64)                                  │
│                                                                      │
│   Example output:                                                    │
│   x_proj = [0.34, -0.12, 1.05, ..., -0.67]  # 64 values            │
│                                                                      │
│   Convert back to numpy (remove batch):                              │
│   x_proj_np = x_proj.squeeze(0).numpy()                              │
│   x_proj_np.shape = (64,)                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 3: Update Sliding Window Buffer                                 │
│                                                                      │
│   Before (first call):                                               │
│   _window_buffer = [[0, 0, 0, ..., 0],                              │
│                     [0, 0, 0, ..., 0],                              │
│                     ...                                              │
│                     [0, 0, 0, ..., 0]]  shape (128, 64)             │
│                                                                      │
│   Shift operation:                                                   │
│   _window_buffer = np.roll(_window_buffer, -1, axis=0)              │
│                                                                      │
│   After roll (shift all rows up by 1):                               │
│   _window_buffer = [[0, 0, 0, ..., 0],        (row 1 moved to 0)    │
│                     [0, 0, 0, ..., 0],        (row 2 moved to 1)    │
│                     ...                                              │
│                     [0, 0, 0, ..., 0]]        (row 127 → 126)       │
│                                                                      │
│   Add new sample at end:                                             │
│   _window_buffer[-1] = x_proj_np                                     │
│                                                                      │
│   After update:                                                      │
│   _window_buffer = [[0, 0, 0, ..., 0],                              │
│                     [0, 0, 0, ..., 0],                              │
│                     ...                                              │
│                     [0.34, -0.12, 1.05, ..., -0.67]]  shape (128, 64)
│                                                                      │
│   Only the last row changed! (127 zeros, 1 real sample)             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 4: Reshape for EEGNet                                           │
│                                                                      │
│   window_tensor = torch.tensor(_window_buffer, dtype=torch.float32)  │
│   window_tensor.shape = (128, 64)                                    │
│                                                                      │
│   Transpose to (channels, time):                                     │
│   window_tensor = window_tensor.T                                    │
│   window_tensor.shape = (64, 128)                                    │
│                                                                      │
│   Add batch and channel dimensions:                                  │
│   window_tensor = window_tensor.unsqueeze(0).unsqueeze(0)            │
│   window_tensor.shape = (1, 1, 64, 128)                              │
│                                                                      │
│   Now in format: (batch, channels, height, width)                   │
│                 (1,     1,       64,     128)                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 5: Forward Through EEGNetCore                                   │
│                                                                      │
│   Input: (1, 1, 64, 128)                                             │
│                                                                      │
│   Block 1 - Temporal Conv:                                           │
│   (1, 1, 64, 128) → (1, 8, 64, 128)                                  │
│   [8 temporal filters applied]                                       │
│                                                                      │
│   Block 2 - Spatial Conv + Pool:                                     │
│   (1, 8, 64, 128) → (1, 16, 1, 32)                                   │
│   [spatial filtering, pooling by 4×]                                 │
│                                                                      │
│   Block 3 - Temporal Smooth + Pool:                                  │
│   (1, 16, 1, 32) → (1, 16, 1, 4)                                     │
│   [temporal smoothing, pooling by 8×]                                │
│                                                                      │
│   Classifier:                                                        │
│   (1, 16, 1, 4) → (1, 64) → (1, 9)                                   │
│   [flatten and dense layer]                                          │
│                                                                      │
│   Output logits: (1, 9)                                              │
│   Example: [[-0.5, 0.2, 1.8, -0.3, 2.1, 0.1, -0.8, 0.4, -0.2]]     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 6: Get Prediction                                               │
│                                                                      │
│   logits = (1, 9)                                                    │
│   class_idx = torch.argmax(logits, dim=1)                            │
│   class_idx = 4  [highest logit at index 4]                          │
│                                                                      │
│   classes_ = [0, 120, 224, 421, 789, 1479, 2772, 5195, 9736]        │
│   prediction = classes_[4] = 789 Hz                                  │
│                                                                      │
│   Return: 789  (as integer)                                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

BUFFER STATE AFTER SAMPLE 0:
  _window_buffer shape: (128, 64)
  Content: 127 rows of zeros, 1 row with real data
  Ready for sample 1


SAMPLE 1 ARRIVES (t=1, 1 ms later):
┌──────────────────────────────────────────────────────────────────────┐
│ Input: raw_ecog_1 shape (1024,)                                      │
│                                                                      │
│ Repeat steps 1-6:                                                    │
│   PCA: (1024,) → (64,)                                               │
│                                                                      │
│   Update buffer:                                                     │
│   Before: [[0, ..., 0],                                              │
│            [0, ..., 0],                                              │
│            ...                                                       │
│            [0.34, ..., -0.67]]  # 127 zeros, 1 sample              │
│                                                                      │
│   After roll: [[0, ..., 0],                                          │
│                [0, ..., 0],                                          │
│                ...                                                   │
│                [0, ..., 0],                                          │
│                [0.34, ..., -0.67]]  # shifted up                    │
│                                                                      │
│   After append: [[0, ..., 0],                                        │
│                  [0, ..., 0],                                        │
│                  ...                                                 │
│                  [0.34, ..., -0.67],      # was last row            │
│                  [0.21, ..., 0.45]]       # new sample 1            │
│                 (126 zeros now, 2 real samples)                     │
│                                                                      │
│   Forward through network, get prediction                            │
│                                                                      │
│ Return: class label for sample 1                                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

BUFFER STATE AFTER SAMPLE 1:
  _window_buffer shape: (128, 64)
  Content: 126 rows of zeros, 2 rows with real data
  Progressing towards full buffer


...

SAMPLE 127 ARRIVES:
  Buffer becomes fully populated with real data (first 127 samples)
  From sample 128 onwards: sliding window (oldest sample drops, new one added)


BUFFER STATE AT STEADY STATE (sample ≥ 128):
  _window_buffer = [[sample_i-127],
                    [sample_i-126],
                    ...
                    [sample_i-1],
                    [sample_i]]

  Most recent context always available
  (128 most recent timesteps at any point)
```

### Complete Streaming Example

```
┌─────────────────────────────────────────────────────────────────────┐
│ FULL STREAMING SEQUENCE                                             │
└─────────────────────────────────────────────────────────────────────┘

Time    Sample          Buffer State              Prediction
────────────────────────────────────────────────────────────────
t=0     Input: (1024)   [zeros] × 127           Class X
                        + [real] × 1
                        Total: 128 rows

t=1     Input: (1024)   [zeros] × 126           Class Y
                        + [real] × 2
                        Total: 128 rows

t=2     Input: (1024)   [zeros] × 125           Class Z
                        + [real] × 3
                        Total: 128 rows

...

t=127   Input: (1024)   [zeros] × 1             Class A
                        + [real] × 127
                        Total: 128 rows

t=128   Input: (1024)   [real] × 128            Class B
        (WARM BUFFER!)  (sliding window starts)  ← Full history!
                        Total: 128 rows

t=129   Input: (1024)   Drop oldest, add newest  Class C
                        [samples 1-128]          ← Full history!
                        Total: 128 rows

t=130   Input: (1024)   [samples 2-129]          Class D
                        Total: 128 rows

...

t=9999  Input: (1024)   [samples 9871-9999]      Class M
                        Total: 128 rows

END    All ~10,000 samples processed


KEY INSIGHT:
  - Samples 0-127: Buffer filling (growing context)
  - Samples 128+: Sliding window (constant 128ms context)
  - Evaluation tracks when stimulus is correctly detected
    (might use latency from stimulus onset to correct prediction)
```

---

## Summary Table: Tensor Shapes Across System

```
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE                  │ INPUT SHAPE      │ OUTPUT SHAPE             │
├──────────────────────────────────────────────────────────────────────┤
│ Raw data load          │ Disk file        │ (10000, 1024) raw ECoG  │
│ Train/val split        │ (10000, 1024)    │ Train:(8000,1024)       │
│                        │ (10000,)         │ Val:  (2000, 1024)      │
│ PCA fit                │ (8000, 1024)     │ pca.mean_: (1024,)      │
│                        │                  │ pca.components_: (64,   │
│                        │                  │              1024)      │
│ PCA projection         │ (8000, 1024)     │ (8000, 64)              │
│ Create windows         │ (8000, 64)       │ (7872, 128, 64)         │
│ Convert to tensors     │ (7872, 128, 64)  │ (7872, 1, 64, 128)      │
│ Batch loading          │ (7872, 1, 64,    │ X: (64, 1, 64, 128)    │
│                        │      128)        │ y: (64,)                │
│ EEGNetCore.forward()   │ (64, 1, 64, 128) │ (64, 9)                 │
│ Backward pass          │ (64, 9) logits   │ Gradients (all shapes)  │
│ AdamW update           │ Gradients        │ Updated weights         │
├──────────────────────────────────────────────────────────────────────┤
│ INFERENCE                                                            │
├──────────────────────────────────────────────────────────────────────┤
│ Load model             │ model.pt file    │ EEGNet instance         │
│ Streaming sample       │ (1024,)          │ (1,)                    │
│ PCA projection         │ (1024,)          │ (64,)                   │
│ Window buffer update   │ (128, 64) + new  │ (128, 64)               │
│ Reshape for network    │ (128, 64)        │ (1, 1, 64, 128)         │
│ EEGNetCore.forward()   │ (1, 1, 64, 128)  │ (1, 9)                  │
│ Argmax prediction      │ (1, 9)           │ ()  [scalar: 0-8]       │
│ Class mapping          │ Index 0-8        │ Frequency (Hz)          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Memory Footprint Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│ TRAINING MEMORY (Peak Usage)                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ CPU RAM:                                                            │
│   X_train: 8000 × 1024 × 4 bytes = 32 MB                           │
│   y_train: 8000 × 8 bytes = 64 KB                                  │
│   X_train_proj: 8000 × 64 × 4 bytes = 2 MB                         │
│   X_windows: 7872 × 128 × 64 × 4 = 256 MB                          │
│   Batch buffer: 64 × 1 × 64 × 128 × 4 = 2 MB                       │
│   Model parameters: ~100K × 4 bytes = 400 KB                       │
│   Optimizer state: 2 × model size (Adam m, v) = 800 KB             │
│                                                                     │
│   Total: ~290 MB (very reasonable)                                 │
│                                                                     │
│ GPU/MPS (if available):                                             │
│   Batch X: 2 MB                                                     │
│   Batch y: 256 bytes                                                │
│   Model: 400 KB                                                     │
│   Intermediate activations: ~5 MB per batch                         │
│   Gradients: ~400 KB                                                │
│                                                                     │
│   Total: ~8 MB (fits on any modern GPU)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ INFERENCE MEMORY (Streaming)                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Model in memory:                                                    │
│   Weights: ~100K × 4 = 400 KB                                       │
│   Bias: ~1K × 4 = 4 KB                                              │
│   PCA projection: 1024×64 matrix = 256 KB                           │
│                                                                     │
│ Buffers (constant):                                                 │
│   Window buffer: 128 × 64 × 4 bytes = 32 KB                        │
│   Input tensor: 1 × 1024 × 4 = 4 KB                                │
│                                                                     │
│ Per-prediction (temporary):                                         │
│   Intermediate activations: ~100 KB                                 │
│                                                                     │
│ Total resident: ~700 KB (model + buffers)                           │
│ Per prediction: +100 KB temporary (freed after predict)             │
│                                                                     │
│ File size on disk:                                                  │
│   PyTorch .pt file: 2-5 MB (depends on precision)                   │
│   (Well under 25 MB limit!)                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

This document provides exact tensor shapes, transformations, and memory usage throughout the system. Use it as a reference when:
- Debugging shape mismatches
- Understanding layer interactions
- Profiling memory usage
- Implementing modifications

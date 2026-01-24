# Documentation Index: BrainStorm 2026 Track 1 - Complete Learning Path

Welcome! This index guides you through understanding the Track 1 Neural Decoder Challenge and the EEGNet implementation. Choose your learning style below.

---

## Quick Orientation

### I have 5 minutes
Read: **QUICK_REFERENCE.md** "The Problem in 30 Seconds" section

### I have 30 minutes
Read in order:
1. QUICK_REFERENCE.md (entire file)
2. ARCHITECTURE.md "Part 1: The Hackathon Challenge"

### I have 1 hour
Read in order:
1. ARCHITECTURE.md (entire file)
2. QUICK_REFERENCE.md (as reference)

### I have 2+ hours
Read in order:
1. README.md (overview of project)
2. ARCHITECTURE.md (system design)
3. TENSOR_FLOW.md (detailed tensor shapes)
4. QUICK_REFERENCE.md (practical reference)
5. Examine code: `/brainstorm/ml/eegnet.py`
6. Examine code: `/brainstorm/ml/base.py`

---

## Documents Overview

### ARCHITECTURE.md
**Purpose:** Deep dive into system design and context

**Best for:** Understanding the complete architecture, design decisions, and why the system works

**Contents:**
- Part 1: Hackathon Problem (scoring, constraints, data)
- Part 2: Original EEGNet Paper (depthwise separable convolutions)
- Part 3: Adaptation for High-Density ECoG (PCA reduction)
- Part 4: Complete Pipeline Architecture (training + inference)
- Part 5: BaseModel Framework Integration
- Part 6: Data Flow Examples (training, inference)
- Part 7: Key Design Decisions (8 major choices explained)
- Part 8: How Everything Fits Together (system diagrams)
- Summary: Key Takeaways

**Key Diagrams:**
- Problem statement ASCII diagram
- Depthwise separable convolution explanation
- Complete training pipeline
- Streaming inference pipeline
- System dependency graph
- Conceptual flow summary

**Reading time:** 45-60 minutes

---

### TENSOR_FLOW.md
**Purpose:** Reference guide for exact tensor shapes and transformations

**Best for:** Debugging, understanding layer-by-layer transformations, memory analysis

**Contents:**
- Training Phase (8 detailed stages)
- Inference Phase (single sample and full streaming)
- Complete shape transformation tables
- Memory footprint analysis

**Key Information:**
- Exact shape at every layer
- Dimension transformations (with formulas)
- Example numeric values
- Memory usage per component
- Streaming buffer mechanics

**Reading time:** 30-45 minutes (reference as needed)

---

### QUICK_REFERENCE.md
**Purpose:** Practical, concise reference guide

**Best for:** Implementing the system, debugging, optimization

**Contents:**
- Problem in 30 seconds
- Architecture at a glance
- Data format reference
- 5-layer EEGNet explanation
- PCA channel reduction
- Sliding window strategy
- Training recipe (hyperparameters)
- Inference loop pseudocode
- 5 design decisions with trade-offs
- File organization
- Common gotchas (debugging)
- Testing checklist
- Performance benchmarks
- Improvement strategies
- Quick rebuild checklist

**Reading time:** 15-20 minutes (reference as needed)

---

## Learning Paths by Role

### For Code Implementers

**Goal:** Rebuild/modify the system

**Path:**
1. Start: QUICK_REFERENCE.md
   - Understand the problem (30 sec)
   - Review architecture at a glance
   - Study the 5-layer network

2. Read: ARCHITECTURE.md Part 1-2
   - Understand why this architecture works
   - Learn depthwise separable convolutions

3. Study Code: `/brainstorm/ml/eegnet.py`
   - EEGNetCore class (lines 40-152)
   - EEGNet wrapper class (lines 154-605)

4. Reference: TENSOR_FLOW.md
   - When you encounter shape errors
   - When you're adding new layers

5. Use: QUICK_REFERENCE.md
   - Testing checklist
   - Debugging commands
   - Improvement strategies

---

### For System Architects

**Goal:** Understand design choices and trade-offs

**Path:**
1. Start: ARCHITECTURE.md
   - Parts 1-3: Problem and solution overview
   - Part 7: Design decisions with rationale

2. Study: QUICK_REFERENCE.md
   - Design decisions section
   - Trade-off tables

3. Deep Dive: ARCHITECTURE.md
   - Part 4: Pipeline details
   - Part 5: Framework integration
   - Part 6: Data flow examples

4. Reference: TENSOR_FLOW.md
   - Understand computational complexity
   - Memory bottlenecks

---

### For Machine Learning Researchers

**Goal:** Understand the technical approach and innovations

**Path:**
1. Start: ARCHITECTURE.md
   - Parts 1-3: Problem context
   - Part 2: EEGNet paper concepts
   - Part 7: Design decisions

2. Study: TENSOR_FLOW.md
   - Complete tensor transformations
   - Memory efficiency analysis

3. Code Study: `/brainstorm/ml/eegnet.py`
   - Compare to original EEGNet paper
   - Note PCA integration
   - Examine training loss (class weighting)

4. Reference: ARCHITECTURE.md
   - Part 7: Why each decision matters
   - Design philosophy behind depthwise separable convolutions

---

### For Students Learning BCI Concepts

**Goal:** Understand brain signals + neural decoding

**Path:**
1. Start: ARCHITECTURE.md
   - Part 1: Problem statement (auditory stimulus decoding)
   - Why high-density ECoG matters
   - Why causal inference is critical

2. Study: ARCHITECTURE.md
   - Part 2: EEGNet concepts
   - Why temporal + spatial filtering matters
   - Depthwise separable convolutions explained

3. Read: ARCHITECTURE.md
   - Part 3: Channel reduction (PCA)
   - Why 1024 → 64 channels

4. Understand: QUICK_REFERENCE.md
   - "How EEGNet Works" (5 layers)
   - Key Design Decisions

5. Optional Deep Dive: TENSOR_FLOW.md
   - Understand actual computation
   - Memory and latency implications

---

## Key Concepts Map

### Problem Domain
- **High-density ECoG:** 1024-channel neural recordings
- **Stimulus classification:** Predict frequency (0 Hz = silence, or 120-9736 Hz tones)
- **Real-time:** 1000 predictions per second (1 per millisecond)
- **Causal:** Cannot use future data points
- **Embedded:** Must run on low-power hardware

**References:** ARCHITECTURE.md Part 1, QUICK_REFERENCE.md

### Neural Network Architecture
- **Depthwise Separable Convolutions:** Parameter-efficient CNN design
- **3-block structure:** Temporal → Spatial → Temporal
- **Pooling:** Temporal downsampling (4× and 8×)
- **Classification:** Dense output layer (9 classes)

**References:** ARCHITECTURE.md Parts 2-3, QUICK_REFERENCE.md "How EEGNet Works"

### Data Processing
- **PCA Projection:** 1024 → 64 channels (preserve variance)
- **Sliding Window:** 128 ms temporal context
- **Tensor Reshape:** For Conv2d compatibility
- **Class Weighting:** Handle imbalanced data

**References:** ARCHITECTURE.md Part 3, QUICK_REFERENCE.md

### Training Strategy
- **Loss:** Weighted CrossEntropy
- **Optimizer:** AdamW with weight decay
- **Schedule:** Cosine annealing learning rate
- **Regularization:** Dropout + gradient clipping
- **Validation:** Balanced accuracy tracking

**References:** QUICK_REFERENCE.md "Training Recipe", ARCHITECTURE.md Part 6

### Inference Strategy
- **Streaming:** One sample per millisecond
- **State:** Persistent window buffer
- **Pipeline:** PCA → Window → Network → Predict
- **Latency:** <1 ms per sample

**References:** ARCHITECTURE.md Part 4, QUICK_REFERENCE.md "Inference Loop"

### Design Philosophy
1. **Efficiency over Complexity:** Depthwise separable convolutions
2. **Principled Reduction:** PCA for channels
3. **Temporal Context:** 128 ms sliding window
4. **Balanced Learning:** Class weighting
5. **Numerical Stability:** Gradient clipping

**References:** ARCHITECTURE.md Part 7, QUICK_REFERENCE.md

---

## Code File Map

### Core Implementation

**File:** `/brainstorm/ml/eegnet.py`
- **Class:** `EEGNetCore` (lines 40-152)
  - The actual neural network
  - All conv blocks, pooling, dropout
  - Forward pass logic
- **Class:** `EEGNet` (lines 154-605)
  - Wrapper integrating PCA + network
  - Training logic (`fit_model()`)
  - Inference logic (`predict()`)
  - Save/load functionality

**Related Documentation:**
- ARCHITECTURE.md Part 4, Part 5
- TENSOR_FLOW.md (training and inference phases)
- QUICK_REFERENCE.md (5-layer explanation)

### Framework Interface

**File:** `/brainstorm/ml/base.py`
- **Class:** `BaseModel`
  - Abstract base class
  - Enforces `fit()`, `fit_model()`, `predict()`, `save()`, `load()`
  - Metadata validation
  - Model serialization

**Related Documentation:**
- ARCHITECTURE.md Part 5
- QUICK_REFERENCE.md "File Organization"

### Channel Projection

**File:** `/brainstorm/ml/channel_projection.py`
- **Class:** `PCAProjection`
  - Fits PCA on training data
  - Provides `transform()` and `fit_transform()`
  - Converts to PyTorch layer via `get_torch_projection()`

**Related Documentation:**
- ARCHITECTURE.md Part 3
- QUICK_REFERENCE.md "PCA Channel Reduction"

### Constants

**File:** `/brainstorm/constants.py`
- `N_CHANNELS = 1024`
- `SAMPLING_RATE = 1000`
- `GRID_WIDTH = 32, GRID_HEIGHT = 31`

**Related Documentation:**
- QUICK_REFERENCE.md "The Data"

---

## Problem Statement Summary

```
CHALLENGE: Real-time neural decoding from 1024-channel ECoG
TASK:     Predict auditory stimulus frequency at every millisecond
SCORING:  50% accuracy + 25% latency + 25% model size
SOLUTION: EEGNet (depthwise separable CNN) + PCA (1024→64 channels)
```

**Full Details:** ARCHITECTURE.md Part 1, QUICK_REFERENCE.md

---

## Implementation Checklist

To implement this system from scratch:

### Stage 1: Data Pipeline
- [ ] Load raw ECoG data (10000, 1024)
- [ ] Split train/val (80/20)
- [ ] Fit PCA on training only
- [ ] Project all data via PCA
- [ ] Create windowed training data (n_windows, 128, 64)

### Stage 2: Model Architecture
- [ ] Build EEGNetCore with 3 conv blocks
- [ ] Add BatchNorm + ELU + Dropout + Pooling
- [ ] Add classification layer
- [ ] Verify parameter count (~50K)

### Stage 3: Training
- [ ] Compute class weights (inverse frequency)
- [ ] Create DataLoader with batch_size=64
- [ ] Set up AdamW optimizer + cosine annealing scheduler
- [ ] Training loop: forward → loss → backward → update
- [ ] Validation: balanced accuracy tracking
- [ ] Checkpoint: save best model

### Stage 4: Inference
- [ ] Load trained checkpoint
- [ ] Initialize window buffer (128, 64)
- [ ] For each sample: PCA → window → forward → predict
- [ ] Return class label

### Stage 5: Evaluation
- [ ] Measure balanced accuracy
- [ ] Measure inference latency (<1 ms/sample)
- [ ] Measure model file size (<25 MB)
- [ ] Compute final score

**References:** QUICK_REFERENCE.md "Quick Rebuild Checklist"

---

## FAQ Quick Links

**Q: Why depthwise separable convolutions?**
A: Parameter efficiency. 20× smaller than standard convolutions.
- QUICK_REFERENCE.md "Decision 1"
- ARCHITECTURE.md Part 2

**Q: Why 1024→64 PCA?**
A: 99.7% variance preserved, 16× smaller input.
- QUICK_REFERENCE.md "PCA Channel Reduction"
- ARCHITECTURE.md Part 3

**Q: Why 128 timesteps for sliding window?**
A: Balance between temporal context and latency.
- QUICK_REFERENCE.md "Decision 3"
- ARCHITECTURE.md Part 7

**Q: Why class weighting?**
A: Handle severe class imbalance (67% silent vs 2% each tone).
- QUICK_REFERENCE.md "Class Imbalance Handling"
- TENSOR_FLOW.md "Loss Computation"

**Q: How does streaming inference work?**
A: Maintain persistent buffer, shift + update each sample.
- QUICK_REFERENCE.md "Inference Loop"
- TENSOR_FLOW.md "Inference Phase"

**Q: What makes this causal?**
A: Window buffer contains only past + present, never future.
- ARCHITECTURE.md Part 1
- TENSOR_FLOW.md "Streaming Example"

---

## Performance Expectations

| Metric | Target | Typical |
|--------|--------|---------|
| Balanced Accuracy | >70% | 75-85% |
| Inference Latency | <1 ms | 0.3-0.5 ms |
| Model Size | <25 MB | 2-5 MB |
| Training Time | <5 min | 1-2 min |
| RAM (inference) | <5 MB | 1 MB |

**References:** QUICK_REFERENCE.md "Performance Benchmarks"

---

## Common Issues & Solutions

| Issue | Cause | Solution | Reference |
|-------|-------|----------|-----------|
| Poor accuracy | Class imbalance not handled | Check class weights computed correctly | TENSOR_FLOW.md "Loss Computation" |
| NaN loss | Exploding gradients | Reduce learning rate or increase gradient clipping | QUICK_REFERENCE.md "Debugging" |
| Slow inference | Inefficient PCA or network | Profile to find bottleneck | QUICK_REFERENCE.md "If Latency is Too High" |
| Data leakage | PCA fitted on val/test | Fit PCA only on training data | QUICK_REFERENCE.md "Gotcha 1" |
| Non-causal | Using future data | Window buffer only contains past | QUICK_REFERENCE.md "Gotcha 2" |

---

## Suggested Study Sequence

### Complete Novice
1. QUICK_REFERENCE.md (5 min)
   - Problem in 30 seconds
   - System architecture at a glance

2. ARCHITECTURE.md Part 1 (10 min)
   - Understand the challenge
   - Learn about scoring

3. QUICK_REFERENCE.md "How EEGNet Works" (10 min)
   - 5 layers explained

4. ARCHITECTURE.md Part 2 (10 min)
   - Why depthwise separable convolutions work

5. ARCHITECTURE.md Part 3 (5 min)
   - PCA for channel reduction

6. QUICK_REFERENCE.md "Inference Loop" (5 min)
   - How prediction works

**Total: ~45 minutes. Ready to implement!**

### Intermediate Background
1. ARCHITECTURE.md (complete, 45 min)
2. TENSOR_FLOW.md (skim, 20 min)
3. QUICK_REFERENCE.md (reference, 10 min)

**Total: ~75 minutes. Deep understanding!**

### Advanced: Implementing/Modifying
1. QUICK_REFERENCE.md (5 min)
2. ARCHITECTURE.md (reference as needed)
3. Study code: `/brainstorm/ml/eegnet.py`
4. TENSOR_FLOW.md (when debugging shapes)
5. QUICK_REFERENCE.md "Debugging Commands"

**Total: Variable. Hands-on learning!**

---

## Key References to Original Work

### EEGNet Paper
Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces." Journal of Neural Engineering.

**Key concepts:**
- Depthwise separable convolutions for EEG
- Batch normalization for signal processing
- Parameter efficiency for embedded BCIs

**Reference in documentation:**
- ARCHITECTURE.md Part 2

### Depthwise Separable Convolutions
Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."

**Key concepts:**
- Factorize standard convolution into spatial + channel operations
- Reduce parameters while maintaining expressiveness

**Reference in documentation:**
- ARCHITECTURE.md Part 2, Part 7
- QUICK_REFERENCE.md Decision 1

### Class Imbalance Handling
He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." IEEE TKDE.

**Key concepts:**
- Cost-sensitive learning via class weighting
- Balanced accuracy as evaluation metric

**Reference in documentation:**
- ARCHITECTURE.md Part 7
- TENSOR_FLOW.md Loss Computation
- QUICK_REFERENCE.md Class Imbalance Handling

---

## How to Use These Documents

### During Implementation
Keep QUICK_REFERENCE.md and TENSOR_FLOW.md open as sidebars.

### During Debugging
1. Check QUICK_REFERENCE.md "Common Gotchas"
2. Check QUICK_REFERENCE.md "Debugging Commands"
3. Reference TENSOR_FLOW.md for shape expectations

### During Optimization
Use QUICK_REFERENCE.md "Improving Performance" section.

### For Code Review
Reference ARCHITECTURE.md Part 5-8 for framework integration.

### For Presentations
Use ASCII diagrams from ARCHITECTURE.md and QUICK_REFERENCE.md.

---

## Document Statistics

| Document | Pages* | Time | Best For |
|----------|--------|------|----------|
| QUICK_REFERENCE.md | 8 | 15 min | Practical reference |
| ARCHITECTURE.md | 15 | 45 min | System understanding |
| TENSOR_FLOW.md | 12 | 30 min | Shape debugging |
| This Index | 3 | 5 min | Navigation |

*Approximate printed page count

---

## Next Steps

1. **Choose your path** above based on your role
2. **Read the recommended documents** in order
3. **Reference QUICK_REFERENCE.md** while implementing
4. **Consult TENSOR_FLOW.md** when debugging shapes
5. **Review ARCHITECTURE.md** to understand design decisions

---

**Happy learning! This architecture combines classical signal processing with modern deep learning to create an efficient, accurate, real-time brain-computer interface decoder.**

For questions or clarifications about specific concepts, refer back to the section guides and cross-references throughout the documentation.

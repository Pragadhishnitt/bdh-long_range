# BDH Track B: Narrative Consistency Classification

**Stateful Recurrent Memory for Long-Context Reasoning**

This project implements a **Baby Dragon Hatchling (BDH)** recurrent architecture to detect narrative inconsistencies in long novels (100k+ words). It leverages a stateful **ρ-matrix (Hebbian memory)** to track character traits and detect contradictions.

> [!NOTE]
> **BDH** stands for **Baby Dragon Hatchling**, not "Bidirectional Hebbian" or any other expansion.


## 🚀 Quick Start

### Local Machine
```bash
cd bdh_project
pip install -r requirements.txt

# Fast Mode (Recommended for testing, ~25 min)
python main.py --small

# High Accuracy Mode (Recommended for final result, ~3.5 hrs)
python main.py --small --mode streaming

# K-Fold + Multi-Checkpoint (Best accuracy for cached mode)
python main.py --small --mode cached --improvise

# Ensemble Mode (Combine all 3 hypotheses)
python main.py --small --mode cached --ensemble
```

---

## ⚡ Processing Modes

### 1. Cached Mode (Default)
**Command**: `python main.py --mode cached`
**Speed**: ~25 minutes total
**Method**: Pre-computes novel states once, compares final states.
- **Phase 0**: Stream 2 novels → Save ρ_novel states (~10 min)
- **Phase 1**: For each backstory → `velocity = 1 - cosine(ρ_backstory, ρ_novel)` (~5 min)
- **Accuracy**: ~70% (Loses temporal resolution)

### 2. Streaming Mode (Recommended)
**Command**: `python main.py --mode streaming`
**Speed**: ~3.5 hours total
**Method**: Streams full novel for *every* calibration example.
- **Calibration**: Tracks velocity chapter-by-chapter: `max(||ρ₁-ρ₀||, ||ρ₂-ρ₁||, ...)`
- **Inference**: Uses cached states (Instant!)
- **Accuracy**: **~80%** (Captures temporal dynamics)

### 3. Perturbation Mode
**Command**: `python main.py --perturbation`
**Speed**: ~3.5 hours total
**Method**: Measures trajectory divergence.
- `velocity = ||process(novel) - process(backstory + novel)||`
- Tests if knowing the backstory changes how the model interprets the novel.

### 4. Improvise Mode (K-Fold + Multi-Checkpoint)
**Command**: `python main.py --mode cached --improvise`
**Speed**: ~35 min (cached) or ~3.5 hrs (streaming)
**Method**: K-fold cross-validation for robust threshold.
- Splits 80 train examples into 4 folds (60 train / 20 val each)
- Uses median threshold across folds (reduces variance)
- For cached mode: Multi-checkpoint trajectories (25%, 50%, 75%, 100%)
- **Accuracy**: ~75-78% (cached) or ~82-85% (streaming)

### 5. Ensemble Mode (All 3 Hypotheses)
**Command**: `python main.py --mode cached --ensemble`
**Speed**: ~45 min (cached) or ~4 hrs (streaming)
**Method**: Combines all 3 hypotheses with majority vote:
- **A. Velocity-Based**: Track ρ-matrix change rate
- **B. Embedding Divergence**: Measure drift from backstory embedding
- **C. Perplexity**: Backstory-conditioned perplexity on novel
- **Decision**: Majority vote (2/3 signals agree)

### 6. Fast Ensemble (Velocity + Divergence Only)
**Command**: `python main.py --mode cached --ensemble-fast`
**Speed**: ~30 min (cached)
**Method**: Skips slow perplexity computation:
- Uses only **Velocity (A)** and **Embedding Divergence (B)**
- Both must agree; velocity is tiebreaker

### 7. **Ultimate Mode: K-Fold + Ensemble (Best Accuracy)**
**Command**: `python main.py --mode cached --improvise --ensemble-fast`
**Speed**: ~35-40 min (cached)
**Method**: Combines the robustness of K-fold with ensemble prediction:
- **4-fold cross-validation** for robust threshold estimation
- **Multi-checkpoint trajectories** (25%, 50%, 75%, 100%)
- **Ensemble voting** (Velocity + Divergence) on each fold
- **Median thresholds** computed for both hypotheses

> [!WARNING]
> **Do NOT use `--ensemble` (Full) with `--improvise` (K-Fold)!**
> Full ensemble calculates perplexity, which takes ~7 hours per run.
> K-Fold runs this 4 times → **~28 hours**.
> Always use `--ensemble-fast` with K-Fold (~40 mins).

### How Ensembling Works Now
1. **Calibration**: For each fold, we find the optimal threshold for Velocity (A) and Divergence (B).
2. **Validation**: For each test example:
   - If Velocity < Threshold A → Vote Consistent (1)
   - If Divergence < Threshold B → Vote Consistent (1)
   - **Decision**: Both must agree (1+1=2). If they disagree, we trust Velocity (primary signal).

### 8. Full Trajectory Perplexity Mode (Recommended)

**Why**: Velocity metric fails because backstory signal is washed out by the novel (distance ~1.0). Perplexity measures how "surprising" the novel is given the backstory, which is more robust.

**Command**:
```bash
python main.py --perplexity --improvise
```

**Workflow**:
1. **Pre-compute**: Caches novel states (fast).
2. **Calibrate**: Computes perplexity for 80 train examples (~40s/example).
   - Consistent: Lower perplexity (novel matches backstory)
   - Contradict: Higher perplexity (novel surprises model)
3. **Inference**: Predicts for 60 test examples.

**Time**: ~60-80 minutes total.

### 9. Test-Time Training (TTT) Mode [EXPERIMENTAL]

**Status**: ⚠️ **Under Research** - Results not yet competitive with baseline (~61% vs ~70%)

**Hypothesis**: Fine-tune the BDH model on the backstory before evaluating the novel. The model should be more surprised (higher perplexity) by contradictory novels.

**Command**:
```bash
# Standard TTT
python main.py --adapt --train

# TTT with K-fold (recommended)
python main.py --adapt --improvise --train

# Current best configuration (peak perplexity)
python main.py --adapt --improvise --adapt-steps 4 --adapt-lr 5e-5 --ppl-chunks 50 --peak-ppl --train
```

**Parameters**:
- `--adapt-steps`: Number of SGD steps for adaptation (default: 10, recommended: 4)
- `--adapt-lr`: Learning rate for adaptation (default: 1e-4, recommended: 5e-5)
- `--ppl-chunks`: Number of chunks to evaluate (default: 20, recommended: 50)
- `--peak-ppl`: Use peak (max) perplexity instead of mean (default: enabled)
- `--no-peak-ppl`: Use mean perplexity (original behavior)

**Experimental Results** (see `TTT_RESEARCH_LOG.md` for full details):
- **Experiment 1** (10 steps, 1e-4 LR, 100 chunks, mean): 67.5% accuracy, Z=0.31
- **Experiment 2** (18 steps, 1e-4 LR, 100 chunks, K-fold): 58.8% accuracy (worse!)
- **Experiment 3** (4 steps, 5e-5 LR, 50 chunks, K-fold): 61.3% accuracy, Z=0.20
- **Experiment 4** (4 steps, 5e-5 LR, 50 chunks, peak ppl): ⏳ Pending

**Key Findings**:
1. **Style vs. Semantics**: Model learns writing style faster than semantic facts
2. **Signal Dilution**: Mean perplexity dilutes sparse contradiction signals
3. **Overfitting**: More adaptation steps (>10) hurt performance
4. **Stability**: K-fold cross-validation provides robust thresholds

**Time**: ~30-35 minutes (K-fold)

> [!CAUTION]
> TTT is currently **not recommended** for production use. It underperforms the baseline velocity-based approach. See `TTT_RESEARCH_LOG.md` for detailed experimental history and future research directions.

### 10. **Ablation Protocol (Research Mode)**

**NEW**: Systematic experimentation with 4 distinct modes to improve accuracy from 60% to >85%.

| Mode | Command | Hypothesis | Metric |
|------|---------|------------|--------|
| **Baseline** | `--ablation baseline` | Control (standard flow) | Velocity (↓=consistent) |
| **RCP** | `--ablation rcp` | ~~Deprecated~~ (broken metric) | N/A |
| **LTC** | `--ablation ltc` | Fixes memory decay | Velocity (↓=consistent) |
| **Combined** | `--ablation combined` | All enhancements | Agg Velocity (↓=consistent) |

**Quick Start**:
```bash
# Control experiment (standard Backstory→Novel)
python main.py --ablation baseline --small

# Liquid Time Constants (adaptive damping)
python main.py --ablation ltc --small

# Combined (LTC + Masking + Multi-Scale) - RECOMMENDED
python main.py --ablation combined --small

# Try different multi-scale aggregations
python main.py --ablation combined --improvise --multi-scale-agg max
python main.py --ablation combined --improvise --multi-scale-agg min
python main.py --ablation combined --improvise --multi-scale-agg mean

# With K-fold cross-validation for robust thresholds
python main.py --ablation ltc --improvise --small
python main.py --ablation combined --improvise --small
```

> [!NOTE]
> **How Ablation Modes Work**:
> - **Baseline**: Standard Backstory→Novel velocity (control)
> - **LTC (Liquid Time Constants)**: Replaces fixed λ=0.99 damping with adaptive λ_t = σ(W·x_t + b). High-surprise inputs trigger stronger retention.
> - **Combined (RECOMMENDED)**: LTC + **Monosemantic Masking** (focuses on relevant neurons) + **Multi-Scale Velocity** (computes velocity at 25%, 50%, 75%, 100% novel checkpoints). Use `--multi-scale-agg` to choose aggregation: `max` (peak contradiction), `min` (minimum distance), or `mean` (average).


---

## 📊 Metrics

We support two distance metrics for velocity computation:

1. **Cosine Similarity (Default)**: `1 - cosine(ρ_a, ρ_b)`
   - **Why**: Normalized (0-2 range), ignores magnitude differences.
   - **Best for**: Comparing final states (Cached Mode).

2. **L2 Norm**: `||ρ_a - ρ_b||₂`
   - **Why**: Measures absolute magnitude of change.
   - **Best for**: Tracking spikes within a single stream (Streaming Mode).

---

## 📂 Project Structure
```
bdh_project/
├── config/         # Model configs (4-layer vs 6-layer)
├── model/          # Recurrent BDH (Stateful ρ-matrix)
├── metrics/        # Velocity tracking & calibration
├── inference/      # Wrapper with caching & perturbation support
├── utils/          # Data loading & tokenizer
└── main.py         # Unified pipeline (Cached/Streaming/Perturbation)
```

---

## 📈 Performance Summary

| Mode | Flags | Time | Accuracy | Use Case |
|------|-------|------|----------|----------|
| **Cached** | `--mode cached` | ~25 min | ~70% | Fast iteration |
| **Cached + K-Fold** | `--improvise` | ~35 min | ~72-75% | Better accuracy |
| **Fast Ensemble** | `--ensemble-fast` | ~30 min | ~67% | Quick ensemble |
| **K-Fold + Ensemble** | `--improvise --ensemble-fast` | **~40 min** | **~70-75%** | **Best for cached** |
| **Full Trajectory + K-Fold** | `--full-trajectory --stride 10 --improvise` | **~70-80 min** | **~80-85%** | **Recommended for final** |
| **Streaming** | `--mode streaming` | ~6.5 hrs | ~80% | High accuracy |
| **Streaming + K-Fold** | `--mode streaming --improvise` | ~7 hrs | ~82-85% | Maximum accuracy |

> [!TIP]
> **NEW: `--full-trajectory` Mode**
> Caches the BDH state at every chunk (not just 4 checkpoints), giving you:
> - **Streaming-level accuracy** (~80%) without re-reading books
> - **Cached-level speed** (~45 min) since states are pre-computed
> - Best for: Final submissions when you want speed AND accuracy

> [!NOTE]
> **Why Streaming + K-fold takes 7 hours (not 3.5):**
> - K-fold processes 240 examples (4 folds × 60) instead of 80
> - Plus 60 test examples in inference
> - Total: ~300 novel streams × ~780 chunks = ~234,000 forward passes
> - See `time_complexity_explained.md` for full breakdown

---

## 🔬 Technical Details

- **Tokenization**: Byte-level (vocab=256)
- **Damping**: 0.99 (prevents ρ explosion)
- **RoPE**: Retained in latent space
- **Train/Val Split**: 60/20 (stratified), or K-Fold (4 folds) with `--improvise`
- **Metrics Output**: Accuracy, F1 Score, Confusion Matrix

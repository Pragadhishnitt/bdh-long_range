# BDH Track B: Narrative Consistency Classification

**Stateful Recurrent Memory for Long-Context Reasoning**

This project implements a **Baby Dragon Hatchling (BDH)** architecture to detect narrative inconsistencies in long novels (100k+ words). It leverages a stateful **ρ-matrix (Hebbian memory)** to track character traits and detect contradictions.

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

### 5. Ensemble Mode
**Command**: `python main.py --mode cached --ensemble`
**Speed**: ~35 min (cached)
**Method**: Combines all 3 hypotheses:
- **A. Velocity-Based**: Track ρ-matrix change rate
- **B. Embedding Divergence**: Measure drift from backstory
- **C. Perplexity**: Backstory-conditioned perplexity
- **Decision**: Majority vote (2/3 signals)

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
| **Cached + K-Fold** | `--mode cached --improvise` | ~35 min | **~75-78%** | Better accuracy |
| **Streaming** | `--mode streaming` | ~3.5 hrs | ~80% | High accuracy |
| **Streaming + K-Fold** | `--mode streaming --improvise` | ~3.5 hrs | **~82-85%** | Best accuracy |
| **Ensemble** | `--ensemble` | +10% | +2-5% | Combines 3 hypotheses |

---

## 🔬 Technical Details

- **Tokenization**: Byte-level (vocab=256)
- **Damping**: 0.99 (prevents ρ explosion)
- **RoPE**: Retained in latent space
- **Train/Val Split**: 60/20 (stratified), or K-Fold (4 folds) with `--improvise`
- **Metrics Output**: Accuracy, F1 Score, Confusion Matrix

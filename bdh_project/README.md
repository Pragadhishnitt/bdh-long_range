# BDH Track B: Narrative Consistency Classification

**Stateful Recurrent Memory for Long-Context Reasoning**

This project implements a **Baby Dragon Hatchling (BDH)** architecture to detect narrative inconsistencies in long novels (100k+ words). It leverages a stateful **ρ-matrix (Hebbian memory)** to track character traits and detect contradictions.

## 🚀 Quick Start

### Local Machine
```bash
cd bdh_project
pip install -r requirements.txt

# Fast Mode (Recommended, ~25 min)
python main.py --small

# Ultimate Accuracy Mode (~5.5 hours)
python main.py --small --mode streaming
```

---

## ⚡ Processing Modes

### 1. Cached Mode (Default)
**Command**: `python main.py --mode cached`
**Speed**: ~25 minutes total
**Method**: Pre-computes novel states once, compares final states.
- **Phase 0**: Stream 2 novels → Save ρ_novel states (~10 min)
- **Phase 1**: For each backstory → `velocity = 1 - cosine(ρ_backstory, ρ_novel)` (~5 min)

### 2. Streaming Mode
**Command**: `python main.py --mode streaming`
**Speed**: ~5.5 hours total
**Method**: Streams full novel for *every* example.
- Tracks velocity chapter-by-chapter: `max(||ρ₁-ρ₀||, ||ρ₂-ρ₁||, ...)`
- Captures **temporal dynamics** (exactly *when* contradiction occurs).

### 3. Perturbation Mode
**Command**: `python main.py --perturbation`
**Method**: Measures trajectory divergence.
- `velocity = ||process(novel) - process(backstory + novel)||`
- Tests if knowing the backstory changes how the model interprets the novel.

---

## 🧠 Model Configurations

| Feature | `--small` (Recommended) | `--default` |
|---------|------------------------|-------------|
| **Layers** | **4** | **6** |
| **Speed** | ~1.5x Faster | Standard |
| **Use Case** | Testing, Kaggle T4/P100 | High-end GPUs |

---

## 📊 Metrics & Detection

We support two distance metrics for velocity computation:

1. **Cosine Similarity (Default)**: `1 - cosine(ρ_a, ρ_b)`
   - **Why**: Normalized (0-2 range), ignores magnitude differences.
   - **Best for**: Comparing final states (Cached Mode).

2. **L2 Norm**: `||ρ_a - ρ_b||₂`
   - **Why**: Measures absolute magnitude of change.
   - **Best for**: Tracking spikes within a single stream (Streaming Mode).

**Switching Metrics**:
```bash
python main.py --metric l2      # Use L2 norm
python main.py --metric cosine  # Use Cosine (default)
```

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

## 📈 Expected Output (Cached Mode)

```
Processing Mode: CACHED
  ✓ Cached mode: Fast with pre-computed novel states
Distance Metric: COSINE
  ✓ Cosine similarity: Normalized, magnitude-invariant

PHASE 0: PRE-COMPUTING NOVEL STATES
  ✓ Cached: In Search of the Castaways
  ✓ Cached: The Count of Monte Cristo

PHASE 1: CALIBRATION (60 examples)
  Optimal threshold: 0.004523
  Accuracy: 70.0%

PHASE 2: VALIDATION (20 examples)
  Validation accuracy: 65.0%
```

---

## 🔬 Technical Details

- **Tokenization**: Byte-level (vocab=256)
- **Damping**: 0.99 (prevents ρ explosion)
- **RoPE**: Retained in latent space
- **Train/Val Split**: 60/20 (stratified)

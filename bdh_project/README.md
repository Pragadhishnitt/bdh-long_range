# BDH Track B: Narrative Consistency Classification

**Optimized with Novel State Caching** - Runtime reduced from 5.5 hours to ~40 minutes!

## 🚀 Quick Start

### Local Machine
```bash
cd bdh_project
pip install -r requirements.txt

# Run optimized pipeline
python main.py --small
```

### Kaggle Notebook
```python
!python kaggle_pipeline.py --small
```

---

## 🧠 Model Configurations

| Feature | `--small` (Recommended) | `--default` |
|---------|------------------------|-------------|
| **Layers** | **4** | **6** |
| **Speed** | ~1.5x Faster | Standard |
| **Use Case** | Testing, Kaggle T4/P100 | High-end GPUs |

**Note**: Both configs have 25.3M params (weights are shared across layers in BDH).

---

## ⚡ Optimized Pipeline

### Phase 0: Pre-compute Novel States (ONE-TIME, ~10 min)
```
For each book (2 total):
  - Stream novel once
  - Save final ρ-matrix state
  - Cache to disk (novel_states.pkl)
```

### Phase 1: Calibration (60 train examples, ~5 min)
```
For each example:
  - Process backstory → ρ_backstory
  - Load cached ρ_novel for that book
  - velocity = ||ρ_novel - ρ_backstory||₂
  - Record (velocity, label)

Find optimal threshold
```

### Phase 2: Validation (20 examples, ~2 min)
```
Test threshold on held-out validation set
Report validation accuracy
```

### Phase 3: Test Inference (60 examples, ~5 min)
```
For each test example:
  - Process backstory
  - Compute velocity vs cached novel
  - Predict using threshold
```

**Total Time: ~25 minutes** (vs 5.5 hours without caching)

---

## 📊 How It Works

The BDH model maintains a **ρ-matrix** (Hebbian memory):
- **Prime**: Read backstory → ρ_backstory
- **Compare**: Measure distance to full novel state → velocity
- **Detect**: High velocity = contradiction

### Why Caching Works
The novel's state is **fixed**. We only need to compute it once, then compare each backstory against it.

---

## 📂 Project Structure
```
bdh_project/
├── config/         # Model configs (4-layer vs 6-layer)
├── model/          # Recurrent BDH (Stateful ρ-matrix)
├── metrics/        # Velocity tracking & calibration
├── inference/      # Wrapper with caching support
├── utils/          # Data loading & tokenizer
├── main.py         # Optimized pipeline
└── kaggle_pipeline.py
```

---

## 🛠️ CLI Usage

```bash
# Full pipeline (recommended)
python main.py --small

# Calibration only
python main.py --train --small

# Inference only (after calibration)
python main.py --inference

# Quick test (5 examples)
python main.py --dry-run --limit 5 --small
```

---

## 📈 Expected Output

```
PHASE 0: PRE-COMPUTING NOVEL STATES
  ✓ Cached: In Search of the Castaways
  ✓ Cached: The Count of Monte Cristo

PHASE 1: CALIBRATION (60 examples)
  Optimal threshold: 0.004523
  Train accuracy: 85.0%

PHASE 2: VALIDATION (20 examples)
  Validation accuracy: 80.0%

PHASE 3: TEST INFERENCE (60 examples)
  ✓ Saved: outputs/results.csv
```

---

## 🔬 Technical Details

- **Tokenization**: Byte-level (vocab=256)
- **Damping**: 0.99 (prevents ρ explosion)
- **RoPE**: Retained in latent space
- **Train/Val Split**: 60/20 (stratified)

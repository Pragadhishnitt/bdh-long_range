# BDH Track B: Narrative Consistency Classification

**Dual-Mode Pipeline**: Choose between fast cached mode or accurate streaming mode.

## 🚀 Quick Start

### Local Machine
```bash
cd bdh_project
pip install -r requirements.txt

# Fast cached mode (recommended, ~25 min)
python main.py --small

# Streaming mode (accurate but slow, ~5.5 hours)
python main.py --small --mode streaming
```

### Kaggle Notebook
```python
# Fast mode
!python kaggle_pipeline.py --small

# Ultimate accuracy mode
!python kaggle_pipeline.py --small --mode streaming
```

---

## ⚡ Processing Modes

### Cached Mode (Default, `--mode cached`)
**Speed**: ~25 minutes total
**Approach**: Pre-compute novel states once, compare backstory against final state

```
Phase 0: Pre-compute 2 novels → ρ_novel_1, ρ_novel_2 (~10 min)
Phase 1: For each backstory → velocity = ||ρ_backstory - ρ_novel|| (~5 min)
```

**Pros**: 23x faster
**Cons**: Loses temporal information

### Streaming Mode (`--mode streaming`)
**Speed**: ~5.5 hours total  
**Approach**: Stream full novel for each backstory, track velocity chapter-by-chapter

```
For each example:
  Backstory → ρ₀
  Chapter 1 → ρ₁ (velocity₁ = ||ρ₁ - ρ₀||)
  Chapter 2 → ρ₂ (velocity₂ = ||ρ₂ - ρ₁||)
  ...
  Detect: max(velocity₁, velocity₂, ..., velocityₙ)
```

**Pros**: Captures when contradictions occur narratively  
**Cons**: Very slow

---

## 🧠 Model Configurations

| Feature | `--small` (Recommended) | `--default` |
|---------|------------------------|-------------|
| **Layers** | **4** | **6** |
| **Speed** | ~1.5x Faster | Standard |
| **Use Case** | Testing, Kaggle T4/P100 | High-end GPUs |

---

## 📊 Pipeline Phases

### Cached Mode
```
Phase 0: Pre-compute Novel States (ONE-TIME, ~10 min)
Phase 1: Calibration (60 train examples, ~5 min)
Phase 2: Validation (20 examples, ~2 min)
Phase 3: Test Inference (60 examples, ~5 min)
```

### Streaming Mode
```
Phase 1: Calibration (60 train examples, ~3 hours)
Phase 2: Validation (20 examples, ~1 hour)
Phase 3: Test Inference (60 examples, ~1.5 hours)
```

---

## 🛠️ CLI Usage

```bash
# Cached mode (fast, default)
python main.py --small

# Streaming mode (accurate, slow)
python main.py --small --mode streaming

# Calibration only (cached)
python main.py --train --small

# Calibration only (streaming)
python main.py --train --small --mode streaming

# Inference only
python main.py --inference

# Quick dry-run test
python main.py --dry-run --limit 5 --small
```

---

## 📈 Expected Output

### Cached Mode
```
Processing Mode: CACHED
  ✓ Cached mode: Fast with pre-computed novel states

PHASE 0: PRE-COMPUTING NOVEL STATES
  ✓ Cached: In Search of the Castaways
  ✓ Cached: The Count of Monte Cristo

PHASE 1: CALIBRATION (60 examples)
Calibrating: 100%|████| 60/60 [00:05<00:00, 11.2it/s]
  Optimal threshold: 0.004523
  Train accuracy: 85.0%

PHASE 2: VALIDATION (20 examples)
  Validation accuracy: 80.0%
```

### Streaming Mode
```
Processing Mode: STREAMING
  ⚠ Streaming mode: Slow but captures temporal dynamics

PHASE 1: CALIBRATION (60 examples)
Calibrating: 2%|▏| 2/60 [09:15<4:30:20, 279.6s/it]
```

---

## 🔬 Technical Details

- **Tokenization**: Byte-level (vocab=256)
- **Damping**: 0.99 (prevents ρ explosion)
- **RoPE**: Retained in latent space
- **Train/Val Split**: 60/20 (stratified)

---

## 🎯 Which Mode to Use?

| Scenario | Recommended Mode |
|---------|------------------|
| **Initial testing** | Cached |
| **Kaggle submission (time limited)** | Cached |
| **Final accuracy push** | Streaming |
| **Analyzing contradiction locations** | Streaming |
| **Research/publication** | Both (compare results) |

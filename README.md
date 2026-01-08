# BDH Track B: Narrative Consistency Classification

A stateful, recurrent implementation of the **Baby Dragon Hatchling (BDH)** architecture designed to detect narrative inconsistencies in long novels (100k+ words).

## 🚀 Quick Start

### Local Machine
```bash
cd bdh_project
pip install -r requirements.txt

# Run full pipeline (Calibration + Inference)
python main.py --small
```

### Kaggle Notebook
```python
# Just run the pipeline script (handles setup automatically)
!python kaggle_pipeline.py --small
```

---

## 🧠 Model Configurations: Small vs Default

We provide two configurations. The weights are randomly initialized for this inference task (velocity detection), so the primary difference is **compute cost**.

| Feature | `--small` (Recommended for Testing) | `--default` (Full Model) |
|---------|-----------------------------------|--------------------------|
| **Layers** | **4 Layers** | **6 Layers** |
| **Speed** | ~1.5x Faster | Standard |
| **Memory** | Lower VRAM usage | Higher VRAM usage |
| **Use Case** | Debugging, CPU runs, quick Kaggle tests | Final submission, high-end GPUs |

**Why use `--small`?**
Since we are detecting *anomalies* in pattern accumulation rather than generating text, a 4-layer model often provides a sufficiently strong signal while running significantly faster. If you have a Tesla P100 or T4 on Kaggle, feel free to use `--default`.

---

## 🛠️ Pipeline Architecture: `main.py` vs `kaggle_pipeline.py`

### 1. `main.py` (The Engine)
This is the core script. It assumes:
*   All libraries (`torch`, `tqdm`, etc.) are installed.
*   The dataset is located exactly at `../Dataset/`.
*   You have write permissions in the current directory.

**What happens if you run `main.py` directly on Kaggle?**
❌ **It will likely fail.**
*   Kaggle stores data in `/kaggle/input`, but `main.py` looks in `./Dataset`.
*   Kaggle's default environment might miss specific dependencies.
*   You cannot write to `/kaggle/input`, so outputting results there crashes the script.

### 2. `kaggle_pipeline.py` (The Mechanic)
This is a wrapper designed specifically for Kaggle. It:
1.  **Installs Dependencies**: Automatically runs `pip install` for missing packages.
2.  **Fixes Paths**: Creates symbolic links so `/kaggle/input` looks like `./Dataset` to the model.
3.  **Manages Output**: Ensures results are saved to `/kaggle/working` (the only writable spot).
4.  **Runs Main**: Internally imports and executes `main.py`.

---

## 📊 How It Works

### The Core Idea: "Velocity"
The BDH model maintains a persistent memory state (the **ρ-matrix**) that evolves as it reads.
1.  **Prime**: We read the character's backstory. The ρ-matrix forms a stable representation of who they are.
2.  **Scan**: We read the novel in 2048-token chunks.
3.  **Detect**: We measure **Velocity** (`||ρ_t - ρ_{t-1}||₂`) - how much the memory state changes.
    *   **Consistent Text**: Low velocity (matches existing patterns).
    *   **Contradiction**: High velocity spike (forces rapid memory re-organization).

### Architecture
*   **Tokenization**: Byte-level (UTF-8 bytes), vocab size = 256.
*   **Recurrence**: Hebbian update rule `ρ_t = 0.99 * ρ_{t-1} + (x ⊗ v)`.
*   **Optimization**: Uses BDH-GPU linear attention for speed.

---

## 📂 Project Structure
```
bdh_project/
├── config/             # Model configurations (4-layer vs 6-layer)
├── model/              # Recurrent BDH implementation (Stateful)
├── metrics/            # Velocity tracking & Threshold calibration
├── inference/          # High-level wrapper (Prime -> Scan -> Detect)
├── utils/              # Data loading & Byte tokenizer
├── main.py             # CLI Entry point
└── kaggle_pipeline.py  # Kaggle automation script
```

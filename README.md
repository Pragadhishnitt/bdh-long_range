# BDH Track B: Narrative Consistency Classification

**Hebbian Memory for Long-Context Narrative Understanding**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a **Baby Dragon Hatchling (BDH)** recurrent architecture with Hebbian memory for detecting narrative inconsistencies in long novels (100K+ tokens). Our approach achieves competitive accuracy using stateful ρ-matrix accumulation and monosemantic masking.

> **Research Report**: See [`bdh_project/research_report.md`](bdh_project/research_report.md) for detailed methodology and ablation study.

---

## 📊 Key Results

| Configuration | Validation Accuracy | Training | Parameters | Time |
|---------------|---------------------|----------|------------|------|
| **BDH Baseline** | 61.25% ± 5.45% | Pre-trained | 25.3M | ~4 min |
| **BDH + LTC** | 61-62% | Pre-trained | 25.3M | ~4 min |
| **BDH Combined** | 60-63% | Pre-trained | 25.3M | ~52 min |
| Mini-GPT2 (BPE) | 63.33% | Fine-tuned | 25M | N/A |
| Llama 3.1 (8B) | 54.5% | Zero-shot | ~8B | N/A |
| Claude 2.0 | 33.0% | Zero-shot | ~100B+ | N/A |

**Key Innovation**: Stateless → Stateful processing enables unlimited context accumulation.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Pragadhishnitt/bdh-long_range.git
cd bdh-long_range

# Install dependencies
pip install -r requirements.txt
```

### Core Training Script

```bash
# Navigate to project directory
cd bdh_project

# Run optimal configuration (Baseline with K-fold)
python main.py --ablation baseline --improvise
```

**Expected Output**:
- Validation Accuracy: ~61-62%
- Processing Time: ~4-5 minutes
- Output: `outputs/results.csv` with predictions

---

## 🔬 Experimental Modes

### 1. Baseline Mode (Recommended)
**Best balance of accuracy and speed**

```bash
python main.py --ablation baseline --improvise
```

- **Accuracy**: 61.25% ± 5.45%
- **Time**: ~4 minutes
- **Method**: Direct state comparison with K-fold CV

### 2. LTC Mode (Adaptive Damping)
**Liquid Time Constants for adaptive memory retention**

```bash
python main.py --ablation ltc --improvise
```

- **Accuracy**: 61-62%
- **Time**: ~4 minutes
- **Method**: Adaptive damping based on input surprise

### 3. Combined Mode (Multi-Component)
**LTC + Monosemantic Masking + Multi-Scale**

```bash
# Default (MAX aggregation)
python main.py --ablation combined --improvise

# Try different aggregations
python main.py --ablation combined --improvise --multi-scale-agg mean
python main.py --ablation combined --improvise --multi-scale-agg min
```

- **Accuracy**: 60-63% (varies by aggregation)
- **Time**: ~52 minutes
- **Method**: Multi-checkpoint velocity with semantic masking

### 4. Custom K-Fold Splits

```bash
# Use 10 folds instead of default 4
python main.py --ablation baseline --improvise --folds 10
```

## 📂 Project Structure

```
bdh-long_range/
├── bdh/                           # Original BDH architecture (reference implementation)
│   ├── bdh.py                     # Core BDH model (stateless, for generation)
│   └── config.py                  # Original configuration
│
├── bdh_project/                   # Main classification project
│   ├── main.py                    # Entry point for classification
│   ├── config/                    # Model configurations (4-layer, 6-layer)
│   ├── model/
│   │   └── bdh_recurrent.py      # Modified BDH with stateful ρ-matrix
│   ├── inference/
│   │   └── model_wrapper.py      # State management & velocity computation
│   ├── utils/
│   │   ├── monosemantic.py       # Semantic neuron masking
│   │   └── data_loader.py        # Dataset utilities
│   ├── bdh graph/                 # Alternative sparse graph implementation
│   │   ├── train.py              # Graph BDH training script
│   │   └── classify.py           # σ-state based classification
│   ├── outputs/                   # Results and checkpoints
│   ├── research_report.md         # Detailed methodology & ablation study
│   └── WALKTHROUGH.md             # Implementation guide
│
├── Dataset/                       # Novel text files and CSV data
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `bdh/` | Original BDH architecture from Pathway (stateless, autoregressive) |
| `bdh_project/` | Our classification implementation with stateful Hebbian memory |
| `bdh_project/bdh graph/` | Alternative sparse graph topology experiment (see Section 4 in research report) |
| `Dataset/` | Novel texts (*In Search of the Castaways*, *The Count of Monte Cristo*) |

---


## 🛠️ Command-Line Arguments

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ablation` | str | `baseline` | Ablation mode: `baseline`, `ltc`, `combined` |
| `--improvise` | flag | False | Enable K-fold cross-validation |
| `--folds` | int | 4 | Number of K-fold splits |
| `--multi-scale-agg` | str | `max` | Aggregation for multi-scale: `max`, `min`, `mean` |
| `--small` | flag | False | Use 4-layer model (faster, less accurate) |

### Advanced Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `cached` | Processing mode: `cached`, `streaming` |
| `--metric` | str | `cosine` | Distance metric: `cosine`, `l2` |
| `--damping` | float | 0.99 | Hebbian damping factor |
| `--chunk-size` | int | 2048 | Tokens per chunk |

### Example Commands

```bash
# Fast iteration (baseline)
python main.py --ablation baseline --improvise

# High accuracy (combined with mean aggregation)
python main.py --ablation combined --improvise --multi-scale-agg mean

# Custom configuration
python main.py --ablation ltc --improvise --folds 10 --damping 0.95
```

---

## 📈 Methodology

### Stateful Hebbian Memory

Our key innovation is converting BDH from **stateless** (original) to **stateful** (ours):

**Original (`bdh/bdh.py`)**: Each chunk processed independently
```python
def forward(self, idx):
    # State reset every chunk - no memory accumulation
    return logits
```

**Our Modification (`inference/model_wrapper.py`)**: Continuous state accumulation
```python
def compute_novel_state(self, novel_path):
    state = self.model.reset_state()  # Initialize once
    for chunk in novel_chunks:
        _, state, _ = self.model(chunk, state=state)  # Accumulate
    return state  # Contains entire novel context
```

### Classification Metric

**State Velocity**: Cosine distance between Hebbian states

```python
velocity = 1 - cosine_similarity(ρ_backstory, ρ_novel)

# Classification
if velocity < threshold:
    prediction = "Consistent"
else:
    prediction = "Contradict"
```

---

## 🔍 Ablation Study Summary

| Experiment | Hypothesis | Result | Key Learning |
|------------|-----------|--------|--------------|
| **Baseline** | Direct state comparison | 61.25% | Captures global patterns |
| **RCP** | Bidirectional priming | ~50% | Asymmetric contexts don't work |
| **LTC** | Adaptive damping | 61-62% | Validates default λ=0.99 |
| **Multi-Scale** | Localized contradictions | 60-63% | Single-scale superior |
| **Combined** | All enhancements | 60-63% | Masking helps, multi-scale doesn't |

**See [`bdh_project/research_report.md`](bdh_project/research_report.md) for detailed analysis.**

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `outputs/results.csv` | Test predictions (id, prediction) |
| `outputs/results_detailed.csv` | With velocity scores |
| `outputs/checkpoints/*.json` | Calibration checkpoints |
| `outputs/plots/*.png` | Velocity distributions |

---

## 🧪 Reproducibility

### System Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: Optional (GPU recommended)
- **RAM**: 16GB minimum
- **Storage**: 5GB for cached states

### Expected Runtime (Tesla P100)
- **Baseline**: ~4 minutes
- **LTC**: ~4 minutes
- **Combined**: ~52 minutes
- **Streaming**: ~7.5 hours (not recommended)

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Damping (λ) | 0.99 | Optimal for novel-length texts |
| Chunk Size | 2048 | Balance context/memory |
| Distance Metric | Cosine | Normalized, magnitude-invariant |
| K-Fold Splits | 4 | Robust threshold estimation |

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{bdh_narrative_2026,
  title={Hebbian Memory for Narrative Consistency Classification},
  author={BDH Research Team},
  year={2026},
  note={Track B: Narrative Consistency Classification}
}
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

This is a competition submission. For questions or issues, please open a GitHub issue.

---

## 📞 Contact

- **GitHub**: [Pragadhishnitt/bdh-long_range](https://github.com/Pragadhishnitt/bdh-long_range)
- **Competition**: Track B - Narrative Consistency Classification

---

## 🙏 Acknowledgments

- Original BDH architecture from the BDH research team
- Competition organizers for the challenging task
- PyTorch and HuggingFace communities

---

**Last Updated**: 2026-01-11  
**Version**: 1.0.0

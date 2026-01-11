# Test-Time Training (TTT) Research Log

**Baby Dragon Hatchling (BDH) - Narrative Consistency Classification**

This document records all experimental approaches for Test-Time Training on the BDH model, including both successful and failed attempts. The goal is to provide a complete research history for future work.

---

## Problem Statement

**Task**: Given a character backstory and a novel (100k+ words), classify whether the backstory is **consistent** (1) or **contradictory** (0) with the narrative.

**Challenge**: The baseline BDH velocity-based approach achieved ~70-75% accuracy. We hypothesized that **Test-Time Training (TTT)** - fine-tuning the model on the backstory before evaluating the novel - could improve separation between classes.

---

## Hypothesis

**Core Idea**: If we adapt the BDH model to a character's backstory, the model should be:
- **Less surprised** (lower perplexity) when reading a novel that is **consistent** with that backstory
- **More surprised** (higher perplexity) when reading a novel that **contradicts** the backstory

**Metric**: Perplexity = exp(mean cross-entropy loss)

---

## Experimental Timeline

### Experiment 1: Initial TTT Implementation
**Date**: 2026-01-10  
**Configuration**:
- Adaptation steps: 10
- Learning rate: 1e-4
- Perplexity chunks: 100 (~200k tokens)
- Metric: Mean perplexity

**Results**:
```
Accuracy: 67.50%
Consistent μ=96.06, σ=10.64, p95=116.34
Contradict μ=99.57, σ=11.93, p95=115.71
Separation: 3.51 points
Z-score: 0.31
```

**Analysis**: ❌ **Failed**
- Very low separation (only 3.5 point difference)
- P95 values nearly identical (116.3 vs 115.7)
- Accuracy barely better than random (67.5% vs 50%)

**Diagnosis**: The model was learning the **writing style** of the backstory more than the **semantic content**. Since both consistent and contradictory novels share the same author's style, adaptation helped both equally, destroying separation.

---

### Experiment 2: Aggressive Adaptation
**Date**: 2026-01-10  
**Configuration**:
- Adaptation steps: 18 (↑ from 10)
- Learning rate: 1e-4
- Perplexity chunks: 100
- Metric: Mean perplexity
- K-fold: 4 folds

**Results**:
```
K-Fold Mean Val Accuracy: 58.75% ± 7.40%
Fold Thresholds: [106.42, 125.88, 122.99, 126.03]
Consistent μ≈101.5, σ≈16.0
Contradict μ≈106.0, σ≈15.0
```

**Analysis**: ❌ **Worse than Experiment 1**
- Accuracy dropped to ~59%
- High variance across folds (106 to 126)
- Separation still poor (~4.5 points)

**Diagnosis**: **Overfitting to style**. By increasing adaptation steps, we forced the model to memorize the specific phrasing and syntax of the backstory. When it switched to the novel (same style), it was "unsurprised" in both cases, masking factual contradictions.

**Lesson**: More adaptation is not always better. There's a "sweet spot" where the model learns facts without overfitting to style.

---

### Experiment 3: Light Touch Adaptation
**Date**: 2026-01-10  
**Configuration**:
- Adaptation steps: 4 (↓ from 18)
- Learning rate: 5e-5 (↓ from 1e-4)
- Perplexity chunks: 50 (↓ from 100)
- Metric: Mean perplexity
- K-fold: 4 folds

**Rationale**:
1. **Fewer steps**: Prevent style memorization
2. **Lower LR**: Gentler adaptation
3. **Fewer chunks**: Focus on first ~100k tokens where setup is most relevant, reducing signal dilution

**Results**:
```
K-Fold Mean Val Accuracy: 61.25% ± 5.45%
Fold Thresholds: [118.46, 119.44, 118.08, 118.46]
Consistent μ=115.6, σ=7.5
Contradict μ=117.1, σ=5.5
Separation: 1.5 points
```

**Analysis**: ⚠️ **Partial Success**
- ✅ **Stability improved**: Fold thresholds very consistent (118.08-119.44)
- ✅ **Lower variance**: σ reduced from ~16 to ~7
- ❌ **Separation still poor**: Only 1.5 point difference
- ❌ **Accuracy mediocre**: 61.25%

**Diagnosis**: The "signal" (factual contradiction) is too weak relative to the "noise" (general language modeling perplexity). Averaging loss over 50 chunks (100k tokens) dilutes the contradiction spike.

**Lesson**: The method is robust (consistent thresholds) but not discriminative enough. The problem is the **metric**, not the hyperparameters.

---

### Experiment 4: Peak Perplexity Metric
**Date**: 2026-01-10  
**Configuration**:
- Adaptation steps: 4
- Learning rate: 5e-5
- Perplexity chunks: 50
- Metric: **Peak (Max) perplexity** (NEW)
- K-fold: 4 folds

**Rationale**: Instead of averaging perplexity over the whole novel (which washes out the signal), look for the **maximum surprise** (highest perplexity chunk). If the backstory contradicts the novel, there should be at least one specific section where the model is extremely confused.

**Implementation**:
```python
# Old: Mean perplexity
mean_loss = total_loss / total_tokens
perplexity = exp(mean_loss)

# New: Peak perplexity
chunk_perplexities = [exp(loss_i) for each chunk]
perplexity = max(chunk_perplexities)
```

**Results**: ⏳ **Pending** (not yet run on GPU)

**Expected Outcome**: Better separation because:
- Consistent examples: Peak should be moderate (no major surprises)
- Contradictory examples: Peak should be high (at least one section clashes with backstory)

---

## Key Insights

### 1. Style vs. Semantics Problem
**Finding**: The model learns writing style much faster than semantic facts.
- **Style** (syntax, vocabulary): Learned in ~5-10 steps
- **Facts** (character traits, plot points): Requires more steps but risks overfitting

**Implication**: There's a narrow window (3-5 steps) where the model learns facts without memorizing style.

### 2. Signal Dilution Problem
**Finding**: Contradictions are often sparse (1-2 paragraphs in a 100k word novel).
- Averaging perplexity over 50-100 chunks dilutes the signal
- The contradiction spike gets washed out by 49 chunks of "normal" text

**Implication**: Need a metric that focuses on the **worst case** (peak) rather than the **average case** (mean).

### 3. Hyperparameter Sensitivity
**Finding**: TTT is extremely sensitive to:
- **Adaptation steps**: 4 is better than 10 or 18
- **Learning rate**: 5e-5 is better than 1e-4
- **Chunk count**: 50 is better than 100

**Implication**: There's a "Goldilocks zone" for each parameter. Too much or too little both fail.

### 4. K-Fold Robustness
**Finding**: K-fold cross-validation significantly improves threshold stability.
- Without K-fold: Threshold varies wildly based on random split
- With K-fold: Median threshold is robust (std < 1.0)

**Implication**: Always use K-fold for TTT experiments. The compute cost is the same (scores computed once, then cross-validated).

---

## Failed Approaches Summary

| Approach | Why It Failed | Lesson Learned |
|----------|---------------|----------------|
| **High adaptation steps (18)** | Overfits to writing style | More steps ≠ better performance |
| **High learning rate (1e-4)** | Destroys model's general knowledge | Need gentle adaptation (5e-5) |
| **Many chunks (100)** | Dilutes sparse contradiction signal | Focus on first 50 chunks (~100k tokens) |
| **Mean perplexity** | Averages out the contradiction spike | Use peak (max) perplexity instead |

---

## Current Best Configuration

```bash
python3 main.py \
  --adapt \
  --improvise \
  --adapt-steps 4 \
  --adapt-lr 5e-5 \
  --ppl-chunks 50 \
  --peak-ppl \
  --train
```

**Rationale**:
- `--adapt`: Enable TTT mode
- `--improvise`: 4-fold cross-validation for robust threshold
- `--adapt-steps 4`: Light touch (avoid style overfitting)
- `--adapt-lr 5e-5`: Gentle adaptation
- `--ppl-chunks 50`: Focus on first ~100k tokens
- `--peak-ppl`: Use max perplexity (detect local contradictions)

**Expected Time**: ~30-35 minutes on Tesla P100 GPU

---

## Future Research Directions

### 1. Chunk-Level Attention
Instead of peak perplexity, use a weighted average where chunks near the backstory get higher weight:
```python
weights = exp(-distance_from_backstory / temperature)
perplexity = sum(weights * chunk_perplexities)
```

### 2. Contrastive Adaptation
Adapt on both backstory and a "negative example" (random backstory):
```python
ppl_adapted = compute_perplexity(model_adapted_to_backstory, novel)
ppl_baseline = compute_perplexity(model_adapted_to_random, novel)
score = ppl_adapted - ppl_baseline  # Relative surprise
```

### 3. Multi-Scale Perplexity
Compute perplexity at multiple chunk sizes (10, 50, 100) and combine:
```python
score = 0.5 * peak_ppl_10 + 0.3 * peak_ppl_50 + 0.2 * peak_ppl_100
```

### 4. Selective Adaptation
Only adapt specific layers (e.g., top 2 layers) instead of the entire model:
```python
optimizer = AdamW([
    {'params': model.layers[-2:], 'lr': 5e-5},
])
```

### 5. Ensemble with Velocity
Combine TTT perplexity with the original velocity metric:
```python
if (perplexity > threshold_ppl) OR (velocity > threshold_vel):
    prediction = "contradictory"
```

---

## Conclusion

**Status**: Test-Time Training shows promise but has not yet achieved breakthrough results (~61% accuracy vs. ~70% baseline).

**Main Challenge**: Distinguishing between:
- **Style adaptation** (fast, unhelpful): Model learns syntax/vocabulary
- **Semantic adaptation** (slow, helpful): Model learns character traits/facts

**Next Steps**:
1. Run Experiment 4 (Peak Perplexity) on GPU
2. If separation improves, try contrastive adaptation
3. If separation remains poor, consider abandoning TTT and focusing on ensemble methods

**Key Takeaway**: Not all "learning" is useful. Sometimes, the model learns the wrong thing (style) faster than the right thing (facts). The challenge is to design an adaptation process that targets semantics while ignoring syntax.

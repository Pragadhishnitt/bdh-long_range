# BDH Architecture Metrics & Visualizations

This module demonstrates key properties of the BDH (Baby Dragon Hatchling) Graph architecture as described in the research paper.

## Quick Start

```bash
python run_metrics_analysis.py
```

This will generate comprehensive visualizations showing all BDH architecture properties.

## What Gets Analyzed

### 1. 📈 Long-Range Memory (Infinite Context)

**What it shows:** BDH's ability to maintain information over increasing context lengths.

**How it works:** 
- Tests the model at context lengths: 128, 256, 512, 1024, 2048 tokens
- Measures perplexity at each length
- Lower perplexity increase = better memory retention

**Output:** `metrics/long_range_memory.png`

**Key insight:** With damping = 0.99, the sigma state maintains information much longer than traditional models. Small perplexity increase over distance shows "infinite context" capability.

### 2. ⚡ Sparsity Analysis

**What it shows:** Efficiency through sparsity at multiple levels.

**Three types of sparsity:**

a) **Connection Sparsity** (Graph Topology)
   - BDH uses sparse random connections
   - Typical: 99.99%+ sparsity (only ~16k out of billions of possible connections)
   - Shows efficient information routing

b) **Activation Sparsity**
   - Percentage of neurons with zero activations
   - ReLU activations create natural sparsity
   - Typical: 60-80% sparsity
   - Enables efficient computation

c) **Synaptic State Sparsity**
   - Distribution of sigma (Hebbian state) values
   - Many near-zero values = selective memory
   - Only important connections strengthen

**Output:** `metrics/sparsity_analysis.png` (4 subplots)

### 3. 🧠 Hebbian Learning Dynamics

**What it shows:** How the model learns through Hebbian plasticity.

**Key visualizations:**

a) **Synaptic State Evolution**
   - Shows how sigma values change over time
   - Individual edge trajectories
   - Demonstrates adaptive memory

b) **Hebbian Update Magnitude**
   - Strength of learning signal
   - Shows when/how much the model learns
   - Damping factor controls decay

c) **Temporal Heatmap**
   - 2D view of sigma evolution
   - Shows patterns of strengthening/weakening

d) **Statistics Over Time**
   - Mean and std dev of sigma
   - Shows stability and diversity

**Output:** `metrics/hebbian_dynamics.png` (4 subplots)

**Key insight:** "Neurons that fire together, wire together" - the model strengthens connections based on co-activation patterns.

### 4. 🔗 Graph Topology Properties

**What it shows:** Structure of the neural graph.

**Metrics:**
- Degree distribution (how connected each neuron is)
- Average in/out degree
- Self-loop count
- Connection patterns

**Key insight:** Sparse random topology is fixed, but information routing adapts through sigma.

## Key Configuration

### Damping Rate: 0.99 (Critical!)

```python
sigma = (sigma + hebbian * Gs) * 0.99
```

**Why 0.99 instead of 0.95?**
- Higher damping = slower decay = longer memory
- 0.95: retains ~37% after 20 steps
- 0.99: retains ~82% after 20 steps
- 0.99: retains ~37% after 100 steps

**Trade-off:**
- Higher damping → better long-term memory
- Lower damping → faster adaptation to new patterns
- 0.99 is optimal for tasks requiring long-context understanding

## Understanding the Outputs

### Long-Range Memory Plot

Good performance looks like:
```
Context 128:  PPL = 30
Context 256:  PPL = 32
Context 512:  PPL = 35
Context 1024: PPL = 38
Context 2048: PPL = 42
```

**Interpretation**: Only 40% increase over 16x context length = excellent memory

### Sparsity Plots

**Graph Topology**: Scattered points show sparse connections
**Activation Timeline**: High sparsity (60-80%) throughout processing
**Sigma Distribution**: Long-tailed → most connections weak, few strong
**Degree Distribution**: Roughly uniform → balanced connectivity

### Hebbian Dynamics

**Evolution Plot**: Individual trajectories show learning
**Update Magnitude**: Non-zero throughout → continuous adaptation
**Heatmap**: Patterns show which connections strengthen together
**Statistics**: Stable mean, varying std → controlled plasticity

## Comparison to Traditional Models

| Property | Traditional Transformer | BDH Graph |
|----------|------------------------|-----------|
| Memory | Fixed context window | Infinite context via sigma |
| Sparsity | Dense attention | 99.99%+ connection sparsity |
| Learning | Gradient-only | Hebbian + Gradient |
| Routing | Learned attention | Fixed graph + adaptive sigma |
| Context scaling | O(n²) | O(n) with persistent sigma |

## Advanced Usage

### Custom Analysis

```python
from bdh_metrics import BDHMetricsAnalyzer

analyzer = BDHMetricsAnalyzer('best_model.pt')

# Individual analyses
analyzer.test_long_context_memory(text, 'output_dir')
analyzer.analyze_sparsity(text, 'output_dir')
analyzer.visualize_hebbian_dynamics(text, 'output_dir')
analyzer.analyze_graph_properties('output_dir')
```

### Custom Text

```python
with open('your_text.txt', 'r') as f:
    text = f.read()

analyzer.generate_full_report(text, 'custom_metrics')
```

## Key Papers/Concepts

1. **Hebbian Learning**: "Cells that fire together, wire together"
2. **Sparse Graphs**: Efficient information routing
3. **Persistent State**: Infinite context through recurrent sigma
4. **Damping**: Controls memory decay rate

## Troubleshooting

**Low sparsity?**
- Check ReLU activations are working
- Verify graph is actually sparse (should be 99.99%+)

**Poor long-range memory?**
- Increase damping (0.99 → 0.995)
- Check sigma is being maintained across forward passes

**Flat Hebbian dynamics?**
- Learning rate might be too low
- Check Gs (synaptic gain) values

## Files Generated

```
metrics/
├── long_range_memory.png       # Context length vs perplexity
├── sparsity_analysis.png        # 4 plots of sparsity
├── hebbian_dynamics.png         # 4 plots of learning
└── bdh_metrics_summary.json     # Numerical results
```

## Citation

If using BDH architecture or these metrics, please cite the relevant BDH paper.

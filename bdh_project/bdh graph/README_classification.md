# BDH Graph Model for Contradiction Classification

This code uses your trained BDH model (perplexity ~30 on Tiny Shakespeare) for classifying whether book backstories are contradicting or not.

## How It Works

1. **BDH Model as Feature Extractor**: Your trained BDH model learns rich representations through its Hebbian synaptic states (sigma). We use these as features for classification.

2. **Classification Head**: A small trainable neural network is added on top of the frozen BDH model to perform binary classification (contradictory vs. consistent).

3. **Book Context**: The model combines reference book context with the content to better detect contradictions.

## Quick Start

### Step 1: Train the Classifier

```bash
python bdh_classifier.py --mode train \
    --model best_model.pt \
    --train_csv Dataset/train.csv \
    --books Dataset/Books \
    --epochs 10 \
    --batch_size 16
```

This will:
- Load your trained BDH model
- Extract representations from training data
- Train a classification head
- Save to `bdh_classifier_head.pt`

### Step 2: Make Predictions

```bash
python bdh_classifier.py --mode predict \
    --model best_model.pt \
    --test_csv Dataset/test.csv \
    --books Dataset/Books
```

This will:
- Load the trained classifier
- Make predictions on test set
- Save results to `predictions.csv`

## Output Format

The `predictions.csv` file will contain:
- `id`: Sample ID from test set
- `prediction`: "contradictory" or "consistent"
- `confidence`: Model confidence (0-1)

## Model Architecture

```
Input Text
    ↓
BDH Model (frozen) → Sigma State (16384 dims)
    ↓
Linear(16384 → 512) + ReLU + Dropout
    ↓
Linear(512 → 128) + ReLU + Dropout
    ↓
Linear(128 → 2) → [consistent, contradictory]
```

## Key Features

1. **Transfer Learning**: Uses your pre-trained BDH model's knowledge
2. **Byte-Level Encoding**: Works with any text, robust to typos/special chars
3. **Contextual**: Combines book reference with content for better detection
4. **Hebbian Memory**: Leverages BDH's unique synaptic state for representation

## Customization

You can adjust:
- `max_length` in `get_representation()` - controls input length (default: 512)
- Classification head architecture in `BDHClassifier.__init__()`
- Learning rate, epochs, batch size in training
- Book context length (currently first 2000 chars)

## Expected Results

With a well-trained BDH model (PPL ~30), you should expect:
- Training accuracy: 70-85%
- Test accuracy: 65-80%
- Better performance with more training epochs

## Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch_size 8`
- Reduce max_length in code to 256 or 128

**Low Accuracy?**
- Train for more epochs: `--epochs 20`
- Ensure book files match the dataset book names
- Check if labels in CSV are correct format

**Slow Training?**
- Use GPU if available (automatically detected)
- Reduce dataset size for quick experiments

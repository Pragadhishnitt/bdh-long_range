"""
Single-Book BDH Classification

Simplified version that:
1. Uses only "In Search of the Castaways" (smaller book)
2. Trains and classifies based on that book alone
3. Includes metrics visualization for this specific book
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================================
# BDH Model (with 0.99 damping)
# ============================================================================

class BDHGraphModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.n = config.n_neurons
        
        edge_index = torch.randint(0, self.n, (2, config.n_edges))
        self.register_buffer('edge_index', edge_index)
        
        self.Gx = nn.Parameter(torch.randn(config.n_edges) * 0.02)
        self.Gy = nn.Parameter(torch.randn(config.n_edges) * 0.02)
        self.Gs = nn.Parameter(torch.ones(config.n_edges))
        self.register_buffer('sigma', torch.zeros(config.n_edges))
        
        self.embedding = nn.Embedding(vocab_size, self.n)
        self.readout = nn.Linear(self.n, vocab_size)

    def forward(self, idx, targets=None, persistent_sigma=None):
        B, T = idx.shape
        X = self.embedding(idx)
        
        sigma = persistent_sigma if persistent_sigma is not None else torch.zeros(self.config.n_edges, device=idx.device)
        logits_list = []
        src, dst = self.edge_index[0], self.edge_index[1]

        for t in range(T):
            x_t = X[:, t, :]
            y_t_prev = torch.zeros_like(x_t)
            
            for _ in range(self.config.n_layers):
                A = torch.zeros_like(x_t)
                A.index_add_(1, dst, x_t[:, src] * sigma)
                
                hebbian = (y_t_prev[:, src] * x_t[:, dst]).mean(0)
                sigma = (sigma + hebbian * self.Gs) * 0.99  # 0.99 damping
                
                y_new = torch.zeros_like(x_t)
                y_new.index_add_(1, dst, F.relu(A[:, src]) * self.Gy)
                y_t_prev = y_new
                
                x_next = torch.zeros_like(x_t)
                x_next.index_add_(1, dst, y_new[:, src] * self.Gx)
                x_t = F.relu(x_next)
                
            logits_list.append(self.readout(x_t))

        logits = torch.stack(logits_list, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, sigma.detach()


class BDHGraphConfig:
    n_neurons = 4096
    n_edges = 16384
    n_layers = 4
    batch_size = 8
    block_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Single-Book Classifier
# ============================================================================

class SingleBookClassifier:
    """Classifier for a single book's contradiction detection."""
    
    def __init__(self, model_path, book_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.config = BDHGraphConfig()
        self.vocab_size = 256
        
        # Load BDH model
        print(f"Loading BDH model from {model_path}...")
        self.model = BDHGraphModel(self.config, self.vocab_size).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load book
        print(f"Loading book from {book_path}...")
        with open(book_path, 'r', encoding='utf-8') as f:
            self.book_text = f.read()
        print(f"✓ Loaded {len(self.book_text):,} characters")
        
        # Get book representation (first 5000 chars as reference)
        self.book_context = self.book_text[:5000]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.n_edges, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        ).to(self.device)
        
        print(f"✓ Classifier initialized on {self.device}")
    
    def encode(self, text):
        return list(text.encode('utf-8'))
    
    def get_representation(self, text, max_length=512):
        """Extract BDH sigma representation."""
        tokens = self.encode(text)[:max_length]
        if len(tokens) < max_length:
            tokens = tokens + [0] * (max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, sigma = self.model(tokens)
        return sigma
    
    def classify(self, text):
        """Classify with book context."""
        full_text = self.book_context[:1000] + "\n\n" + text
        sigma = self.get_representation(full_text)
        
        logits = self.classifier(sigma.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()
        
        return pred, conf


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_single_book_classifier(train_csv, test_csv, model_path, book_path, 
                                 book_name="In Search of the Castaways",
                                 epochs=15, batch_size=16):
    """Train classifier on single book data."""
    
    print("\n" + "="*70)
    print(f"SINGLE-BOOK CLASSIFICATION: {book_name}")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    # Filter for this book only
    df_train = df_train[df_train['book_name'] == book_name].reset_index(drop=True)
    df_test = df_test[df_test['book_name'] == book_name].reset_index(drop=True)
    
    print(f"✓ Training samples: {len(df_train)}")
    print(f"✓ Test samples: {len(df_test)}")
    
    if len(df_train) == 0:
        print(f"\n❌ No training data found for '{book_name}'")
        return None
    
    # Initialize classifier
    classifier = SingleBookClassifier(model_path, book_path)
    
    # Freeze BDH model
    for param in classifier.model.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(classifier.classifier.parameters(), lr=1e-3)
    
    # Prepare training data
    X_train, y_train = [], []
    print("\nPreparing training data...")
    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        X_train.append(str(row['content']))
        label = 1 if str(row['label']).startswith('contra') else 0
        y_train.append(label)
    
    # Class distribution
    n_contradictory = sum(y_train)
    n_consistent = len(y_train) - n_contradictory
    print(f"\nClass distribution:")
    print(f"  Consistent: {n_consistent} ({n_consistent/len(y_train)*100:.1f}%)")
    print(f"  Contradictory: {n_contradictory} ({n_contradictory/len(y_train)*100:.1f}%)")
    
    # Training
    print(f"\nTraining for {epochs} epochs...")
    classifier.classifier.train()
    
    train_losses = []
    train_accs = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        indices = np.random.permutation(len(X_train))
        
        for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_indices = indices[i:i+batch_size]
            
            sigmas = []
            labels = []
            for idx in batch_indices:
                sigma = classifier.get_representation(X_train[idx])
                sigmas.append(sigma)
                labels.append(y_train[idx])
            
            sigmas = torch.stack(sigmas)
            labels = torch.tensor(labels, dtype=torch.long).to(classifier.device)
            
            logits = classifier.classifier(sigmas)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        avg_loss = total_loss / max(1, len(X_train) / batch_size)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # Evaluation
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    if len(df_test) == 0:
        print("No test data available")
        return classifier, train_losses, train_accs, [], []
    
    classifier.classifier.eval()
    
    X_test, y_test = [], []
    for _, row in df_test.iterrows():
        X_test.append(str(row['content']))
        label = 1 if str(row['label']).startswith('contra') else 0
        y_test.append(label)
    
    # Test predictions
    predictions = []
    confidences = []
    
    print("\nMaking predictions on test set...")
    for text in tqdm(X_test):
        pred, conf = classifier.classify(text)
        predictions.append(pred)
        confidences.append(conf)
    
    predictions = np.array(predictions)
    y_test = np.array(y_test)
    
    # Metrics
    test_acc = (predictions == y_test).mean()
    print(f"\n✓ Test Accuracy: {test_acc:.4f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, predictions)
    
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Cons  Contra")
    print(f"Actual Cons    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"       Contra  {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                                target_names=['Consistent', 'Contradictory'],
                                digits=3))
    
    # Save model
    torch.save({
        'classifier_state': classifier.classifier.state_dict(),
        'config': classifier.config.__dict__,
        'book_name': book_name
    }, 'single_book_classifier.pt')
    print("\n✓ Model saved to 'single_book_classifier.pt'")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, linewidth=2, label='Train')
    axes[1].axhline(y=test_acc, color='r', linestyle='--', label=f'Test: {test_acc:.3f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Progress')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('single_book_training.png', dpi=150)
    print("✓ Training plot saved to 'single_book_training.png'")
    
    return classifier, train_losses, train_accs, y_test, predictions


# ============================================================================
# Book-Specific Metrics
# ============================================================================

def analyze_book_metrics(model_path, book_path, save_dir='single_book_metrics'):
    """Analyze BDH metrics specific to this book."""
    
    from bdh_metrics import BDHMetricsAnalyzer
    
    print("\n" + "="*70)
    print("BOOK-SPECIFIC BDH METRICS")
    print("="*70)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load book
    with open(book_path, 'r', encoding='utf-8') as f:
        book_text = f.read()[:10000]  # First 10k chars
    
    analyzer = BDHMetricsAnalyzer(model_path)
    results = analyzer.generate_full_report(book_text, save_dir)
    
    print(f"\n✓ Book-specific metrics saved to {save_dir}/")
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Single-Book BDH Classification')
    parser.add_argument('--model', default='best_model.pt', help='BDH model path')
    parser.add_argument('--book', default='Dataset/Books/In search of the castaways.txt',
                       help='Book text file')
    parser.add_argument('--book_name', default='In Search of the Castaways',
                       help='Book name in CSV')
    parser.add_argument('--train_csv', default='Dataset/train.csv', help='Training CSV')
    parser.add_argument('--test_csv', default='Dataset/test.csv', help='Test CSV')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--mode', choices=['train', 'metrics', 'both'], default='both',
                       help='Mode: train, metrics, or both')
    
    args = parser.parse_args()
    
    if not Path(args.book).exists():
        print(f"❌ Book not found: {args.book}")
        exit(1)
    
    if args.mode in ['train', 'both']:
        # Train and evaluate
        classifier, train_losses, train_accs, y_test, predictions = train_single_book_classifier(
            args.train_csv,
            args.test_csv,
            args.model,
            args.book,
            args.book_name,
            args.epochs,
            args.batch_size
        )
    
    if args.mode in ['metrics', 'both']:
        # Analyze book-specific metrics
        metrics = analyze_book_metrics(args.model, args.book)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)

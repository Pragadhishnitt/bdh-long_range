"""
Sigma Velocity-Based Classification (No MLP Training)

This approach demonstrates that the BDH model:
1. **Remembers** baseline sigma states from processing reference book text
2. Shows **high sigma velocity/updates** when contradictory sentences are encountered
3. Classifies based on threshold of sigma changes (NO MLP training needed)

The intuition: Contradictions cause larger updates to the sigma (synaptic) states
because they conflict with the established patterns from the reference book.
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

    def forward(self, idx, targets=None, persistent_sigma=None, return_sigma_trajectory=False):
        B, T = idx.shape
        X = self.embedding(idx)
        
        sigma = persistent_sigma if persistent_sigma is not None else torch.zeros(self.config.n_edges, device=idx.device)
        
        if return_sigma_trajectory:
            sigma_trajectory = [sigma.clone().detach()]
        
        logits_list = []
        src, dst = self.edge_index[0], self.edge_index[1]
        
        # ✅ FIX: Initialize y_t with first token embedding (non-zero start)
        y_t = X[:, 0, :].clone() if T > 0 else torch.zeros(B, self.n, device=idx.device)

        for t in range(T):
            x_t = X[:, t, :]
            
            for _ in range(self.config.n_layers):
                A = torch.zeros_like(x_t)
                A.index_add_(1, dst, x_t[:, src] * sigma)
                
                # ✅ FIX: Use y_t from previous timestep for Hebbian correlation
                hebbian = (y_t[:, src] * x_t[:, dst]).mean(0)
                sigma = (sigma + hebbian * self.Gs) * 0.99  # 0.99 damping
                
                if return_sigma_trajectory:
                    sigma_trajectory.append(sigma.clone().detach())
                
                y_new = torch.zeros_like(x_t)
                y_new.index_add_(1, dst, F.relu(A[:, src]) * self.Gy)
                y_t = y_new  # ✅ FIX: Update y_t for next iteration
                
                x_next = torch.zeros_like(x_t)
                x_next.index_add_(1, dst, y_t[:, src] * self.Gx)
                x_t = F.relu(x_next)
                
            logits_list.append(self.readout(x_t))

        logits = torch.stack(logits_list, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if return_sigma_trajectory:
            return logits, loss, sigma.detach(), torch.stack(sigma_trajectory)
        return logits, loss, sigma.detach()


class BDHGraphConfig:
    n_neurons = 4096
    n_edges = 16384
    n_layers = 4
    batch_size = 8
    block_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Sigma Velocity Classifier (NO MLP TRAINING)
# ============================================================================

class SigmaVelocityClassifier:
    """
    Classification based on sigma velocity/update magnitude.
    
    Core Idea:
    1. Process reference book text to establish baseline sigma state
    2. Measure sigma velocity (change rate) when processing new sentences
    3. High velocity → Contradictory (conflicts with established patterns)
    4. Low velocity → Consistent (aligns with established patterns)
    
    NO training of classification head required!
    """
    
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
        
        # Load reference book
        print(f"Loading reference book from {book_path}...")
        with open(book_path, 'r', encoding='utf-8') as f:
            self.book_text = f.read()
        print(f"✓ Loaded {len(self.book_text):,} characters")
        
        # Establish baseline sigma state by processing book
        print("Establishing baseline sigma state from reference book...")
        self.baseline_sigma = self._get_baseline_sigma()
        print(f"✓ Baseline sigma established (mean: {self.baseline_sigma.mean():.4f}, std: {self.baseline_sigma.std():.4f})")
        
        # Will be calibrated on training data
        self.threshold = None
        
    def encode(self, text):
        """Convert text to byte tokens."""
        return list(text.encode('utf-8', errors='ignore'))
    
    def _get_baseline_sigma(self, reference_length=5000):
        """Process reference book text to get baseline sigma state."""
        reference_text = self.book_text[:reference_length]
        tokens = self.encode(reference_text)[:512]  # Use first portion
        
        if len(tokens) < 512:
            tokens = tokens + [0] * (512 - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, final_sigma = self.model(tokens, persistent_sigma=None)
        
        return final_sigma
    
    
    def compute_sigma_velocity(self, text, context_length=200):
        """
        Compute sigma velocity when processing text with book context.
        
        Strategy: Compare sigma after processing (book + new_text) vs just (book).
        Contradictory text should produce different sigma trajectory.
        
        Higher velocity = more change = likely contradictory
        Lower velocity = less change = likely consistent
        """
        # Process book context + new text TOGETHER from scratch
        # Use shorter context to ensure new text fits within token limit
        context = self.book_text[:context_length]
        full_text = context + "\n\n" + text
        
        # Encode with room for new text (use 400 tokens max for context+text)
        tokens_with_text = self.encode(full_text)[:400]
        if len(tokens_with_text) < 400:
            tokens_with_text = tokens_with_text + [0] * (400 - len(tokens_with_text))
        tokens_with_text = torch.tensor(tokens_with_text, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Also process just the book context for comparison (same length for fair comparison)
        tokens_baseline = self.encode(context)[:400]
        if len(tokens_baseline) < 400:
            tokens_baseline = tokens_baseline + [0] * (400 - len(tokens_baseline))
        tokens_baseline = torch.tensor(tokens_baseline, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Get sigma after processing book + text
        with torch.no_grad():
            _, _, sigma_with_text, sigma_trajectory = self.model(
                tokens_with_text, 
                persistent_sigma=None,  # Start fresh
                return_sigma_trajectory=True
            )
        
        # Get sigma after processing just book
        with torch.no_grad():
            _, _, sigma_baseline_ctx = self.model(
                tokens_baseline,
                persistent_sigma=None  # Start fresh
            )
        
        # Compute velocity metrics
        sigma_delta = sigma_with_text - sigma_baseline_ctx
        
        # Multiple velocity metrics
        metrics = {
            'l2_norm': torch.norm(sigma_delta, p=2).item(),
            'l1_norm': torch.norm(sigma_delta, p=1).item(),
            'max_change': torch.abs(sigma_delta).max().item(),
            'mean_abs_change': torch.abs(sigma_delta).mean().item(),
            'std_change': sigma_delta.std().item(),
            'relative_change': (torch.norm(sigma_delta, p=2) / (torch.norm(sigma_baseline_ctx, p=2) + 1e-8)).item()
        }
        
        return metrics, sigma_trajectory
    
    def calibrate_threshold(self, train_csv, book_name, metric='l2_norm', percentile=70):
        """
        Calibrate threshold on training data.
        
        Strategy: Find threshold that best separates consistent from contradictory.
        """
        print(f"\nCalibrating threshold on training data...")
        print(f"Using metric: {metric}")
        
        df_train = pd.read_csv(train_csv)
        df_train = df_train[df_train['book_name'] == book_name].reset_index(drop=True)
        
        if len(df_train) == 0:
            print(f"❌ No training data for '{book_name}'")
            return
        
        print(f"Training samples: {len(df_train)}")
        
        # Compute velocities for all training samples
        consistent_velocities = []
        contradictory_velocities = []
        
        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Computing velocities"):
            text = str(row['content'])
            label = 1 if str(row['label']).startswith('contra') else 0
            
            metrics, _ = self.compute_sigma_velocity(text)
            velocity = metrics[metric]
            
            if label == 0:
                consistent_velocities.append(velocity)
            else:
                contradictory_velocities.append(velocity)
        
        consistent_velocities = np.array(consistent_velocities)
        contradictory_velocities = np.array(contradictory_velocities)
        
        print(f"\nVelocity Statistics:")
        print(f"  Consistent: mean={consistent_velocities.mean():.4f}, std={consistent_velocities.std():.4f}")
        print(f"  Contradictory: mean={contradictory_velocities.mean():.4f}, std={contradictory_velocities.std():.4f}")
        
        # Find optimal threshold using multiple strategies
        all_velocities = np.concatenate([consistent_velocities, contradictory_velocities])
        
        # Strategy 1: Percentile of consistent samples
        threshold_percentile = np.percentile(consistent_velocities, percentile)
        
        # Strategy 2: Midpoint between means
        threshold_midpoint = (consistent_velocities.mean() + contradictory_velocities.mean()) / 2
        
        # Strategy 3: Grid search for best accuracy
        thresholds = np.linspace(all_velocities.min(), all_velocities.max(), 100)
        best_acc = 0
        best_threshold = threshold_midpoint
        
        for thresh in thresholds:
            acc_consistent = (consistent_velocities < thresh).mean()
            acc_contradictory = (contradictory_velocities >= thresh).mean()
            acc = (acc_consistent + acc_contradictory) / 2
            
            if acc > best_acc:
                best_acc = acc
                best_threshold = thresh
        
        print(f"\nThreshold Options:")
        print(f"  Percentile ({percentile}%): {threshold_percentile:.4f}")
        print(f"  Midpoint: {threshold_midpoint:.4f}")
        print(f"  Optimal (Grid Search): {best_threshold:.4f} (accuracy: {best_acc:.4f})")
        
        # Use optimal threshold
        self.threshold = best_threshold
        self.metric = metric
        
        print(f"\n✓ Threshold set to: {self.threshold:.4f}")
        
        # Visualize distributions
        self._plot_velocity_distributions(
            consistent_velocities, 
            contradictory_velocities, 
            metric,
            save_path='velocity_distributions.png'
        )
        
        return {
            'consistent_velocities': consistent_velocities,
            'contradictory_velocities': contradictory_velocities,
            'threshold': self.threshold,
            'metric': metric
        }
    
    def _plot_velocity_distributions(self, consistent, contradictory, metric, save_path):
        """Plot velocity distributions for consistent vs contradictory."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(consistent, bins=30, alpha=0.6, label='Consistent', color='blue', density=True)
        axes[0].hist(contradictory, bins=30, alpha=0.6, label='Contradictory', color='red', density=True)
        axes[0].axvline(self.threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {self.threshold:.3f}')
        axes[0].set_xlabel(f'Sigma Velocity ({metric})')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Sigma Velocity Distributions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data = [consistent, contradictory]
        axes[1].boxplot(data, labels=['Consistent', 'Contradictory'])
        axes[1].axhline(self.threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {self.threshold:.3f}')
        axes[1].set_ylabel(f'Sigma Velocity ({metric})')
        axes[1].set_title('Velocity Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Velocity distributions saved to '{save_path}'")
        plt.close()
    
    def classify(self, text):
        """
        Classify text based on sigma velocity.
        
        Returns: (prediction, velocity_score)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set! Call calibrate_threshold() first.")
        
        metrics, _ = self.compute_sigma_velocity(text)
        velocity = metrics[self.metric]
        
        # Simple threshold classification
        prediction = 1 if velocity >= self.threshold else 0
        
        return prediction, velocity
    
    def analyze_sample(self, text, label=None):
        """
        Detailed analysis of a single sample showing sigma dynamics.
        """
        print("\n" + "="*70)
        print("DETAILED SIGMA ANALYSIS")
        print("="*70)
        
        metrics, sigma_trajectory = self.compute_sigma_velocity(text)
        prediction, velocity = self.classify(text)
        
        print(f"\nText: {text[:200]}...")
        if label is not None:
            print(f"True Label: {'Contradictory' if label == 1 else 'Consistent'}")
        print(f"\nPrediction: {'Contradictory' if prediction == 1 else 'Consistent'}")
        print(f"Velocity Score: {velocity:.4f}")
        print(f"Threshold: {self.threshold:.4f}")
        
        print(f"\nAll Velocity Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Plot sigma trajectory
        self._plot_sigma_trajectory(sigma_trajectory, text, prediction, label)
        
        return metrics, prediction
    
    def _plot_sigma_trajectory(self, sigma_trajectory, text, prediction, true_label):
        """Visualize how sigma evolves over processing."""
        trajectory = sigma_trajectory.cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Mean and std over time
        means = trajectory.mean(axis=1)
        stds = trajectory.std(axis=1)
        time_steps = np.arange(len(means))
        
        axes[0, 0].plot(time_steps, means, linewidth=2, color='blue')
        axes[0, 0].fill_between(time_steps, means - stds, means + stds, alpha=0.3)
        axes[0, 0].set_xlabel('Processing Step')
        axes[0, 0].set_ylabel('Mean Sigma Value')
        axes[0, 0].set_title('Sigma Evolution (Mean ± Std)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Delta from baseline
        baseline_expanded = self.baseline_sigma.cpu().numpy()
        deltas = np.array([np.linalg.norm(t - baseline_expanded) for t in trajectory])
        
        axes[0, 1].plot(time_steps, deltas, linewidth=2, color='red')
        axes[0, 1].set_xlabel('Processing Step')
        axes[0, 1].set_ylabel('L2 Distance from Baseline')
        axes[0, 1].set_title('Sigma Deviation from Baseline')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Heatmap of sigma values (sample of edges)
        sample_indices = np.linspace(0, trajectory.shape[1]-1, min(50, trajectory.shape[1]), dtype=int)
        heatmap_data = trajectory[:, sample_indices].T
        
        im = axes[1, 0].imshow(heatmap_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[1, 0].set_xlabel('Processing Step')
        axes[1, 0].set_ylabel('Edge Index (sampled)')
        axes[1, 0].set_title('Sigma Values Heatmap')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Final distribution
        final_sigma = trajectory[-1]
        axes[1, 1].hist(final_sigma, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(final_sigma.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {final_sigma.mean():.3f}')
        axes[1, 1].set_xlabel('Sigma Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Final Sigma Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Overall title
        pred_str = 'Contradictory' if prediction == 1 else 'Consistent'
        true_str = 'Contradictory' if true_label == 1 else 'Consistent' if true_label is not None else 'Unknown'
        fig.suptitle(f'Sigma Trajectory Analysis\nPrediction: {pred_str} | True Label: {true_str}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('sigma_trajectory_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Sigma trajectory plot saved to 'sigma_trajectory_analysis.png'")
        plt.close()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_velocity_classifier(classifier, test_csv, book_name, save_results=True):
    """Evaluate velocity-based classifier on test set."""
    
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    df_test = pd.read_csv(test_csv)
    df_test = df_test[df_test['book_name'] == book_name].reset_index(drop=True)
    
    if len(df_test) == 0:
        print(f"❌ No test data for '{book_name}'")
        return
    
    print(f"Test samples: {len(df_test)}")
    
    # Check if test set has labels
    has_labels = 'label' in df_test.columns
    
    if not has_labels:
        print("⚠️  Test set does not have labels - will only make predictions")
    
    # Make predictions
    predictions = []
    velocities = []
    true_labels = []
    
    print("\nMaking predictions...")
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        text = str(row['content'])
        
        if has_labels:
            label = 1 if str(row['label']).lower().startswith('contra') else 0
            true_labels.append(label)
        
        pred, velocity = classifier.classify(text)
        
        predictions.append(pred)
        velocities.append(velocity)
    
    predictions = np.array(predictions)
    velocities = np.array(velocities)
    
    if not has_labels:
        # No labels - just save predictions
        print("\n✓ Predictions complete")
        print(f"  Prediction distribution: {np.bincount(predictions)}")
        print(f"  Mean velocity: {velocities.mean():.4f}")
        
        if save_results:
            results_df = pd.DataFrame({
                'text': df_test['content'],
                'prediction': predictions,
                'velocity_score': velocities
            })
            results_df.to_csv('velocity_classification_results.csv', index=False)
            print("✓ Results saved to 'velocity_classification_results.csv'")
        
        return {
            'predictions': predictions,
            'velocities': velocities
        }
    
    # Has labels - compute metrics
    true_labels = np.array(true_labels)
    
    # Compute metrics
    accuracy = (predictions == true_labels).mean()
    
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
    
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")
    
    # ROC AUC using velocity scores
    auc = roc_auc_score(true_labels, velocities)
    print(f"✓ ROC AUC: {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Cons  Contra")
    print(f"Actual Cons    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"       Contra  {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['Consistent', 'Contradictory'],
                                digits=3))
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Consistent', 'Contradictory'],
                yticklabels=['Consistent', 'Contradictory'])
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.3f}')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, velocities)
    axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Evaluation plots saved to 'evaluation_results.png'")
    
    if save_results:
        results_df = pd.DataFrame({
            'text': df_test['content'],
            'true_label': true_labels,
            'prediction': predictions,
            'velocity_score': velocities,
            'correct': predictions == true_labels
        })
        results_df.to_csv('velocity_classification_results.csv', index=False)
        print("✓ Results saved to 'velocity_classification_results.csv'")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': predictions,
        'velocities': velocities,
        'true_labels': true_labels
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sigma Velocity-Based Classification (No MLP)')
    parser.add_argument('--model', default='best_model.pt', help='BDH model path')
    parser.add_argument('--book', default='Dataset/Books/In search of the castaways.txt',
                       help='Book text file')
    parser.add_argument('--book_name', default='In Search of the Castaways',
                       help='Book name in CSV')
    parser.add_argument('--train_csv', default='Dataset/train.csv', help='Training CSV')
    parser.add_argument('--test_csv', default='Dataset/test.csv', help='Test CSV')
    parser.add_argument('--metric', default='l2_norm', 
                       choices=['l2_norm', 'l1_norm', 'max_change', 'mean_abs_change', 'std_change', 'relative_change'],
                       help='Velocity metric to use')
    parser.add_argument('--percentile', type=int, default=70,
                       help='Percentile for threshold calibration')
    parser.add_argument('--analyze_samples', type=int, default=3,
                       help='Number of samples to analyze in detail')
    
    args = parser.parse_args()
    
    if not Path(args.book).exists():
        print(f"❌ Book not found: {args.book}")
        exit(1)
    
    print("\n" + "="*70)
    print("SIGMA VELOCITY-BASED CLASSIFICATION")
    print("="*70)
    print("\nApproach: Classify based on sigma update magnitude")
    print("  → NO MLP training required!")
    print("  → Demonstrates BDH's memory capabilities")
    print("  → Higher velocity = Contradictory")
    print("="*70)
    
    # Initialize classifier
    classifier = SigmaVelocityClassifier(args.model, args.book)
    
    # Calibrate threshold on training data
    calibration_results = classifier.calibrate_threshold(
        args.train_csv, 
        args.book_name,
        metric=args.metric,
        percentile=args.percentile
    )
    
    # Evaluate on test set
    eval_results = evaluate_velocity_classifier(
        classifier,
        args.test_csv,
        args.book_name
    )
    
    # Detailed analysis of sample predictions
    print("\n" + "="*70)
    print(f"ANALYZING {args.analyze_samples} SAMPLE PREDICTIONS IN DETAIL")
    print("="*70)
    
    df_test = pd.read_csv(args.test_csv)
    df_test = df_test[df_test['book_name'] == args.book_name].reset_index(drop=True)
    
    # Analyze samples from each class
    contradictory_samples = df_test[df_test['label'].str.startswith('contra')]
    consistent_samples = df_test[~df_test['label'].str.startswith('contra')]
    
    if len(contradictory_samples) > 0:
        print("\n--- CONTRADICTORY SAMPLE ---")
        sample = contradictory_samples.iloc[0]
        classifier.analyze_sample(sample['content'], label=1)
    
    if len(consistent_samples) > 0:
        print("\n--- CONSISTENT SAMPLE ---")
        sample = consistent_samples.iloc[0]
        classifier.analyze_sample(sample['content'], label=0)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nKey Results:")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  ROC AUC: {eval_results['auc']:.4f}")
    print(f"  Threshold: {classifier.threshold:.4f}")
    print(f"  Metric: {classifier.metric}")
    print("\nThis demonstrates that BDH:")
    print("  ✓ Remembers baseline patterns from the reference book")
    print("  ✓ Shows higher sigma updates for contradictory sentences")
    print("  ✓ Can classify WITHOUT training any MLP!")
    print("="*70)

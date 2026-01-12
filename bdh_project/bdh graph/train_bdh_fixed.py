"""
BDH Model Training Script with FIXED Hebbian Dynamics

This script trains the BDH model with the corrected forward pass:
- y_t maintained across timesteps (not reset to zero)
- y_t initialized with first token embedding
- Sigma accumulates meaningful state information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
import urllib.request

# ============================================================================
# Configuration
# ============================================================================

class BDHConfig:
    # Model architecture (matching original)
    n_neurons = 4096
    n_edges = 16384
    n_layers = 4
    vocab_size = 256  # Byte-level
    
    # Training hyperparameters
    batch_size = 8
    block_size = 128
    learning_rate = 1e-3
    max_iters = 10000
    eval_interval = 500
    eval_iters = 100
    
    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Target
    target_ppl = 30.0  # Stop when we reach this perplexity


# ============================================================================
# FIXED BDH Model
# ============================================================================

class BDHGraphModel(nn.Module):
    """BDH Model with CORRECTED Hebbian dynamics."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n = config.n_neurons
        
        # Fixed topology: sparse random connections
        edge_index = torch.randint(0, self.n, (2, config.n_edges))
        self.register_buffer('edge_index', edge_index)
        
        # Learnable parameters
        self.Gx = nn.Parameter(torch.randn(config.n_edges) * 0.02)
        self.Gy = nn.Parameter(torch.randn(config.n_edges) * 0.02)
        self.Gs = nn.Parameter(torch.ones(config.n_edges))
        
        # Input/Output
        self.embedding = nn.Embedding(config.vocab_size, self.n)
        self.readout = nn.Linear(self.n, config.vocab_size)
        
        print(f"✓ BDH Model initialized: {self.n} neurons, {config.n_edges} edges")
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        X = self.embedding(idx)  # (B, T, N)
        
        # Initialize sigma (synaptic state)
        sigma = torch.zeros(self.config.n_edges, device=idx.device)
        
        logits_list = []
        src, dst = self.edge_index[0], self.edge_index[1]
        
        # ✅ FIX: Initialize y_t ONCE with first token embedding
        y_t = X[:, 0, :].clone() if T > 0 else torch.zeros(B, self.n, device=idx.device)
        
        for t in range(T):
            x_t = X[:, t, :]
            
            for _ in range(self.config.n_layers):
                # State-based inference
                A = torch.zeros_like(x_t)
                A.index_add_(1, dst, x_t[:, src] * sigma)
                
                # ✅ FIX: Hebbian update uses y_t from previous timestep
                hebbian = (y_t[:, src] * x_t[:, dst]).mean(0)
                sigma = (sigma + hebbian * self.Gs) * 0.99  # Damping
                
                # Parameter-based inference
                y_new = torch.zeros_like(x_t)
                y_new.index_add_(1, dst, F.relu(A[:, src]) * self.Gy)
                y_t = y_new  # ✅ FIX: Update y_t for next iteration
                
                # Propagate to next layer
                x_next = torch.zeros_like(x_t)
                x_next.index_add_(1, dst, y_t[:, src] * self.Gx)
                x_t = F.relu(x_next)
            
            logits_list.append(self.readout(x_t))
        
        logits = torch.stack(logits_list, dim=1)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, sigma.detach()
    
    def get_param_count(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Data Loading
# ============================================================================

def get_batch(data, config):
    """Generate a batch of training/validation data."""
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+config.block_size].astype(np.int64)) for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """Estimate loss and perplexity on train and val sets."""
    model.eval()
    out = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(config.eval_iters):
            X, Y = get_batch(data, config)
            _, loss, _ = model(X, Y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    
    model.train()
    return out


# ============================================================================
# Training Loop
# ============================================================================

def train():
    print("="*70)
    print("RETRAINING BDH MODEL WITH FIXED HEBBIAN DYNAMICS")
    print("="*70)
    
    config = BDHConfig()
    
    # Download Tiny Shakespeare if needed
    data_path = Path('input.txt')
    if not data_path.exists():
        print("\nDownloading Tiny Shakespeare...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, 'input.txt')
        print("✓ Dataset downloaded")
    
    # Load data
    print("\nLoading dataset...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to bytes
    data = np.array(list(text.encode('utf-8')), dtype=np.uint8)
    
    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"✓ Dataset loaded: {len(data):,} bytes")
    print(f"  Training: {len(train_data):,} bytes")
    print(f"  Validation: {len(val_data):,} bytes")
    
    # Initialize model
    print(f"\nInitializing model on {config.device}...")
    model = BDHGraphModel(config).to(config.device)
    param_count = model.get_param_count()
    print(f"✓ Model has {param_count:,} trainable parameters")
    
    # Test forward pass produces non-zero sigma
    print("\nTesting forward pass...")
    X_test, Y_test = get_batch(train_data, config)
    with torch.no_grad():
        _, _, sigma_test = model(X_test[:1, :10])  # Small test
    print(f"✓ Test sigma: mean={sigma_test.mean():.6f}, max={sigma_test.max():.6f}")
    if sigma_test.abs().max() < 1e-10:
        print("❌ WARNING: Sigma is still zero! Fix didn't work.")
        return
    print("✓ Sigma is non-zero - Hebbian learning is functional!")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print(f"\nStarting training for up to {config.max_iters} iterations...")
    print(f"Target perplexity: {config.target_ppl:.1f}")
    print("="*70)
    
    model.train()
    best_val_loss = float('inf')
    start_time = time.time()
    
    for iter in range(config.max_iters):
        # Evaluate periodically
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            train_ppl = np.exp(losses['train'])
            val_ppl = np.exp(losses['val'])
            elapsed = time.time() - start_time
            
            print(f"\nStep {iter:5d} | Time: {elapsed:.1f}s")
            print(f"  Train: loss={losses['train']:.4f}, ppl={train_ppl:.2f}")
            print(f"  Val:   loss={losses['val']:.4f}, ppl={val_ppl:.2f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.__dict__,
                    'iter': iter,
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                    'train_ppl': train_ppl,
                    'val_ppl': val_ppl
                }, 'bdh_retrained.pt')
                print(f"  ✓ Saved checkpoint (val_ppl={val_ppl:.2f})")
            
            # Check if we reached target
            if val_ppl <= config.target_ppl:
                print(f"\n{'='*70}")
                print(f"🎉 REACHED TARGET PERPLEXITY: {val_ppl:.2f} <= {config.target_ppl:.2f}")
                print(f"{'='*70}")
                break
        
        # Training step
        X, Y = get_batch(train_data, config)
        logits, loss, sigma = model(X, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 100 steps
        if iter % 100 == 0 and iter > 0:
            print(f"  Step {iter}: loss={loss.item():.4f}", end='\r')
    
    # Final evaluation
    print(f"\n\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    final_losses = estimate_loss(model, train_data, val_data, config)
    final_train_ppl = np.exp(final_losses['train'])
    final_val_ppl = np.exp(final_losses['val'])
    
    print(f"\nFinal Results:")
    print(f"  Train PPL: {final_train_ppl:.2f}")
    print(f"  Val PPL: {final_val_ppl:.2f}")
    print(f"  Best Val PPL: {np.exp(best_val_loss):.2f}")
    print(f"\nModel saved to: bdh_retrained.pt")
    print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Test sigma one more time
    print(f"\nFinal sigma check:")
    with torch.no_grad():
        X_final, _ = get_batch(val_data, config)
        _, _, sigma_final = model(X_final[:1, :50])
    print(f"  Mean: {sigma_final.mean():.6f}")
    print(f"  Std: {sigma_final.std():.6f}")
    print(f"  Max: {sigma_final.max():.6f}")
    print(f"  Non-zero: {(sigma_final.abs() > 1e-10).sum()}/{sigma_final.numel()}")
    
    print(f"\n{'='*70}")
    print("✓ Ready for velocity-based classification!")
    print(f"{'='*70}")


if __name__ == "__main__":
    train()

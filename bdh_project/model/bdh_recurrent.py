"""
BDH Track B: Recurrent BDH Model Wrapper

Wraps the GPU-optimized BDH model with stateful processing mechanics
for tracking ρ-matrix velocity during novel scanning.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent bdh directory to path for imports
BDH_PATH = Path(__file__).parent.parent.parent / "bdh"
if str(BDH_PATH) not in sys.path:
    sys.path.insert(0, str(BDH_PATH))


@dataclass
class RecurrentState:
    """State maintained across chunks for recurrent processing."""
    rho_matrix: Optional[torch.Tensor] = None  # Accumulated ρ (Hebbian memory)
    prev_rho: Optional[torch.Tensor] = None    # Previous ρ for velocity computation
    position_offset: int = 0                    # Global position offset for RoPE
    layer_states: Optional[Dict[int, torch.Tensor]] = None  # Per-layer states if needed
    
    def detach(self):
        """Detach all tensors from computation graph."""
        if self.rho_matrix is not None:
            self.rho_matrix = self.rho_matrix.detach()
        if self.prev_rho is not None:
            self.prev_rho = self.prev_rho.detach()
        if self.layer_states is not None:
            self.layer_states = {
                k: v.detach() if v is not None else None 
                for k, v in self.layer_states.items()
            }
        return self
    
    def clone(self) -> 'RecurrentState':
        """Create a copy of the state."""
        return RecurrentState(
            rho_matrix=self.rho_matrix.clone() if self.rho_matrix is not None else None,
            prev_rho=self.prev_rho.clone() if self.prev_rho is not None else None,
            position_offset=self.position_offset,
            layer_states={k: v.clone() for k, v in self.layer_states.items()} if self.layer_states else None
        )
    
    def to_cpu(self) -> 'RecurrentState':
        """Move all tensors to CPU (for caching)."""
        if self.rho_matrix is not None:
            self.rho_matrix = self.rho_matrix.cpu()
        if self.prev_rho is not None:
            self.prev_rho = self.prev_rho.cpu()
        if self.layer_states is not None:
            self.layer_states = {
                k: v.cpu() if v is not None else None 
                for k, v in self.layer_states.items()
            }
        return self
    
    def to_device(self, device: torch.device) -> 'RecurrentState':
        """Move all tensors to specified device."""
        if self.rho_matrix is not None:
            self.rho_matrix = self.rho_matrix.to(device)
        if self.prev_rho is not None:
            self.prev_rho = self.prev_rho.to(device)
        if self.layer_states is not None:
            self.layer_states = {
                k: v.to(device) if v is not None else None 
                for k, v in self.layer_states.items()
            }
        return self


class RecurrentBDH(nn.Module):
    """
    Stateful wrapper around BDH for recurrent processing.
    
    Tracks the ρ-matrix (Hebbian memory) across chunks and computes
    velocity metrics for consistency detection.
    """
    
    def __init__(self, config, damping: float = 0.99, use_ltc: bool = False):
        """
        Initialize RecurrentBDH model.
        
        Args:
            config: Model configuration
            damping: Fixed damping factor for Hebbian updates (default: 0.99)
            use_ltc: If True, use Liquid Time Constants (input-dependent adaptive damping)
                     instead of fixed damping. LTC hypothesis: High-surprise inputs 
                     should trigger stronger retention.
        """
        super().__init__()
        self.config = config
        self.damping = damping
        self.use_ltc = use_ltc
        
        # Core dimensions
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
        
        # LTC: Input-dependent gating for adaptive damping
        # λ_t = σ(W_gate · x_t + b) where high surprise → stronger retention
        if use_ltc:
            self.damping_gate = nn.Linear(config.n_embd, 1)
            nn.init.zeros_(self.damping_gate.bias)  # Start near 0.5 sigmoid output
        
        # Build BDH components (similar to original bdh.py but with state tracking)
        self._build_model()
        
        # RoPE frequencies
        self.register_buffer(
            'freqs',
            self._get_freqs(self.N, theta=2**16, dtype=torch.float32).view(1, 1, 1, self.N)
        )
    
    def _get_freqs(self, n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
        """Compute RoPE frequencies."""
        def quantize(t, q=2):
            return (t / q).floor() * q
        
        return (
            1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
            / (2 * math.pi)
        )
    
    def _build_model(self):
        """Build BDH model components."""
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = self.N
        
        # Embedding
        self.embed = nn.Embedding(C.vocab_size, D)
        
        # Shared weights across layers
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        
        # Layer norm and dropout
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(C.dropout)
        
        # Output projection
        self.lm_head = nn.Parameter(torch.zeros((D, C.vocab_size)).normal_(std=0.02))
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @staticmethod
    def _rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding."""
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)
    
    def _compute_attention(self, x_sparse: torch.Tensor, v: torch.Tensor, 
                            position_offset: int = 0) -> torch.Tensor:
        """Compute attention with RoPE."""
        B, nh, T, N = x_sparse.size()
        
        # RoPE phases
        r_phases = (
            torch.arange(position_offset, position_offset + T, 
                        device=x_sparse.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        # Apply RoPE to queries and keys (Q = K in BDH)
        QR = self._rope(r_phases, x_sparse)
        KR = QR
        
        # Compute attention scores (causal)
        scores = (QR @ KR.mT).tril(diagonal=-1)
        
        # Apply to values
        return scores @ v
    
    def _compute_rho_update(self, x_sparse: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute ρ-matrix update from sparse activations and values.
        
        ρ_update = outer_product(x_sparse, v) averaged over sequence
        """
        # x_sparse: [B, nh, T, N]
        # v: [B, 1, T, D]
        
        # Compute outer product contribution
        # We track the accumulated associative patterns
        x_mean = x_sparse.mean(dim=2)  # [B, nh, N]
        v_mean = v.squeeze(1).mean(dim=1)  # [B, D]
        
        # Outer product: [B, nh, N, D] - but we simplify to [B, nh*N]
        # for velocity computation
        rho_update = torch.einsum('bhn,bd->bhnd', x_mean, v_mean)
        return rho_update.reshape(rho_update.shape[0], -1)  # [B, nh*N*D]
    
    def forward(
        self,
        idx: torch.Tensor,
        state: Optional[RecurrentState] = None,
        return_state: bool = True,
        return_rho_update: bool = True,
    ) -> Tuple[torch.Tensor, Optional[RecurrentState], Optional[torch.Tensor]]:
        """
        Forward pass with optional state tracking.
        
        Args:
            idx: Input token indices [B, T]
            state: Previous recurrent state
            return_state: Whether to return updated state
            return_rho_update: Whether to return ρ update for velocity computation
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            new_state: Updated recurrent state (if return_state=True)
            rho_update: ρ-matrix update for this chunk (if return_rho_update=True)
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = self.N
        
        # Get position offset from state
        position_offset = state.position_offset if state is not None else 0
        
        # Embed input
        x = self.embed(idx).unsqueeze(1)  # [B, 1, T, D]
        x = self.ln(x)
        
        # Track ρ updates across layers
        rho_updates = []
        
        # Process through layers
        for level in range(C.n_layer):
            # Encode to latent space
            x_latent = x @ self.encoder  # [B, nh, T, N]
            
            # Sparse activation (ReLU)
            x_sparse = F.relu(x_latent)  # [B, nh, T, N]
            
            # Compute attention with RoPE
            yKV = self._compute_attention(x_sparse, x, position_offset)
            yKV = self.ln(yKV)
            
            # Second encoding for gating
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            
            # Gated combination
            xy_sparse = x_sparse * y_sparse  # [B, nh, T, N]
            xy_sparse = self.drop(xy_sparse)
            
            # Track ρ update for this layer
            if return_rho_update:
                rho_update = self._compute_rho_update(x_sparse, x)
                rho_updates.append(rho_update)
            
            # Decode back to embedding space
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # [B, 1, T, D]
            
            # Residual connection
            y = self.ln(yMLP)
            x = self.ln(x + y)
        
        # Output projection
        logits = x.view(B, T, D) @ self.lm_head
        
        # Compute aggregate ρ update
        total_rho_update = None
        if return_rho_update and rho_updates:
            # Average across layers
            total_rho_update = torch.stack(rho_updates).mean(dim=0)
        
        # Update state
        new_state = None
        if return_state:
            new_state = RecurrentState(
                position_offset=position_offset + T
            )
            
            if state is not None and state.rho_matrix is not None:
                # Apply damping and accumulate
                new_state.prev_rho = state.rho_matrix.clone()
                
                # LTC: Use adaptive damping based on input
                if self.use_ltc:
                    # Compute adaptive lambda from mean input embedding
                    # x shape: [B, 1, T, D] -> mean over T gives [B, 1, D]
                    x_mean = x.mean(dim=2)  # [B, 1, D]
                    adaptive_lambda = torch.sigmoid(self.damping_gate(x_mean))  # [B, 1, 1]
                    # Squeeze to match rho_matrix shape [B, nh*N*D]
                    adaptive_lambda = adaptive_lambda.squeeze(-1).squeeze(-1).unsqueeze(-1)  # [B, 1]
                    new_state.rho_matrix = adaptive_lambda * state.rho_matrix
                else:
                    new_state.rho_matrix = self.damping * state.rho_matrix
                
                if total_rho_update is not None:
                    new_state.rho_matrix = new_state.rho_matrix + total_rho_update
            elif total_rho_update is not None:
                new_state.rho_matrix = total_rho_update
                new_state.prev_rho = torch.zeros_like(total_rho_update)
            
            new_state.detach()
        
        return logits, new_state, total_rho_update
    
    def compute_velocity(self, state: RecurrentState) -> float:
        """Compute velocity (L2 norm of ρ change) from state."""
        if state is None or state.rho_matrix is None or state.prev_rho is None:
            return 0.0
        
        diff = state.rho_matrix - state.prev_rho
        return float(diff.norm(p=2).item())
    
    def compute_sparsity(self, x_sparse: torch.Tensor) -> float:
        """Compute sparsity of activations."""
        total = x_sparse.numel()
        zeros = (x_sparse == 0).sum().item()
        return zeros / total if total > 0 else 0.0
    
    def reset_state(self) -> RecurrentState:
        """Create fresh initial state."""
        return RecurrentState(
            rho_matrix=None,
            prev_rho=None,
            position_offset=0,
            layer_states=None
        )


def load_pretrained_bdh(config, device: torch.device, use_ltc: bool = False) -> RecurrentBDH:
    """
    Create RecurrentBDH model.
    
    Args:
        config: Model configuration
        device: Target device
        use_ltc: If True, enable Liquid Time Constants for adaptive damping
    
    Note: We use randomly initialized weights since no pretrained checkpoint exists.
    """
    model = RecurrentBDH(config, damping=0.99, use_ltc=use_ltc)
    model = model.to(device)
    return model


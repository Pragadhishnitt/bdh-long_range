"""
BDH Track B: Model Wrapper for Inference

High-level wrapper for priming with backstory and scanning novels
to compute consistency metrics.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from contextlib import nullcontext

import torch
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.bdh_recurrent import RecurrentBDH, RecurrentState, load_pretrained_bdh
from metrics.analysis_metrics import ConsistencyMetrics, compute_velocity, compute_sparsity
from utils.data_loader import ByteTokenizer
from config.model_config import BDHModelConfig, InferenceConfig, get_device, get_dtype



class BDHReasoningWrapper:
    """
    High-level wrapper for BDH consistency reasoning.
    
    Provides methods for:
    1. Priming with backstory text
    2. Scanning novels with state continuity
    3. Computing consistency metrics
    """
    
    def __init__(
        self,
        model_config: BDHModelConfig,
        inference_config: InferenceConfig,
        device: Optional[torch.device] = None,
    ):
        self.model_config = model_config
        self.inference_config = inference_config
        self.device = device or get_device()
        self.dtype = get_dtype(self.device)
        
        # Initialize tokenizer
        self.tokenizer = ByteTokenizer()
        
        # Initialize model
        self.model = self._load_model()
        self.model.eval()
        
        # Mixed precision context
        if inference_config.use_mixed_precision and self.device.type == "cuda":
            self.amp_context = torch.amp.autocast(device_type="cuda", dtype=self.dtype)
        else:
            self.amp_context = nullcontext()
    
    def _load_model(self) -> RecurrentBDH:
        """Load BDH model with configuration."""
        model = RecurrentBDH(
            self.model_config, 
            damping=self.inference_config.damping
        )
        model = model.to(self.device)
        return model
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Convert text to tensor of byte tokens."""
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def _chunk_tokens(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """Split token tensor into chunks."""
        B, T = tokens.size()
        chunk_size = self.inference_config.chunk_size
        chunks = []
        
        for i in range(0, T, chunk_size):
            chunk = tokens[:, i:i + chunk_size]
            if chunk.size(1) > 0:
                chunks.append(chunk)
        
        return chunks
    
    @torch.no_grad()
    def prime_with_backstory(
        self,
        backstory_text: str,
        verbose: bool = False,
    ) -> Tuple[RecurrentState, ConsistencyMetrics]:
        """
        Process backstory to create primed state.
        
        Args:
            backstory_text: Character backstory text
            verbose: Show progress
            
        Returns:
            state: Primed ρ-matrix state
            metrics: Metrics from priming phase
        """
        tokens = self._tokenize(backstory_text)
        chunks = self._chunk_tokens(tokens)
        
        state = self.model.reset_state()
        metrics = ConsistencyMetrics()
        
        chunk_iter = tqdm(chunks, desc="Priming", leave=False) if verbose else chunks
        
        with self.amp_context:
            for chunk_idx, chunk in enumerate(chunk_iter):
                _, state, rho_update = self.model(
                    chunk,
                    state=state,
                    return_state=True,
                    return_rho_update=True,
                )
                
                # Compute velocity
                velocity = self.model.compute_velocity(state)
                
                # Update metrics
                metrics.update(
                    velocity=velocity,
                    chunk_idx=chunk_idx,
                    tokens_in_chunk=chunk.size(1),
                )
        
        metrics.finalize()
        return state, metrics
    
    @torch.no_grad()
    def compute_novel_state(
        self,
        novel_path: Path,
        verbose: bool = True,
    ) -> RecurrentState:
        """
        Compute the final state after reading an entire novel.
        This is cached to avoid re-processing novels for each example.
        
        Args:
            novel_path: Path to novel text file
            verbose: Show progress bar
            
        Returns:
            final_state: Final ρ-matrix state after reading entire novel
        """
        # Load and tokenize novel
        with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
            novel_text = f.read()
        
        tokens = self._tokenize(novel_text)
        chunks = self._chunk_tokens(tokens)
        
        state = self.model.reset_state()
        
        desc = f"Computing novel state: {novel_path.name[:30]}..."
        chunk_iter = tqdm(chunks, desc=desc, leave=False) if verbose else chunks
        
        with self.amp_context:
            for chunk_idx, chunk in enumerate(chunk_iter):
                _, state, _ = self.model(
                    chunk,
                    state=state,
                    return_state=True,
                    return_rho_update=True,  # MUST be True to populate rho_matrix
                )
                
                # Memory management: detach state periodically
                if chunk_idx % 10 == 0:
                    state.detach()
        
        return state
    
    @torch.no_grad()
    def compute_novel_trajectory(
        self,
        novel_path: Path,
        checkpoints: List[float] = [0.25, 0.50, 0.75, 1.0],
        verbose: bool = True,
    ) -> List:
        """
        Compute states at multiple checkpoints throughout the novel.
        Used for --improvise mode to capture temporal dynamics.
        
        Args:
            novel_path: Path to novel text file
            checkpoints: List of progress points (0.0-1.0) to save states
            verbose: Show progress bar
            
        Returns:
            states: List of RecurrentState objects at each checkpoint
        """
        # Load and tokenize novel
        with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
            novel_text = f.read()
        
        tokens = self._tokenize(novel_text)
        chunks = self._chunk_tokens(tokens)
        total_chunks = len(chunks)
        
        # Calculate checkpoint indices
        checkpoint_indices = [int(total_chunks * cp) - 1 for cp in checkpoints]
        checkpoint_indices = [max(0, idx) for idx in checkpoint_indices]  # Ensure non-negative
        
        checkpoint_states = []
        state = self.model.reset_state()
        next_checkpoint = 0
        
        desc = f"Computing trajectory: {novel_path.name[:25]}..."
        chunk_iter = tqdm(chunks, desc=desc, leave=False) if verbose else chunks
        
        with self.amp_context:
            for chunk_idx, chunk in enumerate(chunk_iter):
                _, state, _ = self.model(
                    chunk,
                    state=state,
                    return_state=True,
                    return_rho_update=True,
                )
                
                # Save state at checkpoint
                if (next_checkpoint < len(checkpoint_indices) and 
                    chunk_idx >= checkpoint_indices[next_checkpoint]):
                    checkpoint_states.append(state.clone())
                    next_checkpoint += 1
                
                if chunk_idx % 10 == 0:
                    state.detach()
        
        # Ensure we have all checkpoints (add final state if needed)
        while len(checkpoint_states) < len(checkpoints):
            checkpoint_states.append(state.clone())
        
        return checkpoint_states
    
    @torch.no_grad()
    def compute_trajectory_velocity(
        self,
        backstory_state: RecurrentState,
        novel_trajectory: List,
        metric: str = "cosine",
        aggregation: str = "max",
    ) -> float:
        """
        Compute velocity between backstory and novel trajectory checkpoints.
        Returns max velocity across trajectory to detect contradictions at any point.
        
        Args:
            backstory_state: State after reading backstory
            novel_trajectory: List of states at checkpoints (from compute_novel_trajectory)
            metric: Distance metric ("cosine" or "l2")
            aggregation: How to combine velocities ("max", "mean", "weighted")
            
        Returns:
            velocity: Aggregated velocity across trajectory
        """
        import numpy as np
        
        velocities = []
        for checkpoint_state in novel_trajectory:
            vel = self.compute_velocity_from_states(
                backstory_state,
                checkpoint_state,
                metric=metric,
            )
            velocities.append(vel)
        
        if not velocities:
            return 0.0
        
        if aggregation == "max":
            return max(velocities)
        elif aggregation == "mean":
            return float(np.mean(velocities))
        elif aggregation == "weighted":
            # Weight later checkpoints more heavily
            weights = [0.1, 0.2, 0.3, 0.4][:len(velocities)]
            return float(np.average(velocities, weights=weights))
        else:
            return max(velocities)
    
    @torch.no_grad()
    def compute_velocity_from_states(
        self,
        backstory_state: RecurrentState,
        novel_state: RecurrentState,
        metric: str = "cosine",  # "cosine" or "l2"
    ) -> float:
        """
        Compute velocity between backstory and novel states.
        
        Args:
            backstory_state: Final state after reading backstory
            novel_state: Pre-computed novel state
            metric: "cosine" (normalized) or "l2" (magnitude-sensitive)
            
        Returns:
            velocity: Distance metric (cosine: 0-2, l2: unbounded)
        """
        if (backstory_state is None or backstory_state.rho_matrix is None or
            novel_state is None or novel_state.rho_matrix is None):
            return 0.0
        
        if metric == "cosine":
            # Cosine distance (1 - cosine_similarity)
            # More robust to magnitude differences
            rho_b = backstory_state.rho_matrix.flatten()
            rho_n = novel_state.rho_matrix.flatten()
            
            # Normalize
            rho_b_norm = rho_b / (rho_b.norm(p=2) + 1e-8)
            rho_n_norm = rho_n / (rho_n.norm(p=2) + 1e-8)
            
            # Cosine similarity
            cos_sim = (rho_b_norm * rho_n_norm).sum()
            
            # Return cosine distance (0 = same, 2 = opposite)
            return float((1.0 - cos_sim).item())
        else:
            # L2 norm (original approach)
            diff = novel_state.rho_matrix - backstory_state.rho_matrix
            return float(diff.norm(p=2).item())
    
    @torch.no_grad()
    def compute_embedding_divergence(
        self,
        backstory_state: RecurrentState,
        novel_state: RecurrentState,
        metric: str = "cosine",
    ) -> float:
        """
        Hypothesis B: Measure embedding drift from backstory to novel.
        
        Uses the mean of rho_matrix as an aggregate embedding representation.
        Lower divergence = backstory aligns with novel.
        Higher divergence = backstory contradicts novel.
        
        Args:
            backstory_state: State after reading backstory
            novel_state: State after reading novel
            metric: Distance metric ("cosine" or "l2")
            
        Returns:
            divergence: Embedding divergence score
        """
        if (backstory_state is None or backstory_state.rho_matrix is None or
            novel_state is None or novel_state.rho_matrix is None):
            return 0.0
        
        # Aggregate rho_matrix to a single embedding vector
        # Shape: [batch, seq_len, d_model] -> [d_model]
        backstory_emb = backstory_state.rho_matrix.mean(dim=1).squeeze(0)  # [d_model]
        novel_emb = novel_state.rho_matrix.mean(dim=1).squeeze(0)  # [d_model]
        
        if metric == "cosine":
            # Cosine distance
            backstory_norm = backstory_emb / (backstory_emb.norm(p=2) + 1e-8)
            novel_norm = novel_emb / (novel_emb.norm(p=2) + 1e-8)
            cos_sim = (backstory_norm * novel_norm).sum()
            return float((1.0 - cos_sim).item())
        else:
            # L2 distance
            diff = novel_emb - backstory_emb
            return float(diff.norm(p=2).item())
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        backstory_text: str,
        novel_path: Path,
        max_chunks: Optional[int] = None,
    ) -> float:
        """
        Hypothesis C: Compute perplexity of novel given backstory context.
        
        Lower perplexity = backstory aligns with novel (model is less surprised).
        Higher perplexity = backstory contradicts novel (model is more surprised).
        
        Args:
            backstory_text: Character backstory
            novel_path: Path to novel file
            max_chunks: Limit chunks for testing (None = all)
            
        Returns:
            perplexity: exp(mean_loss) over novel chunks
        """
        import math
        
        # Prime with backstory
        primed_state, _ = self.prime_with_backstory(backstory_text, verbose=False)
        
        # Load novel
        with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
            novel_text = f.read()
        
        tokens = self._tokenize(novel_text)
        chunks = self._chunk_tokens(tokens)
        
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        
        state = primed_state.clone()
        total_loss = 0.0
        total_tokens = 0
        
        with self.amp_context:
            for chunk in chunks:
                # Get logits and compute cross-entropy loss
                logits, state, _ = self.model(
                    chunk,
                    state=state,
                    return_state=True,
                    return_rho_update=True,
                )
                
                # Shift for next-token prediction
                # logits: [batch, seq_len, vocab_size]
                # targets: chunk shifted by 1
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = chunk[:, 1:].contiguous()
                
                # Compute loss
                loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1)
                )
                
                total_loss += loss.item()
                total_tokens += shift_targets.numel()
                
                state.detach()
        
        # Compute perplexity
        if total_tokens == 0:
            return 1.0
        
        mean_loss = total_loss / total_tokens
        perplexity = math.exp(min(mean_loss, 20))  # Cap to prevent overflow
        
        return float(perplexity)
    
    @torch.no_grad()
    def compute_perturbation(
        self,
        backstory_text: str,
        novel_path: Path,
        verbose: bool = False,
        metric: str = "cosine",
        novel_state_baseline: Optional[RecurrentState] = None,  # Use cached if provided
    ) -> float:
        """
        Measure how much the backstory perturbs the novel's trajectory.
        
        Compares:
        - Baseline: Reading novel from scratch
        - Perturbed: Reading novel after priming with backstory
        
        Args:
            backstory_text: Character backstory
            novel_path: Path to novel file
            verbose: Show progress
            metric: Distance metric to use
            novel_state_baseline: Pre-computed baseline state (if cached)
            
        Returns:
            perturbation: How much backstory changes novel processing
        """
        # 1. Get or compute baseline (novel alone)
        if novel_state_baseline is None:
            novel_state_baseline = self.compute_novel_state(novel_path, verbose=verbose)
        
        # 2. Compute perturbed (backstory -> novel)
        backstory_state, _ = self.prime_with_backstory(backstory_text, verbose=False)
        
        # Now process novel starting from backstory state
        with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
            novel_text = f.read()
        
        tokens = self._tokenize(novel_text)
        chunks = self._chunk_tokens(tokens)
        
        state = backstory_state.clone()
        
        desc = f"Computing perturbed state..."
        chunk_iter = tqdm(chunks, desc=desc, leave=False) if verbose else chunks
        
        with self.amp_context:
            for chunk_idx, chunk in enumerate(chunk_iter):
                _, state, _ = self.model(
                    chunk,
                    state=state,
                    return_state=True,
                    return_rho_update=True,
                )
                
                if chunk_idx % 10 == 0:
                    state.detach()
        
        # 3. Compare baseline vs perturbed
        return self.compute_velocity_from_states(
            novel_state_baseline,
            state,
            metric=metric,
        )
    
    @torch.no_grad()
    def scan_novel(
        self,
        novel_path: Path,
        initial_state: RecurrentState,
        verbose: bool = True,
        max_chunks: Optional[int] = None,
    ) -> ConsistencyMetrics:
        """
        Scan novel with primed state, tracking velocity.
        
        Args:
            novel_path: Path to novel text file
            initial_state: Primed state from backstory
            verbose: Show progress bar
            max_chunks: Limit chunks for testing (None = all)
            
        Returns:
            metrics: Consistency metrics including max velocity
        """
        # Load and tokenize novel
        with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
            novel_text = f.read()
        
        tokens = self._tokenize(novel_text)
        chunks = self._chunk_tokens(tokens)
        
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        
        state = initial_state.clone() if initial_state else self.model.reset_state()
        metrics = ConsistencyMetrics()
        
        desc = f"Scanning {novel_path.name[:30]}..."
        chunk_iter = tqdm(chunks, desc=desc, leave=False) if verbose else chunks
        
        with self.amp_context:
            for chunk_idx, chunk in enumerate(chunk_iter):
                _, state, rho_update = self.model(
                    chunk,
                    state=state,
                    return_state=True,
                    return_rho_update=True,
                )
                
                # Compute velocity
                velocity = self.model.compute_velocity(state)
                
                # Update metrics
                metrics.update(
                    velocity=velocity,
                    chunk_idx=chunk_idx,
                    tokens_in_chunk=chunk.size(1),
                )
                
                # Memory management: detach state periodically
                if chunk_idx % 10 == 0:
                    state.detach()
        
        metrics.finalize()
        return metrics

    
    @torch.no_grad()
    def process_example(
        self,
        backstory: str,
        novel_path: Path,
        verbose: bool = True,
        max_chunks: Optional[int] = None,
    ) -> ConsistencyMetrics:
        """
        Process a single example: prime with backstory, scan novel.
        
        Args:
            backstory: Character backstory text
            novel_path: Path to corresponding novel
            verbose: Show progress
            max_chunks: Limit chunks for testing
            
        Returns:
            metrics: Combined consistency metrics
        """
        # Prime with backstory
        primed_state, prime_metrics = self.prime_with_backstory(
            backstory, verbose=False
        )
        
        # Scan novel
        scan_metrics = self.scan_novel(
            novel_path,
            initial_state=primed_state,
            verbose=verbose,
            max_chunks=max_chunks,
        )
        
        return scan_metrics
    
    def predict(self, metrics: ConsistencyMetrics, threshold: float) -> int:
        """
        Predict consistency label based on max velocity.
        
        Args:
            metrics: Consistency metrics from scanning
            threshold: Calibrated threshold
            
        Returns:
            1 if consistent, 0 if contradict
        """
        # Lower velocity → consistent (backstory aligned with novel)
        # Higher velocity → contradict (backstory conflicts with novel)
        return 1 if metrics.max_velocity < threshold else 0
    
    def get_model_info(self) -> dict:
        """Get model information for logging."""
        return {
            "n_layer": self.model_config.n_layer,
            "n_embd": self.model_config.n_embd,
            "n_head": self.model_config.n_head,
            "vocab_size": self.model_config.vocab_size,
            "params": self.model_config.estimate_params(),
            "damping": self.inference_config.damping,
            "chunk_size": self.inference_config.chunk_size,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

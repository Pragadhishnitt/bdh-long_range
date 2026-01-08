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

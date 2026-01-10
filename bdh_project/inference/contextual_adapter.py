"""
BDH Track B: Contextual Adapter (Test-Time Training)

Performs adaptation/fine-tuning of the model on backstory before evaluating the novel.
This creates a stronger "imprint" than simple Hebbian priming.
"""

import copy
import math
from pathlib import Path
from typing import Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
from tqdm import tqdm

from model.bdh_recurrent import RecurrentBDH, RecurrentState
from utils.data_loader import ByteTokenizer
from config.model_config import BDHModelConfig, InferenceConfig, get_device, get_dtype


class ContextualAdapter:
    """
    Test-Time Training (TTT) adapter for BDH.
    
    Temporarily fine-tunes the model on backstory before evaluating the novel.
    This should create stronger separation between consistent/contradictory cases.
    """
    
    def __init__(
        self,
        model_config: BDHModelConfig,
        inference_config: InferenceConfig,
        device: Optional[torch.device] = None,
        # Adaptation hyperparameters
        adapt_lr: float = 1e-4,
        adapt_steps: int = 10,
        adapt_batch_size: int = 1,
    ):
        self.model_config = model_config
        self.inference_config = inference_config
        self.device = device or get_device()
        self.dtype = get_dtype(self.device)
        
        # Adaptation hyperparameters
        self.adapt_lr = adapt_lr
        self.adapt_steps = adapt_steps
        self.adapt_batch_size = adapt_batch_size
        
        # Initialize tokenizer
        self.tokenizer = ByteTokenizer()
        
        # Initialize base model (freezable reference)
        self.base_model = self._load_model()
        self.base_model.eval()
        
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
    
    def _chunk_tokens(self, tokens: torch.Tensor, chunk_size: int = 256) -> list:
        """Split token tensor into chunks for training."""
        B, T = tokens.size()
        chunks = []
        for i in range(0, T - 1, chunk_size):  # -1 to leave room for target
            chunk = tokens[:, i:i + chunk_size + 1]  # +1 for target
            if chunk.size(1) > 1:
                chunks.append(chunk)
        return chunks
    
    def adapt(
        self,
        backstory_text: str,
        verbose: bool = False,
    ) -> RecurrentBDH:
        """
        Adapt model to backstory via Test-Time Training.
        
        Args:
            backstory_text: Character backstory to adapt on
            verbose: Show training progress
            
        Returns:
            adapted_model: Fine-tuned model (copy of base model)
        """
        # Create a fresh copy of the model for adaptation
        adapted_model = copy.deepcopy(self.base_model)
        adapted_model.train()
        
        # Setup optimizer (only train encoder/decoder, not embedding)
        optimizer = torch.optim.AdamW(
            [
                {'params': adapted_model.encoder, 'lr': self.adapt_lr},
                {'params': adapted_model.encoder_v, 'lr': self.adapt_lr},
                {'params': adapted_model.decoder, 'lr': self.adapt_lr},
            ],
            weight_decay=0.01
        )
        
        # Tokenize and chunk backstory
        tokens = self._tokenize(backstory_text)
        chunks = self._chunk_tokens(tokens, chunk_size=256)
        
        if not chunks:
            # Backstory too short, return unchanged model
            adapted_model.eval()
            return adapted_model
        
        # Training loop
        loss_fn = nn.CrossEntropyLoss()
        
        step_iter = range(self.adapt_steps)
        if verbose:
            step_iter = tqdm(step_iter, desc="Adapting", leave=False)
        
        for step in step_iter:
            total_loss = 0.0
            
            for chunk in chunks:
                # Input: all tokens except last
                # Target: all tokens except first (shifted by 1)
                input_ids = chunk[:, :-1]
                target_ids = chunk[:, 1:]
                
                optimizer.zero_grad()
                
                with self.amp_context:
                    # Forward pass
                    logits, _, _ = adapted_model(
                        input_ids,
                        state=None,
                        return_state=False,
                        return_rho_update=False,
                    )
                    
                    # Compute loss
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and hasattr(step_iter, 'set_postfix'):
                step_iter.set_postfix({'loss': total_loss / len(chunks)})
        
        adapted_model.eval()
        return adapted_model
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        model: RecurrentBDH,
        novel_path: Path,
        max_chunks: int = 20,
    ) -> float:
        """
        Compute perplexity of novel using adapted model.
        
        Args:
            model: Adapted model
            novel_path: Path to novel file
            max_chunks: Maximum chunks to process
            
        Returns:
            perplexity: exp(mean_cross_entropy_loss)
        """
        # Load novel
        with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
            novel_text = f.read()
        
        tokens = self._tokenize(novel_text)
        chunks = self._chunk_tokens(tokens, chunk_size=self.inference_config.chunk_size)
        
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        total_loss = 0.0
        total_tokens = 0
        state = model.reset_state()
        
        with self.amp_context:
            for chunk in chunks:
                if chunk.size(1) <= 1:
                    continue
                    
                input_ids = chunk[:, :-1]
                target_ids = chunk[:, 1:]
                
                logits, state, _ = model(
                    input_ids,
                    state=state,
                    return_state=True,
                    return_rho_update=False,
                )
                
                loss_fn = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                total_tokens += target_ids.numel()
                
                state.detach()
        
        if total_tokens == 0:
            return 1.0
        
        mean_loss = total_loss / total_tokens
        perplexity = math.exp(min(mean_loss, 20))  # Cap to prevent overflow
        
        return float(perplexity)
    
    def score_consistency(
        self,
        backstory_text: str,
        novel_path: Path,
        max_chunks: int = 20,
        verbose: bool = False,
    ) -> float:
        """
        Score consistency by adapting to backstory and computing novel perplexity.
        
        Lower perplexity = backstory is CONSISTENT with novel
        Higher perplexity = backstory CONTRADICTS novel
        
        Args:
            backstory_text: Character backstory
            novel_path: Path to novel file
            max_chunks: Max chunks for perplexity computation
            verbose: Show progress
            
        Returns:
            perplexity: Consistency score (lower = more consistent)
        """
        # Step 1: Adapt model to backstory
        adapted_model = self.adapt(backstory_text, verbose=verbose)
        
        # Step 2: Compute perplexity on novel
        perplexity = self.compute_perplexity(
            adapted_model,
            novel_path,
            max_chunks=max_chunks,
        )
        
        # Cleanup adapted model
        del adapted_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return perplexity
    
    def predict(
        self,
        backstory_text: str,
        novel_path: Path,
        threshold: float,
        max_chunks: int = 20,
        verbose: bool = False,
    ) -> Tuple[int, float]:
        """
        Predict consistency label.
        
        Args:
            backstory_text: Character backstory
            novel_path: Path to novel file
            threshold: Perplexity threshold (below = consistent)
            max_chunks: Max chunks for perplexity
            verbose: Show progress
            
        Returns:
            label: 1 = consistent, 0 = contradictory
            score: Perplexity score
        """
        score = self.score_consistency(
            backstory_text,
            novel_path,
            max_chunks=max_chunks,
            verbose=verbose,
        )
        
        # Lower perplexity = consistent
        label = 1 if score < threshold else 0
        
        return label, score

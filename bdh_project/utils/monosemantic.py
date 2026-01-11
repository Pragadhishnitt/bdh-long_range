"""
Monosemantic Masking for BDH Ablation Protocol

Creates binary masks for neurons relevant to backstory keywords.
Used in 'combined' ablation mode to focus scoring on semantically relevant synapses.

Algorithm:
1. Extract top nouns/entities from backstory (simple heuristic)
2. Run each keyword through untrained model
3. Record indices of top 5% active neurons
4. Return binary mask for weighting loss/velocity
"""

import re
from typing import List, Set, Optional
from pathlib import Path

import torch
import torch.nn as nn


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract top nouns/entities from text using simple heuristics.
    
    Algorithm:
    - Split into words, filter by length (>4 chars)
    - Filter out common stopwords
    - Capitalize to identify proper nouns (likely entities)
    - Return top N unique keywords by frequency
    
    Args:
        text: Input text (backstory)
        max_keywords: Maximum keywords to return
        
    Returns:
        List of extracted keywords
    """
    # Common stopwords to filter out
    STOPWORDS = {
        'the', 'and', 'that', 'this', 'with', 'have', 'from', 'they',
        'been', 'were', 'said', 'would', 'could', 'should', 'their',
        'what', 'when', 'where', 'which', 'there', 'about', 'into',
        'more', 'some', 'very', 'just', 'also', 'than', 'then',
        'only', 'other', 'after', 'before', 'being', 'through',
    }
    
    # Extract words (alphanumeric, 4+ chars)
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        word_lower = word.lower()
        if word_lower not in STOPWORDS:
            # Prefer capitalized words (likely proper nouns)
            if word[0].isupper():
                word_freq[word] = word_freq.get(word, 0) + 2  # Boost proper nouns
            else:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
    keywords = [word for word, _ in sorted_words[:max_keywords]]
    
    return keywords


def get_active_neurons(
    text: str,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    top_percent: float = 0.05,
) -> Set[int]:
    """
    Get indices of neurons activated by the given text.
    
    Args:
        text: Input text
        model: RecurrentBDH model
        tokenizer: ByteTokenizer
        device: Target device
        top_percent: Fraction of top activations to consider (default: 5%)
        
    Returns:
        Set of neuron indices that are highly active
    """
    # Tokenize
    tokens = tokenizer.encode(text)
    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Forward pass with no grad
    with torch.no_grad():
        # We need to capture intermediate activations
        # Use a hook to capture sparse activations after ReLU
        activations = []
        
        def hook_fn(module, input, output):
            # Capture ReLU output (sparse activations)
            activations.append(output.detach())
        
        # Register hook on first layer's activation (after ReLU)
        # Note: BDH uses F.relu directly, so we capture x_sparse from forward
        state = model.reset_state()
        _, _, rho_update = model(token_tensor, state=state, return_rho_update=True)
        
        # Use rho_update as proxy for activation pattern
        if rho_update is not None:
            flat_rho = rho_update.flatten()
            
            # Find top 5% most active indices
            k = max(1, int(len(flat_rho) * top_percent))
            top_values, top_indices = torch.topk(flat_rho.abs(), k)
            
            return set(top_indices.cpu().tolist())
    
    return set()


def get_monosemantic_mask(
    backstory_text: str,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    max_keywords: int = 10,
    top_percent: float = 0.05,
) -> torch.Tensor:
    """
    Create a binary mask for neurons relevant to backstory keywords.
    
    This is the main entry point for monosemantic masking.
    
    Args:
        backstory_text: Character backstory text
        model: RecurrentBDH model (untrained/randomly initialized is fine)
        tokenizer: ByteTokenizer instance
        device: Target device
        max_keywords: Number of keywords to extract
        top_percent: Fraction of top neurons to include per keyword
        
    Returns:
        Binary mask tensor [total_neurons] where 1 = relevant, 0 = irrelevant
    """
    # Extract keywords from backstory
    keywords = extract_keywords(backstory_text, max_keywords)
    
    if not keywords:
        # Fallback: return all-ones mask
        # Estimate mask size from model dimensions
        n_head = model.n_head if hasattr(model, 'n_head') else 4
        N = model.N if hasattr(model, 'N') else 8192
        n_embd = model.n_embd if hasattr(model, 'n_embd') else 256
        total_dim = n_head * N * n_embd // n_head  # Approximate rho_update size
        return torch.ones(total_dim, device=device)
    
    # Collect active neurons for each keyword
    all_active = set()
    for keyword in keywords:
        active = get_active_neurons(keyword, model, tokenizer, device, top_percent)
        all_active.update(active)
    
    # Create binary mask
    # Estimate total dimension from first rho_update
    with torch.no_grad():
        state = model.reset_state()
        dummy_tokens = torch.tensor(tokenizer.encode("test"), device=device).unsqueeze(0)
        _, _, rho_update = model(dummy_tokens, state=state, return_rho_update=True)
        
        if rho_update is not None:
            total_dim = rho_update.numel() // rho_update.size(0)  # Per-sample dimension
        else:
            total_dim = 8192 * 256  # Fallback
    
    # Create mask
    mask = torch.zeros(total_dim, device=device)
    for idx in all_active:
        if idx < total_dim:
            mask[idx] = 1.0
    
    # Ensure at least some neurons are active
    if mask.sum() == 0:
        mask = torch.ones(total_dim, device=device)
    
    return mask


def apply_monosemantic_mask(
    rho_update: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply monosemantic mask to rho update for focused scoring.
    
    Args:
        rho_update: Hebbian update tensor [B, dim]
        mask: Binary mask [dim]
        
    Returns:
        Masked rho_update (element-wise multiplication)
    """
    # Broadcast mask to match rho_update shape
    if rho_update.dim() == 2:
        mask = mask.unsqueeze(0).expand_as(rho_update)
    
    return rho_update * mask

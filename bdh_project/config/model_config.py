"""
BDH Track B: Model Configuration

Provides default and lightweight configurations for BDH model,
plus inference settings for chunked processing.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class BDHModelConfig:
    """BDH model architecture configuration."""
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256  # Byte-level tokenization
    dropout: float = 0.1
    
    @property
    def latent_dim_per_head(self) -> int:
        """N = mlp_internal_dim_multiplier * n_embd // n_head"""
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head
    
    @property
    def total_latent_dim(self) -> int:
        """Total latent dimension across all heads."""
        return self.latent_dim_per_head * self.n_head
    
    def estimate_params(self) -> int:
        """Estimate total parameters (weights shared across layers)."""
        N = self.latent_dim_per_head
        decoder = self.n_head * N * self.n_embd
        encoder = self.n_head * self.n_embd * N
        encoder_v = self.n_head * self.n_embd * N
        embed = self.vocab_size * self.n_embd
        lm_head = self.n_embd * self.vocab_size
        return decoder + encoder + encoder_v + embed + lm_head


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    chunk_size: int = 2048
    damping: float = 0.99
    batch_size: int = 1
    use_mixed_precision: bool = True
    checkpoint_every: int = 10
    velocity_ema_alpha: float = 0.1  # For smoothing velocity tracking
    

@dataclass
class PathConfig:
    """Dataset and output path configuration."""
    train_csv: str = "Dataset/train.csv"
    test_csv: str = "Dataset/test.csv"
    books_dir: str = "Dataset/Books"
    output_dir: str = "outputs"
    results_file: str = "results.csv"
    checkpoint_dir: str = "checkpoints"
    
    # Book name to file mapping
    book_mapping: dict = None
    
    def __post_init__(self):
        self.book_mapping = {
            "In Search of the Castaways": "In search of the castaways.txt",
            "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
        }


# Preset configurations
def get_default_config() -> BDHModelConfig:
    """Default BDH configuration (6 layers, 25.3M params)."""
    return BDHModelConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        mlp_internal_dim_multiplier=128,
        vocab_size=256,
        dropout=0.1
    )


def get_small_config() -> BDHModelConfig:
    """Lightweight BDH configuration (4 layers, ~25.3M params - same weights shared)."""
    return BDHModelConfig(
        n_layer=4,
        n_embd=256,
        n_head=4,
        mlp_internal_dim_multiplier=128,
        vocab_size=256,
        dropout=0.1
    )


def get_config_by_name(name: str) -> BDHModelConfig:
    """Get configuration by name."""
    configs = {
        "default": get_default_config,
        "small": get_small_config,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
    return configs[name]()


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Get optimal dtype for the device."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

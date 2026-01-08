"""
BDH Track B: Main Pipeline

Narrative consistency classification using BDH stateful architecture.

Usage:
    python main.py                      # Full pipeline (train + inference)
    python main.py --train              # Train/calibrate only
    python main.py --inference          # Inference only (requires checkpoint)
    python main.py --default            # Use 6-layer model (default)
    python main.py --small              # Use 4-layer model
    python main.py --dry-run --limit 5  # Quick test with 5 examples
"""

import argparse
import json
import os
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    get_config_by_name,
    InferenceConfig,
    PathConfig,
    get_device,
    get_dtype,
)
from metrics import ConsistencyMetrics, CalibrationResult
from utils import DataLoader, get_dataset_stats
from inference import BDHReasoningWrapper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BDH Track B: Narrative Consistency Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Full pipeline with default (6-layer) model
  python main.py --small         # Full pipeline with 4-layer model
  python main.py --train         # Calibration only
  python main.py --inference     # Test inference only
  python main.py --dry-run       # Quick test run
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--train", action="store_true",
        help="Run calibration/training only"
    )
    mode_group.add_argument(
        "--inference", action="store_true",
        help="Run test inference only (requires checkpoint)"
    )
    
    # Model size selection
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--default", action="store_true",
        help="Use default 6-layer model (25.3M params)"
    )
    size_group.add_argument(
        "--small", action="store_true",
        help="Use lightweight 4-layer model"
    )
    
    # Pipeline options
    parser.add_argument(
        "--mode", type=str, default="cached",
        choices=["cached", "streaming"],
        help="Processing mode: 'cached' (fast, default) or 'streaming' (slow, accurate)"
    )
    parser.add_argument(
        "--metric", type=str, default="cosine",
        choices=["cosine", "l2"],
        help="Distance metric: 'cosine' (normalized, recommended) or 'l2' (magnitude-sensitive)"
    )
    parser.add_argument(
        "--perturbation", action="store_true",
        help="Use perturbation measurement (slower, compares baseline vs primed novel)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick test run without full processing"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of examples to process"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Limit chunks per novel (for testing)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to calibration checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def setup_directories(output_dir: str) -> Dict[str, Path]:
    """Create output directories."""
    paths = {
        "output": Path(output_dir),
        "checkpoints": Path(output_dir) / "checkpoints",
        "plots": Path(output_dir) / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_checkpoint(
    calibration: CalibrationResult,
    path: Path,
    model_config_name: str,
):
    """Save calibration checkpoint."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "model_config": model_config_name,
        "calibration": calibration.to_dict(),
        "example_ids": calibration.example_ids,
        "max_velocities": calibration.max_velocities,
        "labels": calibration.labels,
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved checkpoint: {path}")


def load_checkpoint(path: Path) -> CalibrationResult:
    """Load calibration checkpoint."""
    with open(path, "r") as f:
        data = json.load(f)
    
    calibration = CalibrationResult(
        optimal_threshold=data["calibration"]["optimal_threshold"],
        train_accuracy=data["calibration"]["train_accuracy"],
        example_ids=data["example_ids"],
        max_velocities=data["max_velocities"],
        labels=data["labels"],
    )
    calibration.consistent_mean = data["calibration"]["consistent_mean"]
    calibration.consistent_std = data["calibration"]["consistent_std"]
    calibration.contradict_mean = data["calibration"]["contradict_mean"]
    calibration.contradict_std = data["calibration"]["contradict_std"]
    
    return calibration


def precompute_novel_states(
    wrapper: BDHReasoningWrapper,
    loader: DataLoader,
    paths: Dict[str, Path],
) -> Dict[str, any]:
    """Pre-compute novel states once to avoid redundant processing."""
    cache_path = paths["checkpoints"] / "novel_states.pkl"
    
    # Check if cache exists
    if cache_path.exists():
        print(f"\n✓ Loading cached novel states from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("\n" + "="*60)
    print("PHASE 0: PRE-COMPUTING NOVEL STATES (ONE-TIME)")
    print("="*60)
    
    novel_states = {}
    
    for book_name in loader.book_mapping.keys():
        print(f"\nProcessing: {book_name}")
        novel_path = loader.get_book_path(book_name)
        
        # Compute novel state
        novel_state = wrapper.compute_novel_state(novel_path, verbose=True)
        novel_states[book_name] = novel_state
        
        print(f"✓ Cached state for {book_name}")
    
    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(novel_states, f)
    
    print(f"\n✓ Saved novel states to {cache_path}")
    return novel_states


def run_calibration(
    wrapper: BDHReasoningWrapper,
    loader: DataLoader,
    novel_states: Dict[str, any],
    paths: Dict[str, Path],
    args: argparse.Namespace,
    config_name: str,
    mode: str = "cached",
    metric: str = "cosine",
    is_validation: bool = False,
) -> CalibrationResult:
    """Run calibration on training set using cached novel states."""
    phase_name = "VALIDATION" if is_validation else "CALIBRATION"
    print("\n" + "="*60)
    print(f"PHASE {'2' if is_validation else '1'}: {phase_name}")
    print("="*60)
    
    train_examples = loader.get_train_examples()
    
    # Split into train (60) and validation (20)
    if not args.dry_run and not args.limit:
        train_split, val_split = train_test_split(
            train_examples, 
            train_size=60, 
            test_size=20,
            random_state=42,
            stratify=[ex['label_binary'] for ex in train_examples]
        )
        examples = val_split if is_validation else train_split
    else:
        # Use all for dry-run or limited mode
        examples = train_examples[:args.limit] if args.limit else train_examples
    
    calibration = CalibrationResult()
    
    # Process each example
    desc = "Validating" if is_validation else "Calibrating"
    pbar = tqdm(examples, desc=desc)
    
    for i, example in enumerate(pbar):
        try:
            book_name = example['book_name']
            
            # Choose processing mode
            if mode == "streaming":
                # Original approach: stream the full novel
                novel_path = loader.get_book_path(book_name)
                metrics = wrapper.process_example(
                    backstory=example['content'],
                    novel_path=novel_path,
                    verbose=False,
                    max_chunks=args.max_chunks if args.dry_run else None,
                )
                velocity = metrics.max_velocity
            else:
                # Cached approach: compare final states
                if book_name not in novel_states:
                    print(f"\n⚠ Novel state not cached for {book_name}, skipping")
                    continue
                
                novel_state = novel_states[book_name]
                
                if args.perturbation:
                    # Perturbation mode: measure how much backstory changes novel
                    novel_path = loader.get_book_path(book_name)
                    velocity = wrapper.compute_perturbation(
                        backstory_text=example['content'],
                        novel_path=novel_path,
                        verbose=False,
                        metric=metric,
                    )
                else:
                    # Standard cached mode: compare final states
                    # Process only the backstory (fast!)
                    backstory_state, _ = wrapper.prime_with_backstory(
                        example['content'],
                        verbose=False,
                    )
                    
                    # Compute velocity against cached novel state
                    velocity = wrapper.compute_velocity_from_states(
                        backstory_state,
                        novel_state,
                        metric=metric,
                    )
            
            # Record result
            calibration.add_example(
                example_id=example['id'],
                max_velocity=velocity,
                label=example['label_binary'],
            )
            
            # Update progress bar
            pbar.set_postfix({
                "vel": f"{velocity:.4f}",
                "label": "C" if example['label_binary'] == 1 else "X",
            })
            
            # Periodic checkpoint (only during training)
            if not is_validation and (i + 1) % 10 == 0:
                checkpoint_path = paths["checkpoints"] / f"calibration_partial_{i+1}.json"
                calibration.compute_optimal_threshold()
                save_checkpoint(calibration, checkpoint_path, config_name)
                
        except Exception as e:
            print(f"\n⚠ Error processing example {example['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute final threshold
    threshold = calibration.compute_optimal_threshold()
    
    print(f"\n{'─'*40}")
    print(f"{phase_name} RESULTS:")
    print(f"  Optimal threshold: {threshold:.6f}")
    print(f"  Accuracy: {calibration.train_accuracy:.2%}")
    print(f"  Consistent μ={calibration.consistent_mean:.4f}, σ={calibration.consistent_std:.4f}")
    print(f"  Contradict μ={calibration.contradict_mean:.4f}, σ={calibration.contradict_std:.4f}")
    print(f"{'─'*40}")
    
    # Save final checkpoint (only during training)
    if not is_validation:
        checkpoint_path = paths["checkpoints"] / "calibration_final.json"
        save_checkpoint(calibration, checkpoint_path, config_name)
    
    return calibration


def run_inference(
    wrapper: BDHReasoningWrapper,
    loader: DataLoader,
    novel_states: Dict[str, any],
    calibration: CalibrationResult,
    paths: Dict[str, Path],
    args: argparse.Namespace,
    mode: str = "cached",
    metric: str = "cosine",
) -> pd.DataFrame:
    """Run inference on test set using cached novel states."""
    print("\n" + "="*60)
    print("PHASE 3: TEST INFERENCE")
    print("="*60)
    
    test_examples = loader.get_test_examples()
    
    if args.limit:
        test_examples = test_examples[:args.limit]
    
    results = []
    
    # Process each test example
    pbar = tqdm(test_examples, desc="Predicting")
    
    for example in pbar:
        try:
            book_name = example['book_name']
            
            # Choose processing mode
            if mode == "streaming":
                # Original approach: stream the full novel
                novel_path = loader.get_book_path(book_name)
                metrics = wrapper.process_example(
                    backstory=example['content'],
                    novel_path=novel_path,
                    verbose=False,
                    max_chunks=args.max_chunks if args.dry_run else None,
                )
                velocity = metrics.max_velocity
                prediction = calibration.predict(velocity)
            else:
                # Cached approach: compare final states
                if book_name not in novel_states:
                    print(f"\n⚠ Novel state not cached for {book_name}, using default")
                    prediction = 1  # Default to consistent
                    velocity = 0.0
                else:
                    novel_state = novel_states[book_name]
                    
                if args.perturbation:
                    # Perturbation mode
                    novel_path = loader.get_book_path(book_name)
                    velocity = wrapper.compute_perturbation(
                        backstory_text=example['content'],
                        novel_path=novel_path,
                        verbose=False,
                        metric=metric,
                    )
                else:
                    # Standard cached mode
                    # Process only the backstory
                    backstory_state, _ = wrapper.prime_with_backstory(
                        example['content'],
                        verbose=False,
                    )
                    
                    # Compute velocity
                    velocity = wrapper.compute_velocity_from_states(
                        backstory_state,
                        novel_state,
                        metric=metric,
                    )
                    
                    # Predict
                    prediction = calibration.predict(velocity)
            
            results.append({
                "id": example['id'],
                "prediction": prediction,
                "max_velocity": velocity,
                "mean_velocity": velocity,  # Same for cached approach
            })
            
            pbar.set_postfix({
                "pred": prediction,
                "vel": f"{velocity:.4f}",
            })
            
        except Exception as e:
            print(f"\n⚠ Error processing example {example['id']}: {e}")
            import traceback
            traceback.print_exc()
            # Default prediction for errors
            results.append({
                "id": example['id'],
                "prediction": 1,  # Default to consistent
                "max_velocity": 0.0,
                "mean_velocity": 0.0,
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = paths["output"] / "results.csv"
    results_df[["id", "prediction"]].to_csv(results_path, index=False)
    print(f"\n✓ Saved predictions: {results_path}")
    
    # Save detailed results
    detailed_path = paths["output"] / "results_detailed.csv"
    results_df.to_csv(detailed_path, index=False)
    print(f"✓ Saved detailed results: {detailed_path}")
    
    # Summary
    print(f"\n{'─'*40}")
    print("INFERENCE RESULTS:")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Consistent (1): {(results_df['prediction'] == 1).sum()}")
    print(f"  Contradict (0): {(results_df['prediction'] == 0).sum()}")
    print(f"{'─'*40}")
    
    return results_df


def generate_plots(
    calibration: CalibrationResult,
    paths: Dict[str, Path],
):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠ matplotlib/seaborn not installed, skipping plots")
        return
    
    print("\nGenerating plots...")
    
    # 1. Velocity distributions by label
    fig, ax = plt.subplots(figsize=(10, 6))
    
    velocities = np.array(calibration.max_velocities)
    labels = np.array(calibration.labels)
    
    consistent = velocities[labels == 1]
    contradict = velocities[labels == 0]
    
    ax.hist(consistent, bins=20, alpha=0.6, label='Consistent', color='green')
    ax.hist(contradict, bins=20, alpha=0.6, label='Contradict', color='red')
    ax.axvline(calibration.optimal_threshold, color='black', linestyle='--', 
               label=f'Threshold: {calibration.optimal_threshold:.4f}')
    ax.set_xlabel('Max Velocity')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Distribution by Label')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(paths["plots"] / "velocity_distribution.png", dpi=150)
    plt.close()
    
    print(f"✓ Saved: velocity_distribution.png")
    
    # 2. Example velocities scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if l == 1 else 'red' for l in labels]
    ax.scatter(range(len(velocities)), velocities, c=colors, alpha=0.7)
    ax.axhline(calibration.optimal_threshold, color='black', linestyle='--')
    ax.set_xlabel('Example Index')
    ax.set_ylabel('Max Velocity')
    ax.set_title('Training Examples: Max Velocity')
    
    plt.tight_layout()
    plt.savefig(paths["plots"] / "velocity_scatter.png", dpi=150)
    plt.close()
    
    print(f"✓ Saved: velocity_scatter.png")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup
    print("="*60)
    print("BDH TRACK B: NARRATIVE CONSISTENCY CLASSIFICATION")
    print("="*60)
    
    # Determine model configuration
    if args.small:
        config_name = "small"
    else:
        config_name = "default"  # 6-layer is default
    
    model_config = get_config_by_name(config_name)
    inference_config = InferenceConfig()
    
    # Adjust for dry-run
    if args.dry_run:
        args.max_chunks = args.max_chunks or 10
        args.limit = args.limit or 5
    
    # Print configuration
    device = get_device()
    print(f"\nConfiguration:")
    print(f"  Model: {config_name} ({model_config.n_layer} layers)")
    print(f"  Parameters: {model_config.estimate_params():,}")
    print(f"  Device: {device}")
    print(f"  Chunk size: {inference_config.chunk_size}")
    print(f"  Damping: {inference_config.damping}")
    
    if args.dry_run:
        print(f"  [DRY-RUN] Limit: {args.limit}, Max chunks: {args.max_chunks}")
    
    # Setup directories
    paths = setup_directories(args.output_dir)
    
    # Initialize data loader
    base_path = PROJECT_ROOT.parent
    loader = DataLoader(base_path=base_path)
    
    # Load datasets
    loader.load_train()
    loader.load_test()
    
    # Show dataset stats
    stats = get_dataset_stats(loader)
    print(f"\nDataset:")
    print(f"  Train: {stats['train_total']} examples")
    print(f"    - Consistent: {stats['consistent_count']}")
    print(f"    - Contradict: {stats['contradict_count']}")
    print(f"  Test: {stats['test_total']} examples")
    print(f"  Books: {list(stats['train_by_book'].keys())}")
    
    # Initialize model wrapper
    print(f"\nInitializing BDH model...")
    wrapper = BDHReasoningWrapper(
        model_config=model_config,
        inference_config=inference_config,
        device=device,
    )
    print(f"✓ Model ready")
    
    # Determine mode
    run_train = not args.inference
    run_infer = not args.train
    mode = args.mode
    metric = args.metric
    
    print(f"\nProcessing Mode: {mode.upper()}")
    if mode == "streaming":
        print("  ⚠ Streaming mode: Slow but captures temporal dynamics")
    else:
        print("  ✓ Cached mode: Fast with pre-computed novel states")
    
    print(f"Distance Metric: {metric.upper()}")
    if metric == "cosine":
        print("  ✓ Cosine similarity: Normalized, magnitude-invariant")
    else:
        print("  ⚠ L2 norm: Sensitive to magnitude differences")
        
    if args.perturbation:
        print("  ⚠ Perturbation mode: Measuring trajectory divergence (slower)")
    
    # Phase 0: Pre-compute novel states (only for cached mode)
    novel_states = {}
    if mode == "cached":
        novel_states = precompute_novel_states(wrapper, loader, paths)
    
    calibration = None
    validation_result = None
    
    # Phase 1: Calibration (60 examples)
    if run_train:
        calibration = run_calibration(
            wrapper=wrapper,
            loader=loader,
            novel_states=novel_states,
            paths=paths,
            args=args,
            config_name=config_name,
            mode=mode,
            metric=metric,
            is_validation=False,
        )
        
        # Phase 2: Validation (20 examples)
        if not args.dry_run and not args.limit:
            validation_result = run_calibration(
                wrapper=wrapper,
                loader=loader,
                novel_states=novel_states,
                paths=paths,
                args=args,
                config_name=config_name,
                mode=mode,
                metric=metric,
                is_validation=True,
            )
        
        # Generate plots
        generate_plots(calibration, paths)
    
    # Phase 3: Test Inference (60 examples)
    if run_infer:
        # Load checkpoint if inference-only
        if calibration is None:
            checkpoint_path = args.checkpoint
            if checkpoint_path is None:
                checkpoint_path = paths["checkpoints"] / "calibration_final.json"
            
            if not Path(checkpoint_path).exists():
                print(f"✗ Checkpoint not found: {checkpoint_path}")
                print("  Run with --train first, or specify --checkpoint")
                sys.exit(1)
            
            print(f"\nLoading checkpoint: {checkpoint_path}")
            calibration = load_checkpoint(Path(checkpoint_path))
            print(f"  Threshold: {calibration.optimal_threshold:.6f}")
        
        results = run_inference(
            wrapper=wrapper,
            loader=loader,
            novel_states=novel_states,
            calibration=calibration,
            paths=paths,
            args=args,
            mode=mode,
            metric=metric,
        )
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

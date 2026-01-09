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
from dataclasses import dataclass

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
    
    # Advanced optimization flags
    parser.add_argument(
        "--improvise", action="store_true",
        help="K-fold cross-validation (4 folds) + multi-checkpoint caching for better accuracy"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Combine all 3 hypotheses: velocity + embedding divergence + perplexity (majority vote)"
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


def precompute_novel_trajectories(
    wrapper: BDHReasoningWrapper,
    loader: DataLoader,
    paths: Dict[str, Path],
    checkpoints: List[float] = [0.25, 0.50, 0.75, 1.0],
) -> Dict[str, List]:
    """Pre-compute multi-checkpoint novel trajectories for --improvise mode.
    
    Args:
        wrapper: BDH model wrapper
        loader: Data loader
        paths: Output paths
        checkpoints: Progress points to save states (default: 25%, 50%, 75%, 100%)
    
    Returns:
        Dict mapping book_name -> list of states at each checkpoint
    """
    cache_path = paths["checkpoints"] / "novel_trajectories.pkl"
    
    # Check if cache exists
    if cache_path.exists():
        print(f"\n✓ Loading cached novel trajectories from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("\n" + "="*60)
    print("PHASE 0: PRE-COMPUTING NOVEL TRAJECTORIES (MULTI-CHECKPOINT)")
    print(f"  Checkpoints: {[f'{cp*100:.0f}%' for cp in checkpoints]}")
    print("="*60)
    
    novel_trajectories = {}
    
    for book_name in loader.book_mapping.keys():
        print(f"\nProcessing: {book_name}")
        novel_path = loader.get_book_path(book_name)
        
        # Compute trajectory with multiple checkpoints
        trajectory = wrapper.compute_novel_trajectory(
            novel_path, 
            checkpoints=checkpoints,
            verbose=True
        )
        novel_trajectories[book_name] = trajectory
        
        print(f"✓ Cached {len(trajectory)}-checkpoint trajectory for {book_name}")
    
    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(novel_trajectories, f)
    
    print(f"\n✓ Saved novel trajectories to {cache_path}")
    return novel_trajectories


def run_kfold_calibration(
    wrapper: BDHReasoningWrapper,
    loader: DataLoader,
    novel_data: Dict[str, any],
    paths: Dict[str, Path],
    args: argparse.Namespace,
    config_name: str,
    mode: str = "cached",
    metric: str = "cosine",
    use_trajectories: bool = False,
) -> Tuple[CalibrationResult, List[Dict]]:
    """Run K-fold cross-validation for robust threshold estimation.
    
    Args:
        wrapper: BDH model wrapper
        loader: Data loader
        novel_data: Pre-computed novel states or trajectories
        paths: Output paths
        args: Command line arguments
        config_name: Model config name
        mode: Processing mode ("cached" or "streaming")
        metric: Distance metric
        use_trajectories: If True, novel_data contains trajectories (list of states)
    
    Returns:
        final_calibration: Calibration with median threshold
        fold_results: Per-fold statistics
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION (4 FOLDS)")
    print("="*60)
    
    train_examples = loader.get_train_examples()
    labels = [ex['label_binary'] for ex in train_examples]
    
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    fold_results = []
    all_velocities = []
    all_labels = []
    
    import gc
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_examples, labels)):
        print(f"\n{'-'*40}")
        print(f"FOLD {fold_idx + 1}/4")
        print(f"  Train: {len(train_idx)} examples")
        print(f"  Val: {len(val_idx)} examples")
        print(f"{'-'*40}")
        
        fold_train = [train_examples[i] for i in train_idx]
        fold_val = [train_examples[i] for i in val_idx]
        
        # Run calibration on fold_train
        fold_calibration = CalibrationResult()
        
        for example in tqdm(fold_train, desc=f"Fold {fold_idx+1} Train"):
            try:
                book_name = example['book_name']
                
                if book_name not in novel_data:
                    continue
                
                if mode == "streaming":
                    # Full streaming for each example
                    novel_path = loader.get_book_path(book_name)
                    metrics = wrapper.process_example(
                        backstory=example['content'],
                        novel_path=novel_path,
                        verbose=False,
                        max_chunks=args.max_chunks if args.dry_run else None,
                    )
                    velocity = metrics.max_velocity
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    # Cached mode
                    backstory_state, _ = wrapper.prime_with_backstory(
                        example['content'], verbose=False
                    )
                    
                    if use_trajectories:
                        # Use trajectory-based velocity
                        trajectory = novel_data[book_name]
                        velocity = wrapper.compute_trajectory_velocity(
                            backstory_state, trajectory, metric=metric
                        )
                    else:
                        # Use single state
                        novel_state = novel_data[book_name]
                        velocity = wrapper.compute_velocity_from_states(
                            backstory_state, novel_state, metric=metric
                        )
                
                fold_calibration.add_example(
                    example_id=example['id'],
                    max_velocity=velocity,
                    label=example['label_binary']
                )
                all_velocities.append(velocity)
                all_labels.append(example['label_binary'])
                
            except Exception as e:
                print(f"\n⚠ Error: {e}")
                continue
        
        fold_calibration.compute_optimal_threshold()
        
        # Validate on fold_val
        val_correct = 0
        val_total = 0
        
        for example in tqdm(fold_val, desc=f"Fold {fold_idx+1} Val"):
            try:
                book_name = example['book_name']
                if book_name not in novel_data:
                    continue
                
                backstory_state, _ = wrapper.prime_with_backstory(
                    example['content'], verbose=False
                )
                
                if use_trajectories:
                    trajectory = novel_data[book_name]
                    velocity = wrapper.compute_trajectory_velocity(
                        backstory_state, trajectory, metric=metric
                    )
                else:
                    novel_state = novel_data[book_name]
                    velocity = wrapper.compute_velocity_from_states(
                        backstory_state, novel_state, metric=metric
                    )
                
                prediction = fold_calibration.predict(velocity)
                if prediction == example['label_binary']:
                    val_correct += 1
                val_total += 1
                
            except Exception as e:
                continue
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        fold_results.append({
            "fold": fold_idx + 1,
            "threshold": fold_calibration.optimal_threshold,
            "train_accuracy": fold_calibration.train_accuracy,
            "train_f1": fold_calibration.f1,
            "val_accuracy": val_accuracy,
        })
        
        print(f"\n  Fold {fold_idx+1} Results:")
        print(f"    Threshold: {fold_calibration.optimal_threshold:.6f}")
        print(f"    Train Acc: {fold_calibration.train_accuracy:.2%}, F1: {fold_calibration.f1:.4f}")
        print(f"    Val Acc: {val_accuracy:.2%}")
    
    # Aggregate results
    thresholds = [r["threshold"] for r in fold_results]
    final_threshold = float(np.median(thresholds))
    
    print(f"\n{'='*60}")
    print("K-FOLD AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"  Thresholds: {[f'{t:.6f}' for t in thresholds]}")
    print(f"  Median Threshold: {final_threshold:.6f}")
    print(f"  Train Acc: {np.mean([r['train_accuracy'] for r in fold_results]):.2%} ± {np.std([r['train_accuracy'] for r in fold_results]):.2%}")
    print(f"  Val Acc: {np.mean([r['val_accuracy'] for r in fold_results]):.2%} ± {np.std([r['val_accuracy'] for r in fold_results]):.2%}")
    
    # Create final calibration with collected data
    final_calibration = CalibrationResult(optimal_threshold=final_threshold)
    for vel, lbl in zip(all_velocities, all_labels):
        final_calibration.max_velocities.append(vel)
        final_calibration.labels.append(lbl)
    final_calibration.compute_optimal_threshold()
    # Override with median threshold
    final_calibration.optimal_threshold = final_threshold
    
    # Save K-fold results
    kfold_path = paths["checkpoints"] / "kfold_results.json"
    with open(kfold_path, 'w') as f:
        json.dump({
            "fold_results": fold_results,
            "final_threshold": final_threshold,
        }, f, indent=2)
    print(f"\n✓ Saved K-fold results to {kfold_path}")
    
    return final_calibration, fold_results


@dataclass
class EnsembleCalibration:
    """Calibration results for ensemble of 3 hypotheses."""
    # Individual calibrations
    velocity_calibration: CalibrationResult = None
    divergence_calibration: CalibrationResult = None
    perplexity_calibration: CalibrationResult = None
    
    # Ensemble results
    ensemble_accuracy: float = 0.0
    
    def predict_ensemble(self, velocity: float, divergence: float, perplexity: float) -> int:
        """Majority vote across 3 hypotheses."""
        votes = []
        
        # Hypothesis A: Velocity (lower = consistent)
        if self.velocity_calibration:
            votes.append(1 if velocity < self.velocity_calibration.optimal_threshold else 0)
        
        # Hypothesis B: Divergence (lower = consistent)
        if self.divergence_calibration:
            votes.append(1 if divergence < self.divergence_calibration.optimal_threshold else 0)
        
        # Hypothesis C: Perplexity (lower = consistent)
        if self.perplexity_calibration:
            votes.append(1 if perplexity < self.perplexity_calibration.optimal_threshold else 0)
        
        # Majority vote (2/3)
        return 1 if sum(votes) >= 2 else 0


def run_ensemble_calibration(
    wrapper: BDHReasoningWrapper,
    loader: DataLoader,
    novel_states: Dict[str, any],
    paths: Dict[str, Path],
    args: argparse.Namespace,
    config_name: str,
    mode: str = "cached",
    metric: str = "cosine",
) -> EnsembleCalibration:
    """Run calibration for all 3 ensemble hypotheses."""
    print("\n" + "="*60)
    print("ENSEMBLE CALIBRATION (3 HYPOTHESES)")
    print("="*60)
    
    train_examples = loader.get_train_examples()
    
    # Split into train (60) and validation (20)
    if not args.dry_run and not args.limit:
        from sklearn.model_selection import train_test_split
        train_split, val_split = train_test_split(
            train_examples,
            train_size=60,
            test_size=20,
            random_state=42,
            stratify=[ex['label_binary'] for ex in train_examples]
        )
        examples = train_split
    else:
        examples = train_examples[:args.limit] if args.limit else train_examples
    
    # Initialize calibrations
    velocity_cal = CalibrationResult()
    divergence_cal = CalibrationResult()
    perplexity_cal = CalibrationResult()
    
    # Collect signals from all 3 hypotheses
    print("\nCollecting signals from all 3 hypotheses...")
    pbar = tqdm(examples, desc="Ensemble Calibration")
    
    import gc
    
    for i, example in enumerate(pbar):
        try:
            book_name = example['book_name']
            
            if mode == "cached" and book_name not in novel_states:
                continue
            
            # Get novel path
            novel_path = loader.get_book_path(book_name)
            
            # Prime with backstory
            backstory_state, _ = wrapper.prime_with_backstory(example['content'], verbose=False)
            
            # Hypothesis A: Velocity
            if mode == "streaming":
                metrics = wrapper.process_example(
                    backstory=example['content'],
                    novel_path=novel_path,
                    verbose=False,
                    max_chunks=args.max_chunks if args.dry_run else None,
                )
                velocity = metrics.max_velocity
            else:
                novel_state = novel_states[book_name]
                velocity = wrapper.compute_velocity_from_states(
                    backstory_state, novel_state, metric=metric
                )
            
            # Hypothesis B: Embedding Divergence
            if mode == "cached":
                novel_state = novel_states[book_name]
                divergence = wrapper.compute_embedding_divergence(
                    backstory_state, novel_state, metric=metric
                )
            else:
                # For streaming, recompute novel state for divergence
                novel_state = wrapper.compute_novel_state(novel_path, verbose=False)
                divergence = wrapper.compute_embedding_divergence(
                    backstory_state, novel_state, metric=metric
                )
            
            # Hypothesis C: Perplexity
            perplexity = wrapper.compute_perplexity(
                backstory_text=example['content'],
                novel_path=novel_path,
                max_chunks=5 if args.dry_run else None,  # Limit for speed
            )
            
            # Record all signals
            velocity_cal.add_example(example['id'], velocity, example['label_binary'])
            divergence_cal.add_example(example['id'], divergence, example['label_binary'])
            perplexity_cal.add_example(example['id'], perplexity, example['label_binary'])
            
            pbar.set_postfix({
                "vel": f"{velocity:.2f}",
                "div": f"{divergence:.4f}",
                "ppl": f"{perplexity:.2f}",
            })
            
            if mode == "streaming":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            continue
    
    # Compute optimal thresholds for each hypothesis
    print("\n" + "-"*40)
    print("HYPOTHESIS A: Velocity")
    velocity_cal.compute_optimal_threshold()
    print(f"  Threshold: {velocity_cal.optimal_threshold:.6f}")
    print(f"  Accuracy: {velocity_cal.train_accuracy:.2%}")
    print(f"  F1: {velocity_cal.f1:.4f}")
    
    print("\n" + "-"*40)
    print("HYPOTHESIS B: Embedding Divergence")
    divergence_cal.compute_optimal_threshold()
    print(f"  Threshold: {divergence_cal.optimal_threshold:.6f}")
    print(f"  Accuracy: {divergence_cal.train_accuracy:.2%}")
    print(f"  F1: {divergence_cal.f1:.4f}")
    
    print("\n" + "-"*40)
    print("HYPOTHESIS C: Perplexity")
    perplexity_cal.compute_optimal_threshold()
    print(f"  Threshold: {perplexity_cal.optimal_threshold:.6f}")
    print(f"  Accuracy: {perplexity_cal.train_accuracy:.2%}")
    print(f"  F1: {perplexity_cal.f1:.4f}")
    
    # Compute ensemble accuracy (majority vote)
    ensemble_cal = EnsembleCalibration(
        velocity_calibration=velocity_cal,
        divergence_calibration=divergence_cal,
        perplexity_calibration=perplexity_cal,
    )
    
    ensemble_correct = 0
    total = len(velocity_cal.labels)
    
    for i in range(total):
        vel = velocity_cal.max_velocities[i]
        div = divergence_cal.max_velocities[i]
        ppl = perplexity_cal.max_velocities[i]
        label = velocity_cal.labels[i]
        
        pred = ensemble_cal.predict_ensemble(vel, div, ppl)
        if pred == label:
            ensemble_correct += 1
    
    ensemble_cal.ensemble_accuracy = ensemble_correct / total if total > 0 else 0.0
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS (Majority Vote)")
    print(f"  Accuracy: {ensemble_cal.ensemble_accuracy:.2%}")
    print("="*60)
    
    # Save ensemble calibration
    ensemble_path = paths["checkpoints"] / "ensemble_calibration.json"
    with open(ensemble_path, 'w') as f:
        json.dump({
            "velocity_threshold": velocity_cal.optimal_threshold,
            "divergence_threshold": divergence_cal.optimal_threshold,
            "perplexity_threshold": perplexity_cal.optimal_threshold,
            "ensemble_accuracy": ensemble_cal.ensemble_accuracy,
        }, f, indent=2)
    print(f"\n✓ Saved ensemble calibration to {ensemble_path}")
    
    return ensemble_cal


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
    
    import gc
    
    for i, example in enumerate(pbar):
        try:
            book_name = example['book_name']
            
            # Explicit logging for Kaggle (since tqdm might be buffered)
            if i % 5 == 0:
                print(f"Processing example {i+1}/{len(train_examples)}: {book_name}...", flush=True)
            
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
                
                # Force cleanup after heavy streaming
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                        novel_state_baseline=novel_state,  # Use cached state!
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
    print(f"  F1 Score: {calibration.f1:.4f}")
    print(f"  Consistent μ={calibration.consistent_mean:.4f}, σ={calibration.consistent_std:.4f}")
    print(f"  Contradict μ={calibration.contradict_mean:.4f}, σ={calibration.contradict_std:.4f}")
    print(f"\n  Confusion Matrix:")
    print(calibration.print_confusion_matrix())
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
                # Streaming mode: Use cached states for inference
                # (Streaming is only needed during calibration to find threshold)
                if book_name not in novel_states:
                    print(f"\n⚠ Novel state not cached for {book_name}, using default")
                    prediction = 1
                    velocity = 0.0
                else:
                    novel_state = novel_states[book_name]
                    backstory_state, _ = wrapper.prime_with_backstory(
                        example['content'],
                        verbose=False,
                    )
                    velocity = wrapper.compute_velocity_from_states(
                        backstory_state,
                        novel_state,
                        metric=metric,
                    )
                    prediction = calibration.predict(velocity)
            else:
                # Cached approach: compare final states or trajectories
                if book_name not in novel_states:
                    print(f"\n⚠ Novel state not cached for {book_name}, using default")
                    prediction = 1  # Default to consistent
                    velocity = 0.0
                else:
                    novel_data = novel_states[book_name]
                    
                    # Check if it's a trajectory (list) or single state
                    is_trajectory = isinstance(novel_data, list)
                    
                    if args.perturbation:
                        # Perturbation mode
                        novel_path = loader.get_book_path(book_name)
                        # Use last state of trajectory or single state
                        novel_state_baseline = novel_data[-1] if is_trajectory else novel_data
                        velocity = wrapper.compute_perturbation(
                            backstory_text=example['content'],
                            novel_path=novel_path,
                            verbose=False,
                            metric=metric,
                            novel_state_baseline=novel_state_baseline,
                        )
                    else:
                        # Standard cached mode
                        # Process only the backstory
                        backstory_state, _ = wrapper.prime_with_backstory(
                            example['content'],
                            verbose=False,
                        )
                        
                        # Compute velocity (handle both trajectories and single states)
                        if is_trajectory:
                            # Use trajectory-based velocity
                            velocity = wrapper.compute_trajectory_velocity(
                                backstory_state,
                                novel_data,
                                metric=metric,
                            )
                        else:
                            # Use single state velocity
                            velocity = wrapper.compute_velocity_from_states(
                                backstory_state,
                                novel_data,
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
    
    if args.improvise:
        print("\n➡ IMPROVISE MODE ENABLED:")
        print("  • K-fold cross-validation (4 folds, median threshold)")
        if mode == "cached":
            print("  • Multi-checkpoint trajectory caching (25%, 50%, 75%, 100%)")
    
    if args.ensemble:
        print("\n➡ ENSEMBLE MODE ENABLED:")
        print("  • Combining: Velocity + Embedding Divergence + Perplexity")
        print("  • Decision: Majority vote (2/3 signals)")
    
    # Phase 0: Pre-compute novel states or trajectories
    novel_data = {}
    use_trajectories = False
    
    if mode == "cached":
        if args.improvise:
            # Multi-checkpoint trajectories for --improvise
            novel_data = precompute_novel_trajectories(wrapper, loader, paths)
            use_trajectories = True
        else:
            # Single final states (original)
            novel_data = precompute_novel_states(wrapper, loader, paths)
    
    calibration = None
    validation_result = None
    fold_results = None
    ensemble_calibration = None
    
    # Phase 1: Calibration
    if run_train:
        if args.ensemble:
            # Ensemble mode: calibrate all 3 hypotheses
            ensemble_calibration = run_ensemble_calibration(
                wrapper=wrapper,
                loader=loader,
                novel_states=novel_data,
                paths=paths,
                args=args,
                config_name=config_name,
                mode=mode,
                metric=metric,
            )
            # Use velocity calibration as fallback for plots
            calibration = ensemble_calibration.velocity_calibration
        elif args.improvise:
            # K-fold cross-validation for robust threshold
            calibration, fold_results = run_kfold_calibration(
                wrapper=wrapper,
                loader=loader,
                novel_data=novel_data,
                paths=paths,
                args=args,
                config_name=config_name,
                mode=mode,
                metric=metric,
                use_trajectories=use_trajectories,
            )
        else:
            # Standard calibration (60/20 split)
            calibration = run_calibration(
                wrapper=wrapper,
                loader=loader,
                novel_states=novel_data,
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
                    novel_states=novel_data,
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
            novel_states=novel_data,
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

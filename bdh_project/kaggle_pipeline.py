"""
BDH Track B: Kaggle Pipeline

Automated script for running on Kaggle with proper path handling
and dependency installation.
"""

import os
import sys
import subprocess
from pathlib import Path


def is_kaggle() -> bool:
    """Check if running on Kaggle."""
    return os.path.exists("/kaggle/input")


def install_dependencies():
    """Install required dependencies."""
    packages = [
        "torch",
        "pandas",
        "numpy",
        "tqdm",
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "-q"
            ])


def setup_kaggle_paths():
    """Setup Kaggle-specific paths."""
    kaggle_input = Path("/kaggle/input")
    kaggle_working = Path("/kaggle/working")
    
    # Common dataset names on Kaggle
    possible_dataset_paths = [
        kaggle_input / "kharagpur-data-science-hackathon",
        kaggle_input / "kds-hackathon",
        kaggle_input / "bdh-track-b",
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if path.exists():
            dataset_path = path
            break
    
    if dataset_path is None:
        # Check for any directory with required files
        for item in kaggle_input.iterdir():
            if item.is_dir():
                if (item / "train.csv").exists() or (item / "Dataset" / "train.csv").exists():
                    dataset_path = item
                    break
    
    return {
        "input": dataset_path,
        "working": kaggle_working,
    }


def run_kaggle_pipeline(model_size: str = "small"):
    """Run the full pipeline on Kaggle."""
    print("="*60)
    print("BDH TRACK B - KAGGLE PIPELINE")
    print("="*60)
    
    # Install dependencies
    print("\n[1/4] Installing dependencies...")
    install_dependencies()
    
    # Setup paths
    print("\n[2/4] Setting up paths...")
    
    if is_kaggle():
        paths = setup_kaggle_paths()
        
        if paths["input"] is None:
            print("ERROR: Could not find dataset in /kaggle/input/")
            print("Available directories:")
            for item in Path("/kaggle/input").iterdir():
                print(f"  - {item}")
            return
        
        # Create symlinks or copy to expected locations
        working_dir = paths["working"]
        dataset_dir = working_dir / "Dataset"
        
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True)
            
            # Link or copy data files
            source_dataset = paths["input"] / "Dataset"
            if source_dataset.exists():
                os.system(f"cp -r {source_dataset}/* {dataset_dir}/")
            else:
                # Files might be directly in input
                if (paths["input"] / "train.csv").exists():
                    os.system(f"cp {paths['input']}/train.csv {dataset_dir}/")
                    os.system(f"cp {paths['input']}/test.csv {dataset_dir}/")
                if (paths["input"] / "Books").exists():
                    os.system(f"cp -r {paths['input']}/Books {dataset_dir}/")
        
        base_path = working_dir
        output_dir = working_dir / "outputs"
    else:
        # Local development
        base_path = Path(__file__).parent.parent
        output_dir = Path(__file__).parent / "outputs"
    
    print(f"  Base path: {base_path}")
    print(f"  Output: {output_dir}")
    
    # Validate dataset
    print("\n[3/4] Validating dataset...")
    
    required_files = [
        base_path / "Dataset" / "train.csv",
        base_path / "Dataset" / "test.csv",
        base_path / "Dataset" / "Books",
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return
    
    print("  ✓ All required files found")
    
    # Run pipeline
    print("\n[4/4] Running pipeline...")
    
    # Import and run main
    sys.path.insert(0, str(Path(__file__).parent))
    
    from main import (
        setup_directories,
        get_config_by_name,
        InferenceConfig,
        get_device,
        BDHReasoningWrapper,
        DataLoader,
        get_dataset_stats,
        run_calibration,
        run_inference,
        generate_plots,
    )
    
    # Configuration
    config_name = model_size
    model_config = get_config_by_name(config_name)
    inference_config = InferenceConfig()
    device = get_device()
    
    print(f"  Model: {config_name} ({model_config.n_layer} layers)")
    print(f"  Device: {device}")
    
    # Setup
    paths_dict = setup_directories(str(output_dir))
    
    loader = DataLoader(base_path=base_path)
    loader.load_train()
    loader.load_test()
    
    stats = get_dataset_stats(loader)
    print(f"  Train: {stats['train_total']} | Test: {stats['test_total']}")
    
    # Initialize model
    wrapper = BDHReasoningWrapper(
        model_config=model_config,
        inference_config=inference_config,
        device=device,
    )
    
    # Create mock args
    class Args:
        dry_run = False
        limit = None
        max_chunks = None
        verbose = True
    
    args = Args()
    
    # Run calibration
    calibration = run_calibration(
        wrapper=wrapper,
        loader=loader,
        paths=paths_dict,
        args=args,
        config_name=config_name,
    )
    
    # Generate plots
    generate_plots(calibration, paths_dict)
    
    # Run inference
    results = run_inference(
        wrapper=wrapper,
        loader=loader,
        calibration=calibration,
        paths=paths_dict,
        args=args,
    )
    
    print("\n" + "="*60)
    print("KAGGLE PIPELINE COMPLETE")
    print(f"Results saved to: {output_dir / 'results.csv'}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use 4-layer model")
    args = parser.parse_args()
    
    model_size = "small" if args.small else "default"
    run_kaggle_pipeline(model_size)

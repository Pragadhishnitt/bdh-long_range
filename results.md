**!python main.py --mode cached --improvise**

============================================================
K-FOLD CROSS-VALIDATION (4 FOLDS)
============================================================

----------------------------------------
FOLD 1/4
  Train: 60 examples
  Val: 20 examples
----------------------------------------
Fold 1 Train: 100%|█████████████████████████████| 60/60 [00:03<00:00, 16.65it/s]
Fold 1 Val: 100%|███████████████████████████████| 20/20 [00:01<00:00, 17.60it/s]

  Fold 1 Results:
    Velocity Threshold: 1.055161
    Train Acc: 66.67%, F1: 0.7778
    Val Acc: 80.00%

----------------------------------------
FOLD 2/4
  Train: 60 examples
  Val: 20 examples
----------------------------------------
Fold 2 Train: 100%|█████████████████████████████| 60/60 [00:03<00:00, 17.42it/s]
Fold 2 Val: 100%|███████████████████████████████| 20/20 [00:01<00:00, 17.11it/s]

  Fold 2 Results:
    Velocity Threshold: 1.055161
    Train Acc: 70.00%, F1: 0.8043
    Val Acc: 70.00%

----------------------------------------
FOLD 3/4
  Train: 60 examples
  Val: 20 examples
----------------------------------------
Fold 3 Train: 100%|█████████████████████████████| 60/60 [00:03<00:00, 17.27it/s]
Fold 3 Val: 100%|███████████████████████████████| 20/20 [00:01<00:00, 17.49it/s]

  Fold 3 Results:
    Velocity Threshold: 1.055161
    Train Acc: 70.00%, F1: 0.7907
    Val Acc: 70.00%

----------------------------------------
FOLD 4/4
  Train: 60 examples
  Val: 20 examples
----------------------------------------
Fold 4 Train: 100%|█████████████████████████████| 60/60 [00:03<00:00, 17.40it/s]
Fold 4 Val: 100%|███████████████████████████████| 20/20 [00:01<00:00, 17.14it/s]

  Fold 4 Results:
    Velocity Threshold: 1.055161
    Train Acc: 73.33%, F1: 0.8140
    Val Acc: 60.00%

============================================================
K-FOLD AGGREGATE RESULTS
============================================================
  Velocity Thresholds: ['1.0552', '1.0552', '1.0552', '1.0552']
  Median Velocity Threshold: 1.055161
  Train Acc: 70.00% ± 2.36%
  Val Acc: 70.00% ± 7.07%

✓ Saved K-fold results to outputs/checkpoints/kfold_results.json

Generating plots...
✓ Saved: velocity_distribution.png
✓ Saved: velocity_scatter.png

============================================================
PHASE 3: TEST INFERENCE
============================================================
Predicting: 100%|███████████| 60/60 [00:03<00:00, 17.15it/s, pred=1, vel=1.0391]

✓ Saved predictions: outputs/results.csv
✓ Saved detailed results: outputs/results_detailed.csv

────────────────────────────────────────
INFERENCE RESULTS:
  Total predictions: 60
  Consistent (1): 53
  Contradict (0): 7
────────────────────────────────────────

============================================================
"""
zero_shot.py
------------
Reproduces Table 5 zero-shot TimesFM and Chronos rows for all datasets.

Usage:
  python zero_shot.py --dataset weather  --csv data/cleaned_weather.csv
  python zero_shot.py --all
"""

import argparse
import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from guard.data_processors import get_processor
from guard.teachers import TimesFMTeacher, ChronosTeacher

DATASET_CSVS = {
    'weather':      'data/cleaned_weather.csv',
    'etth1':        'data/ETTh1.csv',
    'ettm1':        'data/ETTm1.csv',
    'flux':         'data/monthly_input_output.csv',
    'soilmoisture': 'data/quench_soil_moisture.csv',
}


def evaluate_zero_shot(windows, horizons, teacher, teacher_name):
    """Compute RMSE for a zero-shot teacher across all horizons."""
    results = {}
    for h in horizons:
        preds, labels = [], []
        for w in windows:
            mu, _ = teacher.predict(w['context_X'], h)
            preds.append(mu)
            labels.append(w[f'labels_h{h}'])
        p = np.array(preds)
        l = np.array(labels)
        results[h] = {
            'rmse': float(np.sqrt(np.mean((p - l) ** 2))),
        }
        print(f"  {teacher_name} H={h}: RMSE={results[h]['rmse']:.4f}")
    return results


def run_dataset(dataset: str, csv_path: str, out_dir: str = 'results',
                device: str = None):
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"ZERO-SHOT | {dataset.upper()} | device={device}")
    print(f"{'='*70}")

    processor = get_processor(dataset, csv_path)
    processor.create_splits()
    test_w   = processor.create_windows('test')
    horizons = processor.horizons
    ctx_len  = processor.context_len
    max_h    = max(horizons)

    tf_teacher = TimesFMTeacher(ctx_len, max_h, device)
    ch_teacher = ChronosTeacher(ctx_len, device)

    print("\nTimesFM zero-shot:")
    tf_results = evaluate_zero_shot(test_w, horizons, tf_teacher, 'TimesFM')
    print("\nChronos zero-shot:")
    ch_results = evaluate_zero_shot(test_w, horizons, ch_teacher, 'Chronos')

    # Summary
    print(f"\n{'─'*50}")
    print(f"  {'Model':<12} " + "  ".join([f"H={h}" for h in horizons]) + "  avg")
    print(f"{'─'*50}")
    for name, res in [('TimesFM', tf_results), ('Chronos', ch_results)]:
        rmses = [res[h]['rmse'] for h in horizons]
        cols  = "  ".join([f"{r:.4f}" for r in rmses])
        print(f"  {name:<12} {cols}  {np.mean(rmses):.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out = {'TimesFM': {str(k): v for k, v in tf_results.items()},
           'Chronos': {str(k): v for k, v in ch_results.items()}}
    out_path = os.path.join(out_dir, f'zero_shot_{dataset}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=list(DATASET_CSVS.keys()))
    parser.add_argument('--csv',     type=str, default=None)
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--out-dir', type=str, default='results')
    args = parser.parse_args()

    if args.all:
        for ds, csv in DATASET_CSVS.items():
            if not os.path.exists(csv):
                print(f"⚠️  Skipping {ds}: {csv} not found")
                continue
            run_dataset(ds, csv, args.out_dir)
    elif args.dataset:
        csv = args.csv or DATASET_CSVS[args.dataset]
        run_dataset(args.dataset, csv, args.out_dir)
    else:
        parser.print_help()

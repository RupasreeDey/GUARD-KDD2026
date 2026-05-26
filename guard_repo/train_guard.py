"""
train_guard.py
--------------
Reproduces Table 2 (ablation) and Table 5 (GUARD results) for all datasets.

Usage:
  # Train full GUARD on Weather (Table 5):
  python train_guard.py --dataset weather --csv data/cleaned_weather.csv

  # Run ablation study on ETTh1 (Table 2):
  python train_guard.py --dataset etth1 --csv data/ETTh1.csv --ablation all

  # Run all datasets:
  python train_guard.py --all

  # Hyperparameter sensitivity sweep (Appendix C):
  python train_guard.py --dataset weather --csv data/cleaned_weather.csv --hp-sweep
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from guard.data_processors import get_processor
from guard.teachers import generate_teacher_predictions, generate_teacher_predictions_3teacher
from guard.models import (VotingRouter, TemperatureNetwork,
                           TransformerStudentModel, MultiHorizonDataset,
                           SelectiveAdaptiveKDTrainer,
                           VotingRouter3, TemperatureNetwork3,
                           MultiHorizonDataset3, SelectiveAdaptiveKDTrainer3)

DATASET_CSVS = {
    'weather':      'data/cleaned_weather.csv',
    'etth1':        'data/ETTh1.csv',
    'ettm1':        'data/ETTm1.csv',
    'flux':         'data/monthly_input_output.csv',
    'soilmoisture': 'data/quench_soil_moisture.csv',
}

# ============================================================================
# Training loop
# ============================================================================

def train_guard(dataset, csv_path, ablation='voting_temp',
                epochs=30, batch_size=32, patience=5,
                sample_fraction=1.0, out_dir='results', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"GUARD | {dataset.upper()} | ablation={ablation} | device={device}")
    print(f"{'='*70}")

    # ── data ─────────────────────────────────────────────────────────────────
    processor = get_processor(dataset, csv_path)
    processor.create_splits()
    train_w = processor.create_windows('train', sample_fraction)
    val_w   = processor.create_windows('val')
    test_w  = processor.create_windows('test')
    horizons = processor.horizons
    n_feat   = len(processor.feature_cols)
    ctx_len  = processor.context_len
    max_h    = max(horizons)

    print(f"  Train windows: {len(train_w)} | Val: {len(val_w)} | Test: {len(test_w)}")

    # ── Phase 1: teacher caching ──────────────────────────────────────────────
    phase1_log = {}
    print("\n[Phase 1] Generating teacher predictions...")
    train_w = generate_teacher_predictions(train_w, horizons, device,
                                           phase1_log, 'train')
    val_w   = generate_teacher_predictions(val_w,   horizons, device,
                                           phase1_log, 'val')
    test_w  = generate_teacher_predictions(test_w,  horizons, device,
                                           phase1_log, 'test')

    # ── datasets / loaders ────────────────────────────────────────────────────
    train_ds = MultiHorizonDataset(train_w, horizons, mode='train')
    val_ds   = MultiHorizonDataset(val_w,   horizons, mode='val')
    test_ds  = MultiHorizonDataset(test_w,  horizons, mode='test')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ── model init ────────────────────────────────────────────────────────────
    student       = TransformerStudentModel(n_feat, ctx_len, max_h)
    voting_router = VotingRouter(len(horizons))
    temp_network  = TemperatureNetwork(len(horizons))
    total_params  = sum(p.numel() for p in student.parameters()) + \
                    sum(p.numel() for p in voting_router.parameters()) + \
                    sum(p.numel() for p in temp_network.parameters())
    print(f"\n  Student params:  {sum(p.numel() for p in student.parameters()):,}")
    print(f"  Total params:    {total_params:,}")

    trainer = SelectiveAdaptiveKDTrainer(
        student, voting_router, temp_network,
        horizons, device, ablation=ablation)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val   = float('inf')
    best_states = None
    pat_count  = 0

    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_dl, epoch)
        val_metrics   = trainer.evaluate(val_dl)
        avg_val_rmse  = np.mean([val_metrics[h]['rmse'] for h in horizons])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"loss={train_metrics['total']:.4f} | "
                  f"val_rmse={avg_val_rmse:.4f}")

        if avg_val_rmse < best_val:
            best_val    = avg_val_rmse
            best_states = {
                'student':  {k: v.clone() for k, v in student.state_dict().items()},
                'router':   {k: v.clone() for k, v in voting_router.state_dict().items()},
                'temp_net': {k: v.clone() for k, v in temp_network.state_dict().items()},
            }
            pat_count = 0
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    student.load_state_dict(best_states['student'])
    voting_router.load_state_dict(best_states['router'])
    temp_network.load_state_dict(best_states['temp_net'])

    # ── test evaluation ───────────────────────────────────────────────────────
    test_metrics = trainer.evaluate(test_dl)

    print(f"\n  Test Results [{ablation}]:")
    print(f"  {'Horizon':<10} {'RMSE':<10} {'Normal':<10} {'Extreme':<10}")
    print(f"  {'-'*40}")
    for h in horizons:
        m = test_metrics[h]
        print(f"  {h:<10} {m['rmse']:<10.4f} "
              f"{m['rmse_normal']:<10.4f} {m['rmse_extreme']:<10.4f}")
    avg = np.mean([test_metrics[h]['rmse'] for h in horizons])
    print(f"\n  Average RMSE: {avg:.4f}")

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'guard_{dataset}_{ablation}.json')
    with open(out_path, 'w') as f:
        json.dump({str(k): v for k, v in test_metrics.items()}, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    # save Phase 1 log
    if phase1_log:
        p1_path = os.path.join(out_dir, f'phase1_cost_{dataset}.json')
        with open(p1_path, 'w') as f:
            json.dump(phase1_log, f, indent=2)
        print(f"✓ Phase 1 log saved to {p1_path}")

    return test_metrics, trainer


# ============================================================================
# Hyperparameter sensitivity sweep (Appendix C)
# ============================================================================

HP_CONFIGS = [
    {'alpha': 1.0, 'beta': 0.3,  'epsilon': 0.15, 'label': 'default'},
    {'alpha': 1.0, 'beta': 0.1,  'epsilon': 0.15, 'label': 'weak_kd'},
    {'alpha': 1.0, 'beta': 0.5,  'epsilon': 0.15, 'label': 'strong_kd'},
    {'alpha': 1.0, 'beta': 0.3,  'epsilon': 0.05, 'label': 'weak_entropy'},
    {'alpha': 1.0, 'beta': 0.3,  'epsilon': 0.30, 'label': 'strong_entropy'},
    {'alpha': 1.0, 'beta': 0.1,  'epsilon': 0.05, 'label': 'weak_reg'},
]


def run_hp_sweep(dataset, csv_path, epochs=30, batch_size=32,
                 patience=5, out_dir='results', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"HP SWEEP | {dataset.upper()}")
    print(f"{'='*70}")

    processor = get_processor(dataset, csv_path)
    processor.create_splits()
    train_w  = processor.create_windows('train')
    val_w    = processor.create_windows('val')
    test_w   = processor.create_windows('test')
    horizons = processor.horizons
    n_feat   = len(processor.feature_cols)
    ctx_len  = processor.context_len
    max_h    = max(horizons)

    train_w = generate_teacher_predictions(train_w, horizons, device)
    val_w   = generate_teacher_predictions(val_w,   horizons, device)
    test_w  = generate_teacher_predictions(test_w,  horizons, device)

    train_dl = DataLoader(MultiHorizonDataset(train_w, horizons, 'train'),
                          batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(MultiHorizonDataset(val_w,   horizons, 'val'),
                          batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(MultiHorizonDataset(test_w,  horizons, 'test'),
                          batch_size=batch_size, shuffle=False, num_workers=0)

    hp_results = []
    for cfg in HP_CONFIGS:
        print(f"\n  Config: {cfg['label']} "
              f"(α={cfg['alpha']} β={cfg['beta']} ε={cfg['epsilon']})")

        student       = TransformerStudentModel(n_feat, ctx_len, max_h)
        voting_router = VotingRouter(len(horizons))
        temp_network  = TemperatureNetwork(len(horizons))
        trainer = SelectiveAdaptiveKDTrainer(
            student, voting_router, temp_network,
            horizons, device, ablation='voting_temp')
        trainer.alpha   = cfg['alpha']
        trainer.beta    = cfg['beta']
        trainer.epsilon = cfg['epsilon']

        best_val  = float('inf')
        best_sts  = None
        pat       = 0
        for epoch in range(epochs):
            trainer.train_epoch(train_dl, epoch)
            vm  = trainer.evaluate(val_dl)
            avg = np.mean([vm[h]['rmse'] for h in horizons])
            if avg < best_val:
                best_val = avg
                best_sts = {
                    'student':  {k: v.clone() for k, v in student.state_dict().items()},
                    'router':   {k: v.clone() for k, v in voting_router.state_dict().items()},
                    'temp_net': {k: v.clone() for k, v in temp_network.state_dict().items()},
                }
                pat = 0
            else:
                pat += 1
                if pat >= patience:
                    break

        student.load_state_dict(best_sts['student'])
        voting_router.load_state_dict(best_sts['router'])
        temp_network.load_state_dict(best_sts['temp_net'])
        tm = trainer.evaluate(test_dl)

        row = {**cfg}
        for h in horizons:
            row[f'rmse_h{h}'] = round(tm[h]['rmse'], 4)
        row['avg_rmse'] = round(np.mean([tm[h]['rmse'] for h in horizons]), 4)
        hp_results.append(row)
        print(f"    avg RMSE: {row['avg_rmse']:.4f}")

    # Print table
    print(f"\n{'─'*70}")
    print(f"  {'Config':<18} {'β':<6} {'ε':<6} " +
          " ".join([f"H={h}" for h in horizons]) + "  avg")
    print(f"{'─'*70}")
    default_rmse = hp_results[0]['avg_rmse']
    for r in hp_results:
        rmses = " ".join([f"{r[f'rmse_h{h}']:.4f}" for h in horizons])
        dev   = f"  ({100*(r['avg_rmse']-default_rmse)/default_rmse:+.1f}%)" \
                if r['label'] != 'default' else "  [default]"
        print(f"  {r['label']:<18} {r['beta']:<6} {r['epsilon']:<6} "
              f"{rmses}  {r['avg_rmse']:.4f}{dev}")

    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(hp_results)
    out_path = os.path.join(out_dir, f'hp_sweep_{dataset}.csv')
    df.to_csv(out_path, index=False)
    print(f"\n✓ HP sweep saved to {out_path}")
    return hp_results


# ============================================================================
# 3-Teacher GUARD (Section 7: multi-teacher scalability)
# ============================================================================

def train_guard_3teacher(dataset, csv_path, epochs=30, batch_size=32,
                         patience=5, sample_fraction=1.0,
                         out_dir='results', device=None):
    """
    Train 3-teacher GUARD (TimesFM + Chronos + Moirai).
    Reproduces Section 7 routing weight table and teacher selection criterion.

    Note: Phase 1 takes ~15 hrs on RTX 3090 for Weather (all splits).
    Use --sample-fraction 0.3 for a quick test run.
    To reduce Moirai cost ~6×: set MoiraiTeacher.MOIRAI_SIZE = 'small'
    in guard/teachers.py before running.
    """
    from guard.teachers import MoiraiTeacher
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"GUARD-3T | {dataset.upper()} | device={device}")
    print(f"  Moirai size: {MoiraiTeacher.MOIRAI_SIZE}")
    print(f"  Tip: set MoiraiTeacher.MOIRAI_SIZE='small' to reduce Phase 1 ~6×")
    print(f"{'='*70}")

    processor = get_processor(dataset, csv_path)
    processor.create_splits()
    train_w  = processor.create_windows('train', sample_fraction)
    val_w    = processor.create_windows('val')
    test_w   = processor.create_windows('test')
    horizons = processor.horizons
    n_feat   = len(processor.feature_cols)
    ctx_len  = processor.context_len
    max_h    = max(horizons)

    print(f"  Train: {len(train_w)} | Val: {len(val_w)} | Test: {len(test_w)}")

    # Phase 1: run once per split, in order
    phase1_log = {}
    train_w = generate_teacher_predictions_3teacher(
        train_w, horizons, device, phase1_log, 'train')
    val_w   = generate_teacher_predictions_3teacher(
        val_w,   horizons, device, phase1_log, 'val')
    test_w  = generate_teacher_predictions_3teacher(
        test_w,  horizons, device, phase1_log, 'test')

    train_dl = DataLoader(MultiHorizonDataset3(train_w, horizons, 'train'),
                          batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(MultiHorizonDataset3(val_w,   horizons, 'val'),
                          batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(MultiHorizonDataset3(test_w,  horizons, 'test'),
                          batch_size=batch_size, shuffle=False, num_workers=0)

    student  = TransformerStudentModel(n_feat, ctx_len, max_h)
    router   = VotingRouter3(len(horizons))
    temp_net = TemperatureNetwork3(len(horizons))
    total_params = (sum(p.numel() for p in student.parameters()) +
                    sum(p.numel() for p in router.parameters()) +
                    sum(p.numel() for p in temp_net.parameters()))
    print(f"  Total params: {total_params:,}")

    trainer  = SelectiveAdaptiveKDTrainer3(
        student, router, temp_net, horizons, device)

    best_val, best_states, pat_count = float('inf'), None, 0
    for epoch in range(epochs):
        tm = trainer.train_epoch(train_dl, epoch)
        vm = trainer.evaluate(val_dl)
        avg_val = np.mean([vm[h]['rmse'] for h in horizons])
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"loss={tm['total']:.4f} | val={avg_val:.4f}")
        if avg_val < best_val:
            best_val    = avg_val
            best_states = {
                'student': {k: v.clone() for k, v in student.state_dict().items()},
                'router':  {k: v.clone() for k, v in router.state_dict().items()},
                'temp':    {k: v.clone() for k, v in temp_net.state_dict().items()},
            }
            pat_count = 0
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    student.load_state_dict(best_states['student'])
    router.load_state_dict(best_states['router'])
    temp_net.load_state_dict(best_states['temp'])

    test_metrics = trainer.evaluate(test_dl)
    print(f"\n  Test Results [3-teacher GUARD]:")
    print(f"  {'H':<6} {'RMSE':<10} {'CH-w':<8} {'TF-w':<8} {'MO-w':<8}")
    print(f"  {'-'*44}")
    for h in horizons:
        m = test_metrics[h]
        print(f"  {h:<6} {m['rmse']:<10.4f} "
              f"{m['avg_ch_weight']:.3f}    {m['avg_tf_weight']:.3f}    "
              f"{m['avg_mo_weight']:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'guard_{dataset}_3teacher.json')
    with open(out_path, 'w') as f:
        json.dump({str(k): v for k, v in test_metrics.items()}, f, indent=2)
    p1_path = os.path.join(out_dir, f'phase1_cost_{dataset}_3teacher.json')
    with open(p1_path, 'w') as f:
        json.dump(phase1_log, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")
    print(f"✓ Phase 1 log saved to {p1_path}")
    return test_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GUARD')
    parser.add_argument('--dataset',  type=str, choices=list(DATASET_CSVS.keys()))
    parser.add_argument('--csv',      type=str, default=None)
    parser.add_argument('--ablation', type=str, default='voting_temp',
                        choices=['base', 'voting_only', 'voting_temp', 'all'],
                        help='"all" runs all three ablation modes (Table 2)')
    parser.add_argument('--all',      action='store_true',
                        help='Run all datasets with full GUARD')
    parser.add_argument('--three-teacher', action='store_true',
                        help='Run 3-teacher GUARD with Moirai (Section 7)')
    parser.add_argument('--hp-sweep', action='store_true',
                        help='Run hyperparameter sensitivity sweep (Appendix C)')
    parser.add_argument('--epochs',   type=int,   default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int,   default=5)
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--out-dir',  type=str,   default='results')
    args = parser.parse_args()

    if args.all:
        for ds, csv in DATASET_CSVS.items():
            if not os.path.exists(csv):
                print(f"⚠️  Skipping {ds}: {csv} not found")
                continue
            train_guard(ds, csv, ablation='voting_temp',
                        epochs=args.epochs, batch_size=args.batch_size,
                        patience=args.patience, out_dir=args.out_dir,
                        sample_fraction=args.sample_fraction)
    elif args.dataset:
        csv = args.csv or DATASET_CSVS[args.dataset]
        if args.three_teacher:
            train_guard_3teacher(args.dataset, csv,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 patience=args.patience, out_dir=args.out_dir,
                                 sample_fraction=args.sample_fraction)
        elif args.hp_sweep:
            run_hp_sweep(args.dataset, csv, args.epochs, args.batch_size,
                         args.patience, args.out_dir)
        elif args.ablation == 'all':
            for abl in SelectiveAdaptiveKDTrainer.ABLATION_MODES:
                train_guard(args.dataset, csv, ablation=abl,
                            epochs=args.epochs, batch_size=args.batch_size,
                            patience=args.patience, out_dir=args.out_dir,
                            sample_fraction=args.sample_fraction)
        else:
            train_guard(args.dataset, csv, ablation=args.ablation,
                        epochs=args.epochs, batch_size=args.batch_size,
                        patience=args.patience, out_dir=args.out_dir,
                        sample_fraction=args.sample_fraction)
    else:
        parser.print_help()

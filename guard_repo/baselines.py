"""
baselines.py
------------
Reproduces Table 5 deep learning baseline results for all datasets:
  DLinear | Transformer | PatchTST | iTransformer

Usage:
  python baselines.py --dataset weather  --csv data/cleaned_weather.csv
  python baselines.py --dataset etth1    --csv data/ETTh1.csv
  python baselines.py --dataset ettm1   --csv data/ETTm1.csv
  python baselines.py --dataset flux    --csv data/monthly_input_output.csv
  python baselines.py --dataset soilmoisture --csv data/quench_soil_moisture.csv
  python baselines.py --all              # run all datasets (requires all CSVs in data/)
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from guard.data_processors import get_processor

# ============================================================================
# Baseline models
# ============================================================================

class DLinear(nn.Module):
    def __init__(self, context_len, n_features, horizon):
        super().__init__()
        self.linear = nn.Linear(context_len * n_features, horizon)

    def forward(self, x):
        return self.linear(x.reshape(x.size(0), -1))


class SimpleTransformerTS(nn.Module):
    def __init__(self, n_features, context_len, horizon,
                 d_model=128, n_heads=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=0.1,
            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1),)
        self.fc   = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Linear(d_model, horizon))

    def forward(self, x):
        enc = self.encoder(self.input_proj(x))         # (B, L, d)
        pooled = enc.mean(dim=1)                       # (B, d)
        return self.fc(pooled)


class PatchTSTBaseline(nn.Module):
    def __init__(self, n_features, context_len, horizon,
                 patch_len=16, stride=8, d_model=128, n_heads=4,
                 num_layers=2, dim_feedforward=256):
        super().__init__()
        self.patch_len   = patch_len
        self.stride      = stride
        self.n_features  = n_features
        num_patches      = (context_len - patch_len) // stride + 1
        self.patch_proj  = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=0.1,
            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                     nn.Linear(d_model, horizon))

    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L)
        patches = torch.stack([x[:, s:s+self.patch_len]
                               for s in range(0, L - self.patch_len + 1, self.stride)], dim=1)
        enc     = self.encoder(self.patch_proj(patches))
        pooled  = enc.mean(dim=1).reshape(B, C, -1).mean(dim=1)
        return self.head(pooled)


class iTransformerBaseline(nn.Module):
    def __init__(self, n_features, context_len, horizon,
                 d_model=128, n_heads=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.embedding         = nn.Linear(context_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=0.1,
            batch_first=True, activation='gelu')
        self.encoder           = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.reverse_embedding = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (B, L, C) → iTransformer: attention across channels
        embedded = self.embedding(x.permute(0, 2, 1))   # (B, C, d)
        encoded  = self.encoder(embedded)               # (B, C, d)
        return self.reverse_embedding(encoded).mean(dim=1)  # (B, H)


MODELS = {
    'DLinear':      DLinear,
    'Transformer':  SimpleTransformerTS,
    'PatchTST':     PatchTSTBaseline,
    'iTransformer': iTransformerBaseline,
}

# ============================================================================
# Dataset
# ============================================================================

class SingleHorizonDataset(Dataset):
    def __init__(self, windows, horizon):
        self.windows = windows
        self.horizon = horizon

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return {
            'context': torch.FloatTensor(w['context_X']),
            'labels':  torch.FloatTensor(w[f'labels_h{self.horizon}']),
            'regime':  torch.tensor(
                1.0 if w[f'regime_h{self.horizon}'] == 'extreme' else 0.0,
                dtype=torch.float32),
        }


# ============================================================================
# Train / evaluate
# ============================================================================

def train_and_eval(model, train_loader, val_loader, test_loader,
                   epochs, device, patience=10):
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)
    best_val  = float('inf')
    best_state = None
    pat_count  = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            ctx  = batch['context'].to(device)
            lbl  = batch['labels'].to(device)
            loss = ((model(ctx) - lbl) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                val_preds.append(model(batch['context'].to(device)).cpu().numpy())
                val_labels.append(batch['labels'].numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(val_preds) - np.concatenate(val_labels)) ** 2)))
        scheduler.step(val_rmse)

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_count  = 0
        else:
            pat_count += 1
            if pat_count >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, labels, regimes = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            preds.append(model(batch['context'].to(device)).cpu().numpy())
            labels.append(batch['labels'].numpy())
            regimes.append(batch['regime'].numpy())

    preds   = np.concatenate(preds)
    labels  = np.concatenate(labels)
    regimes = np.concatenate(regimes)
    nm, em  = regimes == 0, regimes == 1

    return {
        'rmse':         float(np.sqrt(np.mean((preds - labels) ** 2))),
        'rmse_normal':  float(np.sqrt(np.mean((preds[nm] - labels[nm]) ** 2))) if nm.sum() else 0.,
        'rmse_extreme': float(np.sqrt(np.mean((preds[em] - labels[em]) ** 2))) if em.sum() else 0.,
    }


# ============================================================================
# Per-dataset runner
# ============================================================================

def run_dataset(dataset: str, csv_path: str, out_dir: str = 'results',
                epochs: int = 30, batch_size: int = 32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"BASELINES | {dataset.upper()} | device={device}")
    print(f"{'='*70}")

    processor = get_processor(dataset, csv_path)
    processor.create_splits()
    train_w = processor.create_windows('train')
    val_w   = processor.create_windows('val')
    test_w  = processor.create_windows('test')
    horizons = processor.horizons
    n_feat   = len(processor.feature_cols)
    ctx_len  = processor.context_len

    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for h in horizons:
        tr_dl = DataLoader(SingleHorizonDataset(train_w, h), batch_size=batch_size,
                           shuffle=True,  num_workers=0)
        va_dl = DataLoader(SingleHorizonDataset(val_w,   h), batch_size=batch_size,
                           shuffle=False, num_workers=0)
        te_dl = DataLoader(SingleHorizonDataset(test_w,  h), batch_size=batch_size,
                           shuffle=False, num_workers=0)

        all_results[h] = {}
        for name, ModelCls in MODELS.items():
            print(f"\n  [{name}] horizon={h}")
            if name == 'DLinear':
                model = ModelCls(ctx_len, n_feat, h).to(device)
            else:
                model = ModelCls(n_feat, ctx_len, h).to(device)
            metrics = train_and_eval(model, tr_dl, va_dl, te_dl, epochs, device)
            all_results[h][name] = metrics
            print(f"    RMSE={metrics['rmse']:.4f}  "
                  f"normal={metrics['rmse_normal']:.4f}  "
                  f"extreme={metrics['rmse_extreme']:.4f}")

    # Print summary table
    print(f"\n{'─'*70}")
    print(f"  {'Model':<16} " +
          "  ".join([f"H={h}" for h in horizons]) + "  avg")
    print(f"{'─'*70}")
    for name in MODELS:
        rmses = [all_results[h][name]['rmse'] for h in horizons]
        cols  = "  ".join([f"{r:.4f}" for r in rmses])
        print(f"  {name:<16} {cols}  {np.mean(rmses):.4f}")

    out_path = os.path.join(out_dir, f'baselines_{dataset}.json')
    with open(out_path, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\n✓ Saved results to {out_path}")
    return all_results


# ============================================================================
# CLI
# ============================================================================

DATASET_CSVS = {
    'weather':     'data/cleaned_weather.csv',
    'etth1':       'data/ETTh1.csv',
    'ettm1':       'data/ETTm1.csv',
    'flux':        'data/monthly_input_output.csv',
    'soilmoisture': 'data/quench_soil_moisture.csv',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GUARD baseline experiments')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=list(DATASET_CSVS.keys()),
                        help='Dataset to run (omit with --all)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV (overrides default)')
    parser.add_argument('--all', action='store_true',
                        help='Run all datasets')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out-dir', type=str, default='results')
    args = parser.parse_args()

    if args.all:
        for ds, csv in DATASET_CSVS.items():
            if not os.path.exists(csv):
                print(f"⚠️  Skipping {ds}: {csv} not found")
                continue
            run_dataset(ds, csv, args.out_dir, args.epochs, args.batch_size)
    elif args.dataset:
        csv = args.csv or DATASET_CSVS[args.dataset]
        run_dataset(args.dataset, csv, args.out_dir, args.epochs, args.batch_size)
    else:
        parser.print_help()

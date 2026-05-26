"""
models.py
---------
Core GUARD model components:
  - VotingRouter        : regime-aware teacher weighting
  - TemperatureNetwork  : uncertainty-gated circuit breaker (softplus, non-saturating)
  - TransformerStudentModel : lightweight ~0.3M param student
  - MultiHorizonDataset : PyTorch dataset for multi-horizon windows
  - SelectiveAdaptiveKDTrainer : joint training with ablation support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


# ============================================================================
# VotingRouter
# ============================================================================

class VotingRouter(nn.Module):
    """
    Regime-aware mixing router for two teachers (TimesFM + Chronos).
    Input features: regime flag, rolling stats (3), teacher uncertainty
    scalars (2), learned horizon embedding (8). Total: 15.
    Output: softmax weights [w_ch, w_tf].
    """

    def __init__(self, n_horizons: int, hidden_dim: int = 64):
        super().__init__()
        self.horizon_embedding = nn.Embedding(n_horizons, 8)
        # 1 + 3 + 2 + 8 = 14
        self.net = nn.Sequential(
            nn.Linear(1 + 3 + 2 + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, regime: torch.Tensor, stats: torch.Tensor,
                tf_std_mean: torch.Tensor, ch_std_mean: torch.Tensor,
                horizon_indices: torch.Tensor) -> torch.Tensor:
        horizon_emb = self.horizon_embedding(horizon_indices)
        x = torch.cat([
            regime.unsqueeze(1),       # [B, 1]
            stats,                     # [B, 3]
            tf_std_mean.unsqueeze(1),  # [B, 1]
            ch_std_mean.unsqueeze(1),  # [B, 1]
            horizon_emb,               # [B, 8]
        ], dim=1)
        return self.net(x)  # [B, 2]: [w_ch, w_tf]


# ============================================================================
# TemperatureNetwork
# ============================================================================

class TemperatureNetwork(nn.Module):
    """
    Uncertainty-gated adaptive temperature (circuit breaker).

    Uses softplus activation: T = 0.5 + softplus(logit).
    - Unbounded upper range: aggressive attenuation in high-uncertainty
      regimes (e.g., T > 6000 on Flux data) without saturation.
    - 0.5 floor: prevents over-softening of reliable short-horizon signals.

    Output: per-teacher temperatures [T_ch, T_tf].
    """

    def __init__(self, n_horizons: int, hidden_dim: int = 32):
        super().__init__()
        self.horizon_embedding = nn.Embedding(n_horizons, 4)
        # 1 + 2 + 2 + 4 = 9
        self.net = nn.Sequential(
            nn.Linear(1 + 2 + 2 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Softplus floor: T >= 0.5
        self.temp_floor = 0.5

    def forward(self, regime: torch.Tensor, voting_weights: torch.Tensor,
                tf_std_mean: torch.Tensor, ch_std_mean: torch.Tensor,
                horizon_indices: torch.Tensor) -> torch.Tensor:
        horizon_emb = self.horizon_embedding(horizon_indices)
        x = torch.cat([
            regime.unsqueeze(1),       # [B, 1]
            voting_weights,            # [B, 2]
            tf_std_mean.unsqueeze(1),  # [B, 1]
            ch_std_mean.unsqueeze(1),  # [B, 1]
            horizon_emb,               # [B, 4]
        ], dim=1)
        logits = self.net(x)
        # Non-saturating: T = 0.5 + softplus(logit), unbounded upper range
        return self.temp_floor + torch.nn.functional.softplus(logits)  # [B, 2]


# ============================================================================
# TransformerStudentModel
# ============================================================================

class TransformerStudentModel(nn.Module):
    """
    Compact 2-layer Transformer student (~0.3M parameters).
    d_model=128, nhead=4, dim_feedforward=256.
    Maps (B, L, F) context windows to (B, H_max) multi-horizon forecasts.
    """

    def __init__(self, n_features: int, context_len: int, max_horizon: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256):
        super().__init__()
        self.max_horizon = max_horizon
        self.context_len = context_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb    = nn.Embedding(context_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=False, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        x_proj   = self.input_proj(x)
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x_proj   = x_proj + self.pos_emb(positions)
        x_enc    = self.encoder(x_proj.transpose(0, 1)).transpose(0, 1)
        return self.head(x_enc.mean(dim=1))


# ============================================================================
# MultiHorizonDataset
# ============================================================================

class MultiHorizonDataset(Dataset):
    """
    PyTorch dataset wrapping cached teacher windows.
    train mode: iterates (window × horizon) pairs.
    val/test mode: iterates windows, returns max horizon.
    """

    def __init__(self, windows: List[Dict], horizons: List[int],
                 mode: str = 'train'):
        self.windows        = windows
        self.horizons       = horizons
        self.mode           = mode
        self.horizon_to_idx = {h: i for i, h in enumerate(horizons)}

    def __len__(self):
        return len(self.windows) * len(self.horizons) if self.mode == 'train' \
               else len(self.windows)

    def __getitem__(self, idx):
        if self.mode == 'train':
            window_idx  = idx % len(self.windows)
            horizon_idx = idx // len(self.windows)
            horizon     = self.horizons[horizon_idx]
        else:
            window_idx = idx
            horizon    = max(self.horizons)

        w     = self.windows[window_idx]
        max_h = max(self.horizons)

        labels  = np.zeros(max_h, dtype=np.float32)
        labels[:len(w[f'labels_h{horizon}'])] = w[f'labels_h{horizon}']

        tf_mean = np.zeros(max_h, dtype=np.float32)
        tf_std  = np.zeros(max_h, dtype=np.float32)
        ch_mean = np.zeros(max_h, dtype=np.float32)
        ch_std  = np.zeros(max_h, dtype=np.float32)

        tf_mean[:horizon] = w[f'tf_mean_h{horizon}']
        tf_std[:horizon]  = w[f'tf_std_h{horizon}']
        ch_mean[:horizon] = w[f'ch_mean_h{horizon}']
        ch_std[:horizon]  = w[f'ch_std_h{horizon}']

        return {
            'context':      torch.FloatTensor(w['context_X']),
            'labels':       torch.FloatTensor(labels),
            'tf_mean':      torch.FloatTensor(tf_mean),
            'tf_std':       torch.FloatTensor(tf_std),
            'ch_mean':      torch.FloatTensor(ch_mean),
            'ch_std':       torch.FloatTensor(ch_std),
            'regime':       torch.tensor(
                                1.0 if w[f'regime_h{horizon}'] == 'extreme' else 0.0,
                                dtype=torch.float32),
            'last_value':   torch.tensor(w[f'stats_h{horizon}']['last_value'],  dtype=torch.float32),
            'rolling_std':  torch.tensor(w[f'stats_h{horizon}']['rolling_std'], dtype=torch.float32),
            'trend':        torch.tensor(w[f'stats_h{horizon}']['trend'],        dtype=torch.float32),
            'horizon':      horizon,
            'horizon_idx':  self.horizon_to_idx[horizon],
            'horizon_mask': torch.FloatTensor([1.0]*horizon + [0.0]*(max_h - horizon)),
            'vote_tf':      torch.tensor(w.get(f'vote_tf_h{horizon}', 0.5), dtype=torch.float32),
            'tf_std_mean':  torch.tensor(w[f'tf_std_mean_h{horizon}'], dtype=torch.float32),
            'ch_std_mean':  torch.tensor(w[f'ch_std_mean_h{horizon}'], dtype=torch.float32),
            'current_idx':  w['current_idx'],
        }


# ============================================================================
# SelectiveAdaptiveKDTrainer
# ============================================================================

class SelectiveAdaptiveKDTrainer:
    """
    Joint trainer for student + VotingRouter + TemperatureNetwork.

    ablation options (reproduces Table 2):
      'base'        -- fixed 50/50 weights, no temperature scaling
      'voting_only' -- learned router, fixed temperatures
      'voting_temp' -- learned router + adaptive temperatures  (= GUARD)
    """

    ABLATION_MODES = ('base', 'voting_only', 'voting_temp')

    def __init__(self, student: TransformerStudentModel,
                 voting_router: VotingRouter,
                 temp_network: TemperatureNetwork,
                 horizons: List[int],
                 device: str = 'cpu',
                 ablation: str = 'voting_temp'):
        assert ablation in self.ABLATION_MODES, \
            f"ablation must be one of {self.ABLATION_MODES}"

        self.student       = student.to(device)
        self.voting_router = voting_router.to(device)
        self.temp_network  = temp_network.to(device)
        self.horizons      = horizons
        self.horizon_to_idx = {h: i for i, h in enumerate(horizons)}
        self.device        = device
        self.ablation      = ablation

        # Loss weights (Section 5.4 / Appendix C)
        self.alpha   = 1.0   # forecast loss
        self.beta    = 0.3   # KD loss
        self.epsilon = 0.15  # entropy regularisation

        self.optimizer = optim.Adam(
            list(student.parameters()) +
            list(voting_router.parameters()) +
            list(temp_network.parameters()),
            lr=1e-3,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)

    # ---- helpers ------------------------------------------------------------

    def _router_inputs(self, batch):
        regime      = batch['regime'].to(self.device)
        tf_std_mean = batch['tf_std_mean'].to(self.device).float()
        ch_std_mean = batch['ch_std_mean'].to(self.device).float()
        stats = torch.stack([
            batch['last_value'].to(self.device),
            batch['rolling_std'].to(self.device),
            batch['trend'].to(self.device),
        ], dim=1).float()
        h_idx = torch.LongTensor(
            [self.horizon_to_idx[h] for h in batch['horizon'].numpy()]
        ).to(self.device)
        return regime, stats, tf_std_mean, ch_std_mean, h_idx

    def _aggregate_teacher(self, batch, weights, horizon_mask):
        """Law-of-total-variance aggregation (Section 4.7)."""
        w_ch = weights[:, 0].unsqueeze(1)
        w_tf = weights[:, 1].unsqueeze(1)
        ch_m = batch['ch_mean'].to(self.device)
        tf_m = batch['tf_mean'].to(self.device)
        mu_agg = w_ch * ch_m + w_tf * tf_m
        var_ch = batch['ch_std'].to(self.device) ** 2
        var_tf = batch['tf_std'].to(self.device) ** 2
        sigma_agg = torch.sqrt(
            w_ch * var_ch + w_tf * var_tf +
            w_ch * w_tf * (ch_m - tf_m) ** 2
        )
        return mu_agg * horizon_mask, sigma_agg * horizon_mask

    # ---- training -----------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        self.student.train()
        self.voting_router.train()
        self.temp_network.train()

        totals = dict(total=0., forecast=0., kd=0., entropy=0.)

        for batch in dataloader:
            context      = batch['context'].to(self.device)
            labels       = batch['labels'].to(self.device)
            regime, stats, tf_std_mean, ch_std_mean, h_idx = self._router_inputs(batch)
            horizon_mask = batch['horizon_mask'].to(self.device)

            student_pred   = self.student(context)
            voting_weights = self.voting_router(regime, stats, tf_std_mean, ch_std_mean, h_idx)

            if self.ablation == 'base':
                temps = torch.ones(len(voting_weights), 2, device=self.device)
            else:
                temps = self.temp_network(regime, voting_weights,
                                          tf_std_mean, ch_std_mean, h_idx)

            masked_pred   = student_pred * horizon_mask
            masked_labels = labels       * horizon_mask
            n_valid       = horizon_mask.sum()

            forecast_loss = ((masked_pred - masked_labels) ** 2).sum() / n_valid

            if self.ablation == 'base':
                mu_agg = (0.5 * batch['tf_mean'].to(self.device) +
                          0.5 * batch['ch_mean'].to(self.device)) * horizon_mask
                kd_loss = ((masked_pred - mu_agg) ** 2).sum() / n_valid
            else:
                mu_agg, _ = self._aggregate_teacher(batch, voting_weights, horizon_mask)
                kd_loss   = ((masked_pred - mu_agg) ** 2).sum() / n_valid
                if self.ablation == 'voting_temp':
                    avg_temp = (voting_weights[:, 0] * temps[:, 0] +
                                voting_weights[:, 1] * temps[:, 1]).mean()
                    kd_loss  = kd_loss / (avg_temp ** 2 + 1e-8)

            entropy_loss = torch.tensor(0., device=self.device)
            if self.ablation != 'base':
                ent          = -(voting_weights * torch.log(voting_weights + 1e-8)).sum(dim=1)
                entropy_loss = -ent.mean()

            regime_w  = (1.0 + 2.0 * regime).mean()
            loss      = (self.alpha * forecast_loss * regime_w +
                         self.beta * kd_loss +
                         self.epsilon * entropy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            for m in (self.student, self.voting_router, self.temp_network):
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.optimizer.step()

            totals['total']    += loss.item()
            totals['forecast'] += forecast_loss.item()
            totals['kd']       += kd_loss.item()
            totals['entropy']  += entropy_loss.item()

        n = len(dataloader)
        self.scheduler.step(totals['total'] / n)
        return {k: v / n for k, v in totals.items()}

    # ---- evaluation ---------------------------------------------------------

    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.student.eval()
        self.voting_router.eval()
        self.temp_network.eval()

        store = {h: {'preds': [], 'labels': [], 'regimes': [],
                     'weights': [], 'temps': []}
                 for h in self.horizons}

        with torch.no_grad():
            for batch in dataloader:
                context      = batch['context'].to(self.device)
                pred         = self.student(context).cpu().numpy()
                regime, stats, tf_std_mean, ch_std_mean, h_idx = self._router_inputs(batch)

                weights = self.voting_router(
                    regime, stats, tf_std_mean, ch_std_mean, h_idx).cpu().numpy()

                if self.ablation == 'voting_temp':
                    w_t   = torch.FloatTensor(weights).to(self.device)
                    temps = self.temp_network(
                        regime, w_t, tf_std_mean, ch_std_mean, h_idx).cpu().numpy()
                else:
                    temps = np.ones_like(weights)

                for h in self.horizons:
                    store[h]['preds'].append(pred[:, :h])
                    store[h]['labels'].append(batch['labels'].numpy()[:, :h])
                    store[h]['regimes'].append(batch['regime'].numpy())
                    store[h]['weights'].append(weights)
                    store[h]['temps'].append(temps)

        metrics = {}
        for h in self.horizons:
            preds   = np.concatenate(store[h]['preds'],   axis=0)
            labels  = np.concatenate(store[h]['labels'],  axis=0)
            regimes = np.concatenate(store[h]['regimes'], axis=0)
            weights = np.concatenate(store[h]['weights'], axis=0)
            temps_  = np.concatenate(store[h]['temps'],   axis=0)

            nm = regimes == 0
            em = regimes == 1

            metrics[h] = {
                'rmse':         float(np.sqrt(np.mean((preds - labels) ** 2))),
                'rmse_normal':  float(np.sqrt(np.mean((preds[nm] - labels[nm]) ** 2)))
                                if nm.sum() > 0 else 0.,
                'rmse_extreme': float(np.sqrt(np.mean((preds[em] - labels[em]) ** 2)))
                                if em.sum() > 0 else 0.,
                'num_extreme':  int(em.sum()),
                'num_normal':   int(nm.sum()),
                'avg_ch_weight': float(weights[:, 0].mean()),
                'avg_tf_weight': float(weights[:, 1].mean()),
                'avg_ch_temp':   float(temps_[:, 0].mean()),
                'avg_tf_temp':   float(temps_[:, 1].mean()),
                'std_ch_temp':   float(temps_[:, 0].std()),
                'std_tf_temp':   float(temps_[:, 1].std()),
            }
        return metrics


# ============================================================================
# 3-Teacher variants (Section 7: multi-teacher scalability)
# ============================================================================

class VotingRouter3(nn.Module):
    """
    3-teacher router: [w_ch, w_tf, w_mo].
    Input: regime(1) + stats(3) + teacher_stds(3) + horizon_emb(8) = 15.
    """

    def __init__(self, n_horizons: int, hidden_dim: int = 64):
        super().__init__()
        self.horizon_embedding = nn.Embedding(n_horizons, 8)
        self.net = nn.Sequential(
            nn.Linear(1 + 3 + 3 + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, regime, stats, tf_std, ch_std, mo_std,
                horizon_indices) -> torch.Tensor:
        h_emb = self.horizon_embedding(horizon_indices)
        x = torch.cat([regime.unsqueeze(1), stats,
                        tf_std.unsqueeze(1), ch_std.unsqueeze(1),
                        mo_std.unsqueeze(1), h_emb], dim=1)
        return self.net(x)   # [B, 3]: [w_ch, w_tf, w_mo]


class TemperatureNetwork3(nn.Module):
    """
    3-teacher temperature network: [T_ch, T_tf, T_mo].
    Softplus activation — non-saturating, unbounded upper range.
    Input: regime(1) + weights(3) + teacher_stds(3) + horizon_emb(4) = 11.
    """

    def __init__(self, n_horizons: int, hidden_dim: int = 32):
        super().__init__()
        self.horizon_embedding = nn.Embedding(n_horizons, 4)
        self.net = nn.Sequential(
            nn.Linear(1 + 3 + 3 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.temp_floor = 0.5

    def forward(self, regime, voting_weights, tf_std, ch_std, mo_std,
                horizon_indices) -> torch.Tensor:
        h_emb = self.horizon_embedding(horizon_indices)
        x = torch.cat([regime.unsqueeze(1), voting_weights,
                        tf_std.unsqueeze(1), ch_std.unsqueeze(1),
                        mo_std.unsqueeze(1), h_emb], dim=1)
        return self.temp_floor + torch.nn.functional.softplus(self.net(x))


class MultiHorizonDataset3(Dataset):
    """MultiHorizonDataset extended for 3 teachers (adds Moirai fields)."""

    def __init__(self, windows: List[Dict], horizons: List[int],
                 mode: str = 'train'):
        self.windows        = windows
        self.horizons       = horizons
        self.mode           = mode
        self.horizon_to_idx = {h: i for i, h in enumerate(horizons)}

    def __len__(self):
        return len(self.windows) * len(self.horizons) if self.mode == 'train' \
               else len(self.windows)

    def __getitem__(self, idx):
        if self.mode == 'train':
            window_idx  = idx % len(self.windows)
            horizon_idx = idx // len(self.windows)
            horizon     = self.horizons[horizon_idx]
        else:
            window_idx = idx
            horizon    = max(self.horizons)

        w     = self.windows[window_idx]
        max_h = max(self.horizons)

        def _pad(key, h):
            arr = np.zeros(max_h, dtype=np.float32)
            src = w.get(f'{key}_h{h}', np.zeros(h, dtype=np.float32))
            arr[:h] = src
            return arr

        labels = _pad('labels', horizon)

        return {
            'context':      torch.FloatTensor(w['context_X']),
            'labels':       torch.FloatTensor(labels),
            'tf_mean':      torch.FloatTensor(_pad('tf_mean', horizon)),
            'tf_std':       torch.FloatTensor(_pad('tf_std',  horizon)),
            'ch_mean':      torch.FloatTensor(_pad('ch_mean', horizon)),
            'ch_std':       torch.FloatTensor(_pad('ch_std',  horizon)),
            'mo_mean':      torch.FloatTensor(_pad('mo_mean', horizon)),
            'mo_std':       torch.FloatTensor(_pad('mo_std',  horizon)),
            'regime':       torch.tensor(
                                1. if w[f'regime_h{horizon}'] == 'extreme' else 0.,
                                dtype=torch.float32),
            'last_value':   torch.tensor(w[f'stats_h{horizon}']['last_value'],  dtype=torch.float32),
            'rolling_std':  torch.tensor(w[f'stats_h{horizon}']['rolling_std'], dtype=torch.float32),
            'trend':        torch.tensor(w[f'stats_h{horizon}']['trend'],        dtype=torch.float32),
            'horizon':      horizon,
            'horizon_idx':  self.horizon_to_idx[horizon],
            'horizon_mask': torch.FloatTensor([1.]*horizon + [0.]*(max_h - horizon)),
            'tf_std_mean':  torch.tensor(w.get(f'tf_std_mean_h{horizon}', 0.), dtype=torch.float32),
            'ch_std_mean':  torch.tensor(w.get(f'ch_std_mean_h{horizon}', 0.), dtype=torch.float32),
            'mo_std_mean':  torch.tensor(w.get(f'mo_std_mean_h{horizon}', 0.), dtype=torch.float32),
            'vote_tf':      torch.tensor(w.get(f'vote_tf_h{horizon}', 1/3), dtype=torch.float32),
            'vote_ch':      torch.tensor(w.get(f'vote_ch_h{horizon}', 1/3), dtype=torch.float32),
            'vote_mo':      torch.tensor(w.get(f'vote_mo_h{horizon}', 1/3), dtype=torch.float32),
            'current_idx':  w['current_idx'],
        }


class SelectiveAdaptiveKDTrainer3:
    """
    Joint trainer for 3-teacher GUARD (Section 7).
    Extends SelectiveAdaptiveKDTrainer to handle Moirai as a third teacher.
    Uses law-of-total-variance aggregation with all pairwise disagreement terms.
    """

    def __init__(self, student: TransformerStudentModel,
                 router: VotingRouter3,
                 temp_net: TemperatureNetwork3,
                 horizons: List[int],
                 device: str = 'cpu'):
        self.student   = student.to(device)
        self.router    = router.to(device)
        self.temp_net  = temp_net.to(device)
        self.horizons  = horizons
        self.horizon_to_idx = {h: i for i, h in enumerate(horizons)}
        self.device    = device

        self.alpha   = 1.0
        self.beta    = 0.3
        self.epsilon = 0.15

        self.optimizer = optim.Adam(
            list(student.parameters()) +
            list(router.parameters()) +
            list(temp_net.parameters()),
            lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)

    def _router_inputs(self, batch):
        regime     = batch['regime'].to(self.device)
        tf_std     = batch['tf_std_mean'].to(self.device).float()
        ch_std     = batch['ch_std_mean'].to(self.device).float()
        mo_std     = batch['mo_std_mean'].to(self.device).float()
        stats      = torch.stack([batch['last_value'].to(self.device),
                                   batch['rolling_std'].to(self.device),
                                   batch['trend'].to(self.device)], dim=1).float()
        h_idx      = torch.LongTensor(
            [self.horizon_to_idx[h] for h in batch['horizon'].numpy()]
        ).to(self.device)
        return regime, stats, tf_std, ch_std, mo_std, h_idx

    def _aggregate(self, batch, weights, mask):
        """Law-of-total-variance for 3 teachers (Section 4.7 extension)."""
        w_ch = weights[:, 0].unsqueeze(1)
        w_tf = weights[:, 1].unsqueeze(1)
        w_mo = weights[:, 2].unsqueeze(1)
        ch_m = batch['ch_mean'].to(self.device)
        tf_m = batch['tf_mean'].to(self.device)
        mo_m = batch['mo_mean'].to(self.device)
        mu   = w_ch * ch_m + w_tf * tf_m + w_mo * mo_m
        var  = (w_ch * batch['ch_std'].to(self.device) ** 2 +
                w_tf * batch['tf_std'].to(self.device) ** 2 +
                w_mo * batch['mo_std'].to(self.device) ** 2 +
                w_ch * w_tf * (ch_m - tf_m) ** 2 +
                w_ch * w_mo * (ch_m - mo_m) ** 2 +
                w_tf * w_mo * (tf_m - mo_m) ** 2)
        return mu * mask, torch.sqrt(var) * mask

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        self.student.train(); self.router.train(); self.temp_net.train()
        totals = dict(total=0., forecast=0., kd=0., entropy=0.)

        for batch in dataloader:
            context  = batch['context'].to(self.device)
            labels   = batch['labels'].to(self.device)
            regime, stats, tf_std, ch_std, mo_std, h_idx = self._router_inputs(batch)
            mask     = batch['horizon_mask'].to(self.device)

            pred     = self.student(context)
            weights  = self.router(regime, stats, tf_std, ch_std, mo_std, h_idx)
            temps    = self.temp_net(regime, weights, tf_std, ch_std, mo_std, h_idx)

            mp = pred * mask; ml = labels * mask; nv = mask.sum()
            forecast_loss = ((mp - ml) ** 2).sum() / nv

            mu_agg, _ = self._aggregate(batch, weights, mask)
            kd_loss   = ((mp - mu_agg) ** 2).sum() / nv
            avg_temp  = (weights * temps).sum(dim=1).mean()
            kd_loss   = kd_loss / (avg_temp ** 2 + 1e-8)

            ent          = -(weights * torch.log(weights + 1e-8)).sum(dim=1)
            entropy_loss = -ent.mean()

            loss = (self.alpha * forecast_loss * (1. + 2. * regime).mean() +
                    self.beta * kd_loss + self.epsilon * entropy_loss)

            self.optimizer.zero_grad(); loss.backward()
            for m in (self.student, self.router, self.temp_net):
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.optimizer.step()

            totals['total']    += loss.item()
            totals['forecast'] += forecast_loss.item()
            totals['kd']       += kd_loss.item()
            totals['entropy']  += entropy_loss.item()

        n = len(dataloader)
        self.scheduler.step(totals['total'] / n)
        return {k: v / n for k, v in totals.items()}

    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.student.eval(); self.router.eval(); self.temp_net.eval()
        store = {h: {'preds': [], 'labels': [], 'regimes': [], 'weights': []}
                 for h in self.horizons}

        with torch.no_grad():
            for batch in dataloader:
                pred   = self.student(batch['context'].to(self.device)).cpu().numpy()
                regime, stats, tf_std, ch_std, mo_std, h_idx = self._router_inputs(batch)
                weights = self.router(
                    regime, stats, tf_std, ch_std, mo_std, h_idx).cpu().numpy()
                for h in self.horizons:
                    store[h]['preds'].append(pred[:, :h])
                    store[h]['labels'].append(batch['labels'].numpy()[:, :h])
                    store[h]['regimes'].append(batch['regime'].numpy())
                    store[h]['weights'].append(weights)

        metrics = {}
        for h in self.horizons:
            p  = np.concatenate(store[h]['preds'])
            l  = np.concatenate(store[h]['labels'])
            r  = np.concatenate(store[h]['regimes'])
            w  = np.concatenate(store[h]['weights'])
            nm = r == 0; em = r == 1
            metrics[h] = {
                'rmse':         float(np.sqrt(np.mean((p - l) ** 2))),
                'rmse_normal':  float(np.sqrt(np.mean((p[nm] - l[nm]) ** 2))) if nm.sum() else 0.,
                'rmse_extreme': float(np.sqrt(np.mean((p[em] - l[em]) ** 2))) if em.sum() else 0.,
                'avg_ch_weight': float(w[:, 0].mean()),
                'avg_tf_weight': float(w[:, 1].mean()),
                'avg_mo_weight': float(w[:, 2].mean()),
            }
        return metrics

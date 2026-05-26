"""
teachers.py
-----------
Foundation model teacher wrappers (TimesFM, Chronos) and Phase 1 caching.

IQR-based uncertainty estimator for Chronos:
  sigma = (q90 - q10) / 1.35
This is distribution-agnostic and valid under heavy-tailed distributions
(e.g., carbon flux), unlike the Gaussian 2.56 approximation.
"""

import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# ── optional imports ──────────────────────────────────────────────────────────

try:
    from chronos import Chronos2Pipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    print("⚠️  Chronos not available. Install: pip install chronos-forecasting")
    CHRONOS_AVAILABLE = False

try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    print("⚠️  TimesFM not available. Install: pip install timesfm")
    TIMESFM_AVAILABLE = False

try:
    from uni2ts.model.moirai import MoiraiForecast
    from huggingface_hub import hf_hub_download
    MOIRAI_AVAILABLE = True
except ImportError:
    print("⚠️  Moirai not available. Install: pip install uni2ts gluonts huggingface_hub safetensors")
    MOIRAI_AVAILABLE = False


# ============================================================================
# TimesFM teacher
# ============================================================================

class TimesFMTeacher:
    """Zero-shot TimesFM teacher wrapper.

    Falls back to a linear extrapolation if the model cannot be loaded,
    allowing the rest of the pipeline to run for debugging.
    """

    def __init__(self, context_len: int = 96, max_horizon: int = 36,
                 device: str = 'cpu'):
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.device      = device
        self.model       = None
        self.is_compiled = False

        if not TIMESFM_AVAILABLE:
            print("⚠️  TimesFM not available, using linear fallback")
            return

        print("Loading TimesFM teacher...")
        try:
            ModelClass = getattr(timesfm, 'TimesFM_2p5_200M_torch', None)
            if ModelClass is None:
                raise AttributeError("TimesFM_2p5_200M_torch not found")
            self.model = ModelClass.from_pretrained(
                "google/timesfm-2.5-200m-pytorch",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            if isinstance(self.model, torch.nn.Module):
                self.model = self.model.to(device).eval()
                for p in self.model.parameters():
                    p.requires_grad = False
            self.model.compile(timesfm.ForecastConfig(
                max_context=context_len, max_horizon=max_horizon,
                normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=False,
                fix_quantile_crossing=True,
            ))
            self.is_compiled = True
            print("✓ TimesFM loaded and compiled")
        except Exception as e:
            print(f"Warning: TimesFM load failed ({e}). Using linear fallback.")

    def _fallback(self, context: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
        last  = context[-1, 0]
        trend = (context[-1, 0] - context[-12, 0]) / 12 if len(context) >= 12 else 0
        means = last + trend * np.arange(1, h + 1)
        stds  = 0.3 + 0.01 * np.arange(1, h + 1)
        return means[:h].astype(np.float32), stds[:h].astype(np.float32)

    def predict(self, context: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_compiled:
            return self._fallback(context, horizon)
        try:
            ctx = np.mean(context, axis=-1)
            if len(ctx) > self.context_len:
                ctx = ctx[-self.context_len:]
            elif len(ctx) < self.context_len:
                ctx = np.concatenate([np.zeros(self.context_len - len(ctx)), ctx])

            with torch.no_grad():
                pt, qt = self.model.forecast(horizon=self.max_horizon,
                                             inputs=[ctx.tolist()])
            means = np.array(pt[0], dtype=np.float32)

            if qt is not None and len(qt) > 0:
                q = np.array(qt[0], dtype=np.float32)
                if q.ndim == 2 and q.shape[1] >= 3:
                    # IQR-based estimator: valid under heavy-tailed distributions
                    stds = ((q[:, -1] - q[:, 1]) / 1.35).astype(np.float32)
                else:
                    stds = np.full(self.max_horizon, 0.3, dtype=np.float32)
            else:
                stds = (0.3 + 0.01 * np.arange(1, self.max_horizon + 1)).astype(np.float32)

            h = horizon
            if len(means) < h:
                means = np.pad(means, (0, h - len(means)), mode='edge')
                stds  = np.pad(stds,  (0, h - len(stds)),  constant_values=0.3)
            return means[:h], stds[:h]
        except Exception as e:
            print(f"TimesFM error: {e}, falling back")
            return self._fallback(context, horizon)


# ============================================================================
# Chronos teacher
# ============================================================================

class ChronosTeacher:
    """Zero-shot Chronos teacher wrapper.

    Uncertainty is estimated from quantiles using the IQR estimator:
        sigma = (q90 - q10) / 1.35
    which is valid for both Gaussian and heavy-tailed distributions.
    """

    def __init__(self, context_len: int = 96, device: str = 'cpu'):
        self.context_len = context_len
        self.device      = device
        self.pipeline    = None

        if not CHRONOS_AVAILABLE:
            print("⚠️  Chronos not available, using seasonal fallback")
            return

        print("Loading Chronos teacher...")
        try:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            if hasattr(self.pipeline, 'model'):
                self.pipeline.model.eval()
                for p in self.pipeline.model.parameters():
                    p.requires_grad = False
            print("✓ Chronos loaded")
        except Exception as e:
            print(f"Warning: Chronos load failed ({e}). Using seasonal fallback.")

    def _fallback(self, context: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
        last     = context[-1, 0] if context.ndim > 1 else context[-1]
        seasonal = np.sin(2 * np.pi * np.arange(1, h + 1) / 144)
        means    = (last + seasonal * 0.5).astype(np.float32)
        stds     = (0.25 + 0.015 * np.arange(1, h + 1)).astype(np.float32)
        return means, stds

    def predict(self, context: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.pipeline is None:
            return self._fallback(context, horizon)
        try:
            seq_len    = len(context)
            timestamps = pd.date_range('2000-01-01', periods=seq_len, freq='10min')
            rows = [{'id': 'series_0', 'timestamp': ts,
                     **{f'feature_{j}': context[i, j]
                        for j in range(context.shape[1] if context.ndim > 1 else 1)}}
                    for i, ts in enumerate(timestamps)]
            df = pd.DataFrame(rows)

            fcast = self.pipeline.predict_df(
                df, prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column='id', timestamp_column='timestamp',
                target='feature_0',
            )
            if fcast is None or len(fcast) == 0:
                raise ValueError("Empty forecast")

            means = fcast['0.5'].values[:horizon].astype(np.float32)
            # IQR-based estimator: valid under heavy-tailed distributions
            stds  = ((fcast['0.9'].values[:horizon] -
                      fcast['0.1'].values[:horizon]) / 1.35).astype(np.float32)

            if len(means) < horizon:
                means = np.pad(means, (0, horizon - len(means)), mode='edge')
                stds  = np.pad(stds,  (0, horizon - len(stds)),  mode='edge')
            return means, stds
        except Exception as e:
            print(f"Chronos error: {e}, falling back")
            return self._fallback(context, horizon)


# ============================================================================
# Phase 1: cache teacher predictions for all splits
# ============================================================================

def generate_teacher_predictions(
        windows: List[Dict],
        horizons: List[int],
        device: str = 'cpu',
        phase1_log: Optional[Dict] = None,
        split_name: str = 'split') -> List[Dict]:
    """
    Run Phase 1 zero-shot teacher inference and cache results in windows.

    Computes EMA-smoothed pseudo-oracle voting weights for router supervision.
    Records wall-time and cache size in phase1_log[split_name] if provided.
    """
    context_len = len(windows[0]['context_X'])
    max_h       = max(horizons)
    ema_alpha   = 0.9
    ema_losses  = {h: {'tf': None, 'ch': None} for h in horizons}

    # ── TimesFM ──────────────────────────────────────────────────────────────
    print(f"\n[Phase 1] Loading TimesFM for {split_name}...")
    tf_teacher = TimesFMTeacher(context_len, max_h, device)
    t0 = time.time()
    for w in tqdm(windows, desc="TimesFM"):
        ctx = w['context_X']
        for h in horizons:
            mu, sig          = tf_teacher.predict(ctx, h)
            w[f'tf_mean_h{h}']     = mu
            w[f'tf_std_h{h}']      = sig
            w[f'tf_std_mean_h{h}'] = float(sig.mean())
    timesfm_s = time.time() - t0
    del tf_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Chronos ───────────────────────────────────────────────────────────────
    print(f"\n[Phase 1] Loading Chronos for {split_name}...")
    ch_teacher = ChronosTeacher(context_len, device)
    t0 = time.time()
    for w in tqdm(windows, desc="Chronos"):
        ctx = w['context_X']
        for h in horizons:
            mu, sig          = ch_teacher.predict(ctx, h)
            w[f'ch_mean_h{h}']     = mu
            w[f'ch_std_h{h}']      = sig
            w[f'ch_std_mean_h{h}'] = float(sig.mean())
    chronos_s = time.time() - t0
    del ch_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── EMA pseudo-oracle voting weights ──────────────────────────────────────
    print("\n[Phase 1] Computing pseudo-oracle routing weights...")
    for w in windows:
        for h in horizons:
            labels  = w[f'labels_h{h}']
            loss_tf = float(np.mean((labels - w[f'tf_mean_h{h}']) ** 2))
            loss_ch = float(np.mean((labels - w[f'ch_mean_h{h}']) ** 2))

            if ema_losses[h]['tf'] is None:
                ema_losses[h]['tf'] = loss_tf
                ema_losses[h]['ch'] = loss_ch
            else:
                ema_losses[h]['tf'] = ema_alpha * ema_losses[h]['tf'] + (1 - ema_alpha) * loss_tf
                ema_losses[h]['ch'] = ema_alpha * ema_losses[h]['ch'] + (1 - ema_alpha) * loss_ch

            vote_tf = ema_losses[h]['ch'] / (ema_losses[h]['tf'] + ema_losses[h]['ch'] + 1e-8)
            w[f'vote_tf_h{h}'] = float(np.clip(vote_tf, 0.1, 0.9))
            w[f'vote_ch_h{h}'] = 1.0 - w[f'vote_tf_h{h}']

    # ── log ───────────────────────────────────────────────────────────────────
    if phase1_log is not None:
        import pickle, tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            tmppath = f.name
        with open(tmppath, 'wb') as f:
            pickle.dump(windows, f)
        cache_mb = os.path.getsize(tmppath) / 1e6
        os.unlink(tmppath)

        phase1_log[split_name] = {
            'n_windows': len(windows),
            'timesfm_s': round(timesfm_s, 1),
            'chronos_s': round(chronos_s, 1),
            'total_s':   round(timesfm_s + chronos_s, 1),
            'cache_mb':  round(cache_mb, 2),
        }
        print(f"  Phase 1 [{split_name}]: TimesFM={timesfm_s:.0f}s, "
              f"Chronos={chronos_s:.0f}s, cache={cache_mb:.1f}MB")

    return windows


# ============================================================================
# MoiraiTeacher (Section 7: multi-teacher scalability)
# ============================================================================

class MoiraiTeacher:
    """
    Zero-shot Moirai-1.0-R teacher wrapper (Salesforce uni2ts).

    Loads model.safetensors from HuggingFace Hub. Model size is controlled
    by MOIRAI_SIZE: 'small' | 'base' | 'large'. Swap to 'small' if
    GPU memory is constrained (~6× faster than large).

    Used in Section 7 (multi-teacher scalability) to validate that GUARD's
    router autonomously detects latent teacher family structure.
    """

    MOIRAI_SIZE = "large"  # change to 'small' or 'base' if memory-constrained

    def __init__(self, context_len: int = 96, max_horizon: int = 36,
                 device: str = 'cpu'):
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.device      = device
        self.predictor   = None
        self.num_samples = 100

        if not MOIRAI_AVAILABLE:
            print("⚠️  Moirai not available, using linear fallback")
            return

        print(f"Loading Moirai-1.0-R-{self.MOIRAI_SIZE} teacher...")
        try:
            from safetensors.torch import load_file as load_safetensors
            from uni2ts.model.moirai import MoiraiModule
            from uni2ts.distribution.mixture import MixtureOutput
            from uni2ts.distribution.student_t import StudentTOutput
            from uni2ts.distribution.normal import NormalFixedScaleOutput
            from uni2ts.distribution.negative_binomial import NegativeBinomialOutput
            from uni2ts.distribution.log_normal import LogNormalOutput

            map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
            checkpoint_path = hf_hub_download(
                repo_id=f"Salesforce/moirai-1.0-R-{self.MOIRAI_SIZE}",
                filename="model.safetensors",
            )

            distr_output = MixtureOutput(components=[
                StudentTOutput(), NormalFixedScaleOutput(scale=0.001),
                NegativeBinomialOutput(), LogNormalOutput(),
            ])
            module = MoiraiModule(
                distr_output=distr_output, d_model=1024, num_layers=24,
                patch_sizes=(8, 16, 32, 64, 128), max_seq_len=512,
                attn_dropout_p=0.0, dropout_p=0.0, scaling=True,
            )
            model = MoiraiForecast(
                prediction_length=max_horizon, context_length=context_len,
                patch_size="auto", num_samples=self.num_samples,
                target_dim=1, feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0, module=module,
            )
            # safetensors keys are unprefixed; MoiraiForecast expects "module." prefix
            raw_weights     = load_safetensors(checkpoint_path, device=map_location)
            prefixed_weights = {"module." + k: v for k, v in raw_weights.items()}
            model.load_state_dict(prefixed_weights)
            model.eval()
            self.predictor = model.create_predictor(batch_size=32)
            print(f"✓ Moirai-1.0-R-{self.MOIRAI_SIZE} loaded")
        except Exception as e:
            print(f"Warning: Moirai load failed ({e}). Using linear fallback.")

    def _make_gluonts_entry(self, context_ts: np.ndarray):
        from gluonts.dataset.common import ListDataset
        return ListDataset(
            [{"start": pd.Period("2000-01-01", freq="10T"),
              "target": context_ts.astype(np.float32)}],
            freq="10T")

    def _fallback(self, context: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
        last  = float(context[-1, 0]) if context.ndim > 1 else float(context[-1])
        trend = (float(context[-1, 0]) - float(context[-12, 0])) / 12 \
                if len(context) >= 12 else 0.
        means = last + trend * np.arange(1, h + 1)
        stds  = 0.28 + 0.012 * np.arange(1, h + 1)
        return means[:h].astype(np.float32), stds[:h].astype(np.float32)

    def predict(self, context: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.predictor is None:
            return self._fallback(context, horizon)
        try:
            ctx = context[:, 0] if context.ndim > 1 else context
            if len(ctx) > self.context_len:
                ctx = ctx[-self.context_len:]
            elif len(ctx) < self.context_len:
                ctx = np.concatenate([np.zeros(self.context_len - len(ctx),
                                                dtype=np.float32), ctx])

            dataset   = self._make_gluonts_entry(ctx)
            forecasts = list(self.predictor.predict(dataset))
            if not forecasts:
                raise ValueError("Moirai returned no forecasts")

            samples = forecasts[0].samples   # [num_samples, prediction_length]
            h = horizon
            if samples.shape[1] >= h:
                means = samples[:, :h].mean(axis=0)
                stds  = samples[:, :h].std(axis=0)
            else:
                means = np.pad(samples.mean(axis=0),
                               (0, h - samples.shape[1]), mode='edge')
                stds  = np.pad(samples.std(axis=0),
                               (0, h - samples.shape[1]), constant_values=0.3)
            stds = np.clip(stds, 1e-4, None)
            return means.astype(np.float32), stds.astype(np.float32)
        except Exception as e:
            print(f"Moirai error: {e}, falling back")
            return self._fallback(context, horizon)


# ============================================================================
# Phase 1: 3-teacher caching (TimesFM + Chronos + Moirai)
# ============================================================================

def generate_teacher_predictions_3teacher(
        windows: List[Dict],
        horizons: List[int],
        device: str = 'cpu',
        phase1_log: Optional[Dict] = None,
        split_name: str = 'split') -> List[Dict]:
    """
    Phase 1 for 3-teacher GUARD (Section 7).

    Runs Moirai pre-flight check first to fail fast before expensive
    TimesFM/Chronos inference. Each teacher is freed from GPU memory
    before the next is loaded to avoid OOM on single-GPU setups.

    Phase 1 wall times on NVIDIA RTX 3090 (Weather, all splits):
      TimesFM  ~1.7 hr | Chronos ~0.8 hr | Moirai-large ~12.7 hr
    Use MOIRAI_SIZE='small' to reduce Moirai cost ~6×.
    """
    context_len = len(windows[0]['context_X'])
    max_h       = max(horizons)
    ema_alpha   = 0.9
    ema_losses  = {h: {'tf': None, 'ch': None, 'mo': None} for h in horizons}

    def _safe_device(min_gb=2.0):
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if free_gb >= min_gb:
                return device
        return 'cpu'

    def _free(obj):
        try:
            if hasattr(obj, 'model') and isinstance(obj.model, torch.nn.Module):
                obj.model.cpu()
            if hasattr(obj, 'pipeline') and hasattr(obj.pipeline, 'model'):
                obj.pipeline.model.cpu()
        except Exception:
            pass
        del obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Pre-flight: verify Moirai loads before spending time on other teachers
    print(f"\n[Phase 1 — 3-teacher] Pre-flight Moirai check ({split_name})...")
    mo_check = MoiraiTeacher(context_len, max_h, _safe_device(min_gb=3.0))
    if mo_check.predictor is None:
        raise RuntimeError(
            "Moirai failed to load. Fix the checkpoint before proceeding.\n"
            f"Check available files: huggingface_hub.list_repo_files("
            f"'Salesforce/moirai-1.0-R-{MoiraiTeacher.MOIRAI_SIZE}')")
    _free(mo_check)
    print("✓ Moirai pre-flight OK")

    # Pass 1: TimesFM
    print(f"\n[Teacher 1/3] TimesFM ({split_name})...")
    tf_teacher = TimesFMTeacher(context_len, max_h, _safe_device(min_gb=2.0))
    t0 = time.time()
    for w in tqdm(windows, desc="TimesFM"):
        for h in horizons:
            mu, sig = tf_teacher.predict(w['context_X'], h)
            w[f'tf_mean_h{h}'] = mu;  w[f'tf_std_h{h}'] = sig
            w[f'tf_std_mean_h{h}'] = float(sig.mean())
    timesfm_s = time.time() - t0
    _free(tf_teacher)

    # Pass 2: Chronos
    print(f"\n[Teacher 2/3] Chronos ({split_name})...")
    ch_teacher = ChronosTeacher(context_len, _safe_device(min_gb=2.0))
    t0 = time.time()
    for w in tqdm(windows, desc="Chronos"):
        for h in horizons:
            mu, sig = ch_teacher.predict(w['context_X'], h)
            w[f'ch_mean_h{h}'] = mu;  w[f'ch_std_h{h}'] = sig
            w[f'ch_std_mean_h{h}'] = float(sig.mean())
    chronos_s = time.time() - t0
    _free(ch_teacher)

    # Pass 3: Moirai (reloaded after others freed GPU memory)
    print(f"\n[Teacher 3/3] Moirai ({split_name})...")
    mo_teacher = MoiraiTeacher(context_len, max_h, _safe_device(min_gb=3.0))
    t0 = time.time()
    for w in tqdm(windows, desc="Moirai"):
        for h in horizons:
            mu, sig = mo_teacher.predict(w['context_X'], h)
            w[f'mo_mean_h{h}'] = mu;  w[f'mo_std_h{h}'] = sig
            w[f'mo_std_mean_h{h}'] = float(sig.mean())
    moirai_s = time.time() - t0
    _free(mo_teacher)

    # Pass 4: EMA pseudo-oracle voting weights (3-way)
    print("\nComputing 3-teacher EMA pseudo-oracle weights...")
    for w in windows:
        for h in horizons:
            labels = w[f'labels_h{h}']
            l_tf = float(np.mean((labels - w[f'tf_mean_h{h}']) ** 2))
            l_ch = float(np.mean((labels - w[f'ch_mean_h{h}']) ** 2))
            l_mo = float(np.mean((labels - w[f'mo_mean_h{h}']) ** 2))

            if ema_losses[h]['tf'] is None:
                ema_losses[h].update({'tf': l_tf, 'ch': l_ch, 'mo': l_mo})
            else:
                for k, l in (('tf', l_tf), ('ch', l_ch), ('mo', l_mo)):
                    ema_losses[h][k] = ema_alpha * ema_losses[h][k] + (1 - ema_alpha) * l

            inv = {k: 1. / (ema_losses[h][k] + 1e-8)
                   for k in ('tf', 'ch', 'mo')}
            tot = sum(inv.values())
            for k in ('tf', 'ch', 'mo'):
                w[f'vote_{k}_h{h}'] = float(np.clip(inv[k] / tot, 0.05, 0.90))
            # renormalise after clipping
            s = sum(w[f'vote_{k}_h{h}'] for k in ('tf', 'ch', 'mo'))
            for k in ('tf', 'ch', 'mo'):
                w[f'vote_{k}_h{h}'] /= s

    # Log
    if phase1_log is not None:
        total_s  = timesfm_s + chronos_s + moirai_s
        # Estimate cache size from actual stored arrays
        cache_bytes = 0
        for w in windows:
            for t in ('tf', 'ch', 'mo'):
                for h in horizons:
                    for key in (f'{t}_mean_h{h}', f'{t}_std_h{h}'):
                        arr = w.get(key)
                        if arr is not None and hasattr(arr, 'nbytes'):
                            cache_bytes += arr.nbytes
        cache_mb = cache_bytes / 1e6
        phase1_log[split_name] = {
            'n_windows': len(windows),
            'timesfm_s': round(timesfm_s, 1),
            'chronos_s': round(chronos_s, 1),
            'moirai_s':  round(moirai_s,  1),
            'total_s':   round(total_s,   1),
            'cache_mb':  round(cache_mb,  2),
        }
        print(f"  Phase 1 [{split_name}]: TimesFM={timesfm_s:.0f}s "
              f"Chronos={chronos_s:.0f}s Moirai={moirai_s:.0f}s "
              f"total={total_s:.0f}s cache={cache_mb:.1f}MB")

    return windows

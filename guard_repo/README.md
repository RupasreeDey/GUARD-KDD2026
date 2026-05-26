# GUARD: Gated Uncertainty-Aware Routing for Distillation

**When to Trust, How to Distill: Multi-Foundation Model Guidance for Lightweight, Robust Scientific Time Series Forecasting**

*Rupasree Dey, Abdul Matin, Nathan Orwick, Yao Zhang, Shrideep Pallickara, Sangmi Lee Pallickara*
*Department of Computer Science (and Soil and Crop Sciences), Colorado State University*

Accepted at **KDD 2026 AI for Sciences Track**.

---

## Overview

GUARD distills knowledge from multiple misaligned Time-Series Foundation Models (TimesFM, Chronos) into a lightweight ~0.3M-parameter student suitable for edge deployment. Two adaptive mechanisms drive the framework:

1. **Contextual Router** — regime-aware mixing of teacher signals based on local volatility, magnitude, and trend statistics.
2. **Uncertainty-Gated Temperature Network** — a circuit breaker that suppresses distillation when teachers are uncertain, using a non-saturating softplus formulation.

GUARD achieves a 28.3% average RMSE reduction across five scientific domains vs. deep SOTA baselines, with >390× parameter compression vs. TimesFM.

---

## Repository Structure

```
guard_repo/
├── guard/                      # Core library
│   ├── __init__.py
│   ├── data_processors.py      # Dataset loaders: Weather, ETTh1/m1, Flux, Soil Moisture
│   ├── teachers.py             # TimesFM + Chronos wrappers; Phase 1 caching
│   └── models.py               # VotingRouter, TemperatureNetwork, Student, Trainer
├── train_guard.py              # ← Main: train GUARD / ablations / HP sweep
├── baselines.py                # ← Deep baselines: DLinear, Transformer, PatchTST, iTransformer
├── zero_shot.py                # ← Zero-shot teacher evaluation (TimesFM, Chronos)
├── data/                       # Place dataset CSVs here (see Data section)
├── results/                    # Output JSON files written here
└── requirements.txt
```

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/your-repo/guard.git
cd guard
conda create -n guard python=3.10
conda activate guard
```

### 2. Install core dependencies

```bash
pip install -r requirements.txt
```

### 3. Install TimesFM (Google)

TimesFM requires a separate install due to its JAX/PyTorch dual-backend:

```bash
pip install timesfm
```

If you encounter issues, refer to the [TimesFM GitHub](https://github.com/google-research/timesfm).

### 4. Install Chronos

```bash
pip install chronos-forecasting
```

### 5. (Optional) Install Moirai — for multi-teacher rebuttal experiments only

```bash
pip install uni2ts gluonts huggingface_hub safetensors
```

---

## Data Preparation

Place all dataset CSVs in `data/`. Expected filenames and formats:

| Dataset | File | Key columns |
|---------|------|-------------|
| Weather (MPI-BGC Jena) | `data/cleaned_weather.csv` | `date`, `T` (temperature target), other meteo variables |
| ETTh1 | `data/ETTh1.csv` | `date`, `OT` (oil temp target), HUFL, HULL, MUFL, MULL, LUFL, LULL |
| ETTm1 | `data/ETTm1.csv` | same as ETTh1 |
| Flux (DayCent NEE) | `data/monthly_input_output.csv` | `Year`, `Month`, `NEE` (target), other flux variables |
| Soil Moisture (Quench) | `data/quench_soil_moisture.csv` | `network`, `station_id`, `timestamp`, `soil_moisture` |

**Weather**: Download from [MPI-BGC Jena Climate Dataset](https://www.bgc-jena.mpg.de/wetter/).

**ETT**: Download from the [ETDataset GitHub repository](https://github.com/zhouhaoyi/ETDataset).

**Flux**: DayCent-simulated NEE data from Midwestern cropland sites (2000–2020). Contact the authors for access.

**Soil Moisture**: Collected via the [Quench platform](https://spatial.colostate.edu/quench/) from 42 Colorado agricultural stations (January 2024–January 2026). Contact the authors for access.

---

## Reproducing Paper Results

All experiments use `context_len=96`, `epochs=30`, `batch_size=32`, `patience=5`.

### Section 7 — Multi-teacher scalability (Moirai integration)

Requires Moirai dependencies:
```bash
pip install uni2ts gluonts huggingface_hub safetensors
```

Then:
```bash
python train_guard.py --dataset weather --csv data/cleaned_weather.csv --three-teacher
```

> **Note:** Phase 1 with Moirai-large takes ~12.7 hrs on RTX 3090. Set `MoiraiTeacher.MOIRAI_SIZE = 'small'` in `guard/teachers.py` for ~6× speedup at some accuracy cost.

Results saved to `results/guard_weather_3teacher.json`.

### Table 5 — GUARD results (full framework)

```bash
# Single dataset
python train_guard.py --dataset weather     --csv data/cleaned_weather.csv
python train_guard.py --dataset etth1       --csv data/ETTh1.csv
python train_guard.py --dataset ettm1       --csv data/ETTm1.csv
python train_guard.py --dataset flux        --csv data/monthly_input_output.csv
python train_guard.py --dataset soilmoisture --csv data/quench_soil_moisture.csv

# All datasets at once
python train_guard.py --all
```

Results saved to `results/guard_{dataset}_voting_temp.json`.

### Table 5 — Deep baseline results

```bash
python baselines.py --dataset weather --csv data/cleaned_weather.csv
python baselines.py --all
```

Results saved to `results/baselines_{dataset}.json`.

### Table 5 — Zero-shot teacher baselines (TimesFM, Chronos)

```bash
python zero_shot.py --dataset weather --csv data/cleaned_weather.csv
python zero_shot.py --all
```

### Table 2 — Ablation study

```bash
# Runs base / voting_only / voting_temp for a dataset
python train_guard.py --dataset weather --csv data/cleaned_weather.csv --ablation all
python train_guard.py --dataset flux    --csv data/monthly_input_output.csv --ablation all
python train_guard.py --dataset etth1   --csv data/ETTh1.csv --ablation all
```

### Appendix C — Hyperparameter sensitivity sweep

```bash
python train_guard.py --dataset weather --csv data/cleaned_weather.csv --hp-sweep
```

Output: `results/hp_sweep_weather.csv` with RMSE for each (β, ε) configuration.

---

## Key Implementation Details

### IQR-based uncertainty estimation
For quantile-based teachers (Chronos), standard deviations are estimated using a distribution-agnostic IQR estimator valid under heavy-tailed distributions:
```
σ = (q90 − q10) / 1.35
```
This replaces the Gaussian-only 2.56 denominator used in prior work.

### Non-saturating temperature (softplus)
The TemperatureNetwork uses:
```
T = 0.5 + softplus(logit)
```
This provides an unbounded upper range (enabling aggressive attenuation, e.g., T > 6000 on Flux data under catastrophic teacher failure) while the 0.5 floor prevents over-softening of reliable short-horizon signals.

### Phase 1 compute costs
Phase 1 (teacher caching) is a one-time offline cost. Approximate wall times on an NVIDIA RTX 3090 for the full Weather dataset (all splits):

| Teacher | Wall Time | Notes |
|---------|-----------|-------|
| TimesFM | ~1.7 hr | Regression-based |
| Chronos | ~0.8 hr | Quantile-based |
| Total | ~2.5 hr | 52 MB cached |

The cached student (1.15 MB) runs at 0.754 ms/sample on CPU — well within the 10-minute sampling intervals of agricultural sensor networks.

---

## Citation

```bibtex
@inproceedings{dey2026guard,
  title     = {When to Trust, How to Distill: Multi-Foundation Model Guidance
               for Lightweight, Robust Scientific Time Series Forecasting},
  author    = {Dey, Rupasree and Matin, Abdul and Orwick, Nathan and
               Zhang, Yao and Pallickara, Shrideep and Pallickara, Sangmi Lee},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on
               Knowledge Discovery and Data Mining (KDD)},
  year      = {2026},
  address   = {Jeju, Korea},
  publisher = {ACM}
}
```

---

## Acknowledgments

This research was supported by the National Science Foundation (1931363, 2312319), the National Institute of Food Agriculture (COL014021223), NSF/NIFA AI Institutes AI-LEAF [2023-03616], and the Clare Boothe Luce Professorship.

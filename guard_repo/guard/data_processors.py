"""
data_processors.py
------------------
Dataset-specific processors for GUARD: Weather, Flux, ETTh1, ETTm1, Soil Moisture.
Each processor handles loading, preprocessing, chronological splitting, and window creation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
import random


# ============================================================================
# Weather (MPI-BGC Jena Climate, 10-minute resolution)
# ============================================================================

class WeatherDataProcessor:
    """MPI-BGC Jena Climate dataset. Target: temperature (T)."""

    def __init__(self, csv_path: str, context_len: int = 96,
                 horizons: List[int] = [6, 18, 36]):
        self.context_len = context_len
        self.horizons    = horizons
        self.max_horizon = max(horizons)

        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        self.target_col   = 'T'
        self.feature_cols = [c for c in self.df.columns if c not in ['date', 'T']]

        self.df['hour']      = self.df['date'].dt.hour
        self.df['dayofyear'] = self.df['date'].dt.dayofyear
        self.df['hour_sin']  = np.sin(2 * np.pi * self.df['hour'] / 24).astype(np.float32)
        self.df['hour_cos']  = np.cos(2 * np.pi * self.df['hour'] / 24).astype(np.float32)
        self.df['day_sin']   = np.sin(2 * np.pi * self.df['dayofyear'] / 365).astype(np.float32)
        self.df['day_cos']   = np.cos(2 * np.pi * self.df['dayofyear'] / 365).astype(np.float32)
        self.feature_cols.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])

    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        n         = len(self.df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))
        self.train_df = self.df.iloc[:train_end].copy()
        self.val_df   = self.df.iloc[train_end:val_end].copy()
        self.test_df  = self.df.iloc[val_end:].copy()

        self.feature_scaler = StandardScaler()
        self.target_scaler  = StandardScaler()
        self.feature_scaler.fit(self.train_df[self.feature_cols].values)
        self.target_scaler.fit(self.train_df[[self.target_col]].values)

        train_vals         = self.train_df[self.target_col]
        self.val_mean      = float(train_vals.mean())
        self.val_std       = float(train_vals.std())
        self.extreme_z_threshold = 1.5   # ~93rd percentile
        return self.train_df, self.val_df, self.test_df

    def scale_data(self, df):
        X = self.feature_scaler.transform(df[self.feature_cols].values).astype(np.float32)
        y = self.target_scaler.transform(df[[self.target_col]].values).flatten().astype(np.float32)
        return X, y

    def create_windows(self, split: str = 'train',
                       sample_fraction: float = 1.0) -> List[Dict]:
        df = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}[split]
        X, y = self.scale_data(df)
        all_idx = list(range(self.context_len, len(df) - self.max_horizon))
        if sample_fraction < 1.0:
            all_idx = random.sample(all_idx, int(len(all_idx) * sample_fraction))

        windows = []
        for i in all_idx:
            window = {'context_X':    X[i - self.context_len:i],
                      'context_dates': df['date'].iloc[i - self.context_len:i].values,
                      'current_idx':  i}
            for h in self.horizons:
                window[f'labels_h{h}'] = y[i:i + h]
                future = df[self.target_col].iloc[i:i + h].values
                z = (future - self.val_mean) / self.val_std
                window[f'regime_h{h}'] = 'extreme' if np.any(z > self.extreme_z_threshold) else 'normal'
                recent = df[self.target_col].iloc[i - 12:i].values
                window[f'stats_h{h}'] = {
                    'last_value':  float(df[self.target_col].iloc[i - 1]),
                    'rolling_std': float(np.std(recent)),
                    'trend':       float(recent[-1] - recent[0]),
                }
            windows.append(window)
        return windows


# ============================================================================
# ETTh1 / ETTm1 (electricity transformer temperature)
# ============================================================================

class ETTDataProcessor:
    """ETT electricity transformer benchmarks. Target: OT (oil temperature)."""

    def __init__(self, csv_path: str, dataset: str = 'ETTh1',
                 context_len: int = 96, horizons: List[int] = None):
        self.dataset     = dataset
        self.context_len = context_len
        self.horizons    = horizons or [6, 18, 36]
        self.max_horizon = max(self.horizons)
        self.target_col  = 'OT'

        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.feature_cols = [c for c in self.df.columns
                             if c not in ['date', self.target_col]]

        self.df['hour']      = self.df['date'].dt.hour
        self.df['dayofyear'] = self.df['date'].dt.dayofyear
        self.df['hour_sin']  = np.sin(2 * np.pi * self.df['hour'] / 24).astype(np.float32)
        self.df['hour_cos']  = np.cos(2 * np.pi * self.df['hour'] / 24).astype(np.float32)
        self.df['day_sin']   = np.sin(2 * np.pi * self.df['dayofyear'] / 365).astype(np.float32)
        self.df['day_cos']   = np.cos(2 * np.pi * self.df['dayofyear'] / 365).astype(np.float32)
        self.feature_cols.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])

    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        n         = len(self.df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))
        self.train_df = self.df.iloc[:train_end].copy()
        self.val_df   = self.df.iloc[train_end:val_end].copy()
        self.test_df  = self.df.iloc[val_end:].copy()

        self.feature_scaler = StandardScaler()
        self.target_scaler  = StandardScaler()
        self.feature_scaler.fit(self.train_df[self.feature_cols].values)
        self.target_scaler.fit(self.train_df[[self.target_col]].values)

        train_vals               = self.train_df[self.target_col]
        self.val_mean            = float(train_vals.mean())
        self.val_std             = float(train_vals.std())
        self.extreme_z_threshold = 1.5
        return self.train_df, self.val_df, self.test_df

    def scale_data(self, df):
        X = self.feature_scaler.transform(df[self.feature_cols].values).astype(np.float32)
        y = self.target_scaler.transform(df[[self.target_col]].values).flatten().astype(np.float32)
        return X, y

    def create_windows(self, split: str = 'train',
                       sample_fraction: float = 1.0) -> List[Dict]:
        df = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}[split]
        X, y = self.scale_data(df)
        all_idx = list(range(self.context_len, len(df) - self.max_horizon))
        if sample_fraction < 1.0:
            all_idx = random.sample(all_idx, int(len(all_idx) * sample_fraction))

        windows = []
        for i in all_idx:
            window = {'context_X':    X[i - self.context_len:i],
                      'context_dates': df['date'].iloc[i - self.context_len:i].values,
                      'current_idx':  i}
            for h in self.horizons:
                window[f'labels_h{h}'] = y[i:i + h]
                future = df[self.target_col].iloc[i:i + h].values
                z = (future - self.val_mean) / self.val_std
                window[f'regime_h{h}'] = 'extreme' if np.any(z > self.extreme_z_threshold) else 'normal'
                recent = df[self.target_col].iloc[i - 12:i].values
                window[f'stats_h{h}'] = {
                    'last_value':  float(df[self.target_col].iloc[i - 1]),
                    'rolling_std': float(np.std(recent)),
                    'trend':       float(recent[-1] - recent[0]),
                }
            windows.append(window)
        return windows


# ============================================================================
# Flux (DayCent NEE)
# ============================================================================

class FluxDataProcessor:
    """DayCent-simulated Net Ecosystem Exchange (NEE). Target: NEE."""

    def __init__(self, csv_path: str, context_len: int = 96,
                 horizons: List[int] = [6, 18, 36]):
        self.context_len = context_len
        self.horizons    = horizons
        self.max_horizon = max(horizons)
        self.target_col  = 'NEE'

        self.df = pd.read_csv(csv_path)
        if 'date' not in self.df.columns:
            self.df['date'] = pd.to_datetime(
                self.df['Year'].astype(str) + '-' +
                self.df['Month'].astype(str).str.zfill(2),
                format='%Y-%m')
        else:
            self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in num_cols if c != self.target_col]

        self.df['month_sin'] = np.sin(2 * np.pi * self.df['date'].dt.month / 12).astype(np.float32)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['date'].dt.month / 12).astype(np.float32)
        self.feature_cols.extend(['month_sin', 'month_cos'])

    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        n         = len(self.df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))
        self.train_df = self.df.iloc[:train_end].copy()
        self.val_df   = self.df.iloc[train_end:val_end].copy()
        self.test_df  = self.df.iloc[val_end:].copy()

        self.feature_scaler = StandardScaler()
        self.target_scaler  = StandardScaler()
        self.feature_scaler.fit(self.train_df[self.feature_cols].values)
        self.target_scaler.fit(self.train_df[[self.target_col]].values)

        train_vals               = self.train_df[self.target_col]
        self.val_mean            = float(train_vals.mean())
        self.val_std             = float(train_vals.std())
        self.extreme_z_threshold = 1.5
        return self.train_df, self.val_df, self.test_df

    def scale_data(self, df):
        X = self.feature_scaler.transform(df[self.feature_cols].values).astype(np.float32)
        y = self.target_scaler.transform(df[[self.target_col]].values).flatten().astype(np.float32)
        return X, y

    def create_windows(self, split: str = 'train',
                       sample_fraction: float = 1.0) -> List[Dict]:
        df = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}[split]
        X, y = self.scale_data(df)
        all_idx = list(range(self.context_len, len(df) - self.max_horizon))
        if sample_fraction < 1.0:
            all_idx = random.sample(all_idx, int(len(all_idx) * sample_fraction))

        windows = []
        for i in all_idx:
            window = {'context_X':    X[i - self.context_len:i],
                      'context_dates': df['date'].iloc[i - self.context_len:i].values,
                      'current_idx':  i}
            for h in self.horizons:
                window[f'labels_h{h}'] = y[i:i + h]
                future = df[self.target_col].iloc[i:i + h].values
                z = (future - self.val_mean) / self.val_std
                window[f'regime_h{h}'] = 'extreme' if np.any(z > self.extreme_z_threshold) else 'normal'
                recent = df[self.target_col].iloc[i - 12:i].values
                window[f'stats_h{h}'] = {
                    'last_value':  float(df[self.target_col].iloc[i - 1]),
                    'rolling_std': float(np.std(recent)),
                    'trend':       float(recent[-1] - recent[0]),
                }
            windows.append(window)
        return windows


# ============================================================================
# Soil Moisture (Quench platform)
# ============================================================================

class SoilMoistureProcessor:
    """Quench in-situ soil moisture at 50 cm depth from 42 Colorado stations.
    Expected CSV columns: network, station_id, timestamp, soil_moisture.
    """

    def __init__(self, csv_path: str, context_len: int = 96,
                 horizons: List[int] = [6, 12, 18]):
        self.context_len = context_len
        self.horizons    = horizons
        self.max_horizon = max(horizons)
        self.target_col  = 'soil_moisture'

        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = (df.groupby('timestamp')['soil_moisture']
                .mean().reset_index()
                .rename(columns={'timestamp': 'date'}))
        df = df.sort_values('date').reset_index(drop=True)
        df['soil_moisture'] = df['soil_moisture'].interpolate(method='linear')

        df['month']     = df['date'].dt.month
        df['dayofyear'] = df['date'].dt.dayofyear
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
        df['day_sin']   = np.sin(2 * np.pi * df['dayofyear'] / 365).astype(np.float32)
        df['day_cos']   = np.cos(2 * np.pi * df['dayofyear'] / 365).astype(np.float32)

        self.df           = df
        self.feature_cols = ['month_sin', 'month_cos', 'day_sin', 'day_cos']

    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        n         = len(self.df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))
        self.train_df = self.df.iloc[:train_end].copy()
        self.val_df   = self.df.iloc[train_end:val_end].copy()
        self.test_df  = self.df.iloc[val_end:].copy()

        self.feature_scaler = StandardScaler()
        self.target_scaler  = StandardScaler()
        self.feature_scaler.fit(self.train_df[self.feature_cols].values)
        self.target_scaler.fit(self.train_df[[self.target_col]].values)

        train_vals               = self.train_df[self.target_col]
        self.val_mean            = float(train_vals.mean())
        self.val_std             = float(train_vals.std())
        self.extreme_z_threshold = 1.5
        return self.train_df, self.val_df, self.test_df

    def scale_data(self, df):
        X = self.feature_scaler.transform(df[self.feature_cols].values).astype(np.float32)
        y = self.target_scaler.transform(df[[self.target_col]].values).flatten().astype(np.float32)
        return X, y

    def create_windows(self, split: str = 'train',
                       sample_fraction: float = 1.0) -> List[Dict]:
        df = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}[split]
        X, y = self.scale_data(df)
        all_idx = list(range(self.context_len, len(df) - self.max_horizon))
        if sample_fraction < 1.0:
            all_idx = random.sample(all_idx, int(len(all_idx) * sample_fraction))

        windows = []
        for i in all_idx:
            window = {'context_X':    X[i - self.context_len:i],
                      'context_dates': df['date'].iloc[i - self.context_len:i].values,
                      'current_idx':  i}
            for h in self.horizons:
                window[f'labels_h{h}'] = y[i:i + h]
                future = df[self.target_col].iloc[i:i + h].values
                z = (future - self.val_mean) / self.val_std
                window[f'regime_h{h}'] = 'extreme' if np.any(z > self.extreme_z_threshold) else 'normal'
                recent = df[self.target_col].iloc[i - 12:i].values
                window[f'stats_h{h}'] = {
                    'last_value':  float(df[self.target_col].iloc[i - 1]),
                    'rolling_std': float(np.std(recent)),
                    'trend':       float(recent[-1] - recent[0]),
                }
            windows.append(window)
        return windows


# ============================================================================
# Factory
# ============================================================================

def get_processor(dataset: str, csv_path: str, context_len: int = 96,
                  horizons: Optional[List[int]] = None):
    """Return the correct processor for a given dataset name."""
    key = dataset.lower().replace('-', '').replace('_', '')
    if key == 'weather':
        return WeatherDataProcessor(csv_path, context_len, horizons or [6, 18, 36])
    elif key == 'etth1':
        return ETTDataProcessor(csv_path, 'ETTh1', context_len, horizons or [6, 18, 36])
    elif key == 'ettm1':
        return ETTDataProcessor(csv_path, 'ETTm1', context_len, horizons or [6, 18, 36])
    elif key == 'flux':
        return FluxDataProcessor(csv_path, context_len, horizons or [6, 18, 36])
    elif key == 'soilmoisture':
        return SoilMoistureProcessor(csv_path, context_len, horizons or [6, 12, 18])
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         "Choose: weather, etth1, ettm1, flux, soilmoisture")

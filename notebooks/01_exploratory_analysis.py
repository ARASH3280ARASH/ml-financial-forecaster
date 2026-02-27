"""
Notebook 01 — Exploratory Data Analysis
========================================

Explores raw financial data, checks distributions, stationarity,
and identifies patterns before feature engineering.

Run as: python notebooks/01_exploratory_analysis.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.data_loader import DataConfig, DataLoader

# ── Load Data ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Synthetic OHLCV Data")
print("=" * 60)

loader = DataLoader(DataConfig(source="synthetic", symbol="BTCUSD"))
df = loader.load()

print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} → {df.index.max()}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── Basic Statistics ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Descriptive Statistics")
print("=" * 60)

print(f"\n{df.describe()}")

# ── Return Analysis ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Return Analysis")
print("=" * 60)

returns = df["close"].pct_change().dropna()
print(f"Mean return:    {returns.mean():.6f}")
print(f"Std deviation:  {returns.std():.6f}")
print(f"Skewness:       {returns.skew():.4f}")
print(f"Kurtosis:       {returns.kurtosis():.4f}")
print(f"Min return:     {returns.min():.6f}")
print(f"Max return:     {returns.max():.6f}")
print(f"VaR (5%):       {returns.quantile(0.05):.6f}")

# ── Stationarity Check ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Stationarity Check")
print("=" * 60)

try:
    from statsmodels.tsa.stattools import adfuller

    # Price level (expected non-stationary)
    adf_price = adfuller(df["close"].dropna(), autolag="AIC")
    print(f"ADF on price:   stat={adf_price[0]:.4f}, p={adf_price[1]:.4f}")
    print(f"  → {'Stationary' if adf_price[1] < 0.05 else 'Non-stationary'}")

    # Returns (expected stationary)
    adf_ret = adfuller(returns, autolag="AIC")
    print(f"ADF on returns: stat={adf_ret[0]:.4f}, p={adf_ret[1]:.4f}")
    print(f"  → {'Stationary' if adf_ret[1] < 0.05 else 'Non-stationary'}")
except ImportError:
    print("  (statsmodels not installed — skipping ADF test)")

# ── Autocorrelation ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Autocorrelation Structure")
print("=" * 60)

for lag in [1, 5, 10, 20]:
    acf = returns.autocorr(lag=lag)
    print(f"Autocorrelation (lag={lag:2d}): {acf:+.4f}")

# ── Volume Analysis ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Volume Analysis")
print("=" * 60)

volume = df["volume"]
print(f"Mean volume:    {volume.mean():,.0f}")
print(f"Median volume:  {volume.median():,.0f}")
print(f"Volume std:     {volume.std():,.0f}")

vol_return_corr = df["volume"].pct_change().corr(returns.abs())
print(f"Volume-|Return| correlation: {vol_return_corr:.4f}")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
print(f"Total bars:       {len(df):,}")
print(f"Trading days:     {len(df) // 24:,}")
print("Next: Run 02_feature_engineering.py")

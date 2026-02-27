"""
Notebook 02 — Feature Engineering Pipeline
===========================================

Demonstrates the full feature engineering pipeline:
technical indicators, statistical features, sentiment, and selection.

Run as: python notebooks/02_feature_engineering.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.data_loader import DataConfig, DataLoader
from src.data.feature_store import FeatureStore
from src.features.technical_features import TechnicalFeatureEngine
from src.features.statistical_features import StatisticalFeatureEngine
from src.features.sentiment_features import SentimentFeatureEngine

# ── Load Data ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

loader = DataLoader(DataConfig(source="synthetic"))
df = loader.load()
print(f"Loaded {len(df)} bars")

# ── Technical Features ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Technical Features")
print("=" * 60)

tech_engine = TechnicalFeatureEngine()
tech_features = tech_engine.compute_all(df)
print(f"Technical features: {tech_features.shape[1]} columns")
print(f"Sample features: {list(tech_features.columns[:10])}")

# ── Statistical Features ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Statistical Features")
print("=" * 60)

stat_engine = StatisticalFeatureEngine()
stat_features = stat_engine.compute_all(df)
print(f"Statistical features: {stat_features.shape[1]} columns")
print(f"Sample features: {list(stat_features.columns[:10])}")

# ── Sentiment Features ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Sentiment Features (Simulated)")
print("=" * 60)

sent_engine = SentimentFeatureEngine()
sent_features = sent_engine.generate_simulated(df)
print(f"Sentiment features: {sent_features.shape[1]} columns")
print(f"Sample features: {list(sent_features.columns[:10])}")

# ── Combine All Features ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Combined Feature Matrix")
print("=" * 60)

all_features = pd.concat([tech_features, stat_features, sent_features], axis=1)
print(f"Total features: {all_features.shape[1]}")
print(f"Total samples:  {all_features.shape[0]}")

# Clean NaN/Inf
all_features = all_features.replace([np.inf, -np.inf], np.nan)
nan_pct = all_features.isna().mean()
print(f"\nFeatures with >50% NaN (drop candidates):")
high_nan = nan_pct[nan_pct > 0.5]
for feat, pct in high_nan.items():
    print(f"  {feat}: {pct:.1%}")

# ── Feature Store ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Register in Feature Store")
print("=" * 60)

store = FeatureStore()
for col in all_features.columns:
    group = "technical" if col in tech_features.columns else \
            "statistical" if col in stat_features.columns else "sentiment"
    store.register(col, group=group, data=all_features[col])

print(store.summary())

# ── Feature Statistics ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Feature Statistics")
print("=" * 60)

clean = all_features.dropna()
print(f"Clean samples: {len(clean)} (dropped {len(all_features) - len(clean)} rows)")
print(f"\nCorrelation with close returns (top 15):")

returns = df["close"].pct_change().shift(-1)
correlations = clean.corrwith(returns.loc[clean.index]).abs().sort_values(ascending=False)
for feat, corr in correlations.head(15).items():
    print(f"  {feat:40s} {corr:.4f}")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"Features ready: {all_features.shape[1]}")
print("Next: Run 03_model_comparison.py")

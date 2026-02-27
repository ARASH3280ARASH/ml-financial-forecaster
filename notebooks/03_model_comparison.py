"""
Notebook 03 — Model Comparison
===============================

Trains and compares multiple model architectures on the same dataset:
- Gradient Boosting (XGBoost / LightGBM)
- LSTM with Attention
- Transformer
- Ensemble

Run as: python notebooks/03_model_comparison.py
"""

import sys
sys.path.insert(0, ".")

import logging
import numpy as np
import pandas as pd

from src.data.data_loader import DataConfig, DataLoader
from src.data.preprocessor import Preprocessor, ScalingMethod
from src.features.technical_features import TechnicalFeatureEngine
from src.features.statistical_features import StatisticalFeatureEngine
from src.models.gradient_boosting import GradientBoostingModel
from src.models.ensemble_model import EnsembleModel, EnsembleMethod
from src.training.trainer import Trainer, TrainingConfig
from src.training.cross_validator import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Prepare Data ───────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Preparing Data")
print("=" * 60)

loader = DataLoader(DataConfig(source="synthetic"))
df = loader.load()

# Generate features
tech = TechnicalFeatureEngine(periods=[7, 14, 21, 50]).compute_all(df)
stat = StatisticalFeatureEngine(windows=[10, 20, 50]).compute_all(df)
features = pd.concat([tech, stat], axis=1)

# Target: next-period return
target = df["close"].pct_change().shift(-1)

# Align and clean
common = features.dropna().index.intersection(target.dropna().index)
X = features.loc[common].replace([np.inf, -np.inf], 0).fillna(0)
y = target.loc[common]

print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")

# ── Train Models ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training Models")
print("=" * 60)

config = TrainingConfig(train_ratio=0.7, val_ratio=0.15, target_horizon=0)
trainer = Trainer(config=config)

# Model 1: XGBoost
print("\n--- XGBoost ---")
xgb_model = GradientBoostingModel("xgboost", backend="xgboost")
xgb_result = trainer.train(xgb_model, X, y)

# Model 2: LightGBM
print("\n--- LightGBM ---")
lgb_model = GradientBoostingModel("lightgbm", backend="lightgbm")
lgb_result = trainer.train(lgb_model, X, y)

# Model 3: Ensemble
print("\n--- Ensemble (Mean) ---")
xgb2 = GradientBoostingModel("xgb_ens", backend="xgboost")
lgb2 = GradientBoostingModel("lgb_ens", backend="lightgbm")
ensemble = EnsembleModel("ensemble_mean", models=[xgb2, lgb2], method=EnsembleMethod.MEAN)
ens_result = trainer.train(ensemble, X, y)

# ── Comparison Table ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Model Comparison")
print("=" * 60)

comparison = trainer.comparison_table()
print(f"\n{comparison.to_string(index=False)}")

# ── Walk-Forward Validation ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Walk-Forward Cross-Validation")
print("=" * 60)

wf = WalkForwardValidator(n_splits=5, min_train_size=5000, embargo_bars=24)

for name, model_cls in [("XGBoost", lambda: GradientBoostingModel("xgb_cv", backend="xgboost")),
                          ("LightGBM", lambda: GradientBoostingModel("lgb_cv", backend="lightgbm"))]:
    print(f"\n--- {name} Walk-Forward CV ---")
    cv_result = wf.validate(model_cls(), X, y)
    print(f"  {cv_result.summary()}")

# ── Feature Importance ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature Importance (Best Model)")
print("=" * 60)

best = trainer.get_best_model()
if best and best.model.get_feature_importance():
    importance = best.model.get_feature_importance()
    print(f"\nTop 20 features ({best.model.name}):")
    for i, (feat, score) in enumerate(list(importance.items())[:20]):
        print(f"  {i+1:2d}. {feat:40s} {score:.4f}")

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETE")
print("=" * 60)
print("Next: Run 04_production_pipeline.py")

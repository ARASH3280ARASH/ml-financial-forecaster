"""
Notebook 04 — Production Pipeline
==================================

Demonstrates the full end-to-end pipeline from raw data
to backtested strategy with performance reporting.

Run as: python notebooks/04_production_pipeline.py
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
from src.features.sentiment_features import SentimentFeatureEngine
from src.models.gradient_boosting import GradientBoostingModel
from src.models.ensemble_model import EnsembleModel, EnsembleMethod
from src.models.model_registry import ModelRegistry
from src.training.trainer import Trainer, TrainingConfig
from src.training.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.backtester import Backtester, BacktestConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ══════════════════════════════════════════════════════════════
# STAGE 1: DATA INGESTION
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STAGE 1: Data Ingestion")
print("=" * 60)

loader = DataLoader(DataConfig(source="synthetic", symbol="BTCUSD"))
df = loader.load()
print(f"Loaded {len(df)} bars for BTCUSD")

# ══════════════════════════════════════════════════════════════
# STAGE 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 2: Feature Engineering")
print("=" * 60)

tech_features = TechnicalFeatureEngine(periods=[7, 14, 21, 50]).compute_all(df)
stat_features = StatisticalFeatureEngine(windows=[10, 20, 50]).compute_all(df)
sent_features = SentimentFeatureEngine().generate_simulated(df)

features = pd.concat([tech_features, stat_features, sent_features], axis=1)
features = features.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Feature matrix: {features.shape}")

# ══════════════════════════════════════════════════════════════
# STAGE 3: MODEL TRAINING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 3: Model Training")
print("=" * 60)

target = df["close"].pct_change().shift(-1)
common = features.index.intersection(target.dropna().index)
X, y = features.loc[common], target.loc[common]

config = TrainingConfig(train_ratio=0.7, val_ratio=0.15, target_horizon=0)
trainer = Trainer(config=config)

# Train individual models
xgb = GradientBoostingModel("xgboost_prod", backend="xgboost")
lgb = GradientBoostingModel("lightgbm_prod", backend="lightgbm")

xgb_result = trainer.train(xgb, X, y)
lgb_result = trainer.train(lgb, X, y)

# Train ensemble
xgb2 = GradientBoostingModel("xgb_ens", backend="xgboost")
lgb2 = GradientBoostingModel("lgb_ens", backend="lightgbm")
ensemble = EnsembleModel("production_ensemble", models=[xgb2, lgb2], method=EnsembleMethod.WEIGHTED)
ens_result = trainer.train(ensemble, X, y)

# ══════════════════════════════════════════════════════════════
# STAGE 4: MODEL REGISTRY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 4: Model Registry")
print("=" * 60)

registry = ModelRegistry("models/registry")
registry.register(ens_result.model, version="1.0.0", tags=["production"])
print(registry.summary())

# ══════════════════════════════════════════════════════════════
# STAGE 5: EXPERIMENT TRACKING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 5: Experiment Tracking")
print("=" * 60)

tracker = ExperimentTracker("experiments")
for result in [xgb_result, lgb_result, ens_result]:
    tracker.log_experiment(
        model_name=result.model.name,
        params=result.model.params,
        metrics={
            "test_rmse": result.test_metrics.rmse,
            "test_mae": result.test_metrics.mae,
            "test_r2": result.test_metrics.r2,
            "test_dir_acc": result.test_metrics.directional_accuracy,
        },
        tags=["production_run"],
    )
print(tracker.summary())

# ══════════════════════════════════════════════════════════════
# STAGE 6: BACKTESTING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 6: Backtesting")
print("=" * 60)

# Generate predictions on test split
split_idx = int(len(X) * 0.85)
X_test = X.iloc[split_idx:]
predictions = pd.Series(
    ens_result.model.predict(X_test.fillna(0).replace([np.inf, -np.inf], 0)),
    index=X_test.index,
)

bt = Backtester(BacktestConfig(
    initial_capital=100_000,
    transaction_cost_bps=10,
    slippage_bps=5,
))
bt_result = bt.run(predictions, df["close"].loc[X_test.index], strategy_name="ML Ensemble")
print(bt_result.summary())

# ══════════════════════════════════════════════════════════════
# STAGE 7: FINANCIAL METRICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 7: Financial Performance Metrics")
print("=" * 60)

calc = MetricsCalculator()
fin_metrics = calc.from_returns(bt_result.returns)
print(fin_metrics.summary())

print(f"\n  Total Return:        {fin_metrics.total_return:.2%}")
print(f"  Annualized Return:   {fin_metrics.annualized_return:.2%}")
print(f"  Annualized Vol:      {fin_metrics.annualized_volatility:.2%}")
print(f"  Sharpe Ratio:        {fin_metrics.sharpe_ratio:.3f}")
print(f"  Sortino Ratio:       {fin_metrics.sortino_ratio:.3f}")
print(f"  Max Drawdown:        {fin_metrics.max_drawdown:.2%}")
print(f"  VaR (95%):           {fin_metrics.var_95:.4f}")
print(f"  Win Rate:            {fin_metrics.win_rate:.2%}")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PRODUCTION PIPELINE COMPLETE")
print("=" * 60)
print("All stages executed successfully.")

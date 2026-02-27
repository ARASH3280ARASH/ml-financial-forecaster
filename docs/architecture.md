# Architecture Overview

## System Design

ML Financial Forecaster follows a modular pipeline architecture where each stage
is independently testable and configurable.

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐    ┌──────────────┐
│ Data Loader  │───▶│   Features   │───▶│  Training  │───▶│ Evaluation │───▶│ Visualization│
│ (Ingestion)  │    │ (Engineering)│    │ (Models)   │    │ (Metrics)  │    │ (Reports)    │
└─────────────┘    └──────────────┘    └───────────┘    └────────────┘    └──────────────┘
       │                  │                  │                 │                  │
       ▼                  ▼                  ▼                 ▼                  ▼
   CSV/Synthetic    50+ Technical     XGBoost/LightGBM    Sharpe/Sortino      HTML Reports
   Yahoo Finance    Statistical       LSTM+Attention       Backtesting        Plot Suite
                    Sentiment         Transformer          Statistical Tests
                    Feature Store     Ensemble/Stacking    Monte Carlo
```

## Module Responsibilities

### `src/data/`
- **DataLoader**: Unified data ingestion from CSV, synthetic, or API sources
- **Preprocessor**: Normalization, missing value handling, temporal splitting with embargo
- **FeatureStore**: Registry for computed features with metadata and versioning

### `src/features/`
- **TechnicalFeatureEngine**: 50+ indicators (RSI, MACD, Bollinger, ATR, etc.)
- **StatisticalFeatureEngine**: Rolling moments, entropy, Hurst exponent, regime detection
- **SentimentFeatureEngine**: Lexicon-based scoring and simulated sentiment
- **FeatureSelector**: Correlation filtering, mutual information, RFE

### `src/models/`
- **BaseModel**: Abstract interface enforcing fit/predict/evaluate contract
- **GradientBoostingModel**: XGBoost and LightGBM with early stopping
- **LSTMAttentionModel**: LSTM + multi-head self-attention
- **TransformerModel**: Encoder-only Transformer for time series
- **EnsembleModel**: Mean, median, weighted, and stacking combinations
- **ModelRegistry**: Versioned model storage with metadata

### `src/training/`
- **Trainer**: End-to-end training orchestrator
- **HyperparameterTuner**: Optuna-based Bayesian optimization
- **WalkForwardValidator**: Expanding/sliding window temporal CV
- **PurgedKFold**: K-Fold with purging to prevent leakage
- **ExperimentTracker**: Lightweight experiment logging and comparison

### `src/evaluation/`
- **MetricsCalculator**: Sharpe, Sortino, Calmar, VaR/CVaR, drawdown
- **Backtester**: Strategy simulation with transaction costs and slippage
- **StatisticalTester**: t-tests, Diebold-Mariano, bootstrap CI, normality tests

### `src/visualization/`
- **ModelPlotter**: Feature importance, learning curves, residual analysis
- **PerformancePlotter**: Equity curves, drawdown, rolling metrics, heatmaps
- **ReportGenerator**: Automated HTML report generation

## Key Design Decisions

1. **Temporal-first splitting**: All cross-validation respects time ordering with embargo gaps
2. **No look-ahead bias**: Features are computed causally (past data only)
3. **Graceful degradation**: Models fall back to simpler alternatives when dependencies are unavailable
4. **Pluggable architecture**: New models implement BaseModel; new features register in FeatureStore
5. **Financial-native metrics**: Sharpe, Sortino, and drawdown alongside standard ML metrics

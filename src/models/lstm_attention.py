"""LSTM with self-attention — deep learning model for sequential financial data.

Combines LSTM memory cells with a self-attention mechanism
to capture both short-term patterns and long-range dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMAttentionModel(BaseModel):
    """LSTM network with self-attention for time series forecasting.

    Architecture:
        Input → LSTM(hidden) → LSTM(hidden) → SelfAttention → Dense → Output

    Args:
        name: Model identifier.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        attention_heads: Number of attention heads.
        sequence_length: Input sequence length (look-back window).
        dropout: Dropout rate.
        learning_rate: Optimizer learning rate.
        epochs: Maximum training epochs.
        batch_size: Training batch size.

    Example:
        >>> model = LSTMAttentionModel("lstm_attn", sequence_length=60)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    def __init__(
        self,
        name: str = "lstm_attention",
        hidden_size: int = 128,
        num_layers: int = 2,
        attention_heads: int = 4,
        sequence_length: int = 60,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "attention_heads": attention_heads,
            "sequence_length": sequence_length,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        super().__init__(name, params)
        self._model: Any = None
        self._scaler: Any = None

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Build and train the LSTM-Attention model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.warning("TensorFlow not available — using fallback linear model")
            self._fit_fallback(X, y)
            return

        X_seq, y_seq = self._create_sequences(X.values, y.values)
        if len(X_seq) == 0:
            logger.warning("Not enough data for sequences, using fallback")
            self._fit_fallback(X, y)
            return

        n_features = X.shape[1]
        seq_len = self.params["sequence_length"]

        model = self._build_model(seq_len, n_features, keras, tf)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
            loss="mse",
            metrics=["mae"],
        )

        # Temporal validation split
        val_split = int(len(X_seq) * 0.85)
        X_tr, X_val = X_seq[:val_split], X_seq[val_split:]
        y_tr, y_val = y_seq[:val_split], y_seq[val_split:]

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        self._model = model
        logger.info("LSTM-Attention trained: %d parameters", model.count_params())

    def _build_model(self, seq_len: int, n_features: int, keras: Any, tf: Any) -> Any:
        """Construct the Keras model."""
        inputs = keras.Input(shape=(seq_len, n_features))

        # LSTM layers
        x = inputs
        for i in range(self.params["num_layers"]):
            return_seq = i < self.params["num_layers"] - 1
            x = keras.layers.LSTM(
                self.params["hidden_size"],
                return_sequences=True if return_seq or i == self.params["num_layers"] - 1 else False,
                dropout=self.params["dropout"],
                recurrent_dropout=self.params["dropout"] * 0.5,
            )(x)

        # Self-attention
        attention = keras.layers.MultiHeadAttention(
            num_heads=self.params["attention_heads"],
            key_dim=self.params["hidden_size"] // self.params["attention_heads"],
        )(x, x)
        x = keras.layers.Add()([x, attention])
        x = keras.layers.LayerNormalization()(x)

        # Global average pooling and dense head
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dropout(self.params["dropout"])(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(self.params["dropout"] * 0.5)(x)
        outputs = keras.layers.Dense(1)(x)

        return keras.Model(inputs, outputs)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self._model is None:
            return self._predict_fallback(X)

        X_seq, _ = self._create_sequences(X.values, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.zeros(len(X))

        preds = self._model.predict(X_seq, verbose=0).flatten()

        # Pad to match original length
        pad_len = len(X) - len(preds)
        if pad_len > 0:
            preds = np.concatenate([np.full(pad_len, preds[0]), preds])

        return preds

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target array.

        Returns:
            Tuple of (X_sequences, y_targets).
        """
        seq_len = self.params["sequence_length"]
        if len(X) <= seq_len:
            return np.array([]), np.array([])

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        sequences = []
        targets = []
        for i in range(seq_len, len(X)):
            sequences.append(X_clean[i - seq_len: i])
            targets.append(y[i])

        return np.array(sequences), np.array(targets)

    def _fit_fallback(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fallback to sklearn when TensorFlow is unavailable."""
        from sklearn.linear_model import Ridge

        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        self._model = None
        self._fallback = Ridge(alpha=1.0)
        self._fallback.fit(X_clean, y)

    def _predict_fallback(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using fallback model."""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        return self._fallback.predict(X_clean)

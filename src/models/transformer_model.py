"""Transformer model — attention-based architecture for time series forecasting.

Implements a Transformer encoder for capturing complex temporal
patterns in financial data without recurrence.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class TransformerModel(BaseModel):
    """Transformer encoder for financial time series prediction.

    Architecture:
        Input → Positional Encoding → N × TransformerEncoder → Pooling → Dense → Output

    Uses learned positional embeddings and multi-head self-attention
    to model temporal dependencies.

    Args:
        name: Model identifier.
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        d_ff: Feed-forward hidden dimension.
        sequence_length: Input sequence look-back.
        dropout: Dropout rate.
        learning_rate: Optimizer learning rate.
        epochs: Maximum training epochs.
        batch_size: Training batch size.

    Example:
        >>> model = TransformerModel("transformer_v1", d_model=64)
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        name: str = "transformer",
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        sequence_length: int = 60,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        params = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "sequence_length": sequence_length,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        super().__init__(name, params)
        self._model: Any = None
        self._fallback: Any = None

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Build and train the Transformer model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.warning("TensorFlow not available — using Ridge fallback")
            self._fit_fallback(X, y)
            return

        X_seq, y_seq = self._create_sequences(X.values, y.values)
        if len(X_seq) == 0:
            self._fit_fallback(X, y)
            return

        n_features = X.shape[1]
        seq_len = self.params["sequence_length"]

        model = self._build_model(seq_len, n_features, keras, tf)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.params["learning_rate"]
            ),
            loss="mse",
            metrics=["mae"],
        )

        val_split = int(len(X_seq) * 0.85)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        model.fit(
            X_seq[:val_split], y_seq[:val_split],
            validation_data=(X_seq[val_split:], y_seq[val_split:]),
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        self._model = model
        logger.info("Transformer trained: %d parameters", model.count_params())

    def _build_model(
        self, seq_len: int, n_features: int, keras: Any, tf: Any
    ) -> Any:
        """Construct the Keras Transformer model."""
        d_model = self.params["d_model"]

        inputs = keras.Input(shape=(seq_len, n_features))

        # Project input to d_model dimension
        x = keras.layers.Dense(d_model)(inputs)

        # Add learned positional encoding
        positions = keras.layers.Embedding(seq_len, d_model)(
            tf.range(seq_len)
        )
        x = x + positions
        x = keras.layers.Dropout(self.params["dropout"])(x)

        # Transformer encoder blocks
        for _ in range(self.params["n_layers"]):
            x = self._encoder_block(x, keras)

        # Global pooling and prediction head
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(self.params["dropout"])(x)
        outputs = keras.layers.Dense(1)(x)

        return keras.Model(inputs, outputs)

    def _encoder_block(self, x: Any, keras: Any) -> Any:
        """Single Transformer encoder block."""
        d_model = self.params["d_model"]

        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=self.params["n_heads"],
            key_dim=d_model // self.params["n_heads"],
            dropout=self.params["dropout"],
        )(x, x)
        attn_output = keras.layers.Dropout(self.params["dropout"])(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Position-wise feed-forward
        ff = keras.layers.Dense(self.params["d_ff"], activation="relu")(x)
        ff = keras.layers.Dense(d_model)(ff)
        ff = keras.layers.Dropout(self.params["dropout"])(ff)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

        return x

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self._model is None:
            return self._predict_fallback(X)

        X_seq, _ = self._create_sequences(X.values, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.zeros(len(X))

        preds = self._model.predict(X_seq, verbose=0).flatten()

        pad_len = len(X) - len(preds)
        if pad_len > 0:
            preds = np.concatenate([np.full(pad_len, preds[0]), preds])

        return preds

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences."""
        seq_len = self.params["sequence_length"]
        if len(X) <= seq_len:
            return np.array([]), np.array([])

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        sequences, targets = [], []
        for i in range(seq_len, len(X)):
            sequences.append(X_clean[i - seq_len: i])
            targets.append(y[i])

        return np.array(sequences), np.array(targets)

    def _fit_fallback(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fallback to Ridge regression."""
        from sklearn.linear_model import Ridge

        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        self._fallback = Ridge(alpha=1.0)
        self._fallback.fit(X_clean, y)

    def _predict_fallback(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with fallback model."""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        return self._fallback.predict(X_clean)

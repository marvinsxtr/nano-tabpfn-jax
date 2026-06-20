from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class NanoTabPFNClassifier:
    """scikit-learn-like interface for the JAX model."""

    def __init__(self, model: eqx.Module, predict_fn: Callable) -> None:
        self.model = eqx.nn.inference_mode(model, value=True)
        self.X_train = None
        self.y_train = None
        self.num_classes = None
        self.predict_fn = predict_fn

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Store training data and infer the number of classes."""
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = int(max(set(y_train))) + 1

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Run the model and return per-class probabilities for the test rows."""
        x = jnp.concatenate((self.X_train, X_test))

        num_features = x.shape[1]
        if num_features < 10:
            x = jnp.concatenate([x, jnp.zeros((x.shape[0], 10 - num_features))], axis=1)

        num_test = len(X_test)
        mean = self.y_train.mean()
        y = jnp.concatenate([self.y_train, np.full(num_test, mean)])

        num_train = len(self.X_train)
        train_mask = jnp.arange(len(x)) < num_train

        out = self.predict_fn(self.model, x, y, train_mask=train_mask)
        out = out[num_train:, : self.num_classes]
        return np.array(jax.nn.softmax(out, axis=1))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for the test data."""
        return self.predict_proba(X_test).argmax(axis=1)

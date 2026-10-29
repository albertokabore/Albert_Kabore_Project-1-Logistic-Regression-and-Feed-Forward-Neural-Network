"""Feature standardisation and conversion of weights back to the raw feature scale."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """z-score scaling: x_tilde = (x - mu) / s, with mu and s estimated on training data only.

    Raw counts range from 0 to more than 100 while the binary features are 0 or 1.
    Putting every feature on the same scale makes gradient descent converge in far
    fewer steps, because no single feature dominates the gradient.
    """

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale

    def to_raw_weights(self, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
        """Rewrite w . x_tilde + b as w_raw . x + b_raw so the model applies to raw features.

        w . (x - mu) / s + b  =  (w / s) . x  +  (b - sum(w * mu / s)).
        """
        w_raw = w / self.scale
        return w_raw, float(b - w_raw @ self.mean)

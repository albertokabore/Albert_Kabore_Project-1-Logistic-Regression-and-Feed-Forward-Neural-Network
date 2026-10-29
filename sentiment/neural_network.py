"""A one-hidden-layer feed-forward network trained with backpropagation.

Architecture: 6 inputs -> H tanh hidden units -> 1 sigmoid output.
The hidden layer learns non-linear combinations of the features. Logistic
regression is the separate model with a direct input-to-output connection.
"""

from dataclasses import dataclass, field

import numpy as np

from .logistic_regression import cross_entropy, sigmoid


@dataclass
class FeedForwardNetwork:
    n_hidden: int = 4
    learning_rate: float = 0.5
    epochs: int = 2_000
    l2: float = 0.0
    seed: int = 0
    params: dict[str, np.ndarray] = field(default_factory=dict)
    loss_history: list[float] = field(default_factory=list)

    def init_params(self, n_features: int) -> None:
        """Use zero-mean Gaussian weights with variance 1 / fan_in.

        Weights must not all start at zero: every hidden unit would then receive the
        same gradient and they would stay identical forever.
        """
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": rng.normal(0.0, np.sqrt(1.0 / n_features), (self.n_hidden, n_features)),
            "b1": np.zeros(self.n_hidden),
            "w2": rng.normal(0.0, np.sqrt(1.0 / self.n_hidden), self.n_hidden),
            "b2": np.zeros(1),
        }

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        p = self.params
        h = np.tanh(X @ p["W1"].T + p["b1"])  # hidden activations, (m, H)
        z = h @ p["w2"] + p["b2"][0]  # output logit, (m,)
        return {"h": h, "z": z, "y_hat": sigmoid(z)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["y_hat"]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.params
        penalty = 0.5 * self.l2 * (np.sum(p["W1"] ** 2) + np.sum(p["w2"] ** 2))
        return cross_entropy(self.forward(X)["z"], y) + float(penalty)

    def gradients(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagation (chain rule applied from the output back to the inputs)."""
        p = self.params
        cache = self.forward(X)
        m = len(y)
        dz = (cache["y_hat"] - y) / m  # dL/dz for sigmoid + cross-entropy, (m,)
        dh = np.outer(dz, p["w2"]) * (1.0 - cache["h"] ** 2)  # tanh'(a) = 1 - tanh(a)^2
        return {
            "W1": dh.T @ X + self.l2 * p["W1"],
            "b1": dh.sum(axis=0),
            "w2": cache["h"].T @ dz + self.l2 * p["w2"],
            "b2": np.array([dz.sum()]),
        }

    def step(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        grads = self.gradients(X, y)
        for name, grad in grads.items():
            self.params[name] = self.params[name] - self.learning_rate * grad
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeedForwardNetwork":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_params(X.shape[1])
        self.loss_history = [self.loss(X, y)]
        for _ in range(self.epochs):
            self.step(X, y)
            self.loss_history.append(self.loss(X, y))
        return self

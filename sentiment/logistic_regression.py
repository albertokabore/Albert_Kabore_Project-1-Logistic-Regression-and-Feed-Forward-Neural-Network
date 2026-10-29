"""Binary logistic regression trained by gradient descent on the cross-entropy loss."""

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z):
    """Numerically stable logistic function: sigma(z) = 1 / (1 + e^-z).

    Written as exp(-log(1 + e^-z)) so that large |z| never overflows.
    """
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))


def cross_entropy(z: np.ndarray, y: np.ndarray) -> float:
    """Mean of -[y log sigma(z) + (1 - y) log(1 - sigma(z))], computed from logits z.

    Uses the identity  -log sigma(z) = log(1 + e^-z)  to stay finite for any z.
    """
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


@dataclass
class LogisticRegression:
    """P(y = 1 | x) = sigma(w . x + b).

    learning_rate: step size eta of gradient descent.
    epochs:        maximum number of full passes over the training data.
    l2:            strength lambda of an optional (lambda / 2) * ||w||^2 penalty; the
                   bias is not penalised. The default 0 gives the plain maximum-
                   likelihood estimate. A small penalty keeps w finite when the
                   training data are linearly separable.
    tol:           stop early once the gradient norm drops below this value.
    """

    learning_rate: float = 0.5
    epochs: int = 100_000
    l2: float = 0.0
    tol: float = 1e-6
    w: np.ndarray | None = None
    b: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def init_params(self, n_features: int) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        penalty = 0.5 * self.l2 * float(self.w @ self.w)
        return cross_entropy(self.decision_function(X), y) + penalty

    def gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Gradient of the mean regularised loss:
        dL/dw = (1/m) X^T (y_hat - y) + lambda * w,   dL/db = (1/m) sum(y_hat - y).
        """
        error = self.predict_proba(X) - y
        grad_w = X.T @ error / len(y) + self.l2 * self.w
        grad_b = float(np.mean(error))
        return grad_w, grad_b

    def step(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """One gradient-descent update on (X, y); returns the gradient that was used."""
        grad_w, grad_b = self.gradient(X, y)
        self.w = self.w - self.learning_rate * grad_w
        self.b = self.b - self.learning_rate * grad_b
        return grad_w, grad_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Full-batch gradient descent from w = 0, b = 0.

        The loss is convex. Full column rank of [X, 1] gives strict convexity at
        finite parameters; a finite unregularized optimum also needs absence of
        complete or quasi separation. A suitable step size gives convergence.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_params(X.shape[1])
        self.loss_history = [self.loss(X, y)]
        for _ in range(self.epochs):
            grad_w, grad_b = self.gradient(X, y)
            if np.sqrt(grad_w @ grad_w + grad_b**2) < self.tol:
                break
            self.w = self.w - self.learning_rate * grad_w
            self.b = self.b - self.learning_rate * grad_b
            self.loss_history.append(self.loss(X, y))
        return self

"""Model tests.

These use small randomly generated feature matrices so that they run in seconds
without downloading the IMDB data. The generated numbers only exercise the code;
no reported result is based on them.
"""

import numpy as np
import pytest

from sentiment import FeedForwardNetwork, LogisticRegression, Standardizer, sigmoid, stratified_split


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    true_w = np.array([1.5, -1.2, -0.5, 0.3, 0.2, 0.0])
    y = (rng.random(400) < sigmoid(X @ true_w + 0.1)).astype(float)
    return X, y


def numerical_gradient(f, theta, eps=1e-6):
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step.flat[i] = eps
        grad.flat[i] = (f(theta + step) - f(theta - step)) / (2 * eps)
    return grad


def newton_solution(X, y, l2=0.0, iterations=30):
    A = np.hstack([X, np.ones((len(y), 1))])
    reg = np.diag([l2] * X.shape[1] + [0.0])
    theta = np.zeros(A.shape[1])
    for _ in range(iterations):
        p = sigmoid(A @ theta)
        grad = A.T @ (p - y) / len(y) + reg @ theta
        hessian = A.T @ (A * (p * (1 - p))[:, None]) / len(y) + reg
        theta -= np.linalg.solve(hessian, grad)
    return theta


def test_sigmoid_is_stable_and_symmetric():
    z = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    assert s[2] == 0.5
    np.testing.assert_allclose(s + sigmoid(-z), 1.0)


@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_logistic_regression_gradient_matches_finite_differences(data, l2):
    X, y = data
    model = LogisticRegression(l2=l2)
    theta = np.append(np.random.default_rng(1).normal(size=6), 0.3)

    def loss_at(t):
        model.w, model.b = t[:6], t[6]
        return model.loss(X, y)

    numeric = numerical_gradient(loss_at, theta.copy())
    model.w, model.b = theta[:6], theta[6]
    grad_w, grad_b = model.gradient(X, y)
    np.testing.assert_allclose(np.append(grad_w, grad_b), numeric, atol=1e-6)


def test_network_gradient_matches_finite_differences(data):
    X, y = data
    net = FeedForwardNetwork(n_hidden=3, l2=0.1)
    net.init_params(X.shape[1])
    analytic = net.gradients(X, y)
    for name, param in net.params.items():

        def loss_at(value, name=name):
            original = net.params[name]
            net.params[name] = value
            result = net.loss(X, y)
            net.params[name] = original
            return result

        np.testing.assert_allclose(
            analytic[name], numerical_gradient(loss_at, param.copy()), atol=1e-6, err_msg=name
        )


def test_gradient_descent_reaches_newton_optimum(data):
    X, y = data
    model = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(np.append(model.w, model.b), newton_solution(X, y), atol=1e-4)


def test_loss_decreases_monotonically(data):
    X, y = data
    history = np.array(LogisticRegression(epochs=500).fit(X, y).loss_history)
    assert np.all(np.diff(history) <= 1e-12)


def test_network_training_reduces_loss(data):
    X, y = data
    net = FeedForwardNetwork(epochs=300, seed=0).fit(X, y)
    assert net.loss_history[-1] < 0.8 * net.loss_history[0]


def test_one_step_matches_hand_calculation():
    """The single-step example from Jurafsky & Martin's logistic regression chapter."""
    x = np.array([[3.0, 2.0, 1.0, 3.0, 0.0, np.log(66)]])
    model = LogisticRegression(learning_rate=0.1)
    model.init_params(6)
    grad_w, grad_b = model.step(x, np.array([1.0]))
    np.testing.assert_allclose(grad_w, -0.5 * x[0])
    np.testing.assert_allclose(model.w, 0.05 * x[0])
    assert grad_b == -0.5 and model.b == pytest.approx(0.05)


def test_raw_weights_give_identical_predictions(data):
    X, y = data
    X_raw = X * np.array([10, 5, 1, 3, 1, 0.5]) + np.array([10, 9, 0, 6, 0, 5])
    scaler = Standardizer().fit(X_raw)
    model = LogisticRegression().fit(scaler.transform(X_raw), y)
    w_raw, b_raw = scaler.to_raw_weights(model.w, model.b)
    np.testing.assert_allclose(sigmoid(X_raw @ w_raw + b_raw), model.predict_proba(scaler.transform(X_raw)))
    # Unregularised maximum likelihood does not depend on feature scaling.
    np.testing.assert_allclose(np.append(w_raw, b_raw), newton_solution(X_raw, y), atol=1e-4)


def test_stratified_split_keeps_class_balance():
    y = np.array([0] * 100 + [1] * 100)
    parts = stratified_split(y, (0.7, 0.15, 0.15), seed=1)
    assert [len(p) for p in parts] == [140, 30, 30]
    assert len(np.unique(np.concatenate(parts))) == 200
    assert all(y[p].mean() == 0.5 for p in parts)

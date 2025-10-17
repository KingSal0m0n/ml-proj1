# implementations.py
# ------------------------------------------------------------------------------
# Self-contained implementations required by the public tests:
#   - mean_squared_error_gd
#   - mean_squared_error_sgd
#   - least_squares
#   - ridge_regression
#   - logistic_regression
#   - reg_logistic_regression
#
# Helpers are defined locally (no imports from other project files).
# ------------------------------------------------------------------------------

from __future__ import annotations
import numpy as np


# ----------------------------- MSE utilities ----------------------------------
def _mse_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Mean Squared Error loss: (1/(2N)) * ||y - tx @ w||^2
    """
    e = y - tx @ w
    return float(0.5 * np.mean(e**2))


def _mse_grad(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t. w: (1/N) * X^T (Xw - y)
    """
    N = y.shape[0]
    return (tx.T @ (tx @ w - y)) / N


# ------------------------- Gradient Descent (MSE) -----------------------------
def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
):
    """
    Full-batch gradient descent for the MSE objective.

    Returns
    -------
    w_final : np.ndarray
    loss_final : float
    """
    w = initial_w.astype(float).copy()
    for _ in range(max_iters):
        w -= gamma * _mse_grad(y, tx, w)
    #return w, _mse_loss(y, tx, w)
    return w.reshape(-1), float(_mse_loss(y, tx, w))



# -------------------- Stochastic Gradient Descent (MSE) -----------------------
def mean_squared_error_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
):
    """
    SGD for MSE with batch_size = 1 (as used in the public tests).

    We do exactly one SGD update per iteration to match the tests' behavior.
    """
    rng = np.random.default_rng()
    w = initial_w.astype(float).copy()
    N = y.shape[0]

    for _ in range(max_iters):
        i = int(rng.integers(0, N))  # pick one index
        y_b = y[i : i + 1]
        tx_b = tx[i : i + 1]
        # stochastic gradient on the single sample
        grad = (tx_b.T @ (tx_b @ w - y_b)) / 1.0
        w -= gamma * grad

    #return w, _mse_loss(y, tx, w)
    return w.reshape(-1), float(_mse_loss(y, tx, w))



# --------------------------- Closed-form (OLS) --------------------------------
def least_squares(y: np.ndarray, tx: np.ndarray):
    """
    Ordinary Least Squares using a robust solver.

    Solves: w = argmin (1/2)||y - Xw||^2
    We return the MSE with the (1/(2N)) factor.
    """
    X = tx
    y = y.astype(float)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)  # robust to rank deficiency
    #return w, _mse_loss(y, X, w)
    return w.reshape(-1), float(_mse_loss(y, X, w))


# ------------------------------ Ridge (L2) ------------------------------------
def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float):
    """
    Ridge regression using normal equations.

    Minimize: (1/(2N))||y - Xw||^2 + lambda * ||w||^2
    => (X^T X + 2N*lambda*I) w = X^T y
    The returned loss is the MSE part only (as in the project tests).
    """
    N, D = tx.shape
    X = tx
    y = y.astype(float)
    A = X.T @ X + (2.0 * N * lambda_) * np.eye(D)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    #return w, _mse_loss(y, X, w)
    return w.reshape(-1), float(_mse_loss(y, X, w))


# ------------------------- Logistic Regression (NLL) --------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid: clip z to avoid overflow in exp.
    """
    z = np.asarray(z)
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Average negative log-likelihood (binary cross-entropy):
    L = - mean( y log p + (1-y) log(1-p) ), with p = sigmoid(Xw).
    """
    p = _sigmoid(tx @ w)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    val = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()
    #return np.array(val, dtype=float)
    return float(val)


def _logistic_grad(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Gradient of the average NLL: (1/N) * X^T (p - y)
    """
    N = y.shape[0]
    p = _sigmoid(tx @ w)
    return (tx.T @ (p - y)) / N


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
):
    """
    Unregularized logistic regression trained with gradient descent.
    Returns (w_final, loss_final) where loss is the average NLL.
    """
    w = initial_w.astype(float).copy()
    for _ in range(max_iters):
        w -= gamma * _logistic_grad(y, tx, w)
    #return w, _logistic_loss(y, tx, w)
    return w.reshape(-1), float(_logistic_loss(y, tx, w))



# --------------- Regularized Logistic Regression (L2) -------------------------
def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
):
    """
    L2-regularized logistic regression (no bias regularization).

    Objective: NLL(w) + lambda * ||w||^2
    We do NOT regularize w[0].
    """
    w = initial_w.astype(float).copy()
    for _ in range(max_iters):
        g_nll = _logistic_grad(y, tx, w)
        g_reg = lambda_ * w
        if g_reg.size > 0:
            g_reg = g_reg.copy()
            g_reg[0] = 0.0
        w -= gamma * (g_nll + g_reg)
    return w, _logistic_loss(y, tx, w)

"""Synthetic processes and anomaly injection.

Processes (innovation variance SIGMA2 = 0.5):
    WN    x_t = e_t
    RW    x_t = x_{t-1} + e_t
    AR1   x_t = 0.7 x_{t-1} + e_t
    SEAS  x_t = 2 sin(2 pi t / 24) + y_t,  y_t = AR1 noise
"""
from __future__ import annotations

import numpy as np

T_CTX = 512
H = 32
SIGMA2 = 0.5
SIGMA = np.sqrt(SIGMA2)
PHI = 0.7
SEAS_PERIOD = 24
SEAS_AMP = 2.0
BURN_IN = 512

PROCESSES = ("WN", "RW", "AR1", "SEAS")
CASES = ("N0", "A0", "A1", "A2", "A3", "A4")

QUANTILE_LEVELS = np.array(
    [0.01, 0.05]
    + [round(0.10 + 0.05 * i, 4) for i in range(17)]   # 0.10 .. 0.90
    + [0.95, 0.99]
)
assert QUANTILE_LEVELS.size == 21, QUANTILE_LEVELS.size
assert np.all(np.diff(QUANTILE_LEVELS) > 0)


def _seasonal(t_index: np.ndarray) -> np.ndarray:
    return SEAS_AMP * np.sin(2.0 * np.pi * t_index / SEAS_PERIOD)


def generate_windows(process: str, n: int, rng: np.random.Generator,
                     var_scale: float = 1.0):
    """Return (context, future, t0) arrays.

    context : (n, T_CTX)   observed history
    future  : (n, H)       the true continuation
    t0      : (n,)         absolute time index of context[:, 0]

    ``var_scale`` multiplies the innovation variance of the FUTURE segment
    only (used by case A4); the context is always drawn from the nominal
    process so the forecaster sees a clean history.
    """
    total = BURN_IN + T_CTX + H
    eps = rng.normal(0.0, SIGMA, size=(n, total))
    if var_scale != 1.0:
        eps[:, BURN_IN + T_CTX:] *= np.sqrt(var_scale)

    if process == "WN":
        x = eps
    elif process == "RW":
        x = np.cumsum(eps, axis=1)
    elif process in ("AR1", "SEAS"):
        x = np.empty_like(eps)
        x[:, 0] = eps[:, 0] / np.sqrt(1.0 - PHI**2)  # stationary start
        for t in range(1, total):
            x[:, t] = PHI * x[:, t - 1] + eps[:, t]
    else:
        raise ValueError(f"unknown process {process!r}")

    # Random absolute time offsets so the seasonal phase varies across windows.
    t0 = rng.integers(0, SEAS_PERIOD, size=n).astype(np.int64)
    if process == "SEAS":
        t_abs = t0[:, None] + np.arange(total)[None, :]
        x = x + _seasonal(t_abs)

    ctx = x[:, BURN_IN:BURN_IN + T_CTX]
    fut = x[:, BURN_IN + T_CTX:]
    # absolute index of the first CONTEXT point
    return ctx, fut, t0 + BURN_IN


def oracle_moments(process: str, ctx: np.ndarray, t0: np.ndarray):
    """Analytic conditional mean/std of x_{T+h}, h = 1..H, given the context.

    Returns (mu, sd) each of shape (n, H).
    """
    n = ctx.shape[0]
    h = np.arange(1, H + 1)
    x_T = ctx[:, -1]
    # absolute time index of the last context point, and of the horizon points
    t_last = t0 + T_CTX - 1
    t_fut = t_last[:, None] + h[None, :]

    if process == "WN":
        mu = np.zeros((n, H))
        var = np.full((n, H), SIGMA2)
    elif process == "RW":
        mu = np.repeat(x_T[:, None], H, axis=1)
        var = np.repeat((SIGMA2 * h)[None, :], n, axis=0)
    elif process == "AR1":
        mu = x_T[:, None] * (PHI ** h)[None, :]
        var = np.repeat((SIGMA2 * (1 - PHI ** (2 * h)) / (1 - PHI**2))[None, :], n, 0)
    elif process == "SEAS":
        y_T = x_T - _seasonal(t_last)
        mu = _seasonal(t_fut) + y_T[:, None] * (PHI ** h)[None, :]
        var = np.repeat((SIGMA2 * (1 - PHI ** (2 * h)) / (1 - PHI**2))[None, :], n, 0)
    else:
        raise ValueError(f"unknown process {process!r}")
    return mu, np.sqrt(var)


def theoretical_psi(process: str) -> np.ndarray:
    """psi_h = d q50_h / d x_T for h = 0..H (psi_0 = 1 by definition)."""
    h = np.arange(0, H + 1)
    if process == "WN":
        psi = np.zeros(H + 1)
    elif process == "RW":
        psi = np.ones(H + 1)
    elif process in ("AR1", "SEAS"):
        psi = PHI ** h
    else:
        raise ValueError(f"unknown process {process!r}")
    psi[0] = 1.0
    return psi


def inject(case: str, future: np.ndarray, mu: np.ndarray, sd: np.ndarray,
           rng: np.random.Generator, future_hivar: np.ndarray | None = None):
    """Build the observed future path for an anomaly case.

    ``mu``/``sd`` are the ORACLE median/std, so the injected deviation is
    identical no matter which forecaster is later scored against it.
    """
    if case == "N0":
        return future.copy()
    if case == "A0":                       # median path: variance deficit only
        return mu.copy()
    if case == "A1":                       # constant 0.7-sigma offset path
        return mu + 0.7 * sd
    if case == "A2":                       # level shift, noise preserved
        return future + 0.7 * sd[:, [0]]
    if case == "A3":                       # single spike
        out = future.copy()
        idx = rng.integers(0, H, size=out.shape[0])
        out[np.arange(out.shape[0]), idx] += 4.0 * sd[:, 0]
        return out
    if case == "A4":                       # inflated innovation variance
        if future_hivar is None:
            raise ValueError("A4 needs the high-variance continuation")
        return future_hivar.copy()
    raise ValueError(f"unknown case {case!r}")


def continue_from(process: str, ctx: np.ndarray, t0: np.ndarray,
                  rng: np.random.Generator, var_scale: float = 1.0):
    """Draw a fresh H-step continuation of ``ctx`` from the true process.

    Lets N0 and A4 share the exact same context, so every case in a window
    is scored against one and the same forecast.
    """
    n = ctx.shape[0]
    sd = SIGMA * np.sqrt(var_scale)
    eps = rng.normal(0.0, sd, size=(n, H))
    x_T = ctx[:, -1]
    t_last = t0 + T_CTX - 1
    t_fut = t_last[:, None] + np.arange(1, H + 1)[None, :]

    if process == "WN":
        return eps
    if process == "RW":
        return x_T[:, None] + np.cumsum(eps, axis=1)
    if process in ("AR1", "SEAS"):
        y_T = x_T - _seasonal(t_last) if process == "SEAS" else x_T
        y = np.empty((n, H))
        prev = y_T
        for h in range(H):
            prev = PHI * prev + eps[:, h]
            y[:, h] = prev
        return y + _seasonal(t_fut) if process == "SEAS" else y
    raise ValueError(f"unknown process {process!r}")

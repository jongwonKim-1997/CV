"""PIT -> z transform, horizon-correlation estimation, and detection scores."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from data import H, QUANTILE_LEVELS

U_CLIP = (0.005, 0.995)
SHRINK = 0.05


# ---------------------------------------------------------------- PIT -> z

def pit(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear-interpolated PIT of x within the quantile vector q.

    q : (..., Q) quantile forecasts (sorted internally to kill crossing)
    x : (...)    observation
    returns u in [U_CLIP[0], U_CLIP[1]] with the same shape as x.
    """
    q = np.sort(q, axis=-1)
    lev = QUANTILE_LEVELS
    Q = q.shape[-1]

    xe = x[..., None]
    # number of quantiles strictly below x -> 0 .. Q
    idx = np.sum(q < xe, axis=-1)

    below = idx == 0
    above = idx == Q
    j = np.clip(idx - 1, 0, Q - 2)          # left bracket index

    ql = np.take_along_axis(q, j[..., None], -1)[..., 0]
    qr = np.take_along_axis(q, (j + 1)[..., None], -1)[..., 0]
    ll, lr = lev[j], lev[j + 1]

    width = qr - ql
    frac = np.where(width > 0, (x - ql) / np.where(width > 0, width, 1.0), 0.0)
    u = ll + np.clip(frac, 0.0, 1.0) * (lr - ll)

    u = np.where(below, U_CLIP[0], u)
    u = np.where(above, U_CLIP[1], u)
    return np.clip(u, *U_CLIP)


def to_z(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    return norm.ppf(pit(q, x))


def model_sigma(q: np.ndarray) -> np.ndarray:
    """(q75 - q25) / 1.349 as a robust scale estimate, shape (..., )."""
    q = np.sort(q, axis=-1)
    i25 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.25)))
    i75 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.75)))
    return (q[..., i75] - q[..., i25]) / 1.349


# ------------------------------------------------------- correlation matrix

def sigma_from_psi(psi: np.ndarray) -> np.ndarray:
    """Forecast-error covariance implied by an impulse response.

    Sigma_hk = sum_{j=0}^{min(h,k)-1} psi_j psi_{j+|h-k|},  h,k = 1..H.
    psi : (n, H+1) with psi[:, 0] == 1.  Returns (n, H, H).
    """
    psi = np.atleast_2d(psi)
    n = psi.shape[0]
    P = psi[:, :H]                                    # psi_0 .. psi_{H-1}
    # C[d][:, m-1] = sum_{j<m} P_j P_{j+d}
    cum = np.zeros((H, n, H))
    for d in range(H):
        prod = np.zeros((n, H))
        prod[:, : H - d] = P[:, : H - d] * P[:, d:]
        cum[d] = np.cumsum(prod, axis=1)

    h = np.arange(1, H + 1)
    d_mat = np.abs(h[:, None] - h[None, :])
    m_mat = np.minimum(h[:, None], h[None, :]) - 1
    return cum[d_mat, :, m_mat].transpose(2, 0, 1)     # (n, H, H)


def corr_from_sigma(sig: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    d = np.sqrt(np.clip(np.diagonal(sig, axis1=-2, axis2=-1), 1e-12, None))
    r = sig / (d[..., :, None] * d[..., None, :])
    r = 0.5 * (r + np.swapaxes(r, -1, -2))
    eye = np.eye(r.shape[-1])
    return (1.0 - shrink) * r + shrink * eye


def r_from_psi(psi: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    return corr_from_sigma(sigma_from_psi(psi), shrink)


def r_empirical(z: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    """Sample correlation of z over normal windows.  z : (n, H) -> (H, H)."""
    r = np.corrcoef(z, rowvar=False)
    r = 0.5 * (r + r.T)
    return (1.0 - shrink) * r + shrink * np.eye(H)


def _chol_solve(R: np.ndarray, B: np.ndarray, max_tries: int = 6):
    """Solve R X = B via Cholesky, inflating the ridge if ill-conditioned.

    R : (n, H, H) or (H, H);  B : (n, H, k).  Returns (X, lam_used).
    """
    from scipy.linalg import cho_factor, cho_solve
    R = np.asarray(R, dtype=float)
    single = R.ndim == 2
    Rb = R[None] if single else R
    eye = np.eye(Rb.shape[-1])
    lam = 0.0
    for k in range(max_tries):
        try:
            out = np.empty_like(B)
            Rl = Rb * (1 - lam) + lam * eye if lam else Rb
            for i in range(Rb.shape[0]):
                c = cho_factor(Rl[i] if not single else Rl[0], lower=True)
                out[i] = cho_solve(c, B[i])
            return out, lam
        except np.linalg.LinAlgError:
            lam = 0.05 * (2**k)
        except ValueError:
            lam = 0.05 * (2**k)
    raise np.linalg.LinAlgError("R stayed indefinite after ridge inflation")


# ------------------------------------------------------------------ scores

def causal_ema_psi(psi: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """EMA of psi along the window axis, using only the current and past windows.

    Pooling psi over a whole (model, process, seed) block needs the process
    label and peeks at future windows, so it cannot be used in deployment.
    Smoothing along the time axis of one stream can: at window i it has seen
    windows 0..i only.  lam = 0.05 is a half-life of about 13 windows.
    """
    out = np.empty_like(psi)
    acc = psi[0].copy()
    for i in range(psi.shape[0]):
        acc = acc if i == 0 else (1.0 - lam) * acc + lam * psi[i]
        out[i] = acc
    return out


def s_max_raw(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """max_h |x_h - q50_h| / sigma_h, with no PIT and therefore no clipping.

    S_max is capped at Phi^-1(0.995) = 2.576 by the 21-point quantile grid, so
    a large spike saturates and cannot be told from a moderate one.  This
    baseline reads the deviation directly off the model's own IQR scale.
    """
    qs = np.sort(q, axis=-1)
    i50 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.50)))
    sd = np.maximum(model_sigma(q), 1e-9)
    return np.max(np.abs(x - qs[..., i50]) / sd, axis=1)


def _gls(R: np.ndarray, z: np.ndarray, d: np.ndarray):
    """Directed GLS level test along direction d.

    T = |d' R^-1 z| / sqrt(d' R^-1 d).  Returns (T, d'R^-1 d, z'R^-1 z).
    The direction must match the SHAPE of the anomaly being looked for:
    d = 1 tests a constant deviation in z, d = 1/sigma_h a constant deviation
    in x.  They agree only when sigma_h is flat across the horizon.
    """
    n, H_ = z.shape
    if np.ndim(R) == 2:
        R = np.broadcast_to(R, (n, H_, H_))
    rhs = np.stack([d, z], axis=-1)
    sol, _ = _chol_solve(np.ascontiguousarray(R), rhs)
    Rinv_d, Rinv_z = sol[..., 0], sol[..., 1]
    quad = np.einsum("nh,nh->n", d, Rinv_d)
    num = np.einsum("nh,nh->n", d, Rinv_z)
    maha = np.einsum("nh,nh->n", z, Rinv_z)
    return np.abs(num) / np.sqrt(np.clip(quad, 1e-12, None)), quad, maha


def all_scores(z: np.ndarray, q: np.ndarray, x: np.ndarray, Rs: dict,
               band_from_z: bool = False) -> dict:
    """z:(n,H)  q:(n,H,Q)  x:(n,H).

    ``Rs`` maps a name to a correlation matrix, (n,H,H) or (H,H).  "pert" must
    be present; "pool", "emp", "ema" and "hybrid" are scored when supplied.
    """
    n = z.shape[0]
    ones = np.ones((n, H))

    out = {}
    for name, R in Rs.items():
        t, quad, maha = _gls(R, z, ones)
        out[f"S_gls_{name}"] = t
        out[f"n_eff_{name}"] = quad
        if name == "pert":
            out["S_maha"] = maha

    # Direction matched to a constant deviation in X space: d_h = 1/sigma_h.
    # Scaling by sigma_1 makes n_eff_b comparable to the d = 1 version -- for
    # an x-space offset of c the statistic is (c/sigma_1)*sqrt(n_eff_b), the
    # same form as (0.7)*sqrt(n_eff) for the z-space case.
    sd_h = np.maximum(model_sigma(q), 1e-9)
    b = sd_h[:, [0]] / sd_h
    t_b, quad_b, _ = _gls(Rs["pert"], z, b)
    out["S_gls_b"] = t_b
    out["n_eff_b"] = quad_b

    if band_from_z:
        outside = np.abs(z) > norm.ppf(0.90)
    else:
        qs = np.sort(q, axis=-1)
        i10 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.10)))
        i90 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.90)))
        outside = (x < qs[..., i10]) | (x > qs[..., i90])

    out.update({
        "S_max": np.max(np.abs(z), axis=1),
        "S_max_raw": s_max_raw(q, x),
        "S_band": outside.mean(axis=1),
        "S_mean": np.abs(np.sqrt(H) * z.mean(axis=1)),
        "S_chi2": np.abs((z**2).sum(axis=1) - H) / np.sqrt(2 * H),
    })
    out["n_eff"] = out["n_eff_pert"]
    return out

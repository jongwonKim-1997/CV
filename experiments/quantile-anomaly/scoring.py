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

def all_scores(z: np.ndarray, q: np.ndarray, x: np.ndarray,
               R_pert: np.ndarray, R_emp: np.ndarray,
               band_from_z: bool = False, R_pool: np.ndarray | None = None) -> dict:
    """z:(n,H)  q:(n,H,Q)  x:(n,H)  R_pert:(n,H,H) or (H,H)  R_emp:(H,H).

    ``band_from_z`` computes S_band as |z| > Phi^-1(0.9) instead of comparing x
    against the raw q10/q90; needed when z has been recalibrated, so the band
    moves with the correction.
    """
    n = z.shape[0]
    ones = np.ones((n, H, 1))
    zb = z[:, :, None]

    def gls(R):
        Rb = np.broadcast_to(np.atleast_3d(R).reshape(-1, H, H), (n, H, H)) \
            if np.ndim(R) == 2 else R
        rhs = np.concatenate([ones, zb], axis=-1)          # (n, H, 2)
        sol, _ = _chol_solve(np.ascontiguousarray(Rb), rhs)
        Rinv1, Rinvz = sol[..., 0], sol[..., 1]
        n_eff = np.einsum("nh,nh->n", np.ones((n, H)), Rinv1)
        num = np.einsum("nh,nh->n", np.ones((n, H)), Rinvz)
        maha = np.einsum("nh,nh->n", z, Rinvz)
        return np.abs(num) / np.sqrt(np.clip(n_eff, 1e-9, None)), n_eff, maha

    s_gls_pert, n_eff_pert, maha = gls(R_pert)
    s_gls_emp, n_eff_emp, _ = gls(R_emp)
    # Pooled variant: one R built from the psi averaged over the windows of
    # this (model, process, seed).  1' R^-1 1 is very sensitive to a handful
    # of small negative off-diagonals, so a per-window psi estimated on a
    # noisy learned model makes n_eff explode; pooling first removes that.
    if R_pool is None:
        R_pool = R_pert
    s_gls_pool, n_eff_pool, _ = gls(R_pool)

    if band_from_z:
        outside = np.abs(z) > norm.ppf(0.90)
    else:
        qs = np.sort(q, axis=-1)
        i10 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.10)))
        i90 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.90)))
        outside = (x < qs[..., i10]) | (x > qs[..., i90])

    return {
        "S_max": np.max(np.abs(z), axis=1),
        "S_band": outside.mean(axis=1),
        "S_mean": np.abs(np.sqrt(H) * z.mean(axis=1)),
        "S_gls_pert": s_gls_pert,
        "S_gls_pool": s_gls_pool,
        "S_gls_emp": s_gls_emp,
        "S_chi2": np.abs((z**2).sum(axis=1) - H) / np.sqrt(2 * H),
        "S_maha": maha,
        "n_eff": n_eff_pert,
        "n_eff_pool": n_eff_pool,
        "n_eff_emp": n_eff_emp,
    }

"""Unit tests: the detector must reproduce the analytic targets of the spec.

    ORACLE + theoretical psi
      WN  / A1 -> S_gls ~ 0.7*sqrt(H)     (level shift is fully visible)
      RW  / A1 -> S_gls ~ 0.7             (only one effective observation)
      any / A0 -> S_gls ~ 0               (no level deviation at all)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import data, scoring
from data import H

TOL = 0.06

# n_eff of a random walk, in closed form.  z_h = (x_{T+h} - x_T)/sqrt(0.5 h)
# has corr(z_h, z_k) = min(h,k)/sqrt(hk), so 1' R^-1 1 = sum_h (sqrt h -
# sqrt(h-1))^2 = 1.8641 for H = 32 -- NOT 1, which would need perfectly
# correlated z.  The spec's "RW ~ 1" is the value for unnormalised errors.
RW_NEFF = float((np.diff(np.sqrt(np.arange(0, H + 1))) ** 2).sum())


def _q_from_moments(mu, sd):
    from scipy.stats import norm
    return mu[..., None] + sd[..., None] * norm.ppf(data.QUANTILE_LEVELS)


def _scores(z, q, x, R):
    return scoring.all_scores(z, q, x, {"pert": R, "emp": R})


def _setup(process, case, n=400, seed=0, shrink=scoring.SHRINK):
    rng = np.random.default_rng(seed)
    ctx, fut, t0 = data.generate_windows(process, n, rng)
    mu, sd = data.oracle_moments(process, ctx, t0)
    x = data.inject(case, fut, mu, sd, rng)
    q = _q_from_moments(mu, sd)
    z = scoring.to_z(q, x)
    psi = np.broadcast_to(data.theoretical_psi(process), (n, H + 1))
    R = scoring.r_from_psi(psi, shrink=shrink)
    return z, q, x, R


def test_sigma_from_psi_matches_bruteforce():
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(2, H + 1)); psi[:, 0] = 1.0
    sig = scoring.sigma_from_psi(psi)
    for n in range(2):
        for h in range(1, H + 1):
            for k in range(1, H + 1):
                ref = sum(psi[n, j] * psi[n, j + abs(h - k)]
                          for j in range(min(h, k)))
                assert abs(sig[n, h - 1, k - 1] - ref) < 1e-10
    print("  sigma_from_psi matches the brute-force definition")


def test_n_eff():
    """n_eff at lambda = 0 must hit the analytic values (WN = H, RW = 1)."""
    for lam, tag in [(0.0, "lambda=0   "), (scoring.SHRINK, "lambda=0.05")]:
        got = {}
        for p in data.PROCESSES:
            psi = np.broadcast_to(data.theoretical_psi(p), (1, H + 1))
            R = scoring.r_from_psi(psi, shrink=lam)
            q0 = np.zeros((1, H, 21)) + np.linspace(-1, 1, 21)
            got[p] = _scores(np.zeros((1, H)), q0,
                             np.zeros((1, H)), R)["n_eff"][0]
        print(f"  {tag}  n_eff  WN={got['WN']:6.2f}  RW={got['RW']:5.2f}  "
              f"AR1={got['AR1']:5.2f}  SEAS={got['SEAS']:5.2f}")
        if lam == 0.0:
            # WN: R = I exactly            -> n_eff = H
            # RW: R_hk = min(h,k)/sqrt(hk) -> n_eff = sum (sqrt h - sqrt(h-1))^2
            v = np.sqrt(np.arange(0, H + 1))
            rw_exact = float((np.diff(v) ** 2).sum())
            print(f"               RW analytic n_eff = {rw_exact:.6f} "
                  f"(the spec's 'RW ~ 1' is not the right constant)")
            assert abs(got["WN"] - H) < 1e-6, got["WN"]
            assert abs(got["RW"] - rw_exact) < 1e-6, got["RW"]
        assert 1.0 < got["AR1"] < got["WN"], got["AR1"]


def test_numerical_psi_matches_theory():
    for p in data.PROCESSES:
        rng = np.random.default_rng(1)
        ctx, _, t0 = data.generate_windows(p, 64, rng)
        d = 0.05 * np.sqrt(data.SIGMA2)
        cp, cm = ctx.copy(), ctx.copy()
        cp[:, -1] += d; cm[:, -1] -= d
        mp, _ = data.oracle_moments(p, cp, t0)
        mm, _ = data.oracle_moments(p, cm, t0)
        psi_num = (mp - mm) / (2 * d)
        psi_th = data.theoretical_psi(p)[1:]
        err = np.abs(psi_num - psi_th).max()
        print(f"  psi[{p:5s}] max|numeric - theory| = {err:.2e}")
        assert err < 1e-8, (p, err)


def test_spec_targets_unshrunk():
    """The spec targets (0.7*sqrt(H), 0.7, 0) hold for the RAW R, lambda = 0."""
    print(f"  targets: WN/A1 = 0.7*sqrt(H) = {0.7*np.sqrt(H):.3f},"
          f"  RW/A1 = 0.7*sqrt({RW_NEFF:.3f}) = {0.7*np.sqrt(RW_NEFF):.3f},"
          f"  A0 = 0.000")
    for p, case, target in [("WN", "A1", 0.7 * np.sqrt(H)),
                            ("RW", "A1", 0.7 * np.sqrt(RW_NEFF)),
                            ("AR1", "A0", 0.0),
                            ("WN", "A0", 0.0),
                            ("RW", "A0", 0.0)]:
        z, q, x, R = _setup(p, case, shrink=0.0)
        s = _scores(z, q, x, R)["S_gls_pert"]
        print(f"  {p:5s}/{case}: S_gls = {s.mean():.4f} +- {s.std():.4f}"
              f"   target {target:.4f}")
        assert abs(s.mean() - target) < TOL, (p, case, s.mean(), target)


def test_shrinkage_effect():
    """S_gls = 0.7*sqrt(n_eff) must hold for whatever lambda is in force.

    lambda = 0.05 moves RW's n_eff 1.864 -> 1.935 (its R is near-singular so
    a 5% ridge is not a negligible perturbation) and leaves WN's at H = 32.
    """
    z, q, x, R = _setup("RW", "A1", shrink=scoring.SHRINK)
    sc = _scores(z, q, x, R)
    s, ne = sc["S_gls_pert"].mean(), sc["n_eff"].mean()
    print(f"  RW/A1 at lambda=0.05: S_gls = {s:.4f}, n_eff = {ne:.4f}, "
          f"0.7*sqrt(n_eff) = {0.7*np.sqrt(ne):.4f}")
    assert abs(s - 0.7 * np.sqrt(ne)) < 0.02, (s, ne)
    assert s < 1.2, s          # still far below the WN value of ~3.95


def test_a0_is_caught_by_chi2_only():
    z, q, x, R = _setup("WN", "A0")
    sc = _scores(z, q, x, R)
    print(f"  WN/A0: S_gls={sc['S_gls_pert'].mean():.3f} (low)  "
          f"S_chi2={sc['S_chi2'].mean():.3f} (high, target {H/np.sqrt(2*H):.3f})")
    assert sc["S_gls_pert"].mean() < 0.1
    assert abs(sc["S_chi2"].mean() - H / np.sqrt(2 * H)) < 0.05


def test_pit_roundtrip():
    from scipy.stats import norm
    rng = np.random.default_rng(5)
    mu = rng.normal(size=(200, H)); sd = np.exp(rng.normal(size=(200, H)) * .3)
    q = _q_from_moments(mu, sd)
    x = mu + sd * 1.0                       # exactly +1 sigma everywhere
    z = scoring.to_z(q, x)
    print(f"  PIT round-trip at +1 sigma: z = {z.mean():.4f} "
          f"(interp on a 21-point grid, target 1.0)")
    assert abs(z.mean() - 1.0) < 0.02




def test_direction_must_match_the_anomaly():
    """d = 1 and d = 1/sigma_h test DIFFERENT deviations; only WN merges them.

    A1  is a constant offset in z  (x_h = q50_h + 0.7 sigma_h) -> matched by d = 1
    A1p is a constant offset in x  (x_h = q50_h + 0.7 sigma_1) -> matched by d = 1/sigma_h

    For a random walk the matched statistic on A1p is exactly 0.7, because
    n_eff_b = sigma_1^2 * b'R^-1 b = 1: with Sigma = 0.5 min(h,k) and
    b_h = sigma_1/sigma_h, the vector D b is flat, so b'R^-1 b = 1'Sigma^-1 1
    scaled, and the Brownian quadratic form of a flat vector keeps only its
    first step.  That 1 is the constant the original spec was reaching for.
    """
    print(f"  A1  target (d=1,   WN) = 0.7*sqrt(H)  = {0.7*np.sqrt(H):.4f}")
    print(f"  A1p target (d=1/s, RW) = 0.7*sqrt(1)  = 0.7000")
    for p, case, key, target in [
            ("WN", "A1", "S_gls_pert", 0.7 * np.sqrt(H)),
            ("WN", "A1", "S_gls_b", 0.7 * np.sqrt(H)),      # sigma_h flat -> same
            ("WN", "A1p", "S_gls_b", 0.7 * np.sqrt(H)),     # and A1 == A1p on WN
            ("RW", "A1p", "S_gls_b", 0.7),
            ("AR1", "A1p", "S_gls_b", None),
    ]:
        z, q, x, R = _setup(p, case, shrink=0.0)
        sc = _scores(z, q, x, R)
        v, neb = sc[key].mean(), sc["n_eff_b"].mean()
        tag = "" if target is None else f"   target {target:.4f}"
        print(f"  {p:5s}/{case:3s} {key:11s} = {v:.4f}  n_eff_b = {neb:7.4f}{tag}")
        if target is not None:
            assert abs(v - target) < TOL, (p, case, key, v, target)

    # the SAME anomaly read with the WRONG direction loses power
    z, q, x, R = _setup("RW", "A1p", shrink=0.0)
    sc = _scores(z, q, x, R)
    print(f"  RW/A1p with the mismatched d=1: {sc['S_gls_pert'].mean():.4f} "
          f"vs matched {sc['S_gls_b'].mean():.4f}")


def test_s_max_raw_is_not_clipped():
    """S_max saturates at 2.576 on a 21-point grid; S_max_raw does not."""
    z, q, x, R = _setup("WN", "A3")
    sc = _scores(z, q, x, R)
    sat = (sc["S_max"] > 2.57).mean()
    print(f"  A3 spike: S_max mean={sc['S_max'].mean():.3f} "
          f"(saturated in {sat:.0%} of windows, hard cap 2.576)")
    print(f"            S_max_raw mean={sc['S_max_raw'].mean():.3f} "
          f"(injected spike is 4 sigma)")
    assert sc["S_max"].max() <= 2.5763
    assert sc["S_max_raw"].mean() > 3.0


def test_causal_ema_uses_no_future():
    rng = np.random.default_rng(11)
    psi = rng.normal(size=(50, H + 1)); psi[:, 0] = 1.0
    e = scoring.causal_ema_psi(psi, lam=0.05)
    assert np.allclose(e[0], psi[0])
    psi2 = psi.copy(); psi2[30:] = 999.0          # perturb only the future
    e2 = scoring.causal_ema_psi(psi2, lam=0.05)
    assert np.allclose(e[:30], e2[:30]), "EMA leaked future windows"
    print("  EMA at window i depends only on windows 0..i")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nALL TESTS PASSED")

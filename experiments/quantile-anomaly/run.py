"""Full experiment driver: forecasts -> z -> R -> scores -> csv."""
from __future__ import annotations

import argparse, json, os, time, zlib
import numpy as np
import pandas as pd

import data, models, scoring
from data import H, PROCESSES, CASES, QUANTILE_LEVELS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
DIAG_H = (1, 8, 16, 32)
HYBRID_ALPHA = 0.5      # weight on R_emp in the hybrid estimator


def stable_seed(*parts) -> int:
    """A seed that survives a restart.

    Python's str hash is salted per process (PYTHONHASHSEED), so hash()-derived
    seeds silently regenerate DIFFERENT windows on a rerun while the npz cache
    still holds forecasts made for the old ones.  crc32 is stable.
    """
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF


def windows_for(process, seed, n, n_emp):
    """Deterministic given (process, seed) so every model sees identical data."""
    rng = np.random.default_rng(stable_seed(process, seed, "ctx"))
    ctx, _, t0 = data.generate_windows(process, n, rng)
    mu, sd = data.oracle_moments(process, ctx, t0)
    fut = data.continue_from(process, ctx, t0, rng, 1.0)
    fut_hi = data.continue_from(process, ctx, t0, rng, 2.5)
    ectx, _, et0 = data.generate_windows(process, n_emp, rng)
    efut = data.continue_from(process, ectx, et0, rng, 1.0)
    return dict(ctx=ctx, t0=t0, mu=mu, sd=sd, fut=fut, fut_hi=fut_hi,
                ectx=ectx, et0=et0, efut=efut, rng=rng)


def forecast(model, process, ctx, t0, tag, use_cache=True):
    path = os.path.join(CACHE, f"{tag}.npz")
    if use_cache and os.path.exists(path):
        d = np.load(path)
        return d["q"], d["psi"], d["delta"]
    psi, q, delta = models.psi_by_perturbation(model, process, ctx, t0)
    np.savez_compressed(path, q=q, psi=psi, delta=delta)
    return q, psi, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-emp", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000, help="MLPQR train steps")
    ap.add_argument("--pool", type=int, default=40000, help="MLPQR train windows")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--chronos-path", default=None,
                    help="local Chronos-2 checkpoint directory (the one with "
                         "config.json). Use when huggingface.co is unreachable; "
                         "pair it with HF_HUB_OFFLINE=1.")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    zoo = [models.OracleModel()]
    prov = {"ORACLE": "analytic"}

    print("resolving a pretrained forecaster ...")
    ch, ch_name = models.try_load_chronos(device, args.chronos_path)
    if ch is not None:
        zoo.append(ch); prov[ch_name] = "pretrained (huggingface)"
    else:
        print("  -> no hub access; training the MLPQR surrogate instead")
        mp = os.path.join(CACHE, f"mlpqr_{args.steps}_{args.pool}.pt")
        m = models.MLPQR(device="cpu", seed=0)
        if os.path.exists(mp) and not args.no_cache:
            m.load(mp); print("  loaded cached MLPQR weights")
        else:
            t = time.time()
            pc, pf = models.training_pool(args.pool, np.random.default_rng(1234))
            print(f"  training pool {pc.shape} in {time.time()-t:.1f}s")
            m.fit(pc, pf, steps=args.steps)
            m.save(mp)
        zoo.append(m); prov["MLPQR"] = "trained here, zero-shot on the 4 targets"

    rows, diag, pit_store, psi_store = [], [], {}, {}

    for model in zoo:
        for seed in args.seeds:
            for proc in PROCESSES:
                t = time.time()
                W = windows_for(proc, seed, args.n, args.n_emp)
                tag = f"{model.name}_{proc}_s{seed}"

                q, psi, _ = forecast(model, proc, W["ctx"], W["t0"], tag,
                                     not args.no_cache)
                R_pert = scoring.r_from_psi(psi)
                # pooled: the whole block's psi averaged.  Diagnostic only --
                # it needs the process label and peeks at future windows.
                R_pool = scoring.r_from_psi(psi.mean(0, keepdims=True))[0]
                # causal EMA along the window axis: deployable on one stream.
                R_ema = scoring.r_from_psi(scoring.causal_ema_psi(psi))

                # ---- empirical R from a disjoint set of normal windows
                qe = model.predict(proc, W["ectx"], W["et0"])
                ze = scoring.to_z(qe, W["efut"])
                R_emp = scoring.r_empirical(ze)

                # Per-horizon recalibration learned on those same normal
                # windows: z -> (z - mean_h) / std_h.  A perfectly calibrated
                # model leaves this a no-op; it separates "the forecaster's
                # intervals are the wrong width" from "the detector is wrong".
                cal_m, cal_s = ze.mean(0), np.maximum(ze.std(0), 1e-6)
                ze_cal = (ze - cal_m) / cal_s
                R_emp_cal = scoring.r_empirical(ze_cal)
                # hybrid: shrink the noisy per-window R_pert toward the
                # empirical R of this series.  Both are correlation matrices,
                # so the convex combination keeps a unit diagonal.  R_emp needs
                # only normal history, so this is deployable too.
                def _mix(Re):
                    return (1.0 - HYBRID_ALPHA) * R_pert + HYBRID_ALPHA * Re

                variants = [(model.name, False, None, R_emp, _mix(R_emp))]
                if not getattr(model, "analytic_psi", False):
                    variants.append((model.name + "_CAL", True,
                                     (cal_m, cal_s), R_emp_cal, _mix(R_emp_cal)))

                psi_store[(model.name, proc, seed)] = psi.mean(0)
                Rp_mean = R_pert.mean(0)
                fro = float(np.linalg.norm(Rp_mean - R_emp, "fro"))

                for case in CASES:
                    crng = np.random.default_rng(stable_seed(proc, seed, case))
                    x = data.inject(case, W["fut"], W["mu"], W["sd"], crng,
                                    future_hivar=W["fut_hi"])
                    z_raw = scoring.to_z(q, x)
                    for mname, is_cal, cal, Re, Rhyb in variants:
                        z = (z_raw - cal[0]) / cal[1] if is_cal else z_raw
                        sc = scoring.all_scores(
                            z, q, x,
                            {"pert": R_pert, "pool": R_pool, "ema": R_ema,
                             "emp": Re, "hybrid": Rhyb},
                            band_from_z=is_cal)
                        for i in range(x.shape[0]):
                            rows.append(dict(
                                process=proc, model=mname, case=case,
                                seed=seed, window_id=i,
                                **{k: float(v[i]) for k, v in sc.items()}))
                    z = z_raw
                    sc = scoring.all_scores(
                        z_raw, q, x,
                        {"pert": R_pert, "pool": R_pool, "ema": R_ema,
                         "emp": R_emp, "hybrid": _mix(R_emp)})
                    if case == "N0":
                        pit_store[(model.name, proc, seed)] = \
                            scoring.pit(q, x)[:, [h - 1 for h in DIAG_H]]
                        for h in range(H):
                            diag.append(dict(
                                process=proc, model=model.name, seed=seed,
                                horizon=h + 1, z_mean=float(z[:, h].mean()),
                                z_var=float(z[:, h].var()),
                                psi=float(psi[:, h + 1].mean()),
                                psi_theory=float(
                                    data.theoretical_psi(proc)[h + 1]),
                                n_eff_pert=float(sc["n_eff"].mean()),
                                n_eff_pert_med=float(np.median(sc["n_eff"])),
                                n_eff_pool=float(sc["n_eff_pool"].mean()),
                                n_eff_ema=float(sc["n_eff_ema"].mean()),
                                n_eff_hybrid=float(sc["n_eff_hybrid"].mean()),
                                n_eff_b=float(sc["n_eff_b"].mean()),
                                n_eff_emp=float(sc["n_eff_emp"].mean()),
                                R_frobenius=fro))
                print(f"  {model.name:8s} {proc:5s} seed={seed}  "
                      f"n_eff={sc['n_eff'].mean():6.2f}  "
                      f"||Rp-Re||_F={fro:5.2f}  ({time.time()-t:.1f}s)")

    pd.DataFrame(rows).to_csv(os.path.join(RES, "scores.csv"), index=False)
    pd.DataFrame(diag).to_csv(os.path.join(RES, "diagnostics.csv"), index=False)
    np.savez_compressed(
        os.path.join(RES, "figdata.npz"),
        pit_keys=np.array([f"{a}|{b}|{c}" for a, b, c in pit_store]),
        pit_vals=np.stack(list(pit_store.values())),
        psi_keys=np.array([f"{a}|{b}|{c}" for a, b, c in psi_store]),
        psi_vals=np.stack(list(psi_store.values())),
        diag_h=np.array(DIAG_H))
    with open(os.path.join(RES, "provenance.json"), "w") as f:
        json.dump(dict(models=prov, n=args.n, n_emp=args.n_emp,
                       seeds=args.seeds, device=device, H=H,
                       shrink=scoring.SHRINK, hybrid_alpha=HYBRID_ALPHA,
                       quantile_levels=QUANTILE_LEVELS.tolist()), f, indent=2)
    print(f"\nwrote {len(rows)} score rows -> results/scores.csv")


if __name__ == "__main__":
    main()

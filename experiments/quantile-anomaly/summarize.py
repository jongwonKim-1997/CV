"""Pull the exact numbers the report's hypothesis verdicts need."""
import numpy as np, pandas as pd, os
HERE=os.path.dirname(os.path.abspath(__file__)); RES=f"{HERE}/results"
pd.set_option("display.width", 200, "display.max_columns", 40)

agg=pd.read_csv(f"{RES}/metrics.csv"); df=pd.read_csv(f"{RES}/scores.csv")
diag=pd.read_csv(f"{RES}/diagnostics.csv")
MODELS=list(dict.fromkeys(df.model))

def block(t): print("\n"+"="*78+f"\n{t}\n"+"="*78)

block("1. AUROC by score  (rows = case, cols = score) -- per model/process")
for m in MODELS:
    for p in ["WN","RW","AR1","SEAS"]:
        t=(agg[(agg.model==m)&(agg.process==p)]
           .pivot(index="case",columns="score",values="auroc")
           [["S_max","S_band","S_mean","S_gls_pert","S_gls_emp","S_chi2","S_maha"]])
        print(f"\n-- {m} / {p} --"); print(t.round(3).to_string())

block("2. TPR @ FPR 5%  (same layout)")
for m in MODELS:
    for p in ["WN","RW","AR1","SEAS"]:
        t=(agg[(agg.model==m)&(agg.process==p)]
           .pivot(index="case",columns="score",values="tpr5")
           [["S_max","S_band","S_mean","S_gls_pert","S_gls_emp","S_chi2","S_maha"]])
        print(f"\n-- {m} / {p} --"); print(t.round(3).to_string())

block("3. H1/H2: WN, case A1 and A2 -- S_max vs S_gls_pert")
for m in MODELS:
    for c in ["A1","A2"]:
        s=agg[(agg.model==m)&(agg.process=="WN")&(agg.case==c)].set_index("score")
        a,b=s.loc["S_max"],s.loc["S_gls_pert"]
        print(f"  {m}/WN/{c}: AUROC S_max={a.auroc:.3f}(sd {a.auroc_sd:.3f})  "
              f"S_gls_pert={b.auroc:.3f}(sd {b.auroc_sd:.3f})  "
              f"delta={b.auroc-a.auroc:+.3f} | TPR5 {a.tpr5:.3f} -> {b.tpr5:.3f}")

block("4. H3: RW, case A1 -- S_mean false positive vs S_gls_pert")
for m in MODELS:
    s=agg[(agg.model==m)&(agg.process=="RW")&(agg.case=="A1")].set_index("score")
    g=df[(df.model==m)&(df.process=="RW")&(df.case=="A1")]
    n0=df[(df.model==m)&(df.process=="RW")&(df.case=="N0")]
    print(f"  {m}/RW/A1: S_mean AUROC={s.loc['S_mean'].auroc:.3f} "
          f"TPR5={s.loc['S_mean'].tpr5:.3f} mean={g.S_mean.mean():.3f}   ||   "
          f"S_gls_pert AUROC={s.loc['S_gls_pert'].auroc:.3f} "
          f"TPR5={s.loc['S_gls_pert'].tpr5:.3f} mean={g.S_gls_pert.mean():.3f}"
          f"  (n_eff={g.n_eff.mean():.2f}, 0.7*sqrt(n_eff)="
          f"{0.7*np.sqrt(g.n_eff.mean()):.3f})")

block("5. H4: case A0 -- S_gls_pert must be low, S_chi2 high")
for m in MODELS:
    for p in ["WN","RW","AR1","SEAS"]:
        g=df[(df.model==m)&(df.process==p)&(df.case=="A0")]
        s=agg[(agg.model==m)&(agg.process==p)&(agg.case=="A0")].set_index("score")
        print(f"  {m}/{p}/A0: S_gls mean={g.S_gls_pert.mean():.3f} "
              f"AUROC={s.loc['S_gls_pert'].auroc:.3f}  |  "
              f"S_chi2 mean={g.S_chi2.mean():.2f} AUROC={s.loc['S_chi2'].auroc:.3f}")

block("6. H5: n_eff and R distance; psi fidelity")
d=(diag.groupby(["model","process"]).agg(
    n_eff_pert=("n_eff_pert","mean"), n_eff_emp=("n_eff_emp","mean"),
    R_fro=("R_frobenius","mean")).reset_index())
import scoring, data
th={p: float(np.ones(32)@np.linalg.solve(
    scoring.r_from_psi(np.broadcast_to(data.theoretical_psi(p),(1,33)))[0],np.ones(32)))
    for p in ["WN","RW","AR1","SEAS"]}
d["n_eff_theory"]=d.process.map(th)
print(d.round(3).to_string(index=False))
print("\npsi max abs error vs theory (mean over seeds):")
pe=(diag.assign(err=(diag.psi-diag.psi_theory).abs())
        .groupby(["model","process"]).err.agg(["max","mean"]).reset_index())
print(pe.round(4).to_string(index=False))

block("7. H6: A3 spike -- point-wise vs level scores")
for m in MODELS:
    for p in ["WN","RW","AR1","SEAS"]:
        s=agg[(agg.model==m)&(agg.process==p)&(agg.case=="A3")].set_index("score")
        print(f"  {m}/{p}/A3: S_max={s.loc['S_max'].auroc:.3f} "
              f"S_band={s.loc['S_band'].auroc:.3f} "
              f"S_mean={s.loc['S_mean'].auroc:.3f} "
              f"S_gls_pert={s.loc['S_gls_pert'].auroc:.3f} "
              f"S_chi2={s.loc['S_chi2'].auroc:.3f}")

block("8. Calibration of N0: z mean / var by horizon")
for m in MODELS:
    t=(diag[diag.model==m][diag.horizon.isin([1,8,16,32])]
       .groupby(["process","horizon"]).agg(z_mean=("z_mean","mean"),
                                           z_var=("z_var","mean")).reset_index())
    print(f"\n-- {m} --"); print(t.round(3).to_string(index=False))

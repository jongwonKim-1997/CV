# Quantile-forecast anomaly detection: level shifts vs point-wise error

Does a horizon-correlation-aware level test (`S_gls`) catch a sustained level
deviation that point-wise scores (`S_max`, `S_band`) miss, on top of a quantile
forecaster?

## Layout

| file | contents |
|---|---|
| `data.py` | the four synthetic processes, analytic conditional moments, anomaly injection |
| `models.py` | ORACLE, Chronos-2 / Chronos-Bolt wrappers, the MLPQR surrogate, impulse response by input perturbation |
| `scoring.py` | quantile -> PIT -> z, R estimation (`R_pert`, `R_emp`), the seven detection scores |
| `evaluate.py` | AUROC / AUPRC / TPR@FPR5%, the four diagnostic figures |
| `run.py` | driver: forecasts -> z -> R -> scores -> `results/*.csv` |
| `summarize.py` | prints the tables the report's hypothesis verdicts quote |
| `test_core.py` | unit tests pinning the detector to its analytic targets |
| `REPORT.md` | hypothesis-by-hypothesis verdicts with the numbers |

## Running

```bash
python -m venv .venv && .venv/bin/pip install numpy scipy pandas matplotlib scikit-learn torch
.venv/bin/pip install chronos-forecasting        # needs huggingface.co reachable

cd experiments/quantile-anomaly
python test_core.py                              # must print ALL TESTS PASSED
python run.py --n 300 --n-emp 500 --seeds 0 1 2  # forecasts are cached in cache/
python evaluate.py
python summarize.py
```

`run.py` tries `amazon/chronos-2` first, falls back to `amazon/chronos-bolt-small`,
and if neither hub download succeeds it trains the `MLPQR` surrogate locally so
that a second, non-analytic model is always present. Forecasts are cached per
`(model, process, seed)` under `cache/`; delete that directory to force a refit.
On a GPU box the Chronos path is picked up automatically (`torch.cuda.is_available()`).

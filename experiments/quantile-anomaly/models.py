"""Quantile forecasters and the input-perturbation impulse response.

ORACLE   analytic conditional distribution of the true process
CHRONOS2 Chronos-2 zero-shot          (requires huggingface.co)
BOLT     chronos-bolt-small zero-shot (requires huggingface.co)
MLPQR    a quantile MLP trained here from scratch on a broad synthetic
         family, then applied zero-shot to the four target processes.
         This is the stand-in for a pretrained forecaster when the model
         hub is unreachable: it is genuinely learned, so its psi_h and its
         calibration are estimated, not analytic.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.stats import norm

from data import H, T_CTX, QUANTILE_LEVELS, SIGMA2, oracle_moments
import scoring

NQ = QUANTILE_LEVELS.size


# --------------------------------------------------------------- ORACLE

class OracleModel:
    name = "ORACLE"
    analytic_psi = True

    def predict(self, process, ctx, t0):
        mu, sd = oracle_moments(process, ctx, t0)
        return mu[..., None] + sd[..., None] * norm.ppf(QUANTILE_LEVELS)


# ------------------------------------------------------- Chronos wrappers

def try_load_chronos(device: str = "cpu", path: str | None = None):
    """Return (model, name) for Chronos-2, else chronos-bolt-small, else None.

    ``path`` is a local checkpoint directory (one holding config.json).  Pass it
    when huggingface.co is unreachable but the weights were fetched elsewhere
    and copied in; from_pretrained takes a directory just as it takes a repo id.
    Setting HF_HUB_OFFLINE=1 alongside it stops the hub being contacted at all.

    The pipeline class and predict signature differ between the two models, so
    each branch is written against its own package README.
    """
    c2 = path or os.environ.get("CHRONOS2_PATH") or "amazon/chronos-2"
    bolt = os.environ.get("CHRONOS_BOLT_PATH") or "amazon/chronos-bolt-small"
    local = os.path.isdir(c2)
    if local:
        print(f"  loading Chronos-2 from a local directory: {c2}")

    try:
        from chronos import Chronos2Pipeline           # chronos-forecasting >= 2
        p = Chronos2Pipeline.from_pretrained(c2, device_map=device)
        return _Chronos2(p), "CHRONOS2"
    except Exception as e:                             # noqa: BLE001
        print(f"  chronos-2 unavailable: {type(e).__name__}: {str(e)[:160]}")
        if local:
            print(f"    (the path exists; contents: "
                  f"{sorted(os.listdir(c2))[:8]})")
    try:
        import torch
        from chronos import BaseChronosPipeline
        p = BaseChronosPipeline.from_pretrained(
            bolt, device_map=device, torch_dtype=torch.float32)
        return _ChronosBolt(p), "CHRONOS_BOLT"
    except Exception as e:                             # noqa: BLE001
        print(f"  chronos-bolt unavailable: {type(e).__name__}: {str(e)[:160]}")
    return None, None


class _Chronos2:
    analytic_psi = False

    def __init__(self, pipe):
        self.pipe = pipe
        self.name = "CHRONOS2"

    def predict(self, process, ctx, t0, batch=64):
        """Chronos2Pipeline.predict_quantiles(inputs, prediction_length,
        quantile_levels) -> (list of (n_variates, H, NQ) tensors, means)."""
        import torch
        out = []
        for i in range(0, ctx.shape[0], batch):
            chunk = [torch.tensor(r, dtype=torch.float32) for r in ctx[i:i + batch]]
            q, _ = self.pipe.predict_quantiles(
                chunk, prediction_length=H,
                quantile_levels=QUANTILE_LEVELS.tolist())
            arr = torch.stack([t.reshape(-1, H, NQ)[0] for t in q])   # (b,H,NQ)
            out.append(arr.float().cpu().numpy().astype(np.float64))
        return np.concatenate(out, 0)


class _ChronosBolt(_Chronos2):
    """Bolt returns a single stacked tensor (batch, H, NQ), not a list."""

    def __init__(self, pipe):
        super().__init__(pipe)
        self.name = "CHRONOS_BOLT"

    def predict(self, process, ctx, t0, batch=64):
        import torch
        out = []
        for i in range(0, ctx.shape[0], batch):
            chunk = [torch.tensor(r, dtype=torch.float32) for r in ctx[i:i + batch]]
            q, _ = self.pipe.predict_quantiles(
                chunk, prediction_length=H,
                quantile_levels=QUANTILE_LEVELS.tolist())
            out.append(q.float().cpu().numpy().astype(np.float64))
        return np.concatenate(out, 0)


# ------------------------------------------------- learned quantile MLP

class MLPQR:
    """Monotone-by-construction quantile MLP, trained on a broad family."""
    name = "MLPQR"
    analytic_psi = False

    def __init__(self, device="cpu", hidden=256, seed=0):
        import torch, torch.nn as nn
        self.torch, self.device = torch, device
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Linear(T_CTX, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden), nn.GELU(),
                )
                self.anchor = nn.Linear(hidden, H)
                self.width = nn.Linear(hidden, H * (NQ - 1))

            def forward(self, x):
                z = self.body(x)
                a = self.anchor(z).unsqueeze(-1)                  # (B,H,1)
                w = nn.functional.softplus(
                    self.width(z).view(-1, H, NQ - 1)) + 1e-4     # positive
                return torch.cat([a, a + torch.cumsum(w, -1)], -1)  # (B,H,NQ)

        self.net = Net().to(device)

    # scale each window by its own mean / std, exactly like a foundation model
    @staticmethod
    def _scale(ctx):
        m = ctx.mean(axis=-1, keepdims=True)
        s = ctx.std(axis=-1, keepdims=True)
        s = np.maximum(s, 1e-3)
        return m, s

    def fit(self, pool_ctx, pool_fut, steps=4000, batch=512, lr=1e-3, log=print):
        torch = self.torch
        m, s = self._scale(pool_ctx)
        X = torch.tensor((pool_ctx - m) / s, dtype=torch.float32)
        Y = torch.tensor((pool_fut - m) / s, dtype=torch.float32)
        lev = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
        n = X.shape[0]
        self.net.train()
        for step in range(steps):
            idx = torch.randint(0, n, (batch,))
            q = self.net(X[idx])                                  # (B,H,NQ)
            err = Y[idx].unsqueeze(-1) - q
            loss = torch.maximum(lev * err, (lev - 1) * err).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step(); sched.step()
            if log and (step % 500 == 0 or step == steps - 1):
                log(f"    step {step:5d}/{steps}  pinball={loss.item():.5f}")
        self.net.eval()
        return self

    def predict(self, process, ctx, t0, batch=1024):
        torch = self.torch
        m, s = self._scale(ctx)
        Xn = (ctx - m) / s
        out = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], batch):
                x = torch.tensor(Xn[i:i + batch], dtype=torch.float32)
                out.append(self.net(x).numpy().astype(np.float64))
        q = np.concatenate(out, 0)
        return q * s[..., None] + m[..., None]

    def save(self, path):
        self.torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(self.torch.load(path, map_location=self.device))
        self.net.eval()
        return self


def training_pool(n, rng):
    """Windows from a BROAD family, so MLPQR is zero-shot on the 4 targets."""
    from data import BURN_IN
    total = BURN_IN + T_CTX + H
    ctxs, futs = [], []
    per = 4096
    while sum(c.shape[0] for c in ctxs) < n:
        b = min(per, n - sum(c.shape[0] for c in ctxs))
        phi = rng.uniform(0.0, 1.0, size=(b, 1))
        # 25% of the batch is an exact unit root, the rest is stationary AR(1)
        phi[rng.random((b, 1)) < 0.25] = 1.0
        sig = np.sqrt(rng.uniform(0.1, 2.0, size=(b, 1)))
        eps = rng.normal(0.0, 1.0, size=(b, total)) * sig
        x = np.empty_like(eps)
        x[:, 0] = eps[:, 0]
        p = phi[:, 0]
        for t in range(1, total):
            x[:, t] = p * x[:, t - 1] + eps[:, t]
        # seasonality on a random subset
        has = rng.random(b) < 0.5
        period = rng.integers(6, 49, size=(b, 1))
        amp = rng.uniform(0.0, 4.0, size=(b, 1)) * has[:, None]
        phase = rng.uniform(0, 2 * np.pi, size=(b, 1))
        t_idx = np.arange(total)[None, :]
        x = x + amp * np.sin(2 * np.pi * t_idx / period + phase)
        x = x + rng.normal(0, 3.0, size=(b, 1))          # random level offset
        ctxs.append(x[:, BURN_IN:BURN_IN + T_CTX])
        futs.append(x[:, BURN_IN + T_CTX:])
    return np.concatenate(ctxs, 0), np.concatenate(futs, 0)


# ------------------------------------------------ impulse response by delta

def psi_by_perturbation(model, process, ctx, t0, q_base=None):
    """psi_h = (q50_h(+d) - q50_h(-d)) / (2d),  d = 0.05 * sigma_1^model.

    Returns (psi with psi[:,0] = 1, q_base, delta).
    """
    if q_base is None:
        q_base = model.predict(process, ctx, t0)
    i50 = int(np.argmin(np.abs(QUANTILE_LEVELS - 0.5)))
    sd1 = scoring.model_sigma(q_base)[:, 0]
    delta = np.maximum(0.05 * sd1, 1e-6)

    cp, cm = ctx.copy(), ctx.copy()
    cp[:, -1] += delta
    cm[:, -1] -= delta
    both = np.concatenate([cp, cm], 0)                    # one batched call
    t_both = np.concatenate([t0, t0], 0)
    qb = model.predict(process, both, t_both)
    n = ctx.shape[0]
    med_p = np.sort(qb[:n], -1)[..., i50]
    med_m = np.sort(qb[n:], -1)[..., i50]
    psi = (med_p - med_m) / (2 * delta)[:, None]
    return np.concatenate([np.ones((n, 1)), psi], 1), q_base, delta

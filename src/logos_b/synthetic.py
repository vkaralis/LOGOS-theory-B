"""Generate invented observations from specified models; no real records or fitted source parameters."""

import json
from pathlib import Path
import numpy as np
from .fitting import infusion, sigmoid, war_cp, turnover

SEED = 20261012


def generate(n=4, seed=SEED):
    if n < 1:
        raise ValueError("At least one profile per model is required")
    rng = np.random.default_rng(seed)
    rem, war = [], []
    for i in range(n):
        W = float(rng.uniform(6, 16))
        rate = float(rng.uniform(80, 160))
        w = np.array([0.002, 0.008]) * rng.uniform(0.8, 1.2, 2)
        k = np.array([0.035, 0.8]) * rng.uniform(0.8, 1.2, 2)
        ke, K, hill = (
            0.3 * rng.uniform(0.8, 1.2),
            4 * rng.uniform(0.8, 1.2),
            2 * rng.uniform(0.8, 1.2),
        )
        t = np.unique(np.r_[0.0, np.linspace(0.25, W, 20), np.linspace(W + 1, 70, 20)])
        cp = infusion(t, w, k, rate, W)
        eeg = sigmoid(
            infusion(t, w, k, rate, W, ke), np.r_[np.log([ke, K, hill]), 20.0, 0.8]
        )
        rem.append(
            dict(
                ID=f"SYN-REM-{i+1:03}",
                T_min=W,
                rate_ug_min=rate,
                time=t.tolist(),
                cp_ng_ml=np.maximum(0, cp + rng.normal(0, 0.025, len(t))).tolist(),
                eeg=np.maximum(0, eeg + rng.normal(0, 0.08, len(t))).tolist(),
            )
        )
        dose = float(rng.uniform(8, 14))
        ka, lag = 1.2, 0.5
        pk = np.log([0.015 * rng.uniform(0.8, 1.2), 0.15 * rng.uniform(0.8, 1.2)])
        kout, Kw = 0.04 * rng.uniform(0.8, 1.2), 1.0 * rng.uniform(0.8, 1.2)
        tw = np.unique(
            np.r_[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, np.arange(24.0, 145.0, 12.0)]
        )
        c = war_cp(tw, pk, dose, ka, lag)
        pca = turnover(tw, pk, dose, ka, lag, kout, Kw)
        war.append(
            dict(
                ID=f"SYN-WAR-{i+1:03}",
                dose_mg=dose,
                ka_h=ka,
                lag_h=lag,
                time=tw.tolist(),
                cp_mg_L=np.maximum(0, c + rng.normal(0, 0.005, len(tw))).tolist(),
                pca_percent=np.clip(
                    pca + rng.normal(0, 0.15, len(tw)), 0, 100
                ).tolist(),
            )
        )
    return dict(
        schema_version=1,
        synthetic=True,
        seed=seed,
        profiles_per_model=n,
        description="Entirely invented observations; no patient data and no source-fitted profiles.",
        remifentanil=rem,
        warfarin=war,
    )


def write_dataset(path, n=4, seed=SEED):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(generate(n, seed), indent=2, allow_nan=False) + "\n")

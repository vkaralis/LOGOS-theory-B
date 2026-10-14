"""Sequential bounded synthetic-observation PK/PD fits."""

import numpy as np
from scipy import optimize, special


def step_response(t, w, k, ke=None):
    t = np.maximum(np.asarray(t, float), 0)
    out = np.zeros_like(t)
    for wi, ki in zip(w, k):
        if ke is None:
            out += wi * (-np.expm1(-ki * t)) / ki
        else:
            if abs(ke - ki) < 1e-6 * max(ke, ki):
                out += wi * ((1 - np.exp(-ki * t)) / ki - t * np.exp(-ki * t))
            else:
                out += (
                    wi
                    * ke
                    / (ke - ki)
                    * ((-np.expm1(-ki * t)) / ki - (-np.expm1(-ke * t)) / ke)
                )
    return out


def infusion(t, w, k, rate, T, ke=None):
    t = np.asarray(t)
    return rate * (step_response(t, w, k, ke) - step_response(t - T, w, k, ke))


def decode_pk(p):
    # ordered positive rates avoid component permutation
    return np.exp(p[:2]), np.array([np.exp(p[2]), np.exp(p[2]) + np.exp(p[3])])


def fit_rem_pk(t, y, rate, T):
    floor = 0.1
    lo = np.log([1e-7, 1e-7, 0.001, 0.005])
    hi = np.log([1, 1, 0.5, 5])

    def res(p):
        w, k = decode_pk(p)
        return np.log(np.maximum(infusion(t, w, k, rate, T), 1e-9) + floor) - np.log(
            y + floor
        )

    fits = []
    for slow, fast in [(0.02, 0.6), (0.05, 1.5), (0.008, 0.2)]:
        start = np.log([0.001, 0.008, slow, fast])
        fits.append(
            optimize.least_squares(
                res,
                start,
                bounds=(lo, hi),
                max_nfev=400,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        )
    best = min(fits, key=lambda a: np.dot(a.fun, a.fun))
    w, k = decode_pk(best.x)
    return w, k, best


def sigmoid(C, p):
    ke, K, h, E0, frac = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3], p[4]
    eff = special.expit(h * (np.log(np.maximum(C, 1e-15)) - np.log(K)))
    return E0 * (1 - frac * eff)


def fit_rem_pd(t, y, w, k, rate, T, direct=False):
    def res(p):
        C = infusion(t, w, k, rate, T, None if direct else np.exp(p[0]))
        return sigmoid(C, p) - y

    lo = np.array([np.log(0.01), np.log(0.1), np.log(0.3), 8, 0.1])
    hi = np.array([np.log(5), np.log(200), np.log(8), 35, 1])
    E0 = np.clip(np.median(y[t <= 0.5]) if np.any(t <= 0.5) else np.max(y), 9, 34)
    fits = []
    for ke in [0.15, 0.6, 1.5]:
        p = np.array([np.log(ke), np.log(15), np.log(2), E0, 0.8])
        fits.append(
            optimize.least_squares(
                res, p, bounds=(lo, hi), max_nfev=300, ftol=1e-9, xtol=1e-9, gtol=1e-9
            )
        )
    best = min(fits, key=lambda a: np.dot(a.fun, a.fun))
    return best


def war_cp(t, p, dose, ka, lag):
    k, scale = np.exp(p)
    t = np.maximum(np.asarray(t) - lag, 0)
    return (
        dose
        * scale
        * (
            ka * t * np.exp(-k * t)
            if abs(ka - k) < 1e-8 * max(ka, k)
            else ka / (ka - k) * (np.exp(-k * t) - np.exp(-ka * t))
        )
    )


def fit_war_pk(t, y, dose, ka, lag):
    def res(p):
        return np.log(war_cp(t, p, dose, ka, lag) + 0.1) - np.log(y + 0.1)

    b = optimize.least_squares(
        res,
        np.log([0.015, 0.15]),
        bounds=(np.log([0.001, 0.001]), np.log([0.1, 2])),
        max_nfev=300,
    )
    return b


def turnover(t, pk, dose, ka, lag, kout, K, dt=0.05):
    t = np.asarray(t)
    grid = np.arange(0, max(t) + dt, dt)
    mid = (grid[:-1] + grid[1:]) / 2
    target = K / (K + war_cp(mid, pk, dose, ka, lag))
    e = np.exp(-kout * dt)
    # Linear recurrence R_j=e R_(j-1)+(1-e) target_j, R_0=1
    from scipy.signal import lfilter

    vals = lfilter([1 - e], [1, -e], target, zi=[e])[0]
    return 100 * np.interp(t, grid, np.r_[1, vals])


def fit_war_pd(t, y, pk, dose, ka, lag, direct=False):
    def pred(p):
        return (
            100 * np.exp(p[1]) / (np.exp(p[1]) + war_cp(t, pk, dose, ka, lag))
            if direct
            else turnover(t, pk, dose, ka, lag, np.exp(p[0]), np.exp(p[1]))
        )

    fits = []
    for ko, K in [(0.04, 1), (0.1, 0.5), (0.015, 2)]:
        b = optimize.least_squares(
            lambda p: pred(p) - y,
            np.log([ko, K]),
            bounds=(np.log([0.002, 0.01]), np.log([1, 100])),
            max_nfev=200,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        fits.append(b)
    return min(fits, key=lambda b: np.sum(b.fun**2))

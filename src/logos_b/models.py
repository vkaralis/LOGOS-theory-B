"""Finite-course linear PK/PD kernels and concave optimization.
Adapted from the supplied LOGOS computational algorithms; no source observations.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar, least_squares, linprog
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss


def quad_segments(cuts, order=24):
    z, w = leggauss(order)
    cuts = np.unique(cuts)
    nodes = []
    weights = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            nodes.extend((a + b) / 2 + (b - a) / 2 * z)
            weights.extend((b - a) / 2 * w)
    return np.array(nodes), np.array(weights)


class RemModel:
    def __init__(self, m, initial_fraction=0, n=20):
        self.m = m
        self.n = n
        self.W = m["T_min"]
        self.rate = m["rate_ug_min"]
        self.k = np.array([m["k1_min"], m["k2_min"]])
        self.w = np.array([m["w1"], m["w2"]])
        self.ke = m["ke0_min"]
        self.K = m["K_ng_ml"]
        self.h = m["Hill"]
        self.edges = np.linspace(0, self.W, n + 1)
        self.H = self.W + max(100, 20 / self.ke, 20 / min(self.k))
        self.cp0 = np.zeros(2)
        self.ce0 = 0.0
        cpref = self.peak(np.ones(n), "cp")[0]
        ceref = self.peak(np.ones(n), "ce")[0]
        self.cp0 = initial_fraction * cpref * self.w / sum(self.w)
        self.ce0 = initial_fraction * ceref
        self.cpcap = self.peak(np.ones(n), "cp")[0]
        self.cecap = self.peak(np.ones(n), "ce")[0]
        self.tailcap = self.tail(np.ones(n))
        self.qt, self.qw = quad_segments(self.edges, 32)
        self.Mq = self.matrix(self.qt, "ce")
        self.bq = self.initial(self.qt, "ce")

    def step(self, t, kind):
        t = np.maximum(np.asarray(t), 0)
        cp = sum(w * (-np.expm1(-k * t)) / k for w, k in zip(self.w, self.k))
        if kind == "cp":
            return cp
        return sum(
            (
                w * ((-np.expm1(-k * t)) / k - t * np.exp(-k * t))
                if abs(self.ke - k) < 1e-8 * max(self.ke, k)
                else w
                * self.ke
                / (self.ke - k)
                * ((-np.expm1(-k * t)) / k - (-np.expm1(-self.ke * t)) / self.ke)
            )
            for w, k in zip(self.w, self.k)
        )

    def matrix(self, t, kind):
        t = np.atleast_1d(t)[:, None]
        return self.rate * (
            self.step(t - self.edges[:-1], kind) - self.step(t - self.edges[1:], kind)
        )

    def initial(self, t, kind):
        t = np.atleast_1d(t)
        cp = sum(c * np.exp(-k * t) for c, k in zip(self.cp0, self.k))
        if kind == "cp":
            return cp
        return self.ce0 * np.exp(-self.ke * t) + sum(
            (
                c * self.ke * t * np.exp(-k * t)
                if abs(self.ke - k) < 1e-8 * max(self.ke, k)
                else c
                * self.ke
                / (self.ke - k)
                * (np.exp(-k * t) - np.exp(-self.ke * t))
            )
            for c, k in zip(self.cp0, self.k)
        )

    def values(self, t, x, kind):
        return self.matrix(t, kind) @ x + self.initial(t, kind)

    def tail(self, x):
        # Integral Cp tail + Ce(W)/ke, including all initial states.
        cpcomponents = [
            self.rate
            * w
            / k
            * (
                np.exp(-k * (self.W - self.edges[1:]))
                - np.exp(-k * (self.W - self.edges[:-1]))
            )
            @ x
            + self.cp0[j] * np.exp(-k * self.W)
            for j, (w, k) in enumerate(zip(self.w, self.k))
        ]
        return (
            sum(c / k for c, k in zip(cpcomponents, self.k))
            + float(self.values([self.W], x, "ce")[0]) / self.ke
        )

    def peak(self, x, kind):
        if kind == "cp":
            times = self.edges
        else:
            grid = np.unique(
                np.r_[
                    np.linspace(0, self.W, 20 * self.n + 1),
                    self.W + np.geomspace(0.001, self.H - self.W, 200),
                ]
            )
            vals = self.values(grid, x, kind)
            maxima = []
            for i in range(1, len(grid) - 1):
                if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
                    res = minimize_scalar(
                        lambda t: -self.values([t], x, kind)[0],
                        bounds=(grid[i - 1], grid[i + 1]),
                        method="bounded",
                        options={"xatol": 1e-11},
                    )
                    maxima.append(res.x)
            times = np.r_[self.edges, grid[[0, -1]], maxima]
        vals = self.values(times, x, kind)
        i = int(np.argmax(vals))
        return float(vals[i]), float(times[i])

    def objective(self, x, h=1, from_time=0):
        if from_time == 0:
            ce = self.Mq @ x + self.bq
            ww = self.qw
        else:
            if from_time >= self.W:
                return 0.0
            t, ww = quad_segments(
                np.r_[from_time, self.edges[self.edges > from_time]], 32
            )
            ce = self.values(t, x, "ce")
        ce = np.maximum(ce, 1e-14)
        v = (ce / self.K) ** h
        return float(ww @ (v / (1 + v)))

    def solve(self, cap=2, h=1, start=None):
        grid = np.unique(
            np.r_[
                np.linspace(0, self.W, 8 * self.n + 1),
                self.W + np.geomspace(0.001, self.H - self.W, 90),
            ]
        )

        def fun(x):
            return -self.objective(x, h) / self.W

        def jac(x):
            ce = np.maximum(self.Mq @ x + self.bq, 1e-14)
            v = (ce / self.K) ** h
            return -(self.Mq.T @ (self.qw * h * v / (ce * (1 + v) ** 2))) / self.W

        starts = (
            [np.ones(self.n)] if start is None else [np.array(start), np.ones(self.n)]
        )
        if h != 1:
            for slope in [0.8, -0.8]:
                starts.append(1 + slope * np.linspace(-1, 1, self.n))
        best = None
        bestdetails = None
        for initial in starts:
            constraint_times = grid.copy()
            x0 = initial
            for iteration in range(40):
                A = np.vstack(
                    [
                        self.matrix(constraint_times, "cp") / self.cpcap,
                        self.matrix(constraint_times, "ce") / self.cecap,
                    ]
                )
                b = np.r_[
                    1 - self.initial(constraint_times, "cp") / self.cpcap,
                    1 - self.initial(constraint_times, "ce") / self.cecap,
                ]
                tail0 = self.tail(np.zeros(self.n))
                tailrow = (
                    np.array(
                        [self.tail(np.eye(self.n)[i]) - tail0 for i in range(self.n)]
                    )
                    / self.tailcap
                )
                A = np.vstack([A, tailrow])
                b = np.r_[b, 1 - tail0 / self.tailcap]
                constraints = [
                    {
                        "type": "eq",
                        "fun": lambda x: sum(x) - self.n,
                        "jac": lambda x: np.ones(self.n),
                    },
                    {"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A},
                ]
                res = minimize(
                    fun,
                    x0,
                    jac=jac,
                    bounds=[(0, cap)] * self.n,
                    constraints=constraints,
                    method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 1500},
                )
                x = res.x
                cp, tp = self.peak(x, "cp")
                ce, te = self.peak(x, "ce")
                excess = (
                    max(cp / self.cpcap, ce / self.cecap, self.tail(x) / self.tailcap)
                    - 1
                )
                if excess <= 5e-9:
                    break
                constraint_times = np.unique(np.r_[constraint_times, tp, te])
                x0 = x
            if (
                excess <= 1e-7
                and abs(sum(x) - self.n) < 1e-7
                and (best is None or fun(x) < fun(best))
            ):
                best = x
                bestdetails = (res, excess, A, b, cp, ce)
        if best is None:
            raise RuntimeError(
                "Rem feasible candidate not found ID "
                + str(self.m["ID"])
                + " excess "
                + str(excess)
                + " dose "
                + str(sum(x) - self.n)
                + " "
                + res.message
            )
        res, excess, A, b, cp, ce = bestdetails
        active = np.flatnonzero(best > 1e-4)
        last = float(self.edges[active[-1] + 1])
        J = self.objective(best, h)
        Jref = self.objective(np.ones(self.n), h)
        output = {
            "ID": self.m["ID"],
            "cap": cap,
            "initial_fraction": float(sum(self.cp0) / self.cpcap),
            "h": h,
            "J_ref": Jref,
            "J": J,
            "gain_percent": 100 * (J / Jref - 1),
            "last_active": last,
            "coast_J": self.objective(best, h, last),
            "coast_fraction": self.objective(best, h, last) / J,
            "Cp_ratio": cp / self.cpcap,
            "Ce_ratio": ce / self.cecap,
            "tail_ratio": self.tail(best) / self.tailcap,
            "amount_error": abs(sum(best) / self.n - 1),
            "solver_success": bool(res.success),
            "max_excess": excess,
            "controls_rate_factors": best.tolist(),
            "W": self.W,
        }
        y = np.r_[self.cp0, self.ce0, 0.0]
        for j, (a, btime) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            rate = best[j] * self.rate

            def ode(t, y):
                ce = max(y[2], 1e-14)
                vv = (ce / self.K) ** h
                return [
                    -self.k[0] * y[0] + self.w[0] * rate,
                    -self.k[1] * y[1] + self.w[1] * rate,
                    self.ke * (y[0] + y[1] - y[2]),
                    vv / (1 + vv),
                ]

            sol = solve_ivp(ode, (a, btime), y, method="DOP853", rtol=1e-10, atol=1e-11)
            if not sol.success:
                raise RuntimeError(sol.message)
            y = sol.y[:, -1]
        output["independent_objective_error"] = float(abs(y[3] - J))
        output["independent_tail_error"] = float(
            abs(
                sum(y[j] / self.k[j] for j in range(2))
                + y[2] / self.ke
                - self.tail(best)
            )
        )
        if h == 1:
            grad = -jac(best) * self.W
            lp = linprog(
                -grad,
                A_ub=A,
                b_ub=b,
                A_eq=np.ones((1, self.n)),
                b_eq=[self.n],
                bounds=[(0, cap)] * self.n,
                method="highs",
            )
            output["sampled_relaxation_supporting_gap"] = (
                float(max(0, grad @ (lp.x - best))) if lp.success else None
            )
        return output


class WarModel:
    def __init__(self, m, Rinit=1, kout=None, K=None, ka=None, lag=None):
        self.m = m
        self.Rinit = Rinit
        self.kout = m["kout_h"] if kout is None else kout
        self.K = m["IC50_mg_L"] if K is None else K
        self.ka = m["ka_h"] if ka is None else ka
        self.lag = m["lag_h"] if lag is None else lag
        self.k = m["k_h"]
        self.scale = m["scale_L_inv"]
        self.D = m["dose_mg"]
        self.H = 120
        tp = self.lag + (
            1 / self.k
            if abs(self.ka - self.k) < 1e-8 * max(self.ka, self.k)
            else np.log(self.ka / self.k) / (self.ka - self.k)
        )
        self.cpcap = float(self.cp([tp], [0], [1])[0])

    def cp(self, t, times, fractions):
        dt = np.maximum(
            np.atleast_1d(t)[:, None] - np.array(times)[None, :] - self.lag, 0
        )
        A = (
            self.D * self.scale * self.ka * dt * np.exp(-self.k * dt)
            if abs(self.ka - self.k) < 1e-8 * max(self.ka, self.k)
            else self.D
            * self.scale
            * self.ka
            / (self.ka - self.k)
            * (np.exp(-self.k * dt) - np.exp(-self.ka * dt))
        )
        return A @ fractions

    def matrix(self, t, times):
        dt = np.maximum(
            np.atleast_1d(t)[:, None] - np.array(times)[None, :] - self.lag, 0
        )
        return (
            self.D * self.scale * self.ka * dt * np.exp(-self.k * dt)
            if abs(self.ka - self.k) < 1e-8 * max(self.ka, self.k)
            else self.D
            * self.scale
            * self.ka
            / (self.ka - self.k)
            * (np.exp(-self.k * dt) - np.exp(-self.ka * dt))
        )

    def objective(self, times, fractions, lo=0, order=32):
        if lo >= 120:
            return 0.0
        cuts = np.unique(
            np.r_[lo, 120, np.arange(lo, 120, 1.0), np.array(times) + self.lag]
        )
        cuts = cuts[(cuts >= lo) & (cuts <= 120)]
        t, w = quad_segments(cuts, order)
        cp = self.cp(t, times, fractions)
        value = w @ (cp / (self.K + cp) * (-np.expm1(-self.kout * (120 - t))))
        # For partial integrals, contribution from suppression accumulated before lo also matters.
        if lo > 0:
            Rlo = self.response([lo], times, fractions)[0]
            value += (1 - Rlo) * (-np.expm1(-self.kout * (120 - lo))) / self.kout
        else:
            value += (1 - self.Rinit) * (-np.expm1(-self.kout * 120)) / self.kout
        return float(value)

    def response(self, tobs, times=[0], fractions=[1]):
        sol = solve_ivp(
            lambda t, y: [
                self.kout
                * (self.K / (self.K + self.cp([t], times, fractions)[0]) - y[0])
            ],
            (0, float(max(tobs))),
            [self.Rinit],
            t_eval=np.array(tobs),
            method="DOP853",
            rtol=1e-9,
            atol=1e-11,
            max_step=1,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[0]

    def peak(self, times, x):
        cuts = np.r_[
            0, np.array(times) + self.lag, max(times) + self.lag + max(50, 12 / self.k)
        ]
        maxima = [0.0]
        for a, b in zip(cuts[:-1], cuts[1:]):
            res = minimize_scalar(
                lambda t: -self.cp([t], times, x)[0],
                bounds=(a, b),
                method="bounded",
                options={"xatol": 1e-9},
            )
            maxima.extend([a, b, res.x])
        values = self.cp(maxima, times, x)
        j = int(np.argmax(values))
        return float(values[j]), maxima[j]

    def solve(self, spacing=6, verify=True):
        times = np.arange(0, 73, spacing)
        n = len(times)
        t, w = quad_segments(
            np.unique(np.r_[np.arange(0, 121, 1.0), times + self.lag]), 16
        )
        mask = t <= 120
        t = t[mask]
        w = w[mask]
        M = self.matrix(t, times)
        weight = w * (-np.expm1(-self.kout * (120 - t)))
        constant = (1 - self.Rinit) * (-np.expm1(-self.kout * 120)) / self.kout

        def fun(x):
            cp = M @ x
            return -(weight @ (cp / (self.K + cp)) + constant) / 120

        def jac(x):
            cp = M @ x
            return -M.T @ (weight * self.K / (self.K + cp) ** 2) / 120

        grid = np.unique(np.r_[np.arange(0, 200, 0.25), times + self.lag])
        x = np.eye(n)[0]
        for iteration in range(40):
            A = self.matrix(grid, times) / self.cpcap
            res = minimize(
                fun,
                x,
                jac=jac,
                bounds=[(0, 1)] * n,
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda x: sum(x) - 1,
                        "jac": lambda x: np.ones(n),
                    },
                    {"type": "ineq", "fun": lambda x: 1 - A @ x, "jac": lambda x: -A},
                ],
                method="SLSQP",
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            x = res.x
            peak, tp = self.peak(times, x)
            if peak / self.cpcap <= 1 + 5e-9:
                break
            grid = np.unique(np.r_[grid, tp])
        if peak / self.cpcap > 1 + 1e-7 or abs(sum(x) - 1) > 1e-8:
            raise RuntimeError("War feasibility")
        J = self.objective(times, x)
        ref = self.objective([0], [1])
        last = float(times[np.flatnonzero(x > 1e-6)[-1]])
        coast = self.objective(times, x, last) if verify else None
        grad = -jac(x) * 120
        lp = linprog(
            -grad,
            A_ub=A,
            b_ub=np.ones(len(A)),
            A_eq=np.ones((1, n)),
            b_eq=[1],
            bounds=[(0, 1)] * n,
            method="highs",
        )
        if not lp.success:
            raise RuntimeError(lp.message)
        gap = float(max(0, grad @ (lp.x - x)))
        # Independent objective via augmented turnover ODE, with integration broken at lag changes.
        y = np.array([self.Rinit, 0.0])
        cuts = np.unique(np.r_[0, times + self.lag, 120])
        cuts = cuts[cuts <= 120]
        for a, b in zip(cuts[:-1], cuts[1:]) if verify else []:
            sol = solve_ivp(
                lambda t, y: [
                    self.kout * (self.K / (self.K + self.cp([t], times, x)[0]) - y[0]),
                    1 - y[0],
                ],
                (a, b),
                y,
                method="DOP853",
                rtol=1e-10,
                atol=1e-11,
                max_step=1,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            y = sol.y[:, -1]
        return {
            "ID": self.m["ID"],
            "spacing": spacing,
            "Rinit": self.Rinit,
            "kout": self.kout,
            "K": self.K,
            "ka": self.ka,
            "lag": self.lag,
            "J_ref": ref,
            "J": J,
            "gain_percent": 100 * (J / ref - 1),
            "Cp_ratio": peak / self.cpcap,
            "amount_error": abs(sum(x) - 1),
            "solver_success": bool(res.success),
            "gap": gap,
            "independent_objective_error": float(abs(y[1] - J)) if verify else None,
            "dose_times": times.tolist(),
            "dose_fractions": x.tolist(),
            "last_active": last,
            "coast_J": coast,
            "coast_fraction": coast / J if coast is not None else None,
        }

"""Dimensionless fixed-target safety model; independent of synthetic drug fits."""

import numpy as np
from scipy.optimize import minimize, brentq, linprog
from scipy.integrate import solve_ivp

B = 12.613903616328283
CAP = 2.5


class Family:
    def __init__(self, n, nodes=16):
        self.n = n
        self.e = np.linspace(0, 4, n + 1)
        self.dt = 4 / n
        self.d = np.r_[1, np.full(n, self.dt)]
        q, w = np.polynomial.legendre.leggauss(nodes)
        ts = np.concatenate(
            [(a + b) / 2 + (b - a) / 2 * q for a, b in zip(self.e[:-1], self.e[1:])]
        )
        ww = np.tile(w * self.dt / 2, n)
        M = self.C(ts)
        self.lin = M.T @ (4 * ww)
        self.Q = M.T @ (ww[:, None] * M)

    def C(self, t):
        t = np.atleast_1d(t)
        return np.column_stack(
            [
                np.exp(-t),
                np.exp(-np.maximum(t[:, None] - self.e[1:], 0))
                - np.exp(-np.maximum(t[:, None] - self.e[:-1], 0)),
            ]
        )

    def Z(self, t):
        t = np.atleast_1d(t)

        def s(x):
            x = np.maximum(x, 0)
            return (-np.expm1(-0.2 * x) / 0.2 + np.expm1(-x)) / 0.8

        return np.column_stack(
            [
                (np.exp(-0.2 * t) - np.exp(-t)) / 0.8,
                s(t[:, None] - self.e[:-1]) - s(t[:, None] - self.e[1:]),
            ]
        )

    def J(self, x):
        return float(self.lin @ x - 0.5 * x @ self.Q @ x), self.lin - self.Q @ x

    def peak(self, x):
        stops = np.r_[self.e, 20]
        dz = lambda t: float(((self.C([t]) - 0.2 * self.Z([t])) @ x).item())
        roots = [
            brentq(dz, a, b, xtol=1e-13)
            for a, b in zip(stops[:-1], stops[1:])
            if dz(a) * dz(b) < 0
        ]
        p = np.unique(np.r_[stops, roots])
        z = self.Z(p) @ x
        return float(z.max()), float(p[np.argmax(z)]), roots, dz(20)

    def ode(self, x, method, rtol, atol):
        state = np.r_[x[0], 0.0, 0.0]
        peak = 0.0
        for j, (a, b) in enumerate(zip(np.r_[self.e, 20][:-1], np.r_[self.e, 20][1:])):
            u = x[j + 1] if j < self.n else 0.0
            sol = solve_ivp(
                lambda t, y: [
                    u - y[0],
                    y[0] - 0.2 * y[1],
                    4 * y[0] - 0.5 * y[0] ** 2 if j < self.n else 0,
                ],
                (a, b),
                state,
                method=method,
                rtol=rtol,
                atol=atol,
                dense_output=True,
                max_step=0.05,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            dz = lambda t: float(sol.sol(t)[0] - 0.2 * sol.sol(t)[1])
            p = [a, b]
            if dz(a) * dz(b) < 0:
                p.append(brentq(dz, a, b, xtol=1e-12))
            peak = max(peak, float(sol.sol(p)[1].max()))
            state = sol.y[:, -1]
        return float(state[2]), peak

    def report(self, x, target, run_id, grid, solver, margin):
        J, grad = self.J(x)
        peak, t, roots, tail = self.peak(x)
        dose = float(self.d @ x)
        grid = np.unique(np.r_[grid, roots])
        lp = linprog(
            self.d,
            A_ub=np.vstack([self.Z(grid), -grad, self.d]),
            b_ub=np.r_[np.full(len(grid), CAP), -(target - J + grad @ x), B],
            bounds=[(0, 20)] * (self.n + 1),
            method="highs",
        )
        if not lp.success:
            raise RuntimeError(lp.message)
        dop = self.ode(x, "DOP853", 2e-13, 1e-14)
        rad = self.ode(x, "Radau", 2e-13, 1e-14)
        quad = Family(self.n, 32).J(x)[0]
        r = dict(
            run_id=run_id,
            target=target,
            bins=self.n,
            candidate_J=J,
            realized_D=dose,
            dose_lower_bound=float(lp.fun),
            dose_upper_bound=dose,
            finite_family_gap=dose - float(lp.fun),
            target_residual=J - target,
            injury_residual=CAP - peak,
            available_cap=B,
            available_cap_residual=B - dose,
            minimum_control=float(x.min()),
            upper_control_residual=float(20 - x.max()),
            peak_Z=peak,
            peak_time=t,
            tail_derivative_at20=tail,
            quadrature_16_32_error=abs(quad - J),
            DOP853_J=dop[0],
            DOP853_peak=dop[1],
            Radau_J=rad[0],
            Radau_peak=rad[1],
            independent_max_error=max(
                abs(dop[0] - J), abs(rad[0] - J), abs(dop[1] - peak), abs(rad[1] - peak)
            ),
            requested_injury_margin=margin,
            solver=solver,
            evidence="Numerically verified within tolerance; floating-point finite-family dose bound, no interval certificate",
            provenance="fresh dimensionless optimization",
        )
        if not (
            r["target_residual"] >= -1e-7
            and r["injury_residual"] >= -1e-9
            and r["minimum_control"] >= -1e-8
            and r["upper_control_residual"] >= -1e-8
            and r["available_cap_residual"] >= 0
            and r["independent_max_error"] < 1e-8
            and r["tail_derivative_at20"] <= 0
        ):
            raise RuntimeError("Theoretical frontier verification failed")
        pass
        return r


def run_frontier(n=80):
    f = Family(n)
    unit = np.r_[0.0, np.ones(n)]
    safe_rate = CAP / f.peak(unit)[0]
    I1 = 4 + np.expm1(-4)
    I2 = 4 + 2 * np.expm1(-4) - 0.5 * np.expm1(-8)
    ref = lambda u: 4 * u * I1 - 0.5 * u * u * I2
    maximum = ref(safe_rate)
    rows = []
    controls = []
    for target in [10.0, 11.0, 12.0, 13.0]:
        grid = np.unique(np.r_[f.e, np.linspace(0, 20, 401)])
        x = np.r_[2.0, np.where(f.e[:-1] < 1.0, 1.0, 0.0)]
        for iteration in range(30):
            Z = f.Z(grid)
            res = minimize(
                lambda x: (float(f.d @ x), f.d),
                x,
                jac=True,
                bounds=[(0, 20)] * (n + 1),
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda x: f.J(x)[0] - target,
                        "jac": lambda x: f.J(x)[1],
                    },
                    {
                        "type": "ineq",
                        "fun": lambda x: CAP - 1e-6 - Z @ x,
                        "jac": lambda x: -Z,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda x: B - f.d @ x,
                        "jac": lambda x: -f.d,
                    },
                ],
                method="SLSQP",
                options={"ftol": 2e-12, "maxiter": 2000},
            )
            x = res.x
            peak, _, roots, _ = f.peak(x)
            if peak <= CAP - 1e-6 + 1e-9:
                break
            grid = np.unique(np.r_[grid, roots])
        if not res.success:
            raise RuntimeError(res.message)
        row = f.report(
            x,
            target,
            f"J{int(target)}-{n}",
            grid,
            dict(method="SLSQP", success=bool(res.success)),
            1e-6,
        )
        constant = (
            4 * brentq(lambda u: ref(u) - target, 0, safe_rate)
            if target <= maximum
            else None
        )
        row.update(
            constant_maximum_J=maximum,
            constant_minimum_dose=constant,
            dose_saving_percent=(
                100 * (1 - row["realized_D"] / constant) if constant else None
            ),
            constant_feasible=constant is not None,
        )
        rows.append(row)
        controls.append(
            dict(
                target=target,
                initial_atom=float(x[0]),
                rates=x[1:].tolist(),
                edges=f.e.tolist(),
            )
        )
    return rows, controls

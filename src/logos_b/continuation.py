"""Matched final-quarter continuation obligations and conservative bounds."""

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.integrate import solve_ivp
from .models import quad_segments


def check(name, error, tol):
    if not np.isfinite(error) or error > tol:
        raise RuntimeError(f"{name}: {error} > {tol}")


def solve_floor(model, floor, drug):
    rem = drug == "Remifentanil"
    if rem:
        n = model.n
        times = model.edges[:-1]
        grid = np.unique(
            np.r_[
                np.linspace(0, model.W, 8 * n + 1),
                model.W + np.geomspace(0.001, model.H - model.W, 90),
            ]
        )
        M = model.Mq
        initial = model.bq
        weights = model.qw
        K = model.K
        H = model.W
        total = n
        upper = 2.0
        lb = np.where(times >= 0.75 * model.W - 1e-12, floor, 0.0)

        def matrix_constraint(grid):
            AA = np.vstack(
                [
                    model.matrix(grid, "cp") / model.cpcap,
                    model.matrix(grid, "ce") / model.cecap,
                ]
            )
            bb = np.r_[
                1 - model.initial(grid, "cp") / model.cpcap,
                1 - model.initial(grid, "ce") / model.cecap,
            ]
            tail0 = model.tail(np.zeros(n))
            row = (
                np.array([model.tail(np.eye(n)[j]) - tail0 for j in range(n)])
                / model.tailcap
            )
            return np.vstack([AA, row]), np.r_[bb, 1 - tail0 / model.tailcap]

        def maxima(x):
            cp, tp = model.peak(x, "cp")
            ce, te = model.peak(x, "ce")
            return max(
                cp / model.cpcap, ce / model.cecap, model.tail(x) / model.tailcap
            ) - 1, [tp, te]

        x = np.ones(n)
    else:
        times = np.arange(0, 73, 6)
        n = len(times)
        t, w = quad_segments(
            np.unique(np.r_[np.arange(0, 121, 1.0), times + model.lag]), 32
        )
        mask = t <= 120
        t = t[mask]
        w = w[mask]
        M = model.matrix(t, times)
        initial = np.zeros(len(t))
        weights = w * (-np.expm1(-model.kout * (120 - t)))
        K = model.K
        H = 120
        total = 1.0
        upper = 1.0
        lb = np.where(times >= 54, floor / n, 0.0)
        grid = np.unique(np.r_[np.arange(0, 200, 0.25), times + model.lag])

        def matrix_constraint(grid):
            return model.matrix(grid, times) / model.cpcap, np.ones(len(grid))

        def maxima(x):
            cp, tp = model.peak(times, x)
            return cp / model.cpcap - 1, [tp]

        x = np.ones(n) / n
    constant = (
        0.0 if rem else (1 - model.Rinit) * (-np.expm1(-model.kout * 120)) / model.kout
    )

    def fg(x):
        c = M @ x + initial
        J = weights @ (c / (K + c)) + constant
        grad = M.T @ (weights * K / (K + c) ** 2)
        return -J / H, -grad / H

    for iteration in range(40):
        A, b = matrix_constraint(grid)
        res = minimize(
            fg,
            x,
            jac=True,
            bounds=list(zip(lb, np.full(n, upper))),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: sum(x) - total,
                    "jac": lambda x: np.ones(n),
                },
                {"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A},
            ],
            method="SLSQP",
            options={"ftol": 2e-13, "maxiter": 3000},
        )
        x = res.x
        excess, extrema = maxima(x)
        if excess <= 5e-9:
            break
        grid = np.unique(np.r_[grid, extrema])
    if not res.success:
        raise RuntimeError(res.message)
    J = -fg(x)[0] * H
    grad = -fg(x)[1] * H
    lp = linprog(
        -grad,
        A_ub=A,
        b_ub=b,
        A_eq=np.ones((1, n)),
        b_eq=[total],
        bounds=list(zip(lb, np.full(n, upper))),
        method="highs",
    )
    if not lp.success:
        raise RuntimeError(lp.message)
    gap = float(max(0, grad @ (lp.x - x)))
    if rem:
        state = np.r_[model.cp0, model.ce0, 0.0]
        for j, (left, right) in enumerate(zip(model.edges[:-1], model.edges[1:])):

            def rhs(t, y):
                return [
                    -model.k[0] * y[0] + model.w[0] * model.rate * x[j],
                    -model.k[1] * y[1] + model.w[1] * model.rate * x[j],
                    model.ke * (y[0] + y[1] - y[2]),
                    y[2] / (model.K + y[2]),
                ]

            sol = solve_ivp(
                rhs, (left, right), state, method="DOP853", rtol=2e-11, atol=1e-12
            )
            assert sol.success
            state = sol.y[:, -1]
    else:
        state = np.array([model.Rinit, 0.0])
        cuts = np.unique(np.r_[0, times + model.lag, 120])
        cuts = cuts[cuts <= 120]
        for left, right in zip(cuts[:-1], cuts[1:]):

            def rhs(t, y):
                return [
                    model.kout
                    * (model.K / (model.K + model.cp([t], times, x)[0]) - y[0]),
                    1 - y[0],
                ]

            sol = solve_ivp(
                rhs,
                (left, right),
                state,
                method="DOP853",
                rtol=2e-11,
                atol=1e-12,
                max_step=1,
            )
            assert sol.success
            state = sol.y[:, -1]
    err = abs(float(state[-1]) - J)
    check(f'{drug} ID {model.m["ID"]} floor {floor} amount', abs(sum(x) - total), 1e-7)
    check(
        f'{drug} ID {model.m["ID"]} floor {floor} full path exposure',
        max(0, excess),
        1e-7,
    )
    check(
        f'{drug} ID {model.m["ID"]} floor {floor} input obligation',
        float(max(0, np.max(lb - x))),
        1e-8,
    )
    check(f'{drug} ID {model.m["ID"]} floor {floor} independent ODE', err, 1e-7)
    return dict(
        J=J,
        supporting_gap=gap,
        full_path_excess=float(excess),
        amount_error=float(abs(sum(x) - total)),
        independent_objective_error=err,
        solver_success=bool(res.success),
        iterations=iteration + 1,
        controls=x.tolist(),
        lower_bounds=lb.tolist(),
        times=times.tolist(),
    )

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from logos_b.models import RemModel, WarModel
from logos_b.synthetic import generate
from logos_b.pipeline import validate_dataset


def rem():
    return RemModel(
        dict(
            ID="SYN-T",
            T_min=10.0,
            rate_ug_min=100.0,
            w1=0.002,
            w2=0.008,
            k1_min=0.04,
            k2_min=0.8,
            ke0_min=0.3,
            K_ng_ml=4.0,
            Hill=2.0,
        )
    )


def war():
    return WarModel(
        dict(
            ID="SYN-T",
            dose_mg=10.0,
            ka_h=1.2,
            lag_h=0.5,
            k_h=0.015,
            scale_L_inv=0.15,
            kout_h=0.04,
            IC50_mg_L=1.0,
        )
    )


def test_rem_convolution_and_infinite_tail_agree_with_ode():
    m = rem()
    x = np.linspace(0.2, 1.8, m.n)
    state = np.zeros(3)
    for j, (a, b) in enumerate(zip(m.edges[:-1], m.edges[1:])):
        sol = solve_ivp(
            lambda t, y: [
                -m.k[0] * y[0] + m.w[0] * m.rate * x[j],
                -m.k[1] * y[1] + m.w[1] * m.rate * x[j],
                m.ke * (y[0] + y[1] - y[2]),
            ],
            (a, b),
            state,
            rtol=1e-11,
            atol=1e-12,
        )
        assert sol.success
        state = sol.y[:, -1]
    np.testing.assert_allclose(m.values([m.W], x, "cp"), [state[:2].sum()], rtol=1e-9)
    np.testing.assert_allclose(m.values([m.W], x, "ce"), [state[2]], rtol=1e-9)
    assert m.tail(x) == pytest.approx(
        state[0] / m.k[0] + state[1] / m.k[1] + state[2] / m.ke, rel=1e-9
    )


def test_oral_objective_identity_agrees_with_turnover_ode():
    m = war()
    times = [0, 24, 48, 72]
    fractions = [0.4, 0.3, 0.2, 0.1]
    sol = solve_ivp(
        lambda t, y: [
            m.kout * (m.K / (m.K + m.cp([t], times, fractions)[0]) - y[0]),
            1 - y[0],
        ],
        (0, 120),
        [1, 0],
        rtol=1e-10,
        atol=1e-12,
        max_step=0.1,
    )
    assert sol.success
    assert m.objective(times, fractions) == pytest.approx(sol.y[1, -1], abs=1e-7)
    assert m.cp([0, 0.4], times, fractions).tolist() == [0, 0]


def test_equal_rates_have_finite_limits():
    m = rem()
    m.ke = m.k[0]
    assert np.isfinite(m.matrix([0, 5, 10], "ce")).all()
    w = war()
    w.ka = w.k
    expected = w.D * w.scale * w.k * 4 * np.exp(-w.k * 4)
    assert w.cp([w.lag + 4], [0], [1])[0] == pytest.approx(expected)


def test_cap_one_is_reference_and_grids_are_nested():
    m = rem()
    r = m.solve(cap=1)
    np.testing.assert_allclose(r["controls_rate_factors"], np.ones(m.n), atol=1e-8)
    for coarse, fine in [(24, 12), (12, 6)]:
        assert set(np.arange(0, 73, coarse)) <= set(np.arange(0, 73, fine))


def test_synthetic_reproducibility_and_invalid_input():
    d = generate(1, 7)
    assert d == generate(1, 7)
    assert d != generate(1, 8)
    validate_dataset(d)
    d["warfarin"][0]["cp_mg_L"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_dataset(d)

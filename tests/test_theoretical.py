import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from logos_b.theoretical import Family
from logos_b.continuation import solve_floor
from test_models import rem, war


def test_injury_peak_includes_delayed_tail_and_ode():
    f = Family(20)
    x = np.r_[0, np.ones(20)]
    peak, t, _, tail = f.peak(x)
    assert t > 4 and tail < 0
    obj, independent = f.ode(x, "DOP853", 1e-11, 1e-12)
    assert obj == pytest.approx(f.J(x)[0], abs=1e-8)
    assert independent == pytest.approx(peak, abs=1e-8)


def test_continuation_exact_amount_floor_and_objective_bound():
    for model, drug in [(rem(), "Remifentanil"), (war(), "Warfarin")]:
        free = solve_floor(model, 0.0, drug)
        floor = solve_floor(model, 0.25, drug)
        assert free["J"] >= floor["J"] - 1e-7
        assert min(np.array(floor["controls"]) - floor["lower_bounds"]) >= -1e-8
        assert (
            floor["amount_error"] < 1e-8 and floor["independent_objective_error"] < 1e-7
        )
        assert floor["supporting_gap"] >= 0

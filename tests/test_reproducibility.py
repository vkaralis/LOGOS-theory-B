"""The portability check accepts rounding noise but rejects changed data."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("reproduce", Path(__file__).resolve().parents[1] / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


def test_input_comparison_accepts_roundoff_and_rejects_changed_observations():
    reference = {"synthetic": True, "ID": "SYN-001", "values": [1.0, 0.0]}
    reproduce.compare(reference, {**reference, "values": [1.0 + 1e-14, 1e-14]}, rel_tol=1e-12, abs_tol=1e-12)
    for candidate in [{**reference, "values": [1.01, 0.0]}, {**reference, "ID": "SYN-002"}, {**reference, "values": [True, 0.0]}, {**reference, "values": [float("nan"), 0.0]}]:
        with pytest.raises(AssertionError):
            reproduce.compare(reference, candidate, rel_tol=1e-12, abs_tol=1e-12)

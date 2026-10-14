"""Regenerate data and run twice from the installed package, independent of cwd."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b, path="root"):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(path)
        for key in a:
            compare(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError(path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not isinstance(b, (int, float)) or not math.isclose(
            a, b, rel_tol=1e-9, abs_tol=1e-8
        ):
            raise AssertionError(f"{path}: {a} != {b}")
    elif a != b:
        raise AssertionError(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an absent or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    data = out / "regenerated.json"
    subprocess.run(
        [sys.executable, "-m", "logos_b", "generate", "--output", str(data)],
        check=True,
        cwd=out,
    )
    expected = ROOT / "data/synthetic_observations.json"
    if data.read_bytes() != expected.read_bytes():
        raise AssertionError("Synthetic input failed exact regeneration")
    for name in ["run1", "run2"]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "logos_b",
                "run",
                "--data",
                str(data),
                "--output",
                str(out / name),
            ],
            check=True,
            cwd=out,
        )
    files = sorted(p.name for p in (out / "run1").iterdir())
    if files != sorted(p.name for p in (out / "run2").iterdir()):
        raise AssertionError("Output file sets differ")
    checked = []
    for name in files:
        a = out / "run1" / name
        b = out / "run2" / name
        if name == "run_manifest.json":
            continue
        if a.suffix == ".json":
            compare(json.loads(a.read_text()), json.loads(b.read_text()), name)
        elif a.read_bytes() != b.read_bytes():
            raise AssertionError(f"Non-JSON output differs: {name}")
        checked.append(name)
    report = dict(
        all_passed=True,
        input_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        numerical_json_relative_tolerance=1e-9,
        numerical_json_absolute_tolerance=1e-8,
        other_outputs="byte-identical within this environment",
        compared_files=checked,
        excluded_files=["run_manifest.json"],
        python=sys.version.split()[0],
    )
    (out / "reproducibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

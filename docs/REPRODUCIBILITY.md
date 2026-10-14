# Reproducibility and supported environment

Supported interpreter: **Python 3.12**, with `requires-python = ">=3.12,<3.13"`. The package metadata, README and GitHub Actions agree. Other minor versions are deliberately outside the declared support range. GitHub Actions is configured for Linux/Python 3.12; the complete remote execution passed in [GitHub Actions](https://github.com/vkaralis/LOGOS-theory-B/actions/runs/38576825512) for v0.5.2.

The complete dependency closure is pinned in `requirements-lock.txt`, including build tools and pytest. Python packages are installed into a virtual environment without inherited system packages. Installation needs access to PyPI and compatible wheels. Library licenses are separate from this project's MIT license.

The v0.5.1 local release check on 2026-10-14 used macOS arm64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.1 and Matplotlib 3.10.9. A new environment initially contained only pip. All dependencies were downloaded and installed there; the project was installed as a wheel, with no editable source path and no external workbook/data dependency. `pip check` passed and all 7 tests passed.

Two complete executions regenerated identical input bytes, refitted both synthetic models, optimized 32 regimen candidates, evaluated 16 continuation contrasts, computed four theoretical targets and generated four PNG/SVG figures. Nineteen result files agreed: JSON numerics within relative 10⁻⁹ / absolute 10⁻⁸, and CSV/PNG/SVG files byte-for-byte in this environment. Runtime metadata is excluded from repeat comparison. Cross-platform image hashes and optimizer iterates may differ; local equality is not a claim of byte equality on every OS.

`reference_results/` contains the verified synthetic outputs. `validation/local_reproducibility.json` records the repeat comparison. Every run writes input, Python-module and output SHA256 hashes. The run manifest excludes itself from output hashes. Pipeline failures raise an error before a success report is written. Nonempty output directories are rejected.

## Checks and limits

- Infusion quadrature objectives are checked with independently integrated piecewise ODEs. Plasma and effect-site peaks and analytical infinite-tail exposure are audited.
- Oral objective weighting is checked against an independently integrated turnover ODE; amount and the full-path plasma peak are audited.
- The theoretical frontier is checked with DOP853, Radau, a doubled quadrature order and analytical injury extrema. Each reported target includes residuals, dose bounds and the complete control.
- Nested candidates must not decrease beyond tolerance. Conservative bounds subtract the smaller-family support gap and ODE discrepancies; negative/unresolved bounds are retained.
- Tests cover independent kernel/tail identities, oral objective identities, removable equal-rate singularities, the cap-one null, nested grids, invalid data and continuation obligations.

The profile bootstrap has 1,000 replicates seeded by the synthetic dataset seed and holds fitted models fixed. It does not resample residuals or refit parameters. Synthetic fits and small-cohort intervals have demonstration value only. The default workflow does not reproduce observational biomarker statistics or empirical clinical results.

The v0.5.1 results also agree with the previous v0.5.0 reference outputs under the same comparison rules. Scientific review remains pending; see [the separate checklist](SCIENTIFIC_REVIEW.md).

## Cross-platform input regeneration

Linux/x86-64 and macOS/arm64 may produce different last-bit floating-point values despite identical seeds and library versions. Input fields are compared recursively with relative and absolute tolerances 1e-12; IDs, schema, keys and nonnumeric values must agree exactly. The report records byte equality and SHA256 hashes for both distributed and regenerated inputs. This change retains the existing strict two-run output comparison and every numerical model audit.

The v0.5.2 check passed all 8 tests, including rejection of changed observations and acceptance of last-bit roundoff. Two complete local runs passed all audits and agreed across 19 output files. Outputs also agree with the previous reference results. Input byte equality remained true on macOS; the portable input comparison is recorded separately.

For v0.5.3, all 10 tests and two complete local executions passed. Optional PNG/SVG metadata was removed; decoded PNG pixels and SVG geometry match the previous release exactly. Numerical CSV/JSON results are unchanged. The current local report records the metadata and repeat checks.

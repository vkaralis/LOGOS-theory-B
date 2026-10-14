# Changelog

## v0.5.2 — 2026-10-14

- Fix cross-platform synthetic-input regeneration checks using strict 1e-12 numerical tolerances instead of requiring identical float serialization.
- Record both input hashes and byte equality while retaining complete numerical audits and two-run output comparison.
- Publish the requested Theory B README and repository information.

## v0.5.1 — 2026-10-14

- Add an explicit pending scientific-review checklist.
- Recheck installation, tests and complete reproduction in a fresh Python 3.12 environment.
- Preserve MIT coverage of code and synthetic data, all eleven GitHub topics, and Python 3.12-only support.
- Package the updated release with checksums and recoverable Git history.

## v0.5.0 — 2026-10-12

- Initial synthetic-data release of finite-course LOGOS PK/PD examples.
- Seeded generation of entirely invented remifentanil and warfarin observations.
- Sequential PK/PD fitting and exact-amount nested regimen comparisons.
- Final-quarter continuation contrasts with explicit, drug-specific floors.
- Independent ODE objective checks, exposure audits and concave supporting bounds.
- Dimensionless injury-constrained minimum-dose frontier at targets 10–13.
- CSV/JSON results, complete controls and PNG/SVG figures.
- MIT license for project code, documentation and synthetic data.
- Python 3.12 support with pinned dependencies and automated reproducibility workflow.

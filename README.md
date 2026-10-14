# LOGOS Theory B — Clinical Applications of Mathematical Regimen Optimization

**From Mathematical Optimal Control to Model-Informed Pharmacological Decisions**

**Creator: Vangelis D. Karalis**

> **Distributed data:** This public release contains only synthetic remifentanil-like and warfarin-like observations. Parameters are fitted afresh to these synthetic observations; no patient records or parameters fitted to patient records are included. The description below states the research framework; the distributed examples do not reproduce empirical clinical results.

**LOGOS (from Ancient Greek λόγος: reason, rational account; Longitudinal Optimization of Global Outcomes and Strategies)** is a mathematical and computational framework for evaluating whole-regimen pharmacokinetic/pharmacodynamic (PK/PD) treatment strategies.

This repository presents the computational and pharmacological applications of LOGOS Theory B, investigating how mathematical optimization of dose allocation, administration timing, and treatment cessation changes attainable model-based outcomes under matched resource and exposure constraints.

The study integrates:

- **Mathematical optimization:** Nested admissible treatment classes, constrained numerical optimization, and analytical bounds for quantifying incremental regimen-design value.
- **Individual PK/PD modeling:** Pharmacokinetic and pharmacodynamic models fitted to remifentanil and warfarin observations.
- **Finite-course regimen design:** Optimization of infusion-rate profiles and temporal dose redistribution under fixed total administered amounts.
- **Conditional cessation analysis:** Mathematical isolation of stopping-specific value through matched continuation-constrained and unrestricted regimen families.
- **Longitudinal pharmacological dynamics:** Evaluation of residual exposure, delayed pharmacodynamic responses, and post-administration effects.
- **Statistical and numerical verification:** Paired model comparisons, parameter sensitivity, bootstrap uncertainty analysis, and independent trajectory verification.

The Python implementations support reproducible model fitting, numerical optimization, counterfactual regimen simulations, sensitivity analyses, and independent computational checks.

**Scientific objective:** To quantify the incremental value of treatment-design flexibility using mathematical optimization and individualized PK/PD models, while distinguishing temporal redistribution, conditional treatment cessation, resource efficiency, and feasibility.

The analyses demonstrate model-conditional optimization opportunities in fitted pharmacological profiles. Clinical efficacy, toxicity reduction, and therapeutic superiority of the alternative regimens have not been established.

*Computational pharmacology research; not a clinically validated treatment optimization or decision-support system.*

## Release and synthetic examples

Version **0.5.3**. Python **3.12**. MIT license for code and synthetic data.

This repository implements matched finite-course comparisons for remifentanil-like infusion and warfarin-like oral PK/PD models. It also includes a dimensionless, injury-constrained fixed-target dose example. Every distributed observation is generated synthetically. Parameters are fitted afresh to those observations; no patient records or parameters fitted to patient records are included.

The endpoints are mathematical coverage and PCA-suppression proxies. These examples do not establish clinical benefit, bleeding or respiratory safety, or clinically attainable dosing policies. The theoretical injury state is dimensionless.

## Install and reproduce

Use Python 3.12 in a fresh environment, with network access for dependency installation:

```bash
python3.12 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
python -m pytest
python scripts/reproduce.py --output results
```

The reproduction script regenerates the checked-in dataset, checks every field against the distributed dataset (relative/absolute numerical tolerances 1e-12) and records both SHA256 hashes, and executes the full fitting, optimization, numerical verification and plotting pipeline twice. It compares numerical outputs using declared tolerances and excludes runtime metadata from the numerical comparison. Output directories must be absent or empty to prevent mixing old and new results.

For a single run or an alternative synthetic cohort:

```bash
logos-b generate --output data/custom.json --profiles 8 --seed 12345
logos-b run --data data/custom.json --output results/custom
```

The default dataset has four invented profiles per model. `--profiles` changes this count. Increasing it increases fitting and optimization time. The default is a portable computational demonstration; a small synthetic cohort is not an uncertainty estimate for any real population.

## Outputs

- `fitted_models.json` and `fit_diagnostics.json`: estimated parameters and fitting diagnostics.
- `regimen_controls.json`: all reference and flexible controls, objectives and numerical audits.
- `paired_increments.csv`, `increment_summary.csv`: within-profile nested-class increments and conditional profile-bootstrap summaries.
- `continuation_controls.json`, `continuation_increments.csv`: identical flexible families with and without final-quarter floors.
- `sensitivity_controls.json`: fitted-Hill candidates and a shared 1% initial-state sensitivity.
- `theoretical_controls.json`, `theoretical_dose_frontier.csv`: dimensionless target feasibility, dose saving, complete controls and dose bounds.
- Four PNG/SVG figures: paired increments, continuation contrasts, theoretical dose frontier, representative trajectories.
- `verification.json`, `run_manifest.json`: numerical status, versions and SHA256 provenance.

Verified default outputs are provided in `reference_results/`.

Read [model definitions](docs/MODELS.md), [data definitions](data/README.md), [reproducibility](docs/REPRODUCIBILITY.md) and [release instructions](docs/RELEASE.md).

## GitHub topics

`logos`, `pharmacokinetics`, `pharmacodynamics`, `pkpd`, `dose-optimization`, `dosing-regimens`, `mathematical-modeling`, `computational-pharmacology`, `model-informed-precision-dosing`, `treatment-optimization`, `drug-safety`.

The machine-readable list is `.github/topics.json`. GitHub topics are repository settings; committing this file alone does not apply them.

## Scientific review status

Final scientific review remains pending. See the [review checklist](docs/SCIENTIFIC_REVIEW.md). Computational reproducibility alone does not establish scientific equivalence.

# Models and comparison contract

All comparisons hold parameters, starting state, amount rule, objective and trajectory guardrails common within each pair. Candidate increments are calculated before aggregation. Exact-dose comparisons and minimum-dose comparisons answer different questions.

## Infusion model (minutes)

For piecewise-constant nonnegative infusion `u`,

```
Cp(t) = ∫₀ᵗ [w1 exp(-k1(t-s)) + w2 exp(-k2(t-s))] u(s) ds
Ce' = ke0 (Cp - Ce)
EEG = E0 [1 - f Ce^h / (K^h + Ce^h)]
J = ∫₀ᵂ Ce/(K+Ce) dt
```

The objective fixes exponent one to retain concavity, even though EEG fitting estimates `h`. `w1,w2` convert µg input into ng/mL, `k1,k2,ke0` are min⁻¹ and `K` is ng/mL. Twenty equal bins span `W`. With rate factors `x`, `sum(x)=20` matches `D=rate*W`. Caps are 1, 1.25, 1.5 and 2 times the reference rate. At cap 1, amount equality forces the reference. Starting states are zero, except for a shared 1% initial-state sensitivity with recomputed reference caps.

Common guardrails are the full-path reference Cp and Ce peaks and the reference integral of Ce after W through infinity. Cp extrema occur at bin endpoints; Ce peaks are refined from local maxima of a dense transition/tail grid. The tail integral is analytical. Continuous peak searches are floating-point numerical checks, not interval certificates. The fitted-Hill objective uses deterministic multistart candidates and has no global-optimality claim.

## Oral model (hours)

For an amount `d` administered at `tj`, with `x=t-tj-lag>=0`,

```
Cp contribution = d * scale * ka * [exp(-k*x)-exp(-ka*x)]/(ka-k)
R' = kout [K/(K+Cp)-R], R(0)=1
J = ∫₀¹²⁰ (1-R) dt
```

Before absorption, each contribution is zero. The equal-rate limit is handled explicitly. `scale` is L⁻¹, `ka,k,kout` are h⁻¹, `lag` is h and `K` is mg/L. `R` is a fraction of a fixed 100% baseline. The reference places all amount at zero; flexible grids are 24, 12 and 6 h through 72 h. Fractions sum to one and remain nonnegative. The reference full-path plasma peak is the shared cap. The weighted convolution identity for J is checked by independent turnover ODE integration.

## Continuation and numerical bounds

The cap-two infusion family and 6 h oral family are compared against themselves with final-quarter floors. For ε=0.05 or 0.25, the last five infusion bins have rate-factor lower bound ε. The oral opportunities at 54,60,66,72 h have fraction lower bound ε/13. Obligated amounts are ε/4 and 4ε/13 of total amount, respectively. These policies are drug-specific, hypothetical obligations.

For a concave objective and candidate v, a supporting-plane LP over a sampled relaxation yields `gap=max_w grad(J(v))·(w-v)`. The relaxation contains the continuously feasible family. Under numerical quadrature and path-check assumptions, `J_larger-J_smaller-gap_smaller` is a conservative increment bound. Reported lower bounds also subtract both independent ODE objective discrepancies. Unresolved or negative bounds are retained. A count above 10⁻⁶ is a descriptive numerical count, not statistical significance.

## Dimensionless injury and fixed-target dose

```
C' = u-C, C(0+) = initial_atom
Z' = C-0.2 Z, Z(0)=0
J = ∫₀⁴ (4C - C²/2) dt
D = initial_atom + ∫₀⁴ u dt
```

Minimize D at targets 10,11,12,13 with `max_t Z(t)<=2.5`, a common available amount cap 12.613903616328283, and 80 input bins plus a loading atom. Numerical input bounds are 0–20; a 10⁻⁶ injury margin is requested. After time 4 input is zero. Analytical two-exponential expressions and derivative roots find all injury extrema through 20; a nonpositive derivative at 20 and the single possible post-input stationary point exclude a later maximum. Independent DOP853 and Radau integrations and doubled quadrature order check J and the injury peak.

The constant family has zero initial atom and a constant rate through 4. Its analytical safe objective ceiling is approximately 11.270930. Dose saving is reported only when its target is feasible. Targets 12 and 13 demonstrate restoration of feasibility in this theoretical model; percentage saving is undefined there. LP dose bounds apply to the finite 80-bin-plus-atom family, not unrestricted controls.

## Scope

This release covers synthetic fitting, primary nested permissions, continuation, selected sensitivities and the theoretical frontier. It does not implement observational MTX analyses, empirical cohort denominators, pooled-parameter leave-one-out analyses, residual-refit bootstrap, label-range screening, or clinical validation. Synthetic-data figures describe the synthetic outputs only.

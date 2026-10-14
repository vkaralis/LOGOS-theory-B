# Synthetic data

`synthetic_observations.json` is licensed under MIT. All IDs, design values and observations are invented using seed 20261012. No original observations, copied identifiers, or source-fitted parameter arrays are distributed. Data are generated using `logos_b.synthetic` with four profiles per model.

Schema version 1 has `synthetic: true`, `seed`, `profiles_per_model`, `remifentanil`, and `warfarin`. Times must start at zero and strictly increase. Observation channels are aligned arrays of finite nonnegative numbers.

| Model | Fields | Units |
|---|---|---|
| Infusion | `time`, `T_min`, `rate_ug_min` | min, min, µg/min |
| Infusion observations | `cp_ng_ml`, `eeg` | ng/mL, model EEG readout |
| Oral | `time`, `dose_mg`, `ka_h`, `lag_h` | h, mg, h⁻¹, h |
| Oral observations | `cp_mg_L`, `pca_percent` | mg/L, % |

Generation adds independent normal noise to concentrations and responses, truncating concentrations at zero and PCA to 0–100. Design values and known oral absorption/lag are held fixed during fitting. Oral absorption/lag are simulation settings, not pooled estimates inferred from participants. Synthetic observations do not encode the completeness or missingness patterns of any empirical cohort.

Fitting uses only observations and declared administration/absorption settings. Generated latent parameters are not passed into fitting. The dataset can be regenerated within numerical tolerances of 1e-12 across supported environments; byte equality is also recorded.

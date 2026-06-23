# Paper Diagnostics

This folder contains diagnostic scripts and compact summaries that support the manuscript discussion of inverse-PINN failure modes. These files are not required for the final baseline workflow in `run_all.m`, but they document the ablation and residual-landscape checks used to motivate the observation-anchored method.

## Contents

- `run_naive_inverse_PINN_baseline.m` — diagnostic baseline that disables the observation-anchored geometry residual and attempts geometry recovery through the standard neural automatic-differentiation physics residual. This corresponds to the naive inverse-PINN comparison discussed in the manuscript.
- `emhd1d_scanObsGeomDelta.m` — scans the observation-anchored geometry residual as a function of constriction depth `delta`.
- `emhd1d_scanWarmupPhysVsObsGeom.m` — compares a frozen-network physics residual against the observation-anchored residual over a `delta` scan, illustrating how neural-derivative error can distort the geometry-gradient signal.
- `emhd1d_checkIdentifiability.m` — forward sensitivity diagnostic used to assess whether geometry perturbations produce measurable changes in the state fields.
- `emhd1d_directGeometryFit.m` — direct finite-difference-based geometry-fit diagnostic used as an additional inverse-problem check.
- `naive_pinn_failure_summary.csv` — compact numerical summary of the representative derivative-error and residual-scan values reported in the manuscript.

## Interpretation

The final method is implemented in `src/emhd1d_trainGeometryPINN.m`. It constructs geometry updates from noisy observation-derived fields rather than from higher-order neural derivatives. The scripts in this folder are retained to make the diagnostic path reproducible and to document why the final training strategy differs from a conventional inverse PINN.

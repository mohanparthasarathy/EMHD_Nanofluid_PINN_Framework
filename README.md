# EMHD_Nanofluid_PINN_Framework
This repository contains the MATLAB codebase used in the chapter entitled "Physics-Informed Inverse Modeling of EMHD Nanofluid Transport in Tumor-Constricted Microvessels" in the upcoming edited book "Advances in Nanofluids – Modeling, Simulation, Experimentation, and Applications" to be published in Springer.

The code implements a one-dimensional electro-magnetohydrodynamic (EMHD) nanofluid transport model in a tumor-constricted microvessel and a staged inverse physics-informed neural network (PINN) for recovering the hidden vessel geometry from sparse noisy transport observations.

The inverse problem estimates three constriction parameters:

- `delta`: constriction depth,
- `x_t`: constriction center/location,
- `sigma`: constriction width.

The observations consist of sparse noisy measurements of velocity `u(x,t)`, nanoparticle concentration `C(x,t)`, and temperature `T(x,t)`.

## Main methodological point

The package includes both the final observation-anchored inverse PINN and diagnostic scripts used to study a failure mode of standard inverse PINNs. In the naive inverse-PINN formulation, geometry is updated through the usual automatic-differentiation physics residual. That approach can reconstruct the state fields accurately in value while producing unreliable geometry estimates because geometry gradients are computed in residuals containing inaccurate higher-order neural derivatives.

The final method avoids this failure mode by using an observation-anchored geometry residual. Sparse noisy observations are interpolated onto a structured grid, lightly smoothed, differentiated by finite differences, and then used to evaluate the geometry-sensitive governing-equation residual. The clean finite-difference reference solution is used to generate synthetic measurements and to evaluate final errors, but it is not used inside the geometry-identification loss.

## Repository structure

```text
EMHD_1D_Geometry_PINN_Publication_Package/
├── README.md
├── MANIFEST.md
├── run_all.m
├── self_test.m
├── verify_FDM_MMS.m
├── generate_paper_figures.m
├── src/
├── paper_diagnostics/
├── paper_figures/
├── paper_reported_results/
├── results/
├── results_MMS/
├── results_self_test/
├── results_hard_stress_test/
└── robustness_results/
```

See `MANIFEST.md` for a file-by-file description.

## Requirements

The code is written for MATLAB and uses:

- MATLAB with `dlarray`, `dlgradient`, and related deep-learning functionality,
- Deep Learning Toolbox,
- standard MATLAB plotting and interpolation functions, including `scatteredInterpolant`, `movmean`, and `saveas`.

The scripts were organized to run from the repository root. The full training workflow can be computationally expensive because the baseline inverse PINN uses 19,500 training iterations. Saved result files are included so that the numerical values and paper figures can be inspected without rerunning the full training.

## Quick start

From MATLAB, run the following from the repository root:

```matlab
cd EMHD_1D_Geometry_PINN_Publication_Package
run_all
```

This performs the baseline workflow:

1. build the default configuration,
2. generate the finite-difference synthetic reference solution,
3. sample sparse noisy observations,
4. train the staged inverse PINN,
5. save the trained model, figures, and final summary to `results/`.

For a fast smoke test:

```matlab
self_test
```

For finite-difference solver verification:

```matlab
verify_FDM_MMS
```

For the robustness suite and the 10% noise stress test, run from the repository root:

```matlab
addpath('src')
run_robustness_suite
run_hard_stress_test
```

To regenerate the paper figures from saved outputs without rerunning training:

```matlab
generate_paper_figures
```

## Baseline reported results

The final baseline run in `results/summary.txt` reports:

| Parameter | True value | Estimated value | Error |
|---|---:|---:|---:|
| `delta` | 0.3600000000 | 0.3431229889 | 4.688058% |
| `x_t` | 0.5800000000 | 0.5846211910 | 0.796757% |
| `sigma` | 0.1050000000 | 0.1178169772 | 12.206644% |

The baseline state-reconstruction errors are:

| Field | Relative L2 error |
|---|---:|
| `u` | `7.66796078e-03` |
| `C` | `5.25788379e-03` |
| `T` | `1.08362121e-04` |

Compact copies of the paper-reported values are stored in `paper_reported_results/`.

## Reproducibility notes

- Random seeds are set in the configuration and individual scripts.
- The main physical, geometry, data, and training parameters are centralized in `src/emhd1d_defaultConfig.m`.
- The finite-difference solver is verified independently by `verify_FDM_MMS.m`.
- The final observation-anchored geometry residual is constructed only from sparse noisy observations, not from the clean reference solution.
- Saved `.mat`, `.png`, `.pdf`, and `.csv` files are included to document the final runs used in the manuscript.

Because PINN training is stochastic and hardware-dependent, rerunning the full training may produce small numerical differences from the saved results. The qualitative findings and reported parameter-recovery behavior should be reproducible under the provided configuration.

## Output folders

- `results/`: baseline trained model, final summary, synthetic data, and figures.
- `results_MMS/`: manufactured-solution convergence plot.
- `results_self_test/`: output from a short smoke test.
- `robustness_results/`: random-seed, doubled-noise, and sparse-data robustness outputs.
- `results_hard_stress_test/`: approximately 10% measurement-noise stress-test output.
- `paper_figures/`: figures prepared for the manuscript.
- `paper_reported_results/`: compact CSV copies of the values reported in the chapter.

## Recommended citation

If using this code, cite the accompanying chapter/manuscript and reference this repository as the associated code package.

# Manifest

This manifest describes the contents of the EMHD 1D Geometry PINN publication package.

## Top-level files

- `README.md` — overview, installation notes, quick-start commands, and reported baseline values.
- `MANIFEST.md` — file-by-file description of the repository.
- `.gitignore` — excludes operating-system files, temporary MATLAB files, and local cache files.
- `run_all.m` — complete baseline workflow: configuration, finite-difference data generation, observation sampling, staged inverse-PINN training, diagnostics, and figures.
- `self_test.m` — reduced smoke test for checking that the main code path executes.
- `verify_FDM_MMS.m` — method-of-manufactured-solutions verification for the finite-difference solver.
- `generate_paper_figures.m` — regenerates paper figures and summary tables from saved outputs without rerunning the full inverse-PINN training.

## `src/`

Core configuration and validation:

- `emhd1d_defaultConfig.m` — central configuration file for physical parameters, geometry, data sampling, neural-network architecture, training phases, and optimization settings.
- `emhd1d_validateConfig.m` — basic runtime checks for required configuration fields and parameter ranges.

Forward model and synthetic observations:

- `emhd1d_generateSyntheticData.m` — Crank--Nicolson/Picard finite-difference solver for the coupled 1D EMHD transport system.
- `emhd1d_geometry.m` — tumor-constricted vessel half-width function.
- `emhd1d_hFromTheta.m` — `dlarray`-compatible geometry evaluation for trainable geometry parameters.
- `emhd1d_ku.m` — geometry-dependent nanoparticle uptake coefficient.
- `emhd1d_makeObservations.m` — sparse noisy observations, initial-condition samples, boundary-condition samples, and collocation sets.

PINN model and optimization:

- `emhd1d_initMLP.m` — initializes the multilayer perceptron and trainable raw geometry variables.
- `emhd1d_forwardMLP.m` — neural-network forward pass.
- `emhd1d_normalizeInput.m` — maps physical `(x,t)` coordinates to normalized network inputs.
- `emhd1d_residualsAD.m` — standard PINN residuals evaluated using neural-network automatic differentiation.
- `emhd1d_trainGeometryPINN.m` — final staged inverse-PINN training routine, including the observation-anchored geometry residual constructed from noisy observations.
- `emhd1d_adamUpdate.m` — Adam optimizer update for network and geometry variables.
- `emhd1d_boundFromRaw.m` and `emhd1d_rawFromBound.m` — logistic transforms between unconstrained optimizer variables and bounded physical parameters.
- `emhd1d_paramsFromNet.m` — extracts bounded geometry parameters from the network structure.

Evaluation, plotting, and diagnostics:

- `emhd1d_evalGrid.m` — evaluates a trained PINN on the finite-difference grid.
- `emhd1d_finalDiagnostics.m` — computes final geometry and state errors and writes baseline diagnostic figures.
- `emhd1d_plotSyntheticData.m` — plots synthetic fields and sparse observations.
- `emhd1d_plotTrainingHistory.m` — plots loss histories and geometry trajectories.
- `emhd1d_checkIdentifiability.m` — forward sensitivity/identifiability diagnostic.
- `emhd1d_directGeometryFit.m` — direct FDM-based geometry-fit diagnostic.
- `emhd1d_scanObsGeomDelta.m` — observation-anchored residual scan over constriction depth.
- `emhd1d_scanWarmupPhysVsObsGeom.m` — frozen-network physics-loss versus observation-anchored-loss scan.
- `run_robustness_suite.m` — random-seed, doubled-noise, and sparse-data robustness experiments.
- `run_hard_stress_test.m` — approximately 10% noise stress test.

## `paper_diagnostics/`

Scripts and compact output files used to support manuscript diagnostics, especially the comparison between a naive inverse PINN and the observation-anchored method. These files are included for transparency and are not the recommended final training workflow.

## `paper_figures/`

Figures used in the chapter and generated or copied by `generate_paper_figures.m`. This folder also contains compact table output used for manuscript preparation.

## `paper_reported_results/`

Small CSV summaries of the numerical values reported in the manuscript:

- `baseline_summary.csv`
- `robustness_summary.csv`
- `hard_stress_test_summary.csv`

## Saved result folders

- `results/` — final baseline model, synthetic data, figures, and `summary.txt`.
- `results_MMS/` — manufactured-solution convergence plot.
- `results_self_test/` — saved smoke-test outputs.
- `robustness_results/` — robustness-suite trained models, training-history plots, and summary files.
- `results_hard_stress_test/` — high-noise stress-test output.

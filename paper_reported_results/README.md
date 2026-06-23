# Paper-Reported Results

This folder stores compact CSV files containing the numerical values reported in the manuscript. These files allow the reported values to be checked without rerunning the full PINN training workflow.

## Files

- `baseline_summary.csv` — final baseline inverse-PINN run reported in `results/summary.txt`.
- `robustness_summary.csv` — robustness suite with random-seed, doubled-noise, and sparse-data experiments.
- `hard_stress_test_summary.csv` — approximately 10% measurement-noise stress test.

## Baseline values

The baseline run estimates the true geometry
`(delta, x_t, sigma) = (0.36, 0.58, 0.105)` as approximately
`(0.34312, 0.58462, 0.11782)`, with errors of about `4.69%`, `0.80%`, and `12.21%`.

The corresponding state-reconstruction relative errors are approximately:

- velocity `u`: `7.67e-3`,
- concentration `C`: `5.26e-3`,
- temperature `T`: `1.08e-4`.

## Robustness summary

The robustness experiments show that constriction depth and location remain more stable than constriction width. The sparse-data case particularly affects `sigma`, consistent with the weaker identifiability of width from sparse observations.

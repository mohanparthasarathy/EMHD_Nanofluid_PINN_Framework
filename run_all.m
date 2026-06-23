%RUN_ALL Baseline EMHD nanofluid data generation and geometry recovery.
% This script generates synthetic data, samples noisy observations, trains
% the staged inverse PINN, and writes the baseline results to results/.

clear; clc; close all;
rootDir = fileparts(mfilename('fullpath'));
addpath(fullfile(rootDir,'src'));

cfg = emhd1d_defaultConfig();
outDir = fullfile(rootDir,'results');
if ~exist(outDir,'dir'); mkdir(outDir); end

fprintf('\n============================================================\n');
fprintf('1D EMHD nanofluid PINN geometry recovery\n');
fprintf('============================================================\n');
fprintf('True geometry: delta=%.4f, x_t=%.4f, sigma=%.4f\n', ...
    cfg.geom.deltaTrue, cfg.geom.xtTrue, cfg.geom.sigmaTrue);
fprintf('PINN target: recover delta, x_t, and sigma from non-truth initial guesses\n');
fprintf('Initial guess: delta=%.4f, x_t=%.4f, sigma=%.4f\n', ...
    cfg.pinn.deltaInit, cfg.pinn.xtInit, cfg.pinn.sigmaInit);

fprintf('\n--- Step 1: finite-difference synthetic data generation ---\n');
sol = emhd1d_generateSyntheticData(cfg);
data = emhd1d_makeObservations(sol, cfg);
save(fullfile(outDir,'synthetic_data.mat'),'cfg','sol','data','-v7.3');
emhd1d_plotSyntheticData(sol, data, cfg, outDir);

fprintf('\n--- Step 2: staged PINN geometry inversion ---\n');
pinn = emhd1d_trainGeometryPINN(sol, data, cfg, outDir);

fprintf('\n--- Step 3: final diagnostics ---\n');
diag = emhd1d_finalDiagnostics(pinn, sol, data, cfg, outDir);
save(fullfile(outDir,'final_results.mat'),'cfg','sol','data','pinn','diag','-v7.3');

fprintf('\n============================================================\n');
fprintf('FINAL GEOMETRY RECOVERY SUMMARY\n');
fprintf('============================================================\n');
fprintf('delta: true %.8f | estimated %.8f | error %.3f %%\n', ...
    diag.deltaTrue, diag.deltaEst, diag.deltaErrPct);
fprintf('x_t:   true %.8f | estimated %.8f | error %.3f %%\n', ...
    diag.xtTrue, diag.xtEst, diag.xtErrPct);
fprintf('sigma: true %.8f | estimated %.8f | error %.3f %%\n', ...
    diag.sigmaTrue, diag.sigmaEst, diag.sigmaErrPct);
fprintf('relative field errors: u %.3e | C %.3e | T %.3e\n', ...
    diag.relU, diag.relC, diag.relT);
fprintf('Results saved under:\n  %s\n', outDir);

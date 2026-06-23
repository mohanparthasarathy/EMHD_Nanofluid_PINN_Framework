%SELF_TEST Reduced run that checks the main workflow on a small problem.
clear; clc; close all;
rootDir = fileparts(mfilename('fullpath'));
addpath(fullfile(rootDir,'src'));

cfg = emhd1d_defaultConfig();
cfg.grid.Nx = 41;
cfg.grid.Nt = 101;
cfg.time.tFinal = 0.20;
cfg.data.nObs = 300;
cfg.data.nCol = 500;
cfg.pinn.layers = [2 24 24 3];
cfg.pinn.printEvery = 10;
for k = 1:numel(cfg.pinn.phases)
    cfg.pinn.phases(k).epochs = 15;
end

emhd1d_validateConfig(cfg);
outDir = fullfile(rootDir,'results_self_test');
if ~exist(outDir,'dir'); mkdir(outDir); end

sol = emhd1d_generateSyntheticData(cfg);
data = emhd1d_makeObservations(sol, cfg);
pinn = emhd1d_trainGeometryPINN(sol, data, cfg, outDir);
diag = emhd1d_finalDiagnostics(pinn, sol, data, cfg, outDir);

disp(diag);
fprintf('Self-test complete. This does not imply converged geometry; it only checks execution.\n');

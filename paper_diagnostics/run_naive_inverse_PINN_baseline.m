%RUN_NAIVE_INVERSE_PINN_BASELINE Diagnostic baseline for the paper.
%
% This script intentionally disables the observation-anchored geometry loss
% and attempts to update geometry through the standard neural PINN physics
% residual. It is included to document the naive inverse-PINN baseline
% discussed in the manuscript. It is not the recommended final method.
%
% Expected use:
%   addpath('src') or run this from the package root after addpath('src')
%
% Note: this is a diagnostic reproduction script. Full training can be slow.

clear; clc; close all;
% addpath('src');
scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(fullfile(repoRoot,'src'));
cd(repoRoot);

cfg = emhd1d_defaultConfig();
cfg.rngSeed = 7;
cfg.pinn.seed = 11;

% Disable observation-anchored geometry residual globally and by phase.
cfg.pinn.wObsGeom = 0.0;

% A naive simultaneous inverse-PINN schedule: train network and geometry
% through the AD-based physics residual. The geometry parameters receive
% gradients only through L_phys, not through L_obs-geom.
cfg.pinn.phases = struct([]);

cfg.pinn.phases(1).name = 'naive_inverse_pinn';
cfg.pinn.phases(1).epochs = 6000;
cfg.pinn.phases(1).trainNet = true;
cfg.pinn.phases(1).trainDelta = true;
cfg.pinn.phases(1).trainXt = true;
cfg.pinn.phases(1).trainSigma = true;
cfg.pinn.phases(1).wData = 14.0;
cfg.pinn.phases(1).wIC = 20.0;
cfg.pinn.phases(1).wBC = 10.0;
cfg.pinn.phases(1).wPhysStart = 0.05;
cfg.pinn.phases(1).wPhysEnd = 0.75;
cfg.pinn.phases(1).netLRFactor = 1.0;
cfg.pinn.phases(1).geomLRFactor = 1.0;
cfg.pinn.phases(1).wObsGeom = 0.0;

outDir = fullfile('paper_diagnostics','naive_inverse_PINN_results');
if ~exist(outDir,'dir'); mkdir(outDir); end

fprintf('Generating synthetic data and observations...\n');
sol = emhd1d_generateSyntheticData(cfg);
data = emhd1d_makeObservations(sol,cfg);

fprintf('Running naive inverse PINN diagnostic...\n');
pinnNaive = emhd1d_trainGeometryPINN(sol,data,cfg,outDir);

fprintf('\nNaive inverse PINN final geometry:\n');
fprintf('delta true %.8f | estimated %.8f\n',cfg.geom.deltaTrue,pinnNaive.theta.delta);
fprintf('x_t   true %.8f | estimated %.8f\n',cfg.geom.xtTrue,pinnNaive.theta.xt);
fprintf('sigma true %.8f | estimated %.8f\n',cfg.geom.sigmaTrue,pinnNaive.theta.sigma);

save(fullfile(outDir,'naive_inverse_pinn_summary.mat'),'cfg','pinnNaive');

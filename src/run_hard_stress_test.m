function pinn = run_hard_stress_test()
%RUN_HARD_STRESS_TEST High-noise robustness test for the final inverse PINN.

clc;

cfg = emhd1d_defaultConfig();

% Hard stress-test settings
cfg.rngSeed = 101;
cfg.pinn.seed = 101;

cfg.data.noiseU = 0.1;
cfg.data.noiseC = 0.1;
cfg.data.noiseT = 0.1;

cfg.data.nObs = 1000;
cfg.data.nVal = 500;
cfg.data.nCol = 3000;
cfg.data.nObsForPhys = 1000;

outDir = fullfile('results_hard_stress_test');
if ~exist(outDir,'dir')
    mkdir(outDir);
end

fprintf('\n============================================================\n');
fprintf('HARD STRESS TEST\n');
fprintf('10x noise, 1000 observations, 3000 collocation points\n');
fprintf('============================================================\n');

sol = emhd1d_generateSyntheticData(cfg);
data = emhd1d_makeObservations(sol,cfg);

fprintf('\nActual generated data:\n');
fprintf('obs=%d | val=%d | col=%d | phys=%d\n', ...
    size(data.obs.X,2), size(data.val.X,2), ...
    size(data.col.X,2), size(data.phys.X,2));
fprintf('noiseU=%g | noiseC=%g | noiseT=%g\n', ...
    cfg.data.noiseU, cfg.data.noiseC, cfg.data.noiseT);

pinn = emhd1d_trainGeometryPINN(sol,data,cfg,outDir);

deltaHat = pinn.theta.delta;
xtHat    = pinn.theta.xt;
sigmaHat = pinn.theta.sigma;

deltaErr = 100*abs(deltaHat-cfg.geom.deltaTrue)/cfg.geom.deltaTrue;
xtErr    = 100*abs(xtHat-cfg.geom.xtTrue)/cfg.geom.xtTrue;
sigmaErr = 100*abs(sigmaHat-cfg.geom.sigmaTrue)/cfg.geom.sigmaTrue;

fprintf('\n============================================================\n');
fprintf('HARD STRESS TEST SUMMARY\n');
fprintf('============================================================\n');
fprintf('delta: true %.8f | estimated %.8f | error %.4f %%\n', ...
    cfg.geom.deltaTrue, deltaHat, deltaErr);
fprintf('x_t:   true %.8f | estimated %.8f | error %.4f %%\n', ...
    cfg.geom.xtTrue, xtHat, xtErr);
fprintf('sigma: true %.8f | estimated %.8f | error %.4f %%\n', ...
    cfg.geom.sigmaTrue, sigmaHat, sigmaErr);

save(fullfile(outDir,'hard_stress_summary.mat'), ...
    'cfg','pinn','deltaErr','xtErr','sigmaErr');

end
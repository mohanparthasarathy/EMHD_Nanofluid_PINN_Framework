function resultsTable = run_robustness_suite()
%RUN_ROBUSTNESS_SUITE Robustness checks for EMHD inverse PINN.
%
% Runs:
%   1) three random seeds,
%   2) one high-noise case,
%   3) one sparse-data case.
%
% Outputs:
%   robustness_results/robustness_summary.csv
%   robustness_results/robustness_summary.mat

clc;

if ~exist('robustness_results','dir')
    mkdir('robustness_results');
end

cases = {
    struct('name',"Seed11",     'seed',11, 'noiseScale',1.0, 'dataFrac',1.0)
    struct('name',"Seed21",     'seed',21, 'noiseScale',1.0, 'dataFrac',1.0)
    struct('name',"Seed37",     'seed',37, 'noiseScale',1.0, 'dataFrac',1.0)
    struct('name',"HighNoise",  'seed',11, 'noiseScale',2.0, 'dataFrac',1.0)
    struct('name',"SparseData", 'seed',11, 'noiseScale',1.0, 'dataFrac',0.5)
};

rows = cell(numel(cases),10);

for k = 1:numel(cases)

    c = cases{k};

    fprintf('\n============================================================\n');
    fprintf('ROBUSTNESS CASE %d/%d: %s\n', k, numel(cases), c.name);
    fprintf('seed=%d | noiseScale=%.2f | dataFrac=%.2f\n', ...
        c.seed, c.noiseScale, c.dataFrac);
    fprintf('============================================================\n');

    cfg = emhd1d_defaultConfig();

    cfg.rngSeed = c.seed;
    cfg.pinn.seed = c.seed;

    cfg.data.noiseU = cfg.data.noiseU * c.noiseScale;
    cfg.data.noiseC = cfg.data.noiseC * c.noiseScale;
    cfg.data.noiseT = cfg.data.noiseT * c.noiseScale;

    cfg.data.nObs = max(100, round(cfg.data.nObs * c.dataFrac));
    cfg.data.nVal = max(100, round(cfg.data.nVal * c.dataFrac));
    cfg.data.nCol = max(500, round(cfg.data.nCol * c.dataFrac));
    cfg.data.nObsForPhys = max(500, round(cfg.data.nObsForPhys * c.dataFrac));

    outDir = fullfile('robustness_results', char(c.name));
    if ~exist(outDir,'dir')
        mkdir(outDir);
    end

    sol = emhd1d_generateSyntheticData(cfg);
    data = emhd1d_makeObservations(sol,cfg);
    pinn = emhd1d_trainGeometryPINN(sol,data,cfg,outDir);

    deltaHat = pinn.theta.delta;
    xtHat    = pinn.theta.xt;
    sigmaHat = pinn.theta.sigma;

    deltaTrue = cfg.geom.deltaTrue;
    xtTrue    = cfg.geom.xtTrue;
    sigmaTrue = cfg.geom.sigmaTrue;

    deltaErr = 100*abs(deltaHat - deltaTrue)/deltaTrue;
    xtErr    = 100*abs(xtHat - xtTrue)/xtTrue;
    sigmaErr = 100*abs(sigmaHat - sigmaTrue)/sigmaTrue;

    rows(k,:) = {char(c.name), c.seed, c.noiseScale, c.dataFrac, ...
        deltaHat, xtHat, sigmaHat, deltaErr, xtErr, sigmaErr};

    fprintf('\nCASE RESULT: %s\n', c.name);
    fprintf('delta: true %.8f | estimated %.8f | error %.4f %%\n', ...
        deltaTrue, deltaHat, deltaErr);
    fprintf('x_t:   true %.8f | estimated %.8f | error %.4f %%\n', ...
        xtTrue, xtHat, xtErr);
    fprintf('sigma: true %.8f | estimated %.8f | error %.4f %%\n', ...
        sigmaTrue, sigmaHat, sigmaErr);
end

resultsTable = cell2table(rows, ...
    'VariableNames', {'Case','Seed','NoiseScale','DataFraction', ...
    'DeltaHat','XtHat','SigmaHat', ...
    'DeltaErrorPct','XtErrorPct','SigmaErrorPct'});

disp(resultsTable);

fprintf('\n============================================================\n');
fprintf('ROBUSTNESS SUMMARY\n');
fprintf('============================================================\n');
fprintf('Mean delta error: %.4f %%\n', mean(resultsTable.DeltaErrorPct));
fprintf('Mean x_t error:   %.4f %%\n', mean(resultsTable.XtErrorPct));
fprintf('Mean sigma error: %.4f %%\n', mean(resultsTable.SigmaErrorPct));
fprintf('Max delta error:  %.4f %%\n', max(resultsTable.DeltaErrorPct));
fprintf('Max x_t error:    %.4f %%\n', max(resultsTable.XtErrorPct));
fprintf('Max sigma error:  %.4f %%\n', max(resultsTable.SigmaErrorPct));

writetable(resultsTable, fullfile('robustness_results','robustness_summary.csv'));
save(fullfile('robustness_results','robustness_summary.mat'), 'resultsTable');

fprintf('\nSaved robustness results under robustness_results/\n');

end
function report = emhd1d_checkIdentifiability(cfg, deltaTest, xtTest)
%EMHD1D_CHECKIDENTIFIABILITY Forward sensitivity / identifiability check.
% Diagnostic only. Compares the CN/FDM solution at the true geometry against
% a candidate geometry and reports relative L2 field differences.

if nargin < 2 || isempty(deltaTest); deltaTest = cfg.pinn.deltaInit; end
if nargin < 3 || isempty(xtTest);    xtTest    = cfg.pinn.xtInit;    end

fprintf('\n--- Forward sensitivity / identifiability check ---\n');
fprintf('True geometry:      delta=%.4f, x_t=%.4f, sigma=%.4f\n', ...
    cfg.geom.deltaTrue, cfg.geom.xtTrue, cfg.geom.sigmaTrue);
fprintf('Candidate geometry: delta=%.4f, x_t=%.4f, sigma=%.4f\n', ...
    deltaTest, xtTest, cfg.prior.sigmaKnown);

solTrue = emhd1d_generateSyntheticData(cfg);

cfgTest = cfg;
cfgTest.geom.deltaTrue = deltaTest;
cfgTest.geom.xtTrue    = xtTest;
cfgTest.geom.sigmaTrue = cfg.prior.sigmaKnown;
solTest = emhd1d_generateSyntheticData(cfgTest);

relU = norm(double(solTest.u(:)) - double(solTrue.u(:))) / norm(double(solTrue.u(:)));
relC = norm(double(solTest.C(:)) - double(solTrue.C(:))) / norm(double(solTrue.C(:)));
relT = norm(double(solTest.T(:)) - double(solTrue.T(:))) / norm(double(solTrue.T(:)));

fprintf('\nRelative L2 field differences (true vs candidate geometry):\n');
fprintf('  u: %7.4f %%   (nominal noise coefficient: %.4f %%)\n', 100*relU, 100*cfg.data.noiseU);
fprintf('  C: %7.4f %%   (nominal noise coefficient: %.4f %%)\n', 100*relC, 100*cfg.data.noiseC);
fprintf('  T: %7.4f %%   (nominal noise coefficient: %.4f %%)\n', 100*relT, 100*cfg.data.noiseT);

report.relU = relU;
report.relC = relC;
report.relT = relT;
report.solTrue = solTrue;
report.solTest = solTest;
report.deltaTest = deltaTest;
report.xtTest = xtTest;

bestSignal = max(relU, relC);
noiseFloor = max(cfg.data.noiseU, cfg.data.noiseC);
fprintf('\nSignal-to-noise heuristic (best of u,C vs largest noise coefficient): %.1fx\n', bestSignal/noiseFloor);
end

function fit = emhd1d_directGeometryFit(cfg, data)
%EMHD1D_DIRECTGEOMETRYFIT Direct FDM-based inverse check for delta and x_t.
% Diagnostic only. It is not called by run_all.

fprintf('\n--- Direct FDM inverse geometry check ---\n');

z0 = [ ...
    emhd1d_rawFromBound(cfg.pinn.deltaInit,cfg.pinn.deltaMin,cfg.pinn.deltaMax), ...
    emhd1d_rawFromBound(cfg.pinn.xtInit,cfg.pinn.xtMin,cfg.pinn.xtMax)];

obj = @(z) objective(z,cfg,data);

opts = optimset('Display','iter','MaxIter',80,'MaxFunEvals',200,'TolX',1e-4,'TolFun',1e-5);
[zBest,fBest] = fminsearch(obj,z0,opts);

deltaBest = sigmoidBound(zBest(1),cfg.pinn.deltaMin,cfg.pinn.deltaMax);
xtBest    = sigmoidBound(zBest(2),cfg.pinn.xtMin,cfg.pinn.xtMax);

fit.delta = deltaBest;
fit.xt = xtBest;
fit.sigma = cfg.prior.sigmaKnown;
fit.objective = fBest;

fprintf('\nDirect FDM inverse result:\n');
fprintf('  delta = %.8f\n',fit.delta);
fprintf('  x_t   = %.8f\n',fit.xt);
fprintf('  sigma = %.8f fixed\n',fit.sigma);
fprintf('  objective = %.6e\n',fit.objective);
end

function J = objective(z,cfg,data)
delta = sigmoidBound(z(1),cfg.pinn.deltaMin,cfg.pinn.deltaMax);
xt    = sigmoidBound(z(2),cfg.pinn.xtMin,cfg.pinn.xtMax);

cfgTest = cfg;
cfgTest.geom.deltaTrue = delta;
cfgTest.geom.xtTrue = xt;
cfgTest.geom.sigmaTrue = cfg.prior.sigmaKnown;
sol = emhd1d_generateSyntheticData(cfgTest);

xq = double(data.obs.X(1,:));
tq = double(data.obs.X(2,:));

uPred = interp2(double(sol.t), double(sol.x), double(sol.u), tq, xq, 'linear');
CPred = interp2(double(sol.t), double(sol.x), double(sol.C), tq, xq, 'linear');
TPred = interp2(double(sol.t), double(sol.x), double(sol.T), tq, xq, 'linear');

Y = double(data.obs.Y);

ru = (uPred - Y(1,:)) ./ cfg.pinn.scaleU;
rC = (CPred - Y(2,:)) ./ cfg.pinn.scaleC;
rT = (TPred - Y(3,:)) ./ cfg.pinn.scaleT;

J = mean(cfg.pinn.dataWeightU*ru.^2 + cfg.pinn.dataWeightC*rC.^2 + cfg.pinn.dataWeightT*rT.^2,'all');
if ~isfinite(J)
    J = 1e30;
end
fprintf('delta %.5f | xt %.5f | J %.4e\n',delta,xt,J);
end

function y = sigmoidBound(z,lo,hi)
s = 1./(1+exp(-z));
y = lo + (hi-lo).*s;
end

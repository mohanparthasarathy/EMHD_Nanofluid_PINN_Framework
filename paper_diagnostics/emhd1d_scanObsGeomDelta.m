function result = emhd1d_scanObsGeomDelta(cfg, sol, outDir, deltaGrid)
%EMHD1D_SCANOBSGEOMDELTA Scan the observation-anchored geometry residual vs delta.
%
% This diagnostic evaluates L_obs-geom(delta) using reference fields on the
% finite-difference grid rather than neural-network predictions. It answers:
%   "Which delta makes the geometry-dependent PDE residual smallest when
%    the state fields are held fixed?"
%
% Usage:
%   cfg = emhd1d_defaultConfig();
%   sol = emhd1d_generateSyntheticData(cfg);
%   result = emhd1d_scanObsGeomDelta(cfg, sol);
%
% Optional:
%   result = emhd1d_scanObsGeomDelta(cfg, sol, "results");
%   result = emhd1d_scanObsGeomDelta(cfg, sol, "results", linspace(0.05,0.65,301));

if nargin < 3 || isempty(outDir)
    outDir = pwd;
end
if ~exist(outDir,'dir')
    mkdir(outDir);
end
if nargin < 4 || isempty(deltaGrid)
    if isfield(cfg,'pinn') && isfield(cfg.pinn,'deltaMin') && isfield(cfg.pinn,'deltaMax')
        deltaGrid = linspace(cfg.pinn.deltaMin,cfg.pinn.deltaMax,301);
    else
        deltaGrid = linspace(0.05,0.65,301);
    end
end

% Fixed prior geometry components.
if isfield(cfg,'prior') && isfield(cfg.prior,'xtKnown')
    xt = cfg.prior.xtKnown;
elseif isfield(cfg,'geom') && isfield(cfg.geom,'xtTrue')
    xt = cfg.geom.xtTrue;
else
    error('Could not find fixed x_t in cfg.prior.xtKnown or cfg.geom.xtTrue.');
end

if isfield(cfg,'prior') && isfield(cfg.prior,'sigmaKnown')
    sigma = cfg.prior.sigmaKnown;
elseif isfield(cfg,'geom') && isfield(cfg.geom,'sigmaTrue')
    sigma = cfg.geom.sigmaTrue;
else
    error('Could not find fixed sigma in cfg.prior.sigmaKnown or cfg.geom.sigmaTrue.');
end

x = double(sol.x(:));
t = double(sol.t(:));
dx = x(2)-x(1);
dt = t(2)-t(1);

u = double(sol.u);
C = double(sol.C);
T = double(sol.T);

% Derivatives on the FDM grid.
ut  = zeros(size(u));
Ct  = zeros(size(C));
Cx  = zeros(size(C));
Cxx = zeros(size(C));
uxx = zeros(size(u));

ut(:,2:end-1) = (u(:,3:end)-u(:,1:end-2))/(2*dt);
Ct(:,2:end-1) = (C(:,3:end)-C(:,1:end-2))/(2*dt);

ut(:,1)   = (u(:,2)-u(:,1))/dt;
ut(:,end) = (u(:,end)-u(:,end-1))/dt;
Ct(:,1)   = (C(:,2)-C(:,1))/dt;
Ct(:,end) = (C(:,end)-C(:,end-1))/dt;

Cx(2:end-1,:)  = (C(3:end,:)-C(1:end-2,:))/(2*dx);
Cxx(2:end-1,:) = (C(3:end,:)-2*C(2:end-1,:)+C(1:end-2,:))/(dx^2);
uxx(2:end-1,:) = (u(3:end,:)-2*u(2:end-1,:)+u(1:end-2,:))/(dx^2);

% Neumann boundary second-derivative stencils.
Cx(1,:) = 0; Cx(end,:) = 0;
Cxx(1,:)   = 2*(C(2,:)-C(1,:))/(dx^2);
Cxx(end,:) = 2*(C(end-1,:)-C(end,:))/(dx^2);
uxx(1,:)   = 2*(u(2,:)-u(1,:))/(dx^2);
uxx(end,:) = 2*(u(end-1,:)-u(end,:))/(dx^2);

[Xg,Tg] = ndgrid(x,t);

% Avoid first and last time slice because time derivatives are one-sided there.
mask = true(size(Xg));
mask(:,1) = false;
mask(:,end) = false;

idxAll = find(mask(:));
rng(cfg.rngSeed + 202);
if isfield(cfg,'pinn') && isfield(cfg.pinn,'obsGeomN')
    n = min(cfg.pinn.obsGeomN,numel(idxAll));
else
    n = min(10000,numel(idxAll));
end
idx = idxAll(randperm(numel(idxAll),n));

xs   = Xg(idx);
us   = u(idx);
Cs   = C(idx);
Ts   = T(idx);
uts  = ut(idx);
Cts  = Ct(idx);
Cxs  = Cx(idx);
Cxxs = Cxx(idx);
uxxs = uxx(idx);

loss = zeros(size(deltaGrid));
lossU = zeros(size(deltaGrid));
lossC = zeros(size(deltaGrid));

for k = 1:numel(deltaGrid)
    delta = deltaGrid(k);

    h = cfg.geom.a + cfg.geom.b*cos(2*pi*xs) ...
        - delta.*exp(-((xs-xt).^2)./(sigma.^2));
    h = max(h,cfg.geom.hMinFloor);

    ku = cfg.phys.k0*(cfg.geom.h0./h).^cfg.phys.mUptake;

    Ru = uts ...
        - cfg.phys.nu*uxxs ...
        + (1/cfg.phys.rho)*cfg.phys.dpdx ...
        - cfg.phys.alphaE*cfg.phys.E0 ...
        + cfg.phys.alphaB*(cfg.phys.B0^2).*us ...
        + cfg.phys.alphaH*(us./h) ...
        - cfg.phys.betaB*(Ts-cfg.phys.T0);

    RC = Cts + us.*Cxs ...
        - cfg.phys.DC*Cxxs ...
        + ku.*Cs;

    lossU(k) = mean((Ru/cfg.pinn.scalePhysU).^2,'all');
    lossC(k) = mean((RC/cfg.pinn.scalePhysC).^2,'all');
    loss(k) = lossU(k) + lossC(k);
end

[bestLoss,bestIdx] = min(loss);
bestDelta = deltaGrid(bestIdx);

result.deltaGrid = deltaGrid;
result.loss = loss;
result.lossU = lossU;
result.lossC = lossC;
result.bestDelta = bestDelta;
result.bestLoss = bestLoss;
result.xt = xt;
result.sigma = sigma;

if isfield(cfg.geom,'deltaTrue')
    result.trueDelta = cfg.geom.deltaTrue;
    [~,trueIdx] = min(abs(deltaGrid-cfg.geom.deltaTrue));
    result.lossAtTrue = loss(trueIdx);
else
    result.trueDelta = NaN;
    result.lossAtTrue = NaN;
end

fprintf('\n============================================================\n');
fprintf('L_obs-geom(delta) diagnostic\n');
fprintf('============================================================\n');
fprintf('Fixed x_t = %.6f | fixed sigma = %.6f\n',xt,sigma);
fprintf('Minimum L_obs-geom at delta = %.8f | loss = %.6e\n',bestDelta,bestLoss);
if ~isnan(result.trueDelta)
    fprintf('True delta = %.8f | nearest-grid loss = %.6e\n',result.trueDelta,result.lossAtTrue);
    fprintf('Offset of L_obs-geom minimum from truth: %.4f percent of true delta\n', ...
        100*abs(bestDelta-result.trueDelta)/max(abs(result.trueDelta),eps));
end

fig = figure('Visible','off');
plot(deltaGrid,loss,'LineWidth',2); hold on;
plot(deltaGrid,lossU,'--','LineWidth',1.25);
plot(deltaGrid,lossC,'--','LineWidth',1.25);
xline(bestDelta,':','LineWidth',1.5);
if ~isnan(result.trueDelta)
    xline(result.trueDelta,'-.','LineWidth',1.5);
    legend('L_{obs-geom}','R_u contribution','R_C contribution','minimum','true \delta', ...
           'Location','best');
else
    legend('L_{obs-geom}','R_u contribution','R_C contribution','minimum', ...
           'Location','best');
end
xlabel('\delta');
ylabel('scaled residual loss');
title('Observation-anchored geometry residual scan');
grid on;
saveas(fig,fullfile(outDir,'obsgeom_delta_scan.png'));
save(fullfile(outDir,'obsgeom_delta_scan.mat'),'result');
close(fig);

fprintf('Saved diagnostic plot to: %s\n',fullfile(outDir,'obsgeom_delta_scan.png'));
fprintf('Saved diagnostic data to: %s\n',fullfile(outDir,'obsgeom_delta_scan.mat'));
end

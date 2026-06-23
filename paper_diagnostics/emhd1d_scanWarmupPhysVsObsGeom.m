function result = emhd1d_scanWarmupPhysVsObsGeom(cfg,sol,net,outDir,Xphys)
%EMHD1D_SCANWARMUPPHYSVSOBSGEOM Scan frozen-network physics loss and obs-geometry loss over delta.
%
% Usage:
%   cfg = emhd1d_defaultConfig();
%   sol = emhd1d_generateSyntheticData(cfg);
%   data = emhd1d_makeObservations(sol,cfg);
%   cfgWarm = cfg;
%   cfgWarm.pinn.phases(2).epochs = 0;
%   cfgWarm.pinn.phases(3).epochs = 0;
%   pinnWarm = emhd1d_trainGeometryPINN(sol,data,cfgWarm,'warmup_only');
%   result = emhd1d_scanWarmupPhysVsObsGeom(cfg,sol,pinnWarm.net,'results',data.phys.X);
%
% The residual evaluation is wrapped in dlfeval so the physics coordinates
% remain traced before dlgradient is called inside emhd1d_residualsAD.

if nargin < 4 || isempty(outDir)
    outDir = 'results';
end
if ~exist(outDir,'dir')
    mkdir(outDir);
end
if nargin < 5 || isempty(Xphys)
    rng(cfg.rngSeed+404);
    n = min(5000, cfg.data.nCol);
    x = cfg.geom.L*rand(1,n);
    t = cfg.time.tFinal*rand(1,n);
    Xphys = single([x;t]);
end

nScan = 151;
deltas = linspace(cfg.pinn.deltaMin,cfg.pinn.deltaMax,nScan);
lossPhys = zeros(size(deltas));
lossObsGeom = zeros(size(deltas));

obs = buildObsGeomSetForScan(sol,cfg);
XphysDL = dlarray(single(Xphys));

fprintf('\n============================================================\n');
fprintf('Frozen-network L_phys(delta) vs L_obs-geom(delta) diagnostic\n');
fprintf('============================================================\n');
fprintf('True delta = %.8f\n',cfg.geom.deltaTrue);
fprintf('Fixed x_t = %.8f | fixed sigma = %.8f\n',cfg.prior.xtKnown,cfg.prior.sigmaKnown);
fprintf('Scanning delta in [%.4f, %.4f] using %d points\n',deltas(1),deltas(end),nScan);

for k = 1:nScan
    delta = deltas(k);
    netTmp = net;
    netTmp.rawDelta = dlarray(single(emhd1d_rawFromBound(delta,cfg.pinn.deltaMin,cfg.pinn.deltaMax)));
    if isfield(netTmp,'rawXt')
        netTmp.rawXt = dlarray(single(emhd1d_rawFromBound(cfg.prior.xtKnown,cfg.pinn.xtMin,cfg.pinn.xtMax)));
    end
    if isfield(netTmp,'rawSigma')
        netTmp.rawSigma = dlarray(single(emhd1d_rawFromBound(cfg.prior.sigmaKnown,cfg.pinn.sigmaMin,cfg.pinn.sigmaMax)));
    end

    lossPhysDL = dlfeval(@frozenPhysLossAtDelta,netTmp,XphysDL,cfg);
    lossPhys(k) = gather(extractdata(lossPhysDL));
    lossObsGeom(k) = gather(extractdata(obsGeomLossAtDelta(netTmp,obs,cfg)));
end

[physMin, iPhys] = min(lossPhys);
[obsMin, iObs] = min(lossObsGeom);
[~, iTrue] = min(abs(deltas-cfg.geom.deltaTrue));

fprintf('\nMinimum frozen-network L_phys at delta = %.8f | loss = %.6e\n',deltas(iPhys),physMin);
fprintf('Minimum L_obs-geom at delta          = %.8f | loss = %.6e\n',deltas(iObs),obsMin);
fprintf('At true delta %.8f: L_phys = %.6e | L_obs-geom = %.6e\n', ...
    deltas(iTrue),lossPhys(iTrue),lossObsGeom(iTrue));

result.deltas = deltas;
result.lossPhys = lossPhys;
result.lossObsGeom = lossObsGeom;
result.deltaPhysMin = deltas(iPhys);
result.deltaObsGeomMin = deltas(iObs);
result.trueDelta = cfg.geom.deltaTrue;

fig = figure('Visible','off');
plot(deltas,lossPhys,'LineWidth',2); hold on;
plot(deltas,lossObsGeom,'LineWidth',2);
xline(cfg.geom.deltaTrue,'--','true \delta','LineWidth',1.5);
xline(deltas(iPhys),':','phys min','LineWidth',1.5);
xline(deltas(iObs),':','obs-geom min','LineWidth',1.5);
xlabel('\delta');
ylabel('loss');
title('Frozen-network L_{phys}(\delta) vs L_{obs-geom}(\delta)');
legend({'Frozen-network L_{phys}','L_{obs-geom}','true \delta','phys min','obs-geom min'},'Location','best');
grid on;
plotPath = fullfile(outDir,'phys_vs_obsgeom_delta_scan.png');
saveas(fig,plotPath);
close(fig);

matPath = fullfile(outDir,'phys_vs_obsgeom_delta_scan.mat');
save(matPath,'result');
fprintf('Saved diagnostic plot to: %s\n',plotPath);
fprintf('Saved diagnostic data to: %s\n',matPath);
end

function lossPhys = frozenPhysLossAtDelta(net,XphysDL,cfg)
[Ru,RC,RT] = emhd1d_residualsAD(net,XphysDL,cfg);
lossPhys = mean((Ru/cfg.pinn.scalePhysU).^2 + ...
                (RC/cfg.pinn.scalePhysC).^2 + ...
                (RT/cfg.pinn.scalePhysT).^2,'all');
end

function obs = buildObsGeomSetForScan(sol,cfg)
x = double(sol.x(:));
t = double(sol.t(:));
dx = x(2)-x(1);
dt = t(2)-t(1);

u = double(sol.u);
C = double(sol.C);
T = double(sol.T);

ut  = zeros(size(u));
Ct  = zeros(size(C));
Tt  = zeros(size(T));
Cx  = zeros(size(C));
Cxx = zeros(size(C));
uxx = zeros(size(u));

ut(:,2:end-1) = (u(:,3:end)-u(:,1:end-2))/(2*dt);
ut(:,1) = (u(:,2)-u(:,1))/dt;
ut(:,end) = (u(:,end)-u(:,end-1))/dt;

Ct(:,2:end-1) = (C(:,3:end)-C(:,1:end-2))/(2*dt);
Ct(:,1) = (C(:,2)-C(:,1))/dt;
Ct(:,end) = (C(:,end)-C(:,end-1))/dt;

Tt(:,2:end-1) = (T(:,3:end)-T(:,1:end-2))/(2*dt);
Tt(:,1) = (T(:,2)-T(:,1))/dt;
Tt(:,end) = (T(:,end)-T(:,end-1))/dt;

Cx(2:end-1,:) = (C(3:end,:)-C(1:end-2,:))/(2*dx);
Cx(1,:) = 0;
Cx(end,:) = 0;

Cxx(2:end-1,:) = (C(3:end,:)-2*C(2:end-1,:)+C(1:end-2,:))/(dx^2);
Cxx(1,:) = 2*(C(2,:)-C(1,:))/(dx^2);
Cxx(end,:) = 2*(C(end-1,:)-C(end,:))/(dx^2);

uxx(2:end-1,:) = (u(3:end,:)-2*u(2:end-1,:)+u(1:end-2,:))/(dx^2);
uxx(1,:) = 2*(u(2,:)-u(1,:))/(dx^2);
uxx(end,:) = 2*(u(end-1,:)-u(end,:))/(dx^2);

[Xg,Tg] = ndgrid(x,t);
mask = true(size(Xg));
mask(:,1) = false;
mask(:,end) = false;
idxAll = find(mask(:));
rng(cfg.rngSeed+77);
if isfield(cfg.pinn,'obsGeomN')
    n = min(cfg.pinn.obsGeomN,numel(idxAll));
else
    n = min(5000,numel(idxAll));
end
idx = idxAll(randperm(numel(idxAll),n));

obs.x = dlarray(single(Xg(idx)'));
obs.t = dlarray(single(Tg(idx)'));
obs.u = dlarray(single(u(idx)'));
obs.C = dlarray(single(C(idx)'));
obs.T = dlarray(single(T(idx)'));
obs.ut = dlarray(single(ut(idx)'));
obs.Ct = dlarray(single(Ct(idx)'));
obs.Tt = dlarray(single(Tt(idx)'));
obs.Cx = dlarray(single(Cx(idx)'));
obs.Cxx = dlarray(single(Cxx(idx)'));
obs.uxx = dlarray(single(uxx(idx)'));
end

function lossObsGeom = obsGeomLossAtDelta(net,obs,cfg)
theta = emhd1d_paramsFromNet(net,cfg);
h = emhd1d_hFromTheta(obs.x,theta,cfg);
ku = cfg.phys.k0*(cfg.geom.h0./h).^cfg.phys.mUptake;

Ru = obs.ut ...
     - cfg.phys.nu*obs.uxx ...
     + (1/cfg.phys.rho)*cfg.phys.dpdx ...
     - cfg.phys.alphaE*cfg.phys.E0 ...
     + cfg.phys.alphaB*(cfg.phys.B0^2).*obs.u ...
     + cfg.phys.alphaH*(obs.u./h) ...
     - cfg.phys.betaB*(obs.T-cfg.phys.T0);

RC = obs.Ct + obs.u.*obs.Cx ...
     - cfg.phys.DC*obs.Cxx ...
     + ku.*obs.C;

lossObsGeom = mean((Ru/cfg.pinn.scalePhysU).^2 + ...
                   (RC/cfg.pinn.scalePhysC).^2,'all');
end

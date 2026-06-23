function pinn = emhd1d_trainGeometryPINN(sol, data, cfg, outDir)
%EMHD1D_TRAINGEOMETRYPINN Staged inverse recovery of delta, x_t, and sigma.
% The geometry-fitting phase uses an observation-anchored residual evaluated
% on fixed fields reconstructed from noisy observations. Geometry gradients
% therefore come through h(x;theta_g) and k_u(h), rather than through
% higher-order derivatives of the neural surrogate.
if nargin < 4 || isempty(outDir); outDir = pwd; end
if ~exist(outDir,'dir'); mkdir(outDir); end
emhd1d_validateConfig(cfg);
rng(cfg.pinn.seed);

net = emhd1d_initMLP(cfg.pinn.layers, cfg);
obsGeom = buildObsGeomSetFromNoisyObservations(data,cfg);

avg = []; avgSq = [];
hist = initHistory(sum([cfg.pinn.phases.epochs]));
iter = 0;

fprintf('PINN initial geometry: delta=%.4f, x_t=%.4f, sigma=%.4f\n', ...
    cfg.pinn.deltaInit, cfg.pinn.xtInit, cfg.pinn.sigmaInit);

for p = 1:numel(cfg.pinn.phases)
    phase = cfg.pinn.phases(p);
    fprintf('\n[Phase %d/%d] %s | epochs=%d | train [delta xt sigma]=[%d %d %d]\n', ...
        p, numel(cfg.pinn.phases), phase.name, phase.epochs, phase.trainDelta, phase.trainXt, phase.trainSigma);
    for ep = 1:phase.epochs
        iter = iter + 1;
        wPhys = phase.wPhysStart + (phase.wPhysEnd-phase.wPhysStart)*min(1,(ep-1)/max(1,phase.epochs-1));
        if isfield(data,'phys') && isfield(data.phys,'X')
            XphysDL = dlarray(single(data.phys.X));
        else
            XphysDL = dlarray(single(data.col.X));
        end
        
        [loss, grad, metrics] = dlfeval(@emhd1d_loss, net, data, cfg, phase, wPhys, XphysDL, obsGeom);
        [net, avg, avgSq] = emhd1d_adamUpdate(net, grad, avg, avgSq, iter, cfg, phase);

        hist.iter(iter) = iter;
        hist.phase(iter) = p;
        hist.total(iter) = metrics.total;
        hist.data(iter) = metrics.data;
        hist.ic(iter) = metrics.ic;
        hist.bc(iter) = metrics.bc;
        hist.phys(iter) = metrics.phys;
        hist.obsGeom(iter) = metrics.obsGeom;
        hist.wPhys(iter) = wPhys;
        hist.delta(iter) = metrics.delta;
        hist.xt(iter) = metrics.xt;
        hist.sigma(iter) = metrics.sigma;

        if mod(ep,cfg.pinn.printEvery)==0 || ep==1 || ep==phase.epochs
            fprintf('iter %5d | %-18s | loss %.3e | data %.2e phys %.2e obsGeom %.2e | delta %.5f x_t %.5f sigma %.5f\n', ...
                iter, phase.name, metrics.total, metrics.data, metrics.phys, metrics.obsGeom, metrics.delta, metrics.xt, metrics.sigma);
        end
    end
end

pinn.net = net;
pinn.history = trimHistory(hist, iter);
pinn.cfg = cfg;
pinn.theta = gatherTheta(net,cfg);
save(fullfile(outDir,'pinn_model.mat'),'pinn','-v7.3');
emhd1d_plotTrainingHistory(pinn, cfg, outDir);
end

function [loss, grad, metrics] = emhd1d_loss(net, data, cfg, phase, wPhys, XphysDL, obsGeom)
Xd = dlarray(emhd1d_normalizeInput(data.obs.X,cfg));
Yd = dlarray(data.obs.Y);
Yhat = emhd1d_forwardMLP(net, Xd, cfg);
lossData = mean(cfg.pinn.dataWeightU*((Yhat(1,:)-Yd(1,:))/cfg.pinn.scaleU).^2 + ...
                cfg.pinn.dataWeightC*((Yhat(2,:)-Yd(2,:))/cfg.pinn.scaleC).^2 + ...
                cfg.pinn.dataWeightT*((Yhat(3,:)-Yd(3,:))/cfg.pinn.scaleT).^2,'all');

Xi = dlarray(emhd1d_normalizeInput(data.ic.X,cfg));
Yi = dlarray(data.ic.Y);
Yhi = emhd1d_forwardMLP(net, Xi, cfg);
lossIC = mean(0.5*((Yhi(1,:)-Yi(1,:))/cfg.pinn.scaleU).^2 + ...
              2.0*((Yhi(2,:)-Yi(2,:))/cfg.pinn.scaleC).^2 + ...
              0.5*((Yhi(3,:)-Yi(3,:))/cfg.pinn.scaleT).^2,'all');

Xb = dlarray(emhd1d_normalizeInput(data.bc.X,cfg));
Yb = dlarray(data.bc.Y);
Yhb = emhd1d_forwardMLP(net, Xb, cfg);
lossBC = mean(0.5*((Yhb(1,:)-Yb(1,:))/cfg.pinn.scaleU).^2 + ...
              2.0*((Yhb(2,:)-Yb(2,:))/cfg.pinn.scaleC).^2 + ...
              0.5*((Yhb(3,:)-Yb(3,:))/cfg.pinn.scaleT).^2,'all');

if wPhys > 0
    [Ru,RC,RT] = emhd1d_residualsAD(net, XphysDL, cfg);

    lossPhys = mean((Ru/cfg.pinn.scalePhysU).^2 + ...
                    (RC/cfg.pinn.scalePhysC).^2 + ...
                    (RT/cfg.pinn.scalePhysT).^2,'all');
else
    lossPhys = dlarray(single(0));
end

if isfield(phase,'wObsGeom')
    wObsGeom = phase.wObsGeom;
else
    wObsGeom = 0.0;
end

if wObsGeom > 0
    lossObsGeom = observationGeometryLoss(net,obsGeom,cfg);
else
    lossObsGeom = dlarray(single(0));
end

lossReg = dlarray(single(0));
for k = 1:numel(net.W)
    lossReg = lossReg + mean(net.W{k}.^2,'all');
end

loss = phase.wData*lossData + phase.wIC*lossIC + phase.wBC*lossBC ...
     + wPhys*lossPhys ...
     + cfg.pinn.wObsGeom*wObsGeom*lossObsGeom ...
     + cfg.pinn.wReg*lossReg;

grad = dlgradient(loss, net);

theta = emhd1d_paramsFromNet(net,cfg);
metrics.total = gather(extractdata(loss));
metrics.data = gather(extractdata(lossData));
metrics.ic = gather(extractdata(lossIC));
metrics.bc = gather(extractdata(lossBC));
metrics.phys = gather(extractdata(lossPhys));
metrics.obsGeom = gather(extractdata(lossObsGeom));
metrics.delta = gather(extractdata(theta.delta));
metrics.xt = gather(extractdata(theta.xt));
metrics.sigma = gather(extractdata(theta.sigma));
end

function obs = buildObsGeomSetFromNoisyObservations(data,cfg)
% Construct observation-derived fields from sparse noisy measurements only.
% The clean finite-difference solution is not used in this geometry loss.

xObs = double(data.obs.X(1,:));
tObs = double(data.obs.X(2,:));
uObs = double(data.obs.Y(1,:));
CObs = double(data.obs.Y(2,:));
TObs = double(data.obs.Y(3,:));

Nx = cfg.grid.Nx;
Nt = cfg.grid.Nt;

x = linspace(0,cfg.geom.L,Nx);
t = linspace(0,cfg.time.tFinal,Nt);
[Xg,Tg] = ndgrid(x,t);

Fu = scatteredInterpolant(xObs(:),tObs(:),uObs(:),'natural','nearest');
FC = scatteredInterpolant(xObs(:),tObs(:),CObs(:),'natural','nearest');
FT = scatteredInterpolant(xObs(:),tObs(:),TObs(:),'natural','nearest');

u = Fu(Xg,Tg);
C = FC(Xg,Tg);
T = FT(Xg,Tg);

% Apply a separable three-point moving average before finite-difference differentiation.
u = movmean(movmean(u,3,1),3,2);
C = movmean(movmean(C,3,1),3,2);
T = movmean(movmean(T,3,1),3,2);

dx = x(2)-x(1);
dt = t(2)-t(1);

ut  = zeros(size(u));
Ct  = zeros(size(C));
Tt  = zeros(size(T));
Cx  = zeros(size(C));
Cxx = zeros(size(C));
uxx = zeros(size(u));

ut(:,2:end-1) = (u(:,3:end)-u(:,1:end-2))/(2*dt);
Ct(:,2:end-1) = (C(:,3:end)-C(:,1:end-2))/(2*dt);
Tt(:,2:end-1) = (T(:,3:end)-T(:,1:end-2))/(2*dt);

ut(:,1) = (u(:,2)-u(:,1))/dt;
ut(:,end) = (u(:,end)-u(:,end-1))/dt;

Ct(:,1) = (C(:,2)-C(:,1))/dt;
Ct(:,end) = (C(:,end)-C(:,end-1))/dt;

Tt(:,1) = (T(:,2)-T(:,1))/dt;
Tt(:,end) = (T(:,end)-T(:,end-1))/dt;

Cx(2:end-1,:) = (C(3:end,:)-C(1:end-2,:))/(2*dx);
Cxx(2:end-1,:) = (C(3:end,:)-2*C(2:end-1,:)+C(1:end-2,:))/(dx^2);
uxx(2:end-1,:) = (u(3:end,:)-2*u(2:end-1,:)+u(1:end-2,:))/(dx^2);

Cx(1,:) = 0;
Cx(end,:) = 0;
Cxx(1,:) = 2*(C(2,:)-C(1,:))/(dx^2);
Cxx(end,:) = 2*(C(end-1,:)-C(end,:))/(dx^2);
uxx(1,:) = 2*(u(2,:)-u(1,:))/(dx^2);
uxx(end,:) = 2*(u(end-1,:)-u(end,:))/(dx^2);

mask = true(size(Xg));
mask(:,1) = false;
mask(:,end) = false;

idxAll = find(mask(:));
rng(cfg.rngSeed+77);
n = min(cfg.pinn.obsGeomN,numel(idxAll));
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

function lossObsGeom = observationGeometryLoss(net,obs,cfg)
%OBSERVATIONGEOMETRYLOSS Geometry residual evaluated on observation-derived fields.
% The geometry parameters enter only through h(x;theta_g) and k_u(h); no
% direct comparison with the synthetic truth is used during optimization.

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

function H = initHistory(n)
fields = {'iter','phase','total','data','ic','bc','phys','obsGeom','wPhys','delta','xt','sigma'};
for k = 1:numel(fields); H.(fields{k}) = zeros(n,1); end
end

function H = trimHistory(H,n)
fields = fieldnames(H);
for k = 1:numel(fields); H.(fields{k}) = H.(fields{k})(1:n); end
end

function th = gatherTheta(net,cfg)
theta = emhd1d_paramsFromNet(net,cfg);
th.delta = gather(extractdata(theta.delta));
th.xt = gather(extractdata(theta.xt));
th.sigma = gather(extractdata(theta.sigma));
end

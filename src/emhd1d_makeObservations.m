function data = emhd1d_makeObservations(sol, cfg)
%EMHD1D_MAKEOBSERVATIONS Sparse noisy measurements and PINN training sets.
rng(cfg.rngSeed+3);

x = double(sol.x);
t = double(sol.t);
Nx = numel(x);
Nt = numel(t);
[Xg,Tg] = ndgrid(x,t);

nObs = min(cfg.data.nObs, Nx*Nt);
xo = cfg.geom.L*rand(1,nObs);
to = cfg.time.tFinal*rand(1,nObs);
[uo,Co,To] = sampleGrid(sol,xo,to);

uo = uo + cfg.data.noiseU*std(double(sol.u(:)))*randn(size(uo),'single');
Co = Co + cfg.data.noiseC*std(double(sol.C(:)))*randn(size(Co),'single');
To = To + cfg.data.noiseT*std(double(sol.T(:)))*randn(size(To),'single');

nVal = min(cfg.data.nVal, Nx*Nt);
idxv = randperm(Nx*Nt, nVal);

nCol = cfg.data.nCol;
hfd = cfg.geom.L/(cfg.grid.Nx-1);
xc = hfd + (cfg.geom.L-2*hfd)*rand(1,nCol);
tc = cfg.time.tFinal*rand(1,nCol);

nIC = cfg.data.nIC;
xic = linspace(0,cfg.geom.L,nIC);
tic = zeros(1,nIC);
uic = reshape(interp1(x,double(sol.u(:,1)),xic,'linear'),1,[]);
Cic = reshape(interp1(x,double(sol.C(:,1)),xic,'linear'),1,[]);
Tic = reshape(interp1(x,double(sol.T(:,1)),xic,'linear'),1,[]);

nBC = cfg.data.nBC;
tb = cfg.time.tFinal*rand(1,nBC);
xb = [zeros(1,ceil(nBC/2)), cfg.geom.L*ones(1,floor(nBC/2))];
tb = tb(1:numel(xb));
[ub,Cb,Tb] = sampleGrid(sol,xb,tb);

data.obs.X = single([xo;to]);
data.obs.Y = single([uo;Co;To]);

data.val.X = single([reshape(Xg(idxv),1,[]); reshape(Tg(idxv),1,[])]);
data.val.Y = single([reshape(double(sol.u(idxv)),1,[]); ...
                     reshape(double(sol.C(idxv)),1,[]); ...
                     reshape(double(sol.T(idxv)),1,[])]);

data.col.X = single([xc;tc]);
data.ic.X = single([xic;tic]);
data.ic.Y = single([uic;Cic;Tic]);
data.bc.X = single([xb;tb]);
data.bc.Y = single([reshape(ub,1,[]); reshape(Cb,1,[]); reshape(Tb,1,[])]);

nObsPhys = min(cfg.data.nObsForPhys,size(data.obs.X,2));
idxPhysObs = randperm(size(data.obs.X,2),nObsPhys);
data.phys.X = single([data.col.X data.obs.X(:,idxPhysObs)]);

fprintf('Synthetic observations: data=%d, collocation=%d, obs-physics=%d, IC=%d, BC=%d, validation=%d\n', ...
    size(data.obs.X,2), size(data.col.X,2), nObsPhys, ...
    size(data.ic.X,2), size(data.bc.X,2), size(data.val.X,2));
end

function [u,C,T] = sampleGrid(sol,xq,tq)
x = double(sol.x);
t = double(sol.t);
u = interp2(t,x,double(sol.u),tq,xq,'linear');
C = interp2(t,x,double(sol.C),tq,xq,'linear');
T = interp2(t,x,double(sol.T),tq,xq,'linear');
u = single(reshape(u,1,[]));
C = single(reshape(C,1,[]));
T = single(reshape(T,1,[]));
end

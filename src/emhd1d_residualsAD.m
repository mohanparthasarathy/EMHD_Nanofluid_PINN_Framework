function [Ru,RC,RT] = emhd1d_residualsAD(net, Xphys, cfg)
%EMHD1D_RESIDUALSAD PINN residuals using automatic differentiation.

if isa(Xphys,'dlarray')
    X = Xphys;
else
    X = dlarray(single(Xphys));
end

Y = localForwardRaw(net,X,cfg);
u = Y(1,:);
C = Y(2,:);
T = Y(3,:);

theta = emhd1d_paramsFromNet(net,cfg);
h = emhd1d_hFromTheta(X(1,:),theta,cfg);
ku = cfg.phys.k0*(cfg.geom.h0./h).^cfg.phys.mUptake;

du = dlgradient(sum(u,'all'),X,'EnableHigherDerivatives',true);
dC = dlgradient(sum(C,'all'),X,'EnableHigherDerivatives',true);
dT = dlgradient(sum(T,'all'),X,'EnableHigherDerivatives',true);

ut = du(2,:);
Cx = dC(1,:);
Ct = dC(2,:);
Tx = dT(1,:);
Tt = dT(2,:);

ux = du(1,:);
dUx = dlgradient(sum(ux,'all'),X,'EnableHigherDerivatives',true);
uxx = dUx(1,:);

dCx = dlgradient(sum(Cx,'all'),X,'EnableHigherDerivatives',true);
Cxx = dCx(1,:);

dTx = dlgradient(sum(Tx,'all'),X,'EnableHigherDerivatives',true);
Txx = dTx(1,:);

Ru = ut ...
     - cfg.phys.nu*uxx ...
     + (1/cfg.phys.rho)*cfg.phys.dpdx ...
     - cfg.phys.alphaE*cfg.phys.E0 ...
     + cfg.phys.alphaB*(cfg.phys.B0^2).*u ...
     + cfg.phys.alphaH*(u./h) ...
     - cfg.phys.betaB*(T-cfg.phys.T0);

RC = Ct + u.*Cx ...
     - cfg.phys.DC*Cxx ...
     + ku.*C;

RT = Tt + u.*Tx ...
     - cfg.phys.kappa*Txx ...
     - cfg.phys.gammaE*(cfg.phys.E0^2) ...
     - cfg.phys.gammaB*(cfg.phys.B0^2).*(u.^2);

end

function Y = localForwardRaw(net,X,cfg)
% Normalize raw physical coordinates while preserving the computation graph.
Xn = [X(1,:)./cfg.geom.L; X(2,:)./cfg.time.tFinal];
Y = emhd1d_forwardMLP(net,Xn,cfg);
end
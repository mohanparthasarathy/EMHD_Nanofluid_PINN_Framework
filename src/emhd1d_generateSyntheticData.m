function sol = emhd1d_generateSyntheticData(cfg)
%EMHD1D_GENERATESYNTHETICDATA Crank-Nicolson FDM solve of the 1D model.
% Linear diffusion, damping, uptake, advection, and source coupling are
% evaluated at the temporal midpoint. The nonlinear coupling is resolved by
% Picard iteration inside each time step.
rng(cfg.rngSeed);
Nx = cfg.grid.Nx; Nt = cfg.grid.Nt;
L = cfg.geom.L; tf = cfg.time.tFinal;
x = linspace(0,L,Nx);
t = linspace(0,tf,Nt);
dx = x(2)-x(1); dt = t(2)-t(1);

h = emhd1d_geometry(x, cfg.geom.deltaTrue, cfg.geom.xtTrue, cfg.geom.sigmaTrue, cfg);
ku = emhd1d_ku(h, cfg);

u = zeros(Nx,Nt,'single');
C = zeros(Nx,Nt,'single');
T = zeros(Nx,Nt,'single');

u(:,1) = single(cfg.ic.uBase + cfg.ic.uAmp*cos(2*pi*x));
C(:,1) = single(cfg.ic.CBase + cfg.ic.CAmp*exp(-((x-cfg.ic.CCenter).^2)/(cfg.ic.CWidth^2)));
T(:,1) = single(cfg.ic.TBase + 0*x);

Lxx = neumannLaplacian(Nx,dx);
D1 = neumannFirstDerivative(Nx,dx);
I = speye(Nx);

dampU = cfg.phys.alphaB*(cfg.phys.B0^2) + cfg.phys.alphaH./h(:);
Lu = cfg.phys.nu*Lxx - spdiags(dampU,0,Nx,Nx);
AuLeft = I - 0.5*dt*Lu;
AuRight = I + 0.5*dt*Lu;

maxPicard = 8;
picardTol = 1.0e-10;

fprintf('FDM grid: Nx=%d Nt=%d dx=%.4g dt=%.4g | Crank-Nicolson/Picard\n',Nx,Nt,dx,dt);

for n = 1:Nt-1
    un = double(u(:,n));
    Cn = double(C(:,n));
    Tn = double(T(:,n));

    uNext = un;
    CNext = Cn;
    TNext = Tn;

    for q = 1:maxPicard
        uOld = uNext;
        COld = CNext;
        TOld = TNext;

        Tmid = 0.5*(Tn + TNext);
        sourceU = -(1/cfg.phys.rho)*cfg.phys.dpdx + cfg.phys.alphaE*cfg.phys.E0 + ...
                  cfg.phys.betaB*(Tmid - cfg.phys.T0);
        uNext = AuLeft \ (AuRight*un + dt*sourceU);
        umid = 0.5*(un + uNext);

        LC = cfg.phys.DC*Lxx - spdiags(umid,0,Nx,Nx)*D1 - spdiags(ku(:),0,Nx,Nx);
        CNext = (I - 0.5*dt*LC) \ ((I + 0.5*dt*LC)*Cn);
        CNext = max(CNext,0);

        LT = cfg.phys.kappa*Lxx - spdiags(umid,0,Nx,Nx)*D1;
        sourceT = cfg.phys.gammaE*(cfg.phys.E0^2) + cfg.phys.gammaB*(cfg.phys.B0^2)*(umid.^2);
        TNext = (I - 0.5*dt*LT) \ ((I + 0.5*dt*LT)*Tn + dt*sourceT);

        change = max([norm(uNext-uOld,inf), norm(CNext-COld,inf), norm(TNext-TOld,inf)]);
        if change < picardTol
            break
        end
    end

    u(:,n+1) = single(uNext(:));
    C(:,n+1) = single(CNext(:));
    T(:,n+1) = single(TNext(:));

    if any(~isfinite(uNext)) || any(~isfinite(CNext)) || any(~isfinite(TNext))
        error('FDM solve became non-finite at step %d. Reduce dt or coefficients.', n);
    end
end

fprintf('FDM complete. Ranges: u=[%.4g %.4g], C=[%.4g %.4g], T=[%.4g %.4g]\n', ...
    min(u(:)), max(u(:)), min(C(:)), max(C(:)), min(T(:)), max(T(:)));

sol.x = single(x); sol.t = single(t); sol.h = single(h); sol.ku = single(ku);
sol.u = u; sol.C = C; sol.T = T;
sol.true.delta = cfg.geom.deltaTrue; sol.true.xt = cfg.geom.xtTrue; sol.true.sigma = cfg.geom.sigmaTrue;
sol.fdm.method = 'Crank-Nicolson finite difference with Picard iteration';
sol.fdm.dx = dx; sol.fdm.dt = dt;
end

function L = neumannLaplacian(N,dx)
e = ones(N,1);
L = spdiags([e -2*e e],[-1 0 1],N,N)/(dx^2);
L(1,:) = 0; L(1,1) = -2/(dx^2); L(1,2) = 2/(dx^2);
L(end,:) = 0; L(end,end) = -2/(dx^2); L(end,end-1) = 2/(dx^2);
end

function D = neumannFirstDerivative(N,dx)
e = ones(N,1);
D = spdiags([-e e],[-1 1],N,N)/(2*dx);
D(1,:) = 0;
D(end,:) = 0;
end

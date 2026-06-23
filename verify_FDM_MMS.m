%% verify_FDM_MMS.m
% Method of Manufactured Solutions verification for the 1D EMHD
% Crank-Nicolson/Picard finite-difference solver.

clear; close all; clc;

fprintf('\n============================================================\n');
fprintf('MMS verification for 1D EMHD Crank-Nicolson FDM solver\n');
fprintf('============================================================\n');

levels = [41 81 161 321];

errU = zeros(size(levels));
errC = zeros(size(levels));
errT = zeros(size(levels));
dxs  = zeros(size(levels));

for r = 1:numel(levels)
    Nx = levels(r);
    Nt = 4*(Nx-1) + 1;

    [errU(r),errC(r),errT(r),dxs(r)] = runOneMMS(Nx,Nt);

    fprintf('Nx=%4d Nt=%5d dx=%.3e | err u=%.3e C=%.3e T=%.3e\n', ...
        Nx,Nt,dxs(r),errU(r),errC(r),errT(r));
end

rateU = log(errU(1:end-1)./errU(2:end))./log(dxs(1:end-1)./dxs(2:end));
rateC = log(errC(1:end-1)./errC(2:end))./log(dxs(1:end-1)./dxs(2:end));
rateT = log(errT(1:end-1)./errT(2:end))./log(dxs(1:end-1)./dxs(2:end));

fprintf('\nObserved convergence rates:\n');
for r = 1:numel(rateU)
    fprintf('level %d -> %d | u %.2f | C %.2f | T %.2f\n', ...
        r,r+1,rateU(r),rateC(r),rateT(r));
end

figure;
loglog(dxs,errU,'-o','LineWidth',1.5); hold on;
loglog(dxs,errC,'-s','LineWidth',1.5);
loglog(dxs,errT,'-^','LineWidth',1.5);
set(gca,'XDir','reverse');
grid on;
xlabel('\Delta x');
ylabel('relative L^2 error');
legend('u','C','T','Location','best');
title('MMS convergence of Crank-Nicolson FDM solver');

function [relU,relC,relT,dx] = runOneMMS(Nx,Nt)

L = 1.0;
tf = 0.25;

x = linspace(0,L,Nx)';
t = linspace(0,tf,Nt);

dx = x(2)-x(1);
dt = t(2)-t(1);

nu = 2.0e-3;
DC = 1.2e-3;
kappa = 1.6e-3;

rho = 1.0;
dpdx = -0.20;
alphaE = 0.075;
E0 = 1.0;
alphaB = 0.10;
B0 = 1.0;
alphaH = 0.055;
betaB = 0.08;
Tref = 1.0;

k0 = 0.060;
m = 1.5;
h0 = 1.0;

gammaE = 1.5e-3;
gammaB = 1.0e-3;

a = 1.0;
b = 0.10;
delta = 0.36;
xt = 0.58;
sigma = 0.105;

h = a + b*cos(2*pi*x) - delta*exp(-((x-xt).^2)/(sigma^2));
ku = k0*(h0./h).^m;

D1 = neumannD1(Nx,dx);
Lxx = neumannLxx(Nx,dx);
I = speye(Nx);

dampU = alphaB*B0^2 + alphaH./h;
Lu = nu*Lxx - spdiags(dampU,0,Nx,Nx);

AuL = I - 0.5*dt*Lu;
AuR = I + 0.5*dt*Lu;

uNum = zeros(Nx,Nt);
CNum = zeros(Nx,Nt);
TNum = zeros(Nx,Nt);

[u0,C0,T0] = exactFields(x,0);
uNum(:,1) = u0;
CNum(:,1) = C0;
TNum(:,1) = T0;

maxPicard = 12;
picardTol = 1e-12;

for n = 1:Nt-1
    tn = t(n);
    tnp1 = t(n+1);
    tm = 0.5*(tn+tnp1);

    un = uNum(:,n);
    Cn = CNum(:,n);
    Tn = TNum(:,n);

    uNext = un;
    CNext = Cn;
    TNext = Tn;

    for q = 1:maxPicard
        uOld = uNext;
        COld = CNext;
        TOld = TNext;

        Tmid = 0.5*(Tn + TNext);

        Su = sourceU(x,tm,h,nu,rho,dpdx,alphaE,E0,alphaB,B0,alphaH,betaB,Tref);

        driveU = -(1/rho)*dpdx + alphaE*E0 + betaB*(Tmid-Tref);
        uNext = AuL \ (AuR*un + dt*(driveU + Su));

        umid = 0.5*(un + uNext);

        LC = DC*Lxx - spdiags(umid,0,Nx,Nx)*D1 - spdiags(ku,0,Nx,Nx);
        SC = sourceC(x,tm,DC,ku);
        CNext = (I - 0.5*dt*LC) \ ((I + 0.5*dt*LC)*Cn + dt*SC);

        LT = kappa*Lxx - spdiags(umid,0,Nx,Nx)*D1;
        heat = gammaE*E0^2 + gammaB*B0^2*(umid.^2);
        ST = sourceT(x,tm,kappa,gammaE,E0,gammaB,B0);
        TNext = (I - 0.5*dt*LT) \ ((I + 0.5*dt*LT)*Tn + dt*(heat + ST));

        change = max([norm(uNext-uOld,inf), norm(CNext-COld,inf), norm(TNext-TOld,inf)]);
        if change < picardTol
            break;
        end
    end

    uNum(:,n+1) = uNext;
    CNum(:,n+1) = CNext;
    TNum(:,n+1) = TNext;
end

[Ue,Ce,Te] = exactGrid(x,t);

relU = norm(uNum(:)-Ue(:))/norm(Ue(:));
relC = norm(CNum(:)-Ce(:))/norm(Ce(:));
relT = norm(TNum(:)-Te(:))/norm(Te(:));

end

function [u,C,T] = exactFields(x,t)

u = 0.25 + 0.04*cos(pi*x).*exp(-t) + 0.02*cos(2*pi*x).*cos(2*t);
C = 0.03 + 0.01*cos(pi*x).*exp(-0.7*t) + 0.006*cos(2*pi*x).*sin(t);
T = 1.0 + 0.01*cos(pi*x).*exp(-0.5*t) + 0.004*cos(2*pi*x).*cos(t);

end

function [U,C,T] = exactGrid(x,t)
U = zeros(numel(x),numel(t));
C = U;
T = U;
for j = 1:numel(t)
    [U(:,j),C(:,j),T(:,j)] = exactFields(x,t(j));
end
end

function Su = sourceU(x,t,h,nu,rho,dpdx,alphaE,E0,alphaB,B0,alphaH,betaB,Tref)
[u,~,T] = exactFields(x,t);
ut = -0.04*cos(pi*x).*exp(-t) - 0.04*cos(2*pi*x).*sin(2*t);
uxx = -0.04*pi^2*cos(pi*x).*exp(-t) - 0.08*pi^2*cos(2*pi*x).*cos(2*t);
Su = ut - nu*uxx + (alphaB*B0^2 + alphaH./h).*u + (1/rho)*dpdx - alphaE*E0 - betaB*(T-Tref);
end

function SC = sourceC(x,t,DC,ku)
[u,C,~] = exactFields(x,t);
Ct = -0.007*cos(pi*x).*exp(-0.7*t) + 0.006*cos(2*pi*x).*cos(t);
Cx = -0.01*pi*sin(pi*x).*exp(-0.7*t) - 0.012*pi*sin(2*pi*x).*sin(t);
Cxx = -0.01*pi^2*cos(pi*x).*exp(-0.7*t) - 0.024*pi^2*cos(2*pi*x).*sin(t);
SC = Ct + u.*Cx - DC*Cxx + ku.*C;
end

function ST = sourceT(x,t,kappa,gammaE,E0,gammaB,B0)
[u,~,T] = exactFields(x,t);
Tt = -0.005*cos(pi*x).*exp(-0.5*t) - 0.004*cos(2*pi*x).*sin(t);
Tx = -0.01*pi*sin(pi*x).*exp(-0.5*t) - 0.008*pi*sin(2*pi*x).*cos(t);
Txx = -0.01*pi^2*cos(pi*x).*exp(-0.5*t) - 0.016*pi^2*cos(2*pi*x).*cos(t);
ST = Tt + u.*Tx - kappa*Txx - gammaE*E0^2 - gammaB*B0^2*(u.^2);
end

function D = neumannD1(N,dx)
e = ones(N,1);
D = spdiags([-e e],[-1 1],N,N)/(2*dx);
D(1,:) = 0;
D(end,:) = 0;
end

function L = neumannLxx(N,dx)
e = ones(N,1);
L = spdiags([e -2*e e],[-1 0 1],N,N)/(dx^2);
L(1,1) = -2/dx^2;
L(1,2) = 2/dx^2;
L(N,N) = -2/dx^2;
L(N,N-1) = 2/dx^2;
end

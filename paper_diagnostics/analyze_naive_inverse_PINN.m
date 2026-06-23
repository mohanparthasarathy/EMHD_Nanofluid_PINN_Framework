%ANALYZE_NAIVE_INVERSE_PINN Compute naive inverse-PINN diagnostics from saved outputs.
%
% This script reads the output of paper_diagnostics/run_naive_inverse_PINN_baseline.m,
% regenerates the same synthetic reference data, computes value-level and derivative
% errors, evaluates an oracle residual along the naive geometry trajectory, and writes
% reproducible diagnostics for the manuscript.
%
% Usage from the repository root:
%   run('paper_diagnostics/run_naive_inverse_PINN_baseline.m')   % if not already run
%   run('paper_diagnostics/analyze_naive_inverse_PINN.m')
%
% Outputs:
%   paper_diagnostics/naive_inverse_PINN_results/naive_pinn_failure_summary.csv
%   paper_diagnostics/naive_inverse_PINN_results/naive_oracle_trajectory.csv
%   paper_figures/fig05_naive_derivative_error_bar.png
%   paper_figures/fig05_naive_derivative_error_bar.pdf

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir); scriptDir = pwd; end
if exist(fullfile(scriptDir,'src'),'dir')
    rootDir = scriptDir;
else
    rootDir = fileparts(scriptDir);
end
addpath(fullfile(rootDir,'src'));

outDir = fullfile(rootDir,'paper_diagnostics','naive_inverse_PINN_results');
figDir = fullfile(rootDir,'paper_figures');
if ~exist(outDir,'dir'); mkdir(outDir); end
if ~exist(figDir,'dir'); mkdir(figDir); end

modelFile = fullfile(outDir,'pinn_model.mat');
summaryFile = fullfile(outDir,'naive_inverse_pinn_summary.mat');

if exist(modelFile,'file')
    S = load(modelFile,'pinn');
    pinnNaive = S.pinn;
    cfg = pinnNaive.cfg;
elseif exist(summaryFile,'file')
    S = load(summaryFile,'cfg','pinnNaive');
    cfg = S.cfg;
    pinnNaive = S.pinnNaive;
else
    error(['Naive inverse-PINN output was not found. First run ', ...
           'paper_diagnostics/run_naive_inverse_PINN_baseline.m from the package root.']);
end

fprintf('Regenerating reference solution for diagnostics...\n');
sol = emhd1d_generateSyntheticData(cfg);

fprintf('Evaluating naive PINN on the full grid...\n');
pred = emhd1d_evalGrid(pinnNaive.net, sol, cfg);

stateRelPct.u = 100*relativeError(pred.u, sol.u);
stateRelPct.C = 100*relativeError(pred.C, sol.C);
stateRelPct.T = 100*relativeError(pred.T, sol.T);

fprintf('Computing finite-difference reference derivatives...\n');
Dref = referenceDerivatives(sol);

% Sample interior grid points for derivative diagnostics. This keeps the
% analysis reproducible while avoiding a very large automatic-differentiation graph.
Nx = numel(sol.x);
Nt = numel(sol.t);
[Xg,Tg] = ndgrid(double(sol.x(:)), double(sol.t(:)));
mask = true(Nx,Nt);
mask([1 end],:) = false;
mask(:,[1 end]) = false;
idxAll = find(mask(:));
rng(cfg.rngSeed + 991);
nDeriv = min(8000,numel(idxAll));
idx = idxAll(randperm(numel(idxAll),nDeriv));
Xphys = single([Xg(idx)'; Tg(idx)']);

fprintf('Computing neural derivatives at %d sampled grid points...\n',nDeriv);
Dnet = evalNetworkDerivatives(pinnNaive.net, Xphys, cfg);

fields = {'ut','ux','uxx','Ct','Cx','Cxx','Tt','Tx','Txx'};
derivErrPct = struct();
derivRmsRatio = struct();
for k = 1:numel(fields)
    f = fields{k};
    refVals = Dref.(f)(idx);
    netVals = Dnet.(f)(:);
    derivErrPct.(f) = 100*relativeError(netVals, refVals);
    derivRmsRatio.(f) = rmsSafe(netVals)/max(rmsSafe(refVals),eps);
end

fprintf('Evaluating oracle residual along the naive geometry trajectory...\n');
traj = oracleTrajectoryMetrics(pinnNaive, cfg, Dref, idx, Xg, Tg, sol);

% Summary table in long format. The values in this file are computed from
% the saved naive run and regenerated reference data.
Metric = strings(0,1);
Value = zeros(0,1);
Units = strings(0,1);
Description = strings(0,1);

[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'state_rel_error_u',stateRelPct.u,'percent','Relative value error for velocity over the full grid.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'state_rel_error_C',stateRelPct.C,'percent','Relative value error for concentration over the full grid.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'state_rel_error_T',stateRelPct.T,'percent','Relative value error for temperature over the full grid.');

for k = 1:numel(fields)
    f = fields{k};
    [Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,['derivative_rel_error_' f],derivErrPct.(f),'percent',['Relative error for neural derivative ' f '.']);
    [Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,['derivative_rms_ratio_' f],derivRmsRatio.(f),'ratio',['RMS(neural derivative)/RMS(reference derivative) for ' f '.']);
end

[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_loss_true_geometry',traj.lossTrue,'scaled loss','Scaled oracle residual loss at the true geometry.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_loss_initial_trajectory',traj.lossInitial,'scaled loss','Scaled oracle residual loss at the first naive trajectory point.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_loss_final_trajectory',traj.lossFinal,'scaled loss','Scaled oracle residual loss at the final naive trajectory point.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_loss_final_over_true',traj.lossFinalOverTrue,'ratio','Final naive-trajectory oracle loss divided by true-geometry oracle loss.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_loss_increase_initial_to_final',traj.lossIncreasePct,'percent','Percent increase in oracle loss from first to final naive trajectory point.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_Ru_rms_true_geometry',traj.RuRmsTrue,'RMS','Momentum residual RMS at the true geometry.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_Ru_rms_final_trajectory',traj.RuRmsFinal,'RMS','Momentum residual RMS at the final naive trajectory point.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'oracle_Ru_rms_final_over_true',traj.RuRmsFinalOverTrue,'ratio','Final naive-trajectory momentum RMS divided by true-geometry momentum RMS.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'naive_final_delta',pinnNaive.theta.delta,'value','Final naive estimate of delta.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'naive_final_xt',pinnNaive.theta.xt,'value','Final naive estimate of x_t.');
[Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,'naive_final_sigma',pinnNaive.theta.sigma,'value','Final naive estimate of sigma.');

summary = table(Metric,Value,Units,Description);
writetable(summary,fullfile(outDir,'naive_pinn_failure_summary.csv'));

trajectory = table(traj.iter(:),traj.delta(:),traj.xt(:),traj.sigma(:),traj.loss(:),traj.RuRms(:), ...
    'VariableNames',{'Iteration','Delta','Xt','Sigma','OracleScaledLoss','OracleRuRms'});
writetable(trajectory,fullfile(outDir,'naive_oracle_trajectory.csv'));

% Also place the compact summary where generate_paper_figures.m can find it.
copyfile(fullfile(outDir,'naive_pinn_failure_summary.csv'), ...
    fullfile(rootDir,'paper_diagnostics','naive_pinn_failure_summary.csv'));

% Rebuild Figure 5 directly from computed diagnostics.
figureValues = [stateRelPct.u, stateRelPct.C, stateRelPct.T, ...
                derivErrPct.ux, derivErrPct.uxx, derivErrPct.Cx, derivErrPct.Cxx];
figureLabels = categorical({'u','C','T','u_x','u_{xx}','C_x','C_{xx}'});
figureLabels = reordercats(figureLabels,cellstr(figureLabels));
fig = figure('Color','w','Position',[100 100 980 440]);
bar(figureLabels,figureValues,'FaceAlpha',0.85);
grid on; box on;
ylabel('relative error (%)');
title('Naive inverse PINN: value-level errors versus derivative errors');
yline(10,'--','10%','LabelHorizontalAlignment','left');
exportgraphics(fig,fullfile(figDir,'fig05_naive_derivative_error_bar.png'),'Resolution',300);
exportgraphics(fig,fullfile(figDir,'fig05_naive_derivative_error_bar.pdf'),'ContentType','vector');
close(fig);

fprintf('\nSaved naive diagnostics to:\n  %s\n',outDir);
fprintf('Rebuilt Figure 5 in:\n  %s\n',figDir);

function [Metric,Value,Units,Description] = addMetric(Metric,Value,Units,Description,name,value,units,description)
    Metric(end+1,1) = string(name); %#ok<AGROW>
    Value(end+1,1) = double(value); %#ok<AGROW>
    Units(end+1,1) = string(units); %#ok<AGROW>
    Description(end+1,1) = string(description); %#ok<AGROW>
end

function e = relativeError(a,b)
    a = double(a(:));
    b = double(b(:));
    denom = norm(b);
    if denom < eps
        denom = eps;
    end
    e = norm(a-b)/denom;
end

function r = rmsSafe(x)
    x = double(x(:));
    r = sqrt(mean(x.^2));
end

function D = referenceDerivatives(sol)
    x = double(sol.x(:));
    t = double(sol.t(:));
    dx = x(2)-x(1);
    dt = t(2)-t(1);
    D = struct();
    [D.ut,D.ux,D.uxx] = finiteDiffDerivatives(double(sol.u),dx,dt);
    [D.Ct,D.Cx,D.Cxx] = finiteDiffDerivatives(double(sol.C),dx,dt);
    [D.Tt,D.Tx,D.Txx] = finiteDiffDerivatives(double(sol.T),dx,dt);
end

function [Yt,Yx,Yxx] = finiteDiffDerivatives(Y,dx,dt)
    Yt = zeros(size(Y));
    Yx = zeros(size(Y));
    Yxx = zeros(size(Y));

    Yt(:,2:end-1) = (Y(:,3:end)-Y(:,1:end-2))/(2*dt);
    Yt(:,1) = (Y(:,2)-Y(:,1))/dt;
    Yt(:,end) = (Y(:,end)-Y(:,end-1))/dt;

    Yx(2:end-1,:) = (Y(3:end,:)-Y(1:end-2,:))/(2*dx);
    Yx(1,:) = (Y(2,:)-Y(1,:))/dx;
    Yx(end,:) = (Y(end,:)-Y(end-1,:))/dx;

    Yxx(2:end-1,:) = (Y(3:end,:)-2*Y(2:end-1,:)+Y(1:end-2,:))/(dx^2);
    Yxx(1,:) = 2*(Y(2,:)-Y(1,:))/(dx^2);
    Yxx(end,:) = 2*(Y(end-1,:)-Y(end,:))/(dx^2);
end

function D = evalNetworkDerivatives(net,Xphys,cfg)
    X = dlarray(single(Xphys));
    Ddl = dlfeval(@networkDerivativesDL,net,X,cfg);
    names = fieldnames(Ddl);
    D = struct();
    for i = 1:numel(names)
        D.(names{i}) = gather(extractdata(Ddl.(names{i})))';
    end
end

function D = networkDerivativesDL(net,X,cfg)
    Xn = [X(1,:)./cfg.geom.L; X(2,:)./cfg.time.tFinal];
    Y = emhd1d_forwardMLP(net,Xn,cfg);
    u = Y(1,:);
    C = Y(2,:);
    T = Y(3,:);

    du = dlgradient(sum(u,'all'),X,'EnableHigherDerivatives',true);
    dC = dlgradient(sum(C,'all'),X,'EnableHigherDerivatives',true);
    dT = dlgradient(sum(T,'all'),X,'EnableHigherDerivatives',true);

    ux = du(1,:); ut = du(2,:);
    Cx = dC(1,:); Ct = dC(2,:);
    Tx = dT(1,:); Tt = dT(2,:);

    dux = dlgradient(sum(ux,'all'),X,'EnableHigherDerivatives',true);
    dCx = dlgradient(sum(Cx,'all'),X,'EnableHigherDerivatives',true);
    dTx = dlgradient(sum(Tx,'all'),X,'EnableHigherDerivatives',true);

    D.ut = ut; D.ux = ux; D.uxx = dux(1,:);
    D.Ct = Ct; D.Cx = Cx; D.Cxx = dCx(1,:);
    D.Tt = Tt; D.Tx = Tx; D.Txx = dTx(1,:);
end

function traj = oracleTrajectoryMetrics(pinn,cfg,Dref,idx,Xg,Tg,sol)
    H = pinn.history;
    nHist = numel(H.iter);
    nEval = min(250,nHist);
    sampleHist = unique(round(linspace(1,nHist,nEval)));

    ref.x = Xg(idx);
    ref.t = Tg(idx); %#ok<NASGU>
    ref.u = double(sol.u(idx));
    ref.C = double(sol.C(idx));
    ref.T = double(sol.T(idx));
    ref.ut = double(Dref.ut(idx));
    ref.uxx = double(Dref.uxx(idx));
    ref.Ct = double(Dref.Ct(idx));
    ref.Cx = double(Dref.Cx(idx));
    ref.Cxx = double(Dref.Cxx(idx));
    ref.Tt = double(Dref.Tt(idx));
    ref.Tx = double(Dref.Tx(idx));
    ref.Txx = double(Dref.Txx(idx));

    trueVals = oracleResidual(cfg,ref,cfg.geom.deltaTrue,cfg.geom.xtTrue,cfg.geom.sigmaTrue);
    traj.lossTrue = trueVals.loss;
    traj.RuRmsTrue = trueVals.RuRms;

    loss = zeros(numel(sampleHist),1);
    RuRms = zeros(numel(sampleHist),1);
    delta = zeros(numel(sampleHist),1);
    xt = zeros(numel(sampleHist),1);
    sigma = zeros(numel(sampleHist),1);
    iter = zeros(numel(sampleHist),1);

    for j = 1:numel(sampleHist)
        ii = sampleHist(j);
        delta(j) = H.delta(ii);
        xt(j) = H.xt(ii);
        sigma(j) = H.sigma(ii);
        iter(j) = H.iter(ii);
        vals = oracleResidual(cfg,ref,delta(j),xt(j),sigma(j));
        loss(j) = vals.loss;
        RuRms(j) = vals.RuRms;
    end

    traj.iter = iter;
    traj.delta = delta;
    traj.xt = xt;
    traj.sigma = sigma;
    traj.loss = loss;
    traj.RuRms = RuRms;
    traj.lossInitial = loss(1);
    traj.lossFinal = loss(end);
    traj.RuRmsFinal = RuRms(end);
    traj.lossFinalOverTrue = loss(end)/max(traj.lossTrue,eps);
    traj.RuRmsFinalOverTrue = RuRms(end)/max(traj.RuRmsTrue,eps);
    traj.lossIncreasePct = 100*(loss(end)-loss(1))/max(abs(loss(1)),eps);
end

function vals = oracleResidual(cfg,ref,delta,xt,sigma)
    h = emhd1d_geometry(ref.x,delta,xt,sigma,cfg);
    ku = cfg.phys.k0*(cfg.geom.h0./h).^cfg.phys.mUptake;

    Ru = ref.ut ...
        - cfg.phys.nu*ref.uxx ...
        + (1/cfg.phys.rho)*cfg.phys.dpdx ...
        - cfg.phys.alphaE*cfg.phys.E0 ...
        + cfg.phys.alphaB*(cfg.phys.B0^2).*ref.u ...
        + cfg.phys.alphaH*(ref.u./h) ...
        - cfg.phys.betaB*(ref.T-cfg.phys.T0);

    RC = ref.Ct + ref.u.*ref.Cx ...
        - cfg.phys.DC*ref.Cxx ...
        + ku.*ref.C;

    RT = ref.Tt + ref.u.*ref.Tx ...
        - cfg.phys.kappa*ref.Txx ...
        - cfg.phys.gammaE*(cfg.phys.E0^2) ...
        - cfg.phys.gammaB*(cfg.phys.B0^2).*(ref.u.^2);

    vals.loss = mean((Ru/cfg.pinn.scalePhysU).^2 + ...
                     (RC/cfg.pinn.scalePhysC).^2 + ...
                     (RT/cfg.pinn.scalePhysT).^2,'all');
    vals.RuRms = sqrt(mean(Ru.^2));
end

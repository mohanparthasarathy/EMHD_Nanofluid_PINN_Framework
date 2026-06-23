% GENERATE_PAPER_FIGURES
% Creates publication figures and result tables from saved EMHD 1D Geometry
% PINN outputs without rerunning the full inverse-PINN training workflow.
%
% Usage:
%   1. Put this file in the root folder of EMHD_1D_Geometry_PINN_Publication_Package.
%   2. In MATLAB, cd to that root folder.
%   3. Run:
%        generate_paper_figures
%
% Outputs are written to:
%   paper_figures/
%
% Notes:
% - This script uses saved .mat/.csv/.png outputs already in the package.
% - It may evaluate the saved PINN on the grid, but it does not retrain it.
% - The optional residual scan uses the saved network and saved synthetic data;
%   it does not run the 19,500-iteration training loop.

clear; close all; clc;

%% ------------------------------------------------------------------------
%  Paths and switches
% -------------------------------------------------------------------------
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end
addpath(fullfile(rootDir,'src'));

figDir = fullfile(rootDir,'paper_figures');
if ~exist(figDir,'dir'); mkdir(figDir); end

% Set this flag to false to skip the residual scan. All other figures are
% generated from existing saved outputs.
RUN_RESIDUAL_SCAN = true;

fprintf('\n============================================================\n');
fprintf('Generating paper figures from saved EMHD PINN package\n');
fprintf('Root: %s\n',rootDir);
fprintf('Output: %s\n',figDir);
fprintf('============================================================\n');

%% ------------------------------------------------------------------------
%  Load saved baseline result
% -------------------------------------------------------------------------
finalMat = fullfile(rootDir,'results','final_results.mat');
if ~exist(finalMat,'file')
    error('Could not find %s. Run this script from the package root.', finalMat);
end
S = load(finalMat,'cfg','sol','data','pinn','diag');
cfg = S.cfg; sol = S.sol; data = S.data; pinn = S.pinn; diag = S.diag;

x = double(sol.x(:));
t = double(sol.t(:));

%% ------------------------------------------------------------------------
%  Figure 1: model/geometry schematic
% -------------------------------------------------------------------------
% Creates a clean schematic if the original 1D_baseline_model_a.jpg is not
% present. This can be used as the Section 2 geometry figure.
try
    h = double(sol.h(:));
catch
    h = emhd1d_geometry(x,cfg.geom.deltaTrue,cfg.geom.xtTrue,cfg.geom.sigmaTrue,cfg);
end

fig = figure('Color','w','Position',[100 100 1000 360]);
fill([x; flipud(x)],[h; -flipud(h)],[0.92 0.96 1.00], ...
    'EdgeColor','none'); hold on;
plot(x,h,'k-','LineWidth',2);
plot(x,-h,'k-','LineWidth',2);
plot([cfg.geom.xtTrue cfg.geom.xtTrue],[-max(h) max(h)],'r--','LineWidth',1.5);
text(cfg.geom.xtTrue+0.015,0.82*max(h),'tumor center $x_t$', ...
    'Interpreter','latex','Color','r','FontSize',12);
quiver(0.08,0,0.16,0,0,'LineWidth',2,'MaxHeadSize',0.7);
text(0.08,0.12,'flow + EMHD forcing','FontSize',12);
xlabel('$x$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
title('One-dimensional tumor-constricted vessel geometry','Interpreter','latex');
axis tight; ylim(1.15*[-max(h) max(h)]); grid on; box on;
exportgraphics(fig,fullfile(figDir,'fig01_model_schematic.png'),'Resolution',300);
exportgraphics(fig,fullfile(figDir,'fig01_model_schematic.pdf'),'ContentType','vector');
close(fig);

%% ------------------------------------------------------------------------
%  Figure 2: MMS convergence plot
% -------------------------------------------------------------------------
% Uses the existing MMS plot if present. If not present, tries to run the
% verification script, which is usually much cheaper than PINN training.
sourceMMS = fullfile(rootDir,'results_MMS','mms_convergence_plot.png');
if ~exist(sourceMMS,'file') && exist(fullfile(rootDir,'verify_FDM_MMS.m'),'file')
    fprintf('MMS plot not found; attempting to run verify_FDM_MMS.m...\n');
    oldDir = pwd; cd(rootDir);
    try
        run('verify_FDM_MMS.m');
    catch ME
        warning('Could not run verify_FDM_MMS.m: %s',ME.message);
    end
    cd(oldDir);
end
if exist(sourceMMS,'file')
    copyfile(sourceMMS,fullfile(figDir,'fig02_mms_convergence.png'));
else
    makeTextFigure(fullfile(figDir,'fig02_mms_convergence.png'), ...
        {'MMS convergence plot not found.', ...
         'Run verify_FDM_MMS.m to regenerate results_MMS/mms_convergence_plot.png.'});
end

%% ------------------------------------------------------------------------
%  Figure 3: baseline true vs recovered vs initial geometry
% -------------------------------------------------------------------------
hTrue = emhd1d_geometry(x,cfg.geom.deltaTrue,cfg.geom.xtTrue,cfg.geom.sigmaTrue,cfg);
hInit = emhd1d_geometry(x,cfg.pinn.deltaInit,cfg.pinn.xtInit,cfg.pinn.sigmaInit,cfg);
hRec  = emhd1d_geometry(x,diag.deltaEst,diag.xtEst,diag.sigmaEst,cfg);

fig = figure('Color','w','Position',[100 100 950 420]);
plot(x,hTrue,'k-','LineWidth',2.5); hold on;
plot(x,hInit,'--','LineWidth',2.0);
plot(x,hRec,'-.','LineWidth',2.2);
xlabel('$x$','Interpreter','latex'); ylabel('$h(x)$','Interpreter','latex');
title('Baseline geometry recovery','Interpreter','latex');
legend({'true geometry','initial guess','recovered geometry'},'Location','best');
grid on; box on;
subtitle(sprintf('$\\delta$: %.3f $\\to$ %.3f; $x_t$: %.3f $\\to$ %.3f; $\\sigma$: %.3f $\\to$ %.3f', ...
    cfg.geom.deltaTrue,diag.deltaEst,cfg.geom.xtTrue,diag.xtEst,cfg.geom.sigmaTrue,diag.sigmaEst), ...
    'Interpreter','latex');
exportgraphics(fig,fullfile(figDir,'fig03_baseline_geometry_recovery.png'),'Resolution',300);
exportgraphics(fig,fullfile(figDir,'fig03_baseline_geometry_recovery.pdf'),'ContentType','vector');
close(fig);

%% ------------------------------------------------------------------------
%  Figure 4: state reconstruction heatmaps
% -------------------------------------------------------------------------
% Evaluates the saved PINN when possible. If the saved model cannot be
% evaluated, the script copies results/field_reconstruction.png.
try
    pred = emhd1d_evalGrid(pinn.net, sol, cfg);
    fig = figure('Color','w','Position',[100 100 1150 760]);
    tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
    fields = {'u','C','T'};
    names = {'velocity $u$','concentration $C$','temperature $T$'};
    for k = 1:3
        Ytrue = double(sol.(fields{k}));
        Ypred = double(pred.(fields{k}));
        Yerr = Ypred - Ytrue;
        nexttile; imagesc(t,x,Ytrue); axis xy; colorbar;
        title(['FD truth: ',names{k}],'Interpreter','latex'); xlabel('$t$','Interpreter','latex'); ylabel('$x$','Interpreter','latex');
        nexttile; imagesc(t,x,Ypred); axis xy; colorbar;
        title(['PINN: ',names{k}],'Interpreter','latex'); xlabel('$t$','Interpreter','latex'); ylabel('$x$','Interpreter','latex');
        nexttile; imagesc(t,x,Yerr); axis xy; colorbar;
        title(['error: ',names{k}],'Interpreter','latex'); xlabel('$t$','Interpreter','latex'); ylabel('$x$','Interpreter','latex');
    end
    exportgraphics(fig,fullfile(figDir,'fig04_state_reconstruction.png'),'Resolution',300);
    close(fig);
catch ME
    warning('Could not evaluate saved PINN for state reconstruction: %s',ME.message);
    sourceRecon = fullfile(rootDir,'results','field_reconstruction.png');
    if exist(sourceRecon,'file')
        copyfile(sourceRecon,fullfile(figDir,'fig04_state_reconstruction.png'));
    else
        makeTextFigure(fullfile(figDir,'fig04_state_reconstruction.png'), ...
            {'State reconstruction figure could not be generated.', ...
             'Check results/field_reconstruction.png or emhd1d_evalGrid.m.'});
    end
end

%% ------------------------------------------------------------------------
%  Figure 5: derivative-error diagnostic for naive inverse PINN. 
%  For this figure, run paper_diagnostics/analyze_naive_inverse_PINN.m
% -------------------------------------------------------------------------

%% ------------------------------------------------------------------------
%  Figure 6: residual-scan / identifiability diagnostic
% -------------------------------------------------------------------------
scanMat = fullfile(figDir,'phys_vs_obsgeom_delta_scan.mat');
scanPng = fullfile(figDir,'fig06_residual_scan_identifiability.png');

if RUN_RESIDUAL_SCAN
    try
        fprintf('Running saved-network residual scan. This does not retrain the PINN...\n');
        if isfield(data,'phys') && isfield(data.phys,'X')
            Xphys = data.phys.X(:,1:min(1200,size(data.phys.X,2))); % faster than full scan
        else
            Xphys = [];
        end
        resultScan = emhd1d_scanWarmupPhysVsObsGeom(cfg,sol,pinn.net,figDir,Xphys);
        save(scanMat,'resultScan');
    catch ME
        warning('Residual scan failed: %s',ME.message);
    end
end

% Replot scan manually so label formatting is controlled here.
if exist(scanMat,'file')
    R = load(scanMat);

    if isfield(R,'resultScan')
        r = R.resultScan;
    elseif isfield(R,'result')
        r = R.result;
    else
        error('Scan MAT file does not contain resultScan or result.');
    end

    fig = figure('Color','w','Position',[100 100 900 430]);

    plot(r.deltas,r.lossPhys,'LineWidth',2); hold on;
    plot(r.deltas,r.lossObsGeom,'LineWidth',2);

    xline(r.trueDelta,'k--','LineWidth',1.5);
    xline(r.deltaPhysMin,':','LineWidth',1.5);
    xline(r.deltaObsGeomMin,':','LineWidth',1.5);

    xlabel('$\delta$','Interpreter','latex');
    ylabel('scaled residual loss');
    title('Residual scan: neural-derivative physics vs observation-anchored geometry loss');

    legend({'Frozen-network $L_{phys}$','$L_{obs\mathrm{-}geom}$'}, ...
           'Interpreter','latex', ...
           'Location','northwest');

    grid on;
    box on;

    exportgraphics(fig,scanPng,'Resolution',300);
    close(fig);

else
    makeTextFigure(scanPng, ...
        {'Residual-scan figure was not generated.', ...
         'Set RUN_RESIDUAL_SCAN=true and check emhd1d_scanWarmupPhysVsObsGeom.m.'});
end

%% ------------------------------------------------------------------------
%  Figure 7: robustness grouped-bar plot
% -------------------------------------------------------------------------
robustCsv = fullfile(rootDir,'paper_reported_results','robustness_summary.csv');
hardCsv   = fullfile(rootDir,'paper_reported_results','hard_stress_test_summary.csv');
if ~exist(robustCsv,'file')
    robustCsv = fullfile(rootDir,'robustness_results','robustness_summary.csv');
end

if exist(robustCsv,'file')
    Trob = readtable(robustCsv,'TextType','string');
    cases = string(Trob.Case);
    errMat = [Trob.DeltaErrorPct, Trob.XtErrorPct, Trob.SigmaErrorPct];

    % Append the 10% hard-stress case if present.
    if exist(hardCsv,'file')
        Thard = readtable(hardCsv,'TextType','string');
        eDelta = Thard.ErrorPercent(strcmpi(Thard.Quantity,'Delta'));
        eXt = Thard.ErrorPercent(strcmpi(Thard.Quantity,'x_t'));
        eSigma = Thard.ErrorPercent(strcmpi(Thard.Quantity,'Sigma'));
        cases(end+1,1) = "10pctNoise";
        errMat(end+1,:) = [eDelta, eXt, eSigma];
    end

    fig = figure('Color','w','Position',[100 100 1050 460]);
    b = bar(categorical(cases),errMat,'grouped'); %#ok<NASGU>
    ylabel('absolute parameter error (%)');
    title('Robustness of geometry recovery');
    legend({'$\delta$','$x_t$','$\sigma$'},'Interpreter','latex','Location','northwest');
    grid on; box on;
    exportgraphics(fig,fullfile(figDir,'fig07_robustness_errors.png'),'Resolution',300);
    exportgraphics(fig,fullfile(figDir,'fig07_robustness_errors.pdf'),'ContentType','vector');
    close(fig);

    Tout = table(cases,errMat(:,1),errMat(:,2),errMat(:,3), ...
        'VariableNames',{'Case','DeltaErrorPct','XtErrorPct','SigmaErrorPct'});
    writetable(Tout,fullfile(figDir,'table02_recovery_and_robustness.csv'));
else
    makeTextFigure(fullfile(figDir,'fig07_robustness_errors.png'), ...
        {'Robustness CSV not found.', ...
         'Expected paper_reported_results/robustness_summary.csv.'});
end

%% ------------------------------------------------------------------------
%  Figure 8: training history
% -------------------------------------------------------------------------
sourceHist = fullfile(rootDir,'results','training_history.png');
if exist(sourceHist,'file')
    copyfile(sourceHist,fullfile(figDir,'fig08_training_history.png'));
else
    try
        H = pinn.history;
        fig = figure('Color','w','Position',[100 100 1050 600]);
        tiledlayout(2,1,'TileSpacing','compact');
        nexttile;
        semilogy(H.iter,H.total,'LineWidth',1.6); hold on;
        semilogy(H.iter,H.data,'LineWidth',1.1);
        semilogy(H.iter,H.phys,'LineWidth',1.1);
        semilogy(H.iter,H.obsGeom,'LineWidth',1.1);
        xlabel('iteration'); ylabel('loss'); grid on; box on;
        legend({'total','data','physics','obs-geom'},'Location','best');
        title('Training losses');
        nexttile;
        plot(H.iter,H.delta,'LineWidth',1.4); hold on;
        plot(H.iter,H.xt,'LineWidth',1.4);
        plot(H.iter,H.sigma,'LineWidth',1.4);
        yline(cfg.geom.deltaTrue,'--'); yline(cfg.geom.xtTrue,'--'); yline(cfg.geom.sigmaTrue,'--');
        xlabel('iteration'); ylabel('parameter value'); grid on; box on;
        legend({'$\delta$','$x_t$','$\sigma$','truth markers'},'Interpreter','latex','Location','best');
        title('Recovered geometry parameters during training');
        exportgraphics(fig,fullfile(figDir,'fig08_training_history.png'),'Resolution',300);
        close(fig);
    catch ME
        warning('Could not generate training history: %s',ME.message);
    end
end

%% ------------------------------------------------------------------------
%  Table 1
% -------------------------------------------------------------------------

fprintf('\nDone. Figures and tables written to:\n  %s\n',figDir);

%% ========================================================================
%  Local helper functions
% ========================================================================
function makeTextFigure(outPath,lines)
    fig = figure('Color','w','Position',[100 100 900 300]);
    axis off;
    text(0.05,0.65,lines,'FontSize',14,'Interpreter','none');
    exportgraphics(fig,outPath,'Resolution',200);
    close(fig);
end
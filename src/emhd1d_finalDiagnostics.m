function diag = emhd1d_finalDiagnostics(pinn, sol, data, cfg, outDir)
%EMHD1D_FINALDIAGNOSTICS Error report and geometry comparison plots.
if nargin < 5 || isempty(outDir); outDir = pwd; end
pred = emhd1d_evalGrid(pinn.net, sol, cfg);

diag.deltaTrue = cfg.geom.deltaTrue;
diag.xtTrue = cfg.geom.xtTrue;
diag.sigmaTrue = cfg.geom.sigmaTrue;
diag.deltaEst = pred.delta;
diag.xtEst = pred.xt;
diag.sigmaEst = pred.sigma;
diag.deltaErrPct = 100*abs(pred.delta-cfg.geom.deltaTrue)/max(abs(cfg.geom.deltaTrue),eps);
diag.xtErrPct = 100*abs(pred.xt-cfg.geom.xtTrue)/max(abs(cfg.geom.xtTrue),eps);
diag.sigmaErrPct = 100*abs(pred.sigma-cfg.geom.sigmaTrue)/max(abs(cfg.geom.sigmaTrue),eps);
diag.relU = norm(double(pred.u(:))-double(sol.u(:)))/norm(double(sol.u(:)));
diag.relC = norm(double(pred.C(:))-double(sol.C(:)))/norm(double(sol.C(:)));
diag.relT = norm(double(pred.T(:))-double(sol.T(:)))/norm(double(sol.T(:)));

x = double(sol.x);
fig = figure('Color','w','Position',[100 100 900 380]);
plot(x,double(sol.h),'k-','LineWidth',2); hold on;
plot(x,pred.h,'--','LineWidth',2);
xlabel('x'); ylabel('h(x)'); grid on;
legend('True geometry','Recovered geometry','Location','best');
title(sprintf('Geometry recovery: delta true %.3f, est %.3f; x_t true %.3f, est %.3f; sigma true %.3f, est %.3f', ...
    cfg.geom.deltaTrue,pred.delta,cfg.geom.xtTrue,pred.xt,cfg.geom.sigmaTrue,pred.sigma));
exportgraphics(fig,fullfile(outDir,'geometry_true_vs_recovered.png'),'Resolution',200);

fig = figure('Color','w','Position',[100 100 950 700]);
tiledlayout(3,2,'TileSpacing','compact');
fields = {'u','C','T'};
for k = 1:3
    nexttile; imagesc(double(sol.t),double(sol.x),double(sol.(fields{k}))); axis xy; colorbar;
    xlabel('t'); ylabel('x'); title(['True ',fields{k}]);
    nexttile; imagesc(double(sol.t),double(sol.x),double(pred.(fields{k}))); axis xy; colorbar;
    xlabel('t'); ylabel('x'); title(['PINN ',fields{k}]);
end
exportgraphics(fig,fullfile(outDir,'field_reconstruction.png'),'Resolution',200);

fid = fopen(fullfile(outDir,'summary.txt'),'w');
fprintf(fid,'EMHD 1D delta/x_t/sigma PINN final summary\n');
fprintf(fid,'delta true %.10f estimated %.10f error %.6f pct\n',diag.deltaTrue,diag.deltaEst,diag.deltaErrPct);
fprintf(fid,'x_t true %.10f estimated %.10f error %.6f pct\n',diag.xtTrue,diag.xtEst,diag.xtErrPct);
fprintf(fid,'sigma true %.10f estimated %.10f error %.6f pct\n',diag.sigmaTrue,diag.sigmaEst,diag.sigmaErrPct);
fprintf(fid,'relU %.8e relC %.8e relT %.8e\n',diag.relU,diag.relC,diag.relT);
fclose(fid);
end

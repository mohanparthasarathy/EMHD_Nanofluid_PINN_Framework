function emhd1d_plotTrainingHistory(pinn, cfg, outDir)
%EMHD1D_PLOTTRAININGHISTORY Training loss and geometry trajectories.
H = pinn.history;
fig = figure('Color','w','Position',[100 100 1000 650]);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; semilogy(H.iter,H.total,'LineWidth',1.5); hold on; semilogy(H.iter,H.data); semilogy(H.iter,H.phys);
if isfield(H,'obsGeom'); semilogy(H.iter,H.obsGeom); legend('total','data','physics','obsGeom'); else; legend('total','data','physics'); end
grid on; xlabel('iteration'); ylabel('loss'); title('Training losses');
nexttile; plot(H.iter,H.delta,'LineWidth',1.5); yline(cfg.geom.deltaTrue,'--'); grid on; xlabel('iteration'); ylabel('\delta'); title('Constriction depth');
nexttile; plot(H.iter,H.xt,'LineWidth',1.5); yline(cfg.geom.xtTrue,'--'); grid on; xlabel('iteration'); ylabel('x_t'); title('Constriction location');
nexttile; plot(H.iter,H.sigma,'LineWidth',1.5); yline(cfg.geom.sigmaTrue,'--'); grid on; xlabel('iteration'); ylabel('\sigma'); title('Constriction width');
exportgraphics(fig,fullfile(outDir,'training_history.png'),'Resolution',200);
end

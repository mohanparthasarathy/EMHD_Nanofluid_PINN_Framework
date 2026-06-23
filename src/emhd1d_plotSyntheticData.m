function emhd1d_plotSyntheticData(sol, data, cfg, outDir)
%EMHD1D_PLOTSYNTHETICDATA Quick FDM/data diagnostic plots.
if nargin < 4 || isempty(outDir); outDir = pwd; end
x = double(sol.x); t = double(sol.t);
fig = figure('Color','w','Position',[100 100 920 700]);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; plot(x,double(sol.h),'LineWidth',2); grid on; xlabel('x'); ylabel('h(x)'); title('True vessel geometry');
nexttile; imagesc(t,x,double(sol.u)); axis xy; colorbar; xlabel('t'); ylabel('x'); title('u(x,t)');
nexttile; imagesc(t,x,double(sol.C)); axis xy; colorbar; xlabel('t'); ylabel('x'); title('C(x,t)');
nexttile; imagesc(t,x,double(sol.T)); axis xy; colorbar; xlabel('t'); ylabel('x'); title('T(x,t)');
exportgraphics(fig,fullfile(outDir,'synthetic_fields.png'),'Resolution',200);

fig = figure('Color','w','Position',[100 100 800 350]);
scatter(double(data.obs.X(1,:)),double(data.obs.X(2,:)),8,double(data.obs.Y(2,:)),'filled');
colorbar; xlabel('x'); ylabel('t'); title('Sparse synthetic observations colored by C'); grid on;
exportgraphics(fig,fullfile(outDir,'sparse_observations.png'),'Resolution',200);
end

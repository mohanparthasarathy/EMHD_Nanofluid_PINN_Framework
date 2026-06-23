function emhd1d_validateConfig(cfg)
%EMHD1D_VALIDATECONFIG Basic runtime checks.

try
    x = dlarray(single(2.0));
    g = dlfeval(@emhd1d_localGradTest, x);
    if ~isa(g,'dlarray')
        error('Autodiff test did not return a dlarray.');
    end
catch ME
    error(['Deep Learning Toolbox automatic differentiation test failed. ', ...
           'Run: which dlarray -all; which dlfeval -all; which dlgradient -all. ', ...
           'Original error: %s'], ME.message);
end

if cfg.grid.Nx < 11 || cfg.grid.Nt < 11
    error('Grid must have at least 11 points in each dimension.');
end

if cfg.pinn.deltaInit <= cfg.pinn.deltaMin || cfg.pinn.deltaInit >= cfg.pinn.deltaMax
    error('Initial delta must lie inside bounds.');
end

if cfg.pinn.xtInit <= cfg.pinn.xtMin || cfg.pinn.xtInit >= cfg.pinn.xtMax
    error('Initial x_t must lie inside bounds.');
end

if cfg.pinn.sigmaInit <= cfg.pinn.sigmaMin || cfg.pinn.sigmaInit >= cfg.pinn.sigmaMax
    error('Initial sigma must lie inside bounds.');
end

if isfield(cfg,'outDir')
    outDir = cfg.outDir;
elseif isfield(cfg,'paths') && isfield(cfg.paths,'outDir')
    outDir = cfg.paths.outDir;
else
    outDir = fullfile(pwd,'output_emhd1d_geometry_pinn');
end

if ~exist(outDir,'dir')
    mkdir(outDir);
end

end

function g = emhd1d_localGradTest(x)
y = sum(x.^2,'all');
g = dlgradient(y,x);
end
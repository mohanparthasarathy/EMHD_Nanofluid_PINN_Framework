function Xn = emhd1d_normalizeInput(X, cfg)
%EMHD1D_NORMALIZEINPUT Physical [x;t] to normalized network input.
Xn = [X(1,:)./cfg.geom.L; X(2,:)./cfg.time.tFinal];
end

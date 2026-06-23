function h = emhd1d_geometry(x, delta, xt, sigma, cfg)
%EMHD1D_GEOMETRY Tumor-constricted peristaltic vessel half-width.
h = cfg.geom.a + cfg.geom.b*cos(2*pi*x) - delta.*exp(-((x-xt).^2)./(sigma.^2));
h = max(h, cfg.geom.hMinFloor);
end

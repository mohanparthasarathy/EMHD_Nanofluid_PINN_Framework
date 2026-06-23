function h = emhd1d_hFromTheta(x, theta, cfg)
%EMHD1D_HFROMTHETA dlarray-compatible h(x;theta).
h = cfg.geom.a + cfg.geom.b*cos(2*pi*x) - theta.delta.*exp(-((x-theta.xt).^2)./(theta.sigma.^2));
h = max(h, cfg.geom.hMinFloor);
end

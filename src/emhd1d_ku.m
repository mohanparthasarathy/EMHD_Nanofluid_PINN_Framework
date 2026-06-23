function ku = emhd1d_ku(h, cfg)
%EMHD1D_KU Geometry-dependent effective nanoparticle uptake.
ku = cfg.phys.k0*(cfg.geom.h0./h).^cfg.phys.mUptake;
end

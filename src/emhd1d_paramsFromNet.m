function theta = emhd1d_paramsFromNet(net, cfg)
%EMHD1D_PARAMSFROMNET Extract bounded geometry parameters from trainable variables.
theta.delta = emhd1d_boundFromRaw(net.rawDelta, cfg.pinn.deltaMin, cfg.pinn.deltaMax);
theta.xt = emhd1d_boundFromRaw(net.rawXt, cfg.pinn.xtMin, cfg.pinn.xtMax);
theta.sigma = emhd1d_boundFromRaw(net.rawSigma, cfg.pinn.sigmaMin, cfg.pinn.sigmaMax);
end

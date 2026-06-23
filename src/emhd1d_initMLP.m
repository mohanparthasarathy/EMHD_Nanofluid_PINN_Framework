function net = emhd1d_initMLP(layers, cfg)
%EMHD1D_INITMLP Fully connected tanh network plus raw geometry variables.
rng(cfg.pinn.seed);
L = numel(layers)-1;
net.W = cell(L,1); net.b = cell(L,1);
for k = 1:L
    fanIn = layers(k); fanOut = layers(k+1);
    lim = sqrt(6/(fanIn+fanOut));
    net.W{k} = dlarray(single((2*rand(fanOut,fanIn)-1)*lim));
    net.b{k} = dlarray(single(zeros(fanOut,1)));
end
net.rawDelta = dlarray(single(emhd1d_rawFromBound(cfg.pinn.deltaInit,cfg.pinn.deltaMin,cfg.pinn.deltaMax)));
net.rawXt = dlarray(single(emhd1d_rawFromBound(cfg.pinn.xtInit,cfg.pinn.xtMin,cfg.pinn.xtMax)));
net.rawSigma = dlarray(single(emhd1d_rawFromBound(cfg.pinn.sigmaInit,cfg.pinn.sigmaMin,cfg.pinn.sigmaMax)));
end

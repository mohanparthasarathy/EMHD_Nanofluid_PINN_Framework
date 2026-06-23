function Y = emhd1d_forwardMLP(net, Xn, cfg)
%EMHD1D_FORWARDMLP Network forward pass. Xn is normalized [x;t].
if isa(Xn,'dlarray'); A = stripdims(Xn); else; A = dlarray(single(Xn)); end
for k = 1:numel(net.W)-1
    A = tanh(stripdims(net.W{k})*A + stripdims(net.b{k}));
end
Z = stripdims(net.W{end})*A + stripdims(net.b{end});
if cfg.pinn.outputTransform
    u = 2.0 ./ (1 + exp(-Z(1,:)));
    C = 0.000 + 0.080 ./ (1 + exp(-Z(2,:)));
    T = 0.995 + 0.035 ./ (1 + exp(-Z(3,:)));
    Y = [u;C;T];
else
    Y = Z;
end
end

function pred = emhd1d_evalGrid(net, sol, cfg)
%EMHD1D_EVALGRID Evaluate PINN over the FDM grid.
x = double(sol.x); t = double(sol.t);
[X,Tg] = ndgrid(x,t);
Xphys = single([X(:)'; Tg(:)']);
bs = 20000;
Yall = zeros(3,size(Xphys,2),'single');
for i = 1:bs:size(Xphys,2)
    j = min(i+bs-1,size(Xphys,2));
    Xn = dlarray(emhd1d_normalizeInput(Xphys(:,i:j),cfg));
    Y = emhd1d_forwardMLP(net,Xn,cfg);
    Yall(:,i:j) = gather(extractdata(Y));
end
pred.u = reshape(Yall(1,:),numel(x),numel(t));
pred.C = reshape(Yall(2,:),numel(x),numel(t));
pred.T = reshape(Yall(3,:),numel(x),numel(t));
theta = emhd1d_paramsFromNet(net,cfg);
pred.delta = gather(extractdata(theta.delta));
pred.xt = gather(extractdata(theta.xt));
pred.sigma = gather(extractdata(theta.sigma));
pred.h = emhd1d_geometry(x,pred.delta,pred.xt,pred.sigma,cfg);
end

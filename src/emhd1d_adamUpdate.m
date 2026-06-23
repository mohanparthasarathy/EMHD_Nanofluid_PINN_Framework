function [net, avg, avgSq] = emhd1d_adamUpdate(net, grad, avg, avgSq, iter, cfg, phase)
%EMHD1D_ADAMUPDATE Adam update for network weights and geometry parameters.

if isempty(avg)
    avg.W = cell(size(net.W));
    avg.b = cell(size(net.b));
    avgSq.W = cell(size(net.W));
    avgSq.b = cell(size(net.b));

    for k = 1:numel(net.W)
        avg.W{k} = zeros(size(extractdata(net.W{k})),'single');
        avg.b{k} = zeros(size(extractdata(net.b{k})),'single');
        avgSq.W{k} = zeros(size(extractdata(net.W{k})),'single');
        avgSq.b{k} = zeros(size(extractdata(net.b{k})),'single');
    end

    avg.rawDelta = single(0);
    avg.rawXt = single(0);
    avg.rawSigma = single(0);
    avgSq.rawDelta = single(0);
    avgSq.rawXt = single(0);
    avgSq.rawSigma = single(0);
end

if ~isfield(phase,'trainNet')
    phase.trainNet = true;
end
if ~isfield(phase,'netLRFactor')
    phase.netLRFactor = 1.0;
end
if ~isfield(phase,'geomLRFactor')
    phase.geomLRFactor = 1.0;
end

netLR = cfg.pinn.learningRate * phase.netLRFactor;

if phase.trainNet && netLR > 0
    for k = 1:numel(net.W)
        [net.W{k}, avg.W{k}, avgSq.W{k}] = adamOne(net.W{k},grad.W{k},avg.W{k},avgSq.W{k},iter,cfg,netLR);
        [net.b{k}, avg.b{k}, avgSq.b{k}] = adamOne(net.b{k},grad.b{k},avg.b{k},avgSq.b{k},iter,cfg,netLR);
    end
end

geomLR = cfg.pinn.learningRateGeom * phase.geomLRFactor;

if phase.trainDelta && geomLR > 0
    [net.rawDelta, avg.rawDelta, avgSq.rawDelta] = adamOne(net.rawDelta,grad.rawDelta,avg.rawDelta,avgSq.rawDelta,iter,cfg,geomLR);
end

if phase.trainXt && geomLR > 0
    [net.rawXt, avg.rawXt, avgSq.rawXt] = adamOne(net.rawXt,grad.rawXt,avg.rawXt,avgSq.rawXt,iter,cfg,geomLR);
end

if phase.trainSigma && geomLR > 0
    [net.rawSigma, avg.rawSigma, avgSq.rawSigma] = adamOne(net.rawSigma,grad.rawSigma,avg.rawSigma,avgSq.rawSigma,iter,cfg,geomLR);
end

end

function [param, avg, avgSq] = adamOne(param, grad, avg, avgSq, iter, cfg, lr)
g = gather(extractdata(grad));
b1 = cfg.pinn.gradDecay;
b2 = cfg.pinn.sqGradDecay;

avg = b1*avg + (1-b1)*g;
avgSq = b2*avgSq + (1-b2)*(g.^2);

avgHat = avg/(1-b1^iter);
avgSqHat = avgSq/(1-b2^iter);

newVal = gather(extractdata(param)) - lr*avgHat./(sqrt(avgSqHat)+cfg.pinn.epsAdam);
param = dlarray(single(newVal));
end

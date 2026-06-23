function cfg = emhd1d_defaultConfig()
%EMHD1D_DEFAULTCONFIG Parameters for 1D EMHD geometry PINN.

cfg.rngSeed = 7;

cfg.geom.L = 1.0;
cfg.geom.a = 1.00;
cfg.geom.b = 0.10;
cfg.geom.deltaTrue = 0.36;
cfg.geom.xtTrue = 0.58;
cfg.geom.sigmaTrue = 0.105;
cfg.geom.h0 = cfg.geom.a;
cfg.geom.hMinFloor = 0.25;

% Prior values used only for initialization or optional diagnostics.
% In this version, delta, x_t, and sigma are all estimated.
cfg.prior.xtKnown = 0.58;
cfg.prior.sigmaKnown = 0.105;

% Longer observation window keeps the model unchanged but improves geometry observability.
cfg.time.tFinal = 5.0;
cfg.grid.Nx = 121;
cfg.grid.Nt = 5001;

cfg.phys.nu = 2.0e-3;
cfg.phys.rho = 1.0;
cfg.phys.dpdx = -0.20;
cfg.phys.alphaE = 0.075;
cfg.phys.E0 = 1.0;
cfg.phys.alphaB = 0.10;
cfg.phys.B0 = 1.0;
cfg.phys.alphaH = 0.055;
cfg.phys.betaB = 0.08;
cfg.phys.T0 = 1.0;
cfg.phys.DC = 1.2e-3;
cfg.phys.k0 = 0.060;
cfg.phys.mUptake = 1.5;
cfg.phys.kappa = 1.6e-3;
cfg.phys.gammaE = 1.5e-3;
cfg.phys.gammaB = 1.0e-3;

cfg.ic.uBase = 0.20;
cfg.ic.uAmp = 0.015;
cfg.ic.CBase = 0.020;
cfg.ic.CAmp = 0.030;
cfg.ic.CCenter = 0.40;
cfg.ic.CWidth = 0.070;
cfg.ic.TBase = cfg.phys.T0;

cfg.data.nObs = 7000;
cfg.data.nVal = 1200;
cfg.data.nCol = 8000;
cfg.data.nIC = 121;
cfg.data.nBC = 600;
cfg.data.noiseU = 0.005;
cfg.data.noiseC = 0.008;
cfg.data.noiseT = 0.008;
cfg.data.nObsForPhys = 3500;

cfg.pinn.layers = [2 96 96 96 96 3];
cfg.pinn.outputTransform = true;
cfg.pinn.learningRate = 1.0e-3;
cfg.pinn.learningRateGeom = 2.0e-4;
cfg.pinn.gradDecay = 0.90;
cfg.pinn.sqGradDecay = 0.999;
cfg.pinn.epsAdam = 1.0e-8;
cfg.pinn.printEvery = 100;
cfg.pinn.seed = 11;

cfg.pinn.deltaMin = 0.05;
cfg.pinn.deltaMax = 0.65;
cfg.pinn.xtMin = 0.15;
cfg.pinn.xtMax = 0.85;
cfg.pinn.sigmaMin = 0.035;
cfg.pinn.sigmaMax = 0.22;

% Non-truth initialization for all three unknown geometry parameters.
cfg.pinn.deltaInit = 0.22;
cfg.pinn.xtInit = 0.43;
cfg.pinn.sigmaInit = 0.160;

cfg.pinn.scaleU = 0.25;
cfg.pinn.scaleC = 0.030;
cfg.pinn.scaleT = 0.010;
cfg.pinn.scalePhysU = 0.20;
cfg.pinn.scalePhysC = 0.020;
cfg.pinn.scalePhysT = 0.010;
cfg.pinn.wReg = 1.0e-7;

cfg.pinn.dataWeightU = 0.15;
cfg.pinn.dataWeightC = 18.0;
cfg.pinn.dataWeightT = 0.02;

% Observation-anchored geometry residual. This term evaluates the
% governing residual using fixed fields reconstructed from noisy observations,
% so delta, x_t, and sigma receive a geometry signal that is not contaminated
% by neural derivative error.
cfg.pinn.wObsGeom = 2.0;
cfg.pinn.obsGeomN = 8000;

cfg.pinn.phases = struct([]);

cfg.pinn.phases(1).name = 'state_warmup';
cfg.pinn.phases(1).epochs = 1500;
cfg.pinn.phases(1).trainNet = true;
cfg.pinn.phases(1).trainDelta = false;
cfg.pinn.phases(1).trainXt = false;
cfg.pinn.phases(1).trainSigma = false;
cfg.pinn.phases(1).wData = 14.0;
cfg.pinn.phases(1).wIC = 20.0;
cfg.pinn.phases(1).wBC = 10.0;
cfg.pinn.phases(1).wPhysStart = 0.0;
cfg.pinn.phases(1).wPhysEnd = 0.00;
cfg.pinn.phases(1).netLRFactor = 1.0;
cfg.pinn.phases(1).geomLRFactor = 0.0;
cfg.pinn.phases(1).wObsGeom = 0.0;

cfg.pinn.phases(2).name = 'physics_intro';
cfg.pinn.phases(2).epochs = 1500;
cfg.pinn.phases(2).trainNet = true;
cfg.pinn.phases(2).trainDelta = false;
cfg.pinn.phases(2).trainXt = false;
cfg.pinn.phases(2).trainSigma = false;
cfg.pinn.phases(2).wData = 14.0;
cfg.pinn.phases(2).wIC = 20.0;
cfg.pinn.phases(2).wBC = 10.0;
cfg.pinn.phases(2).wPhysStart = 0.05;
cfg.pinn.phases(2).wPhysEnd = 0.5;
cfg.pinn.phases(2).netLRFactor = 1.0;
cfg.pinn.phases(2).geomLRFactor = 0.0;
cfg.pinn.phases(2).wObsGeom = 0.0;

cfg.pinn.phases(3).name = 'geometry_fit';
cfg.pinn.phases(3).epochs = 14000;
cfg.pinn.phases(3).trainNet = false;
cfg.pinn.phases(3).trainDelta = true;
cfg.pinn.phases(3).trainXt = true;
cfg.pinn.phases(3).trainSigma = true;
cfg.pinn.phases(3).wData = 8.0;
cfg.pinn.phases(3).wIC = 3.0;
cfg.pinn.phases(3).wBC = 1.5;
cfg.pinn.phases(3).wPhysStart = 0.0;
cfg.pinn.phases(3).wPhysEnd = 0.0;
cfg.pinn.phases(3).netLRFactor = 0.0;
cfg.pinn.phases(3).geomLRFactor = 1.0;
cfg.pinn.phases(3).wObsGeom = 1.0;

cfg.pinn.phases(4).name = 'joint_refinement';
cfg.pinn.phases(4).epochs = 2500;
cfg.pinn.phases(4).trainNet = true;
cfg.pinn.phases(4).trainDelta = false;
cfg.pinn.phases(4).trainXt = false;
cfg.pinn.phases(4).trainSigma = false;
cfg.pinn.phases(4).wData = 10.0;
cfg.pinn.phases(4).wIC = 2.0;
cfg.pinn.phases(4).wBC = 1.0;
cfg.pinn.phases(4).wPhysStart = 0.05;
cfg.pinn.phases(4).wPhysEnd = 0.2;
cfg.pinn.phases(4).netLRFactor = 0.02;
cfg.pinn.phases(4).geomLRFactor = 0.0;
cfg.pinn.phases(4).wObsGeom = 0;
end

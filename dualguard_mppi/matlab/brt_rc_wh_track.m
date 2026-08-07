clc
clear
% Three-state BRT for the RC car (kinematic bicycle) on the rc_wh track.
% Reconstruction of brt_rc_wh_fine_v3.mat, whose generating script was lost;
% every parameter below was recovered from the contents of that file.
% Writes brt_rc_wh_track.mat and brt_rc_wh_environment.mat.

%% problem parameters

x_0 = [0.5; 0.5; pi/2];

vRange    = [0.7 1.4];        % speed bounds        [m/s]     (control 1)
deltaMax  = deg2rad(25);      % steering-angle bound [rad]     (control 2)
wheelbase = 0.235;            % L                    [m]
dMax      = 0.0;              % no disturbance

% minimum turn radius = L / tan(deltaMax) = 0.504 m

uMode = 'max';
dMode = 'min';

filename = 'brt_rc_wh_track.mat';

%% Grid

grid_min = [-0.04; -0.04; -pi];   % Lower corner of computation domain
grid_max = [ 6.03;  4.03;   pi];  % Upper corner of computation domain
N        = [  608;   408;  181];  % Number of grid points per dimension

pdDims = 3;                    % State variable 3 (theta) is periodic
g = createGrid(grid_min, grid_max, N, pdDims);

%% target set
% Failure set = everything off the track: inside the outer wall and outside
% the infield, both stored as closed polygons. Independent of heading.

% true uses the failure set copied out of the original file rather than
% rebuilding it from the polygons. The two agree to 3.5e-5 rms on the track.
USE_REFERENCE_LX = false;

load('rc_wh_track_data')   % loads outer_x, outer_y, inner_x, inner_y

lx = shapeTrack2D(g, [outer_x(:) outer_y(:)], [inner_x(:) inner_y(:)]);

% Optional check against the failure set of the original file
if isfile('rc_wh_lx_reference.mat')
  ref = load('rc_wh_lx_reference.mat');
  err = abs(lx - ref.lx_ref);
  fprintf('lx vs. original: max err %.2e (on track: %.2e), rms %.2e\n', ...
          max(err(:)), max(err(ref.lx_ref > 0)), sqrt(mean(err(:).^2)));
  if USE_REFERENCE_LX
    lx = ref.lx_ref;
  end
end

data0 = repmat(lx, [1 1 g.N(3)]);

%% time vector
t0 = 0.0;
tMax = 5.0;
dt = 0.1;
tau = t0:dt:tMax;

%% Pack problem parameters

% Define dynamic system
sys = RCBicycleCar(x_0, vRange, deltaMax, wheelbase, dMax);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = sys;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
schemeData.dMode = dMode;

%% Compute value function

HJIextraArgs.stopConverge = false;   % original tau2 has all 51 time steps

% Keep only the latest slice; data(:,:,:,end) still works on the 3-D result.
HJIextraArgs.keepLast = true;

HJIextraArgs.visualize.valueSet = 1;
HJIextraArgs.visualize.initialValueSet = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

% see a 2D slice
HJIextraArgs.visualize.plotData.plotDims = [1 1 0]; %plot x, y, theta
HJIextraArgs.visualize.plotData.projpt = [pi/2];
HJIextraArgs.visualize.viewAngle = [0,90]; % view 2D

HJIextraArgs.saveFilename = 'temp_brt_rc_wh';
HJIextraArgs.saveFrequency = 5;

% HJIPDE_solve's own "Total execution time" is cputime, summed over threads.
wallTimer = tic;
cpuStart  = cputime;

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, 'minVOverTime', HJIextraArgs);

%% Export, using the same variable layout as brt_rc_wh_fine_v3.mat

Vx = data(:,:,:,end);
Deriv = computeGradients(g, Vx);
uOpt = sys.optCtrl([], g.xs, Deriv, uMode);

save(filename, 'Vx', 'lx', 'tau2', 'g', 'Deriv', 'uOpt', '-v7.3')

%% Export BRT for Python
% dV/dtheta recovers the optimal steering angle but not the optimal speed, so
% export_3d_brt_with_gradient() also writes x_deriv and y_deriv.

export_3d_brt_with_gradient('brt_rc_wh_environment.mat', g, data, data0, pdDims, sys)

%% Timing report

wallSec = toc(wallTimer);
cpuSec  = cputime - cpuStart;

fprintf('\nSolve + export timing\n');
fprintf('  wall-clock time : %10.1f s  (%.2f h)\n', wallSec, wallSec/3600);
fprintf('  CPU time        : %10.1f s  (%.2f h)\n', cpuSec,  cpuSec/3600);
fprintf('  mean busy threads: %9.1fx  (maxNumCompThreads = %d)\n', ...
        cpuSec / max(wallSec, eps), maxNumCompThreads);

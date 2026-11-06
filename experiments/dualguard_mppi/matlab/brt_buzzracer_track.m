clc
clear
% Six-state BRT for the Buzzracer dynamic bicycle on the RCP tile track.
% Track geometry from buzzracer_track_data.mat (utils/raceline_to_mat.py).
% Writes brt_buzzracer_environment.mat.

SMOKE_TEST = false;   % tiny grid + short horizon, to check the whole path runs

%% problem parameters

x_0 = [2.1; 2.1; -pi/2; 1.0; 0.0; 0.0];   % raceline start_pos / start_dir

deltaMax      = deg2rad(26.1);   % car.py: audi_11 max_steering_left/right
throttleRange = [-1.0 1.0];      % car.py: min_throttle / max_throttle
dMax          = [0.1 0.1];       % immrax_controller.py: icentpert(0, [1e-1 1e-1])

% Control maximises the value function (stays safe), disturbance opposes it
uMode = 'max';
dMode = 'min';

filename = 'brt_buzzracer_environment.mat';

%% Grid

% x, y, heading, v_forward, v_sideway, omega
grid_min = [-0.05; -0.05; -pi; -0.40; -1.50; -14.0];
grid_max = [ 2.45;  3.65;  pi;  3.20;  1.50;  14.0];
N        = [   39;    57;   33;    28;     5;    11];   % 1.13e8 points

t0   = 0.0;
tMax = 2.0;
dt   = 0.05;

if SMOKE_TEST
  N    = [24; 35; 18; 14; 4; 7];
  tMax = 0.1;
  dt   = 0.05;
  filename = 'brt_buzzracer_environment_smoke.mat';
end

pdDims = 3;                      % heading is periodic
g = createGridND(grid_min, grid_max, N, pdDims);   % createGrid caps at 5-D

%% Pack problem parameters

sys = BuzzracerDynamicBicycle(x_0, deltaMax, throttleRange, dMax);

schemeData.grid = g;
schemeData.dynSys = sys;
schemeData.accuracy = 'high';   % WENO5
schemeData.uMode = uMode;
schemeData.dMode = dMode;

%% Grid report

labels = {'x', 'y', 'psi', 'v_fwd', 'v_side', 'omega'};

alphaMax = zeros(g.dim, 1);
for i = 1:g.dim
  a = sys.partialFunc([], [], [], [], schemeData, i);
  alphaMax(i) = max(a(:));
end
cfl = alphaMax ./ g.dx;

fprintf('Grid: %s = %.3g cells\n', mat2str(g.N'), prod(g.N));
for i = 1:g.dim
  fprintf('  %-7s [%7.3f, %7.3f]  N %4d  d %8.4f  max|dx| %8.3f  CFL %8.2f\n', ...
          labels{i}, g.min(i), g.max(i), g.N(i), g.dx(i), alphaMax(i), cfl(i));
end
fprintf('  CFL sum %.1f  ->  dt ~ %.3e s, ~%d steps for %.2f s\n', ...
        sum(cfl), 0.8/sum(cfl), ceil(tMax/(0.8/sum(cfl))), tMax);

arrayGiB = prod(g.N)*8/2^30;
fprintf('  one full array: %.2f GiB (double), %.2f GiB (single)\n', ...
        arrayGiB, arrayGiB/2);

% Empirical fit from probe solves on this machine (peak RSS vs. array size)
peakGiB = 1.04 + 49.6*arrayGiB;
totalGiB = NaN;
if isfile('/proc/meminfo')
  tok = regexp(fileread('/proc/meminfo'), 'MemTotal:\s+(\d+) kB', 'tokens', 'once');
  if ~isempty(tok)
    totalGiB = str2double(tok{1}) / 2^20;
  end
end
fprintf('  predicted peak RSS: %.1f GiB (system RAM %.1f GiB)\n', peakGiB, totalGiB);
if ~isnan(totalGiB) && peakGiB > 0.85*totalGiB
  warning('brt_buzzracer:memory', ...
          ['Predicted peak %.1f GiB is above 85%% of the %.1f GiB on this ' ...
           'machine. Reduce N before starting a multi-hour solve.'], ...
          peakGiB, totalGiB)
end

%% target set
% Failure set = everything off the track; depends only on (x, y).

T = load('buzzracer_track_data.mat');

lx = shapeRCPTrack2D(g, T.wall_segments, T.wall_arcs, T.tile_grid, ...
                     T.tile_scale, T.tile_deadzone);

% Check against the reference failure set computed in Python, on its own grid
gRef = createGrid([T.lx_ref_x(1); T.lx_ref_y(1)], ...
                  [T.lx_ref_x(end); T.lx_ref_y(end)], ...
                  [numel(T.lx_ref_x); numel(T.lx_ref_y)]);
lxRef = shapeRCPTrack2D(gRef, T.wall_segments, T.wall_arcs, T.tile_grid, ...
                        T.tile_scale, T.tile_deadzone);
err = abs(lxRef - T.lx_ref);
fprintf('lx vs. Python reference: max %.2e, rms %.2e, sign mismatches %d\n', ...
        max(err(:)), sqrt(mean(err(:).^2)), nnz(sign(lxRef) ~= sign(T.lx_ref)));
if max(err(:)) > 1e-9
  error('shapeRCPTrack2D disagrees with the reference failure set')
end
clear gRef lxRef err

data0 = repmat(lx, [1 1 g.N(3:end)']);

%% time vector
tau = t0:dt:tMax;

%% Compute value function

% tMax is a cap, not a target: stop once the tube stops moving. 1e-4 m is
% 0.15% of a cell width per dt, and 10x looser than helperOC's default because
% WENO5 leaves an oscillatory noise floor that can block the tighter one.
HJIextraArgs.stopConverge = true;
HJIextraArgs.convergeThreshold = 1e-4;

% ignoreBoundary is left off: it trims 4 cells per face and v_sideway has 5.

% Keep only the latest time slice; `data` comes back as a g.dim array.
HJIextraArgs.keepLast = true;

% No visualization: helperOC cannot render a 6-D value function.

HJIextraArgs.saveFilename = 'temp_brt_buzzracer';
HJIextraArgs.saveFrequency = 20;

% HJIPDE_solve's own "Total execution time" is cputime, summed over threads.
wallTimer = tic;
cpuStart  = cputime;

[data, tau2] = ...
  HJIPDE_solve(data0, tau, schemeData, 'minVOverTime', HJIextraArgs);

%% Sanity check on the converged solution

% minVOverTime takes a running minimum against lx, so V can never exceed it
excess = max(data(:) - data0(:));
fprintf('max(V - lx) = %.3e (should be ~0 for minVOverTime)\n', excess);
if ~isfinite(excess) || excess > 1e-6
  warning('brt_buzzracer:minVOverTime', ...
          'V exceeds the failure set by %.3e -- solve may have diverged', excess)
end
clear data0

%% Export for the Python controller

exportOpts.saveGradients = true;
% Bang-bang optimal inputs, baked in so Python need not interpolate gradients
exportOpts.saveOptInputs = true;
exportOpts.precision = 'single';

export_brt_with_gradient(filename, g, data, lx, pdDims, sys, ...
                         uMode, dMode, exportOpts)

%% Timing report

wallSec = toc(wallTimer);
cpuSec  = cputime - cpuStart;

fprintf('\nSolve + export timing\n');
fprintf('  wall-clock time : %10.1f s  (%.2f h)\n', wallSec, wallSec/3600);
fprintf('  CPU time        : %10.1f s  (%.2f h)\n', cpuSec,  cpuSec/3600);
fprintf('  mean busy threads: %9.1fx  (maxNumCompThreads = %d)\n', ...
        cpuSec / max(wallSec, eps), maxNumCompThreads);

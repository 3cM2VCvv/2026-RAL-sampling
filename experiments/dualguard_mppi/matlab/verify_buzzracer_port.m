function verify_buzzracer_port()
% verify_buzzracer_port()
%   Checks every piece of the Buzzracer track + dynamic bicycle port that can
%   be checked without running the multi-day BRT solve. Run this after
%   changing any of:
%       shapeRCPTrack2D.m, @BuzzracerDynamicBicycle/*, buzzracer_track_data.mat
%
%   Requires helperOC and ToolboxLS on the path, plus the two fixtures
%   produced by utils/raceline_to_mat.py and utils/gen_dynamics_reference.py.
%
%   Checks, in order:
%     1. failure set     -- shapeRCPTrack2D against the Python reference,
%                           itself validated against buzzracer's own
%                           RCPTrack.check_track_boundary
%     2. dynamics        -- against the jax model it was ported from,
%                           on both branches and exactly at the switch
%     3. optCtrl/optDstb -- against a brute-force scan of the input box
%     4. partialFunc     -- alpha must upper-bound |dx| over every input
%     5. input bounds    -- optimal inputs stay inside their boxes

fprintf('\n=== 1. failure set ===\n');
checkFailureSet();

fprintf('\n=== 2. dynamics vs. the jax model ===\n');
checkDynamics();

fprintf('\n=== 3. optimal control and disturbance ===\n');
checkOptInputs();

fprintf('\n=== 4. Lax-Friedrichs alpha bounds ===\n');
checkPartial();

fprintf('\nAll checks passed.\n');

end


% -------------------------------------------------------------------------
function checkFailureSet()

T = load('buzzracer_track_data.mat');

g = createGrid([T.lx_ref_x(1); T.lx_ref_y(1)], ...
               [T.lx_ref_x(end); T.lx_ref_y(end)], ...
               [numel(T.lx_ref_x); numel(T.lx_ref_y)]);

lx = shapeRCPTrack2D(g, T.wall_segments, T.wall_arcs, T.tile_grid, ...
                     T.tile_scale, T.tile_deadzone);

err = abs(lx - T.lx_ref);
nSign = nnz(sign(lx) ~= sign(T.lx_ref));
fprintf('  %d segments, %d arcs, %d grid points\n', ...
        size(T.wall_segments,1), size(T.wall_arcs,1), numel(lx));
fprintf('  max err %.3e, rms %.3e, sign mismatches %d\n', ...
        max(err(:)), sqrt(mean(err(:).^2)), nSign);
assertTrue(max(err(:)) < 1e-9 && nSign == 0, 'failure set disagrees with reference');

% The raceline must lie strictly inside the drivable surface
d = interpn(g.vs{1}, g.vs{2}, lx, T.raceline_xy(:,1), T.raceline_xy(:,2));
fprintf('  raceline min clearance %.4f m over %d samples\n', min(d), numel(d));
assertTrue(min(d) > 0, 'raceline leaves the drivable surface');

end


% -------------------------------------------------------------------------
function checkDynamics()

R = load('buzzracer_dynamics_reference.mat');
sys = BuzzracerDynamicBicycle();

assertTrue(abs(sys.lf - R.ref_lf) < 1e-12 && abs(sys.lr - R.ref_lr) < 1e-12 && ...
           abs(sys.Iz - R.ref_Iz) < 1e-15 && abs(sys.mass - R.ref_m) < 1e-12, ...
           'body parameters differ from the reference');

n = size(R.ref_x, 1);

% Cell/vectorised path -- the one HJIPDE_solve uses
xc = cell(6,1);  for k = 1:6, xc{k} = R.ref_x(:,k); end
dxc = sys.dynamics(0, xc, {R.ref_u(:,1); R.ref_u(:,2)}, ...
                          {R.ref_w(:,1); R.ref_w(:,2)});
got = zeros(n, 6);  for k = 1:6, got(:,k) = dxc{k}; end
err = abs(got - R.ref_xdot);

kin = R.ref_x(:,4) < sys.vSwitch;
at  = R.ref_x(:,4) == sys.vSwitch;
fprintf('  %d samples: max abs err %.3e\n', n, max(err(:)));
fprintf('    kinematic branch (%4d): %.3e\n', nnz(kin),  max(max(err(kin,:))));
fprintf('    dynamic   branch (%4d): %.3e\n', nnz(~kin), max(max(err(~kin,:))));
fprintf('    exactly at vSwitch (%d): %.3e\n', nnz(at),  max(max(err(at,:))));
assertTrue(max(err(:)) < 1e-10, 'dynamics disagree with the jax model');

% Single-state path must agree with the vectorised one
sub = 1:50:n;
for i = sub
  dxi = sys.dynamics(0, R.ref_x(i,:)', {R.ref_u(i,1); R.ref_u(i,2)}, ...
                                       {R.ref_w(i,1); R.ref_w(i,2)});
  assertTrue(max(abs(dxi(:)' - got(i,:))) < 1e-14, ...
             'single-state and vectorised dynamics paths disagree');
end
fprintf('    single-state path matches vectorised on %d spot checks\n', numel(sub));

end


% -------------------------------------------------------------------------
function checkOptInputs()

rng(3)
sys = BuzzracerDynamicBicycle();

n = 2000;
X = [rand(n,1)*2.5-0.05, rand(n,1)*3.7-0.05, rand(n,1)*2*pi-pi, ...
     rand(n,1)*3.6-0.4,  rand(n,1)*3-1.5,    rand(n,1)*28-14];
X(1:400,4) = rand(400,1)*0.5 - 0.4;      % kinematic branch
P = randn(n,6);
P(1:150,5:6) = P(1:150,5:6)*1e-12;       % exercise the k ~ 0 tie-break

xc = cell(6,1);  for k = 1:6, xc{k} = X(:,k); end
pc = cell(6,1);  for k = 1:6, pc{k} = P(:,k); end

nScan = 2001;
deltas = linspace(-sys.deltaMax, sys.deltaMax, nScan);
thr = sys.throttleRange;
kin = X(:,4) < sys.vSwitch;

for mode = {'max', 'min'}
  m = mode{1};
  dm = 'min';  if strcmp(m, 'min'), dm = 'max'; end

  u = sys.optCtrl(0, xc, pc, m);
  d = sys.optDstb(0, xc, pc, dm);
  Hopt = ham(sys, xc, pc, u, d);

  best = -inf(n,1);  if strcmp(m,'min'), best = inf(n,1); end
  for a = 1:nScan
    for b = 1:2
      H = ham(sys, xc, pc, {deltas(a)*ones(n,1); thr(b)*ones(n,1)}, d);
      if strcmp(m,'max'), best = max(best, H); else, best = min(best, H); end
    end
  end
  short = Hopt - best;
  if strcmp(m,'max'), short = -short; end     % positive means the scan won

  fprintf('  uMode %s vs %d-point scan: worst shortfall %.3e ', ...
          m, nScan*2, max(short));
  fprintf('(dynamic branch %.3e)\n', max(short(~kin)));
  % The dynamic branch is closed-form and must be exact; the kinematic branch
  % uses a finite scan of its own, so it is allowed a small gap.
  assertTrue(max(short(~kin)) < 1e-12, 'closed-form optCtrl is not optimal');
  assertTrue(max(short) < 1e-4, 'kinematic-branch optCtrl is far from optimal');

  % Disturbance
  nd = 101;
  ds = linspace(-sys.dMax(1), sys.dMax(1), nd);
  bestD = inf(n,1);  if strcmp(dm,'max'), bestD = -inf(n,1); end
  for a = 1:nd
    for b = 1:nd
      H = ham(sys, xc, pc, u, {ds(a)*ones(n,1); ds(b)*ones(n,1)});
      if strcmp(dm,'max'), bestD = max(bestD, H); else, bestD = min(bestD, H); end
    end
  end
  shortD = ham(sys, xc, pc, u, d) - bestD;
  if strcmp(dm,'max'), shortD = -shortD; end
  fprintf('  dMode %s vs %d-point scan: worst shortfall %.3e\n', ...
          dm, nd*nd, max(shortD));
  assertTrue(max(shortD) < 1e-12, 'optDstb is not optimal');

  assertTrue(all(abs(u{1}) <= sys.deltaMax + 1e-12), 'steering out of bounds');
  assertTrue(all(u{2} >= thr(1)-1e-12 & u{2} <= thr(2)+1e-12), 'throttle out of bounds');
  assertTrue(all(abs(d{1}) <= sys.dMax(1)+1e-12), 'd1 out of bounds');
  assertTrue(all(abs(d{2}) <= sys.dMax(2)+1e-12), 'd2 out of bounds');
end

end


% -------------------------------------------------------------------------
function checkPartial()

sys = BuzzracerDynamicBicycle();
g = createGridND([-0.05; -0.05; -pi; -0.40; -1.50; -14.0], ...
                 [ 2.45;  3.65;  pi;  3.20;  1.50;  14.0], ...
                 [4; 5; 9; 11; 5; 9], 3);
schemeData.grid = g;

alpha = cell(6,1);
for i = 1:6
  alpha{i} = sys.partialFunc([], [], [], [], schemeData, i);
end

nScan = 25;
deltas = linspace(-sys.deltaMax, sys.deltaMax, nScan);
worst = -inf(6,1);
for a = 1:nScan
  for b = 1:2
    for c = 1:2
      for e = 1:2
        u = {deltas(a)*ones(g.shape); sys.throttleRange(b)*ones(g.shape)};
        d = {(2*c-3)*sys.dMax(1)*ones(g.shape); (2*e-3)*sys.dMax(2)*ones(g.shape)};
        dx = sys.dynamics(0, g.xs, u, d);
        for i = 1:6
          worst(i) = max(worst(i), max(abs(dx{i}(:)) - alpha{i}(:)));
        end
      end
    end
  end
end

labels = {'x','y','psi','v_fwd','v_side','omega'};
fprintf('  %d grid points x %d input combinations\n', prod(g.N), nScan*8);
for i = 1:6
  fprintf('    %-7s max(|dx| - alpha) = %+.3e\n', labels{i}, worst(i));
end
assertTrue(all(worst <= 0), 'partialFunc underestimates alpha -- scheme is unstable');

end


% -------------------------------------------------------------------------
function H = ham(sys, xc, pc, u, d)
dx = sys.dynamics(0, xc, u, d);
H = zeros(size(xc{1}));
for k = 1:6
  H = H + pc{k} .* dx{k};
end
end


function assertTrue(cond, msg)
if ~all(cond(:))
  error('verify_buzzracer_port:failed', '%s', msg)
end
end

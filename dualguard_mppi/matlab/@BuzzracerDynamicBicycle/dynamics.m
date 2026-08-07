function dx = dynamics(obj, ~, x, u, d)
% dx = dynamics(obj, t, x, u, d)
%   Six-state dynamic bicycle, ported term for term from
%   DynamicBicycleCartesian.f() in
%   ../online-reachability/immrax_dynamic_bicycle_cartesian.py
%
%   x = {xpos, ypos, heading, v_forward, v_sideway, omega}
%   u = {steering, throttle}
%   d = {w_v_sideway, w_v_heading}
%
%   Cells above the vSwitch speed use the dynamic (tire force) branch; cells
%   below it use the kinematic branch. The two are evaluated on disjoint
%   subsets rather than blended, because the dynamic branch divides by
%   v_forward and would produce NaNs on the low-speed cells.
%
%   Two ordering details are inherited from the source and matter:
%     * In the dynamic branch, a_sideway and a_heading are computed from the
%       UNDISTURBED v_sideway and omega; the disturbances are added only
%       afterwards, where they rotate into the global velocity and the
%       heading rate.
%     * In the kinematic branch, w_v_sideway is added to the global ypos
%       rate directly, not rotated by the heading.

if nargin < 5
  d = {0; 0};
end

if ~iscell(u)
  u = num2cell(u);
end

if ~iscell(d)
  d = num2cell(d);
end

if ~iscell(x)
  % Single state: evaluate the branch that applies and return a column
  xc = num2cell(x);
  dxc = branchAll(obj, xc, u, d, x(4) >= obj.vSwitch);
  dx = zeros(obj.nx, 1);
  for i = 1:obj.nx
    dx(i) = dxc{i};
  end
  return
end

sz = size(x{1});
useDynamic = x{4} >= obj.vSwitch;

dx = cell(obj.nx, 1);

if all(useDynamic(:))
  dx = branchAll(obj, x, u, d, true);
  return
elseif ~any(useDynamic(:))
  dx = branchAll(obj, x, u, d, false);
  return
end

% Mixed: evaluate each branch on its own subset
for i = 1:obj.nx
  dx{i} = zeros(sz);
end

for pass = 1:2
  isDyn = (pass == 1);
  m = useDynamic;
  if ~isDyn
    m = ~m;
  end

  xs = cell(obj.nx, 1);
  for i = 1:obj.nx
    xs{i} = subset(x{i}, m);
  end
  us = {subset(u{1}, m); subset(u{2}, m)};
  ds = {subset(d{1}, m); subset(d{2}, m)};

  dxs = branchAll(obj, xs, us, ds, isDyn);
  for i = 1:obj.nx
    dx{i}(m) = dxs{i};
  end
end

end


function v = subset(v, m)
% Index into v with the logical mask m, leaving scalars alone
if ~isscalar(v)
  v = v(m);
end
end


function dx = branchAll(obj, x, u, d, isDynamic)
% Evaluate one branch of the model over whatever shape x/u/d have

dx = cell(6, 1);

% Longitudinal motor model is shared by both branches
a_forward = obj.motorCm * (u{2} - x{4}/obj.motorCv - obj.motorC0);

if isDynamic
  slip_f = u{1} - atan((x{6}*obj.lf + x{5}) ./ x{4});
  slip_r =        atan((x{6}*obj.lr - x{5}) ./ x{4});

  mr  = obj.massRatio();
  Ffy =                 obj.tireCurve(slip_f) * mr * obj.lr;
  Fry = obj.rearScale * obj.tireCurve(slip_r) * mr * obj.lf;

  a_sideway = (Fry + Ffy - obj.mass * x{4} .* x{6}) / obj.mass;
  a_heading = (Ffy*obj.lf - Fry*obj.lr) / obj.Iz;

  % Disturbance enters after the force computation
  v_sideway = x{5} + d{1};
  omega     = x{6} + d{2};

  c = cos(x{3});
  s = sin(x{3});

  dx{1} = x{4}.*c - v_sideway.*s;
  dx{2} = x{4}.*s + v_sideway.*c;
  dx{3} = omega;
  dx{4} = a_forward;
  dx{5} = a_sideway;
  dx{6} = a_heading;

else
  beta      = atan(obj.lr/obj.L * tan(u{1}));
  omega_ref = x{4} .* cos(beta) / obj.L .* tan(u{1});

  dx{1} = x{4} .* cos(x{3} + beta);
  dx{2} = x{4} .* sin(x{3} + beta) + d{1};
  dx{3} = omega_ref + d{2};
  dx{4} = a_forward;
  dx{5} = -x{5};
  dx{6} = -(x{6} - omega_ref);
end

end

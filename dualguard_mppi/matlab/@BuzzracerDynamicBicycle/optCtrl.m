function uOpt = optCtrl(obj, ~, x, deriv, uMode)
% uOpt = optCtrl(obj, t, x, deriv, uMode)
%   Optimal (steering, throttle) for the six-state dynamic bicycle.
%
% Throttle enters the Hamiltonian affinely as p4*Cm*throttle: bang-bang on
% sign(p4).
%
% Dynamic branch. Steering appears only inside Ffy, which feeds exactly
%   d/dt v_sideway += Ffy/m       d/dt omega += Ffy*lf/Iz
% so the steering dependence collapses to
%   H(delta) = const + k * Ffy(delta),      k = p5/m + p6*lf/Iz
%   Ffy(delta) = tire(delta + s0) * massRatio * lr
%   s0 = -atan((omega*lf + v_sideway)/v_forward)
% tire(s) = D*sin(C*atan(B*s)) is odd and has its only positive stationary
% point at s* = tan(pi/(2C))/B, so over the reachable slip interval
% [s0 - deltaMax, s0 + deltaMax] the optimum is s* (or -s*) if it lies
% inside, else the better endpoint -- a three-way comparison, no search.
%
% Kinematic branch. Below vSwitch steering enters through beta and omega_ref
% with no closed form, so those cells scan nDeltaScan candidates. Worst
% Hamiltonian shortfall 7.2e-6 at the default nDeltaScan = 21.

if nargin < 5
  uMode = 'max';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

if ~iscell(x)
  x = num2cell(x);
end

if ~strcmp(uMode, 'max') && ~strcmp(uMode, 'min')
  error('Unknown uMode!')
end
maximise = strcmp(uMode, 'max');

uOpt = cell(obj.nu, 1);

%% Throttle: bang-bang on the sign of p4

if maximise
  hi = obj.throttleRange(2);
  lo = obj.throttleRange(1);
else
  hi = obj.throttleRange(1);
  lo = obj.throttleRange(2);
end
uOpt{2} = (deriv{4} >= 0)*hi + (deriv{4} < 0)*lo;

%% Steering, dynamic branch: closed form

% Sensitivity of the Hamiltonian to the front lateral force
k = deriv{5}/obj.mass + deriv{6}*obj.lf/obj.Iz;

% Slip angle the front tire sees at zero steering
s0 = -atan((x{6}*obj.lf + x{5}) ./ x{4});

a = s0 - obj.deltaMax;
b = s0 + obj.deltaMax;

sPeak = obj.tirePeakSlip();
% Interior stationary point: the max sits at +s*, the min at -s*, and which
% one we want flips with the sign of k as well as with uMode.
% The zeros(size(s0)) term broadcasts these to the grid shape even when the
% costate happens to be passed in as a scalar.
wantMax = xor(~maximise, k >= 0) & true(size(s0));  % maximise tire(slip_f)?
sInt = wantMax*sPeak + (~wantMax)*(-sPeak);

fa = obj.tireCurve(a);
fb = obj.tireCurve(b);
fi = obj.tireCurve(sInt);

% The interior point only competes when it is actually reachable
reachable = (sInt >= a) & (sInt <= b);
fi(~reachable) = NaN;

% Pick whichever of the three candidates wins, in the direction wantMax says
better = @(p, q) (wantMax & (p >= q)) | (~wantMax & (p <= q));

slip = a;
fbest = fa;

take = better(fb, fbest);
slip(take) = b(take);
fbest(take) = fb(take);

take = reachable & better(fi, fbest);
slip(take) = sInt(take);

steerDyn = slip - s0;
% Guard against round-off pushing the answer a hair outside the bounds
steerDyn = min(max(steerDyn, -obj.deltaMax), obj.deltaMax);

uOpt{1} = steerDyn;

%% Steering, kinematic branch: scan

useKinematic = x{4} < obj.vSwitch;
if ~any(useKinematic(:))
  return
end

deltas = linspace(-obj.deltaMax, obj.deltaMax, obj.nDeltaScan);

% Restrict to the low-speed cells so the scan costs a fraction of a full grid
p1 = subset(deriv{1}, useKinematic);
p2 = subset(deriv{2}, useKinematic);
p3 = subset(deriv{3}, useKinematic);
p6 = subset(deriv{6}, useKinematic);
psi = subset(x{3}, useKinematic);
vf  = subset(x{4}, useKinematic);

if maximise
  bestVal = -inf(size(vf));
else
  bestVal = inf(size(vf));
end
bestDelta = zeros(size(vf));

for n = 1:numel(deltas)
  delta = deltas(n);
  beta = atan(obj.lr/obj.L * tan(delta));
  omega_ref = vf * (cos(beta)/obj.L * tan(delta));

  % Terms of H that depend on delta:
  %   p1*vf*cos(psi+beta) + p2*vf*sin(psi+beta) + p3*omega_ref + p6*omega_ref
  % (d/dt omega = -(omega - omega_ref), so omega_ref enters p6 with a + sign)
  val = p1.*vf.*cos(psi + beta) + p2.*vf.*sin(psi + beta) ...
        + (p3 + p6).*omega_ref;

  if maximise
    take = val > bestVal;
  else
    take = val < bestVal;
  end
  bestVal(take) = val(take);
  bestDelta(take) = delta;
end

uOpt{1}(useKinematic) = bestDelta;

end


function v = subset(v, m)
if isscalar(v)
  v = v * ones(nnz(m), 1);
else
  v = v(m);
end
end

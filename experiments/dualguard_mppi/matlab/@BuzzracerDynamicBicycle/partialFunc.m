function alpha = partialFunc(obj, ~, ~, ~, ~, schemeData, dim)
% alpha = partialFunc(obj, t, data, derivMin, derivMax, schemeData, dim)
%   Bound on |dH/dp_dim| for the Lax-Friedrichs dissipation term.
%
%   Closed-form bounds on |dx_dim| over every admissible control and
%   disturbance. genericPartial.m short-circuits to this method when the
%   system defines it, avoiding 24 full-grid model evaluations per
%   Lax-Friedrichs call. Each bound covers both branches of the model.

x = schemeData.grid.xs;

vf = abs(x{4});
vs = abs(x{5});
om = abs(x{6});

% Largest |omega_ref| the kinematic branch can command. Gated on the SIGNED
% v_forward, matching dynamics.m: a large negative v_forward is still the
% kinematic branch, so this cannot simply be clipped at vSwitch.
kinYaw = vf * (tan(obj.deltaMax) / obj.L);
kinYaw(x{4} >= obj.vSwitch) = 0;

switch obj.dims(dim)
  case {1, 2}
    % Dynamic:   |v_f*cos - (v_s + d1)*sin| <= |v_f| + |v_s| + d1max
    % Kinematic: |v_f*cos(psi+beta)| <= |v_f|, plus d1 on dimension 2
    alpha = vf + vs + obj.dMax(1);

  case 3
    % Dynamic: |omega + d2|;  kinematic: |omega_ref + d2|
    alpha = max(om, kinYaw) + obj.dMax(2);

  case 4
    % |Cm*(throttle - v_f/Cv - C0)|, maximised at the throttle extreme
    tExtreme = max(abs(obj.throttleRange));
    alpha = obj.motorCm * (tExtreme + vf/obj.motorCv + abs(obj.motorC0));

  case 5
    % Dynamic:   |(Fry + Ffy)/m - v_f*omega|
    % Kinematic: |-v_sideway|
    Fmax = obj.maxFrontForce() + obj.maxRearForce();
    alpha = max(Fmax/obj.mass + vf.*om, vs);

  case 6
    % Dynamic:   |(Ffy*lf - Fry*lr)/Iz|, a constant
    % Kinematic: |-(omega - omega_ref)|
    yawTorque = (obj.maxFrontForce()*obj.lf + obj.maxRearForce()*obj.lr) / obj.Iz;
    alpha = max(yawTorque, om + kinYaw);

  otherwise
    error('Cannot compute alpha for dimension %d!', obj.dims(dim))
end

% Every branch above must return something the size of the grid
if isscalar(alpha)
  alpha = alpha * ones(size(x{1}));
end

end

function dOpt = optDstb(obj, ~, x, deriv, dMode)
% dOpt = optDstb(obj, t, x, deriv, dMode)
%   Optimal disturbance for the six-state dynamic bicycle.
%
%   d = {w_v_sideway, w_v_heading}, each bounded by obj.dMax, matching the
%   box that ImmraxController wraps around the disturbance in
%   ../online-reachability/immrax_controller.py:
%       icentpert(zeros(2), [1e-1, 1e-1])
%
%   Both disturbances enter the model affinely, so both are bang-bang.
%
%   In the dynamic branch w_v_sideway is added to v_sideway after the tire
%   forces are computed, so it reaches the state derivative only through the
%   rotation into global coordinates:
%       d/dt xpos += -w_v_sideway*sin(psi)
%       d/dt ypos += +w_v_sideway*cos(psi)
%   giving it a Hamiltonian coefficient of
%       -p1*sin(psi) + p2*cos(psi)
%   i.e. the costate resolved along the vehicle's lateral axis.
%
%   In the kinematic branch the source adds w_v_sideway straight to the
%   global ypos rate instead, so its coefficient there is just p2. That
%   branch is only active below vSwitch.
%
%   w_v_heading is added to omega in both branches, so its coefficient is p3
%   either way.

if nargin < 5
  dMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

if ~iscell(x)
  x = num2cell(x);
end

if ~strcmp(dMode, 'max') && ~strcmp(dMode, 'min')
  error('Unknown dMode!')
end

dOpt = cell(obj.nd, 1);

% Coefficient of w_v_sideway in the Hamiltonian
lateral = -deriv{1}.*sin(x{3}) + deriv{2}.*cos(x{3});

useKinematic = x{4} < obj.vSwitch;
if any(useKinematic(:))
  if isscalar(lateral)
    lateral = lateral * ones(size(useKinematic));
  end
  p2 = deriv{2};
  if isscalar(p2)
    lateral(useKinematic) = p2;
  else
    lateral(useKinematic) = p2(useKinematic);
  end
end

if strcmp(dMode, 'max')
  dOpt{1} = obj.dMax(1) * sign0(lateral);
  dOpt{2} = obj.dMax(2) * sign0(deriv{3});
else
  dOpt{1} = -obj.dMax(1) * sign0(lateral);
  dOpt{2} = -obj.dMax(2) * sign0(deriv{3});
end

end


function s = sign0(v)
% sign(), but with zero mapped to +1 so the choice is a well-defined vertex
% of the disturbance box rather than the interior point 0
s = (v >= 0) - (v < 0);
end

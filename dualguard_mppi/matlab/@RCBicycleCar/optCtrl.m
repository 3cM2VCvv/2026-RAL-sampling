function uOpt = optCtrl(obj, ~, x, deriv, uMode)
% uOpt = optCtrl(obj, t, x, deriv, uMode)
%
% The Hamiltonian is
%   p1*v*cos(th) + p2*v*sin(th) + p3*(v/L)*tan(delta)
%     = v * ( p1*cos(th) + p2*sin(th) + p3*tan(delta)/L )
%
% Since v > 0 the steering angle is chosen first (it only interacts with p3),
% and the speed is then chosen from the sign of the whole bracket.

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

if ~iscell(x)
  x = num2cell(x);
end

uOpt = cell(obj.nu, 1);

% Coefficient of v coming from the position dynamics
posTerm = deriv{1} .* cos(x{3}) + deriv{2} .* sin(x{3});

%% Optimal control
if strcmp(uMode, 'max')
  uOpt{2} = (deriv{3} >= 0) * obj.deltaMax + (deriv{3} < 0) * (-obj.deltaMax);
  vCoeff  = posTerm + deriv{3} .* tan(uOpt{2}) / obj.wheelbase;
  uOpt{1} = (vCoeff >= 0) * obj.vRange(2) + (vCoeff < 0) * obj.vRange(1);

elseif strcmp(uMode, 'min')
  uOpt{2} = (deriv{3} >= 0) * (-obj.deltaMax) + (deriv{3} < 0) * obj.deltaMax;
  vCoeff  = posTerm + deriv{3} .* tan(uOpt{2}) / obj.wheelbase;
  uOpt{1} = (vCoeff >= 0) * obj.vRange(1) + (vCoeff < 0) * obj.vRange(2);

else
  error('Unknown uMode!')
end

end

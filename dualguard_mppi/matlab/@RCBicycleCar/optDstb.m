function dOpt = optDstb(obj, ~, ~, deriv, dMode)
% dOpt = optDstb(obj, t, y, deriv, dMode)
%     Disturbance acts on the position states only:
%         \dot{x}_1 = v * cos(x_3) + d_1
%         \dot{x}_2 = v * sin(x_3) + d_2
%         norm([d_1 d_2]) <= dMaxMag
%
% Note: brt_rc_wh_fine_v3.mat was generated with dMaxMag = 0 (see
% brt_rc_wh_track.m), so this is only here for completeness / reuse.

%% Input processing
if nargin < 5
  dMode = 'max';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

dOpt = cell(obj.nd, 1);

%% Optimal disturbance
if strcmp(dMode, 'max')
  error('Not implemented!')

elseif strcmp(dMode, 'min')
  % Scale factor is (max magnitude / gradient magnitude)
  derivNorm = sqrt( deriv{1}.^2 + deriv{2}.^2 );
  derivScaling = obj.dMaxMag ./ derivNorm;

  % If gradient magnitude is zero, this results in division by zero,
  %   so set scaling factor to 0 in these cases
  derivScaling(derivNorm == 0) = 0;

  % Opposite direction of value gradient
  dOpt{1} = -deriv{1} .* derivScaling;
  dOpt{2} = -deriv{2} .* derivScaling;
  dOpt{3} = 0;
else
  error('Unknown dMode!')
end

end

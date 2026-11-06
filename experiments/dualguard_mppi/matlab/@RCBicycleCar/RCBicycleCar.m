classdef RCBicycleCar < DynSys
  properties
    vRange      % [vMin vMax] longitudinal speed bounds (control 1)
    deltaMax    % steering-angle bound, |delta| <= deltaMax (control 2)
    wheelbase   % L, distance between front and rear axle
    dMaxMag     % Disturbance maximum magnitude (on the x-y position states)
    dims        % Dimensions that are active
  end

  methods
    function obj = RCBicycleCar(x, vRange, deltaMax, wheelbase, dMaxMag, dims)
      % obj = RCBicycleCar(x, vRange, deltaMax, wheelbase, dMaxMag, dims)
      %     Kinematic bicycle (rear-axle reference point) with speed and
      %     steering angle as controls. This is the model that was used to
      %     generate brt_rc_wh_fine_v3.mat; see brt_rc_wh_track.m for how
      %     the parameters below were recovered from that file.
      %
      % Dynamics:
      %    \dot{x}_1 = v * cos(x_3) + d1
      %    \dot{x}_2 = v * sin(x_3) + d2
      %    \dot{x}_3 = (v / L) * tan(delta) + d3
      %         v     \in [vRange(1), vRange(2)]
      %         delta \in [-deltaMax, deltaMax]
      %         norm(d1+d2) <= dMaxMag
      %
      % Inputs:
      %   x         - initial state [xpos; ypos; heading]
      %   vRange    - speed bounds
      %   deltaMax  - steering angle bound [rad]
      %   wheelbase - L [m]
      %   dMaxMag   - disturbance maximum magnitude
      %
      % Output:
      %   obj       - an RCBicycleCar object

      if numel(x) ~= 3
        error('Initial state does not have right dimension!');
      end

      if ~iscolumn(x)
        x = x';
      end

      if nargin < 2
        vRange = [0.7 1.4];
      end

      if nargin < 3
        deltaMax = deg2rad(25);
      end

      if nargin < 4
        wheelbase = 0.235;
      end

      if nargin < 5
        dMaxMag = 0;
      end

      if nargin < 6
        dims = 1:3;
      end

      if numel(vRange) < 2
        vRange = [-vRange; vRange];
      end

      % Basic vehicle properties
      obj.pdim = [find(dims == 1) find(dims == 2)]; % Position dimensions
      obj.nx = length(dims);
      obj.nu = 2;
      obj.nd = 3;

      obj.x = x;
      obj.xhist = obj.x;
      obj.vRange = vRange;
      obj.deltaMax = deltaMax;
      obj.wheelbase = wheelbase;
      obj.dMaxMag = dMaxMag;
      obj.dims = dims;
    end

  end % end methods
end % end classdef

classdef BuzzracerDynamicBicycle < DynSys
  % Six-state dynamic bicycle model of the Buzzracer 1/28-scale RC car.
  %
  % This is a MATLAB port of DynamicBicycleCartesian in
  %   ../online-reachability/immrax_dynamic_bicycle_cartesian.py
  % with the body parameters of car config `audi_11` from
  %   ../online-reachability/Buzzracer/buzzracer/cars/car.py
  % and the disturbance bounds from ImmraxController.__init__, which wraps the
  % disturbance in icentpert(zeros(2), [1e-1, 1e-1]).
  %
  % State  x = [xpos; ypos; heading; v_forward; v_sideway; omega]
  % Control u = [steering; throttle]
  % Disturb d = [w_v_sideway; w_v_heading]
  %
  % Above `vSwitch` m/s (the dynamic branch):
  %   d/dt xpos      = v_f*cos(psi) - (v_s + d1)*sin(psi)
  %   d/dt ypos      = v_f*sin(psi) + (v_s + d1)*cos(psi)
  %   d/dt heading   = omega + d2
  %   d/dt v_forward = Cm*(throttle - v_f/Cv - C0)
  %   d/dt v_sideway = (Fry + Ffy - m*v_f*omega) / m
  %   d/dt omega     = (Ffy*lf - Fry*lr) / Iz
  % with
  %   slip_f = steering - atan((omega*lf + v_s)/v_f)
  %   slip_r =            atan((omega*lr - v_s)/v_f)
  %   Ffy    =        tire(slip_f) * massRatio * lr
  %   Fry    = 1.15 * tire(slip_r) * massRatio * lf
  %   tire(s) = D*sin(C*atan(B*s)),  massRatio = m*9.8/L
  %
  % Below `vSwitch` the source switches to a kinematic branch (jax.lax.cond in
  % the immrax file), where the lateral and yaw states are driven towards
  % their kinematic values instead of being integrated from tire forces:
  %   beta      = atan(lr/L * tan(steering))
  %   omega_ref = v_f*cos(beta)/L * tan(steering)
  %   d/dt xpos      = v_f*cos(psi + beta)
  %   d/dt ypos      = v_f*sin(psi + beta) + d1
  %   d/dt heading   = omega_ref + d2
  %   d/dt v_forward = Cm*(throttle - v_f/Cv - C0)
  %   d/dt v_sideway = -v_s
  %   d/dt omega     = -(omega - omega_ref)

  properties
    % Body geometry and inertia (CarParams defaults, car config audi_11)
    lf          % CG to front axle [m]
    lr          % CG to rear axle [m]
    L           % wheelbase, lf + lr [m]
    mass        % [kg]
    Iz          % yaw inertia [kg m^2]

    % Pacejka-style lateral tire curve, tire(s) = D*sin(C*atan(B*s)).
    % NOTE these are the constants hardcoded in the immrax file, which are
    % NOT the B/C/Df/Dr fields of CarParams.
    tireB
    tireC
    tireD
    rearScale   % extra scale factor on the rear axle force (1.15 in source)
    gravity     % [m/s^2], 9.8 in source

    % Longitudinal motor model, a = Cm*(throttle - v/Cv - C0)
    motorCm
    motorCv
    motorC0

    % Input bounds
    deltaMax      % |steering| <= deltaMax [rad]
    throttleRange % [min max]
    dMax          % [max |w_v_sideway|, max |w_v_heading|]

    % Low-speed branch
    vSwitch     % v_forward below which the kinematic branch is used [m/s]
    nDeltaScan  % steering samples used for the kinematic branch's optCtrl

    dims        % Dimensions that are active
  end

  methods
    function obj = BuzzracerDynamicBicycle(x, deltaMax, throttleRange, dMax, params)
      % obj = BuzzracerDynamicBicycle(x, deltaMax, throttleRange, dMax, params)
      %
      % Inputs:
      %   x             - initial state, 6 x 1
      %   deltaMax      - steering bound [rad], default deg2rad(26.1)
      %   throttleRange - [min max], default [-1 1]
      %   dMax          - [d1max d2max], default [0.1 0.1]
      %   params        - optional struct overriding any of the physical
      %                   constants below by field name

      if nargin < 1 || isempty(x)
        x = zeros(6, 1);
      end

      if numel(x) ~= 6
        error('Initial state does not have right dimension!');
      end

      if ~iscolumn(x)
        x = x';
      end

      if nargin < 2 || isempty(deltaMax)
        deltaMax = deg2rad(26.1);       % car.py: audi_11 max_steering_left/right
      end

      if nargin < 3 || isempty(throttleRange)
        throttleRange = [-1 1];         % car.py: min_throttle / max_throttle
      end

      if nargin < 4 || isempty(dMax)
        dMax = [0.1 0.1];               % immrax_controller.py: icentpert(0, [1e-1 1e-1])
      end

      if nargin < 5
        params = struct();
      end

      if isscalar(dMax)
        dMax = [dMax dMax];
      end

      % --- defaults, then apply any overrides in `params` ---
      p.lf         = 0.04824;           % car.py: CarParams.lf
      p.lr         = 0.09 - 0.04824;    % car.py: CarParams.lr
      p.mass       = 0.1667;            % car.py: CarParams.m
      p.Iz         = 1/12 * 0.1667 * (0.15^2 + 0.1^2);
      p.tireB      = 2.3;               % immrax file: tire_curve()
      p.tireC      = 1.6;
      p.tireD      = 1.1;
      p.rearScale  = 1.15;
      p.gravity    = 9.8;
      p.motorCm    = 6.17;              % immrax file: a = 6.17*(u - v/15.2 - 0.333)
      p.motorCv    = 15.2;
      p.motorC0    = 0.333;
      p.vSwitch    = 0.1;               % immrax file: lt(v_forward, 0.1*ones_like)
      p.nDeltaScan = 21;
      p.dims       = 1:6;

      fn = fieldnames(params);
      for i = 1:numel(fn)
        if ~isfield(p, fn{i})
          error('Unknown parameter override ''%s''', fn{i});
        end
        p.(fn{i}) = params.(fn{i});
      end

      if mod(p.nDeltaScan, 2) == 0
        % Keep delta = 0 in the sample set
        p.nDeltaScan = p.nDeltaScan + 1;
      end

      obj.lf         = p.lf;
      obj.lr         = p.lr;
      obj.L          = p.lf + p.lr;
      obj.mass       = p.mass;
      obj.Iz         = p.Iz;
      obj.tireB      = p.tireB;
      obj.tireC      = p.tireC;
      obj.tireD      = p.tireD;
      obj.rearScale  = p.rearScale;
      obj.gravity    = p.gravity;
      obj.motorCm    = p.motorCm;
      obj.motorCv    = p.motorCv;
      obj.motorC0    = p.motorC0;
      obj.vSwitch    = p.vSwitch;
      obj.nDeltaScan = p.nDeltaScan;
      obj.dims       = p.dims;

      obj.deltaMax      = deltaMax;
      obj.throttleRange = throttleRange;
      obj.dMax          = dMax(:)';

      obj.pdim = [1 2];
      obj.hdim = 3;
      obj.vdim = 4:6;
      obj.nx = 6;
      obj.nu = 2;
      obj.nd = 2;

      obj.x = x;
      obj.xhist = obj.x;
    end

    function r = massRatio(obj)
      % Load transfer constant used to scale both tire forces
      r = obj.mass * obj.gravity / obj.L;
    end

    function f = tireCurve(obj, slip)
      f = obj.tireD * sin(obj.tireC * atan(obj.tireB * slip));
    end

    function s = tirePeakSlip(obj)
      % Slip angle at which tireCurve attains its global maximum tireD.
      % tireCurve is odd, rises monotonically on [-s*, s*], and decays
      % monotonically outside it, so s* and -s* are its only stationary
      % points. This is what makes the optimal steering closed-form.
      s = tan(pi / (2*obj.tireC)) / obj.tireB;
    end

    function f = maxFrontForce(obj)
      f = obj.tireD * obj.massRatio() * obj.lr;
    end

    function f = maxRearForce(obj)
      f = obj.rearScale * obj.tireD * obj.massRatio() * obj.lf;
    end

  end % end methods
end % end classdef

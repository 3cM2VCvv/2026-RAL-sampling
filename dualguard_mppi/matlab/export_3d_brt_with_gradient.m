function export_3d_brt_with_gradient(filename, g, value, init_value, pdDims, dynSys)

% NOTE: Hardcoded for a periodic 3rd dimension. See
% export_brt_with_gradient.m for the dimension-generic version.
assert(pdDims == 3)


%% Get value grid at final time step and ensure periodicity


% Value grid is final one along time dimension
value = value(:,:,:,end);


%% Sample optimal control at each grid point

% This is heavily based on code from helperOC/valFuncs/computeOptTraj

grid_axes = g.vs;

% Get gradients of value function for passing in deriv to optCtrl function
Deriv = computeGradients(g, value);     % This does level-set magic, apparently

    x_deriv = Deriv{1};
    y_deriv = Deriv{2};
theta_deriv = Deriv{3};

% Note: theta_deriv alone determines the optimal control of a Dubins car
% (whose only input is the turn rate), but a system with more inputs needs
% the other components too. For instance, the optimal speed of a bicycle
% model is set by the sign of
%   x_deriv*cos(theta) + y_deriv*sin(theta) + theta_deriv*tan(delta)/L
% so all three are exported.


%% Handle periodicity
% Append the index=1 values of dim 3 to end since phi is periodic
      value(:,:,end+1) =       value(:,:,1);
 init_value(:,:,end+1) =  init_value(:,:,1);
    x_deriv(:,:,end+1) =     x_deriv(:,:,1);
    y_deriv(:,:,end+1) =     y_deriv(:,:,1);
theta_deriv(:,:,end+1) = theta_deriv(:,:,1);

% Save grid_axes as a cell array: tuple() of the simplify_cells=True load
% gives the structure scipy's interpolator wants.
grid_axes{3}(end+1) = grid_axes{3}(end) + g.dx(3);

%% Write variables to file

save(filename, 'value', 'init_value', 'x_deriv', 'y_deriv', 'theta_deriv', ...
               'grid_axes')

end
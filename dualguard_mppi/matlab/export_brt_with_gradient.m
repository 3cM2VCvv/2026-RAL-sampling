function export_brt_with_gradient(filename, g, value, lx, pdDim, dynSys, uMode, dMode, opts)
% export_brt_with_gradient(filename, g, value, lx, pdDim, dynSys, uMode, dMode, opts)
%
%   Dimension-generic replacement for export_3d_brt_with_gradient(), which is
%   hardcoded to a 3-D grid with a periodic third dimension.
%
% Inputs:
%   filename - output .mat path
%   g        - grid structure
%   value    - value function from HJIPDE_solve. May carry a trailing time
%              dimension (the last slice is taken) or be the final slice
%              already, as when HJIextraArgs.keepLast was set.
%   lx       - failure set, as a compact array over whichever dimensions it
%              actually depends on; for a track it is 2-D over (x, y). Stored
%              as-is with its own axes rather than replicated up to the full
%              state grid. See "Reading this file" below.
%   pdDim    - index of the periodic dimension, or [] if there is none
%   dynSys   - dynamical system, used only when opts.saveOptInputs is set
%   uMode    - 'max' or 'min', as passed to HJIPDE_solve
%   dMode    - 'max' or 'min', as passed to HJIPDE_solve
%   opts     - optional struct:
%                .saveGradients  (default true)  write deriv_1 .. deriv_N
%                .saveOptInputs  (default false) write uOpt_* and dOpt_*
%                .precision      (default 'single') 'single' or 'double'
%
% Output file contents:
%   value           - final-time value function, periodic dimension wrapped
%   grid_axes       - 1 x g.dim cell of axis vectors, periodic axis extended
%                     to match. tuple() of the simplify_cells=1 load is the
%                     structure RegularGridInterpolator wants.
%   lx, lx_axes     - the failure set and the axis vectors it is defined on
%   lx_dims         - which state dimensions lx_axes correspond to
%   deriv_1 .. _N   - components of grad(value)             [saveGradients]
%   x_deriv, y_deriv, theta_deriv
%                   - aliases for deriv_1..3, written only when g.dim == 3 so
%                     that the existing utils/mat_to_h5.py keeps working
%   uOpt_1 .. _nu, dOpt_1 .. _nd
%                   - optimal inputs sampled on the grid    [saveOptInputs]
%   uMode, dMode, grid_min, grid_max, grid_N, pdDim
%                   - provenance
%
% Reading this file
% -----------------
% value and the gradients are full state-space arrays; lx is not. To test a
% state against the failure set, interpolate lx on lx_axes using only its
% lx_dims components -- the same answer replication would have given.
%
% The gradients recover both optimal inputs (closed-form in the costate;
% see optCtrl.m and optDstb.m). opts.saveOptInputs bakes them in instead,
% which suits their discontinuity: they want nearest-neighbour lookup, not
% the linear interpolation that suits the gradients.

if nargin < 7 || isempty(uMode), uMode = 'max'; end
if nargin < 8 || isempty(dMode), dMode = 'min'; end
if nargin < 9, opts = struct(); end

if ~isfield(opts, 'saveGradients'), opts.saveGradients = true;     end
if ~isfield(opts, 'saveOptInputs'), opts.saveOptInputs = false;    end
if ~isfield(opts, 'precision'),     opts.precision     = 'single'; end

if ~any(strcmp(opts.precision, {'single', 'double'}))
  error('opts.precision must be ''single'' or ''double''')
end
cast_ = str2func(opts.precision);

if numel(pdDim) > 1
  error('export_brt_with_gradient handles at most one periodic dimension')
end

%% Take the final time slice

% HJIPDE_solve returns a g.dim+1 dimensional array unless keepLast was set,
% in which case it is already g.dim dimensional. Indexing the trailing
% dimension with `end` is correct either way, since a g.dim dimensional array
% has a singleton dimension g.dim+1 whose end is 1.
clns = repmat({':'}, 1, g.dim);
value = value(clns{:}, size(value, g.dim+1));

%% Gradients, before the periodic wrap

if opts.saveGradients || opts.saveOptInputs
  Deriv = computeGradients(g, value);
else
  Deriv = {};
end

if opts.saveOptInputs
  uOpt = dynSys.optCtrl([], g.xs, Deriv, uMode);
  dOpt = dynSys.optDstb([], g.xs, Deriv, dMode);
else
  uOpt = {};
  dOpt = {};
end

%% Axes, with the periodic one extended by one cell

grid_axes = g.vs;
if ~isempty(pdDim)
  grid_axes{pdDim}(end+1) = grid_axes{pdDim}(end) + g.dx(pdDim);
end

%% Size report

n = prod(g.N);
if strcmp(opts.precision, 'single'), bytes = 4; else, bytes = 8; end
if ~isempty(pdDim)
  n = n / g.N(pdDim) * (g.N(pdDim) + 1);
end
nArrays = 1 + opts.saveGradients*g.dim;
fprintf('export_brt_with_gradient: %d full arrays of %d points (%s)\n', ...
        nArrays, n, opts.precision);
fprintf('  estimated size: %.2f GiB', nArrays*n*bytes / 2^30);
if opts.saveOptInputs
  fprintf(' + %d input arrays (%.2f GiB)', ...
          numel(uOpt) + numel(dOpt), (numel(uOpt)+numel(dOpt))*n*bytes / 2^30);
end
fprintf('\n');

%% Write

save(filename, 'grid_axes', '-v7.3')
save(filename, 'lx', '-append')

lx_axes = g.vs(1:ndims_of(lx));
lx_dims = 1:numel(lx_axes);
save(filename, 'lx_axes', 'lx_dims', '-append')

grid_min = g.min;  grid_max = g.max;  grid_N = g.N;
save(filename, 'grid_min', 'grid_max', 'grid_N', 'pdDim', ...
               'uMode', 'dMode', '-append')

value = cast_(wrapPeriodic(value, pdDim, g.dim));
save(filename, 'value', '-append')
clear value

names3d = {'x_deriv', 'y_deriv', 'theta_deriv'};
for i = 1:numel(Deriv)
  d = cast_(wrapPeriodic(Deriv{i}, pdDim, g.dim));
  Deriv{i} = [];                     % release the double copy as we go

  if opts.saveGradients
    S.(sprintf('deriv_%d', i)) = d;
    % Keep the 3-D names alive so utils/mat_to_h5.py still reads 3-D exports
    if g.dim == 3
      S.(names3d{i}) = d;
    end
    save(filename, '-struct', 'S', '-append')
    clear S
  end
  clear d
end

for i = 1:numel(uOpt)
  S.(sprintf('uOpt_%d', i)) = cast_(wrapPeriodic(uOpt{i}, pdDim, g.dim));
  uOpt{i} = [];
  save(filename, '-struct', 'S', '-append')
  clear S
end

for i = 1:numel(dOpt)
  S.(sprintf('dOpt_%d', i)) = cast_(wrapPeriodic(dOpt{i}, pdDim, g.dim));
  dOpt{i} = [];
  save(filename, '-struct', 'S', '-append')
  clear S
end

fprintf('  wrote %s\n', filename);

end


function a = wrapPeriodic(a, pdDim, dim)
% Duplicate the first slice of the periodic dimension at the end, so the
% consumer can interpolate across the wrap without special-casing it.
if isempty(pdDim)
  return
end
clns = repmat({':'}, 1, dim);
first = clns;
first{pdDim} = 1;
a = cat(pdDim, a, a(first{:}));
end


function n = ndims_of(a)
% ndims(), but reporting 1 for a vector rather than MATLAB's minimum of 2
if isvector(a)
  n = 1;
else
  n = ndims(a);
end
end

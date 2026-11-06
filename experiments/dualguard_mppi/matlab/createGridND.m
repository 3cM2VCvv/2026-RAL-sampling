function g = createGridND(grid_min, grid_max, N, pdDims)
% g = createGridND(grid_min, grid_max, N, pdDims)
%   Drop-in replacement for helperOC's createGrid() that also works above
%   five dimensions, where ToolboxLS's processGrid() errors out as a
%   guardrail. Delegates to createGrid() at five dimensions or fewer.
%
%   Fields match processGrid()'s: dim, min, max, N, dx, vs, xs, bdry,
%   bdryData, axis, shape. A periodic dimension has its max pulled in by one
%   cell so the last grid point does not duplicate the first.
%
% Inputs:
%   grid_min, grid_max - bounds on the computation domain, column vectors
%   N                  - number of grid points per dimension
%   pdDims             - indices of periodic dimensions (default none)
%
% Output:
%   g - grid structure

if nargin < 4
  pdDims = [];
end

grid_min = grid_min(:);
grid_max = grid_max(:);
N = N(:);

if numel(grid_min) ~= numel(grid_max) || numel(grid_min) ~= numel(N)
  error('grid_min, grid_max and N must have the same number of elements!')
end

if numel(grid_min) <= 5
  g = createGrid(grid_min, grid_max, N, pdDims);
  return
end

g.dim = numel(grid_min);
g.min = grid_min;
g.max = grid_max;
g.N = N;

g.bdry = cell(g.dim, 1);
for i = 1:g.dim
  if any(i == pdDims)
    g.bdry{i} = @addGhostPeriodic;
    % Drop the duplicate endpoint, exactly as createGrid does
    g.max(i) = g.min(i) + (g.max(i) - g.min(i)) * (1 - 1/g.N(i));
  else
    g.bdry{i} = @addGhostExtrapolate;
  end
end

g.dx = (g.max - g.min) ./ (g.N - 1);

% linspace rather than the colon operator: same points, cannot miscount
g.vs = cell(g.dim, 1);
for i = 1:g.dim
  g.vs{i} = linspace(g.min(i), g.max(i), g.N(i))';
end

g.xs = cell(g.dim, 1);
[g.xs{:}] = ndgrid(g.vs{:});

g.bdryData = cell(g.dim, 1);
[g.bdryData{:}] = deal([]);

g.axis = [];        % processGrid only populates this for dim 2 and 3
g.shape = g.N';

end

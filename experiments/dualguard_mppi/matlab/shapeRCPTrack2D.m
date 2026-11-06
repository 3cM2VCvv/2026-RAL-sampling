function lx = shapeRCPTrack2D(g, segments, arcs, tile_grid, scale, deadzone)
% lx = shapeRCPTrack2D(g, segments, arcs, tile_grid, scale, deadzone)
%   Implicit surface function for the Buzzracer RCP tile track.
%
%   lx > 0 on the drivable surface, lx < 0 in the failure set (off track),
%   and |lx| is the exact Euclidean distance to the track boundary.
%
%   The free space has interior islands, so unlike shapeTrack2D() the sign
%   cannot come from inpolygon() on two curves. The boundary is given as
%   straight walls plus circular arcs, and the sign comes from the tile-local
%   test that RCPTrack.check_track_boundary() uses.
%
% Inputs:
%   g        - grid structure.  Only the first two dimensions are read, and
%              only through g.vs, so this is safe to call on a large 6-D grid
%              without touching the (multi-GB) g.xs arrays.
%   segments - N x 4 array of wall segments, [x1 y1 x2 y2] per row
%   arcs     - M x 5 array of wall arcs, [cx cy radius th_start th_end] per
%              row, with th_start < th_end
%   tile_grid- cols x rows cell array of tile type strings, indexed
%              {col, row}: tile (i,j) covers x in [(i-1)*scale, i*scale] and
%              y in [(j-1)*scale, j*scale]
%   scale    - tile side length [m]
%   deadzone - wall half-width as a fraction of the tile side
%
%   All of these come straight out of buzzracer_track_data.mat, which is
%   produced by utils/raceline_to_mat.py.
%
% Output:
%   lx - g.N(1) x g.N(2) array of signed distances
%
% To lift this to the full state grid, the failure set does not depend on
% heading or velocity, so simply replicate along the remaining dimensions:
%   data0 = repmat(lx, [1 1 g.N(3:end)']);

x = g.vs{1}(:);
y = g.vs{2}(:);
[X, Y] = ndgrid(x, y);

%% Unsigned distance to the boundary

d2 = inf(size(X));

% Straight walls
for k = 1:size(segments, 1)
  a  = segments(k, 1:2);
  ab = segments(k, 3:4) - a;
  L2 = ab * ab';

  t = ((X - a(1))*ab(1) + (Y - a(2))*ab(2)) / L2;
  t = min(max(t, 0), 1);
  dx = X - (a(1) + t*ab(1));
  dy = Y - (a(2) + t*ab(2));

  d2 = min(d2, dx.^2 + dy.^2);
end

% Circular walls
for k = 1:size(arcs, 1)
  cx = arcs(k, 1);  cy = arcs(k, 2);  r = arcs(k, 3);
  th0 = arcs(k, 4); th1 = arcs(k, 5);

  vx = X - cx;
  vy = Y - cy;
  rho = sqrt(vx.^2 + vy.^2);

  % Inside the angular span the closest point is the radial projection;
  % outside it, one of the two arc endpoints.
  within = mod(atan2(vy, vx) - th0, 2*pi) <= (th1 - th0);

  % Within the angular span the radial distance is correct; outside it the
  % radial distance is meaningless (it measures to a part of the circle that
  % is not wall), so discard it and use the endpoints instead.
  dk2 = (rho - r).^2;
  dk2(~within) = inf;

  for th = [th0 th1]
    ex = cx + r*cos(th);
    ey = cy + r*sin(th);
    de2 = (X - ex).^2 + (Y - ey).^2;
    dk2(~within) = min(dk2(~within), de2(~within));
  end

  d2 = min(d2, dk2);
end

lx = sqrt(d2);

%% Sign: the tile-local drivable test

[cols, rows] = size(tile_grid);

% Points outside the tile grid are off track by definition: the grid covers
% the whole track.
on_track = false(size(X));

ii = floor(X / scale);
jj = floor(Y / scale);
in_grid = (ii >= 0) & (ii < cols) & (jj >= 0) & (jj < rows);

for i = 1:cols
  for j = 1:rows
    sel = in_grid & (ii == i-1) & (jj == j-1);
    if ~any(sel(:))
      continue
    end

    xl = X(sel)/scale - (i-1);
    yl = Y(sel)/scale - (j-1);
    tile = tile_grid{i, j};

    switch tile
      case 'WE'
        wl = yl - deadzone;
        wr = 1 - deadzone - yl;
      case 'NS'
        wl = xl - deadzone;
        wr = 1 - deadzone - xl;
      case {'SE', 'SW', 'NE', 'NW'}
        switch tile
          case 'SE', apex = [1 0];
          case 'SW', apex = [0 0];
          case 'NE', apex = [1 1];
          case 'NW', apex = [0 1];
        end
        rr = sqrt((xl - apex(1)).^2 + (yl - apex(2)).^2);
        wl = 1 - deadzone - rr;
        wr = rr - deadzone;
      otherwise
        error('shapeRCPTrack2D:tile', 'Unknown tile type ''%s''', tile)
    end

    on_track(sel) = min(wl, wr) > 0;
  end
end

lx(~on_track) = -lx(~on_track);

end

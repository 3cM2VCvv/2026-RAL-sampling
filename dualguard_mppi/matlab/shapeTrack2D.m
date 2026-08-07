function lx = shapeTrack2D(g, outer, inner)
% lx = shapeTrack2D(g, outer, inner)
%   Implicit surface function for a racetrack-style free space, i.e. the
%   region that is inside the closed curve `outer` and outside the closed
%   curve `inner`.
%
%   lx > 0 on the drivable track, lx < 0 in the failure set (off track), and
%   |lx| is the exact Euclidean distance to the track boundary.
%
% Inputs:
%   g     - grid structure (only the first two dimensions are used; the
%           result is returned on the 2D x-y slice of the grid)
%   outer - N x 2 array of vertices of the outer track boundary (closed,
%           do not repeat the first vertex)
%   inner - M x 2 array of vertices of the inner track boundary (infield)
%
% Output:
%   lx    - g.N(1) x g.N(2) array of signed distances
%
% To lift this to a 3D (x, y, theta) grid, the failure set does not depend
% on heading, so simply use
%   data0 = repmat(lx, [1 1 g.N(3)]);

X = g.xs{1}(:,:,1);
Y = g.xs{2}(:,:,1);

d_outer = polygonSignedDist(X, Y, outer);   % > 0 inside the room
d_inner = polygonSignedDist(X, Y, inner);   % > 0 inside the infield

% Distance to the failure set = distance to whichever boundary is closer
lx = min(d_outer, -d_inner);

end


function d = polygonSignedDist(X, Y, poly)
% Exact signed distance to a closed polygon; positive inside.

P = [poly; poly(1,:)];

d2 = inf(size(X));
for k = 1:size(P,1)-1
  a  = P(k,   :);
  ab = P(k+1, :) - a;
  L2 = ab*ab';

  if L2 < eps
    dx = X - a(1);
    dy = Y - a(2);
  else
    t = ((X - a(1))*ab(1) + (Y - a(2))*ab(2)) / L2;
    t = min(max(t, 0), 1);
    dx = X - (a(1) + t*ab(1));
    dy = Y - (a(2) + t*ab(2));
  end

  d2 = min(d2, dx.^2 + dy.^2);
end

d = sqrt(d2);

inside = inpolygon(X, Y, poly(:,1), poly(:,2));
d(~inside) = -d(~inside);

end

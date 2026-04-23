function positions = gen_hex_positions(num_cells, isd_m, tx_height)
%GEN_HEX_POSITIONS  Generate hexagonal-grid base-station positions.
%
%   POSITIONS = GEN_HEX_POSITIONS(NUM_CELLS, ISD_M, TX_HEIGHT) returns a
%   [3 x NUM_CELLS] matrix of BS positions (x, y, z) arranged in concentric
%   hexagonal rings centred at the origin.
%
%   Inputs:
%       num_cells  : number of BS sites (1, 7, 19, 37, ...)
%       isd_m      : inter-site distance in metres
%       tx_height  : antenna height in metres (z coordinate for all sites)
%
%   Output:
%       positions  : [3 x num_cells] double matrix; rows are (x, y, z)
%
%   Topology:
%       num_cells = 1  -> single site at origin
%       num_cells = 7  -> centre + ring-1 (6 sites)
%       num_cells = 19 -> centre + ring-1 + ring-2 (12 sites)
%       num_cells = 37 -> centre + ring-1 + ring-2 + ring-3 (18 sites)
%
%   For num_cells values that do not match a perfect hex ring count, the
%   function generates enough rings to cover the request and truncates to
%   exactly num_cells sites.
%
%   Example:
%       >> pos = gen_hex_positions(7, 500, 25);
%       >> size(pos)          % -> [3, 7]
%       >> norm(pos(1:2, 2))  % -> 500  (ring-1 radius equals ISD)

    if nargin < 3 || isempty(tx_height); tx_height = 25; end
    if nargin < 2 || isempty(isd_m);     isd_m = 500; end
    if nargin < 1 || isempty(num_cells); num_cells = 1; end

    assert(num_cells >= 1, 'num_cells must be >= 1');

    % Determine minimum number of hex rings needed.
    if num_cells == 1
        num_rings = 0;
    else
        num_rings = 0;
        while 1 + 3 * num_rings * (num_rings + 1) < num_cells
            num_rings = num_rings + 1;
            if num_rings > 50
                error('gen_hex_positions:TooManyRings', ...
                    'num_cells=%d requires > 50 hex rings', num_cells);
            end
        end
    end

    % Generate the full ring structure (Nx2 XY).
    xy = hex_ring_xy(num_rings, isd_m);

    % Truncate to requested count.
    if size(xy, 1) > num_cells
        xy = xy(1:num_cells, :);
    end

    % Build 3-row output.
    n = size(xy, 1);
    positions = zeros(3, n);
    positions(1, :) = xy(:, 1).';
    positions(2, :) = xy(:, 2).';
    positions(3, :) = tx_height;
end


function xy = hex_ring_xy(num_rings, isd_m)
%HEX_RING_XY  Return Nx2 XY positions for hex rings 0..num_rings.
%
%   Ring 0 is the origin.  Each subsequent ring r has 6*r sites placed at
%   the vertices and edges of a regular hexagon with circumradius r*isd_m.

    if num_rings == 0
        xy = [0, 0];
        return;
    end

    % Unit hex corner directions (pointy-top orientation).
    corners = [ 1.0,               0.0;
                0.5,  sqrt(3)/2;
               -0.5,  sqrt(3)/2;
               -1.0,               0.0;
               -0.5, -sqrt(3)/2;
                0.5, -sqrt(3)/2];

    total = 1 + 3 * num_rings * (num_rings + 1);
    xy = zeros(total, 2);
    xy(1, :) = [0, 0];  % centre site
    idx = 1;

    for r = 1:num_rings
        for c = 1:6
            start_pt = r * corners(c, :);
            next_c = mod(c, 6) + 1;
            end_pt = r * corners(next_c, :);
            for k = 0:(r - 1)
                t = k / r;
                idx = idx + 1;
                xy(idx, :) = ((1 - t) * start_pt + t * end_pt) * isd_m;
            end
        end
    end
end

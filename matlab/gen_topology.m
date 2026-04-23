function [bs_positions, bs_orientations, bs_pcis] = gen_topology(num_cells, isd_m, tx_height_m, sectors)
%GEN_TOPOLOGY  Hex K-site topology with optional 3-sector split.
%
%   [BS_POSITIONS, BS_ORIENTATIONS, BS_PCIS] = GEN_TOPOLOGY(NUM_CELLS, ISD_M,
%   TX_HEIGHT_M, SECTORS) builds a deterministic layout of K base-station
%   cells (site x sector).
%
%   Inputs:
%       num_cells    : 1 | 7 | 19  (interpreted as total cells — for
%                      SECTORS=3 it is the number of sites * 3 only when
%                      num_cells is divisible by 3; otherwise num_cells is
%                      taken as the number of sites and each site yields
%                      SECTORS sectors, so the true K = num_sites*sectors).
%                      To keep parity with Python topology.hex_grid we use:
%                      num_cells as the target number of SITES (1, 7, 19)
%                      and multiply by SECTORS to get total cells only when
%                      sectors > 1. Pass SECTORS=1 for omni.
%       isd_m        : inter-site distance in metres.
%       tx_height_m  : antenna height in metres.
%       sectors      : 1 (omni) or 3 (trisector azimuths 0/120/240 deg).
%
%   Outputs:
%       bs_positions    : 3 x K  (metres; x, y, z)
%       bs_orientations : 3 x K  (radians; bank/tilt/heading for QuaDRiGa).
%                         Only the heading (azimuth) is set here — tilt is
%                         applied by the antenna pattern (handled in
%                         qd_arrayant.generate('3gpp-mmw', ..., downtilt)).
%       bs_pcis         : 1 x K  integer PCIs (mod-3 plan: site_idx mod 3
%                         offset by sector so intra-site sectors differ).
%
%   Topology convention:
%       * num_cells = 1 -> single site at origin, 1 cell.
%       * num_cells = 7 -> centre + ring-1 (6 neighbours) of sites; with
%                          sectors=3 this becomes 21 cells.
%       * num_cells = 19 -> centre + ring-1 + ring-2 (18 neighbours); with
%                           sectors=3 this becomes 57 cells.
%
%   Manual verification:
%       >> [p, o, ids] = gen_topology(7, 500, 25, 1);
%       >> size(p)          % -> [3, 7]
%       >> norm(p(:,2))     % -> 500  (ring-1 radius equals ISD)
%       >> all(diff(ids))   % PCIs distinct across cells

    if nargin < 4 || isempty(sectors); sectors = 1; end
    if nargin < 3 || isempty(tx_height_m); tx_height_m = 25; end
    if nargin < 2 || isempty(isd_m); isd_m = 500; end
    if nargin < 1 || isempty(num_cells); num_cells = 1; end

    % ------------------------------------------------------------------
    % Site positions
    % ------------------------------------------------------------------
    switch num_cells
        case 1
            num_rings = 0;
        case 7
            num_rings = 1;
        case 19
            num_rings = 2;
        otherwise
            % Allow arbitrary (treated as ring count-derived).
            % Find smallest num_rings with 1 + 3*R*(R+1) >= num_cells.
            num_rings = 0;
            while 1 + 3*num_rings*(num_rings+1) < num_cells
                num_rings = num_rings + 1;
            end
    end
    xy = hex_ring_positions(num_rings, isd_m);  % Nx2
    num_sites = size(xy, 1);
    if num_sites > num_cells
        % keep only first num_cells for consistency
        xy = xy(1:num_cells, :);
        num_sites = num_cells;
    end

    % ------------------------------------------------------------------
    % Expand by sectors
    % ------------------------------------------------------------------
    if sectors == 1
        sector_azimuths = 0;
    elseif sectors == 3
        sector_azimuths = [0, 120, 240];
    else
        error('gen_topology:BadSectors', 'sectors must be 1 or 3');
    end
    num_cells_total = num_sites * sectors;

    bs_positions = zeros(3, num_cells_total);
    bs_orientations = zeros(3, num_cells_total);
    bs_pcis = zeros(1, num_cells_total);

    k = 0;
    for i_site = 1:num_sites
        for i_sec = 1:sectors
            k = k + 1;
            bs_positions(:, k) = [xy(i_site, 1); xy(i_site, 2); tx_height_m];
            az_deg = sector_azimuths(i_sec);
            % QuaDRiGa orientation = [bank; tilt; heading] in radians.
            bs_orientations(:, k) = [0; 0; deg2rad(az_deg)];
            % Mod-3 PCI plan: PCI = 3*site_idx + sector_idx (zero-based).
            bs_pcis(k) = 3 * (i_site - 1) + (i_sec - 1);
        end
    end
end


function xy = hex_ring_positions(num_rings, isd_m)
    % Return Nx2 XY positions of a hex lattice centred at origin.
    if num_rings == 0
        xy = [0, 0];
        return;
    end
    corners = [ 1.0,               0.0;
                0.5,  sqrt(3)/2;
               -0.5,  sqrt(3)/2;
               -1.0,               0.0;
               -0.5, -sqrt(3)/2;
                0.5, -sqrt(3)/2];
    xy = zeros(1 + 3*num_rings*(num_rings+1), 2);
    xy(1, :) = [0, 0];
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

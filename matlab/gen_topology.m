function [bs_positions, bs_orientations, bs_pcis] = gen_topology( ...
    num_cells, isd_m, tx_height_m, sectors, topology_layout, ...
    track_offset_m, hypercell_size)
%GEN_TOPOLOGY  Unified topology builder: hex or linear layout.
%
%   [BS_POSITIONS, BS_ORIENTATIONS, BS_PCIS] = GEN_TOPOLOGY(NUM_CELLS,
%   ISD_M, TX_HEIGHT_M, SECTORS, TOPOLOGY_LAYOUT, TRACK_OFFSET_M,
%   HYPERCELL_SIZE) builds a deterministic layout of K base-station cells.
%
%   Inputs:
%       num_cells        : number of sites (1, 7, 19 for hex; any for linear)
%       isd_m            : inter-site distance in metres
%       tx_height_m      : antenna height in metres
%       sectors          : 1 (omni) or 3 (trisector)
%       topology_layout  : 'hexagonal' (default) or 'linear'
%       track_offset_m   : perpendicular site offset for linear (default 80)
%       hypercell_size   : HyperCell PCI group size for linear (default 1)
%
%   Outputs:
%       bs_positions    : [3 x K]  (metres; x, y, z)
%       bs_orientations : [3 x K]  (radians; bank/tilt/heading)
%       bs_pcis         : [1 x K]  integer PCIs

    if nargin < 7 || isempty(hypercell_size); hypercell_size = 1; end
    if nargin < 6 || isempty(track_offset_m); track_offset_m = 80; end
    if nargin < 5 || isempty(topology_layout); topology_layout = 'hexagonal'; end
    if nargin < 4 || isempty(sectors); sectors = 1; end
    if nargin < 3 || isempty(tx_height_m); tx_height_m = 25; end
    if nargin < 2 || isempty(isd_m); isd_m = 500; end
    if nargin < 1 || isempty(num_cells); num_cells = 1; end

    if strcmpi(topology_layout, 'linear')
        % ------ Linear (rail-track) topology ------
        [bs_positions, bs_orientations] = gen_linear_positions( ...
            num_cells, isd_m, tx_height_m, track_offset_m, sectors);

        if hypercell_size > 1
            bs_pcis = assign_pci_hypercell(num_cells, sectors, hypercell_size);
        else
            bs_pcis = assign_pci_hypercell(num_cells, sectors, 1);
        end
    else
        % ------ Hexagonal topology ------
        [bs_positions, bs_orientations, bs_pcis] = gen_topology_hex( ...
            num_cells, isd_m, tx_height_m, sectors);
    end
end


% ======================================================================
% Hexagonal topology (original logic preserved)
% ======================================================================
function [bs_positions, bs_orientations, bs_pcis] = gen_topology_hex( ...
    num_cells, isd_m, tx_height_m, sectors)

    switch num_cells
        case 1
            num_rings = 0;
        case 7
            num_rings = 1;
        case 19
            num_rings = 2;
        otherwise
            num_rings = 0;
            while 1 + 3*num_rings*(num_rings+1) < num_cells
                num_rings = num_rings + 1;
            end
    end
    xy = hex_ring_positions(num_rings, isd_m);
    num_sites = size(xy, 1);
    if num_sites > num_cells
        xy = xy(1:num_cells, :);
        num_sites = num_cells;
    end

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
            bs_orientations(:, k) = [0; 0; deg2rad(az_deg)];
            % Mod-3 PCI plan
            bs_pcis(k) = 3 * (i_site - 1) + (i_sec - 1);
        end
    end
end


function xy = hex_ring_positions(num_rings, isd_m)
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
            for kk = 0:(r - 1)
                t = kk / r;
                idx = idx + 1;
                xy(idx, :) = ((1 - t) * start_pt + t * end_pt) * isd_m;
            end
        end
    end
end

function [bs_positions, bs_orientations] = gen_linear_positions( ...
    num_sites, isd_m, tx_height_m, track_offset_m, sectors)
%GEN_LINEAR_POSITIONS  Linear (rail-track) site layout.
%
%   Sites are placed along the X-axis at equal ISD spacing, staggered
%   on alternating sides of the track (Y direction) by track_offset_m.
%   Even-indexed sites go to +Y, odd-indexed to -Y.
%
%   Sector boresights point toward the track centreline.
%
%   Inputs:
%       num_sites       : number of physical sites
%       isd_m           : inter-site distance along the track (metres)
%       tx_height_m     : antenna height (metres)
%       track_offset_m  : perpendicular offset from track centre (metres)
%       sectors         : 1 (omni) or 3 (trisector)
%
%   Outputs:
%       bs_positions    : [3 x K]  where K = num_sites * sectors
%       bs_orientations : [3 x K]  (bank, tilt, heading in radians)

    if nargin < 5 || isempty(sectors); sectors = 1; end
    if nargin < 4 || isempty(track_offset_m); track_offset_m = 80; end

    K = num_sites * sectors;
    bs_positions = zeros(3, K);
    bs_orientations = zeros(3, K);

    total_len = (num_sites - 1) * isd_m;
    start_x = -total_len / 2;

    idx = 0;
    for i = 0:(num_sites - 1)
        cx = start_x + i * isd_m;

        % Stagger: even on +Y, odd on -Y
        if mod(i, 2) == 0
            side = 1.0;
        else
            side = -1.0;
        end
        cy = side * track_offset_m;

        % Sector azimuths: primary sector points toward track (Y=0)
        toward_track_deg = 270.0;  % +Y side points down toward Y=0
        if side < 0
            toward_track_deg = 90.0;  % -Y side points up toward Y=0
        end

        if sectors == 1
            azimuths = toward_track_deg;
        else
            azimuths = [toward_track_deg, ...
                        toward_track_deg + 120.0, ...
                        toward_track_deg + 240.0];
        end

        for s = 1:sectors
            idx = idx + 1;
            bs_positions(:, idx) = [cx; cy; tx_height_m];
            az_rad = deg2rad(mod(azimuths(s), 360));
            bs_orientations(:, idx) = [0; 0; az_rad];
        end
    end
end

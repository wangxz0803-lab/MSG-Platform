function UE_positions = place_ues(num_ues, cell_radius_m, ue_height_m, ...
    distribution, center, rng_seed)
%PLACE_UES  Generate UE positions using specified distribution mode.
%
%   Supports three placement strategies matching the Python internal_sim:
%     'uniform'   - Uniformly distributed within a disk
%     'clustered' - Multiple Gaussian clusters within the cell
%     'hotspot'   - 70% near centre, 30% at cell edge
%
%   Inputs:
%       num_ues        : number of UEs
%       cell_radius_m  : radius of the placement area (metres)
%       ue_height_m    : UE height (metres)
%       distribution   : 'uniform' | 'clustered' | 'hotspot'
%       center         : [2x1] or [1x2] XY centre (metres), default [0;0]
%       rng_seed       : optional RNG seed for reproducibility
%
%   Output:
%       UE_positions   : [3 x num_ues]

    if nargin < 6 || isempty(rng_seed); rng_seed = []; end
    if nargin < 5 || isempty(center); center = [0; 0]; end
    if nargin < 4 || isempty(distribution); distribution = 'uniform'; end

    center = center(:);  % ensure column

    if ~isempty(rng_seed)
        rng(rng_seed, 'twister');
    end

    switch lower(distribution)
        case 'uniform'
            UE_positions = place_uniform(num_ues, cell_radius_m, ue_height_m, center);
        case 'clustered'
            UE_positions = place_clustered(num_ues, cell_radius_m, ue_height_m, center);
        case 'hotspot'
            UE_positions = place_hotspot(num_ues, cell_radius_m, ue_height_m, center);
        otherwise
            warning('place_ues: unknown distribution "%s", using uniform', distribution);
            UE_positions = place_uniform(num_ues, cell_radius_m, ue_height_m, center);
    end
end


function pos = place_uniform(num_ues, radius, h_ut, center)
    rho = radius * sqrt(rand(1, num_ues));
    theta = 2 * pi * rand(1, num_ues);
    pos = zeros(3, num_ues);
    pos(1, :) = center(1) + rho .* cos(theta);
    pos(2, :) = center(2) + rho .* sin(theta);
    pos(3, :) = h_ut;
end


function pos = place_clustered(num_ues, radius, h_ut, center)
    num_clusters = 3;
    cluster_spread = radius * 0.15;

    % Cluster centres within 60% of cell radius
    cr = radius * 0.6 * sqrt(rand(1, num_clusters));
    ct = 2 * pi * rand(1, num_clusters);
    cx = center(1) + cr .* cos(ct);
    cy = center(2) + cr .* sin(ct);

    % Assign UEs to clusters
    assignments = randi(num_clusters, 1, num_ues);
    dx = cluster_spread * randn(1, num_ues);
    dy = cluster_spread * randn(1, num_ues);

    pos = zeros(3, num_ues);
    pos(1, :) = cx(assignments) + dx;
    pos(2, :) = cy(assignments) + dy;
    pos(3, :) = h_ut;
end


function pos = place_hotspot(num_ues, radius, h_ut, center)
    n_centre = round(num_ues * 0.7);
    n_edge = num_ues - n_centre;

    % Centre UEs: within 30% of radius
    if n_centre > 0
        pos_c = place_uniform(n_centre, radius * 0.3, h_ut, center);
    else
        pos_c = zeros(3, 0);
    end

    % Edge UEs: between 70%-100% of radius
    if n_edge > 0
        r_edge = radius * (0.7 + 0.3 * sqrt(rand(1, n_edge)));
        t_edge = 2 * pi * rand(1, n_edge);
        pos_e = zeros(3, n_edge);
        pos_e(1, :) = center(1) + r_edge .* cos(t_edge);
        pos_e(2, :) = center(2) + r_edge .* sin(t_edge);
        pos_e(3, :) = h_ut;
    else
        pos_e = zeros(3, 0);
    end

    pos = [pos_c, pos_e];
end

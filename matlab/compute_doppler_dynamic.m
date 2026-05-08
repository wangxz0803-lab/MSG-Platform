function [doppler_hz, nearest_cell_ids] = compute_doppler_dynamic( ...
    positions, bs_positions, carrier_freq_hz, dt_s)
%COMPUTE_DOPPLER_DYNAMIC  Per-sample Doppler shift relative to nearest cell.
%
%   Matches Python _mobility.py::compute_doppler_from_trajectory().
%
%   Inputs:
%       positions        : [num_steps x 3] UE trajectory
%       bs_positions     : [3 x K] BS positions
%       carrier_freq_hz  : carrier frequency in Hz
%       dt_s             : time step in seconds
%
%   Outputs:
%       doppler_hz       : [num_steps x 1] Doppler shift in Hz
%       nearest_cell_ids : [num_steps x 1] nearest cell index (1-based)

    c = 3e8;
    wavelength = c / carrier_freq_hz;
    num_steps = size(positions, 1);
    K = size(bs_positions, 2);

    % Find nearest cell for each time step
    nearest_cell_ids = ones(num_steps, 1);
    dist_to_nearest = zeros(num_steps, 1);
    for i = 1:num_steps
        ue_2d = positions(i, 1:2);
        min_d = inf;
        for k = 1:K
            d = norm(ue_2d - bs_positions(1:2, k).');
            if d < min_d
                min_d = d;
                nearest_cell_ids(i) = k;
            end
        end
        dist_to_nearest(i) = max(min_d, 1e-3);
    end

    % Radial velocity: d(distance)/dt
    radial_vel = diff(dist_to_nearest) / dt_s;  % [N-1, 1]
    radial_vel = [radial_vel(1); radial_vel];    % duplicate first

    % Doppler: negative radial velocity = approaching = positive Doppler
    doppler_hz = -radial_vel / wavelength;
end

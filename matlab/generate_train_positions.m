function all_positions = generate_train_positions( ...
    base_trajectory, num_ues, train_length_m, train_width_m, seed)
%GENERATE_TRAIN_POSITIONS  Per-UE positions for passengers riding a train.
%
%   Each UE gets a fixed random offset within the train body.
%   The train moves along *base_trajectory* (the train center).
%
%   Matches Python _mobility.py::generate_train_positions() exactly.
%
%   Inputs:
%       base_trajectory : [num_steps x 3] train centre positions
%       num_ues         : number of passengers
%       train_length_m  : train body length (default 400)
%       train_width_m   : train body width (default 3.4)
%       seed            : RNG seed for reproducibility
%
%   Output:
%       all_positions   : [num_ues x num_steps x 3]

    if nargin < 5 || isempty(seed); seed = 42; end
    if nargin < 4 || isempty(train_width_m); train_width_m = 3.4; end
    if nargin < 3 || isempty(train_length_m); train_length_m = 400.0; end

    rng(seed, 'twister');

    num_steps = size(base_trajectory, 1);
    half_len = train_length_m / 2;
    half_wid = train_width_m / 2;

    % Fixed random offsets for each UE within train body
    offsets_along = -half_len + 2 * half_len * rand(1, num_ues);  % along track
    offsets_perp  = -half_wid + 2 * half_wid * rand(1, num_ues);  % across track

    % Compute track direction at each time step
    diffs = diff(base_trajectory(:, 1:2), 1, 1);  % [N-1, 2]
    norms = sqrt(sum(diffs.^2, 2));
    norms = max(norms, 1e-9);
    dirs = zeros(num_steps, 2);
    dirs(2:end, :) = diffs ./ norms;
    dirs(1, :) = dirs(2, :);

    % Perpendicular direction (rotate 90 degrees)
    perp = [-dirs(:, 2), dirs(:, 1)];  % [num_steps, 2]

    all_positions = zeros(num_ues, num_steps, 3);
    for u = 1:num_ues
        all_positions(u, :, 1) = base_trajectory(:, 1) + ...
            offsets_along(u) * dirs(:, 1) + offsets_perp(u) * perp(:, 1);
        all_positions(u, :, 2) = base_trajectory(:, 2) + ...
            offsets_along(u) * dirs(:, 2) + offsets_perp(u) * perp(:, 2);
        all_positions(u, :, 3) = base_trajectory(:, 3);
    end
end

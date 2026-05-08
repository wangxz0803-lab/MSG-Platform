function positions = generate_track_mobility(waypoints, speed_ms, num_steps, dt_s)
%GENERATE_TRACK_MOBILITY  Fixed-path trajectory along polyline waypoints.
%
%   The UE moves at constant *speed_ms* along the polyline defined by
%   *waypoints*. If it reaches the last waypoint before *num_steps* is
%   exhausted, it reverses direction (ping-pong) and continues.
%
%   Matches Python _mobility.py::_track() exactly.
%
%   Inputs:
%       waypoints  : [N x 2] or [N x 3] XY(Z) coordinates
%       speed_ms   : constant speed in m/s
%       num_steps  : number of time samples
%       dt_s       : time step in seconds
%
%   Output:
%       positions  : [num_steps x 3]

    if size(waypoints, 2) < 2
        error('generate_track_mobility: waypoints must be [N x 2] or [N x 3]');
    end
    wp_2d = waypoints(:, 1:2);
    z = 1.5;  % default height

    segments = diff(wp_2d, 1, 1);           % [N-1, 2]
    seg_lens = sqrt(sum(segments.^2, 2));   % [N-1, 1]
    seg_lens = max(seg_lens, 1e-6);
    seg_dirs = segments ./ seg_lens;        % [N-1, 2]

    n_segs = size(seg_lens, 1);
    positions = zeros(num_steps, 3);
    positions(1, :) = [wp_2d(1, 1), wp_2d(1, 2), z];

    seg_idx = 1;         % 1-based (MATLAB)
    dist_in_seg = 0.0;
    direction = 1;       % +1 forward, -1 backward

    for i = 2:num_steps
        step_remaining = speed_ms * dt_s;
        cur_x = positions(i-1, 1);
        cur_y = positions(i-1, 2);

        while step_remaining > 1e-9
            if seg_idx < 1 || seg_idx > n_segs
                break;
            end

            if direction == 1
                remaining_in_seg = seg_lens(seg_idx) - dist_in_seg;
            else
                remaining_in_seg = dist_in_seg;
            end

            if step_remaining <= remaining_in_seg
                dist_in_seg = dist_in_seg + step_remaining * direction;
                d = seg_dirs(seg_idx, :);
                cur_x = wp_2d(seg_idx, 1) + d(1) * dist_in_seg;
                cur_y = wp_2d(seg_idx, 2) + d(2) * dist_in_seg;
                step_remaining = 0;
            else
                step_remaining = step_remaining - remaining_in_seg;
                if direction == 1
                    seg_idx = seg_idx + 1;
                    dist_in_seg = 0;
                    if seg_idx > n_segs
                        direction = -1;
                        seg_idx = n_segs;
                        dist_in_seg = seg_lens(seg_idx);
                    end
                else
                    seg_idx = seg_idx - 1;
                    if seg_idx < 1
                        direction = 1;
                        seg_idx = 1;
                        dist_in_seg = 0;
                    else
                        dist_in_seg = seg_lens(seg_idx);
                    end
                end
            end
        end

        positions(i, :) = [cur_x, cur_y, z];
    end
end

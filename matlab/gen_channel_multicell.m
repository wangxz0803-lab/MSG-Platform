function [cn, c_builder, Hf_per_cell] = gen_channel_multicell(s1, a_BS, a_mt, ...
    bs_positions, bs_orientations, ue_track, ue_position, scenario, ...
    bandwidth, N_RB, no_ss, BS_ant, ue_ant, Nt_v, Nt_h, device)
%GEN_CHANNEL_MULTICELL  Build K-TX QuaDRiGa layout for a single UE.
%
%   [CN, C_BUILDER, HF_PER_CELL] = GEN_CHANNEL_MULTICELL(...) constructs a
%   qd_layout with no_tx = num_cells transmitters and one receiver, runs
%   get_channels(), and returns per-cell frequency-domain channels.
%
%   Inputs:
%       s1              : qd_simulation_parameters
%       a_BS, a_mt      : qd_arrayant (BS / UE)
%       bs_positions    : 3 x K transmitter positions
%       bs_orientations : 3 x K orientations ([bank; tilt; heading], rad)
%       ue_track        : 1 x 1 qd_track
%       ue_position     : 3 x 1 UE position (track start)
%       scenario        : QuaDRiGa scenario tag
%       bandwidth       : total BW in Hz (= N_RB * sc_inter)
%       N_RB            : number of frequency samples
%       no_ss           : number of snapshots
%       BS_ant, ue_ant  : full antenna counts (after pol de-interleaving)
%       Nt_v, Nt_h      : BS grid dimensions (single-pol panel)
%       device          : qd_mesh GPU flag (0 or 1)
%
%   Output:
%       cn              : QuaDRiGa channel-object array (length K)
%       c_builder       : builder object (delay spread etc.)
%       Hf_per_cell     : complex64 array of size
%                         [K, BS_ant, ue_ant, N_RB, no_ss]
%
%   Single-cell degenerate case: K = 1, output is [1, BS_ant, ue_ant, N_RB, no_ss].
%
%   Manual verification (no MATLAB):
%       * Check size(Hf_per_cell) equals [size(bs_positions,2), 64, 4,
%         N_RB, no_ss] for the default antenna config.
%       * For K=1, Hf_per_cell(1,...) should equal the single-cell legacy
%         output up to permutation / pol reshape.

    num_cells = size(bs_positions, 2);
    l = qd_layout(s1);
    l.no_tx = num_cells;
    % Replicate a_BS K times (QuaDRiGa accepts one array per tx).
    if num_cells == 1
        l.tx_array = a_BS;
    else
        l.tx_array = repmat(a_BS, 1, num_cells);
    end
    l.tx_position = bs_positions;
    % Apply per-TX orientation (azimuth) to the array copies.
    % QuaDRiGa convention: rotate the array object, not the layout.
    try
        for k = 1:num_cells
            % rotate by heading (radians). Only apply if array supports it.
            if isa(l.tx_array(1, k), 'qd_arrayant')
                l.tx_array(1, k).rotate_pattern(rad2deg(bs_orientations(3, k)), 'z');
            end
        end
    catch ME  %#ok<NASGU>
        % Some QuaDRiGa builds store tx_array as a single shared ref — fall
        % back to setting orientation via layout if available.
        try
            l.tx_orientation = bs_orientations; %#ok<NASGU>
        catch
            % Leave orientation as zero — legacy single-cell path.
        end
    end

    l.no_rx = 1;
    l.rx_array = a_mt;
    l.rx_track = ue_track;
    l.rx_position = ue_position(:);
    l.set_scenario(scenario);

    [cn, c_builder] = l.get_channels();

    % cn is a num_rx x num_tx channel object array. For num_rx=1 this is
    % 1 x num_cells.
    Hf_per_cell = zeros(num_cells, BS_ant, ue_ant, N_RB, no_ss, 'single');
    for k = 1:num_cells
        Hf_temp = cn(1, k).fr(bandwidth, N_RB, [], device); % [rx, tx, sc, slot]
        % Follow legacy reshape path (legacy used only k=1):
        % The BS array is dual-pol cross-pol: tx dim = 2 * Nt_v * Nt_h.
        Hf_temp = reshape(Hf_temp, ue_ant, 2, Nt_v, Nt_h, N_RB, no_ss);
        Hf_temp = permute(Hf_temp, [1, 4, 3, 2, 5, 6]);          % [UE, Nt_h, Nt_v, pol, RB, slot]
        Hf_temp = reshape(Hf_temp, ue_ant, BS_ant, N_RB, no_ss); % [UE, BS, RB, slot]
        Hf_temp = permute(Hf_temp, [2, 1, 3, 4]);                % [BS, UE, RB, slot]
        Hf_per_cell(k, :, :, :, :) = single(Hf_temp);
    end
end

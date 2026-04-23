function main_multi(config_path)
%MAIN_MULTI  Multi-cell QuaDRiGa channel generation with JSON config.
%
%   MAIN_MULTI(CONFIG_PATH) reads a JSON configuration file, builds a
%   K-cell hex-grid layout, generates per-UE multi-cell channel responses,
%   and saves batch .mat files with Hf_multi and metadata.
%
%   Output format:
%       Hf_multi : complex single [no_ue, K, BsAnt, ue_ant, N_RB, no_ss]
%       meta     : struct with num_cells, bs_positions, ue_positions,
%                  serving_cell_ids, and full config echo
%
%   When num_cells=1, the output Hf_multi has K=1 in the second dimension,
%   maintaining backward compatibility with the original single-cell script.
%
%   JSON config fields:
%       num_cells              : number of BS sites (1, 7, 19, ...)
%       num_ues                : UEs per .mat file (batch size)
%       num_snapshots          : time snapshots
%       carrier_freq_hz        : carrier frequency in Hz
%       scenario               : QuaDRiGa scenario tag
%       isd_m                  : inter-site distance in metres
%       tx_height_m            : BS antenna height
%       rx_height_m            : UE height
%       cell_radius_m          : UE drop radius around centre
%       ue_speed_kmh           : UE speed in km/h
%       bs_ant_v               : BS vertical antenna elements
%       bs_ant_h               : BS horizontal antenna elements
%       ue_ant_v               : UE vertical elements
%       ue_ant_h               : UE horizontal elements
%       output_dir             : output directory for .mat files
%       seed                   : RNG seed
%       custom_site_positions  : null or [K x 3] array of BS positions
%       custom_ue_positions    : null or [N x 3] array of UE positions
%
%   Usage:
%       main_multi('configs/multicell_7.json')
%
%   See also: gen_hex_positions, gen_channel_multicell

    if nargin < 1 || isempty(config_path)
        error('main_multi:MissingConfig', ...
            'Usage: main_multi(config_path)');
    end

    % ------------------------------------------------------------------
    % Setup paths
    % ------------------------------------------------------------------
    this_file = mfilename('fullpath');
    matlab_dir = fileparts(this_file);
    repo_root = fileparts(matlab_dir);
    addpath(matlab_dir);

    % QuaDRiGa source: try local repo first, fall back to D:\MSG
    qd_src = fullfile(repo_root, 'quadriga_src');
    if ~exist(qd_src, 'dir')
        qd_src = fullfile('D:', 'MSG', 'quadriga_src');
    end
    if exist(qd_src, 'dir')
        addpath(qd_src);
    else
        error('main_multi:QuaDRiGaNotFound', ...
            'quadriga_src not found in %s or D:\\MSG', repo_root);
    end

    % ------------------------------------------------------------------
    % Load config and fill defaults
    % ------------------------------------------------------------------
    cfg = jsondecode(fileread(config_path));
    cfg = apply_defaults(cfg);

    rng(cfg.seed, 'twister');

    % Resolve output directory (relative paths are relative to repo root).
    if ~is_absolute_path(cfg.output_dir)
        cfg.output_dir = fullfile(repo_root, cfg.output_dir);
    end
    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end

    % GPU check
    try
        device = qd_mesh.has_gpu;
    catch
        device = 0;
    end
    if device == 1
        disp('[main_multi] GPU acceleration available');
    else
        disp('[main_multi] CPU only');
    end

    % ------------------------------------------------------------------
    % Derived parameters
    % ------------------------------------------------------------------
    K = cfg.num_cells;
    no_ue = cfg.num_ues;
    no_ss = cfg.num_snapshots;
    fc = cfg.carrier_freq_hz;

    Nt_v = cfg.bs_ant_v;
    Nt_h = cfg.bs_ant_h;
    BsAnt = Nt_v * Nt_h * 2;   % dual-pol cross-pol

    Nr_v = cfg.ue_ant_v;
    Nr_h = cfg.ue_ant_h;
    ue_ant = Nr_v * Nr_h;       % single-pol

    if isfield(cfg, 'n_rb');    N_RB = cfg.n_rb;  else; N_RB = 12 * 136; end
    if isfield(cfg, 'sc_inter'); sc_inter = cfg.sc_inter; else; sc_inter = 30e3; end
    bandwidth = N_RB * 12 * sc_inter;  % 每 RB = 12 子载波
    T_snapshots = 1e-3;
    downtilt_ang = 7;

    ue_speed_kmh = cfg.ue_speed_kmh;
    ue_speed_mps = ue_speed_kmh / 3.6;
    track_length = max((no_ss - 1) * T_snapshots * ue_speed_mps, 1e-3);

    fprintf('[main_multi] K=%d  scenario=%s  fc=%.2f GHz\n', ...
        K, cfg.scenario, fc / 1e9);
    fprintf('[main_multi] BsAnt=%d  ue_ant=%d  N_RB=%d  no_ss=%d  no_ue=%d\n', ...
        BsAnt, ue_ant, N_RB, no_ss, no_ue);

    % ------------------------------------------------------------------
    % Antenna arrays (shared across all cells)
    % ------------------------------------------------------------------
    s_base = qd_simulation_parameters;
    s_base.center_frequency = fc;

    a_BS = qd_arrayant.generate('3gpp-mmw', Nt_v, Nt_h, fc, 3, ...
        downtilt_ang, 0.5, 1, 1, 0.5 * Nt_v, 0.5 * Nt_h);

    a_mt = qd_arrayant.generate('3gpp-mmw', Nr_v, Nr_h, fc, 1, ...
        downtilt_ang, 0.5, 1, 1, 0.5 * Nr_v, 0.5 * Nr_h);

    % ------------------------------------------------------------------
    % BS positions: custom or hex-grid
    % ------------------------------------------------------------------
    if isfield(cfg, 'custom_site_positions') && ~isempty(cfg.custom_site_positions)
        % custom_site_positions is [K x 3] from JSON
        csp = cfg.custom_site_positions;
        if isstruct(csp)
            % JSON array of {x, y, z} objects
            bs_positions = zeros(3, numel(csp));
            for i = 1:numel(csp)
                bs_positions(:, i) = [csp(i).x; csp(i).y; csp(i).z];
            end
        else
            % [K x 3] numeric array
            bs_positions = csp.';  % transpose to [3 x K]
            if size(bs_positions, 1) ~= 3
                bs_positions = bs_positions.';
            end
        end
        K = size(bs_positions, 2);
        fprintf('[main_multi] Using %d custom BS positions\n', K);
    else
        bs_positions = gen_hex_positions(K, cfg.isd_m, cfg.tx_height_m);
        fprintf('[main_multi] Generated hex-grid with %d sites, ISD=%.0fm\n', ...
            K, cfg.isd_m);
    end

    % ------------------------------------------------------------------
    % Simulation parameters
    % ------------------------------------------------------------------
    s1 = qd_simulation_parameters;
    s1.center_frequency = fc;
    s1.set_speed(ue_speed_kmh, T_snapshots);
    s1.use_random_initial_phase = true;
    s1.use_3GPP_baseline = 1;
    s1.show_progress_bars = 0;

    % ------------------------------------------------------------------
    % UE positions: custom or random
    % ------------------------------------------------------------------
    if isfield(cfg, 'custom_ue_positions') && ~isempty(cfg.custom_ue_positions)
        cup = cfg.custom_ue_positions;
        if isstruct(cup)
            UE_positions = zeros(3, numel(cup));
            for i = 1:numel(cup)
                UE_positions(:, i) = [cup(i).x; cup(i).y; cup(i).z];
            end
        else
            UE_positions = cup.';
            if size(UE_positions, 1) ~= 3
                UE_positions = UE_positions.';
            end
        end
        no_ue = size(UE_positions, 2);
    else
        % Drop UEs uniformly within a disk of radius cell_radius_m
        rho = cfg.cell_radius_m * sqrt(rand(1, no_ue));
        phi_deg = 360 * rand(1, no_ue) - 180;
        UE_positions = zeros(3, no_ue);
        UE_positions(1, :) = rho .* cosd(phi_deg);
        UE_positions(2, :) = rho .* sind(phi_deg);
        UE_positions(3, :) = cfg.rx_height_m;
    end

    % ------------------------------------------------------------------
    % Generate UE tracks (mobility_mode aware)
    % ------------------------------------------------------------------
    clear UE_tracks;
    if isfield(cfg, 'mobility_mode')
        mobility_mode = cfg.mobility_mode;
    else
        mobility_mode = 'linear';
    end
    fprintf('[main_multi] Mobility mode: %s\n', mobility_mode);

    use_trajectory = isfield(cfg, 'custom_ue_positions') && ~isempty(cfg.custom_ue_positions) && no_ue > 1;
    if use_trajectory
        % Custom positions: create a SINGLE trajectory track through all
        % points so QuaDRiGa's spatial consistency model is preserved.
        origin = UE_positions(:, 1);
        relative_positions = UE_positions - origin;

        trk = qd_track();
        trk.name = 'trajectory';
        trk.positions = relative_positions;   % [3 x N] relative to start
        trk.no_snapshots = no_ue;
        for seg = 1:no_ue
            trk.scenario{seg} = cfg.scenario;
        end

        for i_ue = 1:no_ue
            UE_tracks(1, i_ue) = trk; %#ok<AGROW>
        end
    else
        for i_ue = 1:no_ue
            trk = generate_ue_track(mobility_mode, track_length, ...
                cfg.cell_radius_m, s1, cfg.scenario, i_ue);
            UE_tracks(1, i_ue) = trk; %#ok<AGROW>
        end
    end

    % ------------------------------------------------------------------
    % Allocate output: both ideal and LS-estimated serving channel
    % ------------------------------------------------------------------
    Hf_serving_ideal = zeros(no_ue, BsAnt, ue_ant, N_RB, no_ss, 'single');
    Hf_serving_est   = zeros(no_ue, BsAnt, ue_ant, N_RB, no_ss, 'single');
    serving_cell_ids = zeros(1, no_ue);
    rsrp_per_cell = zeros(no_ue, K, 'single');   % per-cell |H|^2 mean
    snr_dB  = zeros(1, no_ue, 'single');
    sir_dB  = zeros(1, no_ue, 'single');
    sinr_dB = zeros(1, no_ue, 'single');

    % BS orientations: for multi-cell sectorized deployment, each site has
    % 3 sectors pointing at 0deg / 120deg / 240deg azimuth.
    bs_orientations = zeros(3, K);
    if K > 1
        sector_azimuths = [0, 120, 240];  % degrees
        for k = 1:K
            sector_idx = mod(k-1, 3) + 1;
            bs_orientations(3, k) = sector_azimuths(sector_idx) * pi / 180;  % radians
        end
    end

    % Physical parameters — scenario-aware TX power defaults
    if isfield(cfg, 'tx_power_dbm')
        Ptx_BS_dBm = cfg.tx_power_dbm;
    elseif contains(cfg.scenario, 'InF')
        Ptx_BS_dBm = 24;
    elseif contains(cfg.scenario, 'UMi')
        Ptx_BS_dBm = 33;
    else
        Ptx_BS_dBm = 46;
    end
    Ptx_UE_dBm = 23;
    Ptx_BS_W = 10^(Ptx_BS_dBm / 10) * 1e-3;      % 39.81 W
    noise_dBm = -174 + 5 + 10*log10(bandwidth);   % thermal noise over BW
    noise_W   = 10^(noise_dBm / 10) * 1e-3;

    % SRS configuration for UL channel estimation
    srs_cfg = struct();
    srs_cfg.n_srs_id = 0;
    srs_cfg.n_cs = 0;
    srs_cfg.tx_power_dBm = Ptx_UE_dBm;

    % Channel estimation mode
    if isfield(cfg, 'est_mode')
        est_mode = cfg.est_mode;
    else
        est_mode = 'ls_linear';
    end
    fprintf('[main_multi] Channel estimation mode: %s\n', est_mode);

    t0 = tic;

    % ------------------------------------------------------------------
    % Main loop: generate per-UE channels to all K cells
    % ------------------------------------------------------------------
    for i_ue = 1:no_ue
        ue_pos = UE_positions(:, i_ue);
        ue_track = UE_tracks(1, i_ue);

        % Initial serving cell estimate by nearest distance (needed for
        % gen_channel_multicell; will be refined after channel generation).
        sid = pick_serving_cell(ue_pos, bs_positions);

        % Generate per-cell frequency-domain channels using existing helper.
        [~, ~, Hf_per_cell] = gen_channel_multicell( ...
            s1, a_BS, a_mt, bs_positions, bs_orientations, ...
            ue_track, ue_pos, cfg.scenario, bandwidth, N_RB, ...
            no_ss, BsAnt, ue_ant, Nt_v, Nt_h, device);
        % Hf_per_cell: [K, BsAnt, ue_ant, N_RB, no_ss]

        % Re-select serving cell based on strongest mean channel power (RSRP)
        rsrp_vals = zeros(1, K);
        for k = 1:K
            rsrp_vals(k) = mean(abs(double(Hf_per_cell(k, :))).^2, 'all');
        end
        [~, sid] = max(rsrp_vals);
        serving_cell_ids(i_ue) = sid;

        % Store ideal serving channel (ground truth for evaluation)
        Hf_serving_ideal(i_ue, :, :, :, :) = Hf_per_cell(sid, :, :, :, :);

        % Run SRS pipeline: real inter-cell interference + per-port LS
        [H_est, sinr_srs, sir_srs] = ul_srs_pipeline( ...
            Hf_per_cell, sid, srs_cfg, noise_dBm, est_mode);
        % H_est: [BsAnt, ue_ant, N_RB, no_ss]
        Hf_serving_est(i_ue, :, :, :, :) = H_est;

        % SINR/SIR from actual SRS reception (real interference structure)
        sinr_dB(i_ue) = single(sinr_srs);
        sir_dB(i_ue)  = single(sir_srs);

        % Compute per-cell RSRP and DL SNR from channel gains
        serving_gain = 0;
        inter_gain   = 0;
        for k = 1:K
            gain_k = mean(abs(double(Hf_per_cell(k, :))).^2, 'all');
            rsrp_per_cell(i_ue, k) = single(gain_k);
            if k == sid
                serving_gain = gain_k;
            else
                inter_gain = inter_gain + gain_k;
            end
        end
        % DL SNR (reference, uses BS Tx power)
        rx_serv = Ptx_BS_W * serving_gain;
        snr_dB(i_ue) = single(10*log10(max(rx_serv / noise_W, 1e-10)));

        if mod(i_ue, 10) == 0 || i_ue == no_ue
            fprintf('[main_multi] UE %d/%d  SINR=%.1f dB  SIR=%.1f dB  (%.1fs)\n', ...
                i_ue, no_ue, sinr_dB(i_ue), sir_dB(i_ue), toc(t0));
        end
    end

    % ------------------------------------------------------------------
    % Build metadata struct
    % ------------------------------------------------------------------
    meta = struct();
    meta.num_cells = K;
    meta.bs_positions = bs_positions;           % [3 x K]
    meta.ue_positions = UE_positions;           % [3 x no_ue]
    meta.serving_cell_ids = serving_cell_ids;   % [1 x no_ue], 1-based
    meta.num_ues = no_ue;
    meta.num_snapshots = no_ss;
    meta.carrier_freq_hz = fc;
    meta.scenario = cfg.scenario;
    meta.isd_m = cfg.isd_m;
    meta.bandwidth_hz = bandwidth;
    meta.bs_ant_v = Nt_v;
    meta.bs_ant_h = Nt_h;
    meta.BsAnt = BsAnt;
    meta.ue_ant_v = Nr_v;
    meta.ue_ant_h = Nr_h;
    meta.ue_ant = ue_ant;
    meta.N_RB = N_RB;
    meta.ue_speed_kmh = ue_speed_kmh;
    meta.cell_radius_m = cfg.cell_radius_m;
    meta.tx_height_m = cfg.tx_height_m;
    meta.rx_height_m = cfg.rx_height_m;
    meta.seed = cfg.seed;
    meta.Ptx_BS_dBm = Ptx_BS_dBm;
    meta.Ptx_UE_dBm = Ptx_UE_dBm;
    meta.noise_dBm = noise_dBm;
    meta.est_mode = est_mode;

    % ------------------------------------------------------------------
    % Save output
    % ------------------------------------------------------------------
    if isfield(cfg, 'shard_id')
        fileName = sprintf('QDRG_real_shard%04d.mat', cfg.shard_id);
    else
        fileName = sprintf('multicell_K%d_ue%d_bs%d_nrb%d_nss%d_seed%d.mat', ...
            K, no_ue, BsAnt, N_RB, no_ss, cfg.seed);
    end
    fullFilePath = fullfile(cfg.output_dir, fileName);

    save(fullFilePath, 'Hf_serving_est', 'Hf_serving_ideal', 'rsrp_per_cell', ...
         'snr_dB', 'sir_dB', 'sinr_dB', 'meta', '-v7');

    elapsed = toc(t0);
    fprintf('[main_multi] Saved: %s\n', fullFilePath);
    fprintf('[main_multi] Hf_serving_est shape: [%s]\n', ...
        strjoin(arrayfun(@num2str, size(Hf_serving_est), 'UniformOutput', false), ', '));
    fprintf('[main_multi] Hf_serving_ideal shape: [%s]\n', ...
        strjoin(arrayfun(@num2str, size(Hf_serving_ideal), 'UniformOutput', false), ', '));
    fprintf('[main_multi] SINR range: [%.1f, %.1f] dB\n', min(sinr_dB), max(sinr_dB));
    fprintf('[main_multi] SIR  range: [%.1f, %.1f] dB\n', min(sir_dB), max(sir_dB));
    fprintf('[main_multi] Completed in %.1f s (%.2f h)\n', elapsed, elapsed / 3600);
end


% ======================================================================
% Helper functions
% ======================================================================

function cfg = apply_defaults(cfg)
%APPLY_DEFAULTS  Fill missing JSON fields with sensible defaults.
    if ~isfield(cfg, 'num_cells');          cfg.num_cells = 1; end
    if ~isfield(cfg, 'num_ues');            cfg.num_ues = 200; end
    if ~isfield(cfg, 'num_snapshots');      cfg.num_snapshots = 20; end
    if ~isfield(cfg, 'carrier_freq_hz');    cfg.carrier_freq_hz = 3e9; end
    if ~isfield(cfg, 'scenario');           cfg.scenario = '3GPP_38.901_UMa_NLOS'; end
    if ~isfield(cfg, 'isd_m');              cfg.isd_m = 500; end
    if ~isfield(cfg, 'tx_height_m');        cfg.tx_height_m = 25; end
    if ~isfield(cfg, 'rx_height_m');        cfg.rx_height_m = 1.5; end
    if ~isfield(cfg, 'cell_radius_m');      cfg.cell_radius_m = 200; end
    if ~isfield(cfg, 'ue_speed_kmh');       cfg.ue_speed_kmh = 100; end
    if ~isfield(cfg, 'bs_ant_v');           cfg.bs_ant_v = 4; end
    if ~isfield(cfg, 'bs_ant_h');           cfg.bs_ant_h = 8; end
    if ~isfield(cfg, 'ue_ant_v');           cfg.ue_ant_v = 2; end
    if ~isfield(cfg, 'ue_ant_h');           cfg.ue_ant_h = 2; end
    if ~isfield(cfg, 'output_dir');         cfg.output_dir = 'htt_v2_3gppmmw'; end
    if ~isfield(cfg, 'seed');               cfg.seed = 12345; end
end


function tf = is_absolute_path(p)
    if isempty(p); tf = false; return; end
    if ispc
        tf = numel(p) >= 2 && p(2) == ':';
    else
        tf = p(1) == '/';
    end
end


function cell_id = pick_serving_cell(ue_pos, bs_positions)
%PICK_SERVING_CELL  Select nearest BS by 3D Euclidean distance.
    diffs = bs_positions - ue_pos;
    d2 = sum(diffs .^ 2, 1);
    [~, cell_id] = min(d2);
end


function trk = generate_ue_track(mobility_mode, track_length, cell_radius_m, sim_params, scenario, ue_id)
%GENERATE_UE_TRACK  Create a qd_track based on the mobility mode.
%
%   Supported modes:
%     'static'           - stationary UE (zero-length track)
%     'linear'           - straight line at constant speed
%     'random_walk'      - correlated random walk (piecewise linear segments with turning)
%     'random_waypoint'  - random waypoint within cell radius

    switch lower(mobility_mode)
        case 'static'
            trk = qd_track.generate('linear', 1e-3);
            trk.name = sprintf('user%d', ue_id);

        case 'linear'
            trk = qd_track.generate('linear', track_length);
            trk.name = sprintf('user%d', ue_id);
            trk.interpolate('distance', 1 / sim_params.samples_per_meter, [], [], 1);

        case 'random_walk'
            n_segments = max(10, round(track_length / 5));
            seg_len = track_length / n_segments;
            positions = zeros(3, n_segments + 1);
            heading = 2 * pi * rand();
            turn_std = 25 * pi / 180;

            for s = 2:(n_segments + 1)
                heading = heading + turn_std * randn();
                positions(1, s) = positions(1, s-1) + seg_len * cos(heading);
                positions(2, s) = positions(2, s-1) + seg_len * sin(heading);
            end

            trk = qd_track();
            trk.name = sprintf('user%d', ue_id);
            trk.positions = positions;
            trk.scenario{1} = scenario;
            trk.interpolate('distance', 1 / sim_params.samples_per_meter, [], [], 1);

        case 'random_waypoint'
            n_waypoints = max(5, round(track_length / 20));
            positions = zeros(3, n_waypoints + 1);
            for w = 2:(n_waypoints + 1)
                r_wp = cell_radius_m * sqrt(rand());
                theta_wp = 2 * pi * rand();
                positions(1, w) = r_wp * cos(theta_wp);
                positions(2, w) = r_wp * sin(theta_wp);
            end

            trk = qd_track();
            trk.name = sprintf('user%d', ue_id);
            trk.positions = positions;
            trk.scenario{1} = scenario;
            trk.interpolate('distance', 1 / sim_params.samples_per_meter, [], [], 1);

        otherwise
            warning('Unknown mobility_mode "%s", defaulting to linear', mobility_mode);
            trk = qd_track.generate('linear', track_length);
            trk.name = sprintf('user%d', ue_id);
            trk.interpolate('distance', 1 / sim_params.samples_per_meter, [], [], 1);
    end

    % Apply scenario to all segments if not already set
    if isempty(trk.scenario)
        trk.scenario = {scenario};
    end
end

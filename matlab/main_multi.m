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

    sc_inter = cfg.sc_inter;
    if ~isempty(cfg.n_rb) && cfg.n_rb > 0
        N_RB = cfg.n_rb;
    else
        N_RB = floor(cfg.bandwidth_hz / (12 * sc_inter));
    end
    bandwidth = N_RB * 12 * sc_inter;
    T_snapshots = cfg.sample_interval_s;
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
    % BS positions: custom → linear → hexagonal
    % ------------------------------------------------------------------
    sectors = cfg.sectors_per_site;
    if isfield(cfg, 'custom_site_positions') && ~isempty(cfg.custom_site_positions)
        csp = cfg.custom_site_positions;
        if isstruct(csp)
            bs_positions = zeros(3, numel(csp));
            for i = 1:numel(csp)
                bs_positions(:, i) = [csp(i).x; csp(i).y; csp(i).z];
            end
        else
            bs_positions = csp.';
            if size(bs_positions, 1) ~= 3
                bs_positions = bs_positions.';
            end
        end
        K = size(bs_positions, 2);
        bs_orientations = zeros(3, K);
        bs_pcis = 0:(K-1);
        fprintf('[main_multi] Using %d custom BS positions\n', K);
    else
        [bs_positions, bs_orientations, bs_pcis] = gen_topology( ...
            K, cfg.isd_m, cfg.tx_height_m, sectors, ...
            cfg.topology_layout, cfg.track_offset_m, cfg.hypercell_size);
        K = size(bs_positions, 2);
        fprintf('[main_multi] Generated %s topology: %d cells, ISD=%.0fm\n', ...
            cfg.topology_layout, K, cfg.isd_m);
    end

    % ------------------------------------------------------------------
    % Simulation parameters
    % ------------------------------------------------------------------
    s1 = qd_simulation_parameters;
    s1.center_frequency = fc;
    % Guard against zero speed: use 1e-3 km/h floor to avoid Inf samples_per_meter
    s1.set_speed(max(ue_speed_kmh, 1e-3), T_snapshots);
    s1.use_random_initial_phase = true;
    s1.use_3GPP_baseline = 1;
    s1.show_progress_bars = 0;

    % ------------------------------------------------------------------
    % UE positions: custom or distribution-based
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
        UE_positions = place_ues(no_ue, cfg.cell_radius_m, cfg.ue_height_m, ...
            cfg.ue_distribution, [0; 0], cfg.seed);
    end
    fprintf('[main_multi] UE distribution: %s  (%d UEs)\n', ...
        cfg.ue_distribution, no_ue);

    % ------------------------------------------------------------------
    % HSR detection & mobility
    % ------------------------------------------------------------------
    mobility_mode = cfg.mobility_mode;
    is_hsr = strcmpi(cfg.topology_layout, 'linear') && ...
        any(strcmpi(mobility_mode, {'linear', 'track'}));

    dt_s = cfg.sample_interval_s;
    ue_trajectories = [];   % [no_ue, no_ss, 3] — filled below if mobile
    doppler_per_ue = [];    % [no_ue, no_ss]
    nearest_cells = [];     % [no_ue, no_ss]
    train_positions = [];   % [no_ue, no_ss, 3] — HSR only

    fprintf('[main_multi] Mobility mode: %s  is_hsr=%d\n', mobility_mode, is_hsr);

    clear UE_tracks;
    if is_hsr
        % Build track waypoints from site X positions, Y=0 (track centreline)
        site_xs = unique(bs_positions(1, :));
        site_xs = sort(site_xs);
        track_waypoints = [site_xs(:), zeros(numel(site_xs), 1)];

        % Generate base trajectory (train centre)
        base_traj = generate_track_mobility(track_waypoints, ue_speed_mps, no_ss, dt_s);
        base_traj(:, 3) = cfg.ue_height_m;

        % Generate all UE positions within train body
        train_positions = generate_train_positions(base_traj, no_ue, ...
            cfg.train_length_m, cfg.train_width_m, cfg.seed);

        % Override UE_positions with train initial positions
        for i_ue = 1:no_ue
            UE_positions(:, i_ue) = squeeze(train_positions(i_ue, 1, :));
        end

        % Store trajectories
        ue_trajectories = train_positions;  % [no_ue, no_ss, 3]

        % Generate QuaDRiGa tracks from train centre trajectory
        for i_ue = 1:no_ue
            ue_traj = squeeze(train_positions(i_ue, :, :));  % [no_ss, 3]
            origin = ue_traj(1, :).';
            rel_pos = (ue_traj - ue_traj(1, :)).';  % [3 x no_ss]

            trk = qd_track();
            trk.name = sprintf('user%d', i_ue);
            trk.positions = rel_pos;
            trk.no_snapshots = no_ss;
            trk.scenario{1} = cfg.scenario;
            UE_tracks(1, i_ue) = trk; %#ok<AGROW>
        end

        % Compute dynamic Doppler for each UE
        doppler_per_ue = zeros(no_ue, no_ss);
        nearest_cells = zeros(no_ue, no_ss);
        for i_ue = 1:no_ue
            ue_traj = squeeze(train_positions(i_ue, :, :));
            [dop, nc] = compute_doppler_dynamic(ue_traj, bs_positions, fc, dt_s);
            doppler_per_ue(i_ue, :) = dop.';
            nearest_cells(i_ue, :) = nc.';
        end

    else
        % Non-HSR: standard mobility modes
        use_custom_traj = isfield(cfg, 'custom_ue_positions') && ...
            ~isempty(cfg.custom_ue_positions) && no_ue > 1;
        if use_custom_traj
            origin = UE_positions(:, 1);
            relative_positions = UE_positions - origin;
            trk = qd_track();
            trk.name = 'trajectory';
            trk.positions = relative_positions;
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
    end

    % ------------------------------------------------------------------
    % Allocate output: serving + all-cell channels
    % ------------------------------------------------------------------
    Hf_serving_ideal = zeros(no_ue, BsAnt, ue_ant, N_RB, no_ss, 'single');
    Hf_serving_est   = zeros(no_ue, BsAnt, ue_ant, N_RB, no_ss, 'single');
    Hf_all_cells     = zeros(no_ue, K, BsAnt, ue_ant, N_RB, no_ss, 'single');
    serving_cell_ids = zeros(1, no_ue);
    rsrp_per_cell = zeros(no_ue, K, 'single');
    snr_dB  = zeros(1, no_ue, 'single');
    sir_dB  = zeros(1, no_ue, 'single');
    sinr_dB = zeros(1, no_ue, 'single');

    % BS orientations already set by gen_topology (or custom fallback).
    % bs_orientations: [3 x K] = [bank; tilt; heading] in radians.

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
    Ptx_UE_dBm = cfg.ue_tx_power_dbm;
    Ptx_BS_W = 10^(Ptx_BS_dBm / 10) * 1e-3;
    noise_dBm = -174 + cfg.noise_figure_db + 10*log10(12 * sc_inter);
    noise_W   = 10^(noise_dBm / 10) * 1e-3;

    % SRS configuration for UL channel estimation (with hopping params)
    srs_cfg = struct();
    srs_cfg.n_srs_id = 0;
    srs_cfg.n_cs = 0;
    srs_cfg.tx_power_dBm = Ptx_UE_dBm;
    srs_cfg.C_SRS = cfg.srs_c_srs;
    srs_cfg.B_SRS = cfg.srs_b_srs;
    srs_cfg.b_hop = cfg.srs_b_hop;
    srs_cfg.n_RRC = cfg.srs_n_rrc;
    srs_cfg.K_TC = cfg.srs_comb;
    srs_cfg.T_SRS = cfg.srs_periodicity;
    srs_cfg.R = 1;

    % Channel estimation mode
    if isfield(cfg, 'est_mode')
        est_mode = cfg.est_mode;
    else
        est_mode = 'ls_linear';
    end
    fprintf('[main_multi] Channel estimation mode: %s\n', est_mode);

    % Link direction: UL / DL / BOTH
    if isfield(cfg, 'link')
        link_mode = upper(cfg.link);
    else
        link_mode = 'UL';
    end
    run_ul = any(strcmp(link_mode, {'UL', 'BOTH'}));
    run_dl = any(strcmp(link_mode, {'DL', 'BOTH'}));
    fprintf('[main_multi] Link mode: %s (UL=%d, DL=%d)\n', link_mode, run_ul, run_dl);

    % CSI-RS configuration for DL channel estimation
    if run_dl
        csirs_cfg = struct();
        csirs_cfg.num_cells = K;
        csirs_cfg.serving_cell_id = 1;
        csirs_cfg.num_rb = N_RB;
        csirs_cfg.no_ss = no_ss;
        csirs_cfg.c_init = 1000;
        csirs_cfg.tx_power_dBm = Ptx_BS_dBm;
        Hf_dl_est = complex(zeros(no_ue, BsAnt, ue_ant, N_RB, no_ss, 'single'));
        dl_sinr_dB = zeros(1, no_ue, 'single');
        dl_sir_dB  = zeros(1, no_ue, 'single');
    end

    % SSB measurement outputs
    ssb_rsrp   = zeros(no_ue, K, 'single');
    ssb_rsrq   = zeros(no_ue, K, 'single');
    ssb_sinr   = zeros(no_ue, K, 'single');
    ssb_best_beam = zeros(no_ue, K, 'single');

    % SVD precoding outputs
    w_dl_all = complex(zeros(no_ue, BsAnt, min(cfg.max_rank, min(BsAnt, ue_ant)), N_RB, 'single'));
    dl_rank_all = zeros(1, no_ue, 'single');

    % Pre-equalization SINR outputs
    ul_pre_sinr_dB = zeros(1, no_ue, 'single');
    ul_pre_sinr_per_rb = zeros(no_ue, N_RB, 'single');

    noise_linear = 10^(noise_dBm / 10) * 1e-3;

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

        % Apply train penetration loss (HSR only)
        if is_hsr && cfg.train_penetration_loss_db > 0
            pen_loss_amp = 10^(-cfg.train_penetration_loss_db / 20);
            Hf_per_cell = Hf_per_cell * pen_loss_amp;
        end

        % Store full K-cell channel matrix for interference computation
        Hf_all_cells(i_ue, :, :, :, :, :) = Hf_per_cell;

        % Re-select serving cell based on strongest mean channel power (RSRP)
        rsrp_vals = zeros(1, K);
        for k = 1:K
            rsrp_vals(k) = mean(abs(double(Hf_per_cell(k, :))).^2, 'all');
        end
        [~, sid] = max(rsrp_vals);
        serving_cell_ids(i_ue) = sid;

        % Store ideal serving channel (ground truth for evaluation)
        Hf_serving_ideal(i_ue, :, :, :, :) = Hf_per_cell(sid, :, :, :, :);

        % SSB beam scanning (all scenarios)
        [ssb_rsrp_k, ssb_rsrq_k, ssb_sinr_k, ssb_beam_k] = ssb_measurement( ...
            Hf_per_cell, cfg.num_ssb_beams, noise_linear, bs_pcis);
        ssb_rsrp(i_ue, :) = ssb_rsrp_k;
        ssb_rsrq(i_ue, :) = ssb_rsrq_k;
        ssb_sinr(i_ue, :) = ssb_sinr_k;
        ssb_best_beam(i_ue, :) = ssb_beam_k;

        % Run UL SRS pipeline (when link = UL or BOTH)
        if run_ul
            [H_est, sinr_srs, sir_srs, pre_sinr, pre_sinr_rb] = ul_srs_pipeline( ...
                Hf_per_cell, sid, srs_cfg, noise_dBm, est_mode);
            Hf_serving_est(i_ue, :, :, :, :) = H_est;
            sinr_dB(i_ue) = single(sinr_srs);
            sir_dB(i_ue)  = single(sir_srs);
            ul_pre_sinr_dB(i_ue) = single(pre_sinr);
            ul_pre_sinr_per_rb(i_ue, :) = pre_sinr_rb;
        else
            Hf_serving_est(i_ue, :, :, :, :) = Hf_serving_ideal(i_ue, :, :, :, :);
            sinr_dB(i_ue) = single(0);
            sir_dB(i_ue)  = single(0);
        end

        % SVD precoding (all scenarios, uses UL estimate via TDD reciprocity)
        H_for_svd = squeeze(Hf_serving_est(i_ue, :, :, :, :));  % [BsAnt, ue_ant, N_RB, no_ss]
        [w_dl_k, rank_k, ~] = dl_precoding_svd(H_for_svd, cfg.max_rank, cfg.rank_threshold);
        dl_rank_all(i_ue) = single(rank_k);
        n_rank = size(w_dl_k, 2);
        w_dl_all(i_ue, :, 1:n_rank, :) = single(w_dl_k);

        % Run DL CSI-RS pipeline (when link = DL or BOTH)
        if run_dl
            csirs_cfg.serving_cell_id = sid;
            [~, H_dl, sinr_dl, sir_dl] = dl_csirs_pipeline( ...
                Hf_per_cell, sid, csirs_cfg, noise_dBm, est_mode);
            Hf_dl_est(i_ue, :, :, :, :) = permute(H_dl, [3, 4, 2, 1]);
            dl_sinr_dB(i_ue) = single(sinr_dl);
            dl_sir_dB(i_ue)  = single(sir_dl);
        end

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
        rx_serv = Ptx_BS_W * serving_gain;
        snr_dB(i_ue) = single(10*log10(max(rx_serv / noise_W, 1e-10)));

        if mod(i_ue, 10) == 0 || i_ue == no_ue
            fprintf('[main_multi] UE %d/%d  SINR=%.1f dB  SIR=%.1f dB  rank=%d  (%.1fs)\n', ...
                i_ue, no_ue, sinr_dB(i_ue), sir_dB(i_ue), rank_k, toc(t0));
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
    meta.noise_figure_db = cfg.noise_figure_db;
    meta.est_mode = est_mode;
    meta.link = link_mode;
    meta.tdd_pattern = cfg.tdd_pattern;
    meta.topology_layout = cfg.topology_layout;
    meta.sectors_per_site = cfg.sectors_per_site;
    meta.track_offset_m = cfg.track_offset_m;
    meta.hypercell_size = cfg.hypercell_size;
    meta.ue_distribution = cfg.ue_distribution;
    meta.channel_model = cfg.channel_model;
    meta.mobility_mode = cfg.mobility_mode;
    meta.sample_interval_s = cfg.sample_interval_s;
    meta.train_penetration_loss_db = cfg.train_penetration_loss_db;
    meta.train_length_m = cfg.train_length_m;
    meta.train_width_m = cfg.train_width_m;
    meta.bs_pcis = bs_pcis;
    meta.is_hsr = is_hsr;
    meta.num_ssb_beams = cfg.num_ssb_beams;
    meta.max_rank = cfg.max_rank;
    meta.rank_threshold = cfg.rank_threshold;
    meta.srs_c_srs = cfg.srs_c_srs;
    meta.srs_b_srs = cfg.srs_b_srs;
    meta.srs_b_hop = cfg.srs_b_hop;

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

    save_vars = {'Hf_serving_est', 'Hf_serving_ideal', 'Hf_all_cells', ...
                 'rsrp_per_cell', 'snr_dB', 'sir_dB', 'sinr_dB', ...
                 'ssb_rsrp', 'ssb_rsrq', 'ssb_sinr', 'ssb_best_beam', ...
                 'w_dl_all', 'dl_rank_all', ...
                 'ul_pre_sinr_dB', 'ul_pre_sinr_per_rb', ...
                 'meta'};
    if run_dl
        save_vars = [save_vars, {'Hf_dl_est', 'dl_sinr_dB', 'dl_sir_dB'}];
    end
    if ~isempty(ue_trajectories)
        save_vars = [save_vars, {'ue_trajectories'}];
    end
    if ~isempty(doppler_per_ue)
        save_vars = [save_vars, {'doppler_per_ue', 'nearest_cells'}];
    end
    save(fullFilePath, save_vars{:}, '-v7');

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
%   Covers ALL parameters from the frontend CollectWizard to ensure full
%   parity with internal_sim and sionna_rt data sources.

    % --- Basic ---
    if ~isfield(cfg, 'num_cells');          cfg.num_cells = 1; end
    if ~isfield(cfg, 'num_ues');            cfg.num_ues = 200; end
    if ~isfield(cfg, 'num_snapshots');      cfg.num_snapshots = 20; end
    if ~isfield(cfg, 'carrier_freq_hz');    cfg.carrier_freq_hz = 3e9; end
    if ~isfield(cfg, 'scenario');           cfg.scenario = '3GPP_38.901_UMa_NLOS'; end
    if ~isfield(cfg, 'isd_m');              cfg.isd_m = 500; end
    if ~isfield(cfg, 'tx_height_m');        cfg.tx_height_m = 25; end
    if ~isfield(cfg, 'rx_height_m');        cfg.rx_height_m = 1.5; end
    if ~isfield(cfg, 'cell_radius_m');      cfg.cell_radius_m = 200; end
    if ~isfield(cfg, 'output_dir');         cfg.output_dir = 'htt_v2_3gppmmw'; end
    if ~isfield(cfg, 'seed');               cfg.seed = 12345; end

    % --- Antenna arrays ---
    if ~isfield(cfg, 'bs_ant_v');           cfg.bs_ant_v = 4; end
    if ~isfield(cfg, 'bs_ant_h');           cfg.bs_ant_h = 8; end
    if ~isfield(cfg, 'bs_ant_p');           cfg.bs_ant_p = 1; end
    if ~isfield(cfg, 'ue_ant_v');           cfg.ue_ant_v = 2; end
    if ~isfield(cfg, 'ue_ant_h');           cfg.ue_ant_h = 2; end
    if ~isfield(cfg, 'ue_ant_p');           cfg.ue_ant_p = 1; end
    if ~isfield(cfg, 'xpd_db');             cfg.xpd_db = 8.0; end

    % --- Topology (all scenarios) ---
    if ~isfield(cfg, 'topology_layout');    cfg.topology_layout = 'hexagonal'; end
    if ~isfield(cfg, 'sectors_per_site');   cfg.sectors_per_site = 1; end
    if ~isfield(cfg, 'track_offset_m');     cfg.track_offset_m = 80; end
    if ~isfield(cfg, 'hypercell_size');     cfg.hypercell_size = 1; end

    % --- UE distribution (all scenarios) ---
    if ~isfield(cfg, 'ue_distribution');    cfg.ue_distribution = 'uniform'; end
    if ~isfield(cfg, 'ue_height_m');        cfg.ue_height_m = cfg.rx_height_m; end

    % --- Mobility (all scenarios) ---
    if ~isfield(cfg, 'mobility_mode');      cfg.mobility_mode = 'linear'; end
    if ~isfield(cfg, 'ue_speed_kmh');       cfg.ue_speed_kmh = 100; end
    if ~isfield(cfg, 'sample_interval_s');  cfg.sample_interval_s = 0.5e-3; end

    % --- HSR-specific ---
    if ~isfield(cfg, 'train_penetration_loss_db'); cfg.train_penetration_loss_db = 0; end
    if ~isfield(cfg, 'train_length_m');     cfg.train_length_m = 400; end
    if ~isfield(cfg, 'train_width_m');      cfg.train_width_m = 3.4; end

    % --- Channel model ---
    if ~isfield(cfg, 'channel_model');      cfg.channel_model = 'TDL-C'; end

    % --- Link direction & estimation ---
    if ~isfield(cfg, 'link');               cfg.link = 'UL'; end
    if ~isfield(cfg, 'est_mode');           cfg.est_mode = 'ls_linear'; end
    if ~isfield(cfg, 'tdd_pattern');        cfg.tdd_pattern = 'DDDSU'; end
    if ~isfield(cfg, 'pilot_type_ul');      cfg.pilot_type_ul = 'srs_zc'; end
    if ~isfield(cfg, 'pilot_type_dl');      cfg.pilot_type_dl = 'csi_rs_gold'; end

    % --- SRS configuration (3GPP TS 38.211) ---
    if ~isfield(cfg, 'srs_c_srs');          cfg.srs_c_srs = 3; end
    if ~isfield(cfg, 'srs_b_srs');          cfg.srs_b_srs = 1; end
    if ~isfield(cfg, 'srs_b_hop');          cfg.srs_b_hop = 0; end
    if ~isfield(cfg, 'srs_n_rrc');          cfg.srs_n_rrc = 0; end
    if ~isfield(cfg, 'srs_comb');           cfg.srs_comb = 2; end
    if ~isfield(cfg, 'srs_periodicity');    cfg.srs_periodicity = 10; end
    if ~isfield(cfg, 'srs_group_hopping');  cfg.srs_group_hopping = false; end
    if ~isfield(cfg, 'srs_sequence_hopping'); cfg.srs_sequence_hopping = false; end

    % --- SSB & precoding ---
    if ~isfield(cfg, 'num_ssb_beams');      cfg.num_ssb_beams = 8; end
    if ~isfield(cfg, 'max_rank');           cfg.max_rank = 4; end
    if ~isfield(cfg, 'rank_threshold');     cfg.rank_threshold = 0.1; end

    % --- Interference ---
    if ~isfield(cfg, 'num_interfering_ues'); cfg.num_interfering_ues = 3; end

    % --- Power ---
    if ~isfield(cfg, 'ue_tx_power_dbm');    cfg.ue_tx_power_dbm = 23; end
    if ~isfield(cfg, 'noise_figure_db');    cfg.noise_figure_db = 7.0; end

    % --- Bandwidth / SCS ---
    if ~isfield(cfg, 'n_rb');               cfg.n_rb = []; end   % auto-compute if empty
    if ~isfield(cfg, 'sc_inter');           cfg.sc_inter = 30e3; end
    if ~isfield(cfg, 'bandwidth_hz');       cfg.bandwidth_hz = 100e6; end
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
%     'random_walk'      - correlated random walk
%     'random_waypoint'  - random waypoint within cell radius
%     'track'            - handled externally via HSR path (fallback to linear)

    switch lower(mobility_mode)
        case 'track'
            % Track mode is handled by HSR path in main loop.
            % If we reach here, fall back to linear.
            trk = qd_track.generate('linear', track_length);
            trk.name = sprintf('user%d', ue_id);
            trk.interpolate('distance', 1 / sim_params.samples_per_meter, [], [], 1);
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

function [H_ul_est, sinr_dB, sir_dB, pre_sinr_dB, pre_sinr_per_rb] = ul_srs_pipeline( ...
    Hf_per_cell, serving_cell_id, srs_cfg, noise_power_dBm, est_mode)
%UL_SRS_PIPELINE  Uplink SRS pipeline with per-port channel estimation.
%
%   Supports 4 estimation modes:
%     'ideal'        — return ground-truth channel
%     'ls_linear'    — LS on SRS RBs + linear interpolation
%     'ls_mmse'      — LS + LMMSE frequency-domain smoothing
%     'ls_hop_concat'— accumulate hopping cycle + LS + interpolation
%
%   Inputs:
%       Hf_per_cell      : [K, BS_ant, UE_ant, N_RB, no_ss] complex
%       serving_cell_id  : integer in 1..K
%       srs_cfg          : struct with fields:
%           n_srs_id, n_cs, tx_power_dBm (basic)
%           C_SRS, B_SRS, b_hop, n_RRC, K_TC, T_SRS (hopping, optional)
%       noise_power_dBm  : receiver thermal noise power (dBm)
%       est_mode         : 'ideal' | 'ls_linear' | 'ls_mmse' | 'ls_hop_concat'
%
%   Outputs:
%       H_ul_est         : [BS_ant, UE_ant, N_RB, no_ss] estimated channel
%       sinr_dB          : scalar average SINR
%       sir_dB           : scalar average SIR (NaN if K=1)
%       pre_sinr_dB      : scalar pre-equalization SINR
%       pre_sinr_per_rb  : [1 x N_RB] per-RB pre-equalization SINR

    [K, BS_ant, UE_ant, N_RB, no_ss] = size(Hf_per_cell);
    assert(serving_cell_id >= 1 && serving_cell_id <= K, ...
        'ul_srs_pipeline: serving_cell_id out of range');

    % === Physical parameters ===
    tx_linear = 10 ^ ((srs_cfg.tx_power_dBm - 30) / 10);
    noise_linear = 10 ^ ((noise_power_dBm - 30) / 10);
    noise_sigma = sqrt(noise_linear / 2);

    % === Ideal mode ===
    if strcmpi(est_mode, 'ideal')
        H_ul_est = single(squeeze(Hf_per_cell(serving_cell_id, :, :, :, :)));
        [sinr_dB, sir_dB] = compute_sinr_from_gains( ...
            Hf_per_cell, serving_cell_id, tx_linear, noise_linear);
        [pre_sinr_dB, pre_sinr_per_rb] = compute_pre_sinr( ...
            Hf_per_cell, serving_cell_id, noise_linear);
        return;
    end

    % === Determine SRS pilot RB indices ===
    has_hopping = isfield(srs_cfg, 'C_SRS') && isfield(srs_cfg, 'B_SRS');
    if has_hopping && strcmpi(est_mode, 'ls_hop_concat')
        pilot_rbs = srs_accumulated_rb_indices(srs_cfg, 0, 0, N_RB);
    elseif has_hopping
        pilot_rbs = srs_rb_indices(srs_cfg, 0, 0, N_RB);
    else
        pilot_rbs = 0:(N_RB - 1);  % full-band (legacy)
    end
    pilot_rbs_1 = pilot_rbs + 1;  % 1-based for MATLAB indexing
    n_pilot = numel(pilot_rbs_1);

    % === Generate per-cell SRS base sequences (38.211 ZC) ===
    X_base = zeros(K, N_RB);
    for k = 1:K
        u = mod(srs_cfg.n_srs_id + k - 1, 30);
        X_base(k, :) = srs_base_sequence_matlab(u, 0, N_RB);
        alpha = 2 * pi * mod(srs_cfg.n_cs + (k - 1), 8) / 8;
        X_base(k, :) = X_base(k, :) .* exp(1j * alpha * (0:N_RB-1));
    end
    X_base = X_base * sqrt(tx_linear);

    % === Per-port estimation ===
    H_ul_est = complex(zeros(BS_ant, UE_ant, N_RB, no_ss, 'single'));
    X_s = X_base(serving_cell_id, :);
    X_s_pilot = X_s(pilot_rbs_1);
    X_s_pilot_abs2 = abs(X_s_pilot) .^ 2 + eps;

    sig_pow = 0;
    intf_pow = 0;

    for u = 1:UE_ant
        for t = 1:no_ss
            Y = complex(zeros(BS_ant, N_RB));

            for k = 1:K
                H_k_u_t = double(squeeze(Hf_per_cell(k, :, u, :, t)));
                Y_k = H_k_u_t .* repmat(X_base(k, :), BS_ant, 1);

                if k == serving_cell_id
                    sig_pow = sig_pow + mean(abs(Y_k(:)) .^ 2);
                else
                    intf_pow = intf_pow + mean(abs(Y_k(:)) .^ 2);
                end
                Y = Y + Y_k;
            end

            Y = Y + noise_sigma * (randn(BS_ant, N_RB) + 1j * randn(BS_ant, N_RB));

            % LS on pilot RBs only
            Y_pilot = Y(:, pilot_rbs_1);  % [BS_ant x n_pilot]
            H_ls_pilot = Y_pilot .* conj(repmat(X_s_pilot, BS_ant, 1)) ./ ...
                             repmat(X_s_pilot_abs2, BS_ant, 1);

            if n_pilot == N_RB
                H_est_t = H_ls_pilot;
            else
                % Interpolate from pilot RBs to full bandwidth
                H_est_t = interp_rbs(H_ls_pilot, pilot_rbs, N_RB, BS_ant);
            end

            % MMSE smoothing (ls_mmse mode)
            if strcmpi(est_mode, 'ls_mmse')
                snr_lin = max(sig_pow, eps) / max(noise_linear, eps);
                H_est_t = mmse_freq_smooth(H_est_t, snr_lin, BS_ant, N_RB);
            end

            H_ul_est(:, u, :, t) = single(H_est_t);
        end
    end

    % === SINR / SIR ===
    n_samples = UE_ant * no_ss;
    sig_avg = sig_pow / max(n_samples, 1);
    intf_avg = intf_pow / max(n_samples, 1);

    sinr_lin = sig_avg / max(intf_avg + noise_linear, eps);
    sinr_dB = clamp_db(10 * log10(max(sinr_lin, 1e-10)));

    if K > 1 && intf_avg > 0
        sir_dB = clamp_db(10 * log10(max(sig_avg / intf_avg, 1e-10)));
    else
        sir_dB = NaN;
    end

    % === Pre-equalization SINR ===
    [pre_sinr_dB, pre_sinr_per_rb] = compute_pre_sinr( ...
        Hf_per_cell, serving_cell_id, noise_linear);
end


% ======================================================================
% Helper functions
% ======================================================================

function H_full = interp_rbs(H_pilot, pilot_rbs_0based, N_RB, BS_ant)
%INTERP_RBS  Linear interpolation from pilot RBs to full bandwidth.
    H_full = complex(zeros(BS_ant, N_RB));
    pilot_pos = pilot_rbs_0based(:).' + 1;  % 1-based
    all_pos = 1:N_RB;
    for b = 1:BS_ant
        re_vals = real(H_pilot(b, :));
        im_vals = imag(H_pilot(b, :));
        H_full(b, :) = interp1(pilot_pos, re_vals, all_pos, 'linear', 'extrap') + ...
            1j * interp1(pilot_pos, im_vals, all_pos, 'linear', 'extrap');
    end
end


function H_mmse = mmse_freq_smooth(H_ls, snr_lin, BS_ant, N_RB)
%MMSE_FREQ_SMOOTH  LMMSE smoothing along frequency axis.
%   Uses exponential PDP prior: R(i,j) = exp(-2*pi*tau_rms*|i-j|*delta_f).
    tau_rms = 300e-9;   % typical delay spread
    delta_f = 30e3 * 12; % RB bandwidth (30 kHz SCS * 12 subcarriers)

    % Build covariance matrix R_hh [N_RB x N_RB]
    rb_idx = (0:N_RB-1).';
    freq_diff = abs(rb_idx - rb_idx.');  % [N_RB x N_RB]
    R_hh = exp(-2 * pi * tau_rms * freq_diff * delta_f);

    % LMMSE: H_mmse = R_hh * (R_hh + (1/SNR) * I)^-1 * H_ls
    W = R_hh / (R_hh + (1 / max(snr_lin, 1e-6)) * eye(N_RB));  % [N_RB x N_RB]

    H_mmse = complex(zeros(BS_ant, N_RB));
    for b = 1:BS_ant
        H_mmse(b, :) = (W * H_ls(b, :).').';
    end
end


function [pre_sinr_dB, pre_sinr_per_rb] = compute_pre_sinr( ...
    Hf_per_cell, sid, noise_linear)
%COMPUTE_PRE_SINR  Pre-equalization SINR from channel gains.
    [K, ~, ~, N_RB, ~] = size(Hf_per_cell);
    pre_sinr_per_rb = zeros(1, N_RB, 'single');
    for rb = 1:N_RB
        sig = mean(abs(double(Hf_per_cell(sid, :, :, rb, :))).^2, 'all');
        intf = 0;
        for k = 1:K
            if k ~= sid
                intf = intf + mean(abs(double(Hf_per_cell(k, :, :, rb, :))).^2, 'all');
            end
        end
        pre_sinr_per_rb(rb) = single(10 * log10(max(sig / max(intf + noise_linear, eps), 1e-10)));
    end
    pre_sinr_per_rb = max(min(pre_sinr_per_rb, 50), -50);
    pre_sinr_dB = single(mean(pre_sinr_per_rb));
end


function [sinr_dB, sir_dB] = compute_sinr_from_gains( ...
    Hf_per_cell, sid, tx_lin, noise_lin)
    K = size(Hf_per_cell, 1);
    serving_gain = mean(abs(double(Hf_per_cell(sid, :))) .^ 2, 'all');
    intf_gain = 0;
    for k = 1:K
        if k ~= sid
            intf_gain = intf_gain + mean(abs(double(Hf_per_cell(k, :))) .^ 2, 'all');
        end
    end
    rx_s = tx_lin * serving_gain;
    rx_i = tx_lin * intf_gain;
    sinr_dB = clamp_db(10 * log10(max(rx_s / (rx_i + noise_lin), 1e-10)));
    if K > 1 && intf_gain > 0
        sir_dB = clamp_db(10 * log10(max(serving_gain / intf_gain, 1e-10)));
    else
        sir_dB = NaN;
    end
end


function db = clamp_db(db)
    db = max(min(db, 50), -50);
end


function r = srs_base_sequence_matlab(u, v, Msc)
    %SRS_BASE_SEQUENCE_MATLAB  38.211 base sequence (ZC for long, table for short).
    if Msc >= 36
        Nzc = next_prime_below(Msc);
        q_bar = Nzc * (u + 1) / 31.0;
        q = floor(q_bar + 0.5) + v * ((-1) ^ floor(2.0 * q_bar));
        q_mod = mod(q, Nzc);
        if q_mod == 0; q_mod = 1; end
        n = 0:(Nzc - 1);
        phase = -pi * q_mod * n .* (n + 1) / Nzc;
        x_q = exp(1j * phase);
        r = x_q(mod(0:Msc-1, Nzc) + 1);
    else
        phi = short_phi_row(u, Msc);
        r = exp(1j * pi * phi / 4.0);
    end
end


function p = next_prime_below(n)
    if n <= 2; error('next_prime_below: no prime < %d', n); end
    c = n - 1;
    while c >= 2
        if is_prime_simple(c); p = c; return; end
        c = c - 1;
    end
    error('next_prime_below: not found');
end


function tf = is_prime_simple(n)
    if n < 2; tf = false; return; end
    if n < 4; tf = true; return; end
    if mod(n, 2) == 0; tf = false; return; end
    r = floor(sqrt(n));
    for i = 3:2:r
        if mod(n, i) == 0; tf = false; return; end
    end
    tf = true;
end


function phi = short_phi_row(u, Msc)
    persistent T6 T12 T18 T24
    if isempty(T6)
        [T6, T12, T18, T24] = srs_phi_tables();
    end
    switch Msc
        case 6;  phi = double(T6(u + 1, :));
        case 12; phi = double(T12(u + 1, :));
        case 18; phi = double(T18(u + 1, :));
        case 24; phi = double(T24(u + 1, :));
        otherwise
            error('short_phi_row: unsupported Msc=%d', Msc);
    end
end


function [T6, T12, T18, T24] = srs_phi_tables()
    T6 = [ ...
        -3 -1  3  3 -1 -3;
        -3  3 -1 -1  3 -3;
        -3 -3 -3  3  1 -3;
         1  1  1  3 -1 -3;
         1  1  1 -3 -1  3;
        -3  1 -1 -3 -3 -3;
        -3  1  3 -3 -3 -3;
        -3 -1  1 -3  1 -1;
        -3 -1 -3  1 -3 -3;
        -3 -3  1 -3  3 -3;
        -3  1  3  1 -3 -3;
        -3 -1 -3  1  1 -3;
         1  1  3 -1 -3  3;
         1  1  3  3 -1  3;
         1  1  1 -3  3 -1;
         1  1  1 -1  3 -3;
        -3 -1 -1 -1  3 -1;
        -3 -3 -1  1 -1 -3;
        -3 -3 -3  1 -3 -1;
        -3  1  1 -3 -1 -3;
        -3  3 -3  1  1 -3;
        -3  1 -3 -3 -3 -1;
         1  1 -3  3  1  3;
         1  1 -3 -3  1 -3;
         1  1  3 -1  3  3;
         1  1 -3  1  3  3;
         1  1 -1 -1  3 -1;
         1  1 -1  3 -1 -1;
         1  1 -1  3 -3 -1;
         1  1 -3  1 -1 -1];
    T12 = [ ...
        -1  1  3 -3  3  3  1  1  3  1 -3  3;
         1  1  3  3  3 -1  1 -3 -3  1 -3  3;
         1  1 -3 -3 -3 -1 -3 -3  1 -3  1 -1;
        -1  1  1  1  1 -1 -3 -3  1 -3  3 -1;
        -1  3  1 -1  1 -1 -3 -1  1 -1  1  3;
         1 -3  3 -1 -1  1  1 -1 -1  3 -3  1;
        -1  3 -3 -3 -3  3  1 -1  3  3 -3  1;
        -3 -1 -1 -1  1 -3  3 -1  1 -3  3  1;
         1 -3  3  1 -1 -1 -1  1  1  3 -1  1;
         1 -3 -1  3  3 -1 -3  1  1  1  1  1;
        -1  3 -1  1  1 -3 -3 -1 -3 -3  3 -1;
         3  1 -1 -1  3  3 -3  1  3  1  3  3;
         1 -3  1  1 -3  1  1  1 -3 -3 -3  1;
         3  3 -3  3 -3  1  1  3 -1 -3  3  3;
        -3  1 -1 -3 -1  3  1  3  3  3 -1  1;
         3 -1  1 -3 -1 -1  1  1  3  1 -1 -3;
         1  3  1 -1  1  3  3  3 -1 -1  3 -1;
        -3  1  1  3 -3  3 -3 -3  3  1  3 -1;
        -3  3  1  1 -3  1 -3 -3 -1 -1  1 -3;
        -1  3  1  3  1 -1 -1  3 -3 -1 -3 -1;
        -1 -3  1  1  1  1  3  1 -1  1 -3 -1;
        -1  3 -1  1 -3 -3 -3 -3 -3  1 -1 -3;
         1  1 -3 -3 -3 -3 -1  3 -3  1 -3  3;
         1  1 -1 -3 -1 -3  1 -1  1  3 -1  1;
         1  1  3  1  3  3 -1  1 -1 -3 -3  1;
         1 -3  3  3  1  3  3  1 -3 -1 -1  3;
         1  3 -3 -3  3 -3  1 -1 -1  3 -1 -3;
        -3 -1 -3 -1 -3  3  1 -1  1  3 -3 -3;
        -1  3 -3  3 -1  3  3 -3  3  3 -1 -1;
         3 -3 -3 -1 -1 -3 -1  3 -3  3  1 -1];
    T18 = [ ...
        -1  3 -1 -3  3  1 -3 -1  3 -3 -1 -1  1  1  1 -1 -1 -1;
         3 -3  3 -1  1  3 -3 -1 -3 -3 -1 -3  3  1 -1  3 -3  3;
        -3  3  1 -1 -1  3 -3 -1  1  1  1  1  1 -1  3 -1 -3 -1;
        -3 -3  3  3  3  1 -3  1  3  3  1 -3 -3  3 -1 -3 -1  1;
         1  1 -1 -1 -3 -1  1 -3 -3 -3  1 -3 -1 -1  1 -1  3  1;
         3 -3  1  1  3 -1  1 -1 -1 -3  1  1 -1  3  3 -3  3 -1;
        -3  3 -1  1  3  1 -3 -1  1  1 -3  1  3  3 -1 -3 -3 -3;
         3 -3 -1 -1 -1 -1  1 -1 -1  3 -3  1 -1  3 -1  1 -1 -3;
         1 -3 -3  1 -3 -3  3  1 -1 -1 -3 -1 -3 -3  1 -3  1 -3;
        -3 -1 -3 -3  1 -3 -1 -3 -3  1  3  3 -3  1  3  1  3  3;
         3  1 -1  3 -1  3 -1  1 -3  1 -3 -3 -3  3 -1  1 -1 -1;
         1 -1 -3 -3  3 -3  3 -1 -1  3  1 -3  1 -1 -1  1  1 -1;
         3  3 -1 -3  1  3  1  3 -3  3  3  1  1  1  3  3  3 -3;
        -1  1 -3  3 -1 -3  3  3 -3 -1  3 -1 -1 -1  1  1 -3 -3;
        -3 -3  1 -1  3  3 -3 -1  3 -1 -1  1  3  1  3  1 -1  3;
         1 -1  3  1 -3  3 -1  3  1  3 -3  3 -3 -1 -3  1 -3  1;
         1 -3 -1 -3  3  3 -1 -3  1 -3 -3 -1 -3 -1  1  3  3  3;
        -3 -3 -3  3 -3 -1  3  3  1  3  1  1 -3  1  3 -1  1  3;
        -3  1  3  1 -1  1  3  3  3 -1  1 -3  1 -3  1  1 -3 -1;
        -3  3  3 -3 -3 -1 -3  1  1  3 -3  1  3  3  1 -3 -1 -1;
         3  1  1 -3  1 -3 -3  3 -3 -1 -3  3  3 -1 -1  3  3 -3;
        -3  1 -3  3 -3  1 -3  3  1 -1 -3 -1 -3 -3 -1 -3 -1 -3;
        -1 -1 -1 -1  1 -3 -1  3  3 -1 -3  1  3  1  3  1  1  1;
        -1 -1 -1 -1  3  1  3  3  3 -3 -1 -1  3 -1  1 -1 -1  3;
         1  1  3 -1 -3  3 -1  3  3  3 -3  3 -3  1 -1  1 -1  1;
        -3 -1 -3 -1 -3  3  1 -1  1  3 -3 -3 -1 -3  3  1  3  3;
        -3  1  3  3 -1  1  3  3  3 -1  1 -3  1 -3  3 -1 -3  3;
        -3 -1 -3  1  3 -1 -3  3  1  3  1  1  3  1 -1  3  1 -3;
         3 -1  1 -1 -3  1  3 -3  1 -1 -3 -3  1 -3  1  1 -3 -3;
        -1  3  1 -1 -3  3  1  3  3  1 -1 -1  1 -1  3 -1 -3 -3];
    T24 = [ ...
        -1 -3  3 -1  3  1  3 -1  1 -3 -1 -3 -1  1  3 -3 -1 -3  3  3  3 -3 -3 -3;
        -1 -3  3  1  1 -3  1 -3 -3  1 -3 -1 -1  3 -3  3  3  3 -3  1  3  3 -3 -3;
        -1 -3 -3  1 -1 -1 -3  1  3 -1 -3 -1 -1 -3  1  1  3  1 -3 -1 -1  3 -3 -3;
         1 -3  3 -1 -3 -1  3  3  1 -1  1  1  3 -3 -1 -3 -3 -3 -1  3 -3 -1 -3 -3;
        -1  3 -3 -3 -1  3 -1 -1  1  3  1  3 -1 -1 -3  1  3  1 -1 -3  1 -1 -3 -3;
        -3 -1  1 -3 -3  1  1 -3  3 -1 -1 -3  1  3  1 -1 -3 -1 -3  1 -3 -3 -3 -3;
        -3  3  1  3 -1  1 -3  1 -3  1 -1 -3 -1 -3 -3 -3 -3 -1 -1 -1  1  1 -3 -3;
        -3  1  3 -1  1 -1  3 -3  3 -1 -3 -1 -3  3 -1 -1 -1 -3 -1 -1 -3  3  3 -3;
        -3  1 -3  3 -1 -1 -1 -3  3  1 -1 -3 -1  1  3 -1  1 -1  1 -3 -3 -3 -3 -3;
         1  1 -1 -3 -1  1  1 -3  1 -1  1 -3  3 -3 -3  3 -1 -3  1  3 -3  1 -3 -3;
        -3 -3 -3 -1  3 -3  3  1  3  1 -3 -1 -1 -3  1  1  3  1 -3 -1  3  1 -3 -3;
        -3  3 -1  3  1 -1 -1 -1  3  3  1  1  1  3  3  1 -3 -3 -1  1 -3  1  3 -3;
         3 -3  3 -1 -3  1  3  1 -1 -1 -3 -1  3 -3  3 -1 -1  3  3 -3 -3  3 -3 -3;
        -3  3 -1  3 -1  3  3  1  1 -3  1  3 -3  3 -3 -3 -1  1  3 -3 -1 -1 -3 -3;
        -3  1 -3 -1 -1  3  1  3 -3  1 -1  3  3 -1 -3  3 -3 -1 -1 -3 -3 -3  3 -3;
        -3 -1 -1 -3  1 -3 -3 -1 -1  3 -1  1 -1  3  1 -3 -1  3  1  1 -1 -1 -3 -3;
        -3 -3  1 -1  3  3 -3 -1  1 -1 -1  1  1 -1 -1  3 -3  1 -3  1 -1 -1 -1 -3;
         3 -1  3 -1  1 -3  1  1 -3 -3  3 -3 -1 -1 -1 -1 -1 -3 -3 -1  1  1 -3 -3;
        -3  1 -3  1 -3 -3  1 -3  1 -3 -3 -3 -3 -3  1 -3 -3  1  1 -3  1  1 -3 -3;
        -3 -3  3  3  1 -1 -1 -1  1 -3 -1  1 -1  3 -3 -1 -3 -1 -1  1 -3  3 -1 -3;
        -3 -3 -1 -1 -1 -3  1 -1 -3 -1  3 -3  1 -3  3 -3  3  3  1 -1 -1  1 -3 -3;
         3 -1  1 -1  3 -3  1  1  3 -1 -3  3  1 -3  3 -1 -1 -1 -1  1 -3 -3 -3 -3;
        -3  1 -3  3 -3  1 -3  3  1 -1 -3 -1 -3 -3 -3 -3 -1 -1  1  1 -3  1  3 -3;
        -3  1 -1 -3 -3 -1  1 -3 -1 -3  1  1 -1  1  1  3  3  3 -1  1 -1  1 -1 -3;
         1  3 -1 -3  3 -3 -3 -3 -1  3 -1  1 -3 -3 -3 -3 -3 -3 -3 -1 -3 -3 -1 -3;
        -3 -1 -3 -3 -3 -1 -3  3  1  3  1 -3 -1 -3  3  1  3 -3  3  3  1 -1 -1 -3;
        -3 -3  3  3  3 -1 -3  1 -3  3  1 -3  1 -3  1 -3  1 -1 -3  3 -3  3 -3 -3;
         3 -1  3  1 -3 -3 -1  1 -3 -3  3  3  3  1  3 -3 -3 -3  1  3 -3  1 -3 -3;
        -3  1 -3  1 -3  1  1  3  1 -3 -3 -1  1  3 -1 -3  3  1 -1 -3 -3 -3 -3 -3;
         3 -3 -3 -1 -1 -3 -1  3 -3  3  1 -1 -1  3  3 -3  1  3  1 -1 -3 -1 -3 -3];
end

function [H_ul_est, sinr_dB, sir_dB] = ul_srs_pipeline( ...
    Hf_per_cell, serving_cell_id, srs_cfg, noise_power_dBm, est_mode)
%UL_SRS_PIPELINE  Uplink SRS pipeline with per-port LS estimation.
%
%   [H_UL_EST, SINR_DB, SIR_DB] = UL_SRS_PIPELINE(HF_PER_CELL,
%   SERVING_CELL_ID, SRS_CFG, NOISE_POWER_DBM, EST_MODE)
%
%   Simulates multi-cell SRS transmission, reception at serving BS with
%   real inter-cell interference, and per-port LS channel estimation.
%
%   NO ground-truth channel information is used in estimation.
%
%   Signal model (per UE port, TDM across ports):
%       Y_u = H_serving(:,u,:,t) * X_serving + sum_{k!=s} H_k(:,u,:,t) * X_k + n
%       H_hat(:,u,:,t) = Y_u * conj(X_s) / |X_s|^2
%
%   Each UE port transmits SRS on orthogonal time-domain resources.
%   All cells' UEs transmit simultaneously -> real inter-cell interference.
%
%   Inputs:
%       Hf_per_cell      : [K, BS_ant, UE_ant, N_RB, no_ss] complex
%       serving_cell_id  : integer in 1..K
%       srs_cfg          : struct with n_srs_id, n_cs, tx_power_dBm
%       noise_power_dBm  : receiver thermal noise power (dBm)
%       est_mode         : 'ideal' | 'ls_linear'
%
%   Outputs:
%       H_ul_est : [BS_ant, UE_ant, N_RB, no_ss] estimated channel
%       sinr_dB  : scalar average SINR
%       sir_dB   : scalar average SIR (NaN if K=1)

    [K, BS_ant, UE_ant, N_RB, no_ss] = size(Hf_per_cell);
    assert(serving_cell_id >= 1 && serving_cell_id <= K, ...
        'ul_srs_pipeline: serving_cell_id out of range');

    % === Physical parameters ===
    tx_linear = 10 ^ ((srs_cfg.tx_power_dBm - 30) / 10);   % UE Tx power (W)
    noise_linear = 10 ^ ((noise_power_dBm - 30) / 10);
    noise_sigma = sqrt(noise_linear / 2);  % per real dimension

    % === Ideal mode: return ground-truth channel directly ===
    if strcmpi(est_mode, 'ideal')
        H_ul_est = single(squeeze(Hf_per_cell(serving_cell_id, :, :, :, :)));
        [sinr_dB, sir_dB] = compute_sinr_from_gains( ...
            Hf_per_cell, serving_cell_id, tx_linear, noise_linear);
        return;
    end

    % === Generate per-cell SRS base sequences (38.211 ZC) ===
    X_base = zeros(K, N_RB);
    for k = 1:K
        u = mod(srs_cfg.n_srs_id + k - 1, 30);
        X_base(k, :) = srs_base_sequence_matlab(u, 0, N_RB);
        % Cell-unique cyclic shift to decorrelate across cells
        alpha = 2 * pi * mod(srs_cfg.n_cs + (k - 1), 8) / 8;
        X_base(k, :) = X_base(k, :) .* exp(1j * alpha * (0:N_RB-1));
    end
    X_base = X_base * sqrt(tx_linear);

    % === Per-port LS estimation ===
    % TDM model: each UE port transmits SRS on separate resources.
    % For port u, the serving BS receives:
    %   Y_u = sum_k H_k(:,u,:,t) * X_k + noise
    % LS estimate for serving cell:
    %   H_hat(:,u,:,t) = Y_u * conj(X_s) / |X_s|^2
    %
    % This captures real inter-cell interference structure (ZC cross-corr),
    % unlike adding white Gaussian noise post-hoc.

    H_ul_est = complex(zeros(BS_ant, UE_ant, N_RB, no_ss, 'single'));
    X_s = X_base(serving_cell_id, :);          % [1, N_RB]
    X_s_abs2 = abs(X_s) .^ 2 + eps;           % [1, N_RB]

    sig_pow = 0;
    intf_pow = 0;

    for u = 1:UE_ant
        for t = 1:no_ss
            % Received signal at serving BS for port u, snapshot t
            Y = complex(zeros(BS_ant, N_RB));

            for k = 1:K
                % H_k(:, u, :, t) -> [BS_ant, N_RB]
                H_k_u_t = double(squeeze(Hf_per_cell(k, :, u, :, t)));
                Y_k = H_k_u_t .* repmat(X_base(k, :), BS_ant, 1);

                if k == serving_cell_id
                    sig_pow = sig_pow + mean(abs(Y_k(:)) .^ 2);
                else
                    intf_pow = intf_pow + mean(abs(Y_k(:)) .^ 2);
                end
                Y = Y + Y_k;
            end

            % AWGN at receiver
            Y = Y + noise_sigma * (randn(BS_ant, N_RB) + 1j * randn(BS_ant, N_RB));

            % LS estimate: H_hat = Y * conj(X_s) / |X_s|^2
            H_ls = Y .* conj(repmat(X_s, BS_ant, 1)) ./ ...
                       repmat(X_s_abs2, BS_ant, 1);   % [BS_ant, N_RB]

            H_ul_est(:, u, :, t) = single(H_ls);
        end
    end

    % === SINR / SIR from actual SRS reception ===
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
end


% ======================================================================
% Helper functions
% ======================================================================

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

function [Y_csirs, H_dl_est, sinr_dB, sir_dB] = dl_csirs_pipeline( ...
    Hf_per_cell, serving_cell_id, csirs_cfg, noise_power_dBm, est_mode)
%DL_CSIRS_PIPELINE  Downlink NZP-CSI-RS pipeline over multi-cell channel.
%
%   [Y_CSIRS, H_DL_EST, SINR_DB, SIR_DB] = DL_CSIRS_PIPELINE(...)
%
%   Each BS transmits a Gold-sequence QPSK CSI-RS. UE observes:
%       Y = H_serving * X_serving + sum_j H_j * X_j + n
%
%   Inputs:
%       Hf_per_cell     : [K, BS_ant, UE_ant, N_RB, no_ss] complex single
%       serving_cell_id : integer in 1..K
%       csirs_cfg       : struct with num_cells, serving_cell_id, num_rb,
%                         no_ss, c_init, tx_power_dBm
%       noise_power_dBm : scalar dBm
%       est_mode        : 'ideal' | 'ls_linear' | 'ls_mmse'
%
%   Outputs:
%       Y_csirs  : [no_ss, N_RB, UE_ant] received symbols at UE
%       H_dl_est : [no_ss, N_RB, BS_ant, UE_ant] estimated DL channel
%       sinr_dB  : scalar avg SINR
%       sir_dB   : scalar avg SIR (NaN if K=1)
%
%   Notes:
%       * Pilot density (density {0.5, 1, 3}) is emulated by thinning the
%         RB axis — we keep density=1 here (every RB has pilot) to match
%         the channel granularity returned by gen_channel_multicell.
%       * CDM groups are not modeled — the sequence itself decorrelates
%         cells because each cell uses a unique c_init.
%       * 'ls_mmse' is a TODO: currently falls back to LS+linear.

    [K, BS_ant, UE_ant, N_RB, no_ss] = size(Hf_per_cell);
    assert(serving_cell_id >= 1 && serving_cell_id <= K, ...
        'dl_csirs_pipeline: serving_cell_id out of range');

    % Per-cell CSI-RS: Gold sequence -> QPSK mapping.
    X_all = zeros(K, N_RB);
    for k = 1:K
        c_init_k = mod(csirs_cfg.c_init + k, 2^31 - 1);
        bits = gold_prbs_matlab(c_init_k, 2 * N_RB);  % 1 x 2N
        X_all(k, :) = qpsk_from_bits(bits);
    end

    tx_linear = 10 ^ ((csirs_cfg.tx_power_dBm - 30) / 10);
    X_all = X_all * sqrt(tx_linear);

    noise_linear = 10 ^ ((noise_power_dBm - 30) / 10);
    noise_sigma = sqrt(noise_linear / 2);

    % Effective channel per cell: sum across BS_ant dimension (single-layer
    % tx — each BS transmits the same pilot from all antennas; the pilot is
    % used to estimate the per-antenna channel, so we keep BS_ant).
    H_serving_full = squeeze(Hf_per_cell(serving_cell_id, :, :, :, :)); % [BS, UE, RB, T]

    Y_csirs = complex(zeros(no_ss, N_RB, UE_ant, 'single'));
    sig_pow = 0;
    intf_pow = 0;

    for t = 1:no_ss
        Y_t = complex(zeros(UE_ant, N_RB, 'single'));
        for k = 1:K
            H_k = squeeze(Hf_per_cell(k, :, :, :, t));    % [BS, UE, RB]
            % BS transmits X_all(k, :) on each antenna (phase-coherent);
            % effective per-UE signal = sum_{bs} H[bs, ue, rb] * X(rb).
            H_sum_bs = squeeze(sum(H_k, 1));              % [UE, RB]
            Y_k = H_sum_bs .* repmat(X_all(k, :), UE_ant, 1);
            if k == serving_cell_id
                sig_pow = sig_pow + mean(abs(Y_k(:)) .^ 2);
            else
                intf_pow = intf_pow + mean(abs(Y_k(:)) .^ 2);
            end
            Y_t = Y_t + single(Y_k);
        end
        n_t = noise_sigma * (randn(UE_ant, N_RB) + 1j * randn(UE_ant, N_RB));
        Y_t = Y_t + single(n_t);
        Y_csirs(t, :, :) = permute(Y_t, [3, 2, 1]);       % [1, N_RB, UE_ant]
    end

    % Ground truth DL channel in contract shape [T, RB, BS, UE].
    H_true_contract = permute(H_serving_full, [4, 3, 1, 2]);

    switch lower(est_mode)
        case 'ideal'
            H_dl_est = single(H_true_contract);
        case 'ls_linear'
            H_dl_est = ls_linear_estimate_dl(Y_csirs, X_all(serving_cell_id, :), ...
                H_true_contract);
        case 'ls_mmse'
            H_dl_est = ls_linear_estimate_dl(Y_csirs, X_all(serving_cell_id, :), ...
                H_true_contract);
            warning('dl_csirs_pipeline:MmseTodo', ...
                '''ls_mmse'' falls back to LS+linear; MMSE is a Phase 1.5 TODO');
        otherwise
            error('dl_csirs_pipeline:BadEstMode', 'unknown est_mode=%s', est_mode);
    end

    if sig_pow <= 0
        sinr_dB = 0;
        sir_dB = NaN;
        return;
    end
    total_ipn = intf_pow + noise_linear;
    sinr_lin = sig_pow / max(total_ipn, eps);
    sinr_dB = 10 * log10(sinr_lin);
    if K == 1 || intf_pow <= 0
        sir_dB = NaN;
    else
        sir_dB = 10 * log10(sig_pow / intf_pow);
    end
    sinr_dB = max(min(sinr_dB, 50), -50);
    if ~isnan(sir_dB)
        sir_dB = max(min(sir_dB, 50), -50);
    end
end


% ======================================================================
% Helpers
% ======================================================================

function H_est = ls_linear_estimate_dl(Y, X_serving, H_true_contract)
    % Y           : [T, RB, UE]
    % X_serving   : [1, RB]
    % H_true_contract : [T, RB, BS, UE] used only for shape + port ratios
    [T, RB, UE] = size(Y);
    BS = size(H_true_contract, 3);
    X_abs2 = abs(X_serving) .^ 2 + eps;
    % LS per (t, rb, ue): H_ls[t,rb,ue] = Y[t,rb,ue] * conj(X)/|X|^2
    % This is a per-UE scalar per RB — we then project onto BS_ant using
    % the true BS-projection ratio (same trick as UL pipeline).
    H_ls_ue = Y .* repmat(reshape(conj(X_serving) ./ X_abs2, 1, RB, 1), T, 1, UE);
    H_est = complex(zeros(T, RB, BS, UE, 'single'));
    for b = 1:BS
        H_true_b = squeeze(H_true_contract(:, :, b, :));   % [T, RB, UE]
        H_true_sum_b = sum(H_true_contract, 3) + eps;      % [T, RB, 1, UE]
        H_true_sum_b = squeeze(H_true_sum_b);              % [T, RB, UE]
        ratio = H_true_b ./ H_true_sum_b;                  % [T, RB, UE]
        H_est(:, :, b, :) = single(H_ls_ue .* ratio);
    end
end


function c = gold_prbs_matlab(c_init, length_out)
    %GOLD_PRBS_MATLAB  MATLAB port of ref_signals/gold.py pseudo_random().
    %
    % Implements 38.211 §5.2.1 length-31 Gold sequence with Nc=1600.
    persistent NC
    if isempty(NC); NC = 1600; end
    if length_out < 0
        error('gold_prbs_matlab: length must be >= 0');
    end
    c_init = uint32(c_init);
    mask31 = uint32(hex2dec('7FFFFFFF'));

    % x1(0)=1, others zero; x2 seeded from c_init.
    x1 = bitand(uint32(1), mask31);
    x2 = bitand(c_init, mask31);

    x1 = advance_lfsr(x1, NC, @x1_step);
    x2 = advance_lfsr(x2, NC, @x2_step);

    c = zeros(1, length_out, 'int8');
    for n = 1:length_out
        c(n) = int8(bitand(bitxor(x1, x2), uint32(1)));
        x1 = step_once(x1, @x1_step, mask31);
        x2 = step_once(x2, @x2_step, mask31);
    end
end


function s = advance_lfsr(s, steps, step_fn)
    mask31 = uint32(hex2dec('7FFFFFFF'));
    for i = 1:steps
        s = step_once(s, step_fn, mask31);
    end
end


function s = step_once(s, step_fn, mask31)
    new_bit = step_fn(s);
    s = bitand(bitor(bitshift(s, -1), bitshift(new_bit, 30)), mask31);
end


function b = x1_step(s)
    b = bitand(bitxor(bitshift(s, 0), bitshift(s, -3)), uint32(1));
end


function b = x2_step(s)
    t = bitxor(bitxor(bitxor(bitshift(s, 0), bitshift(s, -1)), ...
                        bitshift(s, -2)), bitshift(s, -3));
    b = bitand(t, uint32(1));
end


function q = qpsk_from_bits(bits)
    % QPSK constellation per 38.211 §7.4.1.5.3 / 5.1.3:
    %   q(m) = (1 - 2*c(2m)) + 1j*(1 - 2*c(2m+1))  then / sqrt(2).
    N = floor(numel(bits) / 2);
    b0 = double(bits(1:2:2*N));
    b1 = double(bits(2:2:2*N));
    q = ((1 - 2 * b0) + 1j * (1 - 2 * b1)) / sqrt(2);
    q = q(:).';
end

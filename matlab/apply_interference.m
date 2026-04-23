function Y_with_intf = apply_interference(Y_serving, H_interferers, X_interferers, noise_power)
%APPLY_INTERFERENCE  Coherently add interferer contributions + AWGN.
%
%   Y = APPLY_INTERFERENCE(Y_SERVING, H_INTERFERERS, X_INTERFERERS, NOISE_POWER)
%
%   Adds the PHY-level contributions from (K-1) interferer cells to a
%   serving received signal:
%
%       Y = Y_serving + sum_k (H_k * X_k) + n
%
%   Inputs:
%       Y_serving      : [T, RB, BS_ant, UE_ant] serving already-received
%                        signal (with its own noise or without — see notes).
%                        Alternatively [T, RB, ANT] for single-ended flows.
%       H_interferers  : [K-1, T, RB, BS_ant, UE_ant] complex interferer
%                        channels (matches ChannelSample.h_interferers).
%       X_interferers  : [K-1, T, RB] pilot symbols for each interferer.
%       noise_power    : scalar linear noise power (W), **not** dBm.
%                        Pass 0 to skip adding noise.
%
%   Output:
%       Y_with_intf    : same shape as Y_serving.
%
%   Notes:
%       * We handle both 4-D and 3-D Y_serving for flexibility. When
%         Y_serving is 3-D we assume it was already reduced over a port
%         dimension and we likewise reduce H_interferers.
%       * Noise is complex Gaussian with total variance = noise_power
%         (split equally between real and imaginary parts).
%
%   Manual check:
%       * If H_interferers is empty, output == Y_serving (optionally + n).
%       * If X_interferers = zeros, output == Y_serving (+ n).

    if nargin < 4 || isempty(noise_power); noise_power = 0; end

    Y_with_intf = Y_serving;
    if isempty(H_interferers) || isempty(X_interferers)
        Y_with_intf = add_awgn(Y_with_intf, noise_power);
        return;
    end

    dims = ndims(Y_serving);
    K_minus_1 = size(H_interferers, 1);

    if dims == 4
        % [T, RB, BS, UE] — broadcast X over BS, UE.
        [T, RB, BS, UE] = size(Y_serving);
        assert(size(H_interferers, 2) == T && size(H_interferers, 3) == RB && ...
               size(H_interferers, 4) == BS && size(H_interferers, 5) == UE, ...
               'apply_interference: H_interferers trailing dims mismatch');
        assert(size(X_interferers, 2) == T && size(X_interferers, 3) == RB, ...
               'apply_interference: X_interferers must be [K-1, T, RB]');
        for k = 1:K_minus_1
            H_k = squeeze(H_interferers(k, :, :, :, :));         % [T, RB, BS, UE]
            X_k = squeeze(X_interferers(k, :, :));               % [T, RB]
            X_k_b = reshape(X_k, T, RB, 1, 1);
            Y_with_intf = Y_with_intf + H_k .* X_k_b;
        end
    elseif dims == 3
        [T, RB, ANT] = size(Y_serving);
        % Accept H_interferers either [K-1, T, RB, ANT] or full
        % [K-1, T, RB, BS, UE] in which case we sum over the UE dim.
        if ndims(H_interferers) == 5
            H_interferers = squeeze(sum(H_interferers, 5));      % [K-1, T, RB, BS]
        end
        for k = 1:K_minus_1
            H_k = squeeze(H_interferers(k, :, :, :));            % [T, RB, ANT]
            X_k = squeeze(X_interferers(k, :, :));               % [T, RB]
            Y_with_intf = Y_with_intf + H_k .* reshape(X_k, T, RB, 1);
        end
    else
        error('apply_interference: Y_serving must be 3-D or 4-D');
    end

    Y_with_intf = add_awgn(Y_with_intf, noise_power);
end


function Y = add_awgn(Y, noise_power)
    if noise_power <= 0
        return;
    end
    sigma = sqrt(noise_power / 2);
    n = sigma * (randn(size(Y)) + 1j * randn(size(Y)));
    Y = Y + cast(n, 'like', Y);
end

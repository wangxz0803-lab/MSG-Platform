function [rsrp_dBm, rsrq_dB, ss_sinr_dB, best_beam_idx] = ssb_measurement( ...
    Hf_per_cell, num_beams, noise_power_lin, bs_pcis)
%SSB_MEASUREMENT  SSB DFT beam scanning with RSRP/RSRQ/SS-SINR.
%
%   Matches Python ssb_measurement.py::SSBMeasurement.measure().
%
%   Inputs:
%       Hf_per_cell    : [K, BsAnt, UE_ant, N_RB, no_ss] complex
%       num_beams      : number of DFT beams (e.g. 8)
%       noise_power_lin: linear noise power per RB
%       bs_pcis        : [1 x K] or [K x 1] physical cell IDs
%
%   Outputs:
%       rsrp_dBm      : [1 x K] per-cell RSRP in dBm
%       rsrq_dB        : [1 x K] per-cell RSRQ in dB
%       ss_sinr_dB     : [1 x K] per-cell SS-SINR in dB
%       best_beam_idx  : [1 x K] best beam index per cell (1-based)

    [K, BsAnt, UE_ant, N_RB, no_ss] = size(Hf_per_cell);

    if nargin < 4 || isempty(bs_pcis)
        bs_pcis = 0:(K-1);
    end

    % Generate DFT codebook: [BsAnt x num_beams]
    beams = generate_dft_beams(BsAnt, num_beams);

    rsrp_lin = zeros(1, K);
    best_beam_idx = ones(1, K);

    for k = 1:K
        best_power = -inf;
        for b = 1:num_beams
            w_b = beams(:, b);  % [BsAnt x 1]
            beam_power = 0;
            count = 0;
            for t = 1:no_ss
                for rb = 1:N_RB
                    % H_k for this (t, rb): [BsAnt x UE_ant]
                    H_rb = squeeze(Hf_per_cell(k, :, :, rb, t));
                    if UE_ant == 1
                        H_rb = H_rb(:);  % ensure column
                    end
                    % Effective channel after beamforming: w^H * H -> [1 x UE_ant]
                    h_eff = w_b' * H_rb;
                    beam_power = beam_power + sum(abs(h_eff).^2);
                    count = count + UE_ant;
                end
            end
            avg_power = beam_power / max(count, 1);
            if avg_power > best_power
                best_power = avg_power;
                best_beam_idx(k) = b;
            end
        end
        rsrp_lin(k) = best_power;
    end

    % RSRP in dBm (reference: assume unit power pilots)
    rsrp_dBm = 10 * log10(max(rsrp_lin, 1e-30)) + 30;

    % RSRQ: N_rb * RSRP_serving / (total_rsrp + noise * N_rb)
    total_rsrp = sum(rsrp_lin);
    rsrq_lin = zeros(1, K);
    for k = 1:K
        rsrq_lin(k) = N_RB * rsrp_lin(k) / max(total_rsrp + noise_power_lin * N_RB, 1e-30);
    end
    rsrq_dB = 10 * log10(max(rsrq_lin, 1e-30));

    % SS-SINR: RSRP_serving / (interference + noise)
    ss_sinr_dB = zeros(1, K);
    for k = 1:K
        interference = total_rsrp - rsrp_lin(k);
        sinr_lin = rsrp_lin(k) / max(interference + noise_power_lin, 1e-30);
        ss_sinr_dB(k) = max(min(10 * log10(max(sinr_lin, 1e-30)), 40), -20);
    end
end


function beams = generate_dft_beams(num_ant, num_beams)
%GENERATE_DFT_BEAMS  DFT codebook for SSB beam scanning.
%   Matches Python ssb_measurement.py::_generate_dft_beams().
    beams = zeros(num_ant, num_beams);
    for b = 1:num_beams
        theta = pi * ((b - 1) / num_beams - 0.5);
        kd = 2 * pi * 0.5 * sin(theta);
        for a = 1:num_ant
            beams(a, b) = exp(1j * kd * (a - 1)) / sqrt(num_ant);
        end
    end
end

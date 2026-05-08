function [w_dl, dl_rank, singular_values] = dl_precoding_svd( ...
    H_ul_est, max_rank, rank_threshold)
%DL_PRECODING_SVD  SVD-based DL precoding with rank selection.
%
%   Matches Python precoding.py::compute_dl_precoding().
%   Uses TDD reciprocity: H_DL = conj(H_UL).
%
%   Inputs:
%       H_ul_est       : [BsAnt, UE_ant, N_RB, no_ss] UL estimated channel
%       max_rank       : maximum allowed rank (default 4)
%       rank_threshold : singular value threshold relative to max (default 0.1)
%
%   Outputs:
%       w_dl           : [BsAnt, dl_rank, N_RB] precoding weights
%       dl_rank        : selected rank (scalar)
%       singular_values: [min(BsAnt,UE_ant), N_RB] SVs per RB

    if nargin < 2 || isempty(max_rank);       max_rank = 4; end
    if nargin < 3 || isempty(rank_threshold); rank_threshold = 0.1; end

    [BsAnt, UE_ant, N_RB, no_ss] = size(H_ul_est);
    max_possible_rank = min(BsAnt, UE_ant);
    max_rank = min(max_rank, max_possible_rank);

    % Time-average the UL channel
    H_avg = mean(H_ul_est, 4);  % [BsAnt, UE_ant, N_RB]

    % TDD reciprocity: H_DL = conj(H_UL) (no transpose in this convention)
    H_dl = conj(H_avg);

    rank_per_rb = zeros(1, N_RB);
    singular_values = zeros(max_possible_rank, N_RB);
    w_all = zeros(BsAnt, max_rank, N_RB);

    for rb = 1:N_RB
        H_rb = H_dl(:, :, rb);  % [BsAnt x UE_ant]
        [U, S, ~] = svd(H_rb, 'econ');
        s_vals = diag(S);
        singular_values(1:numel(s_vals), rb) = s_vals;

        % Rank selection: count SVs above threshold * max SV
        if isempty(s_vals) || s_vals(1) == 0
            n_sig = 1;
        else
            n_sig = sum(s_vals > s_vals(1) * rank_threshold);
        end
        n_sig = min(max(n_sig, 1), max_rank);
        rank_per_rb(rb) = n_sig;

        % Store precoding columns
        n_cols = min(n_sig, size(U, 2));
        w_all(:, 1:n_cols, rb) = U(:, 1:n_cols);
    end

    % Global rank = median across RBs
    dl_rank = max(1, round(median(rank_per_rb)));
    dl_rank = min(dl_rank, max_rank);

    % Extract final precoding matrix
    w_dl = w_all(:, 1:dl_rank, :);  % [BsAnt, dl_rank, N_RB]

    % Per-column unit norm normalization
    for rb = 1:N_RB
        for r = 1:dl_rank
            col = w_dl(:, r, rb);
            n = norm(col);
            if n > 0
                w_dl(:, r, rb) = col / n;
            end
        end
    end
end

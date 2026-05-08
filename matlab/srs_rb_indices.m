function rb_indices = srs_rb_indices(srs_cfg, slot, symbol, total_rb)
%SRS_RB_INDICES  RB indices covered by SRS in a given slot.
%
%   Implements 3GPP 38.211 §6.4.1.4.3 frequency hopping.
%   Matches Python srs.py::srs_rb_indices() exactly.
%
%   Inputs:
%       srs_cfg  : struct with fields C_SRS, B_SRS, b_hop, n_RRC, K_TC,
%                  T_SRS, T_offset, R (default 1)
%       slot     : current slot index (0-based)
%       symbol   : symbol index within slot (0-based)
%       total_rb : total number of RBs in the system bandwidth
%
%   Output:
%       rb_indices : [1 x m_srs_b] int, 0-based RB indices

    row = get_bw_row(srs_cfg.C_SRS);
    B_SRS = srs_cfg.B_SRS;
    b_hop = srs_cfg.b_hop;
    n_RRC = srs_cfg.n_RRC;
    hopping_enabled = b_hop < B_SRS;

    m_srs_b = row.m_srs(B_SRS + 1);  % 1-based index

    if ~hopping_enabled
        n_b_list = zeros(1, B_SRS + 1);
        for b = 0:B_SRS
            N_b = row.n(b + 1);
            n_b_list(b + 1) = mod(n_RRC, max(1, N_b));
        end
    else
        N_symb_slot = 14;
        R = 1;
        if isfield(srs_cfg, 'R'); R = srs_cfg.R; end
        T_SRS = srs_cfg.T_SRS;
        T_offset = 0;
        if isfield(srs_cfg, 'T_offset'); T_offset = srs_cfg.T_offset; end

        if T_SRS > 0
            n_SRS = floor((slot - T_offset) / T_SRS) * R + floor(symbol / N_symb_slot);
        else
            n_SRS = 0;
        end

        n_b_list = zeros(1, B_SRS + 1);
        for b = 0:B_SRS
            N_b = row.n(b + 1);
            if N_b <= 1
                n_b_list(b + 1) = 0;
                continue;
            end
            if b <= b_hop
                n_b_list(b + 1) = mod(n_RRC, N_b);
            else
                prod_N = 1;
                for bp = (b_hop + 1):(b - 1)
                    prod_N = prod_N * row.n(bp + 1);
                end
                if b == b_hop + 1
                    F_b = mod(floor(n_SRS / prod_N), N_b);
                else
                    F_b_prev = 0;
                    F_b = mod(floor(n_SRS / prod_N) + F_b_prev, N_b);
                end
                n_b_list(b + 1) = mod(F_b + mod(n_RRC, N_b), N_b);
            end
        end
    end

    rb_offset = 0;
    for b = 1:B_SRS
        rb_offset = rb_offset + n_b_list(b + 1) * row.m_srs(b + 1);
    end

    rb_start = n_RRC + rb_offset;
    rb_start = max(0, min(rb_start, total_rb - m_srs_b));
    rb_end = min(rb_start + m_srs_b, total_rb);
    rb_indices = rb_start:(rb_end - 1);  % 0-based
end

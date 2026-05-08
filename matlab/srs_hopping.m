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


function cycle_len = srs_hopping_cycle_length(srs_cfg)
%SRS_HOPPING_CYCLE_LENGTH  Number of SRS occasions for one full sweep.
%
%   Matches Python srs.py::srs_hopping_cycle_length().

    hopping_enabled = srs_cfg.b_hop < srs_cfg.B_SRS;
    if ~hopping_enabled
        cycle_len = 1;
        return;
    end
    row = get_bw_row(srs_cfg.C_SRS);
    cycle_len = 1;
    for b = (srs_cfg.b_hop + 1):srs_cfg.B_SRS
        cycle_len = cycle_len * row.n(b + 1);
    end
    cycle_len = max(cycle_len, 1);
end


function rb_indices = srs_accumulated_rb_indices(srs_cfg, current_slot, symbol, total_rb)
%SRS_ACCUMULATED_RB_INDICES  Union of RBs across one hopping cycle.
%
%   Matches Python srs.py::srs_accumulated_rb_indices().

    cycle = srs_hopping_cycle_length(srs_cfg);
    if cycle <= 1
        rb_indices = srs_rb_indices(srs_cfg, current_slot, symbol, total_rb);
        return;
    end

    all_rbs = [];
    T_SRS = srs_cfg.T_SRS;
    for i = 0:(cycle - 1)
        hop_slot = current_slot - (cycle - 1 - i) * T_SRS;
        rbs = srs_rb_indices(srs_cfg, hop_slot, symbol, total_rb);
        all_rbs = union(all_rbs, rbs);
    end
    rb_indices = sort(all_rbs);
end


function row = get_bw_row(C_SRS)
%GET_BW_ROW  Look up 38.211 Table 6.4.1.4.3-1 by C_SRS index.
    table = srs_bw_table();
    for i = 1:size(table, 1)
        if table(i).c_srs == C_SRS
            row = table(i);
            return;
        end
    end
    error('C_SRS=%d not found in SRS_BW_TABLE', C_SRS);
end


function T = srs_bw_table()
%SRS_BW_TABLE  3GPP 38.211 Table 6.4.1.4.3-1 (C_SRS 0..17).
    persistent cached_table;
    if ~isempty(cached_table)
        T = cached_table;
        return;
    end

    data = {
        0,  [4,  4,  4,  4],  [1, 1, 1, 1];
        1,  [8,  4,  4,  4],  [1, 2, 1, 1];
        2,  [12, 4,  4,  4],  [1, 3, 1, 1];
        3,  [16, 4,  4,  4],  [1, 4, 1, 1];
        4,  [16, 8,  4,  4],  [1, 2, 2, 1];
        5,  [20, 4,  4,  4],  [1, 5, 1, 1];
        6,  [24, 4,  4,  4],  [1, 6, 1, 1];
        7,  [24, 12, 4,  4],  [1, 2, 3, 1];
        8,  [28, 4,  4,  4],  [1, 7, 1, 1];
        9,  [32, 16, 8,  4],  [1, 2, 2, 2];
        10, [36, 12, 4,  4],  [1, 3, 3, 1];
        11, [40, 20, 4,  4],  [1, 2, 5, 1];
        12, [48, 16, 8,  4],  [1, 3, 2, 2];
        13, [48, 24, 12, 4],  [1, 2, 2, 3];
        14, [52, 4,  4,  4],  [1, 13, 1, 1];
        15, [56, 28, 4,  4],  [1, 2, 7, 1];
        16, [60, 20, 4,  4],  [1, 3, 5, 1];
        17, [64, 32, 16, 4],  [1, 2, 2, 4];
    };

    N = size(data, 1);
    T = struct('c_srs', cell(N,1), 'm_srs', cell(N,1), 'n', cell(N,1));
    for i = 1:N
        T(i).c_srs = data{i, 1};
        T(i).m_srs = data{i, 2};
        T(i).n     = data{i, 3};
    end
    cached_table = T;
end

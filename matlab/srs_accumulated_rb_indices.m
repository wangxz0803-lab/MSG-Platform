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

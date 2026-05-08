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

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

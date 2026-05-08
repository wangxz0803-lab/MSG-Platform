function [slot_types, symbol_maps] = tdd_slot_config(pattern_name)
%TDD_SLOT_CONFIG  Parse TDD slot pattern into per-slot and per-symbol maps.
%
%   Matches Python tdd_config.py::TDDPattern.
%
%   Inputs:
%       pattern_name : char, e.g. 'DDDSU', 'DDSUU', 'DDDDDDDSUU', 'DSUUD'
%
%   Outputs:
%       slot_types   : [1 x N_slots] char array, 'D'/'U'/'S'
%       symbol_maps  : [N_slots x 14] char array, per-symbol 'D'/'U'/'G'

    if nargin < 1 || isempty(pattern_name)
        pattern_name = 'DDDSU';
    end

    slot_types = upper(char(pattern_name));
    N_slots = numel(slot_types);

    % Special slot symbol counts depend on pattern
    switch upper(pattern_name)
        case {'DDDDDDDSUU', 'DSUUD'}
            dl_sym = 6; guard_sym = 4; ul_sym = 4;
        otherwise
            dl_sym = 10; guard_sym = 2; ul_sym = 2;
    end

    symbol_maps = repmat('D', N_slots, 14);
    for s = 1:N_slots
        switch slot_types(s)
            case 'D'
                symbol_maps(s, :) = repmat('D', 1, 14);
            case 'U'
                symbol_maps(s, :) = repmat('U', 1, 14);
            case 'S'
                sym_row = [repmat('D', 1, dl_sym), ...
                           repmat('G', 1, guard_sym), ...
                           repmat('U', 1, ul_sym)];
                symbol_maps(s, :) = sym_row;
        end
    end
end

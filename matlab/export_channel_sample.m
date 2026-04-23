function export_channel_sample(output_path, sample_data)
%EXPORT_CHANNEL_SAMPLE  Save one ChannelSample-compatible .mat file.
%
%   EXPORT_CHANNEL_SAMPLE(OUTPUT_PATH, SAMPLE_DATA) writes a struct whose
%   fields align with the Python pydantic ChannelSample contract in
%   src/msg_embedding/data/contract.py. The consumer is
%   scipy.io.loadmat, so we use '-v7'. Complex arrays are stored natively.
%
%   Required fields in SAMPLE_DATA:
%       h_serving_true      : complex [T, RB, BS, UE]
%       h_serving_est       : complex [T, RB, BS, UE]
%       h_interferers       : complex [K-1, T, RB, BS, UE] (empty if K=1)
%       interference_signal : complex [T, N_RE_obs, ...] or [] (optional)
%       noise_power_dBm     : double
%       snr_dB              : double
%       sir_dB              : double or []
%       sinr_dB             : double
%       link                : 'UL' | 'DL'
%       channel_est_mode    : 'ideal' | 'ls_linear' | 'ls_mmse'
%       serving_cell_id     : integer (0-based)
%       ue_position         : [1 x 3] double (metres)
%       source              : 'quadriga_multi'
%       sample_id           : UUID4 string
%       created_at          : ISO-8601 UTC string
%       meta                : struct with pci, isd_m, scenario, ...
%
%   Output layout: the MAT file stores a single top-level variable called
%   ``sample`` containing every field listed above. Python consumers can
%   do::
%
%       >>> from scipy.io import loadmat
%       >>> m = loadmat('sample_xxx.mat', simplify_cells=True)
%       >>> d = m['sample']; ChannelSample.from_dict(convert(d))
%
%   Manual check (no MATLAB available):
%       * Confirm the produced file name matches 'sample_<uuid>.mat'.
%       * Load in Python: complex arrays should arrive as np.complex64
%         when they were stored as ``single`` complex in MATLAB.

    required = {'h_serving_true', 'h_serving_est', 'noise_power_dBm', ...
                'snr_dB', 'sinr_dB', 'link', 'channel_est_mode', ...
                'serving_cell_id', 'source', 'sample_id', 'created_at'};
    for i = 1:numel(required)
        f = required{i};
        if ~isfield(sample_data, f)
            error('export_channel_sample:MissingField', ...
                'sample_data is missing required field "%s"', f);
        end
    end

    % Fill optional fields with canonical empties so scipy.io.loadmat sees
    % consistent keys across every file.
    defaults = struct( ...
        'h_interferers',       [], ...
        'interference_signal', [], ...
        'sir_dB',              [], ...
        'ue_position',         [], ...
        'meta',                struct() );
    fn = fieldnames(defaults);
    for i = 1:numel(fn)
        if ~isfield(sample_data, fn{i}) || isempty_native(sample_data.(fn{i}))
            sample_data.(fn{i}) = defaults.(fn{i});
        end
    end

    % Cast complex arrays to single (== np.complex64 after scipy load).
    sample_data.h_serving_true = to_complex_single(sample_data.h_serving_true);
    sample_data.h_serving_est  = to_complex_single(sample_data.h_serving_est);
    if ~isempty(sample_data.h_interferers)
        sample_data.h_interferers = to_complex_single(sample_data.h_interferers);
    end
    if ~isempty(sample_data.interference_signal)
        sample_data.interference_signal = to_complex_single( ...
            sample_data.interference_signal);
    end
    % ue_position: store as double 1x3.
    if ~isempty(sample_data.ue_position)
        sample_data.ue_position = double(sample_data.ue_position(:)).';
    end

    % Guard scalar dB values against NaN (contract bounds [-50, 50]).
    sample_data.snr_dB = clamp_scalar_db(sample_data.snr_dB);
    sample_data.sinr_dB = clamp_scalar_db(sample_data.sinr_dB);
    if ~isempty(sample_data.sir_dB)
        sample_data.sir_dB = clamp_scalar_db(sample_data.sir_dB);
    end

    % Ensure parent dir exists.
    out_dir = fileparts(output_path);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    sample = sample_data; %#ok<NASGU>
    save(output_path, 'sample', '-v7');
end


function tf = isempty_native(v)
    tf = isempty(v) && ~isstruct(v);
end


function y = to_complex_single(x)
    if ~isnumeric(x)
        y = x;
        return;
    end
    % Force complex single regardless of input precision.
    y = complex(single(real(x)), single(imag(x)));
end


function db = clamp_scalar_db(db)
    if isempty(db) || ~isfinite(db)
        db = 0;
        return;
    end
    db = max(min(double(db), 50), -50);
end

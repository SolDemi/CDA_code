function [dataMat, times, usedFiles] = extract_decoding_timeseries(resultDir, cfg)
% extract_decoding_timeseries
%
% Extract subject-by-time matrices from saved LDA/SVM result structs.
% This is only a loader/helper. The statistics should be done by
% cluster_perm_1d_timeseries or plot_group_timeseries_perm.
%
% USAGE
%   cfg = struct();
%   cfg.metric = 'AUC';        % or 'predictAcc'
%   cfg.useDiagonal = true;    % true for time-generalization matrices
%   [groupAUC, times, files] = extract_decoding_timeseries('/path/to/CDA', cfg);
%
% ASSUMED FILE FORMAT
%   Each .mat file contains either:
%     - a struct with field cfg.metric, e.g. CDA.AUC or result.AUC
%     - or cfg.resultVarName points to the desired struct.
%
% CFG FIELDS
%   cfg.metric        : field to extract, default = 'AUC'
%   cfg.useDiagonal   : if metric is nTime x nTime, return diag(metric), default = true
%   cfg.filePattern   : default = '*.mat'
%   cfg.resultVarName : optional exact variable name inside .mat file
%   cfg.times         : fallback time vector when the result struct has no .times field

if nargin < 2 || isempty(cfg)
    cfg = struct();
end
cfg = fill_default_cfg(cfg);

files = dir(fullfile(resultDir, cfg.filePattern));
if isempty(files)
    error('No files matched %s in %s.', cfg.filePattern, resultDir);
end

dataRows = {};
times = [];
usedFiles = {};

for fi = 1:numel(files)
    fpath = fullfile(files(fi).folder, files(fi).name);
    S = load(fpath);
    R = pick_result_struct(S, cfg);

    if isempty(R)
        warning('Skipping %s: no result struct with field %s.', files(fi).name, cfg.metric);
        continue;
    end

    metricVal = R.(cfg.metric);
    if cfg.useDiagonal && ~isvector(metricVal)
        row = diag(metricVal)';
    elseif isvector(metricVal)
        row = metricVal(:)';
    else
        error('Metric %s in %s is a matrix. Set cfg.useDiagonal=true or provide a vector metric.', ...
            cfg.metric, files(fi).name);
    end

    if isfield(R, 'times') && ~isempty(R.times)
        tThis = R.times(:)';
    elseif ~isempty(cfg.times)
        tThis = cfg.times(:)';
    else
        tThis = 1:numel(row);
    end

    if isempty(times)
        times = tThis;
    elseif numel(tThis) ~= numel(times) || any(abs(tThis - times) > 1e-9)
        error('Time vector mismatch in file %s.', files(fi).name);
    end

    dataRows{end+1,1} = row; %#ok<AGROW>
    usedFiles{end+1,1} = fpath; %#ok<AGROW>
end

if isempty(dataRows)
    error('No usable result files found in %s.', resultDir);
end

dataMat = vertcat(dataRows{:});
end

%% ========================================================================
function cfg = fill_default_cfg(cfg)
if ~isfield(cfg, 'metric'),        cfg.metric = 'AUC'; end
if ~isfield(cfg, 'useDiagonal'),   cfg.useDiagonal = true; end
if ~isfield(cfg, 'filePattern'),   cfg.filePattern = '*.mat'; end
if ~isfield(cfg, 'resultVarName'), cfg.resultVarName = ''; end
if ~isfield(cfg, 'times'),         cfg.times = []; end
end

%% ========================================================================
function R = pick_result_struct(S, cfg)
R = [];

if ~isempty(cfg.resultVarName) && isfield(S, cfg.resultVarName)
    if isstruct(S.(cfg.resultVarName)) && isfield(S.(cfg.resultVarName), cfg.metric)
        R = S.(cfg.resultVarName);
        return;
    end
end

fn = fieldnames(S);
for i = 1:numel(fn)
    if isstruct(S.(fn{i})) && isfield(S.(fn{i}), cfg.metric)
        R = S.(fn{i});
        return;
    end
end
end

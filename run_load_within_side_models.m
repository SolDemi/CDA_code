function Results = run_load_within_side_models(cda, alpha, cfg, decoderFcn)
% run_load_within_side_models
% Decode low vs high load separately within attended-left and attended-right
% trials, then average the two side-specific decoding results.
%
% Required input data format
%   cda.trial / alpha.trial must contain side-specific lateralized fields,
%   preferably:
%       diff_L_2, diff_L_6, diff_R_2, diff_R_6
%   The function also supports condition-specific fields such as:
%       diff_L_C2, diff_L_S2, diff_R_C2, diff_R_S2
%       diff_L_C6, diff_L_S6, diff_R_C6, diff_R_S6
%
% Output
%   Results.CDA, Results.Alpha, Results.NoPCA, Results.PCA
% Each result struct contains averaged fields such as Acc, AccShuffle, AUC,
% and AUCShuffle, so it can be read directly by stat_plot.m.

if nargin < 4 || isempty(decoderFcn)
    error('A decoder function handle, e.g. @LDA_function_singleSubj, is required.');
end
if nargin < 3 || isempty(cfg)
    cfg = struct();
end
if ~isfield(cfg, 'analysisWindow'), cfg.analysisWindow = [-inf inf]; end

times = cda.time(:)';
timeIdx = times >= cfg.analysisWindow(1) & times <= cfg.analysisWindow(2);
if ~any(timeIdx)
    error('cfg.analysisWindow does not overlap with cda.time.');
end
times = times(timeIdx);

cdaSide   = extract_side_load_arrays(cda.trial,   'diff', timeIdx);
alphaSide = extract_side_load_arrays(alpha.trial, 'diff', timeIdx);

Results = struct();

cfgModel = cfg;
cfgModel.doPCA = false;
Results.CDA = decode_one_model(cdaSide, times, cfgModel, decoderFcn, 'CDA');

cfgModel = cfg;
cfgModel.doPCA = false;
Results.Alpha = decode_one_model(alphaSide, times, cfgModel, decoderFcn, 'Alpha');

combinedSide = concatenate_feature_sets(cdaSide, alphaSide);

cfgModel = cfg;
cfgModel.doPCA = false;
Results.NoPCA = decode_one_model(combinedSide, times, cfgModel, decoderFcn, 'NoPCA');

cfgModel = cfg;
cfgModel.doPCA = true;
Results.PCA = decode_one_model(combinedSide, times, cfgModel, decoderFcn, 'PCA');

end

%% ========================================================================
function sideData = extract_side_load_arrays(T, prefix, timeIdx)

sideData = struct();

sideData.L.low  = pick_load_array(T, prefix, 'L', 2, timeIdx);
sideData.L.high = pick_load_array(T, prefix, 'L', 6, timeIdx);
sideData.R.low  = pick_load_array(T, prefix, 'R', 2, timeIdx);
sideData.R.high = pick_load_array(T, prefix, 'R', 6, timeIdx);

end

%% ========================================================================
function X = pick_load_array(T, prefix, side, loadVal, timeIdx)

loadStr = num2str(loadVal);
X = first_existing_field(T, direct_candidates(prefix, side, loadStr));

if isempty(X)
    if loadVal == 2
        condNames = {'C2', 'S2'};
    elseif loadVal == 6
        condNames = {'C6', 'S6'};
    else
        error('Unsupported load value: %g.', loadVal);
    end

    parts = cell(1, numel(condNames));
    for ci = 1:numel(condNames)
        parts{ci} = first_existing_field(T, condition_candidates(prefix, side, condNames{ci}));
    end

    if all(cellfun(@(x) ~isempty(x), parts))
        X = cat(1, parts{:});
    end
end

if isempty(X)
    fn = fieldnames(T);
    preview = strjoin(fn(1:min(numel(fn), 30)), ', ');
    error(['Missing side-specific %s field for side %s, load %s.\n' ...
           'Expected fields like %s_L_%s / %s_R_%s, or condition fields like %s_L_C%s and %s_L_S%s.\n' ...
           'Current trial fields begin with: %s\n' ...
           'This cannot be reconstructed from collapsed diff_2/diff_6 alone; rerun cda_alpha.m after adding side-specific fields.'], ...
           prefix, side, loadStr, prefix, loadStr, prefix, loadStr, prefix, loadStr, prefix, loadStr, preview);
end

if ndims(X) ~= 3
    error('Field for side %s, load %s must be trials x channels x time.', side, loadStr);
end
if size(X, 3) ~= numel(timeIdx)
    error('The time dimension of the extracted field does not match the time vector.');
end

X = X(:,:,timeIdx);

end

%% ========================================================================
function names = direct_candidates(prefix, side, loadStr)

sideLower = lower(side);
if strcmpi(side, 'L')
    sideLong = 'left';
    sideLongCap = 'Left';
else
    sideLong = 'right';
    sideLongCap = 'Right';
end

names = {
    sprintf('%s_%s_%s', prefix, side, loadStr)
    sprintf('%s_%s%s', prefix, side, loadStr)
    sprintf('%s%s_%s', prefix, side, loadStr)
    sprintf('%s%s%s', prefix, side, loadStr)
    sprintf('%s_%s_%s', prefix, sideLower, loadStr)
    sprintf('%s_%s%s', prefix, sideLower, loadStr)
    sprintf('%s_%s_%s', prefix, sideLong, loadStr)
    sprintf('%s_%s%s', prefix, sideLong, loadStr)
    sprintf('%s_%s_%s', prefix, sideLongCap, loadStr)
    sprintf('%s_%s%s', prefix, sideLongCap, loadStr)
    sprintf('%s_%s_%s', side, prefix, loadStr)
    sprintf('%s_%s%s', side, prefix, loadStr)
    sprintf('%s_%s_%s', sideLower, prefix, loadStr)
    sprintf('%s_%s%s', sideLower, prefix, loadStr)
    sprintf('%s_%s_%s', sideLong, prefix, loadStr)
    sprintf('%s_%s%s', sideLong, prefix, loadStr)
    sprintf('%s%d_%s', side, str2double(loadStr), prefix)
    sprintf('%s%s_%s', side, loadStr, prefix)
    };

end

%% ========================================================================
function names = condition_candidates(prefix, side, condName)

sideLower = lower(side);
if strcmpi(side, 'L')
    sideLong = 'left';
    sideLongCap = 'Left';
else
    sideLong = 'right';
    sideLongCap = 'Right';
end

names = {
    sprintf('%s_%s_%s', prefix, side, condName)
    sprintf('%s_%s%s', prefix, side, condName)
    sprintf('%s%s_%s', prefix, side, condName)
    sprintf('%s%s%s', prefix, side, condName)
    sprintf('%s_%s_%s', prefix, sideLower, condName)
    sprintf('%s_%s%s', prefix, sideLower, condName)
    sprintf('%s_%s_%s', prefix, sideLong, condName)
    sprintf('%s_%s%s', prefix, sideLong, condName)
    sprintf('%s_%s_%s', prefix, sideLongCap, condName)
    sprintf('%s_%s%s', prefix, sideLongCap, condName)
    sprintf('%s_%s_%s', side, prefix, condName)
    sprintf('%s_%s%s', side, prefix, condName)
    sprintf('%s_%s_%s', sideLower, prefix, condName)
    sprintf('%s_%s%s', sideLower, prefix, condName)
    sprintf('%s_%s_%s', sideLong, prefix, condName)
    sprintf('%s_%s%s', sideLong, prefix, condName)
    sprintf('%s_%s_%s', side, condName, prefix)
    sprintf('%s_%s_%s', sideLower, condName, prefix)
    sprintf('%s_%s_%s', sideLong, condName, prefix)
    };

end

%% ========================================================================
function X = first_existing_field(T, candidates)

X = [];
for i = 1:numel(candidates)
    nm = candidates{i};
    if isfield(T, nm)
        X = T.(nm);
        return;
    end
end

end

%% ========================================================================
function combined = concatenate_feature_sets(A, B)

combined = struct();
sideNames = {'L', 'R'};
loadNames = {'low', 'high'};

for si = 1:numel(sideNames)
    s = sideNames{si};
    for li = 1:numel(loadNames)
        l = loadNames{li};

        XA = A.(s).(l);
        XB = B.(s).(l);

        if size(XA,1) ~= size(XB,1) || size(XA,3) ~= size(XB,3)
            error('CDA and alpha trial/time counts do not match for side %s, load %s.', s, l);
        end

        combined.(s).(l) = cat(2, XA, XB);
    end
end

end

%% ========================================================================
function result = decode_one_model(sideData, times, cfg, decoderFcn, modelName)

[dataL, labelsL] = make_binary_decoding_input(sideData.L.low, sideData.L.high);
[dataR, labelsR] = make_binary_decoding_input(sideData.R.low, sideData.R.high);

resL = decoderFcn(dataL, labelsL, times, cfg);
resR = decoderFcn(dataR, labelsR, times, cfg);

leftCounts  = [size(sideData.L.low,1), size(sideData.L.high,1)];
rightCounts = [size(sideData.R.low,1), size(sideData.R.high,1)];

result = average_result_structs(resL, resR);
result.modelName = modelName;
result.withinSide = struct();
result.withinSide.description = 'Load decoding was run separately within attended-left and attended-right trials, then averaged across sides.';
result.withinSide.leftCountsLowHigh = leftCounts;
result.withinSide.rightCountsLowHigh = rightCounts;
result.withinSide.averageMode = 'unweighted mean of left-side and right-side decoding results';
result.side = struct();
result.side.Left  = keep_plot_relevant_fields(resL);
result.side.Right = keep_plot_relevant_fields(resR);

end

%% ========================================================================
function [data, labels] = make_binary_decoding_input(lowData, highData)

data = cat(1, lowData, highData);           % trials x channels x time
data = permute(data, [2 3 1]);              % channels x time x trials
labels = [ones(size(lowData,1),1); 2*ones(size(highData,1),1)];

end

%% ========================================================================
function result = average_result_structs(A, B)

result = A;
fn = fieldnames(A);

for i = 1:numel(fn)
    f = fn{i};
    if isfield(B, f) && isnumeric(A.(f)) && isnumeric(B.(f)) && isequal(size(A.(f)), size(B.(f)))
        dim = ndims(A.(f)) + 1;
        result.(f) = mean(cat(dim, A.(f), B.(f)), dim, 'omitnan');
    end
end

if isfield(result, 'cfg')
    result.cfg.withinSideAverage = true;
end

end

%% ========================================================================
function S = keep_plot_relevant_fields(R)

keep = {'Acc', 'AUC', 'AccShuffle', 'AUCShuffle', ...
        'AccMinusShuffle', 'AUCMinusShuffle', 'AccTrain', 'times'};
S = struct();

for i = 1:numel(keep)
    f = keep{i};
    if isfield(R, f)
        S.(f) = R.(f);
    end
end

end

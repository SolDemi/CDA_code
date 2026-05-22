function Results = run_load_within_side_models(cda, alpha, cfg, decoderFcn)
% run_load_within_side_models
% Decode low vs high load separately within attended-left and attended-right
% trials, then average the two side-specific decoding results.
%
% Minimal-storage compatible version.
% cda_alpha.m only stores absolute posterior left/right hemisphere data:
%   left_L_2, right_L_2, left_R_2, right_R_2, and same for load 6.
%
% This function constructs features on demand:
%   CDA / Alpha      : contra - ipsi within each attended side
%   GlobalAlpha      : [left posterior channels, right posterior channels]
%   GlobalAlphaMean  : mean over global posterior alpha channels
%   NoPCA / PCA      : [CDA features, lateralized alpha features]

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

cdaSide         = construct_lateralized_side_load_features(cda.trial,   timeIdx);
alphaSide       = construct_lateralized_side_load_features(alpha.trial, timeIdx);
globalAlphaSide = construct_global_alpha_side_load_features(alpha.trial, timeIdx, false);
globalMeanSide  = construct_global_alpha_side_load_features(alpha.trial, timeIdx, true);

Results = struct();

cfgModel = cfg;
cfgModel.doPCA = false;
Results.CDA = decode_one_model(cdaSide, times, cfgModel, decoderFcn, 'CDA');

cfgModel = cfg;
cfgModel.doPCA = false;
Results.Alpha = decode_one_model(alphaSide, times, cfgModel, decoderFcn, 'Alpha');

cfgModel = cfg;
cfgModel.doPCA = false;
Results.GlobalAlpha = decode_one_model(globalAlphaSide, times, cfgModel, decoderFcn, 'GlobalAlpha');

cfgModel = cfg;
cfgModel.doPCA = false;
Results.GlobalAlphaMean = decode_one_model(globalMeanSide, times, cfgModel, decoderFcn, 'GlobalAlphaMean');

combinedSide = concatenate_feature_sets(cdaSide, alphaSide);

cfgModel = cfg;
cfgModel.doPCA = false;
Results.NoPCA = decode_one_model(combinedSide, times, cfgModel, decoderFcn, 'NoPCA');

cfgModel = cfg;
cfgModel.doPCA = true;
Results.PCA = decode_one_model(combinedSide, times, cfgModel, decoderFcn, 'PCA');

end

%% ========================================================================
function sideData = construct_lateralized_side_load_features(T, timeIdx)
% Return contra-minus-ipsi features separately for attended-left and
% attended-right trials.
%
% For attended-left trials:  contra = right posterior, ipsi = left posterior.
% For attended-right trials: contra = left posterior,  ipsi = right posterior.

sideData = struct();
sideData.L.low  = get_right_left_diff(T, 'L', 2, timeIdx);
sideData.L.high = get_right_left_diff(T, 'L', 6, timeIdx);
sideData.R.low  = get_left_right_diff(T, 'R', 2, timeIdx);
sideData.R.high = get_left_right_diff(T, 'R', 6, timeIdx);

end

%% ========================================================================
function sideData = construct_global_alpha_side_load_features(T, timeIdx, doMean)
% Return global posterior alpha features separately for attended-left and
% attended-right trials. If doMean=false, use all left+right posterior
% channels. If doMean=true, average channels to one feature per trial/time.

sideData = struct();
sideData.L.low  = get_global_features(T, 'L', 2, timeIdx, doMean);
sideData.L.high = get_global_features(T, 'L', 6, timeIdx, doMean);
sideData.R.low  = get_global_features(T, 'R', 2, timeIdx, doMean);
sideData.R.high = get_global_features(T, 'R', 6, timeIdx, doMean);

end

%% ========================================================================
function X = get_right_left_diff(T, attendedSide, loadVal, timeIdx)

[leftX, rightX] = get_abs_left_right(T, attendedSide, loadVal, timeIdx);
X = rightX - leftX;

end

%% ========================================================================
function X = get_left_right_diff(T, attendedSide, loadVal, timeIdx)

[leftX, rightX] = get_abs_left_right(T, attendedSide, loadVal, timeIdx);
X = leftX - rightX;

end

%% ========================================================================
function X = get_global_features(T, attendedSide, loadVal, timeIdx, doMean)

[leftX, rightX] = get_abs_left_right(T, attendedSide, loadVal, timeIdx);
X = cat(2, leftX, rightX);

if doMean
    X = mean(X, 2, 'omitnan');
end

end

%% ========================================================================
function [leftX, rightX] = get_abs_left_right(T, attendedSide, loadVal, timeIdx)

loadStr = num2str(loadVal);
leftName  = sprintf('left_%s_%s', attendedSide, loadStr);
rightName = sprintf('right_%s_%s', attendedSide, loadStr);

if isfield(T, leftName) && isfield(T, rightName)
    leftX  = T.(leftName);
    rightX = T.(rightName);
else
    [leftX, rightX] = get_abs_left_right_from_legacy_fields(T, attendedSide, loadVal);
end

check_abs_pair(leftX, rightX, attendedSide, loadStr);
leftX  = leftX(:,:,timeIdx);
rightX = rightX(:,:,timeIdx);

end

%% ========================================================================
function [leftX, rightX] = get_abs_left_right_from_legacy_fields(T, attendedSide, loadVal)
% Backward-compatible fallback for older files that still contain
% contra/ipsi side-specific fields.

loadStr = num2str(loadVal);
contraName = sprintf('contra_%s_%s', attendedSide, loadStr);
ipsiName   = sprintf('ipsi_%s_%s',   attendedSide, loadStr);

if ~(isfield(T, contraName) && isfield(T, ipsiName))
    fn = fieldnames(T);
    preview = strjoin(fn(1:min(numel(fn), 30)), ', ');
    error(['Missing minimal absolute fields left_%s_%s/right_%s_%s.\n' ...
           'Also could not find legacy fields contra_%s_%s/ipsi_%s_%s.\n' ...
           'Current trial fields begin with: %s\n' ...
           'Please rerun cda_alpha.m using the minimal-storage version.'], ...
           attendedSide, loadStr, attendedSide, loadStr, ...
           attendedSide, loadStr, attendedSide, loadStr, preview);
end

contra = T.(contraName);
ipsi   = T.(ipsiName);

if strcmpi(attendedSide, 'L')
    % attended-left: contra = right, ipsi = left
    leftX  = ipsi;
    rightX = contra;
else
    % attended-right: contra = left, ipsi = right
    leftX  = contra;
    rightX = ipsi;
end

end

%% ========================================================================
function check_abs_pair(leftX, rightX, attendedSide, loadStr)

if ndims(leftX) ~= 3 || ndims(rightX) ~= 3
    error('Fields for side %s, load %s must be trials x channels x time.', attendedSide, loadStr);
end
if ~isequal(size(leftX), size(rightX))
    error('Left/right posterior fields do not match for side %s, load %s.', attendedSide, loadStr);
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
result.withinSide.featureConstruction = feature_description(modelName);
result.side = struct();
result.side.Left  = keep_plot_relevant_fields(resL);
result.side.Right = keep_plot_relevant_fields(resR);

end

%% ========================================================================
function txt = feature_description(modelName)

switch lower(modelName)
    case {'cda','alpha'}
        txt = 'Constructed on demand as contra-minus-ipsi within attended side.';
    case 'globalalpha'
        txt = 'Constructed on demand as absolute posterior alpha [left channels, right channels].';
    case 'globalalphamean'
        txt = 'Constructed on demand as the trial-wise mean over absolute posterior alpha channels.';
    case {'nopca','pca'}
        txt = 'Constructed on demand by concatenating CDA contra-minus-ipsi and alpha contra-minus-ipsi features.';
    otherwise
        txt = '';
end

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

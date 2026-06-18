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

sideNames = {'L', 'R'};
loadNames = {'low', 'high'};
loadVals = [2 6];

%% Data selection and feature construction
cdaSide = struct();
alphaSide = struct();
globalAlphaSide = struct();
globalMeanSide = struct();

for sourceIdx = 1:2
    if sourceIdx == 1
        T = cda.trial;
        targetName = 'cda';
    else
        T = alpha.trial;
        targetName = 'alpha';
    end

    sideData = struct();
    for si = 1:numel(sideNames)
        attendedSide = sideNames{si};
        for li = 1:numel(loadNames)
            loadVal = loadVals(li);
            loadStr = num2str(loadVal);
            leftName = sprintf('left_%s_%s', attendedSide, loadStr);
            rightName = sprintf('right_%s_%s', attendedSide, loadStr);

            if isfield(T, leftName) && isfield(T, rightName)
                leftX = T.(leftName);
                rightX = T.(rightName);
            else
                contraName = sprintf('contra_%s_%s', attendedSide, loadStr);
                ipsiName = sprintf('ipsi_%s_%s', attendedSide, loadStr);
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
                ipsi = T.(ipsiName);
                if strcmpi(attendedSide, 'L')
                    leftX = ipsi;
                    rightX = contra;
                else
                    leftX = contra;
                    rightX = ipsi;
                end
            end

            if ndims(leftX) ~= 3 || ndims(rightX) ~= 3
                error('Fields for side %s, load %s must be trials x channels x time.', attendedSide, loadStr);
            end
            if ~isequal(size(leftX), size(rightX))
                error('Left/right posterior fields do not match for side %s, load %s.', attendedSide, loadStr);
            end

            leftX = leftX(:,:,timeIdx);
            rightX = rightX(:,:,timeIdx);
            if strcmpi(attendedSide, 'L')
                sideData.(attendedSide).(loadNames{li}) = rightX - leftX;
            else
                sideData.(attendedSide).(loadNames{li}) = leftX - rightX;
            end

            if sourceIdx == 2
                globalAlphaSide.(attendedSide).(loadNames{li}) = cat(2, leftX, rightX);
                globalMeanSide.(attendedSide).(loadNames{li}) = mean(cat(2, leftX, rightX), 2, 'omitnan');
            end
        end
    end

    if strcmp(targetName, 'cda')
        cdaSide = sideData;
    else
        alphaSide = sideData;
    end
end

combinedSide = struct();
for si = 1:numel(sideNames)
    sideName = sideNames{si};
    for li = 1:numel(loadNames)
        loadName = loadNames{li};
        XA = cdaSide.(sideName).(loadName);
        XB = alphaSide.(sideName).(loadName);

        if size(XA,1) ~= size(XB,1) || size(XA,3) ~= size(XB,3)
            error('CDA and alpha trial/time counts do not match for side %s, load %s.', sideName, loadName);
        end

        combinedSide.(sideName).(loadName) = cat(2, XA, XB);
    end
end

%% Trial balancing, model training/testing, side averaging, and result saving fields
Results = struct();
modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};
modelDoPCA = [false false false false false true];
featureDescriptions = { ...
    'Constructed on demand as contra-minus-ipsi within attended side.', ...
    'Constructed on demand as contra-minus-ipsi within attended side.', ...
    'Constructed on demand as absolute posterior alpha [left channels, right channels].', ...
    'Constructed on demand as the trial-wise mean over absolute posterior alpha channels.', ...
    'Constructed on demand by concatenating CDA contra-minus-ipsi and alpha contra-minus-ipsi features.', ...
    'Constructed on demand by concatenating CDA contra-minus-ipsi and alpha contra-minus-ipsi features.'};

for mi = 1:numel(modelNames)
    modelName = modelNames{mi};
    cfgModel = cfg;
    cfgModel.doPCA = modelDoPCA(mi);

    switch modelName
        case 'CDA'
            sideData = cdaSide;
        case 'Alpha'
            sideData = alphaSide;
        case 'GlobalAlpha'
            sideData = globalAlphaSide;
        case 'GlobalAlphaMean'
            sideData = globalMeanSide;
        case {'NoPCA', 'PCA'}
            sideData = combinedSide;
        otherwise
            error('Unsupported model name: %s.', modelName);
    end

    dataL = cat(1, sideData.L.low, sideData.L.high);
    dataL = permute(dataL, [2 3 1]);
    labelsL = [ones(size(sideData.L.low,1),1); 2*ones(size(sideData.L.high,1),1)];

    dataR = cat(1, sideData.R.low, sideData.R.high);
    dataR = permute(dataR, [2 3 1]);
    labelsR = [ones(size(sideData.R.low,1),1); 2*ones(size(sideData.R.high,1),1)];

    resL = decoderFcn(dataL, labelsL, times, cfgModel);
    resR = decoderFcn(dataR, labelsR, times, cfgModel);

    leftCounts = [size(sideData.L.low,1), size(sideData.L.high,1)];
    rightCounts = [size(sideData.R.low,1), size(sideData.R.high,1)];

    result = resL;
    fn = fieldnames(resL);
    for fi = 1:numel(fn)
        f = fn{fi};
        if isfield(resR, f) && isnumeric(resL.(f)) && isnumeric(resR.(f)) && isequal(size(resL.(f)), size(resR.(f)))
            dim = ndims(resL.(f)) + 1;
            result.(f) = mean(cat(dim, resL.(f), resR.(f)), dim, 'omitnan');
        end
    end

    if isfield(result, 'cfg')
        result.cfg.withinSideAverage = true;
    end

    keep = {'Acc', 'AUC', 'AccShuffle', 'AUCShuffle', ...
            'AccMinusShuffle', 'AUCMinusShuffle', 'AccTrain', 'times'};
    sideLeft = struct();
    sideRight = struct();
    for ki = 1:numel(keep)
        f = keep{ki};
        if isfield(resL, f)
            sideLeft.(f) = resL.(f);
        end
        if isfield(resR, f)
            sideRight.(f) = resR.(f);
        end
    end

    result.modelName = modelName;
    result.withinSide = struct();
    result.withinSide.description = 'Load decoding was run separately within attended-left and attended-right trials, then averaged across sides.';
    result.withinSide.leftCountsLowHigh = leftCounts;
    result.withinSide.rightCountsLowHigh = rightCounts;
    result.withinSide.averageMode = 'unweighted mean of left-side and right-side decoding results';
    result.withinSide.featureConstruction = featureDescriptions{mi};
    result.side = struct();
    result.side.Left = sideLeft;
    result.side.Right = sideRight;

    Results.(modelName) = result;
end

end

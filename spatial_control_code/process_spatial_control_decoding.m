%% process_spatial_control_decoding.m
% Follow-up analyses for CDA vs Alpha project:
% 1) side decoding positive control
% 2) within-side load decoding
% 3) side-balanced load decoding
% 4) cross-side load generalization
%
% Put this file in CDA_code/code or any folder where your existing helper
% functions are on the MATLAB path:
% SVM_function_singleSubj.m
% SVM_crossSide_singleSubj.m
% calculate_high_gamma_power.m
% balance_trials_by_label.m
% func_make_superTrials.m

clear; clc;

maindir   = [erase(pwd, 'code'), 'data0'];
datadir   = fullfile(maindir, 'data');
outputdir = fullfile(maindir, 'decoding_SVM_spatialControl');

%% folders
sideFeatures = {'VoltageRawLR', 'AlphaRawLR', 'VoltageLminusR', 'AlphaLminusR', 'GlobalAlphaMean'};
loadFeatures = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};
analysisNames = {'sideDecoding', 'loadWithinSide', 'loadSideBalanced', 'loadCrossSide'};

for ai = 1:numel(analysisNames)
    if strcmp(analysisNames{ai}, 'sideDecoding')
        theseFeatures = sideFeatures;
    else
        theseFeatures = loadFeatures;
    end
    for fi = 1:numel(theseFeatures)
        tmpDir = fullfile(outputdir, analysisNames{ai}, theseFeatures{fi});
        if ~isfolder(tmpDir), mkdir(tmpDir); end
    end
end

%% decoding config: keep close to process_data0.m
cfg = struct();
cfg.cvType = 'holdout';
cfg.trainRatio = 2/3;
cfg.nFolds = 3;
cfg.superTrial = 10;
cfg.nIter = 100;
cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';
cfg.doTimeGeneralization = 1;
cfg.doPCA = false;
cfg.nPCs = 5;
cfg.discrimType = 'Linear';
cfg.standardize = 1;
cfg.doShuffle = true;
cfg.balanceTrials = true;
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];
cfg.useAUC = 1;
cfg.useParallel = true;
cfg.verbose = 0;
cfg.randomSeed = [];

%% channel / condition config
L_labels = {'O1','OL','P3','PO3','T5'};
R_labels = {'O2','OR','P4','PO4','T6'};
global_labels = [L_labels, R_labels];

% In the original process_data0.m:
% condition 1 vs 3 uses R-L, so the attended/remembered side is left
% condition 4 vs 6 uses L-R, so the attended/remembered side is right
sideCfg(1).name = 'attendLeft';
sideCfg(1).channel_contra = R_labels;
sideCfg(1).channel_ipsi   = L_labels;
sideCfg(1).condLow  = 1;
sideCfg(1).condMid  = 2;
sideCfg(1).condHigh = 3;

sideCfg(2).name = 'attendRight';
sideCfg(2).channel_contra = L_labels;
sideCfg(2).channel_ipsi   = R_labels;
sideCfg(2).condLow  = 4;
sideCfg(2).condMid  = 5;
sideCfg(2).condHigh = 6;

%% alpha config
baselinewindow = [-1400, -1100];
frep = [8, 12];

%% decoding
files = dir(fullfile(datadir, '*.mat'));
for s = 1:numel(files)
    file = files(s).name;
    fprintf('\nNow Processing: %s\n', file);

    tmp = load(fullfile(files(s).folder, file));
    eeg0 = tmp.eeg.baselined;       % condition x trial x channel x time
    time = tmp.eeg.time;
    artifactInd = tmp.eeg.arf.artifactInd;
    chanLabels = tmp.eeg.chanLabels;
    srate = tmp.eeg.settings.srate;

    sideDatCell = cell(1, 2);
    for sidei = 1:2
        sideOne = struct();
        sideOne.name = sideCfg(sidei).name;
        sideDatCell{sidei} = build_side_features(sideOne, eeg0, artifactInd, chanLabels, ...
            sideCfg(sidei), L_labels, R_labels, global_labels, srate, time, baselinewindow, frep);
    end
    sideDat = [sideDatCell{:}];

    %% ============================================================
    % 1) Side decoding positive control
    %    Use fixed anatomical features only. Do NOT use contra-minus-ipsi.
    % ============================================================
    for fi = 1:numel(sideFeatures)
        featName = sideFeatures{fi};
        [dataSide, labelsSide, loadFactor] = make_side_decoding_data(sideDat, featName);

        cfgSide = cfg;
        cfgSide.doPCA = false;
        cfgSide.balanceFactors = loadFactor(:);  % side decoding is load-balanced

        Result = run_svm_if_enough(dataSide, labelsSide, time, cfgSide);
        if isempty(Result), continue; end

        Result.analysis = 'sideDecoding';
        Result.feature = featName;
        Result.labelMeaning = {'attendLeft', 'attendRight'};
        Result.loadFactor = loadFactor;
        Result.baselinewindow = baselinewindow;
        Result.frep = frep;
        Result.channelLabels = channels_for_feature(featName, L_labels, R_labels, global_labels);
        save_result(outputdir, 'sideDecoding', featName, file, Result);
    end

    %% ============================================================
    % 2) Within-side load decoding
    %    Decode low vs high separately within each side, then average.
    % ============================================================
    for fi = 1:numel(loadFeatures)
        featName = loadFeatures{fi};

        cfgLoad = cfg;
        cfgLoad.balanceFactors = [];
        cfgLoad.doPCA = strcmpi(featName, 'PCA');

        ResultSide = cell(1, 2);
        for sidei = 1:2
            [dataLoad, labelsLoad] = make_load_data_one_side(sideDat(sidei), featName);
            ResultSide{sidei} = run_svm_if_enough(dataLoad, labelsLoad, time, cfgLoad);
            if ~isempty(ResultSide{sidei})
                ResultSide{sidei}.analysis = 'loadWithinSide_singleSide';
                ResultSide{sidei}.feature = featName;
                ResultSide{sidei}.sideName = sideDat(sidei).name;
            end
        end

        Result = average_two_results(ResultSide{1}, ResultSide{2});
        if isempty(Result), continue; end

        Result.analysis = 'loadWithinSide';
        Result.feature = featName;
        Result.labelMeaning = {'lowLoad', 'highLoad'};
        Result.sideResults = ResultSide;
        Result.baselinewindow = baselinewindow;
        Result.frep = frep;
        save_result(outputdir, 'loadWithinSide', featName, file, Result);
    end

    %% ============================================================
    % 3) Side-balanced load decoding
    %    Merge two sides, but balance each load x side cell in each iteration.
    % ============================================================
    for fi = 1:numel(loadFeatures)
        featName = loadFeatures{fi};
        [dataLoad, labelsLoad, sideFactor] = make_side_balanced_load_data(sideDat, featName);

        cfgBal = cfg;
        cfgBal.balanceFactors = sideFactor(:);
        cfgBal.doPCA = strcmpi(featName, 'PCA');

        Result = run_svm_if_enough(dataLoad, labelsLoad, time, cfgBal);
        if isempty(Result), continue; end

        Result.analysis = 'loadSideBalanced';
        Result.feature = featName;
        Result.labelMeaning = {'lowLoad', 'highLoad'};
        Result.sideFactor = sideFactor;
        Result.sideMeaning = {'attendLeft', 'attendRight'};
        Result.baselinewindow = baselinewindow;
        Result.frep = frep;
        save_result(outputdir, 'loadSideBalanced', featName, file, Result);
    end

    %% ============================================================
    % 4) Cross-side load generalization
    %    Train low vs high on one side, test on the other side, average directions.
    % ============================================================
    for fi = 1:numel(loadFeatures)
        featName = loadFeatures{fi};

        cfgCross = cfg;
        cfgCross.doPCA = strcmpi(featName, 'PCA');
        cfgCross.useParallel = false;   % cross-side helper is already iteration-level simple loop

        [train12, yTrain12] = make_load_data_one_side(sideDat(1), featName);
        [test12,  yTest12]  = make_load_data_one_side(sideDat(2), featName);
        [train21, yTrain21] = make_load_data_one_side(sideDat(2), featName);
        [test21,  yTest21]  = make_load_data_one_side(sideDat(1), featName);

        Result12 = run_cross_if_enough(train12, yTrain12, test12, yTest12, time, cfgCross);
        Result21 = run_cross_if_enough(train21, yTrain21, test21, yTest21, time, cfgCross);
        Result = average_two_results(Result12, Result21);
        if isempty(Result), continue; end

        Result.analysis = 'loadCrossSide';
        Result.feature = featName;
        Result.labelMeaning = {'lowLoad', 'highLoad'};
        Result.directionMeaning = {'trainAttendLeft_testAttendRight', 'trainAttendRight_testAttendLeft'};
        Result.directionResults = {Result12, Result21};
        Result.baselinewindow = baselinewindow;
        Result.frep = frep;
        save_result(outputdir, 'loadCrossSide', featName, file, Result);
    end

    fprintf('Finished %s\n', file);
end

%% ========================================================================
% local functions
% ========================================================================

function sideOne = build_side_features(sideOne, eeg0, artifactInd, chanLabels, sideCfg, ...
    L_labels, R_labels, global_labels, srate, time, baselinewindow, frep)

    [contraLow, ipsiLow, rawLow, leftLow, rightLow] = get_clean_trials_fixed_order( ...
        eeg0, artifactInd, chanLabels, sideCfg.condLow, sideCfg.channel_contra, ...
        sideCfg.channel_ipsi, L_labels, R_labels, global_labels);

    [~, ~, rawMid, leftMid, rightMid] = get_clean_trials_fixed_order( ...
        eeg0, artifactInd, chanLabels, sideCfg.condMid, sideCfg.channel_contra, ...
        sideCfg.channel_ipsi, L_labels, R_labels, global_labels);

    [contraHigh, ipsiHigh, rawHigh, leftHigh, rightHigh] = get_clean_trials_fixed_order( ...
        eeg0, artifactInd, chanLabels, sideCfg.condHigh, sideCfg.channel_contra, ...
        sideCfg.channel_ipsi, L_labels, R_labels, global_labels);

    alphaContraLow  = calculate_high_gamma_power(contraLow,  srate, time, baselinewindow, frep);
    alphaIpsiLow    = calculate_high_gamma_power(ipsiLow,    srate, time, baselinewindow, frep);
    alphaRawLow     = calculate_high_gamma_power(rawLow,     srate, time, baselinewindow, frep);
    alphaLeftLow    = calculate_high_gamma_power(leftLow,    srate, time, baselinewindow, frep);
    alphaRightLow   = calculate_high_gamma_power(rightLow,   srate, time, baselinewindow, frep);

    alphaRawMid     = calculate_high_gamma_power(rawMid,     srate, time, baselinewindow, frep);
    alphaLeftMid    = calculate_high_gamma_power(leftMid,    srate, time, baselinewindow, frep);
    alphaRightMid   = calculate_high_gamma_power(rightMid,   srate, time, baselinewindow, frep);

    alphaContraHigh = calculate_high_gamma_power(contraHigh, srate, time, baselinewindow, frep);
    alphaIpsiHigh   = calculate_high_gamma_power(ipsiHigh,   srate, time, baselinewindow, frep);
    alphaRawHigh    = calculate_high_gamma_power(rawHigh,    srate, time, baselinewindow, frep);
    alphaLeftHigh   = calculate_high_gamma_power(leftHigh,   srate, time, baselinewindow, frep);
    alphaRightHigh  = calculate_high_gamma_power(rightHigh,  srate, time, baselinewindow, frep);

    % Load-decoding features: side-normalized unless noted.
    sideOne.CDA.low  = contraLow  - ipsiLow;
    sideOne.CDA.high = contraHigh - ipsiHigh;

    sideOne.Alpha.low  = alphaContraLow  - alphaIpsiLow;
    sideOne.Alpha.high = alphaContraHigh - alphaIpsiHigh;

    sideOne.GlobalAlpha.low  = alphaRawLow;   % all posterior alpha channels, fixed anatomical order
    sideOne.GlobalAlpha.high = alphaRawHigh;

    sideOne.GlobalAlphaMean.low  = mean(alphaRawLow,  1, 'omitnan');
    sideOne.GlobalAlphaMean.high = mean(alphaRawHigh, 1, 'omitnan');

    % Side-decoding features: fixed anatomical features; no contra/ipsi recoding.
    % Side decoding uses all load levels: low/mid/high = 1/2/3 vs 4/5/6.
    sideOne.VoltageRawLR.low  = rawLow;
    sideOne.VoltageRawLR.mid  = rawMid;
    sideOne.VoltageRawLR.high = rawHigh;

    sideOne.AlphaRawLR.low  = alphaRawLow;
    sideOne.AlphaRawLR.mid  = alphaRawMid;
    sideOne.AlphaRawLR.high = alphaRawHigh;

    sideOne.VoltageLminusR.low  = leftLow  - rightLow;
    sideOne.VoltageLminusR.mid  = leftMid  - rightMid;
    sideOne.VoltageLminusR.high = leftHigh - rightHigh;

    sideOne.AlphaLminusR.low  = alphaLeftLow  - alphaRightLow;
    sideOne.AlphaLminusR.mid  = alphaLeftMid  - alphaRightMid;
    sideOne.AlphaLminusR.high = alphaLeftHigh - alphaRightHigh;

    sideOne.GlobalAlphaMean.mid = mean(alphaRawMid, 1, 'omitnan');
end

function [Xcontra, Xipsi, Xraw, XL, XR] = get_clean_trials_fixed_order( ...
    eeg0, artifactInd, chanLabels, condIdx, contraLabels, ipsiLabels, L_labels, R_labels, global_labels)

    contraIdx = find_chan_idx(chanLabels, contraLabels);
    ipsiIdx   = find_chan_idx(chanLabels, ipsiLabels);
    L_idx     = find_chan_idx(chanLabels, L_labels);
    R_idx     = find_chan_idx(chanLabels, R_labels);
    rawIdx    = find_chan_idx(chanLabels, global_labels);

    contra = squeeze_condition(eeg0, condIdx, contraIdx);
    ipsi   = squeeze_condition(eeg0, condIdx, ipsiIdx);
    raw    = squeeze_condition(eeg0, condIdx, rawIdx);
    left   = squeeze_condition(eeg0, condIdx, L_idx);
    right  = squeeze_condition(eeg0, condIdx, R_idx);

    keepTrial = ~artifactInd(condIdx,:);
    keepTrial = keepTrial & finite_trials(contra) & finite_trials(ipsi) & finite_trials(raw) & ...
        finite_trials(left) & finite_trials(right);

    Xcontra = permute(contra(keepTrial,:,:), [2 3 1]);
    Xipsi   = permute(ipsi(keepTrial,:,:),   [2 3 1]);
    Xraw    = permute(raw(keepTrial,:,:),    [2 3 1]);
    XL      = permute(left(keepTrial,:,:),   [2 3 1]);
    XR      = permute(right(keepTrial,:,:),  [2 3 1]);
end

function dat = squeeze_condition(eeg0, condIdx, chanIdx)
    nTrial = size(eeg0, 2);
    dat = reshape(eeg0(condIdx,:,chanIdx,:), [nTrial, numel(chanIdx), size(eeg0,4)]);
end

function keep = finite_trials(dat)
    keep = squeeze(~any(any(~isfinite(dat), 2), 3))';
end

function idx = find_chan_idx(chanLabels, labels)
    [tf, loc] = ismember(labels, chanLabels);
    idx = loc(tf);
end

function [dataSide, labelsSide, loadFactor] = make_side_decoding_data(sideDat, featName)
    % Decode spatial side using all load levels.
    % condition 1/2/3 = side 1; condition 4/5/6 = side 2.
    % loadFactor keeps low/mid/high balanced across side during SVM iterations.
    loadLevels = {'low', 'mid', 'high'};

    X = cell(2, numel(loadLevels));
    n = zeros(2, numel(loadLevels));
    for sidei = 1:2
        for li = 1:numel(loadLevels)
            X{sidei, li} = sideDat(sidei).(featName).(loadLevels{li});
            n(sidei, li) = size(X{sidei, li}, 3);
        end
    end

    dataSide = cat(3, X{1,1}, X{1,2}, X{1,3}, X{2,1}, X{2,2}, X{2,3});

    labelsSide = [ones(sum(n(1,:)), 1); ...
                  2 * ones(sum(n(2,:)), 1)];

    loadFactor = [ones(n(1,1),1); 2*ones(n(1,2),1); 3*ones(n(1,3),1); ...
                  ones(n(2,1),1); 2*ones(n(2,2),1); 3*ones(n(2,3),1)];
end

function [dataLoad, labelsLoad] = make_load_data_one_side(sideOne, featName)
    if strcmpi(featName, 'NoPCA') || strcmpi(featName, 'PCA')
        Xlow  = cat(1, sideOne.CDA.low,  sideOne.Alpha.low);
        Xhigh = cat(1, sideOne.CDA.high, sideOne.Alpha.high);
    else
        Xlow  = sideOne.(featName).low;
        Xhigh = sideOne.(featName).high;
    end

    dataLoad = cat(3, Xlow, Xhigh);
    labelsLoad = [ones(size(Xlow,3), 1); 2 * ones(size(Xhigh,3), 1)];
end

function [dataLoad, labelsLoad, sideFactor] = make_side_balanced_load_data(sideDat, featName)
    [data1, y1] = make_load_data_one_side(sideDat(1), featName);
    [data2, y2] = make_load_data_one_side(sideDat(2), featName);

    dataLoad = cat(3, data1, data2);
    labelsLoad = [y1; y2];
    sideFactor = [ones(numel(y1),1); 2 * ones(numel(y2),1)];
end

function Result = run_svm_if_enough(data, labels, time, cfg)
    Result = [];
    labels = labels(:);
    if numel(unique(labels)) ~= 2, return; end
    counts = arrayfun(@(x) sum(labels == x), unique(labels));
    if min(counts) < cfg.superTrial || size(data,3) < 2 * cfg.superTrial
        return;
    end
    if size(data,3) ~= numel(labels), return; end
    Result = SVM_function_singleSubj(data, labels, time, cfg);
    u = unique(labels);
    Result.nClass1 = sum(labels == u(1));
    Result.nClass2 = sum(labels == u(2));
    Result.labels = labels;
end

function Result = run_cross_if_enough(trainData, trainLabels, testData, testLabels, time, cfg)
    Result = [];
    if numel(unique(trainLabels)) ~= 2 || numel(unique(testLabels)) ~= 2, return; end
    if min(histcounts(trainLabels, [0.5 1.5 2.5])) < cfg.superTrial, return; end
    if min(histcounts(testLabels,  [0.5 1.5 2.5])) < cfg.superTrial, return; end
    Result = SVM_crossSide_singleSubj(trainData, trainLabels, testData, testLabels, time, cfg);
end

function Result = average_two_results(R1, R2)
    Result = [];
    valid = {R1, R2};
    valid = valid(~cellfun(@isempty, valid));
    if isempty(valid), return; end
    Result = valid{1};
    fieldsToAverage = {'predictAcc', 'AUC', 'predictAccShuffle', 'predictAccMinusShuffle', ...
        'AUCShuffle', 'AUCMinusShuffle', 'weights', 'predictAccTrain'};
    for fi = 1:numel(fieldsToAverage)
        f = fieldsToAverage{fi};
        vals = {};
        for ri = 1:numel(valid)
            if isfield(valid{ri}, f) && ~isempty(valid{ri}.(f))
                vals{end+1} = valid{ri}.(f); %#ok<AGROW>
            end
        end
        if numel(vals) == 2 && isequal(size(vals{1}), size(vals{2}))
            Result.(f) = mean(cat(ndims(vals{1})+1, vals{:}), ndims(vals{1})+1, 'omitnan');
        elseif isscalar(vals)
            Result.(f) = vals{1};
        end
    end
end

function save_result(outputdir, analysisName, featName, file, Result)
    outDir = fullfile(outputdir, analysisName, featName);
    if ~isfolder(outDir), mkdir(outDir); end
    save(fullfile(outDir, file), 'Result', '-v7.3');
end

function labels = channels_for_feature(featName, L_labels, R_labels, global_labels)
    switch featName
        case {'VoltageRawLR', 'AlphaRawLR'}
            labels = global_labels;
        case {'VoltageLminusR', 'AlphaLminusR'}
            labels = strcat(L_labels, '_minus_', R_labels);
        case 'GlobalAlphaMean'
            labels = {'meanPosteriorAlpha'};
        otherwise
            labels = {};
    end
end

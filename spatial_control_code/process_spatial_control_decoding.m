%% process_spatial_control_decoding.m
% Follow-up analyses for CDA vs Alpha project:
% 1) side decoding positive control
% 2) within-side load decoding: setsize1 vs3
% 3) side-balanced load decoding
% 4) cross-side load generalization
%
% Put this file in CDA_code/code or any folder where your existing helper
% functions are on the MATLAB path:
% LDA_function_singleSubj.m
% LDA_crossSide_singleSubj.m
% calculate_hilbert_band_power.m
% balance_trials_by_label.m
% func_make_superTrials.m

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
mainCodeDir = fileparts(codeDir);
projectRoot = fileparts(mainCodeDir);
addpath(mainCodeDir);
addpath(codeDir);

maindir   = fullfile(projectRoot, 'data1');
datadir   = fullfile(maindir, 'data');
outputdir = fullfile(maindir, 'decoding_LDA_spatialControl');

%% folders
% sideFeatures = {'VoltageRawLR', 'AlphaRawLR', 'VoltageLminusR', 'AlphaLminusR', 'GlobalAlphaMean'};
loadFeatures = {'CDA'}; %, 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
analysisNames = {'loadWithinSide_setsize1_vs3'};

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

%% decoding config: keep close to process_data1.m
cfg = struct();
cfg.cvType = 'kfold';
cfg.trainRatio = 2/3;
cfg.nFolds = 5;
cfg.superTrial = 1;
cfg.nIter = 50;
cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';
cfg.doTimeGeneralization = 1;
cfg.doPCA = false;
cfg.nPCs = 5;
cfg.discrimType = 'diagLinear';
cfg.ldaEngine = 'fitcdiscr';
cfg.standardize = 1;
cfg.doShuffle = 0;
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

% In the original process_data1.m:
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

loadComparisons(1).analysisName = 'loadWithinSide_setsize1_vs3';
loadComparisons(1).comparisonName = 'setsize1_vs3';
loadComparisons(1).classLevels = {'low', 'mid'};
loadComparisons(1).setSizes = [1 3];
loadComparisons(1).labelMeaning = {'setsize1', 'setsize3'};

%% alpha config
baselinewindow = [-1400, -1100];
% baselinewindow = [-500, -200];

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

    [includeSubject, inclusionInfo] = data1_subject_inclusion(tmp.eeg);
    if ~includeSubject
        fprintf('Skip %s: original data1 criterion failed, set-size trial counts = [%s]\n', ...
            file, sprintf('%d ', inclusionInfo.trialCountsPerSetSize));
        continue
    end

    sideDatCell = cell(1, 2);
    for sidei = 1:2
        sideOne = struct();
        sideOne.name = sideCfg(sidei).name;

        levelNames = {'low', 'mid', 'high'};
        condVals = [sideCfg(sidei).condLow, sideCfg(sidei).condMid, sideCfg(sidei).condHigh];
        voltage = struct();
        alphaPower = struct();

        [tf, loc] = ismember(sideCfg(sidei).channel_contra, chanLabels);
        contraIdx = loc(tf);
        [tf, loc] = ismember(sideCfg(sidei).channel_ipsi, chanLabels);
        ipsiIdx = loc(tf);
        [tf, loc] = ismember(L_labels, chanLabels);
        L_idx = loc(tf);
        [tf, loc] = ismember(R_labels, chanLabels);
        R_idx = loc(tf);
        [tf, loc] = ismember(global_labels, chanLabels);
        rawIdx = loc(tf);

        for li = 1:numel(levelNames)
            levelName = levelNames{li};
            condIdx = condVals(li);
            nTrial = size(eeg0, 2);

            contra = reshape(eeg0(condIdx,:,contraIdx,:), [nTrial, numel(contraIdx), size(eeg0,4)]);
            ipsi = reshape(eeg0(condIdx,:,ipsiIdx,:), [nTrial, numel(ipsiIdx), size(eeg0,4)]);
            raw = reshape(eeg0(condIdx,:,rawIdx,:), [nTrial, numel(rawIdx), size(eeg0,4)]);
            left = reshape(eeg0(condIdx,:,L_idx,:), [nTrial, numel(L_idx), size(eeg0,4)]);
            right = reshape(eeg0(condIdx,:,R_idx,:), [nTrial, numel(R_idx), size(eeg0,4)]);

            keepContra = squeeze(~any(any(~isfinite(contra), 2), 3))';
            keepIpsi = squeeze(~any(any(~isfinite(ipsi), 2), 3))';
            keepRaw = squeeze(~any(any(~isfinite(raw), 2), 3))';
            keepLeft = squeeze(~any(any(~isfinite(left), 2), 3))';
            keepRight = squeeze(~any(any(~isfinite(right), 2), 3))';
            keepTrial = ~artifactInd(condIdx,:);
            keepTrial = keepTrial & keepContra & keepIpsi & keepRaw & keepLeft & keepRight;

            voltage.(levelName).contra = permute(contra(keepTrial,:,:), [2 3 1]);
            voltage.(levelName).ipsi = permute(ipsi(keepTrial,:,:), [2 3 1]);
            voltage.(levelName).raw = permute(raw(keepTrial,:,:), [2 3 1]);
            voltage.(levelName).left = permute(left(keepTrial,:,:), [2 3 1]);
            voltage.(levelName).right = permute(right(keepTrial,:,:), [2 3 1]);

            alphaPower.(levelName).contra = calculate_hilbert_band_power(voltage.(levelName).contra, srate, time, baselinewindow, frep);
            alphaPower.(levelName).ipsi = calculate_hilbert_band_power(voltage.(levelName).ipsi, srate, time, baselinewindow, frep);
            alphaPower.(levelName).raw = calculate_hilbert_band_power(voltage.(levelName).raw, srate, time, baselinewindow, frep);
            alphaPower.(levelName).left = calculate_hilbert_band_power(voltage.(levelName).left, srate, time, baselinewindow, frep);
            alphaPower.(levelName).right = calculate_hilbert_band_power(voltage.(levelName).right, srate, time, baselinewindow, frep);
        end

        sideOne.CDA.low = voltage.low.contra - voltage.low.ipsi;
        sideOne.CDA.mid = voltage.mid.contra - voltage.mid.ipsi;
        sideOne.CDA.high = voltage.high.contra - voltage.high.ipsi;

        sideOne.Alpha.low = alphaPower.low.contra - alphaPower.low.ipsi;
        sideOne.Alpha.mid = alphaPower.mid.contra - alphaPower.mid.ipsi;
        sideOne.Alpha.high = alphaPower.high.contra - alphaPower.high.ipsi;

        sideOne.GlobalAlpha.low = alphaPower.low.raw;
        sideOne.GlobalAlpha.mid = alphaPower.mid.raw;
        sideOne.GlobalAlpha.high = alphaPower.high.raw;
        sideOne.GlobalAlphaMean.low = mean(alphaPower.low.raw, 1, 'omitnan');
        sideOne.GlobalAlphaMean.mid = mean(alphaPower.mid.raw, 1, 'omitnan');
        sideOne.GlobalAlphaMean.high = mean(alphaPower.high.raw, 1, 'omitnan');

        sideOne.VoltageRawLR.low = voltage.low.raw;
        sideOne.VoltageRawLR.mid = voltage.mid.raw;
        sideOne.VoltageRawLR.high = voltage.high.raw;

        sideOne.AlphaRawLR.low = alphaPower.low.raw;
        sideOne.AlphaRawLR.mid = alphaPower.mid.raw;
        sideOne.AlphaRawLR.high = alphaPower.high.raw;

        sideOne.VoltageLminusR.low = voltage.low.left - voltage.low.right;
        sideOne.VoltageLminusR.mid = voltage.mid.left - voltage.mid.right;
        sideOne.VoltageLminusR.high = voltage.high.left - voltage.high.right;

        sideOne.AlphaLminusR.low = alphaPower.low.left - alphaPower.low.right;
        sideOne.AlphaLminusR.mid = alphaPower.mid.left - alphaPower.mid.right;
        sideOne.AlphaLminusR.high = alphaPower.high.left - alphaPower.high.right;

        sideDatCell{sidei} = sideOne;
    end
    sideDat = [sideDatCell{:}];

    %% ============================================================
    % 1) Side decoding positive control
    %    Use fixed anatomical features only. Do NOT use contra-minus-ipsi.
    % ============================================================
    % for fi = 1:numel(sideFeatures)
    %     featName = sideFeatures{fi};
    %     [dataSide, labelsSide, loadFactor] = make_side_decoding_data(sideDat, featName);
    % 
    %     cfgSide = cfg;
    %     cfgSide.doPCA = false;
    %     cfgSide.balanceFactors = loadFactor(:);
    % 
    %     Result = run_LDA_if_enough(dataSide, labelsSide, time, cfgSide);
    %     if isempty(Result), continue; end
    % 
    %     Result.analysis = 'sideDecoding';
    %     Result.feature = featName;
    %     Result.labelMeaning = {'attendLeft', 'attendRight'};
    %     Result.loadFactor = loadFactor;
    %     Result.baselinewindow = baselinewindow;
    %     Result.frep = frep;
    %     Result.channelLabels = channels_for_feature(featName, L_labels, R_labels, global_labels);
    %     save_result(outputdir, 'sideDecoding', featName, file, Result);
    % end

    %% ============================================================
    % 2) Within-side load decoding
    %    Decode each set-size pair separately within each side, then average.
    % ============================================================
    for compi = 1:numel(loadComparisons)
        analysisName = loadComparisons(compi).analysisName;
        classLevel1 = loadComparisons(compi).classLevels{1};
        classLevel2 = loadComparisons(compi).classLevels{2};

        for fi = 1:numel(loadFeatures)
            featName = loadFeatures{fi};

            cfgLoad = cfg;
            cfgLoad.balanceFactors = [];
            cfgLoad.doPCA = strcmpi(featName, 'PCA');

            ResultSide = cell(1, 2);
            for sidei = 1:2
                if strcmpi(featName, 'NoPCA') || strcmpi(featName, 'PCA')
                    Xclass1 = cat(1, sideDat(sidei).CDA.(classLevel1), sideDat(sidei).Alpha.(classLevel1));
                    Xclass2 = cat(1, sideDat(sidei).CDA.(classLevel2), sideDat(sidei).Alpha.(classLevel2));
                else
                    Xclass1 = sideDat(sidei).(featName).(classLevel1);
                    Xclass2 = sideDat(sidei).(featName).(classLevel2);
                end

                dataLoad = cat(3, Xclass1, Xclass2);
                labelsLoad = [ones(size(Xclass1,3), 1); 2 * ones(size(Xclass2,3), 1)];

                ResultSide{sidei} = [];
                labelsLoad = labelsLoad(:);
                if numel(unique(labelsLoad)) == 2
                    counts = arrayfun(@(x) sum(labelsLoad == x), unique(labelsLoad));
                    if min(counts) >= cfgLoad.superTrial && size(dataLoad,3) >= 2 * cfgLoad.superTrial && ...
                            size(dataLoad,3) == numel(labelsLoad)
                        ResultSide{sidei} = LDA_function_singleSubj(dataLoad, labelsLoad, time, cfgLoad);
                        u = unique(labelsLoad);
                        ResultSide{sidei}.nClass1 = sum(labelsLoad == u(1));
                        ResultSide{sidei}.nClass2 = sum(labelsLoad == u(2));
                        ResultSide{sidei}.labels = labelsLoad;
                    end
                end
                if ~isempty(ResultSide{sidei})
                    ResultSide{sidei}.analysis = [analysisName '_singleSide'];
                    ResultSide{sidei}.feature = featName;
                    ResultSide{sidei}.sideName = sideDat(sidei).name;
                    ResultSide{sidei}.comparisonName = loadComparisons(compi).comparisonName;
                    ResultSide{sidei}.classLevels = loadComparisons(compi).classLevels;
                    ResultSide{sidei}.setSizes = loadComparisons(compi).setSizes;
                end
            end

            Result = [];
            valid = ResultSide(~cellfun(@isempty, ResultSide));
            if ~isempty(valid)
                Result = valid{1};
                fieldsToAverage = {'Acc', 'AUC', 'AccShuffle', 'AccMinusShuffle', ...
                    'AUCShuffle', 'AUCMinusShuffle', 'weights', 'AccTrain'};
                for avgIdx = 1:numel(fieldsToAverage)
                    f = fieldsToAverage{avgIdx};
                    vals = {};
                    for ri = 1:numel(valid)
                        if isfield(valid{ri}, f) && ~isempty(valid{ri}.(f))
                            vals{end+1} = valid{ri}.(f); 
                        end
                    end
                    if numel(vals) == 2 && isequal(size(vals{1}), size(vals{2}))
                        Result.(f) = mean(cat(ndims(vals{1})+1, vals{:}), ndims(vals{1})+1, 'omitnan');
                    elseif isscalar(vals)
                        Result.(f) = vals{1};
                    end
                end
            end
            if isempty(Result), continue; end

            Result.analysis = analysisName;
            Result.feature = featName;
            Result.comparisonName = loadComparisons(compi).comparisonName;
            Result.classLevels = loadComparisons(compi).classLevels;
            Result.setSizes = loadComparisons(compi).setSizes;
            Result.labelMeaning = loadComparisons(compi).labelMeaning;
            Result.subjectInclusion = inclusionInfo;
            Result.sideResults = ResultSide;
            Result.baselinewindow = baselinewindow;
            Result.frep = frep;
            outDir = fullfile(outputdir, analysisName, featName);
            if ~isfolder(outDir), mkdir(outDir); end
            save(fullfile(outDir, file), 'Result', '-v7.3');
        end
    end

    % %% ============================================================
    % % 3) Side-balanced load decoding
    % %    Merge two sides, but balance each load x side cell in each iteration.
    % % ============================================================
    % for fi = 1:numel(loadFeatures)
    %     featName = loadFeatures{fi};
    %     [dataLoad, labelsLoad, sideFactor] = make_side_balanced_load_data(sideDat, featName);
    % 
    %     cfgBal = cfg;
    %     cfgBal.balanceFactors = sideFactor(:);
    %     cfgBal.doPCA = strcmpi(featName, 'PCA');
    % 
    %     Result = run_LDA_if_enough(dataLoad, labelsLoad, time, cfgBal);
    %     if isempty(Result), continue; end
    % 
    %     Result.analysis = 'loadSideBalanced';
    %     Result.feature = featName;
    %     Result.labelMeaning = {'lowLoad', 'highLoad'};
    %     Result.sideFactor = sideFactor;
    %     Result.sideMeaning = {'attendLeft', 'attendRight'};
    %     Result.baselinewindow = baselinewindow;
    %     Result.frep = frep;
    %     save_result(outputdir, 'loadSideBalanced', featName, file, Result);
    % end
    % 
    % %% ============================================================
    % % 4) Cross-side load generalization
    % %    Train low vs high on one side, test on the other side, average directions.
    % % ============================================================
    % for fi = 1:numel(loadFeatures)
    %     featName = loadFeatures{fi};
    % 
    %     cfgCross = cfg;
    %     cfgCross.doPCA = strcmpi(featName, 'PCA');
    %     cfgCross.useParallel = false;   % cross-side helper is already iteration-level simple loop
    % 
    %     [train12, yTrain12] = make_load_data_one_side(sideDat(1), featName);
    %     [test12,  yTest12]  = make_load_data_one_side(sideDat(2), featName);
    %     [train21, yTrain21] = make_load_data_one_side(sideDat(2), featName);
    %     [test21,  yTest21]  = make_load_data_one_side(sideDat(1), featName);
    % 
    %     Result12 = run_cross_if_enough(train12, yTrain12, test12, yTest12, time, cfgCross);
    %     Result21 = run_cross_if_enough(train21, yTrain21, test21, yTest21, time, cfgCross);
    %     Result = average_two_results(Result12, Result21);
    %     if isempty(Result), continue; end
    % 
    %     Result.analysis = 'loadCrossSide';
    %     Result.feature = featName;
    %     Result.labelMeaning = {'lowLoad', 'highLoad'};
    %     Result.directionMeaning = {'trainAttendLeft_testAttendRight', 'trainAttendRight_testAttendLeft'};
    %     Result.directionResults = {Result12, Result21};
    %     Result.baselinewindow = baselinewindow;
    %     Result.frep = frep;
    %     save_result(outputdir, 'loadCrossSide', featName, file, Result);
    % end

    fprintf('Finished %s\n', file);
end


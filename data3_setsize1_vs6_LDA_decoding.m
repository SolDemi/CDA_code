%% data3 set-size-1/3 vs set-size-6 sequential LDA decoding
% Reads first-memory-locked cda/alpha data saved by data3_cda_alpha.m.
% Low-set-size 460-ms segments are decoded against consecutive set-size 6
% 460-ms segments. Segment matrices are concatenated to low-set-size time x
% set-size-6 time.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
datadir = fullfile(dataDir, 'cda_alpha');
behaviorDirCandidates = { ...
    fullfile(dataDir, 'Behavior_data_script')};


addpath(codeDir);
modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};

analysisModeName = 'maintOnly'; % 'encMaint': encoding + maintenance; 'maintOnly': maintenance only

stimStepMs = 460;      % item onset-to-onset interval
encMaintWidthMs = 460; % 100 ms encoding + 360 ms maintenance
maintWidthMs = 360;    % maintenance only, 100-460 ms after each item onset

switch lower(analysisModeName)
    case 'encmaint'
        segmentStartOffsetMs = 0;
        segmentWidthMs = encMaintWidthMs;
        segmentDescription = 'encoding + maintenance';
    case 'maintonly'
        segmentStartOffsetMs = 100;
        segmentWidthMs = maintWidthMs;
        segmentDescription = 'maintenance only';
    otherwise
        error('Unsupported analysisModeName: %s.', analysisModeName);
end

highSetSize = 6;
stimOnsetsMs = (0:(highSetSize - 1)) * stimStepMs;
highSegmentStartsMs = stimOnsetsMs + segmentStartOffsetMs;

comparisons = repmat(struct('name', '', 'outputdir', '', 'seq', []), 2, 1);

%% set size 1 vs set size 6
seq = struct();
seq.lowSetSize = 1;
seq.highSetSize = highSetSize;
seq.lowWindowsMs = [segmentStartOffsetMs, segmentStartOffsetMs + segmentWidthMs];
seq.highSegmentWidthMs = segmentWidthMs;
seq.highFirstSegmentStartMs = segmentStartOffsetMs;
seq.highSegmentStartsMs = highSegmentStartsMs;

seq.labelLow = 1;
seq.labelHigh = 2;
seq.chanceAccuracy = 0.5;
seq.analysisMode = analysisModeName;
seq.segmentDescription = segmentDescription;
seq.segmentTailPolicy = sprintf( ...
    'Use exactly one %d-ms %s segment per high-set-size item; post-final delay is not decoded.', ...
    segmentWidthMs, segmentDescription);

comparisons(1).name = sprintf('setsize1_vs6_%s', analysisModeName);
comparisons(1).outputdir = fullfile(dataDir, sprintf('decoding_LDA_setsize1_vs6_segments_%s', analysisModeName));
comparisons(1).seq = seq;

%% set size 3 vs set size 6
seq = struct();
seq.lowSetSize = 3;
seq.highSetSize = highSetSize;

lowItemOnsetsMs = (0:(seq.lowSetSize - 1)) * stimStepMs;
seq.lowWindowsMs = [lowItemOnsetsMs(:) + segmentStartOffsetMs, ...
                    lowItemOnsetsMs(:) + segmentStartOffsetMs + segmentWidthMs];

seq.highSegmentWidthMs = segmentWidthMs;
seq.highFirstSegmentStartMs = segmentStartOffsetMs;
seq.highSegmentStartsMs = highSegmentStartsMs;

seq.labelLow = 1;
seq.labelHigh = 2;
seq.chanceAccuracy = 0.5;
seq.analysisMode = analysisModeName;
seq.segmentDescription = segmentDescription;
seq.segmentTailPolicy = sprintf( ...
    'Use exactly one %d-ms %s segment per high-set-size item; post-final delay is not decoded.', ...
    segmentWidthMs, segmentDescription);

comparisons(2).name = sprintf('setsize3_vs6_%s', analysisModeName);
comparisons(2).outputdir = fullfile(dataDir, sprintf('decoding_LDA_setsize3_vs6_segments_%s', analysisModeName));
comparisons(2).seq = seq;

for ci = 1:numel(comparisons)
    if ~isfolder(comparisons(ci).outputdir)
        mkdir(comparisons(ci).outputdir);
    end
    for mi = 1:numel(modelNames)
        modelDir = fullfile(comparisons(ci).outputdir, modelNames{mi});
        if ~isfolder(modelDir)
            mkdir(modelDir);
        end
    end
end

%% config: same as LDA_decoding.m, except no shuffled-label baseline
cfg = struct();
cfg.cvType = 'kfold';
cfg.trainRatio = 2/3;
cfg.nFolds = 5;
cfg.superTrial = 1;
cfg.nIter = 50;

cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';

cfg.analysisWindow = [-200 inf];
cfg.doTimeGeneralization = true;
cfg.doPCA = false;
cfg.nPCs = 5;

cfg.discrimType = 'diagLinear';
cfg.ldaEngine = 'fitcdiscr';
cfg.standardize = 1;

cfg.doShuffle = false;
cfg.balanceTrials = true;
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];
cfg.useAUC = 1;
cfg.returnDecisionValues = true;
cfg.useParallel = true;
cfg.verbose = false;
cfg.randomSeed = [];

files = dir(fullfile(datadir, 'sub*.mat'));
files = data3_filter_subject_mat_files(files, data3_subject_filter());
if isempty(files)
    error('No sub*.mat files found in %s.', datadir);
end

sideNames = {'L', 'R'};
loadNames = {'low', 'high'};

for sf = 1:numel(files)
    file = files(sf).name;
    inFile = fullfile(files(sf).folder, file);
    fprintf('data3 sequential LDA [%s]: %s\n', analysisModeName, file);

    S = load(inFile, 'cda', 'alpha');
    if ~(isfield(S, 'cda') && isfield(S, 'alpha'))
        error('%s does not contain cda and alpha. Rerun data3_cda_alpha.', inFile);
    end

    needed = {'cda', 'alpha'};
    for ni = 1:numel(needed)
        if ~isfield(S.(needed{ni}), 'trial') || ~isfield(S.(needed{ni}), 'timeBySetSize')
            error('Saved data in %s is missing %s.trial or %s.timeBySetSize.', ...
                inFile, needed{ni}, needed{ni});
        end
    end
    savedData = struct('cda', S.cda, 'alpha', S.alpha);

    tok = regexp(file, '^sub(\d+)\.mat$', 'tokens', 'once');
    if isempty(tok)
        subject = NaN;
    else
        subject = str2double(tok{1});
    end
    if isfield(savedData.cda, 'subject') && ~isempty(savedData.cda.subject)
        subject = savedData.cda.subject;
    end

    behaviorDir = '';
    for bdi = 1:numel(behaviorDirCandidates)
        if isfolder(fullfile(behaviorDirCandidates{bdi}, 'Beh_data'))
            behaviorDir = behaviorDirCandidates{bdi};
            break;
        end
    end
    if isempty(behaviorDir)
        error('Cannot find data3 behavior folder. Checked: %s', strjoin(behaviorDirCandidates, '; '));
    end

    behavior = table();
    trialOffset = 0;
    for blockNum = 1:4
        behaviorFile = fullfile(behaviorDir, 'Beh_data', ...
            sprintf('cda_cVl_serial_data%d_%d.mat', subject, blockNum));
        if ~isfile(behaviorFile)
            error('Missing behavior file: %s', behaviorFile);
        end

        B = load(behaviorFile, 'data');
        if ~isfield(B, 'data')
            error('%s does not contain data.', behaviorFile);
        end
        behData = B.data;
        nBehTrial = numel(behData.set_size);

        blockBehavior = table();
        blockBehavior.Subject = repmat(subject, nBehTrial, 1);
        blockBehavior.Block = repmat(blockNum, nBehTrial, 1);
        blockBehavior.BlockTrial = (1:nBehTrial)';
        blockBehavior.TrialIndex = trialOffset + (1:nBehTrial)';
        blockBehavior.MemoryType = behData.mem_lets(:);       % 0=color, 1=letter
        blockBehavior.SetSize = behData.set_size(:);
        blockBehavior.Change = behData.change(:);
        blockBehavior.CueSide = behData.cue_side(:);          % -1=left, 1=right
        blockBehavior.SerialPosition = behData.change_item(:);
        blockBehavior.UncuedSerialPosition = behData.change_item_uncued(:);
        blockBehavior.Response = behData.resp(:);
        blockBehavior.RT = behData.rt(:);
        blockBehavior.Correct = behData.acc(:);

        behavior = [behavior; blockBehavior]; %#ok<AGROW>
        trialOffset = trialOffset + nBehTrial;
    end

    for setSizeCheck = [1 3 6]
        setFieldCheck = sprintf('setsize%d', setSizeCheck);
        nBehavior = sum(behavior.SetSize == setSizeCheck);
        nEvents = savedData.cda.extraction.(setFieldCheck).nRawEventTrials;
        if nBehavior ~= nEvents
            error('Subject %d setsize%d behavior/EEG trial count mismatch: behavior=%d, EEG events=%d.', ...
                subject, setSizeCheck, nBehavior, nEvents);
        end
    end

    for ci = 1:numel(comparisons)
        seq = comparisons(ci).seq;
        fprintf('  %s\n', comparisons(ci).name);

        for mi = 1:numel(modelNames)
            modelName = modelNames{mi};
            cfgModel = cfg;
            cfgModel.doPCA = strcmp(modelName, 'PCA');

            lowField = sprintf('setsize%d', seq.lowSetSize);
            highField = sprintf('setsize%d', seq.highSetSize);
            modelData = struct();

            switch lower(modelName)
                case {'cda', 'alpha', 'globalalpha', 'globalalphamean'}
                    if strcmpi(modelName, 'CDA')
                        sourceData = savedData.cda;
                        featureMode = 'lateralized';
                        modelData.channelLabels = savedData.cda.relPairLabels;
                    elseif strcmpi(modelName, 'Alpha')
                        sourceData = savedData.alpha;
                        featureMode = 'lateralized';
                        modelData.channelLabels = savedData.alpha.relPairLabels;
                    elseif strcmpi(modelName, 'GlobalAlpha')
                        sourceData = savedData.alpha;
                        featureMode = 'global';
                        modelData.channelLabels = [savedData.alpha.leftElecLabels savedData.alpha.rightElecLabels];
                    else
                        sourceData = savedData.alpha;
                        featureMode = 'globalmean';
                        modelData.channelLabels = {'posteriorAlphaMean'};
                    end

                    for si = 1:numel(sideNames)
                        sideName = sideNames{si};
                        for li = 1:numel(loadNames)
                            if strcmp(loadNames{li}, 'low')
                                setSizeNow = seq.lowSetSize;
                            else
                                setSizeNow = seq.highSetSize;
                            end

                            loadStr = num2str(setSizeNow);
                            leftName = sprintf('left_%s_%s', sideName, loadStr);
                            rightName = sprintf('right_%s_%s', sideName, loadStr);
                            if ~(isfield(sourceData.trial, leftName) && isfield(sourceData.trial, rightName))
                                error('Saved data is missing %s/%s.', leftName, rightName);
                            end

                            leftX = sourceData.trial.(leftName);
                            rightX = sourceData.trial.(rightName);
                            switch lower(featureMode)
                                case 'lateralized'
                                    if strcmpi(sideName, 'L')
                                        X = rightX - leftX;
                                    else
                                        X = leftX - rightX;
                                    end
                                case 'global'
                                    X = cat(2, leftX, rightX);
                                case 'globalmean'
                                    X = mean(cat(2, leftX, rightX), 2, 'omitnan');
                                otherwise
                                    error('Unsupported feature mode: %s.', featureMode);
                            end
                            modelData.(sideName).(loadNames{li}) = X;
                        end
                    end
                    modelData.times.low = sourceData.timeBySetSize.(lowField);
                    modelData.times.high = sourceData.timeBySetSize.(highField);

                case {'nopca', 'pca'}
                    cdaModel = struct();
                    alphaModel = struct();
                    sourceList = {'cda', 'alpha'};
                    for srcIdx = 1:numel(sourceList)
                        sourceName = sourceList{srcIdx};
                        sourceData = savedData.(sourceName);
                        tmpModel = struct();

                        for si = 1:numel(sideNames)
                            sideName = sideNames{si};
                            for li = 1:numel(loadNames)
                                if strcmp(loadNames{li}, 'low')
                                    setSizeNow = seq.lowSetSize;
                                else
                                    setSizeNow = seq.highSetSize;
                                end

                                loadStr = num2str(setSizeNow);
                                leftName = sprintf('left_%s_%s', sideName, loadStr);
                                rightName = sprintf('right_%s_%s', sideName, loadStr);
                                if ~(isfield(sourceData.trial, leftName) && isfield(sourceData.trial, rightName))
                                    error('Saved data is missing %s/%s.', leftName, rightName);
                                end

                                leftX = sourceData.trial.(leftName);
                                rightX = sourceData.trial.(rightName);
                                if strcmpi(sideName, 'L')
                                    tmpModel.(sideName).(loadNames{li}) = rightX - leftX;
                                else
                                    tmpModel.(sideName).(loadNames{li}) = leftX - rightX;
                                end
                            end
                        end
                        tmpModel.times.low = sourceData.timeBySetSize.(lowField);
                        tmpModel.times.high = sourceData.timeBySetSize.(highField);

                        if strcmp(sourceName, 'cda')
                            cdaModel = tmpModel;
                        else
                            alphaModel = tmpModel;
                        end
                    end

                    for si = 1:numel(sideNames)
                        sideName = sideNames{si};
                        for li = 1:numel(loadNames)
                            sourceTime = alphaModel.times.(loadNames{li});
                            targetTime = cdaModel.times.(loadNames{li});
                            sourceTime = sourceTime(:)';
                            targetTime = targetTime(:)';
                            idx = zeros(1, numel(targetTime));

                            for ti = 1:numel(targetTime)
                                matchIdx = find(abs(sourceTime - targetTime(ti)) < 1e-9, 1);
                                if isempty(matchIdx)
                                    error('Cannot align source time axis to target time %.6g ms.', targetTime(ti));
                                end
                                idx(ti) = matchIdx;
                            end

                            alphaModel.(sideName).(loadNames{li}) = alphaModel.(sideName).(loadNames{li})(:,:,idx);
                        end
                    end

                    for si = 1:numel(sideNames)
                        sideName = sideNames{si};
                        for li = 1:numel(loadNames)
                            modelData.(sideName).(loadNames{li}) = cat(2, ...
                                cdaModel.(sideName).(loadNames{li}), ...
                                alphaModel.(sideName).(loadNames{li}));
                        end
                    end
                    modelData.times = cdaModel.times;

                    baseLabels = [savedData.cda.relPairLabels savedData.alpha.relPairLabels];
                    prefixes = [repmat({'CDA'}, 1, numel(savedData.cda.relPairLabels)), ...
                                repmat({'Alpha'}, 1, numel(savedData.alpha.relPairLabels))];
                    modelData.channelLabels = cell(size(baseLabels));
                    for labelIdx = 1:numel(baseLabels)
                        modelData.channelLabels{labelIdx} = sprintf('%s:%s', prefixes{labelIdx}, baseLabels{labelIdx});
                    end

                otherwise
                    error('Unsupported model name: %s.', modelName);
            end

            modelData.behavior = struct();
            for si = 1:numel(sideNames)
                sideName = sideNames{si};
                if strcmpi(sideName, 'L')
                    sideCode = -1;
                else
                    sideCode = 1;
                end

                for li = 1:numel(loadNames)
                    if strcmp(loadNames{li}, 'low')
                        setSizeNow = seq.lowSetSize;
                    else
                        setSizeNow = seq.highSetSize;
                    end

                    setField = sprintf('setsize%d', setSizeNow);
                    meta = savedData.cda.extraction.(setField);
                    if isfield(meta, 'cleanSetSizeTrialIndex')
                        cleanSetSizeTrialIndex = meta.cleanSetSizeTrialIndex(:);
                    else
                        cleanSetSizeTrialIndex = meta.Ikeep(meta.HEOG_in);
                        cleanSetSizeTrialIndex = cleanSetSizeTrialIndex(:);
                    end

                    behaviorSet = behavior(behavior.SetSize == setSizeNow, :);
                    behaviorClean = behaviorSet(cleanSetSizeTrialIndex, :);
                    behaviorSide = behaviorClean(behaviorClean.CueSide == sideCode, :);
                    if height(behaviorSide) ~= size(modelData.(sideName).(loadNames{li}), 1)
                        error('Subject %d %s setsize%d behavior rows (%d) do not match model rows (%d).', ...
                            subject, sideName, setSizeNow, height(behaviorSide), ...
                            size(modelData.(sideName).(loadNames{li}), 1));
                    end

                    modelData.behavior.(sideName).(loadNames{li}) = behaviorSide;
                end
            end

            sideResults = struct();
            for si = 1:numel(sideNames)
                sideName = sideNames{si};
                lowData = modelData.(sideName).low;
                highData = modelData.(sideName).high;
                lowTime = modelData.times.low(:)';
                highTime = modelData.times.high(:)';

                starts = seq.highSegmentStartsMs(:)';
                if isempty(starts)
                    starts = seq.highFirstSegmentStartMs + (0:(seq.highSetSize - 1)) * seq.highSegmentWidthMs;
                end

                lastNeededTime = starts(end) + seq.highSegmentWidthMs;
                if highTime(1) > starts(1) || highTime(end) < lastNeededTime
                    error('High-set-size time axis [%g %g] ms does not cover requested high segments [%g %g] ms.', ...
                        highTime(1), highTime(end), starts(1), lastNeededTime);
                end

                nLowSeg = size(seq.lowWindowsMs, 1);
                nHighSeg = numel(starts);
                AccBlocks = cell(nLowSeg, nHighSeg);
                AUCBlocks = cell(nLowSeg, nHighSeg);
                timesLowBySegment = cell(nLowSeg, 1);
                timesHighBySegment = cell(nHighSeg, 1);
                rawLowTimeBySegment = cell(nLowSeg, 1);
                AccTrainByPair = [];
                weightsByPair = [];
                decisionByPair = cell(nLowSeg, nHighSeg);
                highBehavior = modelData.behavior.(sideName).high;

                lowSegmentInfo = repmat(struct('index', [], 'lowWindowMs', [], 'lowTime', []), nLowSeg, 1);
                highSegmentInfo = repmat(struct('index', [], 'highWindowMs', [], 'highTime', []), nHighSeg, 1);

                for li = 1:nLowSeg
                    lowWindow = seq.lowWindowsMs(li,:);
                    lowTimeIdx = lowTime >= lowWindow(1) & lowTime <= lowWindow(2);
                    if ~any(lowTimeIdx)
                        error('Requested time window [%g %g] ms does not overlap data time axis [%g %g] ms.', ...
                            lowWindow(1), lowWindow(2), lowTime(1), lowTime(end));
                    end
                    lowWin = lowData(:,:,lowTimeIdx);
                    lowWinTime = lowTime(lowTimeIdx);
                    relativeTime = lowWinTime - lowWindow(1);

                    lowSegmentInfo(li).index = li;
                    lowSegmentInfo(li).lowWindowMs = lowWindow;
                    lowSegmentInfo(li).lowTime = lowWinTime(:);
                    rawLowTimeBySegment{li} = lowWinTime(:);

                    for hi = 1:nHighSeg
                        highWindow = [starts(hi), starts(hi) + seq.highSegmentWidthMs];
                        highTimeIdx = highTime >= highWindow(1) & highTime <= highWindow(2);
                        if ~any(highTimeIdx)
                            error('Requested time window [%g %g] ms does not overlap data time axis [%g %g] ms.', ...
                                highWindow(1), highWindow(2), highTime(1), highTime(end));
                        end
                        highSeg = highData(:,:,highTimeIdx);
                        highSegTime = highTime(highTimeIdx);

                        if size(lowWin, 3) ~= size(highSeg, 3)
                            error('Low segment and high segment have different sample counts for side %s, low segment %d, high segment %d.', ...
                                sideName, li, hi);
                        end

                        labels = [seq.labelLow * ones(size(lowWin,1), 1); ...
                                  seq.labelHigh * ones(size(highSeg,1), 1)];
                        data = cat(1, lowWin, highSeg);
                        data = permute(data, [2 3 1]);

                        segResult = LDA_function_singleSubj(data, labels, relativeTime, cfgModel);
                        relOutTime = segResult.times(:);

                        AccBlocks{li,hi} = segResult.Acc;
                        if isfield(segResult, 'AUC')
                            AUCBlocks{li,hi} = segResult.AUC;
                        end

                        timesLowBySegment{li} = relOutTime + lowWindow(1);
                        timesHighBySegment{hi} = relOutTime + starts(hi);
                        AccTrainByPair(:,li,hi) = segResult.AccTrain(:); 
                        weightsByPair(:,:,li,hi) = segResult.weights; 

                        if isfield(segResult, 'decisionValues')
                            highRows = size(lowWin, 1) + (1:size(highSeg, 1));
                            decisionInfo = struct();
                            decisionInfo.scoreHighState = segResult.decisionValues.scoreClass2(highRows,:,:);
                            decisionInfo.nTest = segResult.decisionValues.nTest(highRows,:,:);
                            decisionInfo.highBehavior = highBehavior;
                            decisionInfo.lowSegment = li;
                            decisionInfo.highSegment = hi;
                            decisionInfo.relativeTimes = relOutTime(:);
                            decisionInfo.trainTimes = relOutTime(:) + lowWindow(1);
                            decisionInfo.testTimes = relOutTime(:) + starts(hi);
                            decisionInfo.scoreMeaning = sprintf( ...
                                'score/posterior for setsize%d label (%d), i.e. high-load state evidence', ...
                                seq.highSetSize, seq.labelHigh);
                            decisionInfo.dimensions = 'high set-size trial x trainTime x testTime';
                            decisionByPair{li,hi} = decisionInfo;
                        end

                        if li == 1
                            highSegmentInfo(hi).index = hi;
                            highSegmentInfo(hi).highWindowMs = highWindow;
                            highSegmentInfo(hi).highTime = highSegTime(:);
                        end
                    end
                end

                resSide = struct();
                resSide.Acc = cell2mat(AccBlocks);
                if ~isempty(AUCBlocks{1,1})
                    resSide.AUC = cell2mat(AUCBlocks);
                end
                resSide.AccTrainByPair = AccTrainByPair;
                resSide.weightsByPair = weightsByPair;
                resSide.decisionByPair = decisionByPair;
                resSide.timesLow = vertcat(timesLowBySegment{:});
                resSide.timesHigh = vertcat(timesHighBySegment{:});
                resSide.rawLowTime = vertcat(rawLowTimeBySegment{:});
                resSide.lowSegmentInfo = lowSegmentInfo;
                resSide.highSegmentInfo = highSegmentInfo;
                resSide.cfg = cfgModel;
                resSide.temporalDesign = struct( ...
                    'analysisMode', seq.analysisMode, ...
                    'segmentDescription', seq.segmentDescription, ...
                    'lowSetSize', seq.lowSetSize, ...
                    'highSetSize', seq.highSetSize, ...
                    'lowWindowsMs', seq.lowWindowsMs, ...
                    'highSegmentWidthMs', seq.highSegmentWidthMs, ...
                    'highSegmentStartsMs', starts(:)', ...
                    'segmentTailPolicy', seq.segmentTailPolicy, ...
                    'matrixRows', 'concatenated low-set-size binned segment times', ...
                    'matrixColumns', 'concatenated high-set-size binned segment times');

                sideResults.(sideName) = resSide;
            end

            Decode = sideResults.L;
            fieldsNow = fieldnames(sideResults.L);
            for fi = 1:numel(fieldsNow)
                f = fieldsNow{fi};
                if isfield(sideResults.R, f) && isnumeric(sideResults.L.(f)) && ...
                        isnumeric(sideResults.R.(f)) && isequal(size(sideResults.L.(f)), size(sideResults.R.(f)))
                    dim = ndims(sideResults.L.(f)) + 1;
                    Decode.(f) = mean(cat(dim, sideResults.L.(f), sideResults.R.(f)), dim, 'omitnan');
                end
            end
            if isfield(Decode, 'decisionByPair')
                Decode = rmfield(Decode, 'decisionByPair');
            end
            if isfield(Decode, 'cfg')
                Decode.cfg.withinSideAverage = true;
            end

            Decode.modelName = modelName;
            Decode.labelMeaning = sprintf('setsize%d trials label=%d; setsize%d trials label=%d', ...
                seq.lowSetSize, seq.labelLow, seq.highSetSize, seq.labelHigh);
            Decode.chanceAccuracy = seq.chanceAccuracy;
            Decode.shuffle = struct('enabled', false, ...
                'reason', 'Disabled for this large data3 sequential decoding; compare later against theoretical chance accuracy.');
            Decode.withinSide = struct();
            Decode.withinSide.description = 'Decoding was run separately within attended-left and attended-right trials, then averaged across sides.';
            Decode.withinSide.leftCountsLowHigh = [size(modelData.L.low,1), size(modelData.L.high,1)];
            Decode.withinSide.rightCountsLowHigh = [size(modelData.R.low,1), size(modelData.R.high,1)];
            Decode.withinSide.averageMode = 'unweighted mean of attended-left and attended-right decoding results';
            keep = {'Acc', 'AUC', 'AccTrainByPair', 'weightsByPair', ...
                    'decisionByPair', ...
                    'timesLow', 'timesHigh', 'rawLowTime', ...
                    'lowSegmentInfo', 'highSegmentInfo', 'temporalDesign'};
            sideLeft = struct();
            sideRight = struct();
            for ki = 1:numel(keep)
                f = keep{ki};
                if isfield(sideResults.L, f)
                    sideLeft.(f) = sideResults.L.(f);
                end
                if isfield(sideResults.R, f)
                    sideRight.(f) = sideResults.R.(f);
                end
            end
            Decode.side = struct('Left', sideLeft, 'Right', sideRight);

            Decode.subject = subject;
            Decode.comparisonName = comparisons(ci).name;
            Decode.channelLabels = modelData.channelLabels;
            Decode.inputSource = inFile;
            if isfield(savedData.cda, 'extraction')
                Decode.extraction = savedData.cda.extraction;
            end

            out = struct();
            out.(modelName) = Decode;
            save(fullfile(comparisons(ci).outputdir, modelName, file), '-struct', 'out', '-v7.3');
        end
    end
end

for ci = 1:numel(comparisons)
    runInfo = struct('cfg', cfg, 'seq', comparisons(ci).seq, 'modelNames', {modelNames});
    save(fullfile(comparisons(ci).outputdir, 'run_cfg.mat'), '-struct', 'runInfo', '-v7.3');
end
fprintf('data3 sequential LDA finished. Results saved under:\n%s\n', dataDir);

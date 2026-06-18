%% data3 spatial-matched sequential Alpha/CDA LDA decoding
% This script reruns data3 set-size-3 vs set-size-6 sequential decoding after
% matching trials on spatial geometry. It reads the first-memory-locked
% cda/alpha files saved by data3_cda_alpha.m and writes a separate result
% folder so the original decoding outputs are not overwritten.
%
% Main use:
%   1) Set cfg.controlMode = 'currentLocList' to control current item
%      position while keeping the full low x high segment matrix.
%   2) Set cfg.controlMode = 'cumMaxDistBin' to strictly match cumulative
%      spatial range. Cells with no spatial overlap are left as NaN and are
%      documented in the match-summary CSV.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
datadir = fullfile(dataDir, 'cda_alpha');
behaviorDir = fullfile(dataDir, 'Behavior_data_script');

addpath(codeDir);

%% Analysis configuration
cfg = struct();
cfg.analysisModeName = 'maintOnly';       % 'encMaint' or 'maintOnly'
cfg.lowSetSize = 3;
cfg.highSetSize = 6;
cfg.modelNames = {'Alpha', 'CDA'};        % Alpha is the target; CDA is a positive-control comparison

% Spatial control modes:
%   'currentLocList' = match current item location code and location-list code.
%   'cumMaxDistBin' = strict overlap matching on cumulative max pairwise distance.
cfg.controlMode = 'currentLocList';
cfg.cumulativeMetric = 'CumMaxDist';
cfg.nSpatialBins = 3;
cfg.minFactorCellN = 2;
cfg.minBalancedTrialsPerClass = 10;
cfg.outputTag = strtrim(char(getenv('DATA3_SPATIAL_OUTPUT_TAG')));

controlModeOverride = strtrim(char(getenv('DATA3_SPATIAL_CONTROL_MODE')));
if ~isempty(controlModeOverride)
    cfg.controlMode = controlModeOverride;
end

modelsOverride = strtrim(char(getenv('DATA3_SPATIAL_MODELS')));
if ~isempty(modelsOverride)
    cfg.modelNames = regexp(modelsOverride, '[,;\s]+', 'split');
    cfg.modelNames = cfg.modelNames(~cellfun('isempty', cfg.modelNames));
end

cfg.makeGroupStats = true;
cfg.makeGroupFigures = true;
cfg.figureDpi = 300;
cfg.figureVisible = 'off';
cfg.colorLimits = [];
cfg.colorLimitBounds = [0 1];
cfg.metric = 'AUC';
cfg.chance = 0.5;
cfg.nPerm2D = 1000;
cfg.clusterAlpha = 0.05;
cfg.alpha = 0.05;
cfg.randomSeed = 20260610;

nPermOverride = str2double(strtrim(char(getenv('DATA3_SPATIAL_NPERM'))));
if ~isnan(nPermOverride) && nPermOverride > 0
    cfg.nPerm2D = nPermOverride;
end

makeStatsOverride = strtrim(char(getenv('DATA3_SPATIAL_MAKE_STATS')));
if ~isempty(makeStatsOverride)
    cfg.makeGroupStats = ~ismember(lower(makeStatsOverride), {'0', 'false', 'no'});
end

makeFiguresOverride = strtrim(char(getenv('DATA3_SPATIAL_MAKE_FIGURES')));
if ~isempty(makeFiguresOverride)
    cfg.makeGroupFigures = ~ismember(lower(makeFiguresOverride), {'0', 'false', 'no'});
end

stimStepMs = 460;
encMaintWidthMs = 460;
maintWidthMs = 360;

switch lower(cfg.analysisModeName)
    case 'encmaint'
        segmentStartOffsetMs = 0;
        segmentWidthMs = encMaintWidthMs;
        segmentDescription = 'encoding + maintenance';
    case 'maintonly'
        segmentStartOffsetMs = 100;
        segmentWidthMs = maintWidthMs;
        segmentDescription = 'maintenance only';
    otherwise
        error('Unsupported analysisModeName: %s.', cfg.analysisModeName);
end

lowItemOnsetsMs = (0:(cfg.lowSetSize - 1)) * stimStepMs;
highItemOnsetsMs = (0:(cfg.highSetSize - 1)) * stimStepMs;

seq = struct();
seq.lowSetSize = cfg.lowSetSize;
seq.highSetSize = cfg.highSetSize;
seq.lowWindowsMs = [lowItemOnsetsMs(:) + segmentStartOffsetMs, ...
                    lowItemOnsetsMs(:) + segmentStartOffsetMs + segmentWidthMs];
seq.highSegmentStartsMs = highItemOnsetsMs + segmentStartOffsetMs;
seq.highSegmentWidthMs = segmentWidthMs;
seq.labelLow = 1;
seq.labelHigh = 2;
seq.chanceAccuracy = 0.5;
seq.analysisMode = cfg.analysisModeName;
seq.segmentDescription = segmentDescription;
seq.segmentTailPolicy = sprintf( ...
    'Use exactly one %d-ms %s segment per high-set-size item; post-final delay is not decoded.', ...
    segmentWidthMs, segmentDescription);

comparisonName = sprintf('setsize%d_vs%d_%s_spatialMatched_%s', ...
    cfg.lowSetSize, cfg.highSetSize, cfg.analysisModeName, cfg.controlMode);
outputdir = fullfile(dataDir, sprintf('decoding_LDA_setsize%d_vs%d_segments_%s_spatialMatched_%s', ...
    cfg.lowSetSize, cfg.highSetSize, cfg.analysisModeName, cfg.controlMode));
if ~isempty(cfg.outputTag)
    safeTag = regexprep(cfg.outputTag, '[^\w-]', '_');
    comparisonName = sprintf('%s_%s', comparisonName, safeTag);
    outputdir = sprintf('%s_%s', outputdir, safeTag);
end
if ~isfolder(outputdir), mkdir(outputdir); end
for mi = 1:numel(cfg.modelNames)
    modelDir = fullfile(outputdir, cfg.modelNames{mi});
    if ~isfolder(modelDir), mkdir(modelDir); end
end

%% LDA configuration: follows data3_setsize1_vs6_LDA_decoding.m
cfgLDA = struct();
cfgLDA.cvType = 'kfold';
cfgLDA.trainRatio = 2/3;
cfgLDA.nFolds = 5;
cfgLDA.superTrial = 1;
cfgLDA.nIter = 50;

cfgLDA.smooth_window = 50;
cfgLDA.smooth_step = 50;
cfgLDA.timeWindowMode = 'bin';

cfgLDA.analysisWindow = [-200 inf];
cfgLDA.doTimeGeneralization = true;
cfgLDA.doPCA = false;
cfgLDA.nPCs = 5;

cfgLDA.discrimType = 'diagLinear';
cfgLDA.ldaEngine = 'fitcdiscr';
cfgLDA.standardize = 1;

cfgLDA.doShuffle = false;
cfgLDA.balanceTrials = true;
cfgLDA.balanceNPerCell = [];
cfgLDA.balanceFactors = [];
cfgLDA.useAUC = 1;
cfgLDA.returnDecisionValues = false;
cfgLDA.useParallel = true;
cfgLDA.verbose = false;
cfgLDA.randomSeed = cfg.randomSeed;

nIterOverride = str2double(strtrim(char(getenv('DATA3_SPATIAL_NITER'))));
if ~isnan(nIterOverride) && nIterOverride > 0
    cfgLDA.nIter = nIterOverride;
end

files = dir(fullfile(datadir, 'sub*.mat'));
files = data3_filter_subject_mat_files(files, data3_subject_filter());
if isempty(files)
    error('No sub*.mat files found in %s.', datadir);
end

%% Rebuild task location coordinates from the behavior task script
numLocs = 6;
stimDists = [155, 250, 350];
minAng = 60;
maxAng = 120;
spacing = (maxAng - minAng) / (numLocs / numel(stimDists));
stimAngs = minAng:spacing:maxAng;

xPosList = zeros(2, numLocs);
yPosList = zeros(2, numLocs);
for di = 1:numel(stimDists)
    for ai = 1:numel(stimAngs)
        off1 = (mod(di, 2) - 0.5) * spacing;
        off2 = (mod(di - 1, 2) - 0.5) * spacing;
        col = (di - 1) * numel(stimAngs) + ai;
        xPosList(1, col) = sind(stimAngs(ai) + off1) * stimDists(di);
        yPosList(1, col) = cosd(stimAngs(ai) + off1) * stimDists(di);
        xPosList(2, col) = sind(stimAngs(ai) + off2) * stimDists(di);
        yPosList(2, col) = cosd(stimAngs(ai) + off2) * stimDists(di);
    end
end

sideNames = {'L', 'R'};
loadNames = {'low', 'high'};
subjectMatchTables = {};

%% Subject-level spatial-matched decoding
for sf = 1:numel(files)
    file = files(sf).name;
    inFile = fullfile(files(sf).folder, file);
    fprintf('data3 spatial-matched LDA [%s]: %s\n', cfg.controlMode, file);

    S = load(inFile, 'cda', 'alpha');
    if ~(isfield(S, 'cda') && isfield(S, 'alpha'))
        error('%s does not contain cda and alpha. Rerun data3_cda_alpha.', inFile);
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

    %% Read behavior blocks 1:4 and append spatial geometry columns
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
        blockBehavior.MemoryType = behData.mem_lets(:);
        blockBehavior.SetSize = behData.set_size(:);
        blockBehavior.Change = behData.change(:);
        blockBehavior.CueSide = behData.cue_side(:);
        blockBehavior.SerialPosition = behData.change_item(:);
        blockBehavior.UncuedSerialPosition = behData.change_item_uncued(:);
        blockBehavior.Response = behData.resp(:);
        blockBehavior.RT = behData.rt(:);
        blockBehavior.Correct = behData.acc(:);
        blockBehavior.LocList = behData.loc_list_cued(:);

        locs = behData.locs;
        if size(locs, 2) < cfg.highSetSize
            error('%s has fewer than %d location columns.', behaviorFile, cfg.highSetSize);
        end

        for pos = 1:cfg.highSetSize
            blockBehavior.(sprintf('CurrentLoc_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('CurrentRing_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('CurrentRadius_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('CurrentAngle_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('CumHullArea_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('CumMaxDist_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('CumMeanDist_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('NearestPriorDist_Pos%d', pos)) = nan(nBehTrial, 1);
            blockBehavior.(sprintf('AngleSpan_Pos%d', pos)) = nan(nBehTrial, 1);
        end

        for ti = 1:nBehTrial
            setSizeNow = blockBehavior.SetSize(ti);
            locListNow = blockBehavior.LocList(ti);
            if isnan(locListNow) || locListNow < 1 || locListNow > 2
                continue;
            end

            coords = nan(setSizeNow, 2);
            for pos = 1:setSizeNow
                locNow = locs(ti, pos);
                if locNow < 1 || locNow > numLocs
                    continue;
                end

                xNow = xPosList(locListNow, locNow);
                yNow = yPosList(locListNow, locNow);
                coords(pos,:) = [xNow, yNow];

                blockBehavior.(sprintf('CurrentLoc_Pos%d', pos))(ti) = locNow;
                blockBehavior.(sprintf('CurrentRing_Pos%d', pos))(ti) = ceil(locNow / numel(stimAngs));
                blockBehavior.(sprintf('CurrentRadius_Pos%d', pos))(ti) = hypot(xNow, yNow);
                angleNow = atan2d(yNow, xNow);
                if angleNow < 0, angleNow = angleNow + 360; end
                blockBehavior.(sprintf('CurrentAngle_Pos%d', pos))(ti) = angleNow;

                validCoords = coords(1:pos,:);
                validCoords = validCoords(all(~isnan(validCoords), 2), :);
                if isempty(validCoords)
                    continue;
                end

                if size(validCoords, 1) >= 3
                    try
                        hullIdx = convhull(validCoords(:,1), validCoords(:,2));
                        blockBehavior.(sprintf('CumHullArea_Pos%d', pos))(ti) = ...
                            polyarea(validCoords(hullIdx,1), validCoords(hullIdx,2));
                    catch
                        blockBehavior.(sprintf('CumHullArea_Pos%d', pos))(ti) = 0;
                    end
                else
                    blockBehavior.(sprintf('CumHullArea_Pos%d', pos))(ti) = 0;
                end

                pairDists = [];
                for a = 1:size(validCoords, 1)
                    for b = (a + 1):size(validCoords, 1)
                        pairDists(end+1, 1) = hypot(validCoords(a,1) - validCoords(b,1), ...
                            validCoords(a,2) - validCoords(b,2)); %#ok<SAGROW>
                    end
                end
                if isempty(pairDists)
                    blockBehavior.(sprintf('CumMaxDist_Pos%d', pos))(ti) = 0;
                    blockBehavior.(sprintf('CumMeanDist_Pos%d', pos))(ti) = 0;
                    blockBehavior.(sprintf('NearestPriorDist_Pos%d', pos))(ti) = NaN;
                else
                    blockBehavior.(sprintf('CumMaxDist_Pos%d', pos))(ti) = max(pairDists);
                    blockBehavior.(sprintf('CumMeanDist_Pos%d', pos))(ti) = mean(pairDists, 'omitnan');
                    if pos > 1
                        priorCoords = validCoords(1:(end - 1), :);
                        currentCoord = validCoords(end, :);
                        blockBehavior.(sprintf('NearestPriorDist_Pos%d', pos))(ti) = ...
                            min(hypot(priorCoords(:,1) - currentCoord(1), priorCoords(:,2) - currentCoord(2)));
                    end
                end

                angles = atan2d(validCoords(:,2), validCoords(:,1));
                angles(angles < 0) = angles(angles < 0) + 360;
                if numel(angles) <= 1
                    blockBehavior.(sprintf('AngleSpan_Pos%d', pos))(ti) = 0;
                else
                    angles = sort(angles(:));
                    gaps = diff([angles; angles(1) + 360]);
                    blockBehavior.(sprintf('AngleSpan_Pos%d', pos))(ti) = 360 - max(gaps);
                end
            end
        end

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

    %% Build model data and aligned clean behavior rows
    for mi = 1:numel(cfg.modelNames)
        modelName = cfg.modelNames{mi};
        modelData = struct();

        switch lower(modelName)
            case 'alpha'
                sourceData = savedData.alpha;
                modelData.channelLabels = savedData.alpha.relPairLabels;
            case 'cda'
                sourceData = savedData.cda;
                modelData.channelLabels = savedData.cda.relPairLabels;
            otherwise
                error('This script currently supports Alpha and CDA only. Requested: %s.', modelName);
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
                if strcmpi(sideName, 'L')
                    modelData.(sideName).(loadNames{li}) = rightX - leftX;
                else
                    modelData.(sideName).(loadNames{li}) = leftX - rightX;
                end
            end
        end

        lowField = sprintf('setsize%d', seq.lowSetSize);
        highField = sprintf('setsize%d', seq.highSetSize);
        modelData.times.low = sourceData.timeBySetSize.(lowField);
        modelData.times.high = sourceData.timeBySetSize.(highField);

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

        %% Sequential decoding with spatial matching before each LDA call
        sideResults = struct();
        for si = 1:numel(sideNames)
            sideName = sideNames{si};
            lowData = modelData.(sideName).low;
            highData = modelData.(sideName).high;
            lowTime = modelData.times.low(:)';
            highTime = modelData.times.high(:)';
            lowBehavior = modelData.behavior.(sideName).low;
            highBehavior = modelData.behavior.(sideName).high;

            starts = seq.highSegmentStartsMs(:)';
            nLowSeg = size(seq.lowWindowsMs, 1);
            nHighSeg = numel(starts);

            AccBlocks = cell(nLowSeg, nHighSeg);
            AUCBlocks = cell(nLowSeg, nHighSeg);
            timesLowBySegment = cell(nLowSeg, 1);
            timesHighBySegment = cell(nHighSeg, 1);
            rawLowTimeBySegment = cell(nLowSeg, 1);
            AccTrainByPair = [];
            weightsByPair = [];
            matchRows = {};

            lowSegmentInfo = repmat(struct('index', [], 'lowWindowMs', [], 'lowTime', []), nLowSeg, 1);
            highSegmentInfo = repmat(struct('index', [], 'highWindowMs', [], 'highTime', []), nHighSeg, 1);

            for li = 1:nLowSeg
                lowWindow = seq.lowWindowsMs(li,:);
                lowTimeIdx = lowTime >= lowWindow(1) & lowTime <= lowWindow(2);
                if ~any(lowTimeIdx)
                    error('Requested low window [%g %g] ms does not overlap data time axis [%g %g] ms.', ...
                        lowWindow(1), lowWindow(2), lowTime(1), lowTime(end));
                end
                lowWinAll = lowData(:,:,lowTimeIdx);
                lowWinTime = lowTime(lowTimeIdx);
                relativeTime = lowWinTime - lowWindow(1);
                binStarts = relativeTime(1):cfgLDA.smooth_step:(relativeTime(end) - cfgLDA.smooth_window);
                relOutTimeExpected = binStarts + cfgLDA.smooth_window / 2;
                nOutTime = numel(relOutTimeExpected);

                lowSegmentInfo(li).index = li;
                lowSegmentInfo(li).lowWindowMs = lowWindow;
                lowSegmentInfo(li).lowTime = lowWinTime(:);
                rawLowTimeBySegment{li} = lowWinTime(:);

                for hi = 1:nHighSeg
                    highWindow = [starts(hi), starts(hi) + seq.highSegmentWidthMs];
                    highTimeIdx = highTime >= highWindow(1) & highTime <= highWindow(2);
                    if ~any(highTimeIdx)
                        error('Requested high window [%g %g] ms does not overlap data time axis [%g %g] ms.', ...
                            highWindow(1), highWindow(2), highTime(1), highTime(end));
                    end
                    highSegAll = highData(:,:,highTimeIdx);
                    highSegTime = highTime(highTimeIdx);

                    if size(lowWinAll, 3) ~= size(highSegAll, 3)
                        error('Low and high segment sample counts differ for side %s, low %d, high %d.', ...
                            sideName, li, hi);
                    end

                    lowMetricName = sprintf('%s_Pos%d', cfg.cumulativeMetric, li);
                    highMetricName = sprintf('%s_Pos%d', cfg.cumulativeMetric, hi);
                    lowMetricBefore = lowBehavior.(lowMetricName);
                    highMetricBefore = highBehavior.(highMetricName);

                    skipReason = '';
                    switch lower(cfg.controlMode)
                        case 'currentloclist'
                            lowFactor = [lowBehavior.LocList, lowBehavior.(sprintf('CurrentLoc_Pos%d', li))];
                            highFactor = [highBehavior.LocList, highBehavior.(sprintf('CurrentLoc_Pos%d', hi))];

                        case 'cummaxdistbin'
                            lowMetric = lowMetricBefore;
                            highMetric = highMetricBefore;
                            overlapMin = max(min(lowMetric, [], 'omitnan'), min(highMetric, [], 'omitnan'));
                            overlapMax = min(max(lowMetric, [], 'omitnan'), max(highMetric, [], 'omitnan'));

                            if isnan(overlapMin) || isnan(overlapMax)
                                skipReason = 'missingSpatialMetric';
                                lowFactor = nan(numel(lowMetric), 1);
                                highFactor = nan(numel(highMetric), 1);
                            elseif abs(overlapMax - overlapMin) < 1e-9
                                keepLowRange = abs(lowMetric - overlapMin) < 1e-9;
                                keepHighRange = abs(highMetric - overlapMin) < 1e-9;
                                lowFactor = nan(numel(lowMetric), 1);
                                highFactor = nan(numel(highMetric), 1);
                                lowFactor(keepLowRange) = 1;
                                highFactor(keepHighRange) = 1;
                            elseif overlapMax > overlapMin
                                keepLowRange = lowMetric >= overlapMin & lowMetric <= overlapMax;
                                keepHighRange = highMetric >= overlapMin & highMetric <= overlapMax;
                                edges = linspace(overlapMin, overlapMax, cfg.nSpatialBins + 1);
                                edges(1) = -inf;
                                edges(end) = inf;
                                lowFactor = discretize(lowMetric, edges);
                                highFactor = discretize(highMetric, edges);
                                lowFactor(~keepLowRange) = NaN;
                                highFactor(~keepHighRange) = NaN;
                            else
                                skipReason = 'noCumulativeSpatialOverlap';
                                lowFactor = nan(numel(lowMetric), 1);
                                highFactor = nan(numel(highMetric), 1);
                            end

                        otherwise
                            error('Unsupported cfg.controlMode: %s.', cfg.controlMode);
                    end

                    validLow = all(~isnan(lowFactor), 2);
                    validHigh = all(~isnan(highFactor), 2);
                    commonFactor = intersect(lowFactor(validLow,:), highFactor(validHigh,:), 'rows');

                    keepLow = false(size(lowFactor, 1), 1);
                    keepHigh = false(size(highFactor, 1), 1);
                    if ~isempty(commonFactor)
                        keepLow = validLow & ismember(lowFactor, commonFactor, 'rows');
                        keepHigh = validHigh & ismember(highFactor, commonFactor, 'rows');
                    end

                    nCommonFactor = size(commonFactor, 1);
                    minFactorCellN = 0;
                    nBalancedPerClass = 0;
                    if nCommonFactor > 0
                        countsLow = zeros(nCommonFactor, 1);
                        countsHigh = zeros(nCommonFactor, 1);
                        for fi = 1:nCommonFactor
                            countsLow(fi) = sum(ismember(lowFactor, commonFactor(fi,:), 'rows'));
                            countsHigh(fi) = sum(ismember(highFactor, commonFactor(fi,:), 'rows'));
                        end
                        minFactorCellN = min([countsLow(:); countsHigh(:)]);
                        nBalancedPerClass = nCommonFactor * minFactorCellN;
                    end

                    runThisPair = isempty(skipReason) && nCommonFactor > 0 && ...
                        minFactorCellN >= cfg.minFactorCellN && ...
                        nBalancedPerClass >= max(cfg.minBalancedTrialsPerClass, cfgLDA.nFolds);

                    if runThisPair
                        lowWin = lowWinAll(keepLow,:,:);
                        highSeg = highSegAll(keepHigh,:,:);
                        labels = [seq.labelLow * ones(size(lowWin, 1), 1); ...
                                  seq.labelHigh * ones(size(highSeg, 1), 1)];
                        dataNow = cat(1, lowWin, highSeg);
                        dataNow = permute(dataNow, [2 3 1]);

                        cfgPair = cfgLDA;
                        cfgPair.balanceFactors = [lowFactor(keepLow,:); highFactor(keepHigh,:)];
                        cfgPair.randomSeed = cfg.randomSeed + subject * 1000 + si * 100 + li * 10 + hi;

                        segResult = LDA_function_singleSubj(dataNow, labels, relativeTime, cfgPair);
                        relOutTime = segResult.times(:);

                        AccBlocks{li,hi} = segResult.Acc;
                        if isfield(segResult, 'AUC')
                            AUCBlocks{li,hi} = segResult.AUC;
                        else
                            AUCBlocks{li,hi} = [];
                        end

                        AccTrainByPair(:,li,hi) = segResult.AccTrain(:);
                        weightsByPair(:,:,li,hi) = segResult.weights;
                    else
                        if isempty(skipReason)
                            skipReason = 'insufficientMatchedTrials';
                        end
                        relOutTime = relOutTimeExpected(:);
                        AccBlocks{li,hi} = nan(nOutTime, nOutTime);
                        AUCBlocks{li,hi} = nan(nOutTime, nOutTime);
                        AccTrainByPair(:,li,hi) = nan(nOutTime, 1);
                        weightsByPair(:,:,li,hi) = nan(size(lowWinAll, 2), nOutTime);
                    end

                    timesLowBySegment{li} = relOutTime(:) + lowWindow(1);
                    timesHighBySegment{hi} = relOutTime(:) + starts(hi);

                    if li == 1
                        highSegmentInfo(hi).index = hi;
                        highSegmentInfo(hi).highWindowMs = highWindow;
                        highSegmentInfo(hi).highTime = highSegTime(:);
                    end

                    meanLowMetricAfter = mean(lowMetricBefore(keepLow), 'omitnan');
                    meanHighMetricAfter = mean(highMetricBefore(keepHigh), 'omitnan');

                    M = table();
                    M.Subject = subject;
                    M.Model = {modelName};
                    M.Side = {sideName};
                    M.ControlMode = {cfg.controlMode};
                    M.LowSerialPosition = li;
                    M.HighSerialPosition = hi;
                    M.LowWindowStartMs = lowWindow(1);
                    M.LowWindowEndMs = lowWindow(2);
                    M.HighWindowStartMs = highWindow(1);
                    M.HighWindowEndMs = highWindow(2);
                    M.NLowBefore = size(lowWinAll, 1);
                    M.NHighBefore = size(highSegAll, 1);
                    M.NLowAfter = sum(keepLow);
                    M.NHighAfter = sum(keepHigh);
                    M.NCommonFactorCells = nCommonFactor;
                    M.MinFactorCellN = minFactorCellN;
                    M.NBalancedPerClass = nBalancedPerClass;
                    M.MeanLowCumMaxDistBefore = mean(lowMetricBefore, 'omitnan');
                    M.MeanHighCumMaxDistBefore = mean(highMetricBefore, 'omitnan');
                    M.MeanDiffHighMinusLowBefore = M.MeanHighCumMaxDistBefore - M.MeanLowCumMaxDistBefore;
                    M.MeanLowCumMaxDistAfter = meanLowMetricAfter;
                    M.MeanHighCumMaxDistAfter = meanHighMetricAfter;
                    M.MeanDiffHighMinusLowAfter = M.MeanHighCumMaxDistAfter - M.MeanLowCumMaxDistAfter;
                    M.RunThisPair = runThisPair;
                    M.SkipReason = {skipReason};
                    matchRows{end+1, 1} = M; %#ok<SAGROW>
                end
            end

            resSide = struct();
            resSide.Acc = cell2mat(AccBlocks);
            if ~isempty(AUCBlocks{1,1})
                resSide.AUC = cell2mat(AUCBlocks);
            end
            resSide.AccTrainByPair = AccTrainByPair;
            resSide.weightsByPair = weightsByPair;
            resSide.timesLow = vertcat(timesLowBySegment{:});
            resSide.timesHigh = vertcat(timesHighBySegment{:});
            resSide.rawLowTime = vertcat(rawLowTimeBySegment{:});
            resSide.lowSegmentInfo = lowSegmentInfo;
            resSide.highSegmentInfo = highSegmentInfo;
            resSide.matchSummary = vertcat(matchRows{:});
            resSide.cfg = cfgLDA;
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
        Decode.matchSummary = [sideResults.L.matchSummary; sideResults.R.matchSummary];
        subjectMatchTables{end+1, 1} = Decode.matchSummary; %#ok<SAGROW>

        if isfield(Decode, 'cfg')
            Decode.cfg.withinSideAverage = true;
        end

        Decode.modelName = modelName;
        Decode.labelMeaning = sprintf('setsize%d trials label=%d; setsize%d trials label=%d', ...
            seq.lowSetSize, seq.labelLow, seq.highSetSize, seq.labelHigh);
        Decode.chanceAccuracy = seq.chanceAccuracy;
        Decode.shuffle = struct('enabled', false, ...
            'reason', 'Disabled for this spatial-control sequential decoding; compare against theoretical chance accuracy.');
        Decode.spatialControl = struct();
        Decode.spatialControl.controlMode = cfg.controlMode;
        Decode.spatialControl.cumulativeMetric = cfg.cumulativeMetric;
        Decode.spatialControl.nSpatialBins = cfg.nSpatialBins;
        Decode.spatialControl.minFactorCellN = cfg.minFactorCellN;
        Decode.spatialControl.minBalancedTrialsPerClass = cfg.minBalancedTrialsPerClass;
        Decode.spatialControl.description = ['Spatial factors are matched before LDA within each side x low-segment x high-segment pair. ' ...
            'The decoder then balances label x spatial-factor cells using LDA_function_singleSubj.'];
        Decode.withinSide = struct();
        Decode.withinSide.description = 'Decoding was run separately within attended-left and attended-right trials, then averaged across sides.';
        Decode.withinSide.averageMode = 'unweighted mean of attended-left and attended-right decoding results';
        Decode.side = struct('Left', sideResults.L, 'Right', sideResults.R);

        Decode.subject = subject;
        Decode.comparisonName = comparisonName;
        Decode.channelLabels = modelData.channelLabels;
        Decode.inputSource = inFile;
        if isfield(savedData.cda, 'extraction')
            Decode.extraction = savedData.cda.extraction;
        end

        out = struct();
        out.(modelName) = Decode;
        save(fullfile(outputdir, modelName, file), '-struct', 'out', '-v7.3');
    end
end

if ~isempty(subjectMatchTables)
    allMatchSummary = vertcat(subjectMatchTables{:});
    writetable(allMatchSummary, fullfile(outputdir, sprintf('%s_match_summary.csv', comparisonName)));
end

%% Minimal group cluster statistics for the new output folder
if cfg.makeGroupStats
    groupDir = fullfile(outputdir, 'GroupStats');
    if ~isfolder(groupDir), mkdir(groupDir); end
    if cfg.makeGroupFigures
        figDir = fullfile(groupDir, 'figures');
        if ~isfolder(figDir), mkdir(figDir); end
    end
    clusterTables = {};

    for mi = 1:numel(cfg.modelNames)
        modelName = cfg.modelNames{mi};
        modelDir = fullfile(outputdir, modelName);
        modelFiles = dir(fullfile(modelDir, 'sub*.mat'));
        dataAll = [];
        usedFiles = {};
        subjects = [];
        timesLow = [];
        timesHigh = [];
        design = [];

        for fi = 1:numel(modelFiles)
            fpath = fullfile(modelFiles(fi).folder, modelFiles(fi).name);
            S = load(fpath, modelName);
            if ~isfield(S, modelName), continue; end
            R = S.(modelName);
            if ~isfield(R, cfg.metric), continue; end
            if ~any(str2double(data3_original_subjects()) == R.subject)
                continue;
            end

            M = R.(cfg.metric);
            if isempty(timesLow)
                timesLow = R.timesLow(:);
                timesHigh = R.timesHigh(:);
                design = R.temporalDesign;
            else
                if numel(R.timesLow) ~= numel(timesLow) || any(abs(R.timesLow(:) - timesLow) > 1e-9) || ...
                        numel(R.timesHigh) ~= numel(timesHigh) || any(abs(R.timesHigh(:) - timesHigh) > 1e-9)
                    error('Time axis mismatch in %s.', fpath);
                end
            end

            dataAll(end+1,:,:) = M; 
            usedFiles{end+1,1} = fpath; %#ok<SAGROW>
            subjects(end+1,1) = R.subject; %#ok<SAGROW>
        end

        if isempty(dataAll)
            warning('No included data found for %s.', modelName);
            continue;
        end

        statCfg = struct();
        statCfg.null = cfg.chance;
        statCfg.nPerm = cfg.nPerm2D;
        statCfg.tail = 'right';
        statCfg.clusterAlpha = cfg.clusterAlpha;
        statCfg.alpha = cfg.alpha;
        statCfg.clusterStat = 'mass';
        statCfg.minClusterSize = 2;
        statCfg.clusterConnectivity = 'withinSegmentBlocks';
        statCfg.randomSeed = cfg.randomSeed;
        statCfg.verbose = true;

        stat = cluster_perm_2d_matrix(dataAll, timesLow, timesHigh, design, statCfg);

        T = table();
        for ci = 1:numel(stat.clusters)
            idx = stat.clusters(ci).idx(:);
            [rowIdx, colIdx] = ind2sub(size(stat.mean), idx);
            row = table();
            row.comparison = {comparisonName};
            row.model = {modelName};
            row.clusterId = ci;
            row.lowStartTime = min(stat.xTimes(rowIdx));
            row.lowEndTime = max(stat.xTimes(rowIdx));
            row.highStartTime = min(stat.yTimes(colIdx));
            row.highEndTime = max(stat.yTimes(colIdx));
            row.nSamples = numel(idx);
            row.clusterStat = stat.clusters(ci).clusterStat;
            row.p = stat.clusters(ci).p;
            row.significant = stat.clusters(ci).p <= cfg.alpha;
            T = [T; row]; %#ok<AGROW>
        end

        save(fullfile(groupDir, sprintf('%s_%s_2d_cluster_stats.mat', modelName, cfg.metric)), ...
            'stat', 'cfg', 'usedFiles', 'subjects', 'design', '-v7.3');
        writetable(T, fullfile(groupDir, sprintf('%s_%s_clusters.csv', modelName, cfg.metric)));
        clusterTables{end+1,1} = T; %#ok<SAGROW>

        if cfg.makeGroupFigures
            fig = figure('Color', 'w', 'Visible', cfg.figureVisible, 'Position', [100 100 980 720]);
            ax = subplot(1, 1, 1, 'Parent', fig);
            imagesc(ax, timesHigh, timesLow, stat.mean);
            set(ax, 'YDir', 'normal');
            axis(ax, 'tight');
            hold(ax, 'on');
            colormap(ax, parula);
            colorbar(ax);

            if isempty(cfg.colorLimits)
                delta = max(abs(stat.mean(:) - cfg.chance), [], 'omitnan');
                if isempty(delta) || isnan(delta) || delta == 0
                    delta = 0.02;
                end
                delta = max(delta, 0.02);
                colorLim = [cfg.chance - delta, cfg.chance + delta];
                if ~isempty(cfg.colorLimitBounds)
                    colorLim = [max(cfg.colorLimitBounds(1), colorLim(1)), ...
                        min(cfg.colorLimitBounds(2), colorLim(2))];
                end
            else
                colorLim = cfg.colorLimits;
            end
            if colorLim(1) >= colorLim(2)
                colorLim = colorLim + [-0.01 0.01];
            end
            clim(ax, colorLim);

            if any(stat.significantMask(:))
                contour(ax, timesHigh, timesLow, double(stat.significantMask), [1 1], ...
                    'Color', 'k', 'LineWidth', 1.4);
            end

            if isfield(design, 'highSegmentStartsMs') && isfield(design, 'highSegmentWidthMs')
                highBoundaries = [design.highSegmentStartsMs(:); ...
                    design.highSegmentStartsMs(end) + design.highSegmentWidthMs];
                for bi = 1:numel(highBoundaries)
                    xline(ax, highBoundaries(bi), ':', 'Color', [0.45 0.45 0.45], ...
                        'HandleVisibility', 'off');
                end
            end
            if isfield(design, 'lowWindowsMs')
                lowBoundaries = unique([design.lowWindowsMs(:,1); design.lowWindowsMs(:,2)]);
                for bi = 1:numel(lowBoundaries)
                    yline(ax, lowBoundaries(bi), ':', 'Color', [0.45 0.45 0.45], ...
                        'HandleVisibility', 'off');
                end
            end

            if isfield(design, 'analysisMode') && strcmpi(design.analysisMode, 'encMaint') && ...
                    isfield(design, 'highSegmentStartsMs') && isfield(design, 'highSegmentWidthMs')
                starts = design.highSegmentStartsMs(:)';
                nSeg = numel(starts);
                onLabels = arrayfun(@(idx) sprintf('s%d on', idx), 1:nSeg, 'UniformOutput', false);
                offLabels = arrayfun(@(idx) sprintf('s%d off', idx), 1:nSeg, 'UniformOutput', false);
                xline(ax, starts, 'r--', onLabels, ...
                    'HandleVisibility', 'off', ...
                    'LabelVerticalAlignment', 'top', ...
                    'LabelHorizontalAlignment', 'left');
                xline(ax, starts + 100, 'r--', offLabels, ...
                    'HandleVisibility', 'off', ...
                    'LabelVerticalAlignment', 'top', ...
                    'LabelHorizontalAlignment', 'left');
            end

            if isfield(design, 'highSegmentStartsMs') && isfield(design, 'highSegmentWidthMs') && ...
                    ~isempty(design.highSegmentStartsMs)
                xStart = min(design.highSegmentStartsMs(:));
                xEnd = max(design.highSegmentStartsMs(:)) + design.highSegmentWidthMs;
            else
                xStart = min(timesHigh);
                xEnd = max(timesHigh);
            end
            if isfield(design, 'lowWindowsMs') && ~isempty(design.lowWindowsMs)
                yStart = min(design.lowWindowsMs(:,1));
                yEnd = max(design.lowWindowsMs(:,2));
            else
                yStart = min(timesLow);
                yEnd = max(timesLow);
            end
            if xStart == xEnd
                xEnd = xStart + 1;
            end
            if yStart == yEnd
                yEnd = yStart + 1;
            end
            xlim(ax, [xStart xEnd]);
            ylim(ax, [yStart yEnd]);
            daspect(ax, [1 1 1]);

            xlabel(ax, 'Set-size 6 time (ms)');
            if isfield(design, 'lowSetSize')
                ylabel(ax, sprintf('Set-size %d time (ms)', design.lowSetSize));
            else
                ylabel(ax, sprintf('Set-size %d time (ms)', cfg.lowSetSize));
            end
            title(ax, sprintf('%s %s %s', comparisonName, modelName, cfg.metric), 'Interpreter', 'none');
            box(ax, 'off');
            set(ax, 'FontSize', 11);

            print(fig, fullfile(figDir, sprintf('%s_%s_heatmap.png', modelName, cfg.metric)), ...
                '-dpng', sprintf('-r%d', cfg.figureDpi));
            savefig(fig, fullfile(figDir, sprintf('%s_%s_heatmap.fig', modelName, cfg.metric)));
            close(fig);
        end

        fprintf('  Group stats %s: n=%d, matrix=%dx%d, significant clusters=%d\n', ...
            modelName, size(dataAll, 1), size(dataAll, 2), size(dataAll, 3), numel(stat.significantClusters));
    end

    if ~isempty(clusterTables)
        allClusters = vertcat(clusterTables{:});
        writetable(allClusters, fullfile(groupDir, sprintf('%s_all_model_clusters.csv', cfg.metric)));
    end
end

fprintf('\nSpatial-matched sequential LDA finished.\nOutput folder:\n%s\n', outputdir);

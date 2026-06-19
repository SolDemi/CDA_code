%% data3 sequential segment-state RSA for CDA
% Builds empirical RDMs directly from trial-level CDA patterns.
% Conditions are set-size-specific maintenance segments:
% SS1_P1, SS3_P1-3, SS6_P1-6.
%
% Each condition pair is matched within shared cumulative-area bins before
% repeated random-half crossnobis / LDC estimation. Attended-left and
% attended-right RDMs are computed separately and then averaged.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
datadir = fullfile(dataDir, 'cda_alpha');
behaviorDir = fullfile(dataDir, 'Behavior_data_script');
outputDir = fullfile(dataDir, 'RSA_segment_state_maintOnly_areaMatched', 'CDA');
figDir = fullfile(outputDir, 'Figures');

addpath(codeDir);

%% Configuration
cfg = struct();
cfg.projectRoot = projectRoot;
cfg.codeDir = codeDir;
cfg.dataDir = dataDir;
cfg.cdaDir = datadir;
cfg.behaviorDir = behaviorDir;
cfg.outputDir = outputDir;
cfg.figureDir = figDir;

cfg.analysisName = 'data3_segment_state_RSA';
cfg.featureName = 'CDA';
cfg.analysisModeName = 'maintOnly';
cfg.stimStepMs = 460;
cfg.segmentStartOffsetMs = 100;
cfg.segmentWidthMs = 360;
cfg.segmentDescription = 'maintenance only, 100-460 ms after item onset';
cfg.segmentTailPolicy = 'Use only item-specific maintenance windows; do not include the post-final delay.';

cfg.conditionSetSizes = [1 3 3 3 6 6 6 6 6 6];
cfg.conditionPositions = [1 1 2 3 1 2 3 4 5 6];
cfg.conditionNames = {'SS1_P1','SS3_P1','SS3_P2','SS3_P3', ...
    'SS6_P1','SS6_P2','SS6_P3','SS6_P4','SS6_P5','SS6_P6'};

cfg.timeBinWidthMs = 50;
cfg.timeBinStepMs = 50;
cfg.keepOnlyFullTimeBins = true;
cfg.nRdmSplits = 1000;
cfg.minAreaMatchedTrialsPerCondition = 8;
cfg.useCorrectOnly = false;

cfg.randomSeed = 20260618;
cfg.covShrinkage = 0.10;
cfg.ridgeScale = 1e-6;
cfg.minCovRcond = 1e-10;

cfg.alpha = 0.05;
cfg.fdrMethod = 'fdr';
cfg.makeFigures = true;
cfg.figureDpi = 200;

cfg.rectHalfSizePx = 30;
cfg.rectHalfWidthPx = cfg.rectHalfSizePx;
cfg.rectHalfHeightPx = cfg.rectHalfSizePx;
cfg.bboxAreaMode = 'fixationAnchored'; % Same default as data3_rect_area_alpha_decoding.m
cfg.cumulativeAreaMetric = 'CumBBoxArea';
cfg.nSpatialBins = 2; % Pooled pairwise quantile bins, matching data3_rect_area_alpha_decoding.m
cfg.areaMatchingDescription = ['For each condition pair, retain shared pooled-quantile cumulative-area bins, ' ...
    'draw equal trial counts per condition within each bin, and split each bin equally across the two LDC halves.'];

nSplitOverride = str2double(strtrim(char(getenv('DATA3_RSA_N_SPLITS'))));
if ~isnan(nSplitOverride) && nSplitOverride > 0
    cfg.nRdmSplits = round(nSplitOverride);
end

useCorrectOverride = strtrim(char(getenv('DATA3_RSA_USE_CORRECT_ONLY')));
if ~isempty(useCorrectOverride)
    cfg.useCorrectOnly = ismember(lower(useCorrectOverride), {'1','true','yes'});
end

makeFiguresOverride = strtrim(char(getenv('DATA3_RSA_MAKE_FIGURES')));
if ~isempty(makeFiguresOverride)
    cfg.makeFigures = ~ismember(lower(makeFiguresOverride), {'0','false','no'});
end

if ~isfolder(outputDir)
    mkdir(outputDir);
end
if cfg.makeFigures && ~isfolder(figDir)
    mkdir(figDir);
end

conditionTable = table();
conditionTable.conditionName = cfg.conditionNames(:);
conditionTable.setSize = cfg.conditionSetSizes(:);
conditionTable.position = cfg.conditionPositions(:);
conditionTable.currentLoad = conditionTable.position;
conditionTable.capacityComponent = min(conditionTable.position, 3);
conditionTable.postCapacityBinary = double(conditionTable.position >= 4);
conditionTable.postCapacityRamp = max(conditionTable.position - 3, 0);
conditionTableTemplate = conditionTable;

modelSets = struct();
modelSets(1).name = 'ModelSetA';
modelSets(1).description = 'capacityComponent + postCapacityBinary';
modelSets(1).predictors = {'capacityComponent','postCapacityBinary'};
modelSets(1).diagnosticOnly = false;
modelSets(2).name = 'ModelSetB';
modelSets(2).description = 'capacityComponent + postCapacityRamp';
modelSets(2).predictors = {'capacityComponent','postCapacityRamp'};
modelSets(2).diagnosticOnly = false;
modelSets(3).name = 'DiagnosticAll';
modelSets(3).description = 'Diagnostic only: capacityComponent + postCapacityBinary + postCapacityRamp';
modelSets(3).predictors = {'capacityComponent','postCapacityBinary','postCapacityRamp'};
modelSets(3).diagnosticOnly = true;

sideNames = {'L','R'};
sideLongNames = {'Left','Right'};
setSizesToCheck = [1 3 6];
nCond = numel(cfg.conditionNames);
timeBinStarts = 0:cfg.timeBinStepMs:(cfg.segmentWidthMs - cfg.timeBinWidthMs);
timeBinEnds = timeBinStarts + cfg.timeBinWidthMs;
timeBinCenters = timeBinStarts + cfg.timeBinWidthMs / 2;
nTimeBins = numel(timeBinCenters);
timeBinInfo = table((1:nTimeBins)', timeBinStarts(:), timeBinEnds(:), timeBinCenters(:), ...
    'VariableNames', {'binIndex','startMsRelative','endMsRelative','centerMsRelative'});

files = dir(fullfile(datadir, 'sub*.mat'));
files = data3_filter_subject_mat_files(files, data3_subject_filter());
if isempty(files)
    error('No included sub*.mat files found in %s.', datadir);
end

rng(cfg.randomSeed, 'twister');

includedSubjects = [];
skippedSubjects = table();
subjectFiles = {};
subjectSummaries = table();
groupTimeR = struct();
groupSegmentR = struct();
groupEmpiricalRDMTime = [];
groupEmpiricalRDMSegment = [];
groupTheoryCorrelationMatrix = [];
groupAreaMatchTables = {};

for sf = 1:numel(files)
    file = files(sf).name;
    inFile = fullfile(files(sf).folder, file);
    fprintf('\ndata3 segment-state RSA: %s\n', file);

    subjectWarnings = {};
    tok = regexp(file, '^sub(\d+)\.mat$', 'tokens', 'once');
    if isempty(tok)
        subject = NaN;
    else
        subject = str2double(tok{1});
    end

    try
        S = load(inFile, 'cda');
        if ~isfield(S, 'cda')
            error('%s does not contain cda. Rerun data3_cda_alpha.m.', inFile);
        end
        cda = S.cda;
        if isfield(cda, 'subject') && ~isempty(cda.subject)
            subject = cda.subject;
        end
        if ~isfield(cda, 'trial') || ~isfield(cda, 'timeBySetSize') || ~isfield(cda, 'extraction')
            error('%s is missing cda.trial, cda.timeBySetSize, or cda.extraction.', inFile);
        end
        if ~isfield(cda, 'relPairLabels')
            error('%s is missing cda.relPairLabels.', inFile);
        end

        behavior = load_data3_behavior_with_geometry(subject, cfg);

        trialCountCheck = table();
        for siCheck = 1:numel(setSizesToCheck)
            setSizeCheck = setSizesToCheck(siCheck);
            setFieldCheck = sprintf('setsize%d', setSizeCheck);
            if ~isfield(cda.extraction, setFieldCheck) || ~isfield(cda.extraction.(setFieldCheck), 'nRawEventTrials')
                error('Subject %d is missing cda.extraction.%s.nRawEventTrials.', subject, setFieldCheck);
            end
            nBehavior = sum(behavior.SetSize == setSizeCheck);
            nEvents = cda.extraction.(setFieldCheck).nRawEventTrials;
            if nBehavior ~= nEvents
                error('Subject %d setsize%d behavior/EEG trial count mismatch: behavior=%d, EEG events=%d.', ...
                    subject, setSizeCheck, nBehavior, nEvents);
            end
            trialCountCheck = [trialCountCheck; table(subject, setSizeCheck, nBehavior, nEvents, ...
                'VariableNames', {'Subject','SetSize','BehaviorTrials','RawEEGEvents'})]; %#ok<AGROW>
        end

        sideRDMTime = nan(nCond, nCond, nTimeBins, numel(sideNames));
        sideRDMSegment = nan(nCond, nCond, numel(sideNames));
        sideInfo = struct();
        areaRows = {};
        subjectAreaMatchRows = {};

        for sideIdx = 1:numel(sideNames)
            sideName = sideNames{sideIdx};
            sideLong = sideLongNames{sideIdx};
            if strcmpi(sideName, 'L')
                sideCode = -1;
            else
                sideCode = 1;
            end

            conditionDataTime = cell(nCond, nTimeBins);
            conditionDataSegment = cell(nCond, 1);
            conditionBehavior = cell(nCond, 1);
            conditionArea = cell(nCond, 1);
            nTrialsByCondition = nan(nCond, 1);

            for ci = 1:nCond
                setSizeNow = cfg.conditionSetSizes(ci);
                posNow = cfg.conditionPositions(ci);
                setField = sprintf('setsize%d', setSizeNow);
                loadStr = num2str(setSizeNow);
                leftName = sprintf('left_%s_%s', sideName, loadStr);
                rightName = sprintf('right_%s_%s', sideName, loadStr);

                if ~(isfield(cda.trial, leftName) && isfield(cda.trial, rightName))
                    error('Subject %d saved CDA is missing %s/%s.', subject, leftName, rightName);
                end
                if ~isfield(cda.timeBySetSize, setField)
                    error('Subject %d saved CDA is missing timeBySetSize.%s.', subject, setField);
                end
                if ~isfield(cda.extraction, setField)
                    error('Subject %d saved CDA is missing extraction.%s.', subject, setField);
                end

                leftX = cda.trial.(leftName);
                rightX = cda.trial.(rightName);
                if strcmpi(sideName, 'L')
                    X = rightX - leftX;
                else
                    X = leftX - rightX;
                end
                timeAxis = cda.timeBySetSize.(setField)(:)';
                if size(X, 3) ~= numel(timeAxis)
                    error('Subject %d %s setsize%d data/time mismatch: %d samples vs %d time points.', ...
                        subject, sideName, setSizeNow, size(X, 3), numel(timeAxis));
                end
                if any(isnan(X(:)))
                    subjectWarnings{end+1, 1} = sprintf('Subject %d %s setsize%d has NaNs in CDA data.', subject, sideName, setSizeNow); %#ok<SAGROW>
                end

                meta = cda.extraction.(setField);
                if isfield(meta, 'cleanSetSizeTrialIndex')
                    cleanSetSizeTrialIndex = meta.cleanSetSizeTrialIndex(:);
                elseif isfield(meta, 'Ikeep') && isfield(meta, 'HEOG_in')
                    cleanSetSizeTrialIndex = meta.Ikeep(meta.HEOG_in);
                    cleanSetSizeTrialIndex = cleanSetSizeTrialIndex(:);
                else
                    error('Subject %d setsize%d extraction is missing clean trial indices.', subject, setSizeNow);
                end

                behaviorSet = behavior(behavior.SetSize == setSizeNow, :);
                if max(cleanSetSizeTrialIndex) > height(behaviorSet)
                    error('Subject %d setsize%d clean trial index exceeds behavior trial count.', subject, setSizeNow);
                end
                behaviorClean = behaviorSet(cleanSetSizeTrialIndex, :);
                behaviorSide = behaviorClean(behaviorClean.CueSide == sideCode, :);
                if cfg.useCorrectOnly
                    behaviorSide = behaviorSide(behaviorSide.Correct == 1, :);
                end

                if height(behaviorClean(behaviorClean.CueSide == sideCode, :)) ~= size(X, 1)
                    error('Subject %d %s setsize%d behavior rows (%d) do not match CDA rows (%d).', ...
                        subject, sideName, setSizeNow, height(behaviorClean(behaviorClean.CueSide == sideCode, :)), size(X, 1));
                end
                if cfg.useCorrectOnly
                    keepCorrectRows = behaviorClean.CueSide == sideCode & behaviorClean.Correct == 1;
                    X = X(keepCorrectRows, :, :);
                end
                if height(behaviorSide) ~= size(X, 1)
                    error('Subject %d %s setsize%d behavior rows (%d) do not match selected CDA rows (%d).', ...
                        subject, sideName, setSizeNow, height(behaviorSide), size(X, 1));
                end

                segmentStart = (posNow - 1) * cfg.stimStepMs + cfg.segmentStartOffsetMs;
                segmentEnd = segmentStart + cfg.segmentWidthMs;
                if timeAxis(1) > segmentStart || timeAxis(end) < segmentEnd
                    error('Subject %d setsize%d time axis [%g %g] ms does not cover condition %s window [%g %g] ms.', ...
                        subject, setSizeNow, timeAxis(1), timeAxis(end), cfg.conditionNames{ci}, segmentStart, segmentEnd);
                end

                segmentIdx = timeAxis >= segmentStart & timeAxis <= segmentEnd;
                if ~any(segmentIdx)
                    error('Subject %d condition %s has no samples in full segment [%g %g] ms.', ...
                        subject, cfg.conditionNames{ci}, segmentStart, segmentEnd);
                end
                conditionDataSegment{ci} = mean(X(:, :, segmentIdx), 3, 'omitnan');
                conditionBehavior{ci} = behaviorSide;
                nTrialsByCondition(ci) = size(conditionDataSegment{ci}, 1);

                areaField = sprintf('%s_Pos%d', cfg.cumulativeAreaMetric, posNow);
                if ~ismember(areaField, behaviorSide.Properties.VariableNames)
                    error('Behavior table is missing %s.', areaField);
                end
                conditionArea{ci} = behaviorSide.(areaField);
                A = table();
                A.Subject = subject;
                A.Side = {sideLong};
                A.Condition = cfg.conditionNames(ci);
                A.SetSize = setSizeNow;
                A.Position = posNow;
                A.NTrials = height(behaviorSide);
                A.MeanCumulativeArea = mean(behaviorSide.(areaField), 'omitnan');
                A.MedianCumulativeArea = median(behaviorSide.(areaField), 'omitnan');
                A.SDCumulativeArea = std(behaviorSide.(areaField), 'omitnan');
                areaRows{end+1, 1} = A; %#ok<SAGROW>

                for bi = 1:nTimeBins
                    binStart = segmentStart + timeBinStarts(bi);
                    binEnd = segmentStart + timeBinEnds(bi);
                    if cfg.keepOnlyFullTimeBins
                        binIdx = timeAxis >= binStart & timeAxis < binEnd;
                    else
                        binIdx = timeAxis >= binStart & timeAxis <= min(binEnd, segmentEnd);
                    end
                    if ~any(binIdx)
                        error('Subject %d condition %s has no samples in time bin [%g %g] ms.', ...
                            subject, cfg.conditionNames{ci}, binStart, binEnd);
                    end
                    conditionDataTime{ci, bi} = mean(X(:, :, binIdx), 3, 'omitnan');
                end
            end

            sideInfo.(sideLong) = struct();
            sideInfo.(sideLong).nTrialsByCondition = nTrialsByCondition;
            sideInfo.(sideLong).conditionBehavior = conditionBehavior;

            fprintf('  %s side: pairwise area matching, raw min condition trials=%d, splits=%d\n', ...
                sideLong, min(nTrialsByCondition), cfg.nRdmSplits);
            for bi = 1:nTimeBins
                condForBin = conditionDataTime(:, bi);
                [rdmBin, rdmInfo] = compute_area_matched_crossnobis_rdm(condForBin, conditionArea, cfg);
                sideRDMTime(:, :, bi, sideIdx) = rdmBin;
                sideInfo.(sideLong).timeBin(bi).rdmInfo = rdmInfo; %#ok<SAGROW>
            end
            [rdmSegment, segmentInfo] = compute_area_matched_crossnobis_rdm( ...
                conditionDataSegment, conditionArea, cfg);
            sideRDMSegment(:, :, sideIdx) = rdmSegment;
            sideInfo.(sideLong).segmentAverage.rdmInfo = segmentInfo;

            upperMaskSide = triu(true(nCond), 1);
            sideSkipped = ~any(isfinite(rdmSegment(upperMaskSide)));
            if sideSkipped
                sideSkipReason = 'noValidAreaMatchedPairs';
                warning('Subject %d %s skipped: %s.', subject, sideLong, sideSkipReason);
                subjectWarnings{end+1, 1} = sprintf('Subject %d %s skipped: %s.', ...
                    subject, sideLong, sideSkipReason); %#ok<SAGROW>
            else
                sideSkipReason = '';
            end
            sideInfo.(sideLong).skipped = sideSkipped;
            sideInfo.(sideLong).skipReason = sideSkipReason;

            matchNow = segmentInfo.matchSummary;
            matchNow.Subject = repmat(subject, height(matchNow), 1);
            matchNow.Side = repmat({sideLong}, height(matchNow), 1);
            matchNow = movevars(matchNow, {'Subject','Side'}, 'Before', 1);
            subjectAreaMatchRows{end+1, 1} = matchNow; %#ok<SAGROW>
        end

        if isempty(areaRows)
            error('Subject %d has no area rows.', subject);
        end
        areaInfo = vertcat(areaRows{:});
        conditionTableSubject = conditionTableTemplate;
        areaMatchSummary = vertcat(subjectAreaMatchRows{:});

        empiricalRDM_side = struct();
        empiricalRDM_side.Left.time = sideRDMTime(:, :, :, 1);
        empiricalRDM_side.Left.segmentAverage = sideRDMSegment(:, :, 1);
        empiricalRDM_side.Left.info = sideInfo.Left;
        empiricalRDM_side.Right.time = sideRDMTime(:, :, :, 2);
        empiricalRDM_side.Right.segmentAverage = sideRDMSegment(:, :, 2);
        empiricalRDM_side.Right.info = sideInfo.Right;

        empiricalRDM_time = mean(sideRDMTime, 4, 'omitnan');
        empiricalRDM_segmentAverage = mean(sideRDMSegment, 3, 'omitnan');
        upperMaskSubject = triu(true(nCond), 1);
        if ~any(isfinite(empiricalRDM_segmentAverage(upperMaskSubject)))
            newRow = table(subject, {file}, {'bothSidesSkipped'}, ...
                'VariableNames', {'Subject','File','Reason'});
            skippedSubjects = [skippedSubjects; newRow]; %#ok<AGROW>
            warning('Subject %d skipped because both sides were invalid.', subject);
            continue;
        end

        [theoryRDMs, theoryVectors, theoryCorrelationMatrix, theoryInfo] = build_theory_rdms(conditionTableSubject);
        rsaResults = run_semipartial_rsa(empiricalRDM_time, empiricalRDM_segmentAverage, theoryVectors, modelSets, cfg);

        subjectSummary = table();
        subjectSummary.Subject = subject;
        subjectSummary.File = {file};
        subjectSummary.MinTrialsLeft = min(sideInfo.Left.nTrialsByCondition);
        subjectSummary.MinTrialsRight = min(sideInfo.Right.nTrialsByCondition);
        subjectSummary.LeftSkipped = sideInfo.Left.skipped;
        subjectSummary.RightSkipped = sideInfo.Right.skipped;
        subjectSummary.LeftSkipReason = {sideInfo.Left.skipReason};
        subjectSummary.RightSkipReason = {sideInfo.Right.skipReason};
        subjectSummary.ValidPairsLeft = sum(sideInfo.Left.segmentAverage.rdmInfo.matchSummary.ValidDistance);
        subjectSummary.ValidPairsRight = sum(sideInfo.Right.segmentAverage.rdmInfo.matchSummary.ValidDistance);
        subjectSummaries = [subjectSummaries; subjectSummary]; %#ok<AGROW>

        outFile = fullfile(outputDir, sprintf('sub%d_RSA.mat', subject));
        conditionTable = conditionTableSubject; %#ok<NASGU>
        save(outFile, 'subject', 'cfg', 'conditionTableSubject', 'conditionTable', ...
            'timeBinInfo', 'empiricalRDM_time', 'empiricalRDM_segmentAverage', ...
            'empiricalRDM_side', 'theoryRDMs', 'theoryVectors', ...
            'theoryCorrelationMatrix', 'theoryInfo', 'rsaResults', 'areaInfo', ...
            'areaMatchSummary', 'trialCountCheck', 'subjectWarnings', '-v7.3');

        groupSubjectIdx = numel(includedSubjects) + 1;
        includedSubjects(groupSubjectIdx, 1) = subject; %#ok<SAGROW>
        subjectFiles{groupSubjectIdx, 1} = outFile; %#ok<SAGROW>
        groupEmpiricalRDMTime(:, :, :, groupSubjectIdx) = empiricalRDM_time; %#ok<SAGROW>
        groupEmpiricalRDMSegment(:, :, groupSubjectIdx) = empiricalRDM_segmentAverage; %#ok<SAGROW>
        groupTheoryCorrelationMatrix(:, :, groupSubjectIdx) = theoryCorrelationMatrix; %#ok<SAGROW>
        groupAreaMatchTables{groupSubjectIdx, 1} = areaMatchSummary; %#ok<SAGROW>

        for ms = 1:numel(modelSets)
            setName = modelSets(ms).name;
            if ~isfield(groupTimeR, setName)
                groupTimeR.(setName) = [];
                groupSegmentR.(setName) = [];
            end
            groupTimeR.(setName)(:, :, groupSubjectIdx) = rsaResults.(setName).time.semiPartialR; %#ok<SAGROW>
            groupSegmentR.(setName)(:, groupSubjectIdx) = rsaResults.(setName).segmentAverage.semiPartialR; %#ok<SAGROW>
        end

        fprintf('  Saved %s\n', outFile);

    catch ME
        newRow = table(subject, {file}, {ME.message}, ...
            'VariableNames', {'Subject','File','Reason'});
        skippedSubjects = [skippedSubjects; newRow]; %#ok<AGROW>
        warning('Subject %d failed and was skipped: %s', subject, ME.message);
    end
end

if isempty(includedSubjects)
    error('No subjects produced valid RSA outputs.');
end

nIncludedSubjects = numel(includedSubjects);
if size(groupEmpiricalRDMTime, 4) ~= nIncludedSubjects || ...
        size(groupEmpiricalRDMSegment, 3) ~= nIncludedSubjects || ...
        size(groupTheoryCorrelationMatrix, 3) ~= nIncludedSubjects
    error('Group RDM subject dimensions do not match includedSubjects.');
end
for ms = 1:numel(modelSets)
    setName = modelSets(ms).name;
    if size(groupTimeR.(setName), 3) ~= nIncludedSubjects || ...
            size(groupSegmentR.(setName), 2) ~= nIncludedSubjects
        error('%s group subject dimensions do not match includedSubjects.', setName);
    end
end

%% Group statistics and outputs
groupStats = struct();
for ms = 1:numel(modelSets)
    setName = modelSets(ms).name;
    groupStats.(setName) = compute_group_stats(groupTimeR.(setName), groupSegmentR.(setName), ...
        modelSets(ms).predictors, timeBinInfo, cfg);
end

groupAverageEmpiricalRDM_time = mean(groupEmpiricalRDMTime, 4, 'omitnan');
groupAverageEmpiricalRDM_segmentAverage = mean(groupEmpiricalRDMSegment, 3, 'omitnan');
groupAverageTheoryCorrelationMatrix = mean(groupTheoryCorrelationMatrix, 3, 'omitnan');

groupConditionTable = conditionTableTemplate;
[groupTheoryRDMs, groupTheoryVectors, groupTheoryCorrelationMatrix, groupTheoryInfo] = build_theory_rdms(groupConditionTable);
areaMatchSummary = vertcat(groupAreaMatchTables{:});

groupFile = fullfile(outputDir, 'group_RSA_CDA.mat');
conditionTable = groupConditionTable; %#ok<NASGU>
save(groupFile, 'cfg', 'includedSubjects', 'subjectFiles', 'subjectSummaries', ...
    'skippedSubjects', 'conditionTable', 'groupConditionTable', ...
    'groupTimeR', 'groupSegmentR', 'groupStats', ...
    'groupAverageEmpiricalRDM_time', 'groupAverageEmpiricalRDM_segmentAverage', ...
    'groupTheoryRDMs', 'groupTheoryVectors', 'groupTheoryCorrelationMatrix', ...
    'groupAverageTheoryCorrelationMatrix', 'groupTheoryInfo', 'modelSets', ...
    'timeBinInfo', 'areaMatchSummary', '-v7.3');

if ~isempty(subjectSummaries)
    writetable(subjectSummaries, fullfile(outputDir, 'subject_RSA_summary.csv'));
end
if ~isempty(skippedSubjects)
    writetable(skippedSubjects, fullfile(outputDir, 'skipped_subjects.csv'));
end
writetable(areaMatchSummary, fullfile(outputDir, 'area_match_summary.csv'));

for ms = 1:numel(modelSets)
    setName = modelSets(ms).name;
    statsNow = groupStats.(setName);
    writetable(statsNow.timeTable, fullfile(outputDir, sprintf('%s_time_stats.csv', setName)));
    writetable(statsNow.segmentTable, fullfile(outputDir, sprintf('%s_segment_average_stats.csv', setName)));
end

if cfg.makeFigures
    make_rsa_figures(groupAverageEmpiricalRDM_segmentAverage, groupTheoryRDMs, ...
        groupTheoryCorrelationMatrix, groupStats, groupSegmentR, modelSets, ...
        conditionTable.conditionName, timeBinInfo, cfg);
end

fprintf('\nSegment-state RSA finished.\n');
fprintf('Included subjects (%d): %s\n', numel(includedSubjects), mat2str(includedSubjects(:)'));
if ~isempty(skippedSubjects)
    fprintf('Skipped subjects/sides are listed in:\n%s\n', fullfile(outputDir, 'skipped_subjects.csv'));
end
fprintf('Subject outputs and group file saved under:\n%s\n', outputDir);
fprintf('Group file:\n%s\n', groupFile);

%% Local functions
function behavior = load_data3_behavior_with_geometry(subject, cfg)
behavior = table();
trialOffset = 0;

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

for blockNum = 1:4
    behaviorFile = fullfile(cfg.behaviorDir, 'Beh_data', ...
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

    neededFields = {'mem_lets','set_size','change','cue_side','change_item', ...
        'change_item_uncued','resp','rt','acc','loc_list_cued','locs'};
    for fi = 1:numel(neededFields)
        if ~isfield(behData, neededFields{fi})
            error('%s is missing data.%s.', behaviorFile, neededFields{fi});
        end
    end
    if size(behData.locs, 2) < 6
        error('%s has fewer than six location columns.', behaviorFile);
    end

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

    for pos = 1:6
        blockBehavior.(sprintf('CurrentLoc_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CurrentRing_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CurrentRadius_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CurrentAngle_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CumHullArea_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CumBBoxArea_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CumBBoxWidth_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CumBBoxHeight_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CumMaxDist_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('CumMeanDist_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('NearestPriorDist_Pos%d', pos)) = nan(nBehTrial, 1);
        blockBehavior.(sprintf('AngleSpan_Pos%d', pos)) = nan(nBehTrial, 1);
    end

    locs = behData.locs;
    for ti = 1:nBehTrial
        setSizeNow = blockBehavior.SetSize(ti);
        locListNow = blockBehavior.LocList(ti);
        if isnan(setSizeNow) || setSizeNow < 1 || setSizeNow > 6 || ...
                isnan(locListNow) || locListNow < 1 || locListNow > 2
            continue;
        end

        coords = nan(setSizeNow, 2);
        for pos = 1:setSizeNow
            locNow = locs(ti, pos);
            if isnan(locNow) || locNow < 1 || locNow > numLocs
                continue;
            end

            xNow = xPosList(locListNow, locNow);
            yNow = yPosList(locListNow, locNow);
            coords(pos, :) = [xNow, yNow];

            blockBehavior.(sprintf('CurrentLoc_Pos%d', pos))(ti) = locNow;
            blockBehavior.(sprintf('CurrentRing_Pos%d', pos))(ti) = ceil(locNow / numel(stimAngs));
            blockBehavior.(sprintf('CurrentRadius_Pos%d', pos))(ti) = hypot(xNow, yNow);
            angleNow = atan2d(yNow, xNow);
            if angleNow < 0
                angleNow = angleNow + 360;
            end
            blockBehavior.(sprintf('CurrentAngle_Pos%d', pos))(ti) = angleNow;

            validCoords = coords(1:pos, :);
            validCoords = validCoords(all(~isnan(validCoords), 2), :);
            if isempty(validCoords)
                continue;
            end

            switch lower(cfg.bboxAreaMode)
                case 'fixationanchored'
                    bboxCoords = [validCoords; 0 0];
                case 'stimonly'
                    bboxCoords = validCoords;
                otherwise
                    error('Unsupported cfg.bboxAreaMode: %s.', cfg.bboxAreaMode);
            end
            xMinBox = min(bboxCoords(:,1) - cfg.rectHalfWidthPx);
            xMaxBox = max(bboxCoords(:,1) + cfg.rectHalfWidthPx);
            yMinBox = min(bboxCoords(:,2) - cfg.rectHalfHeightPx);
            yMaxBox = max(bboxCoords(:,2) + cfg.rectHalfHeightPx);
            bboxWidth = xMaxBox - xMinBox;
            bboxHeight = yMaxBox - yMinBox;
            blockBehavior.(sprintf('CumBBoxWidth_Pos%d', pos))(ti) = bboxWidth;
            blockBehavior.(sprintf('CumBBoxHeight_Pos%d', pos))(ti) = bboxHeight;
            blockBehavior.(sprintf('CumBBoxArea_Pos%d', pos))(ti) = bboxWidth * bboxHeight;

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
                        validCoords(a,2) - validCoords(b,2)); %#ok<AGROW>
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
end

function [rdm, info] = compute_area_matched_crossnobis_rdm(conditionData, conditionArea, cfg)
nCond = numel(conditionData);
nTrialsRaw = nan(nCond, 1);
nTrialsValid = nan(nCond, 1);
nFeatures = size(conditionData{1}, 2);
for ci = 1:nCond
    if isempty(conditionData{ci})
        error('Condition %d has empty data.', ci);
    end
    if ~ismatrix(conditionData{ci})
        error('Condition %d data must be trials x features.', ci);
    end
    if numel(conditionArea{ci}) ~= size(conditionData{ci}, 1)
        error('Condition %d area/data trial counts differ: %d vs %d.', ...
            ci, numel(conditionArea{ci}), size(conditionData{ci}, 1));
    end
    if size(conditionData{ci}, 2) ~= nFeatures
        error('Condition %d has %d features, expected %d.', ...
            ci, size(conditionData{ci}, 2), nFeatures);
    end

    nTrialsRaw(ci) = size(conditionData{ci}, 1);
    conditionArea{ci} = conditionArea{ci}(:);
    finiteRows = all(isfinite(conditionData{ci}), 2) & isfinite(conditionArea{ci});
    conditionData{ci} = conditionData{ci}(finiteRows, :);
    conditionArea{ci} = conditionArea{ci}(finiteRows);
    nTrialsValid(ci) = size(conditionData{ci}, 1);
end

rdm = nan(nCond, nCond);
rdm(1:(nCond + 1):end) = 0;
pairRows = {};

for i = 1:nCond
    for j = (i + 1):nCond
        areaI = conditionArea{i};
        areaJ = conditionArea{j};
        pooledArea = [areaI; areaJ];
        skipReason = '';

        if isempty(pooledArea)
            edges = [-inf inf];
            binI = nan(size(areaI));
            binJ = nan(size(areaJ));
            skipReason = 'missingCumulativeArea';
        elseif max(pooledArea) - min(pooledArea) < 1e-9
            edges = [-inf inf];
            binI = ones(size(areaI));
            binJ = ones(size(areaJ));
        else
            edges = quantile(pooledArea, linspace(0, 1, cfg.nSpatialBins + 1));
            edges = unique(edges, 'stable');
            if numel(edges) < 2
                edges = [-inf inf];
                binI = ones(size(areaI));
                binJ = ones(size(areaJ));
            else
                edges(1) = -inf;
                edges(end) = inf;
                binI = discretize(areaI, edges);
                binJ = discretize(areaJ, edges);
            end
        end

        sharedBins = intersect(unique(binI(isfinite(binI))), unique(binJ(isfinite(binJ))));
        nAvailableI = zeros(numel(sharedBins), 1);
        nAvailableJ = zeros(numel(sharedBins), 1);
        nMatchedPerBin = zeros(numel(sharedBins), 1);
        for binIdx = 1:numel(sharedBins)
            nAvailableI(binIdx) = sum(binI == sharedBins(binIdx));
            nAvailableJ(binIdx) = sum(binJ == sharedBins(binIdx));
            nMatchedPerBin(binIdx) = 2 * floor(min(nAvailableI(binIdx), nAvailableJ(binIdx)) / 2);
        end

        usableBins = nMatchedPerBin >= 2;
        sharedBins = sharedBins(usableBins);
        nAvailableI = nAvailableI(usableBins);
        nAvailableJ = nAvailableJ(usableBins);
        nMatchedPerBin = nMatchedPerBin(usableBins);
        nMatchedPerCondition = sum(nMatchedPerBin);
        nTrainPerCondition = nMatchedPerCondition / 2;
        nTestPerCondition = nMatchedPerCondition / 2;

        if isempty(skipReason) && isempty(sharedBins)
            skipReason = 'noSharedAreaBinWithTwoTrialsPerCondition';
        elseif isempty(skipReason) && nMatchedPerCondition < cfg.minAreaMatchedTrialsPerCondition
            skipReason = 'insufficientAreaMatchedTrials';
        end

        validSplitCount = 0;
        distanceSum = 0;
        ridgeUsed = nan(cfg.nRdmSplits, 1);
        rcondUsed = nan(cfg.nRdmSplits, 1);
        meanAreaIAfter = nan(cfg.nRdmSplits, 1);
        meanAreaJAfter = nan(cfg.nRdmSplits, 1);

        if isempty(skipReason)
            for splitIdx = 1:cfg.nRdmSplits
                trainIdxI = [];
                testIdxI = [];
                trainIdxJ = [];
                testIdxJ = [];

                for binIdx = 1:numel(sharedBins)
                    rowsI = find(binI == sharedBins(binIdx));
                    rowsJ = find(binJ == sharedBins(binIdx));
                    nUse = nMatchedPerBin(binIdx);
                    orderI = rowsI(randperm(numel(rowsI), nUse));
                    orderJ = rowsJ(randperm(numel(rowsJ), nUse));
                    nHalf = nUse / 2;

                    trainIdxI = [trainIdxI; orderI(1:nHalf)]; %#ok<AGROW>
                    testIdxI = [testIdxI; orderI((nHalf + 1):end)]; %#ok<AGROW>
                    trainIdxJ = [trainIdxJ; orderJ(1:nHalf)]; %#ok<AGROW>
                    testIdxJ = [testIdxJ; orderJ((nHalf + 1):end)]; %#ok<AGROW>
                end

                trainI = conditionData{i}(trainIdxI, :);
                testI = conditionData{i}(testIdxI, :);
                trainJ = conditionData{j}(trainIdxJ, :);
                testJ = conditionData{j}(testIdxJ, :);
                meanTrainI = mean(trainI, 1, 'omitnan');
                meanTestI = mean(testI, 1, 'omitnan');
                meanTrainJ = mean(trainJ, 1, 'omitnan');
                meanTestJ = mean(testJ, 1, 'omitnan');

                trainResiduals = [trainI - meanTrainI; trainJ - meanTrainJ];
                if any(~isfinite(trainResiduals(:))) || ...
                        any(~isfinite([meanTrainI, meanTestI, meanTrainJ, meanTestJ]))
                    continue;
                end

                C = cov(trainResiduals, 1);
                if ~all(isfinite(C(:)))
                    continue;
                end
                C = (C + C') / 2;
                targetScale = trace(C) / max(nFeatures, 1);
                if ~isfinite(targetScale) || targetScale <= 0
                    targetScale = mean(diag(C), 'omitnan');
                end
                if ~isfinite(targetScale) || targetScale <= 0
                    targetScale = 1;
                end
                Creg = (1 - cfg.covShrinkage) * C + ...
                    cfg.covShrinkage * targetScale * eye(nFeatures);
                ridge = cfg.ridgeScale * targetScale;
                Creg = Creg + ridge * eye(nFeatures);
                rcondNow = rcond(Creg);
                while rcondNow < cfg.minCovRcond
                    ridge = ridge * 10;
                    Creg = Creg + ridge * eye(nFeatures);
                    rcondNow = rcond(Creg);
                    if ridge > targetScale
                        break;
                    end
                end

                diffTrain = meanTrainI - meanTrainJ;
                diffTest = meanTestI - meanTestJ;
                distanceNow = (diffTrain / Creg) * diffTest';
                if ~isfinite(distanceNow)
                    continue;
                end

                validSplitCount = validSplitCount + 1;
                distanceSum = distanceSum + distanceNow;
                ridgeUsed(validSplitCount) = ridge;
                rcondUsed(validSplitCount) = rcondNow;
                matchedIdxI = [trainIdxI; testIdxI];
                matchedIdxJ = [trainIdxJ; testIdxJ];
                meanAreaIAfter(validSplitCount) = mean(areaI(matchedIdxI), 'omitnan');
                meanAreaJAfter(validSplitCount) = mean(areaJ(matchedIdxJ), 'omitnan');
            end
        end

        if validSplitCount > 0
            distance = distanceSum / validSplitCount;
            rdm(i, j) = distance;
            rdm(j, i) = distance;
        elseif isempty(skipReason)
            skipReason = 'noValidCrossnobisSplit';
        end

        row = table();
        row.ConditionI = cfg.conditionNames(i);
        row.ConditionJ = cfg.conditionNames(j);
        row.NTrialsRawI = nTrialsRaw(i);
        row.NTrialsRawJ = nTrialsRaw(j);
        row.NTrialsWithFiniteAreaI = nTrialsValid(i);
        row.NTrialsWithFiniteAreaJ = nTrialsValid(j);
        row.AreaBinEdges = {mat2str(edges(:)')};
        row.SharedAreaBins = {mat2str(sharedBins(:)')};
        row.NSharedAreaBins = numel(sharedBins);
        row.NAvailableIBySharedBin = {mat2str(nAvailableI(:)')};
        row.NAvailableJBySharedBin = {mat2str(nAvailableJ(:)')};
        row.NMatchedPerConditionBySharedBin = {mat2str(nMatchedPerBin(:)')};
        row.NMatchedPerCondition = nMatchedPerCondition;
        row.NTrainPerCondition = nTrainPerCondition;
        row.NTestPerCondition = nTestPerCondition;
        row.MeanAreaIBefore = mean(areaI, 'omitnan');
        row.MeanAreaJBefore = mean(areaJ, 'omitnan');
        row.MeanAreaDiffJMinusIBefore = row.MeanAreaJBefore - row.MeanAreaIBefore;
        row.MeanAreaIAfter = mean(meanAreaIAfter(1:validSplitCount), 'omitnan');
        row.MeanAreaJAfter = mean(meanAreaJAfter(1:validSplitCount), 'omitnan');
        row.MeanAreaDiffJMinusIAfter = row.MeanAreaJAfter - row.MeanAreaIAfter;
        row.NRequestedSplits = cfg.nRdmSplits;
        row.NValidSplits = validSplitCount;
        row.ValidDistance = validSplitCount > 0;
        row.SkipReason = {skipReason};
        row.MeanRidgeUsed = mean(ridgeUsed(1:validSplitCount), 'omitnan');
        if validSplitCount > 0
            row.MinCovRcond = min(rcondUsed(1:validSplitCount), [], 'omitnan');
        else
            row.MinCovRcond = NaN;
        end
        pairRows{end+1, 1} = row; %#ok<AGROW>
    end
end

info = struct();
info.nTrialsRawByCondition = nTrialsRaw;
info.nTrialsWithFiniteAreaByCondition = nTrialsValid;
info.nRequestedSplits = cfg.nRdmSplits;
info.nFeatures = nFeatures;
info.covShrinkage = cfg.covShrinkage;
info.ridgeScale = cfg.ridgeScale;
info.areaMetric = cfg.cumulativeAreaMetric;
info.nSpatialBins = cfg.nSpatialBins;
info.areaBinDefinition = 'pooled pairwise quantile bins; only bins shared by both conditions are retained';
info.splitDefinition = 'equal condition counts from every shared area bin in each independent LDC half';
info.distanceDefinition = 'crossvalidated Mahalanobis/LDC using covariance from pairwise matched training residuals';
info.matchSummary = vertcat(pairRows{:});
end

function [theoryRDMs, theoryVectors, theoryCorrelationMatrix, theoryInfo] = build_theory_rdms(conditionTable)
predictorNames = {'capacityComponent','postCapacityBinary','postCapacityRamp'};
values = struct();
values.capacityComponent = conditionTable.capacityComponent(:);
values.postCapacityBinary = conditionTable.postCapacityBinary(:);
values.postCapacityRamp = conditionTable.postCapacityRamp(:);

nCond = height(conditionTable);
upperMask = triu(true(nCond), 1);
theoryRDMs = struct();
theoryVectors = struct();
theoryMatrix = nan(nnz(upperMask), numel(predictorNames));

for pi = 1:numel(predictorNames)
    pred = predictorNames{pi};
    v = values.(pred);
    rdm = abs(v - v');
    theoryRDMs.(pred) = rdm;
    theoryVectors.(pred) = rdm(upperMask);
    theoryMatrix(:, pi) = theoryVectors.(pred);
end

rankedTheoryMatrix = nan(size(theoryMatrix));
for pi = 1:size(theoryMatrix, 2)
    rankedTheoryMatrix(:, pi) = tiedrank(theoryMatrix(:, pi));
end
theoryCorrelationMatrix = corrcoef(rankedTheoryMatrix, 'Rows', 'pairwise');

theoryInfo = struct();
theoryInfo.predictorNames = predictorNames;
theoryInfo.vectorization = 'upper triangle, excluding diagonal';
theoryInfo.correlation = 'Pearson correlation among tied-rank-transformed theoretical RDM vectors';
end

function rsaResults = run_semipartial_rsa(empiricalRDMTime, empiricalRDMSegment, theoryVectors, modelSets, cfg)
nTimeBins = size(empiricalRDMTime, 3);
upperMask = triu(true(size(empiricalRDMSegment, 1)), 1);
empSegmentVector = empiricalRDMSegment(upperMask);

rsaResults = struct();
for ms = 1:numel(modelSets)
    setName = modelSets(ms).name;
    predictors = modelSets(ms).predictors;
    nPred = numel(predictors);

    theoryMatrix = nan(numel(empSegmentVector), nPred);
    for pi = 1:nPred
        theoryMatrix(:, pi) = theoryVectors.(predictors{pi});
    end

    timeSemiR = nan(nPred, nTimeBins);
    timeBeta = nan(nPred, nTimeBins);
    timeDeltaR2 = nan(nPred, nTimeBins);
    timeR2Full = nan(1, nTimeBins);
    for bi = 1:nTimeBins
        rdmNow = empiricalRDMTime(:, :, bi);
        y = rdmNow(upperMask);
        res = semipartial_rank_regression(y, theoryMatrix);
        timeSemiR(:, bi) = res.semiPartialR;
        timeBeta(:, bi) = res.beta;
        timeDeltaR2(:, bi) = res.deltaR2;
        timeR2Full(bi) = res.R2Full;
    end

    segmentRes = semipartial_rank_regression(empSegmentVector, theoryMatrix);
    rankedTheoryMatrix = nan(size(theoryMatrix));
    for pi = 1:nPred
        rankedTheoryMatrix(:, pi) = tiedrank(theoryMatrix(:, pi));
    end
    predictorCorrelation = corrcoef(rankedTheoryMatrix, 'Rows', 'pairwise');
    vif = nan(nPred, 1);
    if nPred > 1 && all(isfinite(predictorCorrelation(:))) && rcond(predictorCorrelation) > 1e-10
        vif = diag(inv(predictorCorrelation));
    end
    highCorr = any(abs(predictorCorrelation(triu(true(nPred), 1))) > 0.80);
    highVif = any(vif > 5 & isfinite(vif));

    rsaResults.(setName) = struct();
    rsaResults.(setName).description = modelSets(ms).description;
    rsaResults.(setName).predictors = predictors;
    rsaResults.(setName).time = struct('semiPartialR', timeSemiR, ...
        'beta', timeBeta, 'deltaR2', timeDeltaR2, 'R2Full', timeR2Full);
    rsaResults.(setName).segmentAverage = segmentRes;
    rsaResults.(setName).predictorCorrelationMatrix = predictorCorrelation;
    rsaResults.(setName).vif = vif;
    rsaResults.(setName).diagnosticWarnings = struct( ...
        'highPredictorCorrelationAbsGreaterThanPoint8', highCorr, ...
        'highVifGreaterThan5', highVif, ...
        'diagnosticOnly', isfield(modelSets(ms), 'diagnosticOnly') && modelSets(ms).diagnosticOnly);
    rsaResults.(setName).cfg = cfg;
end
end

function res = semipartial_rank_regression(empVector, theoryMatrix)
valid = isfinite(empVector) & all(isfinite(theoryMatrix), 2);
yRaw = empVector(valid);
XRaw = theoryMatrix(valid, :);
nPred = size(XRaw, 2);

y = tiedrank(yRaw(:));
X = nan(size(XRaw));
for pi = 1:nPred
    X(:, pi) = tiedrank(XRaw(:, pi));
end

y = y - mean(y, 'omitnan');
for pi = 1:nPred
    X(:, pi) = X(:, pi) - mean(X(:, pi), 'omitnan');
end

XFull = [ones(size(X, 1), 1), X];
if rcond(XFull' * XFull) < 1e-10
    betaFull = pinv(XFull) * y;
else
    betaFull = XFull \ y;
end
yhatFull = XFull * betaFull;
sseFull = sum((y - yhatFull) .^ 2, 'omitnan');
sst = sum((y - mean(y, 'omitnan')) .^ 2, 'omitnan');
if sst <= eps
    R2Full = NaN;
else
    R2Full = 1 - sseFull / sst;
end

deltaR2 = nan(nPred, 1);
semiPartialR = nan(nPred, 1);
for pi = 1:nPred
    keepPred = true(1, nPred);
    keepPred(pi) = false;
    XReduced = [ones(size(X, 1), 1), X(:, keepPred)];
    if rcond(XReduced' * XReduced) < 1e-10
        betaReduced = pinv(XReduced) * y;
    else
        betaReduced = XReduced \ y;
    end
    yhatReduced = XReduced * betaReduced;
    sseReduced = sum((y - yhatReduced) .^ 2, 'omitnan');
    if sst <= eps
        R2Reduced = NaN;
    else
        R2Reduced = 1 - sseReduced / sst;
    end
    deltaR2(pi) = R2Full - R2Reduced;
    semiPartialR(pi) = sign(betaFull(pi + 1)) * sqrt(max(deltaR2(pi), 0));
end

res = struct();
res.semiPartialR = semiPartialR;
res.beta = betaFull(2:end);
res.intercept = betaFull(1);
res.R2Full = R2Full;
res.deltaR2 = deltaR2;
res.nPairs = numel(y);
res.rankTransform = 'tiedrank';
end

function stats = compute_group_stats(timeR, segmentR, predictors, timeBinInfo, cfg)
nPred = numel(predictors);
nTimeBins = size(timeR, 2);
timeTable = table();

pTime = nan(nPred, nTimeBins);
for pi = 1:nPred
    for bi = 1:nTimeBins
        x = squeeze(timeR(pi, bi, :));
        x = x(isfinite(x));
        if isempty(x)
            p = NaN;
        else
            p = signrank(x, 0);
        end
        pTime(pi, bi) = p;
    end
    [pAdj, sig] = correct_pvalues(pTime(pi, :), cfg.fdrMethod, cfg.alpha);
    for bi = 1:nTimeBins
        x = squeeze(timeR(pi, bi, :));
        x = x(isfinite(x));
        row = table();
        row.Predictor = predictors(pi);
        row.TimeBin = bi;
        row.TimeStartMsRelative = timeBinInfo.startMsRelative(bi);
        row.TimeEndMsRelative = timeBinInfo.endMsRelative(bi);
        row.TimeCenterMsRelative = timeBinInfo.centerMsRelative(bi);
        row.N = numel(x);
        row.MeanSemiPartialR = mean(x, 'omitnan');
        row.SEMSemiPartialR = std(x, 'omitnan') ./ sqrt(max(numel(x), 1));
        row.P = pTime(pi, bi);
        row.P_FDR = pAdj(bi);
        row.Significant_FDR = sig(bi);
        timeTable = [timeTable; row]; %#ok<AGROW>
    end
end

segmentTable = table();
for pi = 1:nPred
    x = squeeze(segmentR(pi, :))';
    x = x(isfinite(x));
    if isempty(x)
        p = NaN;
    else
        p = signrank(x, 0);
    end
    row = table();
    row.Predictor = predictors(pi);
    row.N = numel(x);
    row.MeanSemiPartialR = mean(x, 'omitnan');
    row.SEMSemiPartialR = std(x, 'omitnan') ./ sqrt(max(numel(x), 1));
    row.P = p;
    row.P_FDR = p;
    row.Significant_FDR = p < cfg.alpha;
    segmentTable = [segmentTable; row]; %#ok<AGROW>
end

stats = struct();
stats.predictors = predictors;
stats.timeSemiPartialR = timeR;
stats.segmentSemiPartialR = segmentR;
stats.timeTable = timeTable;
stats.segmentTable = segmentTable;
stats.alpha = cfg.alpha;
stats.correction = cfg.fdrMethod;
stats.test = 'Wilcoxon signed-rank against zero';
end

function make_rsa_figures(groupEmpiricalRDMSegment, theoryRDMs, theoryCorr, groupStats, groupSegmentR, modelSets, conditionNames, timeBinInfo, cfg)
if ~isfolder(cfg.figureDir)
    mkdir(cfg.figureDir);
end

fig = figure('Color', 'w', 'Position', [100 100 760 620]);
ax = axes(fig);
imageHandle = imagesc(ax, groupEmpiricalRDMSegment);
set(imageHandle, 'AlphaData', isfinite(groupEmpiricalRDMSegment));
set(ax, 'Color', 'w');
axis(ax, 'square');
colorbar(ax);
title(ax, 'Group-average empirical CDA RDM, segment average (unestimable pairs blank)');
set(ax, 'XTick', 1:numel(conditionNames), 'XTickLabel', conditionNames, ...
    'YTick', 1:numel(conditionNames), 'YTickLabel', conditionNames, ...
    'XTickLabelRotation', 45, 'TickDir', 'out');
print(fig, fullfile(cfg.figureDir, 'group_empirical_RDM_segment_average.png'), '-dpng', sprintf('-r%d', cfg.figureDpi));
savefig(fig, fullfile(cfg.figureDir, 'group_empirical_RDM_segment_average.fig'));
close(fig);

theoryNames = fieldnames(theoryRDMs);
fig = figure('Color', 'w', 'Position', [100 100 1200 900]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for ti = 1:numel(theoryNames)
    nexttile;
    imagesc(theoryRDMs.(theoryNames{ti}));
    axis square;
    colorbar;
    title(theoryNames{ti}, 'Interpreter', 'none');
    set(gca, 'XTick', 1:numel(conditionNames), 'XTickLabel', conditionNames, ...
        'YTick', 1:numel(conditionNames), 'YTickLabel', conditionNames, ...
        'XTickLabelRotation', 45, 'TickDir', 'out');
end
print(fig, fullfile(cfg.figureDir, 'theoretical_RDMs.png'), '-dpng', sprintf('-r%d', cfg.figureDpi));
savefig(fig, fullfile(cfg.figureDir, 'theoretical_RDMs.fig'));
close(fig);

fig = figure('Color', 'w', 'Position', [100 100 650 570]);
imagesc(theoryCorr, [-1 1]);
axis square;
colorbar;
title('Theory-vector correlation matrix');
set(gca, 'XTick', 1:numel(theoryNames), 'XTickLabel', theoryNames, ...
    'YTick', 1:numel(theoryNames), 'YTickLabel', theoryNames, ...
    'XTickLabelRotation', 45, 'TickDir', 'out');
print(fig, fullfile(cfg.figureDir, 'theory_predictor_correlation.png'), '-dpng', sprintf('-r%d', cfg.figureDpi));
savefig(fig, fullfile(cfg.figureDir, 'theory_predictor_correlation.fig'));
close(fig);

for ms = 1:min(2, numel(modelSets))
    setName = modelSets(ms).name;
    statsNow = groupStats.(setName);
    predictors = statsNow.predictors;
    centers = timeBinInfo.centerMsRelative;

    fig = figure('Color', 'w', 'Position', [100 100 850 520]);
    hold on;
    colors = lines(numel(predictors));
    for pi = 1:numel(predictors)
        rows = strcmp(statsNow.timeTable.Predictor, predictors{pi});
        T = statsNow.timeTable(rows, :);
        y = T.MeanSemiPartialR;
        se = T.SEMSemiPartialR;
        fill([centers; flipud(centers)], [y - se; flipud(y + se)], colors(pi, :), ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none');
        plot(centers, y, 'LineWidth', 2, 'Color', colors(pi, :), 'DisplayName', predictors{pi});
        sigY = min(ylim);
        sigBins = T.Significant_FDR;
        if any(sigBins)
            plot(centers(sigBins), repmat(sigY, sum(sigBins), 1), '.', ...
                'Color', colors(pi, :), 'MarkerSize', 12, 'HandleVisibility', 'off');
        end
    end
    yline(0, 'k:');
    xlabel('Time in maintenance segment (ms)');
    ylabel('Semipartial r');
    title(sprintf('%s time-resolved semipartial RSA', setName), 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');
    box off;
    print(fig, fullfile(cfg.figureDir, sprintf('%s_time_semipartial_curves.png', setName)), '-dpng', sprintf('-r%d', cfg.figureDpi));
    savefig(fig, fullfile(cfg.figureDir, sprintf('%s_time_semipartial_curves.fig', setName)));
    close(fig);

    fig = figure('Color', 'w', 'Position', [100 100 760 520]);
    x = 1:numel(predictors);
    y = statsNow.segmentTable.MeanSemiPartialR;
    se = statsNow.segmentTable.SEMSemiPartialR;
    bar(x, y, 0.65, 'FaceColor', [0.55 0.60 0.66], 'EdgeColor', 'none');
    hold on;
    errorbar(x, y, se, 'k.', 'LineWidth', 1.2);
    for pi = 1:numel(predictors)
        dots = squeeze(groupSegmentR.(setName)(pi, :));
        scatter(repmat(x(pi), size(dots)) + (rand(size(dots)) - 0.5) * 0.18, dots, ...
            26, 'k', 'filled', 'MarkerFaceAlpha', 0.55);
    end
    yline(0, 'k:');
    set(gca, 'XTick', x, 'XTickLabel', predictors, 'XTickLabelRotation', 25, 'TickDir', 'out');
    ylabel('Semipartial r');
    title(sprintf('%s segment-average semipartial RSA', setName), 'Interpreter', 'none');
    box off;
    print(fig, fullfile(cfg.figureDir, sprintf('%s_segment_average_bars.png', setName)), '-dpng', sprintf('-r%d', cfg.figureDpi));
    savefig(fig, fullfile(cfg.figureDir, sprintf('%s_segment_average_bars.fig', setName)));
    close(fig);
end
end

%% data1 simultaneous-array CDA neural RSA
% Builds subject-level neural RDMs directly from trial-level CDA patterns.
% Conditions and RDM vector order are fixed as SS1, SS3, SS6 and
% [D13, D16, D36]. The primary distance is random-half crossnobis/LDC.

% All items were presented simultaneously. Therefore, D36 tests
% post-capacity differentiation / supra-capacity separability, not a
% post-capacity ramp.

% Required input:
%   data1/data/*_EEG_timeLockMem.mat
% Required project functions:
%   data1_subject_inclusion.m
%   parse_subject_id_from_filename.m

% Default output:
%   data1/data1_simultaneous_neural_rsa/

% Optional smoke-test overrides (default behavior is unchanged):
%   DATA1_RSA_SUBJECT      numeric subject id
%   DATA1_RSA_N_ITER       number of RDM balancing/split iterations
%   DATA1_RSA_N_PERM       number of sign-flip permutations
%   DATA1_RSA_OUTPUT_DIR   absolute output directory
%   DATA1_RSA_FIGURES      0/false/no disables figures

% The data1 task timing is 250 ms memory array plus 1300 ms retention.
% Time zero is memory-array onset (data1/README_DataExp1.md).

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data1');
eegDir = fullfile(dataDir, 'data');
outputDir = fullfile(dataDir, 'data1_simultaneous_neural_rsa');
addpath(codeDir);

%% Configuration
cfg = struct();
cfg.analysisName = 'data1_simultaneous_neural_rsa';
cfg.condNames = {'SS1','SS3','SS6'};
cfg.loadVals = [1 3 6];
cfg.conditionIndexBySide = [1 2 3; 4 5 6]; % rows: attend left/right
cfg.sideNames = {'attendLeft','attendRight'};
cfg.leftElectrodes = {'O1','OL','P3','PO3','T5'};
cfg.rightElectrodes = {'O2','OR','P4','PO4','T6'};
cfg.retentionWindowMs = [250 1550];
cfg.timeBinWidthMs = 50;
cfg.nIter = 100;
cfg.nPerm = 100000;
cfg.minTrialsPerSetSize = 75;
cfg.covShrinkage = 0.10;
cfg.ridgeScale = 1e-6;
cfg.minCovRcond = 1e-10;
cfg.randomSeed = 20260619;
cfg.alpha = 0.05;
cfg.makeFigures = true;
cfg.figureDpi = 200;
cfg.distanceMetric = 'crossnobis/LDC';
cfg.featureDefinition = ['five paired posterior CDA electrodes x 50-ms retention bins; ' ...
    'contra-minus-ipsilateral voltage'];
cfg.sideHandling = ['attend-left and attend-right crossnobis RDMs computed separately ' ...
    'after equal trial sampling, then averaged'];

subjectOverride = str2double(strtrim(char(getenv('DATA1_RSA_SUBJECT'))));
nIterOverride = str2double(strtrim(char(getenv('DATA1_RSA_N_ITER'))));
nPermOverride = str2double(strtrim(char(getenv('DATA1_RSA_N_PERM'))));
outputOverride = strtrim(char(getenv('DATA1_RSA_OUTPUT_DIR')));
figureOverride = strtrim(char(getenv('DATA1_RSA_FIGURES')));
if isfinite(nIterOverride) && nIterOverride > 0
    cfg.nIter = round(nIterOverride);
end
if isfinite(nPermOverride) && nPermOverride > 0
    cfg.nPerm = round(nPermOverride);
end
if ~isempty(outputOverride)
    outputDir = outputOverride;
end
if ~isempty(figureOverride)
    cfg.makeFigures = ~ismember(lower(figureOverride), {'0','false','no'});
end
cfg.outputDir = outputDir;

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Input files and fixed theoretical RDMs
files = dir(fullfile(eegDir, '*_EEG_timeLockMem.mat'));
if isempty(files)
    error('No data1 EEG files found in %s.', eegDir);
end

fileSubject = nan(numel(files), 1);
for fi = 1:numel(files)
    fileSubject(fi) = parse_subject_id_from_filename(files(fi).name);
end
[fileSubject, sortIdx] = sort(fileSubject);
files = files(sortIdx);
if isfinite(subjectOverride)
    keepFile = fileSubject == subjectOverride;
    files = files(keepFile);
    fileSubject = fileSubject(keepFile);
    if isempty(files)
        error('Requested DATA1_RSA_SUBJECT=%g was not found.', subjectOverride);
    end
end

condNames = cfg.condNames;
loadVals = cfg.loadVals;
capacityState = min(loadVals, 3);
linearLoad = loadVals;
supraBinary = double(loadVals > 3);
lowVsHigher = double(loadVals > 1);

modelCapacity = abs(capacityState' - capacityState);
modelLinear = abs(linearLoad' - linearLoad);
modelSupra = abs(supraBinary' - supraBinary);
modelLowVsHigher = abs(lowVsHigher' - lowVsHigher);
modelVectors = [modelCapacity(1,2), modelCapacity(1,3), modelCapacity(2,3); ...
                modelLinear(1,2), modelLinear(1,3), modelLinear(2,3); ...
                modelSupra(1,2), modelSupra(1,3), modelSupra(2,3); ...
                modelLowVsHigher(1,2), modelLowVsHigher(1,3), modelLowVsHigher(2,3)];
modelNames = {'capacityPlateau','linearLoad','supraDiff','lowVsHigher'};

theoryCorrelation = corr(modelVectors', 'Type', 'Spearman', 'Rows', 'pairwise');
TheoryModelCorrelation = array2table(theoryCorrelation, 'VariableNames', modelNames);
TheoryModelCorrelation = addvars(TheoryModelCorrelation, string(modelNames(:)), ...
    'Before', 1, 'NewVariableNames', 'Model');
writetable(TheoryModelCorrelation, fullfile(outputDir, 'theory_model_correlation.csv'));

%% Build subject-level neural RDMs from trial-level CDA features
nFile = numel(files);
includedSubject = nan(nFile, 1);
subjectDistance = nan(nFile, 3); % [D13 D16 D36]
subjectDistanceSide = nan(nFile, 3, 2);
subjectCleanCount = nan(nFile, 6); % SS1/3/6 x attend-left/right
subjectBalancedPerCondition = nan(nFile, 1);
subjectValidIterations = nan(nFile, 1);
inclusionSubject = nan(nFile, 1);
inclusionPassed = false(nFile, 1);
inclusionTrialCounts = nan(nFile, 3);
inclusionMinCount = nan(nFile, 1);
nIncluded = 0;

pairCondition = [1 2; 1 3; 2 3];

for fileIdx = 1:nFile
    subject = fileSubject(fileIdx);
    fprintf('\nData1 neural RSA: subject %d (%d/%d)\n', subject, fileIdx, nFile);
    S = load(fullfile(files(fileIdx).folder, files(fileIdx).name), 'eeg');
    eeg = S.eeg;

    [includeSubject, inclusionInfo] = data1_subject_inclusion(eeg, cfg.minTrialsPerSetSize);
    inclusionSubject(fileIdx) = subject;
    inclusionPassed(fileIdx) = includeSubject;
    inclusionTrialCounts(fileIdx,:) = inclusionInfo.trialCountsPerSetSize;
    inclusionMinCount(fileIdx) = inclusionInfo.minTrialCount;
    if ~includeSubject
        fprintf('  Skip: set-size clean-trial counts [%s] fail >=%d criterion.\n', ...
            sprintf('%d ', inclusionInfo.trialCountsPerSetSize), cfg.minTrialsPerSetSize);
        continue;
    end

    time = double(eeg.time(:)');
    retentionEndMs = min(cfg.retentionWindowMs(2), time(end));
    retentionMask = time >= cfg.retentionWindowMs(1) & time <= retentionEndMs;
    if ~any(retentionMask)
        error('Subject %d has no samples in the requested retention window.', subject);
    end
    retentionTime = time(retentionMask);
    binStarts = cfg.retentionWindowMs(1):cfg.timeBinWidthMs:retentionEndMs;
    binStarts = binStarts(binStarts < retentionEndMs);
    nTimeBin = numel(binStarts);

    [foundLeft, leftIdx] = ismember(cfg.leftElectrodes, eeg.chanLabels);
    [foundRight, rightIdx] = ismember(cfg.rightElectrodes, eeg.chanLabels);
    if ~all(foundLeft) || ~all(foundRight)
        error('Subject %d is missing one or more project CDA electrodes.', subject);
    end

    featureData = cell(2, 3);
    cleanCount = nan(2, 3);
    nTrialStored = size(eeg.baselined, 2);
    nRetentionSample = nnz(retentionMask);

    for sideIdx = 1:2
        if sideIdx == 1
            contraIdx = rightIdx;
            ipsiIdx = leftIdx;
        else
            contraIdx = leftIdx;
            ipsiIdx = rightIdx;
        end

        for condIdx = 1:3
            eegCondition = cfg.conditionIndexBySide(sideIdx, condIdx);
            contra = reshape(eeg.baselined(eegCondition,:,contraIdx,retentionMask), ...
                [nTrialStored, numel(contraIdx), nRetentionSample]);
            ipsi = reshape(eeg.baselined(eegCondition,:,ipsiIdx,retentionMask), ...
                [nTrialStored, numel(ipsiIdx), nRetentionSample]);
            trialCDA = contra - ipsi;

            finiteTrial = all(isfinite(reshape(trialCDA, nTrialStored, [])), 2);
            keepTrial = ~logical(eeg.arf.artifactInd(eegCondition,:))' & finiteTrial;
            trialCDA = trialCDA(keepTrial,:,:);
            if isempty(trialCDA)
                error('Cannot find trial-level or condition-level CDA neural patterns for neural RSA.');
            end

            trialFeatures = nan(size(trialCDA,1), numel(contraIdx) * nTimeBin);
            for binIdx = 1:nTimeBin
                if binIdx < nTimeBin
                    thisTime = retentionTime >= binStarts(binIdx) & ...
                        retentionTime < binStarts(binIdx + 1);
                else
                    thisTime = retentionTime >= binStarts(binIdx) & ...
                        retentionTime <= retentionEndMs;
                end
                binFeature = mean(trialCDA(:,:,thisTime), 3, 'omitnan');
                colIdx = (binIdx - 1) * numel(contraIdx) + (1:numel(contraIdx));
                trialFeatures(:,colIdx) = binFeature;
            end

            if any(~isfinite(trialFeatures(:)))
                error('Subject %d %s %s produced non-finite CDA features.', ...
                    subject, cfg.sideNames{sideIdx}, condNames{condIdx});
            end
            featureData{sideIdx, condIdx} = trialFeatures;
            cleanCount(sideIdx, condIdx) = size(trialFeatures, 1);
        end
    end

    nUsePerSideCondition = 2 * floor(min(cleanCount(:)) / 2);
    if nUsePerSideCondition < 4
        error('Subject %d has fewer than four usable trials in at least one side/load cell.', subject);
    end
    nFeature = size(featureData{1,1}, 2);
    distanceIterSide = nan(cfg.nIter, 3, 2);
    ridgeIterSide = nan(cfg.nIter, 2);

    rng(cfg.randomSeed + subject, 'twister');
    for iterIdx = 1:cfg.nIter
        for sideIdx = 1:2
            trainMean = nan(3, nFeature);
            testMean = nan(3, nFeature);
            trainResiduals = [];

            for condIdx = 1:3
                X = featureData{sideIdx, condIdx};
                selected = randperm(size(X,1), nUsePerSideCondition);
                nHalf = nUsePerSideCondition / 2;
                trainX = X(selected(1:nHalf),:);
                testX = X(selected((nHalf + 1):end),:);
                trainMean(condIdx,:) = mean(trainX, 1);
                testMean(condIdx,:) = mean(testX, 1);
                trainResiduals = [trainResiduals; trainX - trainMean(condIdx,:)]; %#ok<AGROW>
            end

            C = cov(trainResiduals, 1);
            C = (C + C') / 2;
            targetScale = trace(C) / nFeature;
            if ~isfinite(targetScale) || targetScale <= 0
                targetScale = mean(diag(C), 'omitnan');
            end
            if ~isfinite(targetScale) || targetScale <= 0
                targetScale = 1;
            end
            Creg = (1 - cfg.covShrinkage) * C + ...
                cfg.covShrinkage * targetScale * eye(nFeature);
            ridge = cfg.ridgeScale * targetScale;
            Creg = Creg + ridge * eye(nFeature);
            rcondNow = rcond(Creg);
            while rcondNow < cfg.minCovRcond
                ridge = ridge * 10;
                Creg = Creg + ridge * eye(nFeature);
                rcondNow = rcond(Creg);
                if ridge > targetScale
                    break;
                end
            end
            ridgeIterSide(iterIdx, sideIdx) = ridge;

            for pairIdx = 1:3
                condI = pairCondition(pairIdx,1);
                condJ = pairCondition(pairIdx,2);
                diffTrain = trainMean(condI,:) - trainMean(condJ,:);
                diffTest = testMean(condI,:) - testMean(condJ,:);
                distanceIterSide(iterIdx,pairIdx,sideIdx) = ...
                    (diffTrain / Creg) * diffTest';
            end
        end
    end

    validIteration = squeeze(all(all(isfinite(distanceIterSide), 2), 3));
    if ~any(validIteration)
        error('Subject %d produced no valid crossnobis iterations.', subject);
    end
    distanceIterSide = distanceIterSide(validIteration,:,:);
    meanBySide = squeeze(mean(distanceIterSide, 1, 'omitnan'));
    if size(meanBySide,1) ~= 3
        meanBySide = meanBySide';
    end
    meanAcrossSide = mean(meanBySide, 2, 'omitnan')';

    nIncluded = nIncluded + 1;
    includedSubject(nIncluded) = subject;
    subjectDistance(nIncluded,:) = meanAcrossSide;
    subjectDistanceSide(nIncluded,:,:) = reshape(meanBySide, [1 3 2]);
    subjectCleanCount(nIncluded,:) = [cleanCount(1,:) cleanCount(2,:)];
    subjectBalancedPerCondition(nIncluded) = 2 * nUsePerSideCondition;
    subjectValidIterations(nIncluded) = sum(validIteration);

    fprintf('  Clean trials L=[%s], R=[%s], balanced/condition=%d, features=%d.\n', ...
        sprintf('%d ', cleanCount(1,:)), sprintf('%d ', cleanCount(2,:)), ...
        2 * nUsePerSideCondition, nFeature);
    fprintf('  D13=%.6g, D16=%.6g, D36=%.6g\n', meanAcrossSide);
end

if nIncluded == 0
    error('No subjects produced valid neural RSA outputs.');
end

includedSubject = includedSubject(1:nIncluded);
subjectDistance = subjectDistance(1:nIncluded,:);
subjectDistanceSide = subjectDistanceSide(1:nIncluded,:,:);
subjectCleanCount = subjectCleanCount(1:nIncluded,:);
subjectBalancedPerCondition = subjectBalancedPerCondition(1:nIncluded);
subjectValidIterations = subjectValidIterations(1:nIncluded);

Inclusion = table(inclusionSubject, inclusionPassed, inclusionMinCount, ...
    inclusionTrialCounts(:,1), inclusionTrialCounts(:,2), inclusionTrialCounts(:,3), ...
    'VariableNames', {'Subject','Included','MinCleanTrialCount','N_SS1','N_SS3','N_SS6'});
writetable(Inclusion, fullfile(outputDir, 'subject_inclusion.csv'));

D13 = subjectDistance(:,1);
D16 = subjectDistance(:,2);
D36 = subjectDistance(:,3);
NeuralRDMSubject = table(includedSubject, D13, D16, D36, ...
    subjectCleanCount(:,1), subjectCleanCount(:,2), subjectCleanCount(:,3), ...
    subjectCleanCount(:,4), subjectCleanCount(:,5), subjectCleanCount(:,6), ...
    subjectBalancedPerCondition, subjectValidIterations, ...
    'VariableNames', {'Subject','D13_neural','D16_neural','D36_neural', ...
    'N_SS1_AttendLeft','N_SS3_AttendLeft','N_SS6_AttendLeft', ...
    'N_SS1_AttendRight','N_SS3_AttendRight','N_SS6_AttendRight', ...
    'N_BalancedPerCondition','N_ValidIterations'});
writetable(NeuralRDMSubject, fullfile(outputDir, 'neural_rdm_subject.csv'));

%% Neural-distance and planned-contrast sign-flip statistics
distanceNames = {'D13_neural','D16_neural','D36_neural'};
distanceData = [D13 D16 D36];
distanceMean = nan(3,1);
distanceSEM = nan(3,1);
distanceCI_low = nan(3,1);
distanceCI_high = nan(3,1);
distanceP = nan(3,1);
distanceN = nan(3,1);
for testIdx = 1:3
    statNow = sign_flip_test(distanceData(:,testIdx), cfg.nPerm, 'greater', ...
        cfg.randomSeed + 1000 + testIdx);
    distanceMean(testIdx) = statNow.mean;
    distanceSEM(testIdx) = statNow.sem;
    distanceCI_low(testIdx) = statNow.ci95(1);
    distanceCI_high(testIdx) = statNow.ci95(2);
    distanceP(testIdx) = statNow.p;
    distanceN(testIdx) = statNow.n;
end
NeuralRDMGroupStats = table(string(distanceNames(:)), repmat("greater_than_0",3,1), ...
    distanceMean, distanceSEM, distanceCI_low, distanceCI_high, distanceP, distanceN, ...
    'VariableNames', {'Test','Alternative','Mean','SEM','CI95_Low','CI95_High','PValue','N'});
writetable(NeuralRDMGroupStats, fullfile(outputDir, 'neural_rdm_group_stats.csv'));

capacityLoadingDiff = D13;
postCapacityDiff = D36;
fullLoadSpan = D16;
plateauViolationIndex = D36;
supraRelativeToCapacity = D36 - D13;
loadSpanGain = D16 - D13;
PlannedContrastSubject = table(includedSubject, capacityLoadingDiff, postCapacityDiff, ...
    fullLoadSpan, plateauViolationIndex, supraRelativeToCapacity, loadSpanGain, ...
    'VariableNames', {'Subject','capacityLoadingDiff','postCapacityDiff', ...
    'fullLoadSpan','plateauViolationIndex','supraRelativeToCapacity','loadSpanGain'});
writetable(PlannedContrastSubject, fullfile(outputDir, 'planned_contrast_subject.csv'));

plannedTestNames = {'D13 > 0','D36 > 0','D16 > 0','D16 > D13','D36 < D13','D16 > D36'};
plannedExpression = {'D13','D36','D16','D16-D13','D13-D36','D16-D36'};
plannedAlternative = repmat({'greater'}, 6, 1);
plannedData = [D13, D36, D16, D16-D13, D13-D36, D16-D36];
plannedMean = nan(6,1);
plannedSEM = nan(6,1);
plannedCI_low = nan(6,1);
plannedCI_high = nan(6,1);
plannedP = nan(6,1);
plannedN = nan(6,1);
for testIdx = 1:6
    statNow = sign_flip_test(plannedData(:,testIdx), cfg.nPerm, 'greater', ...
        cfg.randomSeed + 2000 + testIdx);
    plannedMean(testIdx) = statNow.mean;
    plannedSEM(testIdx) = statNow.sem;
    plannedCI_low(testIdx) = statNow.ci95(1);
    plannedCI_high(testIdx) = statNow.ci95(2);
    plannedP(testIdx) = statNow.p;
    plannedN(testIdx) = statNow.n;
end
PlannedContrastGroupStats = table(string(plannedTestNames(:)), string(plannedExpression(:)), ...
    string(plannedAlternative), plannedMean, plannedSEM, plannedCI_low, plannedCI_high, ...
    plannedP, plannedN, 'VariableNames', {'Test','TestedValue','Alternative','Mean', ...
    'SEM','CI95_Low','CI95_High','PValue','N'});
writetable(PlannedContrastGroupStats, fullfile(outputDir, 'planned_contrast_group_stats.csv'));

%% Lightweight theoretical-model fits
nModel = numel(modelNames);
modelFitSpearman = nan(nIncluded, nModel);
modelFitNegSSE = nan(nIncluded, nModel);
for subjectIdx = 1:nIncluded
    y = subjectDistance(subjectIdx,:);
    ySD = std(y, 0, 2);
    if isfinite(ySD) && ySD > 0
        zy = (y - mean(y)) ./ ySD;
    else
        zy = nan(size(y));
    end
    for modelIdx = 1:nModel
        modelNow = modelVectors(modelIdx,:);
        modelFitSpearman(subjectIdx,modelIdx) = corr(y(:), modelNow(:), ...
            'Type', 'Spearman', 'Rows', 'complete');
        zm = (modelNow - mean(modelNow)) ./ std(modelNow, 0, 2);
        modelFitNegSSE(subjectIdx,modelIdx) = -sum((zy - zm).^2);
    end
end

RSAModelFitSubject = table(repmat(includedSubject, nModel, 1), ...
    repelem(string(modelNames(:)), nIncluded), modelFitSpearman(:), modelFitNegSSE(:), ...
    'VariableNames', {'Subject','Model','SpearmanFit','NegSSEFit'});
writetable(RSAModelFitSubject, fullfile(outputDir, 'rsa_model_fit_subject.csv'));

comparisonNames = {'linearLoad - capacityPlateau', ...
    'supraDiff - capacityPlateau', ...
    'linearLoad - supraDiff', ...
    'capacityPlateau - lowVsHigher'};
comparisonA = [2 3 2 1];
comparisonB = [1 1 3 4];
nComparison = numel(comparisonNames);
deltaSpearman = nan(nIncluded, nComparison);
deltaNegSSE = nan(nIncluded, nComparison);
for comparisonIdx = 1:nComparison
    deltaSpearman(:,comparisonIdx) = modelFitSpearman(:,comparisonA(comparisonIdx)) - ...
        modelFitSpearman(:,comparisonB(comparisonIdx));
    deltaNegSSE(:,comparisonIdx) = modelFitNegSSE(:,comparisonA(comparisonIdx)) - ...
        modelFitNegSSE(:,comparisonB(comparisonIdx));
end

fitMetric = strings(nComparison * 2, 1);
fitComparison = strings(nComparison * 2, 1);
fitMean = nan(nComparison * 2, 1);
fitSEM = nan(nComparison * 2, 1);
fitCI_low = nan(nComparison * 2, 1);
fitCI_high = nan(nComparison * 2, 1);
fitP = nan(nComparison * 2, 1);
fitN = nan(nComparison * 2, 1);
rowIdx = 0;
for metricIdx = 1:2
    if metricIdx == 1
        deltaNow = deltaSpearman;
        metricName = 'SpearmanFit';
    else
        deltaNow = deltaNegSSE;
        metricName = 'NegSSEFit';
    end
    for comparisonIdx = 1:nComparison
        rowIdx = rowIdx + 1;
        statNow = sign_flip_test(deltaNow(:,comparisonIdx), cfg.nPerm, 'two-sided', ...
            cfg.randomSeed + 3000 + rowIdx);
        fitMetric(rowIdx) = metricName;
        fitComparison(rowIdx) = comparisonNames{comparisonIdx};
        fitMean(rowIdx) = statNow.mean;
        fitSEM(rowIdx) = statNow.sem;
        fitCI_low(rowIdx) = statNow.ci95(1);
        fitCI_high(rowIdx) = statNow.ci95(2);
        fitP(rowIdx) = statNow.p;
        fitN(rowIdx) = statNow.n;
    end
end
RSAModelFitComparisonStats = table(fitMetric, fitComparison, repmat("two-sided",rowIdx,1), ...
    fitMean, fitSEM, fitCI_low, fitCI_high, fitP, fitN, ...
    'VariableNames', {'Metric','Comparison','Alternative','MeanDelta','SEM', ...
    'CI95_Low','CI95_High','PValue','N'});
writetable(RSAModelFitComparisonStats, fullfile(outputDir, 'rsa_model_fit_comparison_stats.csv'));

%% Optional supplementary decoding-based RDM from existing pairwise LDA outputs
decoderDirs = {fullfile(dataDir, 'decoding_LDA_spatialControl', 'loadWithinSide_setsize1_vs3', 'CDA'), ...
               fullfile(dataDir, 'decoding_LDA_spatialControl', 'loadWithinSide', 'CDA'), ...
               fullfile(dataDir, 'decoding_LDA_spatialControl', 'loadWithinSide_setsize3_vs6', 'CDA')};
decodingAvailable = all(cellfun(@isfolder, decoderDirs));
DecodingRDMSubject = table();
DecodingRDMGroupStats = table();
decodingP_D36 = NaN;
if decodingAvailable
    decodingDistance = nan(nIncluded, 3);
    for subjectIdx = 1:nIncluded
        subject = includedSubject(subjectIdx);
        resultName = sprintf('%d_EEG_timeLockMem.mat', subject);
        for pairIdx = 1:3
            resultFile = fullfile(decoderDirs{pairIdx}, resultName);
            if ~isfile(resultFile)
                continue;
            end
            R = load(resultFile, 'Result');
            auc = R.Result.AUC;
            if ismatrix(auc) && size(auc,1) == size(auc,2)
                auc = diag(auc);
            else
                auc = auc(:);
            end
            decoderTime = R.Result.times(:);
            maintenanceIdx = decoderTime >= cfg.retentionWindowMs(1) & ...
                decoderTime <= cfg.retentionWindowMs(2);
            if any(maintenanceIdx) && numel(auc) == numel(decoderTime)
                decodingDistance(subjectIdx,pairIdx) = mean(auc(maintenanceIdx), 'omitnan') - 0.5;
            end
        end
    end

    completeDecoding = all(isfinite(decodingDistance), 2);
    if any(completeDecoding)
        DecodingRDMSubject = table(includedSubject(completeDecoding), ...
            decodingDistance(completeDecoding,1), decodingDistance(completeDecoding,2), ...
            decodingDistance(completeDecoding,3), ...
            'VariableNames', {'Subject','D13_decoding','D16_decoding','D36_decoding'});
        writetable(DecodingRDMSubject, fullfile(outputDir, 'decoding_rdm_subject.csv'));

        decodingMean = nan(3,1);
        decodingSEM = nan(3,1);
        decodingCI_low = nan(3,1);
        decodingCI_high = nan(3,1);
        decodingP = nan(3,1);
        decodingN = nan(3,1);
        for testIdx = 1:3
            statNow = sign_flip_test(decodingDistance(completeDecoding,testIdx), ...
                cfg.nPerm, 'greater', cfg.randomSeed + 4000 + testIdx);
            decodingMean(testIdx) = statNow.mean;
            decodingSEM(testIdx) = statNow.sem;
            decodingCI_low(testIdx) = statNow.ci95(1);
            decodingCI_high(testIdx) = statNow.ci95(2);
            decodingP(testIdx) = statNow.p;
            decodingN(testIdx) = statNow.n;
        end
        decodingDistanceNames = {'D13_decoding','D16_decoding','D36_decoding'};
        DecodingRDMGroupStats = table(string(decodingDistanceNames(:)), repmat("greater_than_0",3,1), ...
            decodingMean, decodingSEM, decodingCI_low, decodingCI_high, decodingP, decodingN, ...
            'VariableNames', {'Test','Alternative','Mean','SEM','CI95_Low','CI95_High','PValue','N'});
        writetable(DecodingRDMGroupStats, fullfile(outputDir, 'decoding_rdm_group_stats.csv'));
        decodingP_D36 = decodingP(3);
    else
        decodingAvailable = false;
    end
end

%% Figures
groupMeanRDM = [0 mean(D13,'omitnan') mean(D16,'omitnan'); ...
                mean(D13,'omitnan') 0 mean(D36,'omitnan'); ...
                mean(D16,'omitnan') mean(D36,'omitnan') 0];
if cfg.makeFigures
    fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 720 620]);
    ax = axes(fig);
    imagesc(ax, groupMeanRDM);
    axis(ax, 'square');
    colorbar(ax);
    colormap(ax, parula);
    set(ax, 'XTick', 1:3, 'XTickLabel', condNames, ...
        'YTick', 1:3, 'YTickLabel', condNames, 'FontSize', 11);
    title(ax, sprintf('Group empirical CDA crossnobis RDM (N=%d)', nIncluded));
    for row = 1:3
        for col = 1:3
            text(ax, col, row, sprintf('%.3g', groupMeanRDM(row,col)), ...
                'HorizontalAlignment', 'center', 'Color', 'k', 'FontWeight', 'bold');
        end
    end
    exportgraphics(fig, fullfile(outputDir, 'empirical_neural_rdm_group.png'), ...
        'Resolution', cfg.figureDpi);
    close(fig);

    fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 820 560]);
    ax = axes(fig);
    draw_subject_bar(ax, [D13 D36 D16], {'D13','D36','D16'}, ...
        sprintf('Pairwise CDA neural distances; p(D36 > 0) = %.4g', distanceP(3)), ...
        'Crossnobis distance', cfg.randomSeed + 5001);
    exportgraphics(fig, fullfile(outputDir, 'pairwise_neural_distances_bar.png'), ...
        'Resolution', cfg.figureDpi);
    close(fig);

    fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1050 570]);
    ax = axes(fig);
    contrastPlotData = [capacityLoadingDiff postCapacityDiff fullLoadSpan ...
        plateauViolationIndex supraRelativeToCapacity loadSpanGain];
    draw_subject_bar(ax, contrastPlotData, ...
        {'capacityLoading','postCapacity','fullLoadSpan','plateauViolation','supraRelCapacity','loadSpanGain'}, ...
        'Planned CDA neural-RDM contrasts', 'Contrast value', cfg.randomSeed + 5002);
    exportgraphics(fig, fullfile(outputDir, 'planned_contrast_bar.png'), ...
        'Resolution', cfg.figureDpi);
    close(fig);

    fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1200 520]);
    ax1 = subplot(1,2,1, 'Parent', fig);
    draw_subject_bar(ax1, modelFitSpearman, modelNames, 'Spearman model fit', ...
        'Spearman rho', cfg.randomSeed + 5003);
    ax2 = subplot(1,2,2, 'Parent', fig);
    draw_subject_bar(ax2, modelFitNegSSE, modelNames, 'Negative-SSE model fit', ...
        'Negative SSE', cfg.randomSeed + 5004);
    exportgraphics(fig, fullfile(outputDir, 'model_fit_bar.png'), ...
        'Resolution', cfg.figureDpi);
    close(fig);

    shortComparisonLabels = {'linear-capacity','supra-capacity','linear-supra','capacity-lowHigher'};
    fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1200 520]);
    ax1 = subplot(1,2,1, 'Parent', fig);
    draw_subject_bar(ax1, deltaSpearman, shortComparisonLabels, ...
        'Spearman fit differences', 'Delta fit', cfg.randomSeed + 5005);
    ax2 = subplot(1,2,2, 'Parent', fig);
    draw_subject_bar(ax2, deltaNegSSE, shortComparisonLabels, ...
        'Negative-SSE fit differences', 'Delta fit', cfg.randomSeed + 5006);
    exportgraphics(fig, fullfile(outputDir, 'model_fit_delta_bar.png'), ...
        'Resolution', cfg.figureDpi);
    close(fig);

    fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 760 650]);
    ax = axes(fig);
    imagesc(ax, theoryCorrelation, [-1 1]);
    axis(ax, 'square');
    colorbar(ax);
    colormap(ax, parula);
    set(ax, 'XTick', 1:nModel, 'XTickLabel', modelNames, ...
        'YTick', 1:nModel, 'YTickLabel', modelNames, 'FontSize', 10);
    xtickangle(ax, 25);
    title(ax, 'Spearman correlation among theoretical RDM vectors');
    for row = 1:nModel
        for col = 1:nModel
            text(ax, col, row, sprintf('%.2f', theoryCorrelation(row,col)), ...
                'HorizontalAlignment', 'center', 'Color', 'k', 'FontWeight', 'bold');
        end
    end
    exportgraphics(fig, fullfile(outputDir, 'theory_model_correlation.png'), ...
        'Resolution', cfg.figureDpi);
    close(fig);
end

%% Automatic interpretation summary and MATLAB result bundle
pD36 = distanceP(3);
meanD36 = distanceMean(3);
sdD36 = std(D36, 0, 'omitnan');
if isfinite(sdD36) && sdD36 > 0
    dzD36 = meanD36 / sdD36;
else
    dzD36 = NaN;
end

summaryFile = fullfile(outputDir, 'data1_simultaneous_neural_rsa_summary.txt');
fid = fopen(summaryFile, 'w');
if fid < 0
    error('Cannot open summary file for writing: %s', summaryFile);
end
fprintf(fid, 'data1 simultaneous-array CDA neural RSA\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Subjects included: %d of %d files.\n', nIncluded, nFile);
fprintf(fid, 'Conditions: SS1, SS3, SS6; RDM vector order: [D13, D16, D36].\n');
fprintf(fid, 'Primary metric: %s.\n', cfg.distanceMetric);
fprintf(fid, 'Feature definition: %s.\n', cfg.featureDefinition);
fprintf(fid, 'Retention window: %.0f to %.0f ms; nominal bin width: %.0f ms.\n', ...
    cfg.retentionWindowMs(1), min(cfg.retentionWindowMs(2), retentionEndMs), cfg.timeBinWidthMs);
fprintf(fid, 'Side handling: %s.\n', cfg.sideHandling);
fprintf(fid, ['Trial balancing: each iteration sampled the same even number of trials from every ' ...
    'set-size x attended-side cell; %d random-half iterations were averaged.\n'], cfg.nIter);
fprintf(fid, ['Crossnobis covariance used the same shrinkage-plus-ridge LDC logic as the existing ' ...
    'data3 neural RSA (shrinkage %.3f; ridge scale %.3g).\n'], ...
    cfg.covShrinkage, cfg.ridgeScale);
fprintf(fid, ['Group tests: subject-level sign-flip permutation tests with %d permutations; ' ...
    '95%% confidence intervals are subject-level t intervals.\n\n'], cfg.nPerm);

fprintf(fid, 'Primary planned distances\n');
for testIdx = 1:3
    fprintf(fid, '%s: mean=%.8g, SEM=%.8g, 95%% CI=[%.8g, %.8g], p(one-sided >0)=%.8g, N=%d.\n', ...
        distanceNames{testIdx}, distanceMean(testIdx), distanceSEM(testIdx), ...
        distanceCI_low(testIdx), distanceCI_high(testIdx), distanceP(testIdx), distanceN(testIdx));
end
fprintf(fid, '\n');

fprintf(fid, ['This analysis tested whether CDA neural patterns remain distinguishable between ' ...
    'near-capacity and supra-capacity loads in a simultaneous-array dataset. Because all items ' ...
    'were presented simultaneously, significant SS3-vs-SS6 neural distance cannot be attributed ' ...
    'to serial position, elapsed time, trial-internal fatigue, or dynamic set-size expectation. ' ...
    'However, this result should be described as post-capacity differentiation or supra-capacity ' ...
    'separability, not as a post-capacity ramp.\n\n']);

if pD36 < cfg.alpha && meanD36 > 0
    if isfinite(dzD36) && abs(dzD36) < 0.30
        fprintf(fid, ['CDA neural patterns distinguished SS3 and SS6, but the standardized effect ' ...
            'was small (dz=%.3f). Interpret this as weak but statistically reliable post-capacity ' ...
            'differentiation, not strong evidence for continued item-by-item storage beyond capacity.\n'], dzD36);
    else
        fprintf(fid, ['CDA neural patterns distinguished SS3 and SS6 despite both loads being at or ' ...
            'beyond the classical CDA capacity plateau. This supports multivariate CDA load ' ...
            'sensitivity beyond the univariate capacity plateau (dz=%.3f).\n'], dzD36);
    end
else
    fprintf(fid, ['The simultaneous data do not support post-capacity differentiation in the primary ' ...
        'neural RDM (D36 one-sided p=%.6g). Any decoding-only SS3-vs-SS6 effect should be interpreted cautiously.\n'], pD36);
end

fprintf(fid, '\nModel-fit limitation\n');
fprintf(fid, ['Because the simultaneous dataset contains only three load conditions and therefore ' ...
    'only three pairwise distances, model correlations and model-fit comparisons are descriptive ' ...
    'and should be interpreted together with the planned pairwise neural-distance contrasts.\n']);
fprintf(fid, ['The capacityPlateau and lowVsHigher vectors are proportional for these three ' ...
    'conditions, so rank and z-scored-SSE fits cannot distinguish them; their delta is a sanity check only.\n']);

fprintf(fid, '\nSupplementary analyses\n');
if decodingAvailable && ~isempty(DecodingRDMSubject)
    fprintf(fid, ['A supplementary decoding RDM was computed from existing pairwise CDA LDA AUC ' ...
        'results as maintenance-mean AUC minus 0.5 (N=%d complete subjects).\n'], height(DecodingRDMSubject));
    if pD36 < cfg.alpha && meanD36 > 0 && decodingP_D36 < cfg.alpha
        fprintf(fid, ['The neural RDM and decoding-based RDM converged in showing SS3-vs-SS6 ' ...
            'separability, strengthening the evidence for post-capacity differentiation.\n']);
    elseif xor(pD36 < cfg.alpha && meanD36 > 0, decodingP_D36 < cfg.alpha)
        fprintf(fid, ['Neural RDM and decoding-based RDM did not fully converge; interpretation should ' ...
            'prioritize the neural RDM because it is the primary RSA metric.\n']);
    else
        fprintf(fid, ['Neither the primary neural RDM nor the supplementary decoding RDM provided ' ...
            'significant positive D36 evidence.\n']);
    end
else
    fprintf(fid, ['Supplementary decoding RDM was not computed because complete existing pairwise ' ...
        'SS1-vs-SS3, SS1-vs-SS6, and SS3-vs-SS6 CDA decoding inputs were not available.\n']);
end
fprintf(fid, ['Spatial extent control was not performed because the saved EEG RSA input contains no ' ...
    'trial-aligned stimulus positions. Position variables exist in separate behavior files, but ' ...
    'their trial rows do not map one-to-one onto the saved EEG condition rows without reconstructing preprocessing.\n']);
fprintf(fid, ['Time-resolved RSA was not added to this primary module; the requested retention-window ' ...
    'neural RDM is computed from 50-ms-binned trial features across the full retention interval.\n']);
fclose(fid);

save(fullfile(outputDir, 'data1_simultaneous_neural_rsa.mat'), ...
    'cfg', 'condNames', 'loadVals', 'modelNames', 'modelVectors', ...
    'theoryCorrelation', 'includedSubject', 'subjectDistance', ...
    'subjectDistanceSide', 'NeuralRDMSubject', 'NeuralRDMGroupStats', ...
    'PlannedContrastSubject', 'PlannedContrastGroupStats', ...
    'modelFitSpearman', 'modelFitNegSSE', 'deltaSpearman', 'deltaNegSSE', ...
    'RSAModelFitSubject', 'RSAModelFitComparisonStats', ...
    'DecodingRDMSubject', 'DecodingRDMGroupStats', 'Inclusion', '-v7.3');

fprintf('\nData1 simultaneous neural RSA complete.\n%s\n', outputDir);

%% Reused local functions
function stat = sign_flip_test(x, nPerm, alternative, randomSeed)
% Subject-level one-sample sign-flip test, reused for all group contrasts.
x = x(isfinite(x));
x = x(:);
n = numel(x);
if n == 0
    stat = struct('mean',NaN,'sem',NaN,'ci95',[NaN NaN],'p',NaN,'n',0);
    return;
end

observed = mean(x);
sem = std(x, 0) / sqrt(n);
if n > 1
    criticalT = tinv(0.975, n - 1);
    ci95 = observed + [-1 1] * criticalT * sem;
else
    ci95 = [NaN NaN];
end

rng(randomSeed, 'twister');
signMatrix = 2 * (rand(nPerm, n) >= 0.5) - 1;
permutedMean = mean(signMatrix .* x', 2);
switch lower(alternative)
    case 'greater'
        p = (sum(permutedMean >= observed) + 1) / (nPerm + 1);
    case 'less'
        p = (sum(permutedMean <= observed) + 1) / (nPerm + 1);
    case 'two-sided'
        p = (sum(abs(permutedMean) >= abs(observed)) + 1) / (nPerm + 1);
    otherwise
        error('Unknown sign-flip alternative: %s', alternative);
end
stat = struct('mean',observed,'sem',sem,'ci95',ci95,'p',p,'n',n);
end

function draw_subject_bar(ax, data, labels, plotTitle, yLabelText, randomSeed)
% Reused plotting routine: group mean, SEM, and individual subject values.
groupMean = mean(data, 1, 'omitnan');
nValid = sum(isfinite(data), 1);
groupSEM = std(data, 0, 1, 'omitnan') ./ sqrt(nValid);
bar(ax, 1:size(data,2), groupMean, 0.68, 'FaceColor', [0.55 0.68 0.82], ...
    'EdgeColor', [0.2 0.2 0.2]);
hold(ax, 'on');
errorbar(ax, 1:size(data,2), groupMean, groupSEM, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1.4, 'CapSize', 8);
rng(randomSeed, 'twister');
for col = 1:size(data,2)
    valid = isfinite(data(:,col));
    jitter = (rand(sum(valid),1) - 0.5) * 0.22;
    scatter(ax, col + jitter, data(valid,col), 22, [0.15 0.15 0.15], ...
        'filled', 'MarkerFaceAlpha', 0.55);
end
yline(ax, 0, ':k');
set(ax, 'XTick', 1:numel(labels), 'XTickLabel', labels, 'FontSize', 10);
xtickangle(ax, 22);
ylabel(ax, yLabelText);
title(ax, plotTitle);
box(ax, 'off');
hold(ax, 'off');
end

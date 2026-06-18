%% Trial-level behavior correlation for data3 sequential LDA decision values
% Requires data3_setsize1_vs6_LDA_decoding.m to be rerun with
% cfg.returnDecisionValues = true.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
outputDir = fullfile(dataDir, 'decision_behavior_correlation');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

addpath(codeDir);

cfg = struct();
cfg.analysisMode = 'maintOnly';
cfg.modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
cfg.comparisons = { ...
    sprintf('setsize1_vs6_%s', cfg.analysisMode), ...
    fullfile(dataDir, sprintf('decoding_LDA_setsize1_vs6_segments_%s', cfg.analysisMode)); ...
    sprintf('setsize3_vs6_%s', cfg.analysisMode), ...
    fullfile(dataDir, sprintf('decoding_LDA_setsize3_vs6_segments_%s', cfg.analysisMode))};
cfg.evidenceAggregate = 'diagMean';
cfg.evidenceTimeWindowMs = [];  % empty = all binned samples in each segment
cfg.lowSegmentMode = 'last';    % use the set-size-3 segment for setsize3-vs6; do not average low segments

evidenceTables = {};

%% Build one row per subject x model x comparison x probed set-size-6 trial
for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    comparisonDir = cfg.comparisons{ci, 2};

    for mi = 1:numel(cfg.modelNames)
        modelName = cfg.modelNames{mi};
        modelDir = fullfile(comparisonDir, modelName);
        files = dir(fullfile(modelDir, 'sub*.mat'));

        if isempty(files)
            warning('No %s files found in %s.', modelName, modelDir);
            continue;
        end

        for sf = 1:numel(files)
            inFile = fullfile(files(sf).folder, files(sf).name);
            S = load(inFile, modelName);
            if ~isfield(S, modelName)
                warning('%s does not contain %s.', inFile, modelName);
                continue;
            end
            Decode = S.(modelName);

            sideNames = {'Left', 'Right'};
            for si = 1:numel(sideNames)
                sideName = sideNames{si};
                if ~isfield(Decode.side, sideName)
                    continue;
                end

                sideDecode = Decode.side.(sideName);
                if ~isfield(sideDecode, 'decisionByPair') || isempty(sideDecode.decisionByPair)
                    warning('%s is missing decisionByPair. Rerun data3_setsize1_vs6_LDA_decoding.m.', inFile);
                    continue;
                end

                nLowSeg = size(sideDecode.decisionByPair, 1);
                nHighSeg = size(sideDecode.decisionByPair, 2);

                for hi = 1:nHighSeg
                    evidenceByLow = [];
                    testCountByLow = [];
                    highBehavior = table();

                    for li = 1:nLowSeg
                        decisionInfo = sideDecode.decisionByPair{li, hi};
                        if isempty(decisionInfo)
                            continue;
                        end

                        score = decisionInfo.scoreHighState;
                        nTest = decisionInfo.nTest;
                        nDiag = min(size(score, 2), size(score, 3));
                        diagScore = nan(size(score, 1), nDiag);
                        diagCount = zeros(size(score, 1), nDiag);

                        for ti = 1:nDiag
                            diagScore(:,ti) = score(:,ti,ti);
                            diagCount(:,ti) = nTest(:,ti,ti);
                        end

                        relTimes = decisionInfo.relativeTimes(:);
                        if numel(relTimes) >= nDiag
                            relTimes = relTimes(1:nDiag);
                        else
                            relTimes = (1:nDiag)';
                        end

                        if isempty(cfg.evidenceTimeWindowMs)
                            timeKeep = true(nDiag, 1);
                        else
                            timeKeep = relTimes >= cfg.evidenceTimeWindowMs(1) & ...
                                relTimes <= cfg.evidenceTimeWindowMs(2);
                        end

                        evidenceByLow(:,li) = mean(diagScore(:,timeKeep), 2, 'omitnan'); %#ok<SAGROW>
                        testCountByLow(:,li) = sum(diagCount(:,timeKeep), 2, 'omitnan'); %#ok<SAGROW>
                        if isempty(highBehavior)
                            highBehavior = decisionInfo.highBehavior;
                        end
                    end

                    if isempty(evidenceByLow) || isempty(highBehavior)
                        continue;
                    end

                    selectedLowSegment = nan;
                    switch lower(cfg.lowSegmentMode)
                        case 'mean'
                            evidence = mean(evidenceByLow, 2, 'omitnan');
                            decoderTestCount = sum(testCountByLow, 2, 'omitnan');
                            nLowSegmentsAveraged = size(evidenceByLow, 2);
                        case 'first'
                            selectedLowSegment = 1;
                            evidence = evidenceByLow(:,selectedLowSegment);
                            decoderTestCount = testCountByLow(:,selectedLowSegment);
                            nLowSegmentsAveraged = 1;
                        case 'last'
                            selectedLowSegment = size(evidenceByLow, 2);
                            evidence = evidenceByLow(:,selectedLowSegment);
                            decoderTestCount = testCountByLow(:,selectedLowSegment);
                            nLowSegmentsAveraged = 1;
                        otherwise
                            error('Unsupported cfg.lowSegmentMode: %s.', cfg.lowSegmentMode);
                    end

                    probedTrial = highBehavior.SerialPosition == hi & ...
                        highBehavior.SetSize == 6 & ...
                        ~isnan(evidence) & decoderTestCount > 0;

                    if ~any(probedTrial)
                        continue;
                    end

                    nRow = sum(probedTrial);
                    T = table();
                    T.Subject = highBehavior.Subject(probedTrial);
                    T.Comparison = repmat({comparisonName}, nRow, 1);
                    T.Model = repmat({modelName}, nRow, 1);
                    T.Side = repmat({sideName}, nRow, 1);
                    T.TrialIndex = highBehavior.TrialIndex(probedTrial);
                    T.Block = highBehavior.Block(probedTrial);
                    T.BlockTrial = highBehavior.BlockTrial(probedTrial);
                    T.MemoryType = highBehavior.MemoryType(probedTrial);
                    T.Change = highBehavior.Change(probedTrial);
                    T.SerialPosition = highBehavior.SerialPosition(probedTrial);
                    T.Response = highBehavior.Response(probedTrial);
                    T.RT = highBehavior.RT(probedTrial);
                    T.Correct = highBehavior.Correct(probedTrial);
                    T.Evidence = evidence(probedTrial);
                    T.DecoderTestCount = decoderTestCount(probedTrial);
                    T.LowSegmentUsed = repmat(selectedLowSegment, nRow, 1);
                    T.NLowSegmentsAveraged = repmat(nLowSegmentsAveraged, nRow, 1);

                    evidenceTables{end+1,1} = T; %#ok<SAGROW>
                end
            end
        end
    end
end

if isempty(evidenceTables)
    error('No decision evidence rows were built. Rerun data3_setsize1_vs6_LDA_decoding.m first.');
end

TrialEvidence = vertcat(evidenceTables{:});
TrialEvidence.EvidenceZ = nan(height(TrialEvidence), 1);

for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    for mi = 1:numel(cfg.modelNames)
        modelName = cfg.modelNames{mi};
        idx = strcmp(TrialEvidence.Comparison, comparisonName) & strcmp(TrialEvidence.Model, modelName);
        if ~any(idx)
            continue;
        end

        mu = mean(TrialEvidence.Evidence(idx), 'omitnan');
        sd = std(TrialEvidence.Evidence(idx), 0, 'omitnan');
        if sd == 0 || isnan(sd)
            sd = 1;
        end
        TrialEvidence.EvidenceZ(idx) = (TrialEvidence.Evidence(idx) - mu) ./ sd;
    end
end

writetable(TrialEvidence, fullfile(outputDir, 'data3_decision_behavior_trial_evidence.csv'));

%% Logistic mixed-effects models
ModelResults = struct();
summaryTables = {};

for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    for mi = 1:numel(cfg.modelNames)
        modelName = cfg.modelNames{mi};
        useRows = strcmp(TrialEvidence.Comparison, comparisonName) & ...
            strcmp(TrialEvidence.Model, modelName) & ...
            ~isnan(TrialEvidence.Evidence) & ...
            ~isnan(TrialEvidence.Correct) & ...
            ~isnan(TrialEvidence.SerialPosition) & ...
            ~isnan(TrialEvidence.TrialIndex);

        modelTable = TrialEvidence(useRows, :);
        if height(modelTable) == 0 || numel(unique(modelTable.Subject)) < 2
            warning('Skipping %s %s: not enough rows or subjects.', comparisonName, modelName);
            continue;
        end

        modelTable.Subject = categorical(modelTable.Subject);
        modelTable.Correct = double(modelTable.Correct);
        modelTable.SerialPosition = double(modelTable.SerialPosition);
        modelTable.TrialIndex = double(modelTable.TrialIndex);

        formulas = { ...
            'base', 'Correct ~ Evidence + SerialPosition + (1|Subject)'; ...
            'withTrialIndex', 'Correct ~ Evidence + SerialPosition + TrialIndex + (1|Subject)'};

        for fi = 1:size(formulas, 1)
            formulaName = formulas{fi, 1};
            formula = formulas{fi, 2};

            try
                glme = fitglme(modelTable, formula, ...
                    'Distribution', 'Binomial', ...
                    'Link', 'logit', ...
                    'FitMethod', 'Laplace');
            catch ME
                warning('fitglme failed for %s %s %s: %s', ...
                    comparisonName, modelName, formulaName, ME.message);
                continue;
            end

            ModelResults.(comparisonName).(modelName).(formulaName) = glme;

            coef = glme.Coefficients;
            nCoef = height(coef);
            S = table();
            S.Comparison = repmat({comparisonName}, nCoef, 1);
            S.Model = repmat({modelName}, nCoef, 1);
            S.Formula = repmat({formulaName}, nCoef, 1);
            S.NTrial = repmat(height(modelTable), nCoef, 1);
            S.NSubject = repmat(numel(unique(modelTable.Subject)), nCoef, 1);
            S.Term = coef.Name;
            S.Estimate = coef.Estimate;
            S.SE = coef.SE;
            S.tStat = coef.tStat;
            S.pValue = coef.pValue;
            summaryTables{end+1,1} = S; %#ok<SAGROW>
        end
    end
end

if isempty(summaryTables)
    ModelSummary = table();
else
    ModelSummary = vertcat(summaryTables{:});
end

writetable(ModelSummary, fullfile(outputDir, 'data3_decision_behavior_glme_summary.csv'));
save(fullfile(outputDir, 'data3_decision_behavior_correlation.mat'), ...
    'cfg', 'TrialEvidence', 'ModelSummary', 'ModelResults', '-v7.3');

fprintf('Saved trial evidence and GLME summaries to:\n%s\n', outputDir);

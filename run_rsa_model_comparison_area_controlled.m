%% RSA theoretical model comparison after pairwise cumulative-area matching
% Reads existing subject-level empirical RDMs from data3_segment_state_RSA.m.
% This script does not recompute trial-level CDA features or crossnobis RDMs.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
rsaDir = fullfile(projectRoot, 'data3', ...
    'RSA_segment_state_maintOnly_areaMatched', 'CDA');
outputDir = fullfile(rsaDir, 'rsa_model_comparison');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

cfg = struct();
cfg.analysisName = 'RSA theoretical model comparison after pairwise cumulative-area matching';
cfg.rsaDir = rsaDir;
cfg.outputDir = outputDir;
cfg.nPerm = 100000;
cfg.randomSeed = 20260619;
cfg.alpha = 0.05;
cfg.highCorrelationThreshold = 0.85;
cfg.figureVisible = 'off';
cfg.figureDpi = 200;
cfg.familyFitMetric = 'adjusted R2 from tied-rank OLS';

condNames = {'SS1_P1','SS3_P1','SS3_P2','SS3_P3', ...
    'SS6_P1','SS6_P2','SS6_P3','SS6_P4','SS6_P5','SS6_P6'};
setSize = [1 3 3 3 6 6 6 6 6 6];
pos = [1 1 2 3 1 2 3 4 5 6];
ssValues = [1 3 6];
nCond = numel(condNames);
upperMask = triu(true(nCond), 1);

subjectFiles = dir(fullfile(rsaDir, 'sub*_RSA.mat'));
if isempty(subjectFiles)
    error('No subject RSA files found in %s.', rsaDir);
end

subjectFromName = nan(numel(subjectFiles), 1);
for fileIdx = 1:numel(subjectFiles)
    tok = regexp(subjectFiles(fileIdx).name, '^sub(\d+)_RSA\.mat$', 'tokens', 'once');
    if isempty(tok)
        error('Unexpected subject RSA filename: %s.', subjectFiles(fileIdx).name);
    end
    subjectFromName(fileIdx) = str2double(tok{1});
end
[subjectFromName, fileOrder] = sort(subjectFromName);
subjectFiles = subjectFiles(fileOrder);

nSubjects = numel(subjectFiles);
subjects = nan(nSubjects, 1);
empiricalRDMs = nan(nCond, nCond, nSubjects);
areaValuesBySubject = nan(nCond, nSubjects);
setSizeTrialCountsBySubject = nan(numel(ssValues), nSubjects);
hasArea = false(nSubjects, 1);
hasSetSizeCounts = false(nSubjects, 1);
sourceRDMVariable = cell(nSubjects, 1);

rdmCandidates = {'empiricalRDM_segmentAverage','subjRDM','empRDM','rdm_all'};

%% Read existing subject-level empirical RDMs and saved design summaries
for subjectIdx = 1:nSubjects
    inFile = fullfile(subjectFiles(subjectIdx).folder, subjectFiles(subjectIdx).name);
    saved = load(inFile);

    rdmField = '';
    for candidateIdx = 1:numel(rdmCandidates)
        if isfield(saved, rdmCandidates{candidateIdx})
            rdmField = rdmCandidates{candidateIdx};
            break;
        end
    end
    if isempty(rdmField)
        error(['No subject-level empirical RDM found in %s. Expected one of: %s. ' ...
            'The required variable must contain one 10 x 10 subject empirical RDM.'], ...
            inFile, strjoin(rdmCandidates, ', '));
    end

    empiricalRDM = squeeze(saved.(rdmField));
    if ~isequal(size(empiricalRDM), [nCond nCond])
        error('%s.%s has size %s; expected 10 x 10.', ...
            subjectFiles(subjectIdx).name, rdmField, mat2str(size(empiricalRDM)));
    end
    finiteDifference = abs(empiricalRDM - empiricalRDM');
    if any(finiteDifference(isfinite(finiteDifference)) > 1e-10)
        error('%s.%s is not symmetric.', subjectFiles(subjectIdx).name, rdmField);
    end

    if isfield(saved, 'subject') && ~isempty(saved.subject)
        subjects(subjectIdx) = saved.subject;
    else
        subjects(subjectIdx) = subjectFromName(subjectIdx);
    end
    sourceRDMVariable{subjectIdx} = rdmField;
    empiricalRDMs(:, :, subjectIdx) = empiricalRDM;

    conditionTableNow = table();
    if isfield(saved, 'conditionTableSubject') && istable(saved.conditionTableSubject)
        conditionTableNow = saved.conditionTableSubject;
    elseif isfield(saved, 'conditionTable') && istable(saved.conditionTable)
        conditionTableNow = saved.conditionTable;
    end
    if ~isempty(conditionTableNow) && ismember('conditionName', conditionTableNow.Properties.VariableNames)
        savedNames = conditionTableNow.conditionName(:)';
        if ~isequal(savedNames, condNames)
            error('Condition order mismatch in %s.', subjectFiles(subjectIdx).name);
        end
    end

    if ~isempty(conditionTableNow) && ...
            ismember('meanCumulativeArea', conditionTableNow.Properties.VariableNames)
        areaValuesBySubject(:, subjectIdx) = conditionTableNow.meanCumulativeArea(:);
        hasArea(subjectIdx) = all(isfinite(areaValuesBySubject(:, subjectIdx)));
    elseif isfield(saved, 'areaInfo') && istable(saved.areaInfo) && ...
            all(ismember({'Condition','MeanCumulativeArea'}, saved.areaInfo.Properties.VariableNames))
        for conditionIdx = 1:nCond
            areaRows = strcmp(saved.areaInfo.Condition, condNames{conditionIdx});
            areaValuesBySubject(conditionIdx, subjectIdx) = ...
                mean(saved.areaInfo.MeanCumulativeArea(areaRows), 'omitnan');
        end
        hasArea(subjectIdx) = all(isfinite(areaValuesBySubject(:, subjectIdx)));
    end

    if isfield(saved, 'trialCountCheck') && istable(saved.trialCountCheck) && ...
            all(ismember({'SetSize','BehaviorTrials'}, saved.trialCountCheck.Properties.VariableNames))
        for ssIdx = 1:numel(ssValues)
            rows = saved.trialCountCheck.SetSize == ssValues(ssIdx);
            setSizeTrialCountsBySubject(ssIdx, subjectIdx) = ...
                sum(saved.trialCountCheck.BehaviorTrials(rows), 'omitnan');
        end
        hasSetSizeCounts(subjectIdx) = all(isfinite(setSizeTrialCountsBySubject(:, subjectIdx))) && ...
            all(setSizeTrialCountsBySubject(:, subjectIdx) > 0);
    end
end

if numel(unique(subjects)) ~= nSubjects
    error('Duplicate subject identifiers were found in the RSA output folder.');
end

%% Derive the set-size prior from saved behavior counts
if all(hasSetSizeCounts)
    setSizePriorCounts = sum(setSizeTrialCountsBySubject, 2);
    setSizePrior = setSizePriorCounts ./ sum(setSizePriorCounts);
    priorSource = 'summed BehaviorTrials from all subject trialCountCheck tables';
else
    setSizePriorCounts = nan(numel(ssValues), 1);
    setSizePrior = ones(numel(ssValues), 1) ./ numel(ssValues);
    priorSource = 'equal-probability fallback because complete saved trial counts were unavailable';
end

fprintf('Set-size prior source: %s\n', priorSource);
fprintf('P(SS=1,3,6) = [%s]\n', strjoin(compose('%.6f', setSizePrior), ', '));

%% Construct memory-state, subjective-expectation, and objective models
capacity = min(pos, 3);
postRamp = max(pos - 3, 0);
postBinary = double(pos >= 4);
actualSetSize = setSize;
absolutePosition = pos;

uncertaintyByPosition = nan(1, max(pos));
expectedSSByPosition = nan(1, max(pos));
expectedRemainingByPosition = nan(1, max(pos));
hazardByPosition = nan(1, max(pos));

for positionNow = 1:max(pos)
    possible = ssValues >= positionNow;
    conditionalPrior = setSizePrior(possible);
    conditionalPrior = conditionalPrior ./ sum(conditionalPrior);
    possibleSS = ssValues(possible);

    uncertaintyByPosition(positionNow) = ...
        -sum(conditionalPrior .* log2(conditionalPrior), 'omitnan');
    expectedSSByPosition(positionNow) = sum(possibleSS(:) .* conditionalPrior(:));
    expectedRemainingByPosition(positionNow) = ...
        sum((possibleSS(:) - positionNow) .* conditionalPrior(:));
    hazardByPosition(positionNow) = sum(conditionalPrior(possibleSS == positionNow));
end

uncertainty = uncertaintyByPosition(pos);
expectedSS = expectedSSByPosition(pos);
expectedRemaining = expectedRemainingByPosition(pos);
hazard = hazardByPosition(pos);

if ~(uncertainty(1) == uncertainty(2) && uncertainty(2) == uncertainty(5) && ...
        expectedSS(1) == expectedSS(2) && expectedSS(2) == expectedSS(5) && ...
        expectedRemaining(3) == expectedRemaining(6) && ...
        hazard(4) == hazard(7))
    error('Subjective expectation models depend on actual set-size condition rather than position only.');
end

areaAvailable = all(hasArea);
if areaAvailable
    groupMeanArea = mean(areaValuesBySubject, 2, 'omitnan')';
    areaStatus = sprintf('area model included from areaInfo.MeanCumulativeArea in %d subject files', nSubjects);
else
    groupMeanArea = [];
    areaStatus = 'area model skipped: mean cumulative area per condition not found';
    fprintf('%s\n', areaStatus);
end

modelValuesGroup = struct();
modelValuesGroup.capacity = capacity;
modelValuesGroup.postRamp = postRamp;
modelValuesGroup.postBinary = postBinary;
modelValuesGroup.uncertainty = uncertainty;
modelValuesGroup.expectedSS = expectedSS;
modelValuesGroup.expectedRemaining = expectedRemaining;
modelValuesGroup.hazard = hazard;
modelValuesGroup.actualSetSize = actualSetSize;
modelValuesGroup.absolutePosition = absolutePosition;

theoryModelNames = {'capacity','postRamp','postBinary','uncertainty', ...
    'expectedSS','expectedRemaining','hazard','actualSetSize','absolutePosition'};
theoryModelCategory = {'memory','memory','memory','subjectiveExpectation', ...
    'subjectiveExpectation','subjectiveExpectation','subjectiveExpectation', ...
    'objective','objective'};
if areaAvailable
    modelValuesGroup.area = groupMeanArea;
    theoryModelNames{end+1} = 'area';
    theoryModelCategory{end+1} = 'spatialControl';
end

theoryRDMsGroup = struct();
theoryVectorsGroup = nan(nnz(upperMask), numel(theoryModelNames));
for modelIdx = 1:numel(theoryModelNames)
    modelName = theoryModelNames{modelIdx};
    valuesNow = modelValuesGroup.(modelName)(:);
    rdmNow = abs(valuesNow - valuesNow');
    theoryRDMsGroup.(modelName) = rdmNow;
    theoryVectorsGroup(:, modelIdx) = rdmNow(upperMask);
end

conditionTableModel = table(condNames(:), setSize(:), pos(:), capacity(:), ...
    postRamp(:), postBinary(:), uncertainty(:), expectedSS(:), ...
    expectedRemaining(:), hazard(:), actualSetSize(:), absolutePosition(:), ...
    'VariableNames', {'Condition','SetSize','Position','Capacity','PostRamp', ...
    'PostBinary','Uncertainty','ExpectedSS','ExpectedRemaining','Hazard', ...
    'ActualSetSize','AbsolutePosition'});
if areaAvailable
    conditionTableModel.GroupMeanArea = groupMeanArea(:);
end
writetable(conditionTableModel, fullfile(outputDir, 'theory_model_condition_values.csv'));

%% Theoretical RDM Spearman correlation matrix and high-correlation audit
rankedTheoryVectors = nan(size(theoryVectorsGroup));
for modelIdx = 1:numel(theoryModelNames)
    rankedTheoryVectors(:, modelIdx) = tiedrank(theoryVectorsGroup(:, modelIdx));
end
theoryModelCorrelation = corrcoef(rankedTheoryVectors, 'Rows', 'pairwise');

correlationTable = array2table(theoryModelCorrelation, ...
    'VariableNames', theoryModelNames, 'RowNames', theoryModelNames);
writetable(correlationTable, fullfile(outputDir, 'theory_model_correlation.csv'), ...
    'WriteRowNames', true);

fig = figure('Color', 'w', 'Visible', cfg.figureVisible, 'Position', [100 100 980 820]);
ax = axes(fig);
imagesc(ax, theoryModelCorrelation, [-1 1]);
axis(ax, 'square');
colorbar(ax);
colormap(ax, parula);
set(ax, 'XTick', 1:numel(theoryModelNames), 'XTickLabel', theoryModelNames, ...
    'YTick', 1:numel(theoryModelNames), 'YTickLabel', theoryModelNames, ...
    'XTickLabelRotation', 45, 'TickDir', 'out');
title(ax, 'Spearman correlation among theoretical RDMs');
print(fig, fullfile(outputDir, 'theory_model_correlation.png'), ...
    '-dpng', sprintf('-r%d', cfg.figureDpi));
close(fig);

highCorrelationRows = {};
for modelI = 1:numel(theoryModelNames)
    for modelJ = (modelI + 1):numel(theoryModelNames)
        rNow = theoryModelCorrelation(modelI, modelJ);
        if isfinite(rNow) && abs(rNow) > cfg.highCorrelationThreshold
            row = table();
            row.Model1 = theoryModelNames(modelI);
            row.Model2 = theoryModelNames(modelJ);
            row.SpearmanR = rNow;
            highCorrelationRows{end+1, 1} = row; %#ok<SAGROW>
        end
    end
end

warningFile = fullfile(outputDir, 'high_model_correlation_warning.txt');
fid = fopen(warningFile, 'w');
if isempty(highCorrelationRows)
    highCorrelationTable = table();
    fprintf(fid, 'No theoretical model pairs exceeded abs(Spearman r) > %.2f.\n', ...
        cfg.highCorrelationThreshold);
else
    highCorrelationTable = vertcat(highCorrelationRows{:});
    warning('%d theoretical model pairs exceeded abs(Spearman r) > %.2f.', ...
        height(highCorrelationTable), cfg.highCorrelationThreshold);
    fprintf(fid, 'High theoretical model correlation warning: abs(Spearman r) > %.2f\n\n', ...
        cfg.highCorrelationThreshold);
    for rowIdx = 1:height(highCorrelationTable)
        fprintf(fid, '%s vs %s: r = %.6f\n', ...
            highCorrelationTable.Model1{rowIdx}, highCorrelationTable.Model2{rowIdx}, ...
            highCorrelationTable.SpearmanR(rowIdx));
        fprintf('  high correlation: %s vs %s, r=%.4f\n', ...
            highCorrelationTable.Model1{rowIdx}, highCorrelationTable.Model2{rowIdx}, ...
            highCorrelationTable.SpearmanR(rowIdx));
    end
end
fclose(fid);

expectationPredictors = {'uncertainty','expectedSS','expectedRemaining','hazard'};
expectationIdx = find(ismember(theoryModelNames, expectationPredictors));
expectationCorr = theoryModelCorrelation(expectationIdx, expectationIdx);
expectationUpper = abs(expectationCorr( ...
    triu(true(numel(expectationIdx), numel(expectationIdx)), 1)));
expectationNotes = {};
if any(expectationUpper > cfg.highCorrelationThreshold)
    expectationPredictors = {'uncertainty','expectedSS','hazard'};
    expectationNotes{end+1} = ['The four-predictor expectation RDM set exceeded the high-correlation ' ...
        'threshold; expectedRemaining was removed.'];
end

uncertaintyIdx = find(strcmp(theoryModelNames, 'uncertainty'), 1);
expectedSSIdx = find(strcmp(theoryModelNames, 'expectedSS'), 1);
if all(ismember({'uncertainty','expectedSS'}, expectationPredictors)) && ...
        abs(theoryModelCorrelation(uncertaintyIdx, expectedSSIdx)) > cfg.highCorrelationThreshold
    expectationPredictors(strcmp(expectationPredictors, 'expectedSS')) = [];
    expectationNotes{end+1} = sprintf([ ...
        'uncertainty and expectedSS RDMs remained non-separable (Spearman r=%.6f); ' ...
        'expectedSS was removed from multivariable expectation families.'], ...
        theoryModelCorrelation(uncertaintyIdx, expectedSSIdx));
end

if isempty(expectationNotes)
    expectationModelNote = 'All four subjective expectation predictors were retained.';
else
    expectationModelNote = strjoin(expectationNotes, ' ');
end
fprintf('%s\n', expectationModelNote);
fprintf('Expectation predictors used: %s\n', strjoin(expectationPredictors, ', '));

%% Define model families
familyDefs = struct();
familyDefs(1).name = 'memoryOnly';
familyDefs(1).predictors = {'capacity','postRamp'};
familyDefs(2).name = 'memoryDiagnostic';
familyDefs(2).predictors = {'capacity','postBinary','postRamp'};
familyDefs(3).name = 'subjectiveExpectationOnly';
familyDefs(3).predictors = expectationPredictors;

confoundPredictors = [expectationPredictors, {'actualSetSize','absolutePosition'}];
if areaAvailable
    confoundPredictors{end+1} = 'area';
end
familyDefs(4).name = 'confoundOnly';
familyDefs(4).predictors = confoundPredictors;
familyDefs(5).name = 'fullModel';
familyDefs(5).predictors = [confoundPredictors, {'capacity','postRamp'}];
familyDefs(6).name = 'absolutePositionOnly';
familyDefs(6).predictors = {'absolutePosition'};
familyDefs(7).name = 'capacityOnly';
familyDefs(7).predictors = {'capacity'};
familyDefs(8).name = 'capacityPlusPostRamp';
familyDefs(8).predictors = {'capacity','postRamp'};

nFamilies = numel(familyDefs);
familyFitMatrix = nan(nSubjects, nFamilies);
betaRows = {};
familyFitRows = {};

baseTheoryVectors = struct();
for modelIdx = 1:numel(theoryModelNames)
    if ~strcmp(theoryModelNames{modelIdx}, 'area')
        baseTheoryVectors.(theoryModelNames{modelIdx}) = theoryVectorsGroup(:, modelIdx);
    end
end

%% Subject-level tied-rank OLS and semipartial RSA
for subjectIdx = 1:nSubjects
    empiricalVector = empiricalRDMs(:, :, subjectIdx);
    empiricalVector = empiricalVector(upperMask);

    subjectTheoryVectors = baseTheoryVectors;
    if areaAvailable
        areaNow = areaValuesBySubject(:, subjectIdx);
        areaRDMNow = abs(areaNow - areaNow');
        subjectTheoryVectors.area = areaRDMNow(upperMask);
    end

    for familyIdx = 1:nFamilies
        familyName = familyDefs(familyIdx).name;
        predictors = familyDefs(familyIdx).predictors;
        theoryMatrix = nan(numel(empiricalVector), numel(predictors));
        for predictorIdx = 1:numel(predictors)
            theoryMatrix(:, predictorIdx) = subjectTheoryVectors.(predictors{predictorIdx});
        end

        fitResult = fit_rank_rsa(empiricalVector, theoryMatrix);
        familyFitMatrix(subjectIdx, familyIdx) = fitResult.adjustedR2;

        fitRow = table();
        fitRow.Subject = subjects(subjectIdx);
        fitRow.Model = {familyName};
        fitRow.PredictorsUsed = {strjoin(predictors, ' + ')};
        fitRow.FitMetric = {cfg.familyFitMetric};
        fitRow.R2 = fitResult.R2;
        fitRow.AdjustedR2 = fitResult.adjustedR2;
        fitRow.EffectivePredictorRank = fitResult.effectivePredictorRank;
        fitRow.NPairs = fitResult.nPairs;
        familyFitRows{end+1, 1} = fitRow; %#ok<SAGROW>

        for predictorIdx = 1:numel(predictors)
            betaRow = table();
            betaRow.Subject = subjects(subjectIdx);
            betaRow.Model = {familyName};
            betaRow.Predictor = predictors(predictorIdx);
            betaRow.PredictorsUsed = {strjoin(predictors, ' + ')};
            betaRow.SemipartialR = fitResult.semipartialR(predictorIdx);
            betaRow.StandardizedBeta = fitResult.standardizedBeta(predictorIdx);
            betaRow.R2 = fitResult.R2;
            betaRow.AdjustedR2 = fitResult.adjustedR2;
            betaRow.EffectivePredictorRank = fitResult.effectivePredictorRank;
            betaRow.NPairs = fitResult.nPairs;
            betaRows{end+1, 1} = betaRow; %#ok<SAGROW>
        end
    end
end

betaSubject = vertcat(betaRows{:});
familyFitSubject = vertcat(familyFitRows{:});
writetable(betaSubject, fullfile(outputDir, 'rsa_model_betas_subject.csv'));
writetable(familyFitSubject, fullfile(outputDir, 'rsa_model_family_fit_subject.csv'));

%% Group-level sign-flip statistics for subject semipartial R
rng(cfg.randomSeed, 'twister');
betaKeys = unique(betaSubject(:, {'Model','Predictor'}), 'rows', 'stable');
betaGroupRows = {};
for keyIdx = 1:height(betaKeys)
    rows = strcmp(betaSubject.Model, betaKeys.Model{keyIdx}) & ...
        strcmp(betaSubject.Predictor, betaKeys.Predictor{keyIdx});
    values = betaSubject.SemipartialR(rows);
    statsNow = sign_flip_test(values, cfg.nPerm);

    row = table();
    row.Model = betaKeys.Model(keyIdx);
    row.Predictor = betaKeys.Predictor(keyIdx);
    row.Metric = {'SemipartialR'};
    row.Mean = statsNow.mean;
    row.SEM = statsNow.sem;
    row.CI95Low = statsNow.ciLow;
    row.CI95High = statsNow.ciHigh;
    row.PValueSignFlip = statsNow.p;
    row.N = statsNow.n;
    betaGroupRows{end+1, 1} = row; %#ok<SAGROW>
end
betaGroupStats = vertcat(betaGroupRows{:});
writetable(betaGroupStats, fullfile(outputDir, 'rsa_model_betas_group_stats.csv'));

%% Model-family paired comparisons using adjusted R2
comparisonNames = {'memoryOnly_minus_subjectiveExpectationOnly', ...
    'fullModel_minus_confoundOnly', ...
    'memoryOnly_minus_absolutePositionOnly', ...
    'capacityPlusPostRamp_minus_capacityOnly'};
modelA = {'memoryOnly','fullModel','memoryOnly','capacityPlusPostRamp'};
modelB = {'subjectiveExpectationOnly','confoundOnly','absolutePositionOnly','capacityOnly'};
comparisonDifference = nan(nSubjects, numel(comparisonNames));
comparisonRows = {};

familyNames = {familyDefs.name};
for comparisonIdx = 1:numel(comparisonNames)
    idxA = find(strcmp(familyNames, modelA{comparisonIdx}), 1);
    idxB = find(strcmp(familyNames, modelB{comparisonIdx}), 1);
    differenceNow = familyFitMatrix(:, idxA) - familyFitMatrix(:, idxB);
    comparisonDifference(:, comparisonIdx) = differenceNow;
    statsNow = sign_flip_test(differenceNow, cfg.nPerm);

    if statsNow.mean > 0
        effectDirection = sprintf('%s > %s', modelA{comparisonIdx}, modelB{comparisonIdx});
    elseif statsNow.mean < 0
        effectDirection = sprintf('%s < %s', modelA{comparisonIdx}, modelB{comparisonIdx});
    else
        effectDirection = 'no mean difference';
    end

    row = table();
    row.Comparison = comparisonNames(comparisonIdx);
    row.ModelA = modelA(comparisonIdx);
    row.ModelB = modelB(comparisonIdx);
    row.DifferenceDefinition = {sprintf('%s adjustedR2 minus %s adjustedR2', ...
        modelA{comparisonIdx}, modelB{comparisonIdx})};
    row.MeanDifference = statsNow.mean;
    row.SEM = statsNow.sem;
    row.CI95Low = statsNow.ciLow;
    row.CI95High = statsNow.ciHigh;
    row.PValueSignFlip = statsNow.p;
    row.EffectDirection = {effectDirection};
    row.N = statsNow.n;
    comparisonRows{end+1, 1} = row; %#ok<SAGROW>
end
familyComparisonStats = vertcat(comparisonRows{:});
writetable(familyComparisonStats, ...
    fullfile(outputDir, 'rsa_model_family_comparison_stats.csv'));

%% Group figures with subject overlays
familyPValues = nan(1, nFamilies);
plot_subject_bars(familyFitMatrix, familyNames, familyPValues, ...
    'Model-family fit', 'Adjusted R^2', ...
    fullfile(outputDir, 'model_family_fit_bar.png'), cfg);

fullVsConfoundIdx = find(strcmp(comparisonNames, 'fullModel_minus_confoundOnly'), 1);
plot_subject_bars(comparisonDifference(:, fullVsConfoundIdx), ...
    {'Full - confound'}, familyComparisonStats.PValueSignFlip(fullVsConfoundIdx), ...
    'Incremental memory-state fit beyond confounds', 'Adjusted R^2 difference', ...
    fullfile(outputDir, 'full_vs_confound_delta_bar.png'), cfg);

memoryPlotKeys = { ...
    'memoryOnly','capacity'; ...
    'memoryOnly','postRamp'; ...
    'memoryDiagnostic','capacity'; ...
    'memoryDiagnostic','postBinary'; ...
    'memoryDiagnostic','postRamp'};
memoryPlotValues = nan(nSubjects, size(memoryPlotKeys, 1));
memoryPlotLabels = cell(1, size(memoryPlotKeys, 1));
memoryPlotP = nan(1, size(memoryPlotKeys, 1));
for plotIdx = 1:size(memoryPlotKeys, 1)
    rows = strcmp(betaSubject.Model, memoryPlotKeys{plotIdx, 1}) & ...
        strcmp(betaSubject.Predictor, memoryPlotKeys{plotIdx, 2});
    [~, subjectOrder] = ismember(subjects, betaSubject.Subject(rows));
    valuesNow = betaSubject.SemipartialR(rows);
    memoryPlotValues(:, plotIdx) = valuesNow(subjectOrder);
    memoryPlotLabels{plotIdx} = sprintf('%s:%s', ...
        memoryPlotKeys{plotIdx, 1}, memoryPlotKeys{plotIdx, 2});
    statRow = strcmp(betaGroupStats.Model, memoryPlotKeys{plotIdx, 1}) & ...
        strcmp(betaGroupStats.Predictor, memoryPlotKeys{plotIdx, 2});
    memoryPlotP(plotIdx) = betaGroupStats.PValueSignFlip(statRow);
end
plot_subject_bars(memoryPlotValues, memoryPlotLabels, memoryPlotP, ...
    'Memory-model predictor effects', 'Semipartial r', ...
    fullfile(outputDir, 'predictor_semipartial_bar_memory_models.png'), cfg);

fullPredictors = familyDefs(strcmp(familyNames, 'fullModel')).predictors;
fullPlotValues = nan(nSubjects, numel(fullPredictors));
fullPlotP = nan(1, numel(fullPredictors));
for plotIdx = 1:numel(fullPredictors)
    rows = strcmp(betaSubject.Model, 'fullModel') & ...
        strcmp(betaSubject.Predictor, fullPredictors{plotIdx});
    [~, subjectOrder] = ismember(subjects, betaSubject.Subject(rows));
    valuesNow = betaSubject.SemipartialR(rows);
    fullPlotValues(:, plotIdx) = valuesNow(subjectOrder);
    statRow = strcmp(betaGroupStats.Model, 'fullModel') & ...
        strcmp(betaGroupStats.Predictor, fullPredictors{plotIdx});
    fullPlotP(plotIdx) = betaGroupStats.PValueSignFlip(statRow);
end
fullPlotLabels = cellfun(@(x) sprintf('full:%s', x), fullPredictors, 'UniformOutput', false);
plot_subject_bars(fullPlotValues, fullPlotLabels, fullPlotP, ...
    'Full-model predictor effects', 'Semipartial r', ...
    fullfile(outputDir, 'predictor_semipartial_bar_full_model.png'), cfg);

%% Save complete results and an interpretation template
theoryModelInfo = table(theoryModelNames(:), theoryModelCategory(:), ...
    'VariableNames', {'Model','Category'});
priorInfo = struct('ssValues', ssValues, 'counts', setSizePriorCounts, ...
    'probability', setSizePrior, 'source', priorSource);

save(fullfile(outputDir, 'rsa_model_comparison_results.mat'), ...
    'cfg', 'subjects', 'subjectFiles', 'sourceRDMVariable', 'empiricalRDMs', ...
    'conditionTableModel', 'priorInfo', 'areaStatus', 'areaValuesBySubject', ...
    'theoryModelInfo', 'theoryRDMsGroup', 'theoryVectorsGroup', ...
    'theoryModelCorrelation', 'highCorrelationTable', 'expectationPredictors', ...
    'expectationModelNote', 'familyDefs', 'betaSubject', 'betaGroupStats', ...
    'familyFitSubject', 'familyFitMatrix', 'familyComparisonStats', ...
    'comparisonDifference', '-v7.3');

summaryFile = fullfile(outputDir, 'rsa_model_comparison_summary.txt');
fid = fopen(summaryFile, 'w');
fprintf(fid, 'RSA theoretical model comparison summary\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Subjects: N=%d, IDs=%s\n', nSubjects, mat2str(subjects(:)'));
fprintf(fid, 'Empirical RDM: subject-level area-matched segment-average crossnobis RDM.\n');
fprintf(fid, 'Family fit metric: %s (in-sample, not cross-validated).\n', cfg.familyFitMetric);
fprintf(fid, 'Set-size prior source: %s\n', priorSource);
fprintf(fid, 'P(SS=1,3,6)=[%s]\n', strjoin(compose('%.6f', setSizePrior), ', '));
fprintf(fid, 'Expectation predictors used: %s\n', strjoin(expectationPredictors, ', '));
fprintf(fid, '%s\n', expectationModelNote);
fprintf(fid, '%s\n', areaStatus);
fprintf(fid, 'High theoretical model correlations above |r|>%.2f: %d pairs.\n\n', ...
    cfg.highCorrelationThreshold, height(highCorrelationTable));

capacityFullRow = strcmp(betaGroupStats.Model, 'fullModel') & ...
    strcmp(betaGroupStats.Predictor, 'capacity');
postRampFullRow = strcmp(betaGroupStats.Model, 'fullModel') & ...
    strcmp(betaGroupStats.Predictor, 'postRamp');
fullComparisonRow = strcmp(familyComparisonStats.Comparison, ...
    'fullModel_minus_confoundOnly');
memoryPositionRow = strcmp(familyComparisonStats.Comparison, ...
    'memoryOnly_minus_absolutePositionOnly');

fprintf(fid, 'Interpretation template based on current statistics\n');
fprintf(fid, '---------------------------------------------------\n');
if any(capacityFullRow) && betaGroupStats.PValueSignFlip(capacityFullRow) < cfg.alpha
    fprintf(fid, ['Capacity remains significant in the full model (p=%.6g): the CDA RDM contains ' ...
        'stable capacity-limited load geometry.\n'], betaGroupStats.PValueSignFlip(capacityFullRow));
else
    fprintf(fid, ['Capacity is not significant in the full model: evidence for capacity-limited ' ...
        'geometry after the included controls is inconclusive.\n']);
end

if any(postRampFullRow) && betaGroupStats.PValueSignFlip(postRampFullRow) < cfg.alpha
    fprintf(fid, ['PostRamp remains significant in the full model (p=%.6g): post-capacity dynamics ' ...
        'remain after controlling subjective expectation, actual set-size context, absolute ' ...
        'position/elapsed time, and area when available.\n'], ...
        betaGroupStats.PValueSignFlip(postRampFullRow));
else
    fprintf(fid, ['PostRamp is not significant after absolutePosition and other confounds are included: ' ...
        'the ramp may mainly reflect serial position/elapsed time. Interpret it conservatively as ' ...
        'post-capacity state dynamics rather than clear WM-specific updating.\n']);
end

if familyComparisonStats.MeanDifference(fullComparisonRow) > 0 && ...
        familyComparisonStats.PValueSignFlip(fullComparisonRow) < cfg.alpha
    fprintf(fid, ['FullModel exceeds confoundOnly (mean delta adjusted R2=%.6f, p=%.6g): ' ...
        'memory-state predictors explain additional empirical RDM variance.\n'], ...
        familyComparisonStats.MeanDifference(fullComparisonRow), ...
        familyComparisonStats.PValueSignFlip(fullComparisonRow));
else
    fprintf(fid, ['FullModel does not significantly exceed confoundOnly: additional explanatory value ' ...
        'from memory-state predictors is not established.\n']);
end

if ~(familyComparisonStats.MeanDifference(memoryPositionRow) > 0 && ...
        familyComparisonStats.PValueSignFlip(memoryPositionRow) < cfg.alpha)
    fprintf(fid, ['MemoryOnly does not significantly outperform absolutePositionOnly: do not claim that ' ...
        'CDA post-capacity dynamics are independent of temporal progression.\n']);
else
    fprintf(fid, ['MemoryOnly significantly outperforms absolutePositionOnly (mean delta adjusted R2=%.6f, ' ...
        'p=%.6g), supporting information beyond absolute serial position alone.\n'], ...
        familyComparisonStats.MeanDifference(memoryPositionRow), ...
        familyComparisonStats.PValueSignFlip(memoryPositionRow));
end
fclose(fid);

fprintf('\nRSA theoretical model comparison finished.\n');
fprintf('Subjects: %d\n', nSubjects);
fprintf('Area status: %s\n', areaStatus);
fprintf('Outputs saved under:\n%s\n', outputDir);

%% Local functions
function result = fit_rank_rsa(empiricalVector, theoryMatrix)
validRows = isfinite(empiricalVector) & all(isfinite(theoryMatrix), 2);
yRaw = empiricalVector(validRows);
XRaw = theoryMatrix(validRows, :);
nPredictors = size(XRaw, 2);
nPairs = numel(yRaw);

result = struct('semipartialR', nan(nPredictors, 1), ...
    'standardizedBeta', nan(nPredictors, 1), 'R2', NaN, ...
    'adjustedR2', NaN, 'effectivePredictorRank', NaN, 'nPairs', nPairs);
if nPairs <= nPredictors + 1
    return;
end

y = tiedrank(yRaw(:));
X = nan(size(XRaw));
for predictorIdx = 1:nPredictors
    X(:, predictorIdx) = tiedrank(XRaw(:, predictorIdx));
end

ySD = std(y, 0, 'omitnan');
if ~isfinite(ySD) || ySD <= 0
    return;
end
y = (y - mean(y, 'omitnan')) ./ ySD;
for predictorIdx = 1:nPredictors
    xSD = std(X(:, predictorIdx), 0, 'omitnan');
    if ~isfinite(xSD) || xSD <= 0
        return;
    end
    X(:, predictorIdx) = ...
        (X(:, predictorIdx) - mean(X(:, predictorIdx), 'omitnan')) ./ xSD;
end

XFull = [ones(nPairs, 1), X];
effectivePredictorRank = rank(X);
betaFull = pinv(XFull) * y;
yHatFull = XFull * betaFull;
sseFull = sum((y - yHatFull) .^ 2, 'omitnan');
sst = sum((y - mean(y, 'omitnan')) .^ 2, 'omitnan');
R2Full = 1 - sseFull / sst;
adjustedR2 = 1 - (1 - R2Full) * (nPairs - 1) / ...
    (nPairs - effectivePredictorRank - 1);

semipartialR = nan(nPredictors, 1);
for predictorIdx = 1:nPredictors
    keepPredictor = true(1, nPredictors);
    keepPredictor(predictorIdx) = false;
    XReduced = [ones(nPairs, 1), X(:, keepPredictor)];
    betaReduced = pinv(XReduced) * y;
    yHatReduced = XReduced * betaReduced;
    sseReduced = sum((y - yHatReduced) .^ 2, 'omitnan');
    R2Reduced = 1 - sseReduced / sst;
    deltaR2 = max(R2Full - R2Reduced, 0);
    semipartialR(predictorIdx) = sign(betaFull(predictorIdx + 1)) * sqrt(deltaR2);
end

result.semipartialR = semipartialR;
result.standardizedBeta = betaFull(2:end);
result.R2 = R2Full;
result.adjustedR2 = adjustedR2;
result.effectivePredictorRank = effectivePredictorRank;
end

function stats = sign_flip_test(values, nPerm)
values = values(isfinite(values));
n = numel(values);
stats = struct('mean', NaN, 'sem', NaN, 'ciLow', NaN, ...
    'ciHigh', NaN, 'p', NaN, 'n', n);
if n == 0
    return;
end

stats.mean = mean(values, 'omitnan');
stats.sem = std(values, 0, 'omitnan') / sqrt(n);
if n > 1
    criticalT = tinv(0.975, n - 1);
    stats.ciLow = stats.mean - criticalT * stats.sem;
    stats.ciHigh = stats.mean + criticalT * stats.sem;
end

signs = 2 * (rand(nPerm, n) > 0.5) - 1;
nullMean = mean(signs .* values(:)', 2);
stats.p = (sum(abs(nullMean) >= abs(stats.mean)) + 1) / (nPerm + 1);
end

function plot_subject_bars(subjectValues, labels, pValues, plotTitle, yLabelText, outFile, cfg)
if isvector(subjectValues)
    subjectValues = subjectValues(:);
end
nBars = size(subjectValues, 2);
meanValues = mean(subjectValues, 1, 'omitnan');
semValues = std(subjectValues, 0, 1, 'omitnan') ./ ...
    sqrt(sum(isfinite(subjectValues), 1));

fig = figure('Color', 'w', 'Visible', cfg.figureVisible, 'Position', [100 100 1100 620]);
ax = axes(fig);
bar(ax, 1:nBars, meanValues, 0.72, 'FaceColor', [0.35 0.55 0.78], ...
    'EdgeColor', 'none');
hold(ax, 'on');
errorbar(ax, 1:nBars, meanValues, semValues, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1.3, 'CapSize', 8);

for barIdx = 1:nBars
    valuesNow = subjectValues(:, barIdx);
    validNow = isfinite(valuesNow);
    jitter = linspace(-0.16, 0.16, sum(validNow))';
    scatter(ax, barIdx + jitter, valuesNow(validNow), 24, ...
        'MarkerFaceColor', [0.15 0.15 0.15], 'MarkerEdgeColor', 'w', ...
        'LineWidth', 0.4, 'MarkerFaceAlpha', 0.65);
end

yline(ax, 0, ':', 'Color', [0.35 0.35 0.35]);
set(ax, 'XTick', 1:nBars, 'XTickLabel', labels, ...
    'XTickLabelRotation', 35, 'TickDir', 'out');
ylabel(ax, yLabelText);
title(ax, plotTitle, 'Interpreter', 'none');
box(ax, 'off');

finiteValues = subjectValues(isfinite(subjectValues));
if isempty(finiteValues)
    yRange = [-1 1];
else
    yRange = [min([finiteValues; 0]), max([finiteValues; 0])];
    span = max(diff(yRange), 0.1);
    yRange = yRange + [-0.12 0.24] * span;
end
ylim(ax, yRange);

if ~isempty(pValues)
    for barIdx = 1:min(nBars, numel(pValues))
        if isfinite(pValues(barIdx))
            text(ax, barIdx, yRange(2) - 0.05 * diff(yRange), ...
                sprintf('p=%.3g', pValues(barIdx)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, ...
                'Rotation', 90);
        end
    end
end

print(fig, outFile, '-dpng', sprintf('-r%d', cfg.figureDpi));
close(fig);
end

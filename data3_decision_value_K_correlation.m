%% Subject-level behavior correlation for data3 decision values
% Run after data3_decision_behavior_correlation.m.
% This script reads the saved trial-level decision evidence CSV and does
% not rerun LDA decoding.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
resultDir = fullfile(dataDir, 'decision_behavior_correlation');
figureDir = fullfile(resultDir, 'figures');
behaviorDir = fullfile(dataDir, 'Behavior_data_script', 'Beh_data');

if ~isfolder(resultDir)
    mkdir(resultDir);
end
if ~isfolder(figureDir)
    mkdir(figureDir);
end

trialFile = fullfile(resultDir, 'data3_decision_behavior_trial_evidence.csv');
if ~isfile(trialFile)
    error('Missing trial evidence file: %s. Run data3_decision_behavior_correlation.m first.', trialFile);
end

TrialEvidence = readtable(trialFile);

requiredVars = {'Subject', 'Comparison', 'Model', 'Side', 'TrialIndex', ...
    'SerialPosition', 'Change', 'Correct', 'Evidence'};
missingVars = setdiff(requiredVars, TrialEvidence.Properties.VariableNames);
if ~isempty(missingVars)
    error('Trial evidence table is missing required columns: %s', strjoin(missingVars, ', '));
end

cfg = struct();
cfg.comparisons = {'setsize1_vs6_maintOnly', 'setsize3_vs6_maintOnly'};
cfg.models = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
cfg.latePositions = 4:6;
cfg.plateauPosition = 3;
cfg.initialPosition = 1;
cfg.serialPositions = 1:6;
cfg.behaviorMetrics = {'K6', 'K6MinusK1Corrected', 'Accuracy6', 'LateAccuracy6'};
cfg.behaviorBlocks = 1:4;
cfg.ratioDenominatorMinAbs = 0.05;
cfg.rawRatioDenominatorMinAbs = 1e-6;

modelColors = lines(numel(cfg.models));
comparisonColors = [0.13 0.42 0.76; 0.74 0.28 0.20];
comparisonLabels = {'SS1 vs SS6', 'SS3 vs SS6'};

% Evidence is the high-load / set-size-6 decision evidence.
% Do not reverse the sign: larger values mean stronger evidence for the set-size-6 class.

%% Step 1: Compute subject-level behavioral measures
% K6 keeps the original clean-trial behavior used in the earlier K6 figures.
% K6MinusK1Corrected uses the conventional corrected capacity formula from
% the original behavior script: set size * (hit - false alarm) / (1 - false alarm).
behaviorVars = {'Subject', 'TrialIndex', 'Change', 'Correct', 'SerialPosition'};
CleanBehaviorTrials6 = unique(TrialEvidence(:, behaviorVars), 'rows');
subjects = unique(CleanBehaviorTrials6.Subject);
nSubject = numel(subjects);

BehaviorSummary = table(subjects(:), 'VariableNames', {'Subject'});
behaviorSummaryVars = {'NTrial6', 'Accuracy6', 'LateAccuracy6', ...
    'HitRate6', 'FalseAlarmRate6', 'K6', ...
    'BehaviorNTrial1', 'BehaviorNTrial6', 'BehaviorAccuracy1', 'BehaviorAccuracy6', ...
    'BehaviorHitRate1', 'BehaviorFalseAlarmRate1', 'BehaviorK1', 'BehaviorK1Corrected', ...
    'BehaviorHitRate6', 'BehaviorFalseAlarmRate6', 'BehaviorK6', 'BehaviorK6Corrected', ...
    'K6MinusK1', 'K6MinusK1Corrected'};
for vi = 1:numel(behaviorSummaryVars)
    BehaviorSummary.(behaviorSummaryVars{vi}) = nan(nSubject, 1);
end

for si = 1:nSubject
    subjectRows = CleanBehaviorTrials6.Subject == subjects(si);
    T = CleanBehaviorTrials6(subjectRows, :);

    validCorrect = ~isnan(T.Correct);
    lateRows = ismember(T.SerialPosition, cfg.latePositions) & validCorrect;
    changeRows = T.Change == 1 & validCorrect;
    noChangeRows = T.Change == 0 & validCorrect;

    BehaviorSummary.NTrial6(si) = sum(validCorrect);
    BehaviorSummary.Accuracy6(si) = mean(T.Correct(validCorrect), 'omitnan');
    BehaviorSummary.LateAccuracy6(si) = mean(T.Correct(lateRows), 'omitnan');

    if any(changeRows)
        BehaviorSummary.HitRate6(si) = mean(T.Correct(changeRows) == 1);
    end
    if any(noChangeRows)
        BehaviorSummary.FalseAlarmRate6(si) = mean(T.Correct(noChangeRows) == 0);
    end
    if any(changeRows) && any(noChangeRows)
        BehaviorSummary.K6(si) = 6 * (BehaviorSummary.HitRate6(si) - BehaviorSummary.FalseAlarmRate6(si));
    end

    AllBehavior = table();
    for bi = 1:numel(cfg.behaviorBlocks)
        blockNum = cfg.behaviorBlocks(bi);
        behaviorFile = fullfile(behaviorDir, sprintf('cda_cVl_serial_data%d_%d.mat', subjects(si), blockNum));
        if ~isfile(behaviorFile)
            warning('Missing behavior file for subject %d block %d: %s', subjects(si), blockNum, behaviorFile);
            continue;
        end

        B = load(behaviorFile, 'data');
        if ~isfield(B, 'data')
            warning('%s does not contain data.', behaviorFile);
            continue;
        end

        behData = B.data;
        nBehTrial = numel(behData.set_size);
        BlockBehavior = table();
        BlockBehavior.SetSize = behData.set_size(:);
        BlockBehavior.Change = behData.change(:);
        BlockBehavior.Response = behData.resp(:);
        BlockBehavior.Correct = behData.acc(:);
        BlockBehavior.SerialPosition = behData.change_item(:);
        BlockBehavior.Block = repmat(blockNum, nBehTrial, 1);
        BlockBehavior.BlockTrial = (1:nBehTrial)';
        AllBehavior = [AllBehavior; BlockBehavior]; %#ok<AGROW>
    end

    for setSizeNow = [1 6]
        setRows = AllBehavior.SetSize == setSizeNow & ~isnan(AllBehavior.Correct);
        if ~any(setRows)
            continue;
        end

        Tset = AllBehavior(setRows, :);
        changeRows = Tset.Change == 1 & ~isnan(Tset.Response);
        noChangeRows = Tset.Change == 0 & ~isnan(Tset.Response);

        hitRate = nan;
        falseAlarmRate = nan;
        if any(changeRows)
            hitRate = mean(Tset.Response(changeRows) == 1);
        end
        if any(noChangeRows)
            falseAlarmRate = mean(Tset.Response(noChangeRows) == 1);
        end

        behaviorK = nan;
        behaviorKCorrected = nan;
        if ~isnan(hitRate) && ~isnan(falseAlarmRate)
            behaviorK = setSizeNow * (hitRate - falseAlarmRate);
            if falseAlarmRate < 1
                behaviorKCorrected = setSizeNow * (hitRate - falseAlarmRate) / (1 - falseAlarmRate);
            end
        end

        if setSizeNow == 1
            BehaviorSummary.BehaviorNTrial1(si) = sum(setRows);
            BehaviorSummary.BehaviorAccuracy1(si) = mean(Tset.Correct, 'omitnan');
            BehaviorSummary.BehaviorHitRate1(si) = hitRate;
            BehaviorSummary.BehaviorFalseAlarmRate1(si) = falseAlarmRate;
            BehaviorSummary.BehaviorK1(si) = behaviorK;
            BehaviorSummary.BehaviorK1Corrected(si) = behaviorKCorrected;
        else
            BehaviorSummary.BehaviorNTrial6(si) = sum(setRows);
            BehaviorSummary.BehaviorAccuracy6(si) = mean(Tset.Correct, 'omitnan');
            BehaviorSummary.BehaviorHitRate6(si) = hitRate;
            BehaviorSummary.BehaviorFalseAlarmRate6(si) = falseAlarmRate;
            BehaviorSummary.BehaviorK6(si) = behaviorK;
            BehaviorSummary.BehaviorK6Corrected(si) = behaviorKCorrected;
        end
    end

    BehaviorSummary.K6MinusK1(si) = BehaviorSummary.BehaviorK6(si) - BehaviorSummary.BehaviorK1(si);
    BehaviorSummary.K6MinusK1Corrected(si) = BehaviorSummary.BehaviorK6Corrected(si) - BehaviorSummary.BehaviorK1Corrected(si);
end

writetable(BehaviorSummary, fullfile(resultDir, 'data3_subject_K6_behavior_summary.csv'));
writetable(BehaviorSummary, fullfile(resultDir, 'data3_subject_K_behavior_summary.csv'));

%% Step 2: Standardize decision evidence within subject
TrialEvidence.EvidenceZSubject = nan(height(TrialEvidence), 1);

for si = 1:nSubject
    for ci = 1:numel(cfg.comparisons)
        comparisonName = cfg.comparisons{ci};
        for mi = 1:numel(cfg.models)
            modelName = cfg.models{mi};
            rows = TrialEvidence.Subject == subjects(si) & ...
                strcmp(TrialEvidence.Comparison, comparisonName) & ...
                strcmp(TrialEvidence.Model, modelName) & ...
                ~isnan(TrialEvidence.Evidence);

            if ~any(rows)
                continue;
            end

            mu = mean(TrialEvidence.Evidence(rows), 'omitnan');
            sd = std(TrialEvidence.Evidence(rows), 0, 'omitnan');
            if sd == 0 || isnan(sd)
                sd = 1;
            end
            TrialEvidence.EvidenceZSubject(rows) = (TrialEvidence.Evidence(rows) - mu) ./ sd;
        end
    end
end

%% Step 3: Compute subject-level decision evidence by serial position
positionTables = {};

for si = 1:nSubject
    for ci = 1:numel(cfg.comparisons)
        comparisonName = cfg.comparisons{ci};
        for mi = 1:numel(cfg.models)
            modelName = cfg.models{mi};
            for pi = 1:numel(cfg.serialPositions)
                serialPosition = cfg.serialPositions(pi);
                rows = TrialEvidence.Subject == subjects(si) & ...
                    strcmp(TrialEvidence.Comparison, comparisonName) & ...
                    strcmp(TrialEvidence.Model, modelName) & ...
                    TrialEvidence.SerialPosition == serialPosition & ...
                    ~isnan(TrialEvidence.Evidence);

                if ~any(rows)
                    continue;
                end

                S = table();
                S.Subject = subjects(si);
                S.Comparison = {comparisonName};
                S.Model = {modelName};
                S.SerialPosition = serialPosition;
                S.MeanEvidenceRaw = mean(TrialEvidence.Evidence(rows), 'omitnan');
                S.MeanEvidenceZSubject = mean(TrialEvidence.EvidenceZSubject(rows), 'omitnan');
                S.NTrial = sum(~isnan(TrialEvidence.Evidence(rows)));

                positionTables{end+1, 1} = S; %#ok<SAGROW>
            end
        end
    end
end

if isempty(positionTables)
    error('No subject-position decision evidence rows were found.');
end

PositionEvidence = vertcat(positionTables{:});
writetable(PositionEvidence, fullfile(resultDir, 'data3_subject_position_decision_evidence.csv'));

%% Step 4: Compute post-plateau and reference decision-value indices
% For SS3 vs SS6, use only pos6 - pos3. Do not average pos4-6.
% Position 3 is treated as the traditional CDA plateau baseline.
% This tests whether post-plateau decision evidence is related to individual K6.
%
% For SS1 vs SS6, mean(pos4-6) - pos1 is included as a reference contrast
% against a minimal-load initial baseline.
metricTables = {};

for si = 1:nSubject
    for ci = 1:numel(cfg.comparisons)
        comparisonName = cfg.comparisons{ci};
        for mi = 1:numel(cfg.models)
            modelName = cfg.models{mi};

            rows = PositionEvidence.Subject == subjects(si) & ...
                strcmp(PositionEvidence.Comparison, comparisonName) & ...
                strcmp(PositionEvidence.Model, modelName);

            if ~any(rows)
                continue;
            end

            dvZ = nan(1, numel(cfg.serialPositions));
            dvRaw = nan(1, numel(cfg.serialPositions));
            nTrialByPosition = nan(1, numel(cfg.serialPositions));

            for pi = 1:numel(cfg.serialPositions)
                serialPosition = cfg.serialPositions(pi);
                positionRows = rows & PositionEvidence.SerialPosition == serialPosition;
                if any(positionRows)
                    dvZ(pi) = mean(PositionEvidence.MeanEvidenceZSubject(positionRows), 'omitnan');
                    dvRaw(pi) = mean(PositionEvidence.MeanEvidenceRaw(positionRows), 'omitnan');
                    nTrialByPosition(pi) = sum(PositionEvidence.NTrial(positionRows), 'omitnan');
                end
            end

            lateZ = dvZ(cfg.latePositions);
            lateRaw = dvRaw(cfg.latePositions);
            if all(~isnan(lateZ))
                lateMeanDV = mean(lateZ);
            else
                lateMeanDV = nan;
            end
            if all(~isnan(lateRaw))
                lateMeanDVRaw = mean(lateRaw);
            else
                lateMeanDVRaw = nan;
            end

            S = table();
            S.Subject = subjects(si);
            S.Comparison = {comparisonName};
            S.Model = {modelName};
            S.LateMeanDV = lateMeanDV;
            S.PostPlateauDV = nan;
            S.LateMinusInitialDV = nan;
            S.DeltaPos4MinusPos3 = nan;
            S.DeltaPos5MinusPos3 = nan;
            S.DeltaPos6MinusPos3 = nan;
            S.DeltaPos4MinusPos1 = nan;
            S.DeltaPos5MinusPos1 = nan;
            S.DeltaPos6MinusPos1 = nan;
            S.DV_pos1 = dvZ(1);
            S.DV_pos2 = dvZ(2);
            S.DV_pos3 = dvZ(3);
            S.DV_pos4 = dvZ(4);
            S.DV_pos5 = dvZ(5);
            S.DV_pos6 = dvZ(6);
            S.NTrial_pos1 = nTrialByPosition(1);
            S.NTrial_pos2 = nTrialByPosition(2);
            S.NTrial_pos3 = nTrialByPosition(3);
            S.NTrial_pos4 = nTrialByPosition(4);
            S.NTrial_pos5 = nTrialByPosition(5);
            S.NTrial_pos6 = nTrialByPosition(6);
            S.LateMeanDVRaw = lateMeanDVRaw;
            S.PostPlateauDVRaw = nan;
            S.LateMinusInitialDVRaw = nan;
            S.DeltaPos4MinusPos3Raw = nan;
            S.DeltaPos5MinusPos3Raw = nan;
            S.DeltaPos6MinusPos3Raw = nan;
            S.DeltaPos4MinusPos1Raw = nan;
            S.DeltaPos5MinusPos1Raw = nan;
            S.DeltaPos6MinusPos1Raw = nan;
            S.DVRaw_pos1 = dvRaw(1);
            S.DVRaw_pos2 = dvRaw(2);
            S.DVRaw_pos3 = dvRaw(3);
            S.DVRaw_pos4 = dvRaw(4);
            S.DVRaw_pos5 = dvRaw(5);
            S.DVRaw_pos6 = dvRaw(6);

            if strcmp(comparisonName, 'setsize3_vs6_maintOnly')
                S.PostPlateauDV = dvZ(6) - dvZ(cfg.plateauPosition);
                S.DeltaPos4MinusPos3 = dvZ(4) - dvZ(3);
                S.DeltaPos5MinusPos3 = dvZ(5) - dvZ(3);
                S.DeltaPos6MinusPos3 = dvZ(6) - dvZ(3);
                S.PostPlateauDVRaw = dvRaw(6) - dvRaw(cfg.plateauPosition);
                S.DeltaPos4MinusPos3Raw = dvRaw(4) - dvRaw(3);
                S.DeltaPos5MinusPos3Raw = dvRaw(5) - dvRaw(3);
                S.DeltaPos6MinusPos3Raw = dvRaw(6) - dvRaw(3);
            elseif strcmp(comparisonName, 'setsize1_vs6_maintOnly')
                S.LateMinusInitialDV = lateMeanDV - dvZ(cfg.initialPosition);
                S.DeltaPos4MinusPos1 = dvZ(4) - dvZ(1);
                S.DeltaPos5MinusPos1 = dvZ(5) - dvZ(1);
                S.DeltaPos6MinusPos1 = dvZ(6) - dvZ(1);
                S.LateMinusInitialDVRaw = lateMeanDVRaw - dvRaw(cfg.initialPosition);
                S.DeltaPos4MinusPos1Raw = dvRaw(4) - dvRaw(1);
                S.DeltaPos5MinusPos1Raw = dvRaw(5) - dvRaw(1);
                S.DeltaPos6MinusPos1Raw = dvRaw(6) - dvRaw(1);
            end

            metricTables{end+1, 1} = S; %#ok<SAGROW>
        end
    end
end

NeuralMetrics = vertcat(metricTables{:});
Metrics = join(NeuralMetrics, BehaviorSummary, 'Keys', 'Subject');

Metrics.Pos6DV = Metrics.DV_pos6;
Metrics.Pos6MinusPos1DV = Metrics.DeltaPos6MinusPos1;
Metrics.Pos6MinusPos1OverPos1DV = nan(height(Metrics), 1);
Metrics.Pos6MinusPos1OverMeanPos1Pos6DV = nan(height(Metrics), 1);

denomPos1 = Metrics.DV_pos1;
ratioRows = abs(denomPos1) >= cfg.ratioDenominatorMinAbs;
Metrics.Pos6MinusPos1OverPos1DV(ratioRows) = ...
    (Metrics.DV_pos6(ratioRows) - Metrics.DV_pos1(ratioRows)) ./ denomPos1(ratioRows);

denomMeanPos1Pos6 = (Metrics.DV_pos6 + Metrics.DV_pos1) ./ 2;
ratioRows = abs(denomMeanPos1Pos6) >= cfg.ratioDenominatorMinAbs;
Metrics.Pos6MinusPos1OverMeanPos1Pos6DV(ratioRows) = ...
    (Metrics.DV_pos6(ratioRows) - Metrics.DV_pos1(ratioRows)) ./ denomMeanPos1Pos6(ratioRows);

Metrics.Pos6RawDV = Metrics.DVRaw_pos6;
Metrics.Pos6MinusPos1RawDV = Metrics.DeltaPos6MinusPos1Raw;
Metrics.Pos6MinusPos1OverPos1RawDV = nan(height(Metrics), 1);
Metrics.Pos6MinusPos1OverMeanPos1Pos6RawDV = nan(height(Metrics), 1);

denomPos1Raw = Metrics.DVRaw_pos1;
ratioRows = abs(denomPos1Raw) >= cfg.rawRatioDenominatorMinAbs;
Metrics.Pos6MinusPos1OverPos1RawDV(ratioRows) = ...
    (Metrics.DVRaw_pos6(ratioRows) - Metrics.DVRaw_pos1(ratioRows)) ./ denomPos1Raw(ratioRows);

denomMeanPos1Pos6Raw = (Metrics.DVRaw_pos6 + Metrics.DVRaw_pos1) ./ 2;
ratioRows = abs(denomMeanPos1Pos6Raw) >= cfg.rawRatioDenominatorMinAbs;
Metrics.Pos6MinusPos1OverMeanPos1Pos6RawDV(ratioRows) = ...
    (Metrics.DVRaw_pos6(ratioRows) - Metrics.DVRaw_pos1(ratioRows)) ./ denomMeanPos1Pos6Raw(ratioRows);

writetable(Metrics, fullfile(resultDir, 'data3_subject_decision_value_K_metrics.csv'));

%% Step 5: Correlate decision-value indices with K and accuracy
correlationTables = {};

for ci = 1:numel(cfg.comparisons)
    comparisonName = cfg.comparisons{ci};
    if strcmp(comparisonName, 'setsize3_vs6_maintOnly')
        neuralMetric = 'DeltaPos6MinusPos3';
    else
        neuralMetric = 'LateMinusInitialDV';
    end

    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(Metrics.Comparison, comparisonName) & strcmp(Metrics.Model, modelName);

        for bi = 1:numel(cfg.behaviorMetrics)
            behaviorMetric = cfg.behaviorMetrics{bi};
            x = Metrics.(neuralMetric)(rows);
            y = Metrics.(behaviorMetric)(rows);
            validRows = ~isnan(x) & ~isnan(y);
            nValid = sum(validRows);

            pearsonR = nan;
            pearsonP = nan;
            spearmanRho = nan;
            spearmanP = nan;

            if nValid >= 2 && std(x(validRows), 0) > 0 && std(y(validRows), 0) > 0
                [pearsonR, pearsonP] = corr(x(validRows), y(validRows), 'Type', 'Pearson');
                [spearmanRho, spearmanP] = corr(x(validRows), y(validRows), 'Type', 'Spearman');
            end

            S = table();
            S.Comparison = {comparisonName};
            S.Model = {modelName};
            S.NeuralMetric = {neuralMetric};
            S.BehaviorMetric = {behaviorMetric};
            S.NSubject = nValid;
            S.PearsonR = pearsonR;
            S.PearsonP = pearsonP;
            S.SpearmanRho = spearmanRho;
            S.SpearmanP = spearmanP;

            correlationTables{end+1, 1} = S; %#ok<SAGROW>
        end
    end
end

CorrelationSummary = vertcat(correlationTables{:});
writetable(CorrelationSummary, fullfile(resultDir, 'data3_decision_value_K_correlation_summary.csv'));

%% Step 6: Plot main scatter figures
scatterComparisons = {'setsize3_vs6_maintOnly', 'setsize1_vs6_maintOnly'};
scatterMetrics = {'DeltaPos6MinusPos3', 'LateMinusInitialDV'};
scatterBehaviorMetrics = {'K6', 'K6MinusK1Corrected'};
scatterTitles = {'SS3 vs SS6: pos6-minus-pos3 DV vs K6', ...
    'SS1 vs SS6: late-minus-initial DV vs corrected K6-minus-K1'};
scatterXLabels = {{'Post-plateau DV', 'pos6 - pos3'}, ...
    {'Late-minus-initial DV', 'mean(pos4-6) - pos1'}};
scatterYLabels = {'Set-size-6 K', {'Corrected behavior K difference', 'K6 - K1'}};
scatterFileBases = {'data3_SS3vsSS6_post_plateau_DV_K6_correlation', ...
    'data3_SS1vsSS6_late_minus_initial_DV_corrected_K6_minus_K1_correlation'};

for fi = 1:numel(scatterComparisons)
    comparisonName = scatterComparisons{fi};
    neuralMetric = scatterMetrics{fi};
    behaviorMetric = scatterBehaviorMetrics{fi};

    fig = figure('Color', 'w', 'Position', [80 120 1500 360]);
    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(Metrics.Comparison, comparisonName) & strcmp(Metrics.Model, modelName);
        x = Metrics.(neuralMetric)(rows);
        y = Metrics.(behaviorMetric)(rows);
        validRows = ~isnan(x) & ~isnan(y);

        subplot(1, numel(cfg.models), mi);
        hold on;
        scatter(x(validRows), y(validRows), 38, ...
            'MarkerFaceColor', modelColors(mi,:), ...
            'MarkerEdgeColor', [0.15 0.15 0.15], ...
            'LineWidth', 0.5);

        if sum(validRows) >= 2 && std(x(validRows), 0) > 0 && std(y(validRows), 0) > 0
            fitCoef = polyfit(x(validRows), y(validRows), 1);
            xLine = linspace(min(x(validRows)), max(x(validRows)), 100);
            plot(xLine, polyval(fitCoef, xLine), 'k-', 'LineWidth', 1.2);
        end

        if any(validRows)
            xMin = min(x(validRows));
            xMax = max(x(validRows));
            yMin = min(y(validRows));
            yMax = max(y(validRows));
            xPad = max((xMax - xMin) * 0.12, 0.10);
            yPad = max((yMax - yMin) * 0.12, 0.25);
            xlim([xMin - xPad, xMax + xPad]);
            ylim([yMin - yPad, yMax + yPad]);
        end

        statRows = strcmp(CorrelationSummary.Comparison, comparisonName) & ...
            strcmp(CorrelationSummary.Model, modelName) & ...
            strcmp(CorrelationSummary.NeuralMetric, neuralMetric) & ...
            strcmp(CorrelationSummary.BehaviorMetric, behaviorMetric);
        if any(statRows)
            statsText = sprintf('Pearson r = %.2f, p = %.3f\nSpearman rho = %.2f, p = %.3f\nN = %d', ...
                CorrelationSummary.PearsonR(statRows), ...
                CorrelationSummary.PearsonP(statRows), ...
                CorrelationSummary.SpearmanRho(statRows), ...
                CorrelationSummary.SpearmanP(statRows), ...
                CorrelationSummary.NSubject(statRows));
        else
            statsText = 'No complete rows';
        end

        ax = gca;
        xl = xlim(ax);
        yl = ylim(ax);
        text(xl(1) + 0.05 * diff(xl), yl(2) - 0.08 * diff(yl), statsText, ...
            'VerticalAlignment', 'top', ...
            'FontName', 'Arial', ...
            'FontSize', 8);

        title(modelName, 'Interpreter', 'none');
        xlabel(scatterXLabels{fi}, 'FontSize', 8);
        if mi == 1
            ylabel(scatterYLabels{fi});
        end
        box off;
        set(gca, 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 9);
    end

    sgtitle(scatterTitles{fi}, 'FontName', 'Arial', 'FontWeight', 'bold');
    savefig(fig, fullfile(figureDir, [scatterFileBases{fi} '.fig']));
    print(fig, fullfile(figureDir, [scatterFileBases{fi} '.png']), '-dpng', '-r300');
end

%% Step 7: Plot decision evidence by serial position
fig = figure('Color', 'w', 'Position', [120 120 1100 430]);

for ci = 1:numel(cfg.comparisons)
    comparisonName = cfg.comparisons{ci};
    subplot(1, numel(cfg.comparisons), ci);
    hold on;

    allY = [];
    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        subjectByPosition = nan(nSubject, numel(cfg.serialPositions));

        for si = 1:nSubject
            for pi = 1:numel(cfg.serialPositions)
                serialPosition = cfg.serialPositions(pi);
                rows = PositionEvidence.Subject == subjects(si) & ...
                    strcmp(PositionEvidence.Comparison, comparisonName) & ...
                    strcmp(PositionEvidence.Model, modelName) & ...
                    PositionEvidence.SerialPosition == serialPosition;
                if any(rows)
                    subjectByPosition(si, pi) = mean(PositionEvidence.MeanEvidenceZSubject(rows), 'omitnan');
                end
            end
        end

        groupMean = mean(subjectByPosition, 1, 'omitnan');
        groupSem = std(subjectByPosition, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(subjectByPosition), 1));
        allY = [allY, groupMean - groupSem, groupMean + groupSem]; %#ok<AGROW>

        errorbar(cfg.serialPositions, groupMean, groupSem, 'o-', ...
            'Color', modelColors(mi,:), ...
            'MarkerFaceColor', modelColors(mi,:), ...
            'LineWidth', 1.6, ...
            'MarkerSize', 5, ...
            'DisplayName', modelName);
    end

    yline(0, ':k', 'HandleVisibility', 'off');
    if any(~isnan(allY))
        yMin = min(allY(~isnan(allY)));
        yMax = max(allY(~isnan(allY)));
        yPad = max((yMax - yMin) * 0.12, 0.10);
        ylim([yMin - yPad, yMax + yPad]);
    end
    yl = ylim;

    if strcmp(comparisonName, 'setsize3_vs6_maintOnly')
        targetPatch = patch([5.5 6.5 6.5 5.5], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.90 0.90 0.90], ...
            'FaceAlpha', 0.35, ...
            'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
        uistack(targetPatch, 'bottom');
        xline(cfg.plateauPosition, '--k', 'pos3 baseline', ...
            'LabelVerticalAlignment', 'bottom', ...
            'HandleVisibility', 'off');
        xline(6, ':k', 'pos6 target', ...
            'LabelVerticalAlignment', 'bottom', ...
            'HandleVisibility', 'off');
        text(5.65, yl(2) - 0.06 * diff(yl), 'pos6 - pos3', ...
            'HorizontalAlignment', 'center', ...
            'FontName', 'Arial', ...
            'FontSize', 9);
    else
        latePatch = patch([3.5 6.5 6.5 3.5], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.90 0.90 0.90], ...
            'FaceAlpha', 0.35, ...
            'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
        uistack(latePatch, 'bottom');
        xline(cfg.initialPosition, '--k', 'pos1 baseline', ...
            'LabelVerticalAlignment', 'bottom', ...
            'HandleVisibility', 'off');
        text(5, yl(2) - 0.06 * diff(yl), 'late pos4-6', ...
            'HorizontalAlignment', 'center', ...
            'FontName', 'Arial', ...
            'FontSize', 9);
    end

    xlim([0.75 6.25]);
    xlabel('Set-size-6 serial position');
    ylabel('Mean high-load decision evidence (subject z)');
    title(comparisonLabels{ci}, 'Interpreter', 'none');
    if ci == 1
        legend('Location', 'best', 'Box', 'off');
    end
    box off;
    set(gca, 'TickDir', 'out', ...
        'XTick', cfg.serialPositions, ...
        'FontName', 'Arial', ...
        'FontSize', 10);
end

savefig(fig, fullfile(figureDir, 'data3_decision_evidence_by_serial_position.fig'));
print(fig, fullfile(figureDir, 'data3_decision_evidence_by_serial_position.png'), '-dpng', '-r300');

%% Step 8: Plot correlation coefficient summary
fig = figure('Color', 'w', 'Position', [120 120 980 390]);

for ci = 1:numel(cfg.comparisons)
    comparisonName = cfg.comparisons{ci};
    if strcmp(comparisonName, 'setsize3_vs6_maintOnly')
        neuralMetric = 'DeltaPos6MinusPos3';
        behaviorMetric = 'K6';
        behaviorLabel = 'K6';
    else
        neuralMetric = 'LateMinusInitialDV';
        behaviorMetric = 'K6MinusK1Corrected';
        behaviorLabel = 'corrected K6 - K1';
    end

    rho = nan(1, numel(cfg.models));
    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(CorrelationSummary.Comparison, comparisonName) & ...
            strcmp(CorrelationSummary.Model, modelName) & ...
            strcmp(CorrelationSummary.NeuralMetric, neuralMetric) & ...
            strcmp(CorrelationSummary.BehaviorMetric, behaviorMetric);
        if any(rows)
            rho(mi) = CorrelationSummary.SpearmanRho(rows);
        end
    end

    subplot(1, numel(cfg.comparisons), ci);
    plot(1:numel(cfg.models), rho, 'o-', ...
        'Color', comparisonColors(ci,:), ...
        'MarkerFaceColor', comparisonColors(ci,:), ...
        'LineWidth', 1.8, ...
        'MarkerSize', 6);
    hold on;
    yline(0, ':k', 'HandleVisibility', 'off');
    xlim([0.5 numel(cfg.models) + 0.5]);
    ylim([-1 1]);
    set(gca, 'XTick', 1:numel(cfg.models), ...
        'XTickLabel', cfg.models, ...
        'XTickLabelRotation', 30, ...
        'TickDir', 'out', ...
        'FontName', 'Arial', ...
        'FontSize', 10);
    ylabel('Spearman rho');
    title(sprintf('%s vs %s', comparisonLabels{ci}, behaviorLabel), 'Interpreter', 'none');
    box off;
end

savefig(fig, fullfile(figureDir, 'data3_decision_value_behavior_correlation_model_summary.fig'));
print(fig, fullfile(figureDir, 'data3_decision_value_behavior_correlation_model_summary.png'), '-dpng', '-r300');

%% Step 9: Sensitivity check for SS1-vs-SS6 neural metrics against corrected K6-minus-K1
sensitivityComparison = 'setsize1_vs6_maintOnly';
sensitivityBehaviorMetric = 'K6MinusK1Corrected';
sensitivityMetricSpecs = { ...
    'LateMinusInitialDV', 'mean(pos4-6) - pos1', 'subject z'; ...
    'Pos6DV', 'pos6 only', 'subject z'; ...
    'Pos6MinusPos1DV', 'pos6 - pos1', 'subject z'; ...
    'Pos6MinusPos1OverPos1DV', '(pos6 - pos1) / pos1', 'subject z ratio'; ...
    'Pos6MinusPos1OverMeanPos1Pos6DV', '(pos6 - pos1) / mean(pos1,pos6)', 'subject z ratio'; ...
    'LateMinusInitialDVRaw', 'raw mean(pos4-6) - pos1', 'raw'; ...
    'Pos6RawDV', 'raw pos6 only', 'raw'; ...
    'Pos6MinusPos1RawDV', 'raw pos6 - pos1', 'raw'; ...
    'Pos6MinusPos1OverPos1RawDV', 'raw (pos6 - pos1) / pos1', 'raw ratio'; ...
    'Pos6MinusPos1OverMeanPos1Pos6RawDV', 'raw (pos6 - pos1) / mean(pos1,pos6)', 'raw ratio'};

sensitivityTables = {};
for mi = 1:numel(cfg.models)
    modelName = cfg.models{mi};
    modelRows = strcmp(Metrics.Comparison, sensitivityComparison) & ...
        strcmp(Metrics.Model, modelName);

    for vi = 1:size(sensitivityMetricSpecs, 1)
        neuralMetric = sensitivityMetricSpecs{vi, 1};
        metricLabel = sensitivityMetricSpecs{vi, 2};
        metricScale = sensitivityMetricSpecs{vi, 3};

        x = Metrics.(neuralMetric)(modelRows);
        y = Metrics.(sensitivityBehaviorMetric)(modelRows);
        validRows = ~isnan(x) & ~isnan(y);
        nValid = sum(validRows);

        pearsonR = nan;
        pearsonP = nan;
        spearmanRho = nan;
        spearmanP = nan;
        xMean = nan;
        xStd = nan;
        yMean = nan;
        yStd = nan;

        if nValid > 0
            xMean = mean(x(validRows), 'omitnan');
            xStd = std(x(validRows), 0, 'omitnan');
            yMean = mean(y(validRows), 'omitnan');
            yStd = std(y(validRows), 0, 'omitnan');
        end

        if nValid >= 2 && std(x(validRows), 0) > 0 && std(y(validRows), 0) > 0
            [pearsonR, pearsonP] = corr(x(validRows), y(validRows), 'Type', 'Pearson');
            [spearmanRho, spearmanP] = corr(x(validRows), y(validRows), 'Type', 'Spearman');
        end

        S = table();
        S.Comparison = {sensitivityComparison};
        S.Model = {modelName};
        S.NeuralMetric = {neuralMetric};
        S.MetricLabel = {metricLabel};
        S.MetricScale = {metricScale};
        S.BehaviorMetric = {sensitivityBehaviorMetric};
        S.NSubject = nValid;
        S.XMean = xMean;
        S.XStd = xStd;
        S.YMean = yMean;
        S.YStd = yStd;
        S.PearsonR = pearsonR;
        S.PearsonP = pearsonP;
        S.SpearmanRho = spearmanRho;
        S.SpearmanP = spearmanP;
        S.AbsSpearmanRho = abs(spearmanRho);

        sensitivityTables{end+1, 1} = S; %#ok<SAGROW>
    end
end

SS1MetricSensitivity = vertcat(sensitivityTables{:});
writetable(SS1MetricSensitivity, fullfile(resultDir, ...
    'data3_SS1vsSS6_DV_metric_sensitivity_corrected_K6_minus_K1_summary.csv'));

rhoMat = nan(size(sensitivityMetricSpecs, 1), numel(cfg.models));
nMat = nan(size(sensitivityMetricSpecs, 1), numel(cfg.models));
for vi = 1:size(sensitivityMetricSpecs, 1)
    neuralMetric = sensitivityMetricSpecs{vi, 1};
    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(SS1MetricSensitivity.Model, modelName) & ...
            strcmp(SS1MetricSensitivity.NeuralMetric, neuralMetric);
        if any(rows)
            rhoMat(vi, mi) = SS1MetricSensitivity.SpearmanRho(rows);
            nMat(vi, mi) = SS1MetricSensitivity.NSubject(rows);
        end
    end
end

fig = figure('Color', 'w', 'Position', [120 100 1050 560]);
imagesc(rhoMat, [-1 1]);
nColor = 256;
nBlue = floor(nColor / 2);
nRed = nColor - nBlue;
blue = [0.16 0.32 0.72];
white = [1 1 1];
red = [0.78 0.18 0.16];
cmap = [interp1([1 nBlue], [blue; white], 1:nBlue); ...
    interp1([1 nRed], [white; red], 1:nRed)];
colormap(gca, cmap);
colorbar;
set(gca, 'XTick', 1:numel(cfg.models), ...
    'XTickLabel', cfg.models, ...
    'YTick', 1:size(sensitivityMetricSpecs, 1), ...
    'YTickLabel', sensitivityMetricSpecs(:,2), ...
    'TickDir', 'out', ...
    'FontName', 'Arial', ...
    'FontSize', 9);
xlabel('Decoder model');
ylabel('SS1-vs-SS6 neural metric');
title('Spearman rho with corrected K6 - K1', 'FontName', 'Arial', 'FontWeight', 'bold');

for vi = 1:size(rhoMat, 1)
    for mi = 1:size(rhoMat, 2)
        if isnan(rhoMat(vi, mi))
            cellText = 'n/a';
        else
            cellText = sprintf('%.2f\nN=%d', rhoMat(vi, mi), nMat(vi, mi));
        end
        text(mi, vi, cellText, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontName', 'Arial', ...
            'FontSize', 8, ...
            'Color', [0.05 0.05 0.05]);
    end
end

savefig(fig, fullfile(figureDir, ...
    'data3_SS1vsSS6_DV_metric_sensitivity_corrected_K6_minus_K1_heatmap.fig'));
print(fig, fullfile(figureDir, ...
    'data3_SS1vsSS6_DV_metric_sensitivity_corrected_K6_minus_K1_heatmap.png'), '-dpng', '-r300');

cdaScatterMetrics = {'Pos6DV', 'Pos6MinusPos1DV', 'LateMinusInitialDV', ...
    'Pos6MinusPos1OverPos1DV', 'Pos6MinusPos1OverMeanPos1Pos6DV'};
cdaScatterLabels = {'pos6 only', 'pos6 - pos1', 'mean(pos4-6) - pos1', ...
    '(pos6 - pos1) / pos1', '(pos6 - pos1) / mean(pos1,pos6)'};

fig = figure('Color', 'w', 'Position', [90 100 1500 360]);
for vi = 1:numel(cdaScatterMetrics)
    neuralMetric = cdaScatterMetrics{vi};
    modelRows = strcmp(Metrics.Comparison, sensitivityComparison) & strcmp(Metrics.Model, 'CDA');
    x = Metrics.(neuralMetric)(modelRows);
    y = Metrics.(sensitivityBehaviorMetric)(modelRows);
    validRows = ~isnan(x) & ~isnan(y);

    subplot(1, numel(cdaScatterMetrics), vi);
    hold on;
    scatter(x(validRows), y(validRows), 40, ...
        'MarkerFaceColor', modelColors(strcmp(cfg.models, 'CDA'),:), ...
        'MarkerEdgeColor', [0.15 0.15 0.15], ...
        'LineWidth', 0.5);

    if sum(validRows) >= 2 && std(x(validRows), 0) > 0 && std(y(validRows), 0) > 0
        fitCoef = polyfit(x(validRows), y(validRows), 1);
        xLine = linspace(min(x(validRows)), max(x(validRows)), 100);
        plot(xLine, polyval(fitCoef, xLine), 'k-', 'LineWidth', 1.2);
    end

    if any(validRows)
        xMin = min(x(validRows));
        xMax = max(x(validRows));
        yMin = min(y(validRows));
        yMax = max(y(validRows));
        xPad = max((xMax - xMin) * 0.12, 0.10);
        yPad = max((yMax - yMin) * 0.12, 0.25);
        xlim([xMin - xPad, xMax + xPad]);
        ylim([yMin - yPad, yMax + yPad]);
    end

    statRows = strcmp(SS1MetricSensitivity.Model, 'CDA') & ...
        strcmp(SS1MetricSensitivity.NeuralMetric, neuralMetric);
    if any(statRows)
        statsText = sprintf('Pearson r = %.2f, p = %.3f\nSpearman rho = %.2f, p = %.3f\nN = %d', ...
            SS1MetricSensitivity.PearsonR(statRows), ...
            SS1MetricSensitivity.PearsonP(statRows), ...
            SS1MetricSensitivity.SpearmanRho(statRows), ...
            SS1MetricSensitivity.SpearmanP(statRows), ...
            SS1MetricSensitivity.NSubject(statRows));
    else
        statsText = 'No complete rows';
    end

    ax = gca;
    xl = xlim(ax);
    yl = ylim(ax);
    text(xl(1) + 0.05 * diff(xl), yl(2) - 0.08 * diff(yl), statsText, ...
        'VerticalAlignment', 'top', ...
        'FontName', 'Arial', ...
        'FontSize', 8);

    title(cdaScatterLabels{vi}, 'Interpreter', 'none');
    xlabel(cdaScatterLabels{vi}, 'FontSize', 8);
    if vi == 1
        ylabel({'Corrected behavior K difference', 'K6 - K1'});
    end
    box off;
    set(gca, 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 9);
end

sgtitle('CDA SS1-vs-SS6 metric sensitivity vs corrected K6 - K1', ...
    'FontName', 'Arial', 'FontWeight', 'bold');
savefig(fig, fullfile(figureDir, ...
    'data3_SS1vsSS6_CDA_metric_sensitivity_corrected_K6_minus_K1_scatter.fig'));
print(fig, fullfile(figureDir, ...
    'data3_SS1vsSS6_CDA_metric_sensitivity_corrected_K6_minus_K1_scatter.png'), '-dpng', '-r300');

fprintf('Saved decision-value behavior summaries to:\n%s\n', resultDir);
fprintf('Saved decision-value behavior figures to:\n%s\n', figureDir);

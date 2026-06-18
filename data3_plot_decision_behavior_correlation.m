%% Plot data3 decision-evidence behavior correlations
% Run after data3_decision_behavior_correlation.m.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
resultDir = fullfile(dataDir, 'decision_behavior_correlation');
figureDir = fullfile(resultDir, 'figures');

if ~isfolder(figureDir)
    mkdir(figureDir);
end

trialFile = fullfile(resultDir, 'data3_decision_behavior_trial_evidence.csv');
summaryFile = fullfile(resultDir, 'data3_decision_behavior_glme_summary.csv');

if ~isfile(trialFile)
    error('Missing trial evidence file: %s', trialFile);
end
if ~isfile(summaryFile)
    error('Missing GLME summary file: %s', summaryFile);
end

TrialEvidence = readtable(trialFile);
ModelSummary = readtable(summaryFile);

cfg = struct();
cfg.comparisons = {'setsize1_vs6_maintOnly', 'setsize3_vs6_maintOnly'};
cfg.latePositions = 4:6;
cfg.earlyPositions = 1:3;
cfg.positionsForSeparateLines = 4:6;
cfg.nEvidenceBinsLate = 3;
cfg.nEvidenceBinsAll = 5;
cfg.modelsForTrialPlots = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
cfg.modelsForForest = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
cfg.nEvidenceBins = cfg.nEvidenceBinsAll;
cfg.serialPositions = 1:6;
cfg.formulasForForest = {'base', 'withTrialIndex'};

compLabels = {'SS1 vs SS6', 'SS3 vs SS6'};
modelColors = lines(max(numel(cfg.modelsForForest), 3));
comparisonColors = [0.13 0.42 0.76; 0.74 0.28 0.20];

binSummaryTables = {};
serialSummaryTables = {};
lateBinSummaryTables = {};
lateSerialBinSummaryTables = {};

%% Within-position evidence z-score
% EvidenceZWithinPosition reduces serial-position confounding by measuring
% trial-to-trial evidence variation within the same subject and probed
% serial position.
TrialEvidence.EvidenceZWithinPosition = nan(height(TrialEvidence), 1);

subjectsAll = unique(TrialEvidence.Subject);
for si = 1:numel(subjectsAll)
    for ci = 1:numel(cfg.comparisons)
        for mi = 1:numel(cfg.modelsForTrialPlots)
            for pi = 1:numel(cfg.serialPositions)
                rows = TrialEvidence.Subject == subjectsAll(si) & ...
                    strcmp(TrialEvidence.Comparison, cfg.comparisons{ci}) & ...
                    strcmp(TrialEvidence.Model, cfg.modelsForTrialPlots{mi}) & ...
                    TrialEvidence.SerialPosition == cfg.serialPositions(pi) & ...
                    ~isnan(TrialEvidence.Evidence);
                if ~any(rows)
                    continue;
                end

                mu = mean(TrialEvidence.Evidence(rows), 'omitnan');
                sd = std(TrialEvidence.Evidence(rows), 0, 'omitnan');
                if sd == 0 || isnan(sd)
                    sd = 1;
                end
                TrialEvidence.EvidenceZWithinPosition(rows) = ...
                    (TrialEvidence.Evidence(rows) - mu) ./ sd;
            end
        end
    end
end

%% Original evidence bins: subject-level bin means, then group mean +/- SEM
% This all-position figure is descriptive and mixes serial positions 1-6.
% The late-position figures below are the main behavior-relevance plots for
% the current research question.
for mi = 1:numel(cfg.modelsForTrialPlots)
    modelName = cfg.modelsForTrialPlots{mi};

    fig = figure('Color', 'w', 'Position', [100 100 980 420]);
    for ci = 1:numel(cfg.comparisons)
        comparisonName = cfg.comparisons{ci};
        rows = strcmp(TrialEvidence.Comparison, comparisonName) & ...
            strcmp(TrialEvidence.Model, modelName) & ...
            ~isnan(TrialEvidence.EvidenceZ) & ...
            ~isnan(TrialEvidence.Correct);
        T = TrialEvidence(rows, :);

        subjects = unique(T.Subject);
        subjAcc = nan(numel(subjects), cfg.nEvidenceBins);
        subjEvidence = nan(numel(subjects), cfg.nEvidenceBins);

        for si = 1:numel(subjects)
            subjectRows = T.Subject == subjects(si);
            Ts = T(subjectRows, :);
            nTrial = height(Ts);
            if nTrial < cfg.nEvidenceBins
                continue;
            end

            [~, order] = sort(Ts.EvidenceZ);
            binEdges = round(linspace(0, nTrial, cfg.nEvidenceBins + 1));
            for bi = 1:cfg.nEvidenceBins
                binRows = order((binEdges(bi) + 1):binEdges(bi + 1));
                subjAcc(si,bi) = mean(Ts.Correct(binRows), 'omitnan');
                subjEvidence(si,bi) = mean(Ts.EvidenceZ(binRows), 'omitnan');

                S = table();
                S.Subject = subjects(si);
                S.Comparison = {comparisonName};
                S.Model = {modelName};
                S.EvidenceBin = bi;
                S.MeanEvidenceZ = subjEvidence(si,bi);
                S.Accuracy = subjAcc(si,bi);
                S.NTrial = numel(binRows);
                binSummaryTables{end+1,1} = S; %#ok<SAGROW>
            end
        end

        x = mean(subjEvidence, 1, 'omitnan');
        y = mean(subjAcc, 1, 'omitnan');
        semY = std(subjAcc, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(subjAcc), 1));

        subplot(1, numel(cfg.comparisons), ci);
        errorbar(x, y, semY, 'o-', ...
            'Color', comparisonColors(ci,:), ...
            'MarkerFaceColor', comparisonColors(ci,:), ...
            'LineWidth', 1.8, ...
            'MarkerSize', 6);
        hold on;
        plot([min(x)-0.2 max(x)+0.2], [0.5 0.5], 'k:', 'LineWidth', 1);
        xlabel('Decision evidence (z)');
        ylabel('Accuracy');
        title(sprintf('%s, %s', modelName, compLabels{ci}));
        ylim([0.45 1.00]);
        xlim([min(x)-0.25 max(x)+0.25]);
        box off;
        set(gca, 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 10);
    end

    savefig(fig, fullfile(figureDir, sprintf('data3_%s_evidence_bins_accuracy.fig', modelName)));
    print(fig, fullfile(figureDir, sprintf('data3_%s_evidence_bins_accuracy.png', modelName)), '-dpng', '-r300');
end

%% Late positions 4-6: pooled evidence-bin accuracy
% The late-position analysis focuses on serial positions 4-6 because these
% positions are the critical test of whether set-size-6 trials still express
% behaviorally relevant high-load neural evidence after the early loading
% period.
for mi = 1:numel(cfg.modelsForTrialPlots)
    modelName = cfg.modelsForTrialPlots{mi};

    fig = figure('Color', 'w', 'Position', [100 100 980 420]);
    for ci = 1:numel(cfg.comparisons)
        comparisonName = cfg.comparisons{ci};
        rows = strcmp(TrialEvidence.Comparison, comparisonName) & ...
            strcmp(TrialEvidence.Model, modelName) & ...
            ismember(TrialEvidence.SerialPosition, cfg.latePositions) & ...
            ~isnan(TrialEvidence.EvidenceZWithinPosition) & ...
            ~isnan(TrialEvidence.Correct);
        T = TrialEvidence(rows, :);

        subjects = unique(T.Subject);
        subjAcc = nan(numel(subjects), cfg.nEvidenceBinsLate);
        subjEvidence = nan(numel(subjects), cfg.nEvidenceBinsLate);

        for si = 1:numel(subjects)
            posAcc = nan(numel(cfg.latePositions), cfg.nEvidenceBinsLate);
            posEvidence = nan(numel(cfg.latePositions), cfg.nEvidenceBinsLate);
            posNTrial = nan(numel(cfg.latePositions), cfg.nEvidenceBinsLate);
            positionsUsed = [];

            for pi = 1:numel(cfg.latePositions)
                pos = cfg.latePositions(pi);
                posRows = T.Subject == subjects(si) & T.SerialPosition == pos;
                Ts = T(posRows, :);
                nTrial = height(Ts);
                if nTrial < cfg.nEvidenceBinsLate
                    continue;
                end

                [~, order] = sort(Ts.EvidenceZWithinPosition);
                binEdges = round(linspace(0, nTrial, cfg.nEvidenceBinsLate + 1));
                for bi = 1:cfg.nEvidenceBinsLate
                    binRows = order((binEdges(bi) + 1):binEdges(bi + 1));
                    posAcc(pi,bi) = mean(Ts.Correct(binRows), 'omitnan');
                    posEvidence(pi,bi) = mean(Ts.EvidenceZWithinPosition(binRows), 'omitnan');
                    posNTrial(pi,bi) = numel(binRows);
                end
                positionsUsed(end+1) = pos; %#ok<SAGROW>
            end

            if isempty(positionsUsed)
                continue;
            end

            subjAcc(si,:) = mean(posAcc, 1, 'omitnan');
            subjEvidence(si,:) = mean(posEvidence, 1, 'omitnan');
            positionsText = sprintf('%d,', positionsUsed);
            positionsText = positionsText(1:end-1);

            for bi = 1:cfg.nEvidenceBinsLate
                S = table();
                S.Subject = subjects(si);
                S.Comparison = {comparisonName};
                S.Model = {modelName};
                S.EvidenceBin = bi;
                S.MeanEvidenceZWithinPosition = subjEvidence(si,bi);
                S.Accuracy = subjAcc(si,bi);
                S.NTrial = sum(posNTrial(:,bi), 'omitnan');
                S.PositionsIncluded = {positionsText};
                lateBinSummaryTables{end+1,1} = S; %#ok<SAGROW>
            end
        end

        x = mean(subjEvidence, 1, 'omitnan');
        y = mean(subjAcc, 1, 'omitnan');
        semY = std(subjAcc, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(subjAcc), 1));

        subplot(1, numel(cfg.comparisons), ci);
        errorbar(x, y, semY, 'o-', ...
            'Color', comparisonColors(ci,:), ...
            'MarkerFaceColor', comparisonColors(ci,:), ...
            'LineWidth', 1.8, ...
            'MarkerSize', 6);
        hold on;
        plot([min(x)-0.2 max(x)+0.2], [0.5 0.5], 'k:', 'LineWidth', 1);
        xlabel('Within-position decision evidence (z)');
        ylabel('Accuracy');
        title(sprintf('%s, %s, positions 4-6', modelName, compLabels{ci}));
        ylim([0.45 1.00]);
        xlim([min(x)-0.25 max(x)+0.25]);
        box off;
        set(gca, 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 10);
    end

    savefig(fig, fullfile(figureDir, sprintf('data3_%s_late_positions_evidence_bins_accuracy.fig', modelName)));
    print(fig, fullfile(figureDir, sprintf('data3_%s_late_positions_evidence_bins_accuracy.png', modelName)), '-dpng', '-r300');
end

%% Late positions 4, 5, and 6: separate evidence-bin curves
for mi = 1:numel(cfg.modelsForTrialPlots)
    modelName = cfg.modelsForTrialPlots{mi};

    fig = figure('Color', 'w', 'Position', [100 100 980 420]);
    posColors = lines(numel(cfg.positionsForSeparateLines));

    for ci = 1:numel(cfg.comparisons)
        comparisonName = cfg.comparisons{ci};
        rows = strcmp(TrialEvidence.Comparison, comparisonName) & ...
            strcmp(TrialEvidence.Model, modelName) & ...
            ismember(TrialEvidence.SerialPosition, cfg.positionsForSeparateLines) & ...
            ~isnan(TrialEvidence.EvidenceZWithinPosition) & ...
            ~isnan(TrialEvidence.Correct);
        T = TrialEvidence(rows, :);

        subplot(1, numel(cfg.comparisons), ci);
        hold on;

        for pi = 1:numel(cfg.positionsForSeparateLines)
            pos = cfg.positionsForSeparateLines(pi);
            subjects = unique(T.Subject);
            subjAcc = nan(numel(subjects), cfg.nEvidenceBinsLate);
            subjEvidence = nan(numel(subjects), cfg.nEvidenceBinsLate);

            for si = 1:numel(subjects)
                posRows = T.Subject == subjects(si) & T.SerialPosition == pos;
                Ts = T(posRows, :);
                nTrial = height(Ts);
                if nTrial < cfg.nEvidenceBinsLate
                    continue;
                end

                [~, order] = sort(Ts.EvidenceZWithinPosition);
                binEdges = round(linspace(0, nTrial, cfg.nEvidenceBinsLate + 1));
                for bi = 1:cfg.nEvidenceBinsLate
                    binRows = order((binEdges(bi) + 1):binEdges(bi + 1));
                    subjAcc(si,bi) = mean(Ts.Correct(binRows), 'omitnan');
                    subjEvidence(si,bi) = mean(Ts.EvidenceZWithinPosition(binRows), 'omitnan');

                    S = table();
                    S.Subject = subjects(si);
                    S.Comparison = {comparisonName};
                    S.Model = {modelName};
                    S.SerialPosition = pos;
                    S.EvidenceBin = bi;
                    S.MeanEvidenceZWithinPosition = subjEvidence(si,bi);
                    S.Accuracy = subjAcc(si,bi);
                    S.NTrial = numel(binRows);
                    lateSerialBinSummaryTables{end+1,1} = S; %#ok<SAGROW>
                end
            end

            x = mean(subjEvidence, 1, 'omitnan');
            y = mean(subjAcc, 1, 'omitnan');
            semY = std(subjAcc, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(subjAcc), 1));
            errorbar(x, y, semY, 'o-', ...
                'Color', posColors(pi,:), ...
                'MarkerFaceColor', posColors(pi,:), ...
                'LineWidth', 1.6, ...
                'MarkerSize', 5);
        end

        plot([-1.8 1.8], [0.5 0.5], 'k:', 'LineWidth', 1);
        xlabel('Within-position decision evidence (z)');
        ylabel('Accuracy');
        title(sprintf('%s, %s', modelName, compLabels{ci}));
        legendLabels = cell(1, numel(cfg.positionsForSeparateLines));
        for li = 1:numel(cfg.positionsForSeparateLines)
            legendLabels{li} = sprintf('Position %d', cfg.positionsForSeparateLines(li));
        end
        legend(legendLabels, 'Location', 'best', 'Box', 'off');
        ylim([0.45 1.00]);
        xlim([-1.8 1.8]);
        box off;
        set(gca, 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 10);
    end

    savefig(fig, fullfile(figureDir, sprintf('data3_%s_late_positions_separate_evidence_bins_accuracy.fig', modelName)));
    print(fig, fullfile(figureDir, sprintf('data3_%s_late_positions_separate_evidence_bins_accuracy.png', modelName)), '-dpng', '-r300');
end

%% Serial position: accuracy and decision evidence for probed set-size-6 items
for mi = 1:numel(cfg.modelsForTrialPlots)
    modelName = cfg.modelsForTrialPlots{mi};

    fig = figure('Color', 'w', 'Position', [100 100 980 720]);
    for ci = 1:numel(cfg.comparisons)
        comparisonName = cfg.comparisons{ci};
        rows = strcmp(TrialEvidence.Comparison, comparisonName) & ...
            strcmp(TrialEvidence.Model, modelName) & ...
            ~isnan(TrialEvidence.EvidenceZ) & ...
            ~isnan(TrialEvidence.Correct);
        T = TrialEvidence(rows, :);

        subjects = unique(T.Subject);
        subjAcc = nan(numel(subjects), numel(cfg.serialPositions));
        subjEvidence = nan(numel(subjects), numel(cfg.serialPositions));

        for si = 1:numel(subjects)
            for pi = 1:numel(cfg.serialPositions)
                pos = cfg.serialPositions(pi);
                posRows = T.Subject == subjects(si) & T.SerialPosition == pos;
                if ~any(posRows)
                    continue;
                end
                subjAcc(si,pi) = mean(T.Correct(posRows), 'omitnan');
                subjEvidence(si,pi) = mean(T.EvidenceZ(posRows), 'omitnan');

                S = table();
                S.Subject = subjects(si);
                S.Comparison = {comparisonName};
                S.Model = {modelName};
                S.SerialPosition = pos;
                S.MeanEvidenceZ = subjEvidence(si,pi);
                S.Accuracy = subjAcc(si,pi);
                S.NTrial = sum(posRows);
                serialSummaryTables{end+1,1} = S; %#ok<SAGROW>
            end
        end

        accMean = mean(subjAcc, 1, 'omitnan');
        accSem = std(subjAcc, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(subjAcc), 1));
        evidenceMean = mean(subjEvidence, 1, 'omitnan');
        evidenceSem = std(subjEvidence, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(subjEvidence), 1));

        subplot(2, numel(cfg.comparisons), ci);
        errorbar(cfg.serialPositions, accMean, accSem, 'o-', ...
            'Color', comparisonColors(ci,:), ...
            'MarkerFaceColor', comparisonColors(ci,:), ...
            'LineWidth', 1.8, ...
            'MarkerSize', 6);
        hold on;
        plot([cfg.serialPositions(1) cfg.serialPositions(end)], [0.5 0.5], 'k:', 'LineWidth', 1);
        xlabel('Probed serial position');
        ylabel('Accuracy');
        title(sprintf('%s, %s', modelName, compLabels{ci}));
        xlim([0.5 6.5]);
        ylim([0.45 1.00]);
        set(gca, 'XTick', cfg.serialPositions, 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 10);
        box off;

        subplot(2, numel(cfg.comparisons), ci + numel(cfg.comparisons));
        errorbar(cfg.serialPositions, evidenceMean, evidenceSem, 'o-', ...
            'Color', comparisonColors(ci,:), ...
            'MarkerFaceColor', comparisonColors(ci,:), ...
            'LineWidth', 1.8, ...
            'MarkerSize', 6);
        hold on;
        plot([cfg.serialPositions(1) cfg.serialPositions(end)], [0 0], 'k:', 'LineWidth', 1);
        xlabel('Probed serial position');
        ylabel('Decision evidence (z)');
        xlim([0.5 6.5]);
        set(gca, 'XTick', cfg.serialPositions, 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 10);
        box off;
    end

    savefig(fig, fullfile(figureDir, sprintf('data3_%s_serial_position_accuracy_evidence.fig', modelName)));
    print(fig, fullfile(figureDir, sprintf('data3_%s_serial_position_accuracy_evidence.png', modelName)), '-dpng', '-r300');
end

%% GLME evidence coefficient forest plots
fig = figure('Color', 'w', 'Position', [100 100 1120 620]);

for fi = 1:numel(cfg.formulasForForest)
    formulaName = cfg.formulasForForest{fi};

    rows = strcmp(ModelSummary.Formula, formulaName) & strcmp(ModelSummary.Term, 'Evidence');
    S = ModelSummary(rows, :);

    orderedRows = [];
    for ci = 1:numel(cfg.comparisons)
        for mi = 1:numel(cfg.modelsForForest)
            rowNow = find(strcmp(S.Comparison, cfg.comparisons{ci}) & strcmp(S.Model, cfg.modelsForForest{mi}), 1);
            if ~isempty(rowNow)
                orderedRows(end+1,1) = rowNow; %#ok<SAGROW>
            end
        end
    end
    S = S(orderedRows, :);

    subplot(1, numel(cfg.formulasForForest), fi);
    hold on;
    plot([0 0], [0 height(S)+1], 'k--', 'LineWidth', 1);

    yTickLabels = cell(height(S), 1);
    for ri = 1:height(S)
        estimate = S.Estimate(ri);
        ciLow = estimate - 1.96 * S.SE(ri);
        ciHigh = estimate + 1.96 * S.SE(ri);

        compIdx = find(strcmp(cfg.comparisons, S.Comparison{ri}), 1);
        modelIdx = find(strcmp(cfg.modelsForForest, S.Model{ri}), 1);
        if isempty(compIdx), compIdx = 1; end
        if isempty(modelIdx), modelIdx = 1; end

        y = height(S) - ri + 1;
        plot([ciLow ciHigh], [y y], '-', 'Color', comparisonColors(compIdx,:), 'LineWidth', 1.8);
        plot(estimate, y, 'o', ...
            'Color', comparisonColors(compIdx,:), ...
            'MarkerFaceColor', modelColors(modelIdx,:), ...
            'MarkerSize', 6, ...
            'LineWidth', 1.2);

        pVal = S.pValue(ri);
        if pVal < 0.001
            pText = '***';
        elseif pVal < 0.01
            pText = '**';
        elseif pVal < 0.05
            pText = '*';
        else
            pText = sprintf('p=%.3f', pVal);
        end
        text(ciHigh + 0.03, y, pText, 'FontSize', 8, 'FontName', 'Arial');

        if contains(S.Comparison{ri}, 'setsize1')
            compText = 'SS1 vs SS6';
        else
            compText = 'SS3 vs SS6';
        end
        yTickLabels{y} = sprintf('%s  %s', compText, S.Model{ri});
    end

    xlabel('Evidence coefficient (log odds)');
    title(sprintf('GLME: %s', formulaName), 'Interpreter', 'none');
    set(gca, 'YTick', 1:height(S), 'YTickLabel', yTickLabels, ...
        'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 9);
    ylim([0 height(S)+1]);
    box off;
end

savefig(fig, fullfile(figureDir, 'data3_glme_evidence_coefficient_forest.fig'));
print(fig, fullfile(figureDir, 'data3_glme_evidence_coefficient_forest.png'), '-dpng', '-r300');

%% Late-position GLME statistics
lateGlmeSummaryTables = {};
lateInteractionSummaryTables = {};

for ci = 1:numel(cfg.comparisons)
    comparisonName = cfg.comparisons{ci};

    for mi = 1:numel(cfg.modelsForForest)
        modelName = cfg.modelsForForest{mi};

        lateRows = strcmp(TrialEvidence.Comparison, comparisonName) & ...
            strcmp(TrialEvidence.Model, modelName) & ...
            ismember(TrialEvidence.SerialPosition, cfg.latePositions) & ...
            ~isnan(TrialEvidence.EvidenceZWithinPosition) & ...
            ~isnan(TrialEvidence.Correct) & ...
            ~isnan(TrialEvidence.TrialIndex);
        lateModelTable = TrialEvidence(lateRows, :);

        lateModelTable.Subject = categorical(lateModelTable.Subject);
        lateModelTable.Correct = double(lateModelTable.Correct);
        lateModelTable.SerialPositionCat = categorical(lateModelTable.SerialPosition);
        lateModelTable.TrialIndex = double(lateModelTable.TrialIndex);

        lateFormula = 'Correct ~ EvidenceZWithinPosition + SerialPositionCat + TrialIndex + (1|Subject)';
        lateGlme = fitglme(lateModelTable, lateFormula, ...
            'Distribution', 'Binomial', ...
            'Link', 'logit', ...
            'FitMethod', 'Laplace');

        coef = lateGlme.Coefficients;
        nCoef = height(coef);
        S = table();
        S.Comparison = repmat({comparisonName}, nCoef, 1);
        S.Model = repmat({modelName}, nCoef, 1);
        S.Formula = repmat({'lateWithTrialIndex'}, nCoef, 1);
        S.NTrial = repmat(height(lateModelTable), nCoef, 1);
        S.NSubject = repmat(numel(unique(lateModelTable.Subject)), nCoef, 1);
        S.Term = coef.Name;
        S.Estimate = coef.Estimate;
        S.SE = coef.SE;
        S.tStat = coef.tStat;
        S.pValue = coef.pValue;
        lateGlmeSummaryTables{end+1,1} = S; %#ok<SAGROW>

        allRows = strcmp(TrialEvidence.Comparison, comparisonName) & ...
            strcmp(TrialEvidence.Model, modelName) & ...
            ~isnan(TrialEvidence.EvidenceZWithinPosition) & ...
            ~isnan(TrialEvidence.Correct) & ...
            ~isnan(TrialEvidence.SerialPosition) & ...
            ~isnan(TrialEvidence.TrialIndex);
        allModelTable = TrialEvidence(allRows, :);
        allModelTable.Subject = categorical(allModelTable.Subject);
        allModelTable.Correct = double(allModelTable.Correct);
        allModelTable.SerialPosition = double(allModelTable.SerialPosition);
        allModelTable.TrialIndex = double(allModelTable.TrialIndex);
        allModelTable.LatePosition = double(allModelTable.SerialPosition >= min(cfg.latePositions));

        interactionFormula = 'Correct ~ EvidenceZWithinPosition * LatePosition + SerialPosition + TrialIndex + (1|Subject)';
        interactionGlme = fitglme(allModelTable, interactionFormula, ...
            'Distribution', 'Binomial', ...
            'Link', 'logit', ...
            'FitMethod', 'Laplace');

        coef = interactionGlme.Coefficients;
        nCoef = height(coef);
        S = table();
        S.Comparison = repmat({comparisonName}, nCoef, 1);
        S.Model = repmat({modelName}, nCoef, 1);
        S.Formula = repmat({'allPositionLateInteraction'}, nCoef, 1);
        S.NTrial = repmat(height(allModelTable), nCoef, 1);
        S.NSubject = repmat(numel(unique(allModelTable.Subject)), nCoef, 1);
        S.Term = coef.Name;
        S.Estimate = coef.Estimate;
        S.SE = coef.SE;
        S.tStat = coef.tStat;
        S.pValue = coef.pValue;
        lateInteractionSummaryTables{end+1,1} = S; %#ok<SAGROW>
    end
end

if isempty(lateGlmeSummaryTables)
    LateGlmeSummary = table();
else
    LateGlmeSummary = vertcat(lateGlmeSummaryTables{:});
end

if isempty(lateInteractionSummaryTables)
    LateInteractionGlmeSummary = table();
else
    LateInteractionGlmeSummary = vertcat(lateInteractionSummaryTables{:});
end

writetable(LateGlmeSummary, fullfile(resultDir, 'data3_decision_behavior_late_glme_summary.csv'));
writetable(LateInteractionGlmeSummary, fullfile(resultDir, 'data3_decision_behavior_late_interaction_glme_summary.csv'));

%% Late-position GLME evidence coefficient forest plot
fig = figure('Color', 'w', 'Position', [100 100 960 620]);

rows = strcmp(LateGlmeSummary.Term, 'EvidenceZWithinPosition');
S = LateGlmeSummary(rows, :);

orderedRows = [];
for ci = 1:numel(cfg.comparisons)
    for mi = 1:numel(cfg.modelsForForest)
        rowNow = find(strcmp(S.Comparison, cfg.comparisons{ci}) & strcmp(S.Model, cfg.modelsForForest{mi}), 1);
        if ~isempty(rowNow)
            orderedRows(end+1,1) = rowNow; %#ok<SAGROW>
        end
    end
end
S = S(orderedRows, :);

hold on;
plot([0 0], [0 height(S)+1], 'k--', 'LineWidth', 1);

yTickLabels = cell(height(S), 1);
ciLowAll = S.Estimate - 1.96 * S.SE;
ciHighAll = S.Estimate + 1.96 * S.SE;

for ri = 1:height(S)
    estimate = S.Estimate(ri);
    ciLow = estimate - 1.96 * S.SE(ri);
    ciHigh = estimate + 1.96 * S.SE(ri);

    compIdx = find(strcmp(cfg.comparisons, S.Comparison{ri}), 1);
    modelIdx = find(strcmp(cfg.modelsForForest, S.Model{ri}), 1);
    if isempty(compIdx), compIdx = 1; end
    if isempty(modelIdx), modelIdx = 1; end

    y = height(S) - ri + 1;
    plot([ciLow ciHigh], [y y], '-', 'Color', comparisonColors(compIdx,:), 'LineWidth', 1.8);
    plot(estimate, y, 'o', ...
        'Color', comparisonColors(compIdx,:), ...
        'MarkerFaceColor', modelColors(modelIdx,:), ...
        'MarkerSize', 6, ...
        'LineWidth', 1.2);

    pVal = S.pValue(ri);
    if pVal < 0.001
        pText = '***';
    elseif pVal < 0.01
        pText = '**';
    elseif pVal < 0.05
        pText = '*';
    else
        pText = sprintf('p=%.3f', pVal);
    end
    text(ciHigh + 0.03, y, pText, 'FontSize', 8, 'FontName', 'Arial');

    if contains(S.Comparison{ri}, 'setsize1')
        compText = 'SS1 vs SS6';
    else
        compText = 'SS3 vs SS6';
    end
    yTickLabels{y} = sprintf('%s  %s', compText, S.Model{ri});
end

xlabel('Evidence coefficient (log odds)');
title('Late positions 4-6 GLME');
set(gca, 'YTick', 1:height(S), 'YTickLabel', yTickLabels, ...
    'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 9);
set(gca, 'Position', [0.32 0.12 0.55 0.78]);
ylim([0 height(S)+1]);
xlim([min(ciLowAll)-0.2 max(ciHighAll)+0.5]);
box off;

savefig(fig, fullfile(figureDir, 'data3_late_glme_evidence_coefficient_forest.fig'));
print(fig, fullfile(figureDir, 'data3_late_glme_evidence_coefficient_forest.png'), '-dpng', '-r300');

if isempty(binSummaryTables)
    BinSubjectSummary = table();
else
    BinSubjectSummary = vertcat(binSummaryTables{:});
end

if isempty(serialSummaryTables)
    SerialSubjectSummary = table();
else
    SerialSubjectSummary = vertcat(serialSummaryTables{:});
end

if isempty(lateBinSummaryTables)
    LateBinSubjectSummary = table();
else
    LateBinSubjectSummary = vertcat(lateBinSummaryTables{:});
end

if isempty(lateSerialBinSummaryTables)
    LateSerialPositionBinSubjectSummary = table();
else
    LateSerialPositionBinSubjectSummary = vertcat(lateSerialBinSummaryTables{:});
end

writetable(BinSubjectSummary, fullfile(resultDir, 'data3_decision_behavior_evidence_bin_subject_summary.csv'));
writetable(SerialSubjectSummary, fullfile(resultDir, 'data3_decision_behavior_serial_position_subject_summary.csv'));
writetable(LateBinSubjectSummary, fullfile(resultDir, 'data3_decision_behavior_late_evidence_bin_subject_summary.csv'));
writetable(LateSerialPositionBinSubjectSummary, fullfile(resultDir, 'data3_decision_behavior_late_serial_position_bin_subject_summary.csv'));
save(fullfile(resultDir, 'data3_decision_behavior_plot_summaries.mat'), ...
    'cfg', 'BinSubjectSummary', 'SerialSubjectSummary', ...
    'LateBinSubjectSummary', 'LateSerialPositionBinSubjectSummary', ...
    'LateGlmeSummary', 'LateInteractionGlmeSummary', '-v7.3');

fprintf('Saved figures to:\n%s\n', figureDir);

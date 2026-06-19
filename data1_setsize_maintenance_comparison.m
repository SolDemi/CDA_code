%% Compare data1 CDA decoding across set-size pairs during maintenance
% The subject-level value is the mean diagonal AUC from 250 ms to the end.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);

dataDir = fullfile(projectRoot, 'data1');
decodingRoot = fullfile(dataDir, 'decoding_LDA_spatialControl');
outputDir = fullfile(decodingRoot, 'setsizePairMaintenanceComparison');
if ~isfolder(outputDir), mkdir(outputDir); end

cfg = struct();
cfg.metric = 'AUC';
cfg.chance = 0.5;
cfg.maintenanceWindowMs = [250 inf];
cfg.alpha = 0.05;
cfg.multCompMethod = 'holm';
cfg.filePattern = '*.mat';
cfg.useDiagonal = true;
cfg.excludeSubjectIds = [];
[cfg.includeSubjectIds, subjectInclusion] = data1_decoding_subjects( ...
    fullfile(dataDir, 'data'), 75);

comparisonNames = {'SS1vsSS3', 'SS1vsSS6', 'SS3vsSS6'};
comparisonLabels = {'SS1 vs SS3', 'SS1 vs SS6', 'SS3 vs SS6'};
comparisonDirs = { ...
    fullfile(decodingRoot, 'loadWithinSide_setsize1_vs3', 'CDA'), ...
    fullfile(decodingRoot, 'loadWithinSide', 'CDA'), ...
    fullfile(decodingRoot, 'loadWithinSide_setsize3_vs6', 'CDA')};

%% Load diagonal AUC and average the maintenance period for each subject
nComparison = numel(comparisonNames);
maintenanceMean = cell(1, nComparison);
subjectFiles = cell(1, nComparison);
commonTimes = [];

for ci = 1:nComparison
    loadCfg = cfg;
    [diagAUC, times, usedFiles] = extract_decoding_timeseries(comparisonDirs{ci}, loadCfg);

    if isempty(commonTimes)
        commonTimes = times(:)';
    elseif numel(times) ~= numel(commonTimes) || any(abs(times(:)' - commonTimes) > 1e-9)
        error('Time axes differ across set-size comparisons.');
    end

    maintenanceIdx = times >= cfg.maintenanceWindowMs(1);
    if isfinite(cfg.maintenanceWindowMs(2))
        maintenanceIdx = maintenanceIdx & times <= cfg.maintenanceWindowMs(2);
    end
    if ~any(maintenanceIdx)
        error('The maintenance window does not overlap the decoding time axis.');
    end

    maintenanceMean{ci} = mean(diagAUC(:, maintenanceIdx), 2, 'omitnan');
    subjectFiles{ci} = cell(size(usedFiles));
    for si = 1:numel(usedFiles)
        [~, fileName, fileExt] = fileparts(usedFiles{si});
        subjectFiles{ci}{si} = [fileName fileExt];
    end

    fprintf('%s: loaded %d subjects, maintenance window = %g to %g ms.\n', ...
        comparisonLabels{ci}, size(diagAUC, 1), times(find(maintenanceIdx, 1, 'first')), ...
        times(find(maintenanceIdx, 1, 'last')));
end

%% Retain the same subjects in all three comparisons
commonSubjectFiles = subjectFiles{1};
for ci = 2:nComparison
    commonSubjectFiles = intersect(commonSubjectFiles, subjectFiles{ci}, 'stable');
end
if numel(commonSubjectFiles) < 2
    error('Fewer than two subjects are shared by all three comparisons.');
end

subjectData = nan(numel(commonSubjectFiles), nComparison);
for ci = 1:nComparison
    [found, rowIdx] = ismember(commonSubjectFiles, subjectFiles{ci});
    if ~all(found)
        error('Subject alignment failed for %s.', comparisonLabels{ci});
    end
    subjectData(:, ci) = maintenanceMean{ci}(rowIdx);
end

keepSubject = all(isfinite(subjectData), 2);
commonSubjectFiles = commonSubjectFiles(keepSubject);
subjectData = subjectData(keepSubject, :);
nSubject = size(subjectData, 1);
if nSubject < 2
    error('Fewer than two subjects have finite values in all three comparisons.');
end

subjectTable = table(commonSubjectFiles(:), subjectData(:,1), subjectData(:,2), subjectData(:,3), ...
    'VariableNames', {'SubjectFile', 'SS1vsSS3', 'SS1vsSS6', 'SS3vsSS6'});

%% Test each comparison against chance
chanceP = nan(nComparison, 1);
chanceT = nan(nComparison, 1);
chanceDF = nan(nComparison, 1);
chanceDz = nan(nComparison, 1);
comparisonMean = mean(subjectData, 1, 'omitnan');
comparisonSEM = std(subjectData, 0, 1, 'omitnan') ./ sqrt(nSubject);

for ci = 1:nComparison
    [~, chanceP(ci), ~, chanceTest] = ttest(subjectData(:,ci), cfg.chance, 'Tail', 'right');
    chanceT(ci) = chanceTest.tstat;
    chanceDF(ci) = chanceTest.df;
    chanceDz(ci) = (comparisonMean(ci) - cfg.chance) ./ std(subjectData(:,ci), 0, 'omitnan');
end
[chancePHolm, chanceSignificant] = correct_pvalues(chanceP, cfg.multCompMethod, cfg.alpha);

chanceStats = table(comparisonNames(:), comparisonLabels(:), repmat(nSubject, nComparison, 1), ...
    comparisonMean(:), comparisonSEM(:), chanceT, chanceDF, chanceP, chancePHolm, ...
    chanceSignificant, chanceDz, ...
    'VariableNames', {'Comparison', 'Label', 'N', 'MeanAUC', 'SEMAUC', 'T', 'DF', ...
    'P_right_vsChance', 'P_right_Holm', 'Significant_Holm', 'CohenDz'});

%% Paired comparisons between the three set-size contrasts
pairList = [1 2; 1 3; 2 3];
nPair = size(pairList, 1);
pairName = cell(nPair, 1);
firstComparison = cell(nPair, 1);
secondComparison = cell(nPair, 1);
pairMeanDiff = nan(nPair, 1);
pairSEMDiff = nan(nPair, 1);
pairT = nan(nPair, 1);
pairDF = nan(nPair, 1);
pairP = nan(nPair, 1);
pairCILow = nan(nPair, 1);
pairCIHigh = nan(nPair, 1);
pairDz = nan(nPair, 1);

for pi = 1:nPair
    firstIdx = pairList(pi, 1);
    secondIdx = pairList(pi, 2);
    difference = subjectData(:, secondIdx) - subjectData(:, firstIdx);
    [~, pairP(pi), pairCI, pairTest] = ttest(subjectData(:,secondIdx), subjectData(:,firstIdx), ...
        'Tail', 'both');

    pairName{pi} = sprintf('%s_minus_%s', comparisonNames{secondIdx}, comparisonNames{firstIdx});
    firstComparison{pi} = comparisonNames{firstIdx};
    secondComparison{pi} = comparisonNames{secondIdx};
    pairMeanDiff(pi) = mean(difference, 'omitnan');
    pairSEMDiff(pi) = std(difference, 0, 'omitnan') ./ sqrt(nSubject);
    pairT(pi) = pairTest.tstat;
    pairDF(pi) = pairTest.df;
    pairCILow(pi) = pairCI(1);
    pairCIHigh(pi) = pairCI(2);
    pairDz(pi) = pairMeanDiff(pi) ./ std(difference, 0, 'omitnan');
end
[pairPHolm, pairSignificant] = correct_pvalues(pairP, cfg.multCompMethod, cfg.alpha);

pairwiseStats = table(pairName, firstComparison, secondComparison, repmat(nSubject, nPair, 1), ...
    pairMeanDiff, pairSEMDiff, pairT, pairDF, pairP, pairPHolm, pairSignificant, ...
    pairCILow, pairCIHigh, pairDz, ...
    'VariableNames', {'Contrast', 'FirstComparison', 'SecondComparison', 'N', ...
    'MeanDiff_SecondMinusFirst', 'SEMDiff', 'T', 'DF', 'P_two', 'P_two_Holm', ...
    'Significant_Holm', 'CI95Low', 'CI95High', 'CohenDz'});

%% Plot all three comparisons in one figure
plotColors = [0.20 0.55 0.80; 0.90 0.45 0.20; 0.35 0.70 0.40];
fig = figure('Color', 'w', 'Position', [120 100 780 620]);
ax = axes('Parent', fig);
hold(ax, 'on');

for si = 1:nSubject
    plot(ax, 1:nComparison, subjectData(si,:), '-', 'Color', [0.80 0.80 0.80], ...
        'LineWidth', 0.7, 'HandleVisibility', 'off');
end

b = bar(ax, 1:nComparison, comparisonMean, 0.58, 'FaceColor', 'flat', ...
    'EdgeColor', 'none');
b.CData = plotColors;
errorbar(ax, 1:nComparison, comparisonMean, comparisonSEM, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1.3, 'CapSize', 12, 'HandleVisibility', 'off');

jitter = linspace(-0.09, 0.09, nSubject)';
for ci = 1:nComparison
    scatter(ax, ci + jitter, subjectData(:,ci), 28, plotColors(ci,:), 'filled', ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.4, 'HandleVisibility', 'off');
end
yline(ax, cfg.chance, ':k', 'Chance', 'LineWidth', 1.1, 'HandleVisibility', 'off');

yMin = min([subjectData(:); cfg.chance], [], 'omitnan');
yMax = max([subjectData(:); (comparisonMean + comparisonSEM)'], [], 'omitnan');
yRange = yMax - yMin;
if yRange <= 0, yRange = 0.02; end
bracketBase = yMax + 0.10 * yRange;
bracketStep = 0.15 * yRange;
bracketHeight = 0.035 * yRange;

for pi = 1:nPair
    x1 = pairList(pi, 1);
    x2 = pairList(pi, 2);
    bracketY = bracketBase + (pi - 1) * bracketStep;
    plot(ax, [x1 x1 x2 x2], [bracketY bracketY+bracketHeight bracketY+bracketHeight bracketY], ...
        'k-', 'LineWidth', 1.1, 'HandleVisibility', 'off');

    if pairPHolm(pi) < 0.001
        pLabel = '***';
    elseif pairPHolm(pi) < 0.01
        pLabel = '**';
    elseif pairPHolm(pi) < 0.05
        pLabel = '*';
    else
        pLabel = 'n.s.';
    end
    text(ax, mean([x1 x2]), bracketY + bracketHeight + 0.01 * yRange, ...
        sprintf('%s  p_{Holm}=%.3g', pLabel, pairPHolm(pi)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
end

ylim(ax, [yMin - 0.10 * yRange, bracketBase + (nPair - 1) * bracketStep + 0.18 * yRange]);
xlim(ax, [0.45 nComparison + 0.55]);
set(ax, 'XTick', 1:nComparison, 'XTickLabel', comparisonLabels, ...
    'TickDir', 'out', 'FontSize', 11);
ylabel(ax, sprintf('Mean maintenance diagonal %s', cfg.metric));
title(ax, sprintf('data1 CDA set-size decoding, %g ms to end (N=%d)', ...
    cfg.maintenanceWindowMs(1), nSubject));
box(ax, 'off');
grid(ax, 'on');

figureBaseName = sprintf('%s_data1_CDA_three_setsize_pairs_maintenance_mean', cfg.metric);
savefig(fig, fullfile(outputDir, [figureBaseName '.fig']));
print(fig, fullfile(outputDir, [figureBaseName '.png']), '-dpng', '-r300');
close(fig);

%% Save subject values and statistics
writetable(subjectTable, fullfile(outputDir, [figureBaseName '_subjects.csv']));
writetable(chanceStats, fullfile(outputDir, [figureBaseName '_chance_stats.csv']));
writetable(pairwiseStats, fullfile(outputDir, [figureBaseName '_pairwise_stats.csv']));
save(fullfile(outputDir, [figureBaseName '_stats.mat']), 'cfg', 'subjectInclusion', ...
    'subjectTable', 'chanceStats', 'pairwiseStats', 'commonTimes', '-v7.3');

disp(chanceStats);
disp(pairwiseStats);
fprintf('Figure and statistics saved to:\n%s\n', outputDir);

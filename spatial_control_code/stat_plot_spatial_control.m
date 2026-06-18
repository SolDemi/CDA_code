%% stat_plot_spatial_control.m
% Group-level stats and figures for process_spatial_control_decoding.m
% Uses the same logic as stat_plot.m:
% extract_decoding_timeseries -> cluster_perm_1d_timeseries -> plot.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
mainCodeDir = fileparts(codeDir);
projectRoot = fileparts(mainCodeDir);
addpath(mainCodeDir);
addpath(codeDir);

maindir = fullfile(projectRoot, 'data0');
rootdir = fullfile(maindir, 'decoding_LDA_spatialControl');
figdir = fullfile(rootdir, 'group_figures');
if ~isfolder(figdir), mkdir(figdir); end

metricCfg = struct();
metricCfg.metric = 'AccMinusShuffle';
metricCfg.useDiagonal = true;
[metricCfg.includeSubjectIds, subjectInclusion] = data0_decoding_subjects( ...
    fullfile(maindir, 'data'), 75);
metricCfg.excludeSubjectIds = [];
fprintf('Subject inclusion for data0 spatial-control stats: n = %d\n', ...
    numel(metricCfg.includeSubjectIds));

statCfg = struct();
statCfg.null = 0;
statCfg.nPerm = 1000;
statCfg.tail = 'right';
statCfg.clusterAlpha = 0.05;
statCfg.alpha = 0.05;
statCfg.clusterStat = 'mass';
statCfg.minClusterSize = 1;
statCfg.randomSeed = 1;
statCfg.verbose = true;

plotCfg = struct();
plotCfg.ylabel = 'Acc - shuffle';
plotCfg.eventLines = [0 250];
plotCfg.eventLineLabels = {'stim on', 'stim off'};
plotCfg.xlim = [];
plotCfg.ylim = [];

sideFeatures = {'VoltageRawLR', 'AlphaRawLR', 'VoltageLminusR', 'AlphaLminusR', 'GlobalAlphaMean'};
loadFeatures = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};

allStats = struct();
analysisList = {'sideDecoding', 'loadWithinSide', 'loadSideBalanced', 'loadCrossSide'};
featureList = {sideFeatures, loadFeatures, loadFeatures, loadFeatures};

for analysisIdx = 1:numel(analysisList)
    analysisName = analysisList{analysisIdx};
    features = featureList{analysisIdx};
    nFeat = numel(features);
    nCol = 3;
    nRow = ceil(nFeat / nCol);
    statsOut = struct();

    figure('Color', 'w', 'Position', [80 80 420*nCol 300*nRow]);
    tiledlayout(nRow, nCol, 'TileSpacing', 'compact', 'Padding', 'compact');

    for fi = 1:nFeat
        featName = features{fi};
        resultDir = fullfile(rootdir, analysisName, featName);

        nexttile; hold on;
        if ~isfolder(resultDir) || isempty(dir(fullfile(resultDir, '*.mat')))
            title([featName, ' (no files)']); box off;
            continue;
        end

        [dataMat, times, usedFiles] = extract_decoding_timeseries(resultDir, metricCfg);
        stat = cluster_perm_1d_timeseries(dataMat, times, statCfg);
        statsOut.(featName).stat = stat;
        statsOut.(featName).files = usedFiles;

        x = stat.times;
        y = stat.mean;
        e = stat.sem;

        patch([x, fliplr(x)], [y+e, fliplr(y-e)], [0.5 0.5 0.5], ...
            'FaceAlpha', 0.2, 'EdgeColor', 'none');
        plot(x, y, 'k', 'LineWidth', 2);
        yline(0, ':k');

        for li = 1:numel(plotCfg.eventLines)
            xline(plotCfg.eventLines(li), '--k');
        end

        yl = ylim;
        yBar = yl(1) + 0.05 * range(yl);
        for ci = 1:numel(stat.significantClusters)
            idx = stat.significantClusters(ci).idx;
            plot(x(idx), yBar * ones(size(idx)), 'k-', 'LineWidth', 4);
        end

        title(featName, 'Interpreter', 'none');
        xlabel('Time (ms)');
        ylabel(plotCfg.ylabel);
        if ~isempty(plotCfg.xlim), xlim(plotCfg.xlim); end
        if ~isempty(plotCfg.ylim), ylim(plotCfg.ylim); end
        box off; grid on;
    end

    sgtitle(analysisName, 'Interpreter', 'none');
    saveas(gcf, fullfile(figdir, [analysisName, '_', metricCfg.metric, '.png']));
    savefig(gcf, fullfile(figdir, [analysisName, '_', metricCfg.metric, '.fig']));
    allStats.(analysisName) = statsOut;
end

%% Maintenance-period summary: side evidence vs controlled load evidence
summaryCfg = struct();
summaryCfg.timeWin = [250 inf];   % edit this to your preferred maintenance window
summaryCfg.sideAnalysis = 'sideDecoding';
summaryCfg.loadAnalysis = 'loadSideBalanced';
summaryCfg.metricCfg = metricCfg;

summaryMap = struct();
summaryMap(1).label = 'CDA / voltage lateralization';
summaryMap(1).sideFeature = 'VoltageLminusR';
summaryMap(1).loadFeature = 'CDA';
summaryMap(2).label = 'Lateralized alpha';
summaryMap(2).sideFeature = 'AlphaLminusR';
summaryMap(2).loadFeature = 'Alpha';
summaryMap(3).label = 'Global alpha mean';
summaryMap(3).sideFeature = 'GlobalAlphaMean';
summaryMap(3).loadFeature = 'GlobalAlphaMean';
summaryMap(4).label = 'Global alpha topography';
summaryMap(4).sideFeature = 'AlphaRawLR';
summaryMap(4).loadFeature = 'GlobalAlpha';

n = numel(summaryMap);
labels = strings(n,1);
sideMean = nan(n,1);
loadMean = nan(n,1);
sideSEM = nan(n,1);
loadSEM = nan(n,1);

for i = 1:n
    labels(i) = string(summaryMap(i).label);

    sideDir = fullfile(rootdir, summaryCfg.sideAnalysis, summaryMap(i).sideFeature);
    loadDir = fullfile(rootdir, summaryCfg.loadAnalysis, summaryMap(i).loadFeature);

    [sideData, sideTimes] = extract_decoding_timeseries(sideDir, summaryCfg.metricCfg);
    [loadData, loadTimes] = extract_decoding_timeseries(loadDir, summaryCfg.metricCfg);

    sideIdx = sideTimes >= summaryCfg.timeWin(1) & sideTimes <= summaryCfg.timeWin(2);
    loadIdx = loadTimes >= summaryCfg.timeWin(1) & loadTimes <= summaryCfg.timeWin(2);

    sideSubj = mean(sideData(:, sideIdx), 2, 'omitnan');
    loadSubj = mean(loadData(:, loadIdx), 2, 'omitnan');

    sideMean(i) = mean(sideSubj, 'omitnan');
    loadMean(i) = mean(loadSubj, 'omitnan');
    sideSEM(i) = std(sideSubj, 0, 'omitnan') ./ sqrt(sum(~isnan(sideSubj)));
    loadSEM(i) = std(loadSubj, 0, 'omitnan') ./ sqrt(sum(~isnan(loadSubj)));
end

summaryTable = table(labels, sideMean, sideSEM, loadMean, loadSEM);

figure('Color', 'w', 'Position', [120 120 700 550]); hold on;
errorbar(sideMean, loadMean, loadSEM, loadSEM, sideSEM, sideSEM, 'o', ...
    'LineWidth', 1.5, 'MarkerSize', 7);
xline(0, ':k'); yline(0, ':k');
for i = 1:n
    text(sideMean(i), loadMean(i), ['  ', char(labels(i))], 'Interpreter', 'none');
end
xlabel(['Side evidence: ', summaryCfg.metricCfg.metric]);
ylabel(['Controlled load evidence: ', summaryCfg.metricCfg.metric]);
title(sprintf('Maintenance summary: %d-%d ms', summaryCfg.timeWin(1), summaryCfg.timeWin(2)));
box off; grid on;

saveas(gcf, fullfile(figdir, 'side_vs_controlled_load_summary.png'));
savefig(gcf, fullfile(figdir, 'side_vs_controlled_load_summary.fig'));
save(fullfile(figdir, 'spatial_control_group_stats.mat'), ...
    'allStats', 'summaryTable', 'subjectInclusion');

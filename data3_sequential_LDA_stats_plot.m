%% Group statistics and plots for data3 sequential LDA decoding
% Tests subject-level decoding matrices against theoretical chance (0.5)
% using 2-D cluster-based sign-flip permutation tests.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
addpath(codeDir);

cfg = struct();
cfg.metric = 'AUC';                 % 'AUC' or 'Acc'
cfg.chance = 0.5;
cfg.nPerm2D = 1000;
cfg.tail = 'right';
cfg.clusterAlpha = 0.05;
cfg.alpha = 0.05;
cfg.clusterStat = 'mass';
cfg.minClusterSize = 2;
cfg.clusterConnectivity = 'withinSegmentBlocks'; % do not connect clusters across segment boundaries
cfg.randomSeed = 2027;
cfg.analysisMode = 'maintOnly';       % must match data3_setsize1_vs6_LDA_decoding.m

cfg.doPairwiseModelContrasts = true;
cfg.pairTail = 'two';
cfg.pairList = { ...
    'CDA', 'Alpha'; ...
    'NoPCA', 'CDA'; ...
    'NoPCA', 'Alpha'; ...
    'PCA', 'NoPCA'; ...
    'GlobalAlpha', 'CDA'; ...
    'GlobalAlpha', 'PCA'};

% Subject inclusion options:
%   'original'                         = Wang et al. analysis subjects
%   'allDecoded'                       = every subject with a saved result
%   'minSideSetSizeTrials'             = min(left/right x low/high count) >= threshold
%   'originalAndMinSideSetSizeTrials'  = both criteria
cfg.subjectInclusion = 'original';
cfg.minSideSetSizeTrials = 20;

cfg.modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};
cfg.comparisons = { ...
    sprintf('setsize1_vs6_%s', cfg.analysisMode), fullfile(dataDir, sprintf('decoding_LDA_setsize1_vs6_segments_%s', cfg.analysisMode)); ...
    sprintf('setsize3_vs6_%s', cfg.analysisMode), fullfile(dataDir, sprintf('decoding_LDA_setsize3_vs6_segments_%s', cfg.analysisMode))};

cfg.colorLimits = [];               % [] = symmetric around chance
cfg.colorLimitBounds = [0 1];
cfg.pairColorLimits = [];           % [] = symmetric around zero across pairwise contrasts
cfg.figureDpi = 300;

allStats = struct();

for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    comparisonDir = cfg.comparisons{ci, 2};
    groupDir = fullfile(comparisonDir, 'GroupStats');
    figDir = fullfile(groupDir, 'figures');
    if ~isfolder(groupDir), mkdir(groupDir); end
    if ~isfolder(figDir), mkdir(figDir); end

    fprintf('\n%s\n', comparisonName);

    validModels = {};
    modelData = struct();
    clusterTables = {};

    for mi = 1:numel(cfg.modelNames)
        modelName = cfg.modelNames{mi};
        modelDir = fullfile(comparisonDir, modelName);
        if ~isfolder(modelDir)
            fprintf('  Skip %s: no folder.\n', modelName);
            continue;
        end

        [data, timesLow, timesHigh, usedFiles, subjects, design, inclusion] = ...
            load_group_matrices(modelDir, modelName, cfg);

        if isempty(data)
            fprintf('  Skip %s: no included subjects.\n', modelName);
            continue;
        end

        statCfg = make_stat_cfg(cfg);
        stat = cluster_perm_2d_matrix(data, timesLow, timesHigh, design, statCfg);

        modelData.(modelName) = struct( ...
            'data', data, ...
            'timesLow', timesLow, ...
            'timesHigh', timesHigh, ...
            'usedFiles', {usedFiles}, ...
            'subjects', subjects, ...
            'design', design, ...
            'inclusion', inclusion, ...
            'stat', stat);
        validModels{end+1} = modelName; %#ok<SAGROW>

        T = cluster_table(stat, comparisonName, modelName);
        clusterTables{end+1} = T; %#ok<SAGROW>

        save(fullfile(groupDir, sprintf('%s_%s_2d_cluster_stats.mat', modelName, cfg.metric)), ...
            'stat', 'cfg', 'usedFiles', 'subjects', 'design', 'inclusion', '-v7.3');
        writetable(T, fullfile(groupDir, sprintf('%s_%s_clusters.csv', modelName, cfg.metric)));

        fig = figure('Color', 'w', 'Position', [100 100 980 720]);
        ax = subplot(1, 1, 1, 'Parent', fig);
        plot_decoding_heatmap(ax, squeeze(mean(data, 1, 'omitnan')), ...
            timesLow, timesHigh, stat, design, cfg, sprintf('%s %s %s', comparisonName, modelName, cfg.metric));
        export_figure(fig, fullfile(figDir, sprintf('%s_%s_heatmap.png', modelName, cfg.metric)), cfg.figureDpi);
        savefig(fig, fullfile(figDir, sprintf('%s_%s_heatmap.fig', modelName, cfg.metric)));
        close(fig);

        fprintf('  %s: n=%d, matrix=%dx%d, significant clusters=%d\n', ...
            modelName, size(data,1), size(data,2), size(data,3), numel(stat.significantClusters));
    end

    if ~isempty(validModels)
        fig = plot_stacked_model_heatmap(modelData, validModels, comparisonName, cfg);
        export_figure(fig, fullfile(figDir, sprintf('%s_all_models_%s_heatmaps.png', comparisonName, cfg.metric)), cfg.figureDpi);
        savefig(fig, fullfile(figDir, sprintf('%s_all_models_%s_heatmaps.fig', comparisonName, cfg.metric)));
        export_figure(fig, fullfile(figDir, sprintf('%s_all_models_stacked_%s_heatmaps.png', comparisonName, cfg.metric)), cfg.figureDpi);
        savefig(fig, fullfile(figDir, sprintf('%s_all_models_stacked_%s_heatmaps.fig', comparisonName, cfg.metric)));
        close(fig);

        pairwiseStats = struct();
        pairClusterTables = {};
        if cfg.doPairwiseModelContrasts
            pairDir = fullfile(groupDir, 'Pairwise');
            pairFigDir = fullfile(figDir, 'pairwise');
            [pairwiseStats, pairClusterTables] = make_pairwise_model_contrasts( ...
                modelData, validModels, comparisonName, pairDir, pairFigDir, cfg);

            if ~isempty(pairClusterTables)
                allPairClusters = vertcat_tables(pairClusterTables);
                writetable(allPairClusters, fullfile(pairDir, ...
                    sprintf('%s_pairwise_model_clusters.csv', cfg.metric)));
            end
        end

        allClusters = vertcat_tables(clusterTables);
        writetable(allClusters, fullfile(groupDir, sprintf('%s_all_model_clusters.csv', cfg.metric)));
        allStats.(comparisonName) = modelData;
        if cfg.doPairwiseModelContrasts
            allStats.(comparisonName).PairwiseModelContrasts = pairwiseStats;
        end
    end
end

save(fullfile(dataDir, sprintf('data3_sequential_LDA_%s_group_stats.mat', cfg.metric)), ...
    'allStats', 'cfg', '-v7.3');

fprintf('\nSequential LDA group stats finished.\n');

%% ========================================================================
function statCfg = make_stat_cfg(cfg)

statCfg = struct();
statCfg.null = cfg.chance;
statCfg.nPerm = cfg.nPerm2D;
statCfg.tail = cfg.tail;
statCfg.clusterAlpha = cfg.clusterAlpha;
statCfg.alpha = cfg.alpha;
statCfg.clusterStat = cfg.clusterStat;
statCfg.minClusterSize = cfg.minClusterSize;
statCfg.clusterConnectivity = cfg.clusterConnectivity;
statCfg.randomSeed = cfg.randomSeed;
end

%% ========================================================================
function [data, timesLow, timesHigh, usedFiles, subjects, design, inclusion] = ...
    load_group_matrices(modelDir, modelName, cfg)

files = dir(fullfile(modelDir, 'sub*.mat'));
data = [];
timesLow = [];
timesHigh = [];
usedFiles = {};
subjects = [];
design = [];
inclusion = struct('file', {{}}, 'subject', [], 'included', [], ...
    'minSideSetSizeTrials', [], 'reason', {{}});

if isempty(files)
    return;
end

for fi = 1:numel(files)
    fpath = fullfile(files(fi).folder, files(fi).name);
    S = load(fpath, modelName);
    if ~isfield(S, modelName)
        continue;
    end
    R = S.(modelName);
    if ~isfield(R, cfg.metric)
        warning('Skipping %s: missing metric %s.', files(fi).name, cfg.metric);
        continue;
    end

    subject = get_subject_id(R, files(fi).name);
    [includeNow, reason, minCount] = include_subject(R, subject, cfg);

    inclusion.file{end+1,1} = fpath;
    inclusion.subject(end+1,1) = subject;
    inclusion.included(end+1,1) = includeNow;
    inclusion.minSideSetSizeTrials(end+1,1) = minCount;
    inclusion.reason{end+1,1} = reason;

    if ~includeNow
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

    data(end+1,:,:) = M; %#ok<AGROW>
    usedFiles{end+1,1} = fpath; %#ok<AGROW>
    subjects(end+1,1) = subject; %#ok<AGROW>
end
end

%% ========================================================================
function subject = get_subject_id(R, fileName)

if isfield(R, 'subject') && ~isempty(R.subject)
    subject = double(R.subject);
    return;
end

tok = regexp(fileName, '^sub(\d+)\.mat$', 'tokens', 'once');
if isempty(tok)
    subject = NaN;
else
    subject = str2double(tok{1});
end
end

%% ========================================================================
function [includeNow, reason, minCount] = include_subject(R, subject, cfg)

minCount = NaN;
if isfield(R, 'withinSide') && isfield(R.withinSide, 'leftCountsLowHigh') && ...
        isfield(R.withinSide, 'rightCountsLowHigh')
    minCount = min([R.withinSide.leftCountsLowHigh(:); R.withinSide.rightCountsLowHigh(:)]);
end

inclusionMode = lower(cfg.subjectInclusion);
isOriginal = any(str2double(data3_original_subjects()) == subject);
passesMin = ~isnan(minCount) && minCount >= cfg.minSideSetSizeTrials;

switch inclusionMode
    case 'alldecoded'
        includeNow = true;
        reason = 'allDecoded';
    case 'original'
        includeNow = isOriginal;
        reason = 'originalSubjects';
    case 'minsidesetsizetrials'
        includeNow = passesMin;
        reason = sprintf('minSideSetSizeTrials>=%d', cfg.minSideSetSizeTrials);
    case 'originalandminsidesetsizetrials'
        includeNow = isOriginal && passesMin;
        reason = sprintf('originalSubjects and minSideSetSizeTrials>=%d', cfg.minSideSetSizeTrials);
    otherwise
        error('Unknown cfg.subjectInclusion: %s.', cfg.subjectInclusion);
end
end

%% ========================================================================
function [pairwiseStats, clusterTables] = make_pairwise_model_contrasts( ...
    modelData, validModels, comparisonName, pairDir, figDir, cfg)

if ~isfolder(pairDir), mkdir(pairDir); end
if ~isfolder(figDir), mkdir(figDir); end

pairwiseStats = struct();
clusterTables = {};
pairs = filter_pairs(cfg.pairList, validModels);
if isempty(pairs)
    fprintf('  Pairwise model contrasts: no configured model pairs available.\n');
    return;
end

pairInfo = cell(0,3);
maxAbsVal = 0;

for pi = 1:size(pairs, 1)
    modelA = pairs{pi, 1};
    modelB = pairs{pi, 2};
    A = modelData.(modelA);
    B = modelData.(modelB);

    assert_same_decoding_grid(A, B, modelA, modelB);
    [dataA, dataB, subjects, usedFilesA, usedFilesB] = align_pair_model_data(A, B);
    if isempty(subjects)
        fprintf('  Skip pair %s - %s: no common included subjects.\n', modelA, modelB);
        continue;
    end

    diffData = dataA - dataB;
    statCfg = make_stat_cfg(cfg);
    statCfg.null = 0;
    statCfg.tail = cfg.pairTail;
    stat = cluster_perm_2d_matrix(diffData, A.timesLow, A.timesHigh, A.design, statCfg);

    statName = pair_stat_name(modelA, modelB);
    pairwiseStats.(statName) = struct( ...
        'modelA', modelA, ...
        'modelB', modelB, ...
        'data', diffData, ...
        'timesLow', A.timesLow, ...
        'timesHigh', A.timesHigh, ...
        'subjects', subjects, ...
        'usedFilesA', {usedFilesA}, ...
        'usedFilesB', {usedFilesB}, ...
        'design', A.design, ...
        'stat', stat);

    T = pair_cluster_table(stat, comparisonName, modelA, modelB, statName);
    clusterTables{end+1} = T; %#ok<AGROW>

    save(fullfile(pairDir, sprintf('%s_%s_2d_cluster_stats.mat', statName, cfg.metric)), ...
        'stat', 'cfg', 'diffData', 'subjects', 'usedFilesA', 'usedFilesB', ...
        'modelA', 'modelB', '-v7.3');
    writetable(T, fullfile(pairDir, sprintf('%s_%s_clusters.csv', statName, cfg.metric)));

    thisMax = max(abs(stat.mean(:)), [], 'omitnan');
    if ~isempty(thisMax) && ~isnan(thisMax)
        maxAbsVal = max(maxAbsVal, thisMax);
    end
    pairInfo(end+1,:) = {modelA, modelB, statName}; %#ok<AGROW>

    fprintf('  %s - %s: n=%d, matrix=%dx%d, significant clusters=%d\n', ...
        modelA, modelB, size(diffData, 1), size(diffData, 2), size(diffData, 3), ...
        numel(stat.significantClusters));
end

if isempty(pairInfo)
    return;
end

plotCfg = cfg;
plotCfg.chance = 0;
plotCfg.colorLimitBounds = [];
plotCfg.isContrast = true;
if isempty(cfg.pairColorLimits)
    maxAbsVal = max(maxAbsVal, 0.02);
    plotCfg.colorLimits = [-maxAbsVal maxAbsVal];
else
    plotCfg.colorLimits = cfg.pairColorLimits;
end

for pi = 1:size(pairInfo, 1)
    modelA = pairInfo{pi, 1};
    modelB = pairInfo{pi, 2};
    statName = pairInfo{pi, 3};
    P = pairwiseStats.(statName);

    fig = figure('Color', 'w', 'Position', [100 100 980 720]);
    ax = subplot(1, 1, 1, 'Parent', fig);
    plot_decoding_heatmap(ax, squeeze(mean(P.data, 1, 'omitnan')), ...
        P.timesLow, P.timesHigh, P.stat, P.design, plotCfg, ...
        sprintf('%s %s - %s %s', comparisonName, modelA, modelB, cfg.metric));
    export_figure(fig, fullfile(figDir, sprintf('%s_%s_heatmap.png', statName, cfg.metric)), cfg.figureDpi);
    savefig(fig, fullfile(figDir, sprintf('%s_%s_heatmap.fig', statName, cfg.metric)));
    close(fig);
end

fig = figure('Color', 'w', 'Position', [80 80 1500 900]);
[nRow, nCol] = subplot_grid(size(pairInfo, 1));
for pi = 1:size(pairInfo, 1)
    modelA = pairInfo{pi, 1};
    modelB = pairInfo{pi, 2};
    statName = pairInfo{pi, 3};
    P = pairwiseStats.(statName);
    ax = subplot(nRow, nCol, pi);
    plot_decoding_heatmap(ax, squeeze(mean(P.data, 1, 'omitnan')), ...
        P.timesLow, P.timesHigh, P.stat, P.design, plotCfg, ...
        sprintf('%s - %s', modelA, modelB));
end
sgtitle(sprintf('%s pairwise model contrasts, %s, nPerm=%d, tail=%s', ...
    comparisonName, cfg.metric, cfg.nPerm2D, cfg.pairTail), 'Interpreter', 'none');
export_figure(fig, fullfile(figDir, sprintf('%s_pairwise_model_%s_heatmaps.png', comparisonName, cfg.metric)), cfg.figureDpi);
savefig(fig, fullfile(figDir, sprintf('%s_pairwise_model_%s_heatmaps.fig', comparisonName, cfg.metric)));
close(fig);
end

%% ========================================================================
function pairs = filter_pairs(pairList, validModels)

keep = false(size(pairList, 1), 1);
for i = 1:size(pairList, 1)
    keep(i) = ismember(pairList{i, 1}, validModels) && ismember(pairList{i, 2}, validModels);
end
pairs = pairList(keep,:);
end

%% ========================================================================
function [dataA, dataB, subjects, usedFilesA, usedFilesB] = align_pair_model_data(A, B)

[subjects, ia, ib] = intersect(A.subjects(:), B.subjects(:), 'stable');
dataA = A.data(ia,:,:);
dataB = B.data(ib,:,:);
usedFilesA = A.usedFiles(ia);
usedFilesB = B.usedFiles(ib);
end

%% ========================================================================
function assert_same_decoding_grid(A, B, modelA, modelB)

if numel(A.timesLow) ~= numel(B.timesLow) || any(abs(A.timesLow(:) - B.timesLow(:)) > 1e-9) || ...
        numel(A.timesHigh) ~= numel(B.timesHigh) || any(abs(A.timesHigh(:) - B.timesHigh(:)) > 1e-9)
    error('Time axis mismatch between %s and %s.', modelA, modelB);
end
end

%% ========================================================================
function statName = pair_stat_name(modelA, modelB)

statName = sprintf('%s_minus_%s', modelA, modelB);
end

%% ========================================================================
function stat = cluster_perm_2d_matrix(data, timesLow, timesHigh, design, cfg)

if nargin < 5 || isempty(cfg)
    cfg = struct();
end
cfg = fill_stat_defaults(cfg);

validateattributes(data, {'numeric'}, {'3d', 'nonempty'}, mfilename, 'data', 1);
[nSub, nLow, nHigh] = size(data);
timesLow = timesLow(:);
timesHigh = timesHigh(:);
if numel(timesLow) ~= nLow || numel(timesHigh) ~= nHigh
    error('timesLow/timesHigh must match data row/column dimensions.');
end
blockId = make_cluster_block_id(timesLow, timesHigh, design, cfg.clusterConnectivity);

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed, 'twister');
end

nullVal = cfg.null;
if isscalar(nullVal)
    nullMat = repmat(nullVal, nLow, nHigh);
else
    nullMat = nullVal;
    if ~isequal(size(nullMat), [nLow nHigh])
        error('cfg.null must be scalar or nLow x nHigh.');
    end
end

D = data - reshape(nullMat, [1 nLow nHigh]);
[tObs, pObs, dfObs] = one_sample_t_2d(D, cfg.tail);
clusterMask = cluster_forming_mask_2d(tObs, pObs, cfg.clusterAlpha, cfg.tail);
obsClusters = find_clusters_2d(clusterMask, cfg.minClusterSize, blockId);
obsClusterStats = compute_cluster_stats_2d(tObs, obsClusters, cfg.tail, cfg.clusterStat);

maxClusterStatNull = zeros(cfg.nPerm, 1);
for pi = 1:cfg.nPerm
    signs = (randi([0 1], nSub, 1) * 2) - 1;
    Dp = D .* reshape(signs, [nSub 1 1]);

    [tPerm, pPerm] = one_sample_t_2d(Dp, cfg.tail);
    permMask = cluster_forming_mask_2d(tPerm, pPerm, cfg.clusterAlpha, cfg.tail);
    permClusters = find_clusters_2d(permMask, cfg.minClusterSize, blockId);
    permStats = compute_cluster_stats_2d(tPerm, permClusters, cfg.tail, cfg.clusterStat);

    if isempty(permStats)
        maxClusterStatNull(pi) = 0;
    else
        maxClusterStatNull(pi) = max(permStats);
    end
end

clusters = struct('idx', {}, 'rowIdx', {}, 'colIdx', {}, ...
    'lowStartTime', {}, 'lowEndTime', {}, 'highStartTime', {}, 'highEndTime', {}, ...
    'clusterStat', {}, 'p', {}, 'nSamples', {});
significantClusters = clusters;
significantMask = false(nLow, nHigh);
clusterP = nan(1, numel(obsClusters));

for ci = 1:numel(obsClusters)
    idx = obsClusters{ci};
    [rowIdx, colIdx] = ind2sub([nLow nHigh], idx);
    clusterP(ci) = (1 + sum(maxClusterStatNull >= obsClusterStats(ci))) / (cfg.nPerm + 1);

    clusters(ci).idx = idx;
    clusters(ci).rowIdx = rowIdx;
    clusters(ci).colIdx = colIdx;
    clusters(ci).lowStartTime = min(timesLow(rowIdx));
    clusters(ci).lowEndTime = max(timesLow(rowIdx));
    clusters(ci).highStartTime = min(timesHigh(colIdx));
    clusters(ci).highEndTime = max(timesHigh(colIdx));
    clusters(ci).clusterStat = obsClusterStats(ci);
    clusters(ci).p = clusterP(ci);
    clusters(ci).nSamples = numel(idx);

    if clusterP(ci) <= cfg.alpha
        significantMask(idx) = true;
        significantClusters(end+1) = clusters(ci); %#ok<AGROW>
    end
end

stat = struct();
stat.data = data;
stat.diff = D;
stat.null = nullMat;
stat.timesLow = timesLow;
stat.timesHigh = timesHigh;
stat.nSubj = nSub;
stat.mean = squeeze(mean(data, 1, 'omitnan'));
nValid = squeeze(sum(~isnan(data), 1));
stat.sem = squeeze(std(data, 0, 1, 'omitnan')) ./ sqrt(nValid);
stat.tObs = tObs;
stat.pObs = pObs;
stat.dfObs = dfObs;
stat.clusterFormingMask = clusterMask;
stat.clusterBlockId = blockId;
stat.clusters = clusters;
stat.clusterP = clusterP;
stat.significantClusters = significantClusters;
stat.significantMask = significantMask;
stat.maxClusterStatNull = maxClusterStatNull;
stat.cfg = cfg;
end

%% ========================================================================
function fig = plot_stacked_model_heatmap(modelData, validModels, comparisonName, cfg)

nModel = numel(validModels);
refModel = validModels{1};
ref = modelData.(refModel);
timesLow = ref.timesLow(:);
timesHigh = ref.timesHigh(:);
design = ref.design;

[xStart, xEnd, lowStart, lowEnd] = decoding_grid_bounds(design, timesHigh, timesLow);
lowSpan = lowEnd - lowStart;
xSpan = xEnd - xStart;
if lowSpan <= 0
    lowSpan = max(timesLow) - min(timesLow);
end
if lowSpan <= 0
    lowSpan = 1;
end
if xSpan <= 0
    xSpan = max(timesHigh) - min(timesHigh);
end
if xSpan <= 0
    xSpan = 1;
end

nLow = numel(timesLow);
nHigh = numel(timesHigh);
stackMat = nan(nLow * nModel, nHigh);
yTicks = nan(1, nModel);
yLabels = cell(1, nModel);

for mi = 1:nModel
    modelName = validModels{mi};
    D = modelData.(modelName);
    assert_same_decoding_grid(ref, D, refModel, modelName);

    blockFromBottom = nModel - mi + 1;
    rowIdx = (blockFromBottom - 1) * nLow + (1:nLow);
    stackMat(rowIdx,:) = squeeze(mean(D.data, 1, 'omitnan'));

    blockBase = (blockFromBottom - 1) * lowSpan;
    yTicks(blockFromBottom) = blockBase + lowSpan / 2;
    yLabels{blockFromBottom} = sprintf('%s | setsize %d', modelName, D.design.lowSetSize);
end

yTotal = nModel * lowSpan;
figWidth = 1650;
figHeight = stacked_figure_height(figWidth, xSpan, yTotal);
fig = figure('Color', 'w', 'Position', [60 60 figWidth figHeight]);
ax = axes('Parent', fig);

imagesc(ax, [xStart xEnd], [0 yTotal], stackMat);
set(ax, 'YDir', 'normal');
hold(ax, 'on');
colormap(ax, parula);
colorbar(ax);
apply_heatmap_color_limits(ax, stackMat, cfg);

for mi = 1:nModel
    modelName = validModels{mi};
    D = modelData.(modelName);
    blockFromBottom = nModel - mi + 1;
    blockBase = (blockFromBottom - 1) * lowSpan;
    yContour = blockBase + D.timesLow(:)' - lowStart;

    if any(D.stat.significantMask(:))
        contour(ax, D.timesHigh(:)', yContour, double(D.stat.significantMask), [1 1], ...
            'Color', 'k', 'LineWidth', 1.4);
    end
end

plot_stacked_segment_boundaries(ax, design, nModel, lowSpan, lowStart);
add_high_event_lines(ax, design);

xlim(ax, [xStart xEnd]);
ylim(ax, [0 yTotal]);
daspect(ax, [1 1 1]);

set(ax, ...
    'YTick', yTicks, ...
    'YTickLabel', yLabels, ...
    'TickDir', 'out', ...
    'FontSize', 11);

xlabel(ax, 'Set-size 6 time (ms)');
ylabel(ax, 'Model / low-set-size time');
title(ax, sprintf('%s %s, nPerm=%d, inclusion=%s', ...
    comparisonName, cfg.metric, cfg.nPerm2D, cfg.subjectInclusion), ...
    'Interpreter', 'none');
box(ax, 'off');
end

%% ========================================================================
function cfg = fill_stat_defaults(cfg)

if ~isfield(cfg, 'null'), cfg.null = 0.5; end
if ~isfield(cfg, 'nPerm'), cfg.nPerm = 1000; end
if ~isfield(cfg, 'tail'), cfg.tail = 'right'; end
if ~isfield(cfg, 'clusterAlpha'), cfg.clusterAlpha = 0.05; end
if ~isfield(cfg, 'alpha'), cfg.alpha = 0.05; end
if ~isfield(cfg, 'clusterStat'), cfg.clusterStat = 'mass'; end
if ~isfield(cfg, 'minClusterSize'), cfg.minClusterSize = 1; end
if ~isfield(cfg, 'clusterConnectivity'), cfg.clusterConnectivity = 'withinSegmentBlocks'; end
if ~isfield(cfg, 'randomSeed'), cfg.randomSeed = []; end
end

%% ========================================================================
function blockId = make_cluster_block_id(timesLow, timesHigh, design, connectivity)

timesLow = timesLow(:);
timesHigh = timesHigh(:);

switch lower(connectivity)
    case 'continuous'
        blockId = ones(numel(timesLow), numel(timesHigh));

    case 'withinsegmentblocks'
        lowSeg = assign_time_segments(timesLow, design.lowWindowsMs);
        highWindows = [design.highSegmentStartsMs(:), ...
                       design.highSegmentStartsMs(:) + design.highSegmentWidthMs];
        highSeg = assign_time_segments(timesHigh, highWindows);

        blockId = nan(numel(timesLow), numel(timesHigh));
        for li = 1:numel(lowSeg)
            for hi = 1:numel(highSeg)
                blockId(li,hi) = lowSeg(li) + (highSeg(hi) - 1) * size(design.lowWindowsMs, 1);
            end
        end

    otherwise
        error('Unknown clusterConnectivity: %s.', connectivity);
end
end

%% ========================================================================
function segIdx = assign_time_segments(times, windowsMs)

segIdx = nan(numel(times), 1);
for wi = 1:size(windowsMs, 1)
    inWin = times >= windowsMs(wi,1) & times <= windowsMs(wi,2);
    segIdx(inWin) = wi;
end

if any(isnan(segIdx))
    error('Some output time points could not be assigned to segment windows.');
end
end

%% ========================================================================
function [tVals, pVals, df] = one_sample_t_2d(D, tail)

n = squeeze(sum(~isnan(D), 1));
m = squeeze(mean(D, 1, 'omitnan'));
s = squeeze(std(D, 0, 1, 'omitnan'));
se = s ./ sqrt(n);
tVals = m ./ se;
tVals(se == 0 | n < 2 | isnan(tVals)) = 0;

df = n - 1;
pVals = ones(size(tVals));
valid = df > 0;

switch lower(tail)
    case 'right'
        pVals(valid) = 1 - tcdf(tVals(valid), df(valid));
    case 'left'
        pVals(valid) = tcdf(tVals(valid), df(valid));
    case 'two'
        pVals(valid) = 2 * (1 - tcdf(abs(tVals(valid)), df(valid)));
    otherwise
        error('Unsupported tail: %s.', tail);
end
pVals(~valid | isnan(pVals)) = 1;
end

%% ========================================================================
function mask = cluster_forming_mask_2d(tVals, pVals, clusterAlpha, tail)

switch lower(tail)
    case 'right'
        mask = pVals < clusterAlpha & tVals > 0;
    case 'left'
        mask = pVals < clusterAlpha & tVals < 0;
    case 'two'
        mask = pVals < clusterAlpha;
end
mask(isnan(mask)) = false;
end

%% ========================================================================
function clusters = find_clusters_2d(mask, minClusterSize, blockId)

[nRow, nCol] = size(mask);
visited = false(nRow, nCol);
clusters = {};

for r = 1:nRow
    for c = 1:nCol
        if ~mask(r,c) || visited(r,c)
            continue;
        end

        queueR = zeros(numel(mask), 1);
        queueC = zeros(numel(mask), 1);
        head = 1;
        tail = 1;
        queueR(tail) = r;
        queueC(tail) = c;
        visited(r,c) = true;
        idx = zeros(numel(mask), 1);
        nIdx = 0;

        while head <= tail
            rr = queueR(head);
            cc = queueC(head);
            head = head + 1;

            nIdx = nIdx + 1;
            idx(nIdx) = sub2ind([nRow nCol], rr, cc);

            neigh = [rr-1 cc; rr+1 cc; rr cc-1; rr cc+1];
            for ni = 1:size(neigh, 1)
                nr = neigh(ni, 1);
                nc = neigh(ni, 2);
                if nr < 1 || nr > nRow || nc < 1 || nc > nCol
                    continue;
                end
                if blockId(nr,nc) == blockId(rr,cc) && mask(nr,nc) && ~visited(nr,nc)
                    tail = tail + 1;
                    queueR(tail) = nr;
                    queueC(tail) = nc;
                    visited(nr,nc) = true;
                end
            end
        end

        idx = idx(1:nIdx);
        if numel(idx) >= minClusterSize
            clusters{end+1} = idx(:)'; %#ok<AGROW>
        end
    end
end
end

%% ========================================================================
function clusterStats = compute_cluster_stats_2d(tVals, clusters, tail, clusterStat)

clusterStats = nan(1, numel(clusters));
for ci = 1:numel(clusters)
    idx = clusters{ci};
    switch lower(clusterStat)
        case 'mass'
            switch lower(tail)
                case 'right'
                    clusterStats(ci) = sum(tVals(idx));
                case 'left'
                    clusterStats(ci) = sum(-tVals(idx));
                case 'two'
                    clusterStats(ci) = sum(abs(tVals(idx)));
            end
        case 'size'
            clusterStats(ci) = numel(idx);
        otherwise
            error('Unsupported clusterStat: %s.', clusterStat);
    end
end
end

%% ========================================================================
function plot_decoding_heatmap(ax, meanMat, timesLow, timesHigh, stat, design, cfg, titleText)

imagesc(ax, timesHigh, timesLow, meanMat);
set(ax, 'YDir', 'normal');
axis(ax, 'tight');
hold(ax, 'on');
if isfield(cfg, 'isContrast') && cfg.isContrast
    colormap(ax, blue_white_red_colormap(256));
else
    colormap(ax, parula);
end
colorbar(ax);

apply_heatmap_color_limits(ax, meanMat, cfg);

if any(stat.significantMask(:))
    contour(ax, timesHigh, timesLow, double(stat.significantMask), [1 1], ...
        'Color', 'k', 'LineWidth', 1.4);
end

plot_segment_boundaries(ax, design);
add_high_event_lines(ax, design);
apply_equal_time_aspect(ax, design, timesLow, timesHigh);

xlabel(ax, 'Set-size 6 time (ms)');
ylabel(ax, sprintf('Set-size %d time (ms)', design.lowSetSize));
title(ax, titleText, 'Interpreter', 'none');
box(ax, 'off');
set(ax, 'FontSize', 11);
end

%% ========================================================================
function apply_heatmap_color_limits(ax, values, cfg)

if isempty(cfg.colorLimits)
    delta = max(abs(values(:) - cfg.chance), [], 'omitnan');
    if isempty(delta) || isnan(delta) || delta == 0
        delta = 0.02;
    end
    delta = max(delta, 0.02);
    colorLim = [cfg.chance - delta, cfg.chance + delta];
    if isfield(cfg, 'colorLimitBounds') && ~isempty(cfg.colorLimitBounds)
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
end

%% ========================================================================
function apply_equal_time_aspect(ax, design, timesLow, timesHigh)

[xStart, xEnd, yStart, yEnd] = decoding_grid_bounds(design, timesHigh, timesLow);
xlim(ax, [xStart xEnd]);
ylim(ax, [yStart yEnd]);
daspect(ax, [1 1 1]);
end

%% ========================================================================
function [xStart, xEnd, yStart, yEnd] = decoding_grid_bounds(design, timesHigh, timesLow)

timesHigh = timesHigh(:);
timesLow = timesLow(:);

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
end

%% ========================================================================
function figHeight = stacked_figure_height(figWidth, xSpan, ySpan)

plotHeight = figWidth * ySpan / xSpan;
figHeight = round(plotHeight + 260);
figHeight = max(figHeight, 850);
figHeight = min(figHeight, 5200);
end

%% ========================================================================
function plot_stacked_segment_boundaries(ax, design, nModel, lowSpan, lowStart)

if isfield(design, 'highSegmentStartsMs') && isfield(design, 'highSegmentWidthMs')
    xBoundaries = [design.highSegmentStartsMs(:); design.highSegmentStartsMs(end) + design.highSegmentWidthMs];
    for i = 1:numel(xBoundaries)
        xline(ax, xBoundaries(i), ':', 'Color', [0.45 0.45 0.45], 'HandleVisibility', 'off');
    end
end

if isfield(design, 'lowWindowsMs')
    yBoundaries = unique([design.lowWindowsMs(:,1); design.lowWindowsMs(:,2)]);
    for bi = 1:nModel
        blockBase = (bi - 1) * lowSpan;
        for yi = 1:numel(yBoundaries)
            yline(ax, blockBase + yBoundaries(yi) - lowStart, ':', ...
                'Color', [0.45 0.45 0.45], 'HandleVisibility', 'off');
        end
    end
end

for bi = 0:nModel
    yline(ax, bi * lowSpan, '-', 'Color', [0.15 0.15 0.15], ...
        'LineWidth', 0.8, 'HandleVisibility', 'off');
end
end

%% ========================================================================
function add_high_event_lines(ax, design)

if ~(isfield(design, 'highSegmentStartsMs') && isfield(design, 'highSegmentWidthMs'))
    return
end

if isfield(design, 'analysisMode') && strcmpi(char(design.analysisMode), 'maintOnly')
    return
end

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

%% ========================================================================
function cmap = blue_white_red_colormap(n)

if nargin < 1
    n = 256;
end
n1 = floor(n / 2);
n2 = n - n1;
blue = [0.1137 0.3020 0.6235];
white = [1 1 1];
red = [0.6980 0.0941 0.1686];
cmap = [interp1([1 n1], [blue; white], 1:n1); ...
        interp1([1 n2], [white; red], 1:n2)];
end

%% ========================================================================
function plot_segment_boundaries(ax, design)

if isfield(design, 'highSegmentStartsMs') && isfield(design, 'highSegmentWidthMs')
    xBoundaries = [design.highSegmentStartsMs(:); design.highSegmentStartsMs(end) + design.highSegmentWidthMs];
    for i = 1:numel(xBoundaries)
        xline(ax, xBoundaries(i), ':', 'Color', [0.45 0.45 0.45], 'HandleVisibility', 'off');
    end
end

if isfield(design, 'lowWindowsMs')
    yBoundaries = unique([design.lowWindowsMs(:,1); design.lowWindowsMs(:,2)]);
    for i = 1:numel(yBoundaries)
        yline(ax, yBoundaries(i), ':', 'Color', [0.45 0.45 0.45], 'HandleVisibility', 'off');
    end
end
end

%% ========================================================================
function T = cluster_table(stat, comparisonName, modelName)

n = numel(stat.clusters);
comparison = repmat({comparisonName}, n, 1);
model = repmat({modelName}, n, 1);
clusterId = (1:n)';
lowStartTime = nan(n,1);
lowEndTime = nan(n,1);
highStartTime = nan(n,1);
highEndTime = nan(n,1);
nSamples = nan(n,1);
clusterStat = nan(n,1);
p = nan(n,1);
significant = false(n,1);

for i = 1:n
    lowStartTime(i) = stat.clusters(i).lowStartTime;
    lowEndTime(i) = stat.clusters(i).lowEndTime;
    highStartTime(i) = stat.clusters(i).highStartTime;
    highEndTime(i) = stat.clusters(i).highEndTime;
    nSamples(i) = stat.clusters(i).nSamples;
    clusterStat(i) = stat.clusters(i).clusterStat;
    p(i) = stat.clusters(i).p;
    significant(i) = p(i) <= stat.cfg.alpha;
end

T = table(comparison, model, clusterId, lowStartTime, lowEndTime, ...
    highStartTime, highEndTime, nSamples, clusterStat, p, significant);
end

%% ========================================================================
function T = pair_cluster_table(stat, comparisonName, modelA, modelB, contrastName)

n = numel(stat.clusters);
comparison = repmat({comparisonName}, n, 1);
contrast = repmat({contrastName}, n, 1);
modelACol = repmat({modelA}, n, 1);
modelBCol = repmat({modelB}, n, 1);
clusterId = (1:n)';
lowStartTime = nan(n,1);
lowEndTime = nan(n,1);
highStartTime = nan(n,1);
highEndTime = nan(n,1);
nSamples = nan(n,1);
clusterStat = nan(n,1);
p = nan(n,1);
significant = false(n,1);

for i = 1:n
    lowStartTime(i) = stat.clusters(i).lowStartTime;
    lowEndTime(i) = stat.clusters(i).lowEndTime;
    highStartTime(i) = stat.clusters(i).highStartTime;
    highEndTime(i) = stat.clusters(i).highEndTime;
    nSamples(i) = stat.clusters(i).nSamples;
    clusterStat(i) = stat.clusters(i).clusterStat;
    p(i) = stat.clusters(i).p;
    significant(i) = p(i) <= stat.cfg.alpha;
end

T = table(comparison, contrast, modelACol, modelBCol, clusterId, ...
    lowStartTime, lowEndTime, highStartTime, highEndTime, nSamples, ...
    clusterStat, p, significant, 'VariableNames', ...
    {'comparison', 'contrast', 'modelA', 'modelB', 'clusterId', ...
     'lowStartTime', 'lowEndTime', 'highStartTime', 'highEndTime', ...
     'nSamples', 'clusterStat', 'p', 'significant'});
end

%% ========================================================================
function T = vertcat_tables(tablesIn)

if isempty(tablesIn)
    T = table();
    return;
end

T = tablesIn{1};
for i = 2:numel(tablesIn)
    T = [T; tablesIn{i}]; %#ok<AGROW>
end
end

%% ========================================================================
function [nRow, nCol] = subplot_grid(nPlot)

nCol = ceil(sqrt(nPlot));
nRow = ceil(nPlot / nCol);
end

%% ========================================================================
function export_figure(fig, outFile, dpi)

outDir = fileparts(outFile);
if ~isfolder(outDir)
    mkdir(outDir);
end
print(fig, outFile, '-dpng', sprintf('-r%d', dpi));
end

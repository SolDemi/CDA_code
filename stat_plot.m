%% ===================== basic config =====================

clear; clc;
cfg = struct();

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);

sideFeatures = {'VoltageRawLR', 'AlphaRawLR', 'VoltageLminusR', 'AlphaLminusR', 'GlobalAlphaMean'};
loadFeatures = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};
analysisNames = {'loadDecoding', 'sideDecoding', 'loadWithinSide', 'loadSideBalanced', 'loadCrossSide'};

decodefolder = 'LDA';   % 'SVM' or 'LDA'
maindir = fullfile(projectRoot, 'data1');

analysis = 3;           % 1=loadDecoding, 2=sideDecoding, 3=loadWithinSide

switch analysis
    case 1
        decodingDir = fullfile(maindir, ['decoding_' decodefolder]);
        cfg.modelNames = {'CDA', 'Alpha', 'NoPCA', 'PCA', 'GlobalAlpha'};

        cfg.pairList = {
            'CDA',   'Alpha'
            'NoPCA', 'CDA'
            'NoPCA', 'Alpha'
            'PCA',   'NoPCA'
            'GlobalAlpha', 'CDA'
            'GlobalAlpha', 'PCA'
            };
    case 2
        decodingDir = fullfile(maindir, ['decoding_' decodefolder '_spatialControl'],analysisNames{analysis});
        cfg.modelNames = sideFeatures;

        cfg.pairList = {
            'VoltageRawLR',   'AlphaRawLR'
            'VoltageLminusR', 'AlphaLminusR'
            };

    otherwise
        decodingDir = fullfile(maindir, ['decoding_' decodefolder '_spatialControl'],analysisNames{analysis});
        cfg.modelNames = loadFeatures;
        cfg.pairList = {
            'CDA',   'Alpha'
            'NoPCA', 'CDA'
            'NoPCA', 'Alpha'
            'PCA',   'NoPCA'
            'GlobalAlpha', 'CDA'
            'GlobalAlpha', 'PCA'
            };
        
end
saveDir = fullfile(decodingDir, 'GroupStats');

if ~isfolder(saveDir), mkdir(saveDir); end

cfg.metric = 'Acc';                  % 'AUC' or 'predictAcc'
cfg.shuffleMetric = 'AccShuffle';    % 'AUCShuffle' or 'predictAccShuffle'
cfg.filePattern = '*.mat';

cfg.chance = 0.5;
cfg.delayWindow = [250 inf];         % maintenance summary: 250 ms to the end

[~, dataName] = fileparts(maindir);
if strcmpi(dataName, 'data0') 
    cfg.eventLines = [0 250];
else
    cfg.eventLines = [0 150];
end
cfg.eventLineLabels = {'Array onset', 'Array offset'};

cfg.nPerm1D = 1000;
cfg.nPerm2D = 1000;
cfg.clusterAlpha = 0.05;
cfg.alpha = 0.05;
cfg.randomSeed = 1;

cfg.doAgainstShuffleTime = true;
cfg.doDelaySummary = true;
cfg.doPairwiseTime = true;
cfg.doTimeGeneralization = true;
cfg.doTGPairwise = true;



colors = lines(numel(cfg.modelNames));

%% ===================== load diagonal time series =====================
D = struct();
validModels = {};

for mi = 1:numel(cfg.modelNames)
    modelName = cfg.modelNames{mi};
    resultDir = fullfile(decodingDir, modelName);
    files = dir(fullfile(resultDir, cfg.filePattern));

    if isempty(files)
        fprintf('Skip %s: no files found.\n', modelName);
        continue
    end

    loadCfg = struct();
    loadCfg.metric = cfg.metric;
    loadCfg.useDiagonal = true;
    loadCfg.filePattern = cfg.filePattern;
    loadCfg.resultVarName = modelName;

    [diagData, times, usedFiles] = extract_decoding_timeseries(resultDir, loadCfg);

    loadCfg.metric = cfg.shuffleMetric;
    [diagShuffle, ~, shuffleFiles] = extract_decoding_timeseries(resultDir, loadCfg);

    D.(modelName).diag = diagData;
    D.(modelName).shuffle = diagShuffle;
    D.(modelName).files = usedFiles;
    D.(modelName).shuffleFiles = shuffleFiles;
    D.(modelName).color = colors(mi,:);

    validModels{end+1} = modelName; %#ok<SAGROW>

    fprintf('Loaded %s: nSub = %d, nTime = %d\n', ...
        modelName, size(diagData,1), size(diagData,2));
end

if isempty(validModels)
    error('No valid decoding results found in %s.', decodingDir);
end

delayIdx = times >= cfg.delayWindow(1);
if isfinite(cfg.delayWindow(2))
    delayIdx = delayIdx & times <= cfg.delayWindow(2);
end

%% ===================== 1) each metric vs shuffle over time =====================
if cfg.doAgainstShuffleTime
    statAgainstShuffle = struct();

    nModel = numel(validModels);
    [nRow, nCol] = subplot_grid(nModel);

    fig = figure('Color','w','Position',[100 100 1200 780]);

    for mi = 1:nModel
        modelName = validModels{mi};
        dataDiff = D.(modelName).diag - D.(modelName).shuffle;

        statCfg = struct();
        statCfg.null = 0;
        statCfg.nPerm = cfg.nPerm1D;
        statCfg.tail = 'right';
        statCfg.clusterAlpha = cfg.clusterAlpha;
        statCfg.alpha = cfg.alpha;
        statCfg.randomSeed = cfg.randomSeed;
        statCfg.verbose = false;

        statAgainstShuffle.(modelName) = cluster_perm_1d_timeseries(dataDiff, times, statCfg);

        subplot(nRow, nCol, mi); hold on;

        y = mean(D.(modelName).diag, 1, 'omitnan');
        e = sem_rows(D.(modelName).diag);
        ys = mean(D.(modelName).shuffle, 1, 'omitnan');
        es = sem_rows(D.(modelName).shuffle);
        c = D.(modelName).color;

        patch([times fliplr(times)], [y+e fliplr(y-e)], c, ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none');
        plot(times, y, 'LineWidth', 2.2, 'Color', c);

        patch([times fliplr(times)], [ys+es fliplr(ys-es)], [0.4 0.4 0.4], ...
            'FaceAlpha', 0.12, 'EdgeColor', 'none');
        plot(times, ys, '--', 'LineWidth', 1.8, 'Color', [0.35 0.35 0.35]);

        yline(cfg.chance, ':k');
        add_event_lines(cfg);
        plot_sig_bar(gca, times, statAgainstShuffle.(modelName).significantMask);

        xlabel('Time (ms)');
        ylabel(cfg.metric);
        title(sprintf('%s vs shuffle', modelName));
        box off; grid on;
    end

    savefig(fig, fullfile(saveDir, sprintf('%s_againstShuffle_time.fig', cfg.metric)));
    print(fig, fullfile(saveDir, sprintf('%s_againstShuffle_time.png', cfg.metric)), '-dpng', '-r300');

    save(fullfile(saveDir, sprintf('%s_againstShuffle_time_stats.mat', cfg.metric)), ...
        'statAgainstShuffle', 'cfg', 'times');
end

%% ===================== 2) maintenance-period summary: metric - shuffle =====================
cfg.multCompMethod = 'holm';       % 'holm', 'fdr', 'bonferroni', or 'none'
cfg.showNonSigPairBars = false;    % false = only draw corrected significant pairwise bars

if cfg.doDelaySummary

    [DelayStats, PairDelayStats, delayDelta] = make_delay_delta_summary(D, validModels, cfg, delayIdx);

    meanDelta = nan(numel(validModels),1);
    semDelta  = nan(numel(validModels),1);

    for mi = 1:numel(validModels)
        modelName = validModels{mi};
        x = delayDelta.(modelName);

        meanDelta(mi) = mean(x, 'omitnan');
        semDelta(mi)  = std(x, 0, 'omitnan') ./ sqrt(sum(~isnan(x)));
    end

    fig = figure('Color','w','Position',[100 100 980 580]); hold on;

    rng(cfg.randomSeed);

    for mi = 1:numel(validModels)
        modelName = validModels{mi};
        x = delayDelta.(modelName);

        bar(mi, meanDelta(mi), 0.65, ...
            'FaceColor', D.(modelName).color, ...
            'EdgeColor', 'none');

        errorbar(mi, meanDelta(mi), semDelta(mi), ...
            'k.', 'LineWidth', 1.3);

        jitter = 0.07 * randn(size(x));
        plot(mi + jitter, x, 'o', ...
            'MarkerSize', 4.5, ...
            'MarkerFaceColor', [0.85 0.85 0.85], ...
            'MarkerEdgeColor', [0.25 0.25 0.25], ...
            'LineWidth', 0.7);
    end

    yline(0, ':k', 'LineWidth', 1.1);

    set(gca, ...
        'XTick', 1:numel(validModels), ...
        'XTickLabel', validModels, ...
        'TickDir', 'out');

    xlim([0.5 numel(validModels)+0.5]);

    ylabel(sprintf('Mean %s - shuffle', cfg.metric));
    title(sprintf('Maintenance-period decoding evidence: %s - shuffle', cfg.metric));

    box off; grid on;

    add_pairwise_sig_bars(gca, PairDelayStats, validModels, cfg);

    savefig(fig, fullfile(saveDir, sprintf('%s_delayEvidence.fig', cfg.metric)));
    print(fig, fullfile(saveDir, sprintf('%s_delayEvidence.png', cfg.metric)), '-dpng', '-r300');

    writetable(DelayStats, fullfile(saveDir, sprintf('%s_delayEvidence_vs_zero.csv', cfg.metric)));
    writetable(PairDelayStats, fullfile(saveDir, sprintf('%s_delayEvidence_pairwise.csv', cfg.metric)));

    save(fullfile(saveDir, sprintf('%s_delayEvidence_stats.mat', cfg.metric)), ...
        'DelayStats', 'PairDelayStats', 'delayDelta', 'cfg');
end

%% ===================== 3) pairwise time-resolved contrasts =====================
if cfg.doPairwiseTime
    statPairTime = struct();

    pairs = filter_pairs(cfg.pairList, validModels);
    [nRow, nCol] = subplot_grid(size(pairs,1));

    fig = figure('Color','w','Position',[100 100 1250 760]);

    for pi = 1:size(pairs,1)
        A = pairs{pi,1};
        B = pairs{pi,2};

        [XA, XB] = align_by_files(D.(A).diag, D.(A).files, D.(B).diag, D.(B).files);
        dataDiff = XA - XB;

        statCfg = struct();
        statCfg.null = 0;
        statCfg.nPerm = cfg.nPerm1D;
        statCfg.tail = 'two';
        statCfg.clusterAlpha = cfg.clusterAlpha;
        statCfg.alpha = cfg.alpha;
        statCfg.randomSeed = cfg.randomSeed;
        statCfg.verbose = false;

        statPairTime.(sprintf('%s_minus_%s', A, B)) = ...
            cluster_perm_1d_timeseries(dataDiff, times, statCfg);

        subplot(nRow, nCol, pi); hold on;

        y = mean(dataDiff, 1, 'omitnan');
        e = sem_rows(dataDiff);

        patch([times fliplr(times)], [y+e fliplr(y-e)], [0.25 0.25 0.25], ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none');
        plot(times, y, 'k', 'LineWidth', 2.1);

        yline(0, ':k');
        add_event_lines(cfg);
        plot_sig_bar(gca, times, statPairTime.(sprintf('%s_minus_%s', A, B)).significantMask);

        xlabel('Time (ms)');
        ylabel(sprintf('%s - %s', A, B));
        title(sprintf('%s minus %s', A, B));
        box off; grid on;
    end

    savefig(fig, fullfile(saveDir, sprintf('%s_pairwise_time.fig', cfg.metric)));
    print(fig, fullfile(saveDir, sprintf('%s_pairwise_time.png', cfg.metric)), '-dpng', '-r300');

    save(fullfile(saveDir, sprintf('%s_pairwise_time_stats.mat', cfg.metric)), ...
        'statPairTime', 'cfg', 'times');
end

%% ===================== 4) time-generalization: model vs shuffle =====================
if cfg.doTimeGeneralization
    statTG = struct();
    tgPlotModels = {};
    tgTimesAll = [];
    maxAbsVal = 0;

    for mi = 1:numel(validModels)
        modelName = validModels{mi};
        resultDir = fullfile(decodingDir, modelName);

        [matData, tgTimes, matFiles] = load_group_matrix(resultDir, modelName, cfg.metric, cfg.filePattern);
        [matShuffle, ~, shuffleFiles] = load_group_matrix(resultDir, modelName, cfg.shuffleMetric, cfg.filePattern);

        [matData, matShuffle] = align_by_files_3d(matData, matFiles, matShuffle, shuffleFiles);
        matDiff = matData - matShuffle;

        if ~has_offdiag_values(matDiff)
            fprintf('Skip TG for %s: matrix only has diagonal values.\n', modelName);
            continue
        end

        statCfg = struct();
        statCfg.null = 0;
        statCfg.nPerm = cfg.nPerm2D;
        statCfg.tail = 'right';
        statCfg.clusterAlpha = cfg.clusterAlpha;
        statCfg.alpha = cfg.alpha;
        statCfg.randomSeed = cfg.randomSeed;
        statCfg.verbose = false;

        statTG.(modelName) = cluster_perm_2d_matrix(matDiff, tgTimes, tgTimes, statCfg);

        tgPlotModels{end+1} = modelName; %#ok<SAGROW>
        tgTimesAll = tgTimes;

        maxAbsVal = max(maxAbsVal, max(abs(statTG.(modelName).meanDiff(:)), [], 'omitnan'));
    end

    if ~isempty(tgPlotModels)
        [nRow, nCol] = subplot_grid(numel(tgPlotModels));

        fig = figure('Color','w','Position',[80 80 1250 850]);

        for mi = 1:numel(tgPlotModels)
            modelName = tgPlotModels{mi};

            subplot(nRow, nCol, mi);

            imagesc(tgTimesAll, tgTimesAll, statTG.(modelName).meanDiff');
            axis xy; hold on;

            if maxAbsVal > 0
                clim([-maxAbsVal maxAbsVal]);
            end

            colorbar;
            plot(tgTimesAll, tgTimesAll, '--k', 'LineWidth', 1);

            contour(tgTimesAll, tgTimesAll, ...
                statTG.(modelName).significantMask', [1 1], ...
                'k', 'LineWidth', 1.3);

            xlabel('Train time (ms)');
            ylabel('Test time (ms)');
            title(sprintf('%s: %s - shuffle', modelName, cfg.metric));

            add_event_lines_tg(cfg);
            box off;
        end

        sgtitle(sprintf('Time-generalization: %s - shuffle', cfg.metric), ...
            'FontWeight','bold');
        % 
        savefig(fig, fullfile(saveDir, sprintf('%s_TG_againstShuffle_subplot.fig', cfg.metric)));
        print(fig, fullfile(saveDir, sprintf('%s_TG_againstShuffle_subplot.png', cfg.metric)), '-dpng', '-r300');
    end

    save(fullfile(saveDir, sprintf('%s_TG_againstShuffle_stats.mat', cfg.metric)), ...
        'statTG', 'cfg');
end

%% ===================== 5) time-generalization: pairwise contrasts =====================
if cfg.doTGPairwise
    statTGPair = struct();
    pairs = filter_pairs(cfg.pairList, validModels);

    tgPlotPairs = {};
    tgTimesAll = [];
    maxAbsVal = 0;

    for pi = 1:size(pairs,1)
        A = pairs{pi,1};
        B = pairs{pi,2};

        [matA, tgTimes, filesA] = load_group_matrix(fullfile(decodingDir, A), A, cfg.metric, cfg.filePattern);
        [matB, ~, filesB] = load_group_matrix(fullfile(decodingDir, B), B, cfg.metric, cfg.filePattern);

        [matA, matB] = align_by_files_3d(matA, filesA, matB, filesB);
        matDiff = matA - matB;

        if ~has_offdiag_values(matDiff)
            fprintf('Skip TG pair %s - %s: matrix only has diagonal values.\n', A, B);
            continue
        end

        statCfg = struct();
        statCfg.null = 0;
        statCfg.nPerm = cfg.nPerm2D;
        statCfg.tail = 'two';
        statCfg.clusterAlpha = cfg.clusterAlpha;
        statCfg.alpha = cfg.alpha;
        statCfg.randomSeed = cfg.randomSeed;
        statCfg.verbose = false;

        statName = sprintf('%s_minus_%s', A, B);
        statTGPair.(statName) = cluster_perm_2d_matrix(matDiff, tgTimes, tgTimes, statCfg);

        tgPlotPairs(end+1,:) = {A, B, statName}; %#ok<SAGROW>
        tgTimesAll = tgTimes;

        maxAbsVal = max(maxAbsVal, max(abs(statTGPair.(statName).meanDiff(:)), [], 'omitnan'));
    end

    if ~isempty(tgPlotPairs)
        [nRow, nCol] = subplot_grid(size(tgPlotPairs,1));

        fig = figure('Color','w','Position',[80 80 1250 850]);

        for pi = 1:size(tgPlotPairs,1)
            A = tgPlotPairs{pi,1};
            B = tgPlotPairs{pi,2};
            statName = tgPlotPairs{pi,3};

            subplot(nRow, nCol, pi);

            imagesc(tgTimesAll, tgTimesAll, statTGPair.(statName).meanDiff');
            axis xy; hold on;

            if maxAbsVal > 0
                clim([-maxAbsVal maxAbsVal]);
            end

            colorbar;
            plot(tgTimesAll, tgTimesAll, '--k', 'LineWidth', 1);

            contour(tgTimesAll, tgTimesAll, ...
                statTGPair.(statName).significantMask', [1 1], ...
                'k', 'LineWidth', 1.3);

            xlabel('Train time (ms)');
            ylabel('Test time (ms)');
            title(sprintf('%s - %s', A, B));

            add_event_lines_tg(cfg);
            box off;
        end

        sgtitle(sprintf('Time-generalization pairwise contrasts: %s', cfg.metric), ...
            'FontWeight','bold');

        savefig(fig, fullfile(saveDir, sprintf('%s_TG_pairwise_subplot.fig', cfg.metric)));
        print(fig, fullfile(saveDir, sprintf('%s_TG_pairwise_subplot.png', cfg.metric)), '-dpng', '-r300');
    end

    save(fullfile(saveDir, sprintf('%s_TG_pairwise_stats.mat', cfg.metric)), ...
        'statTGPair', 'cfg');
end

%% ========================================================================
function e = sem_rows(X)
n = sum(~isnan(X), 1);
e = std(X, 0, 1, 'omitnan') ./ sqrt(n);
end

%% ========================================================================
function [nRow, nCol] = subplot_grid(n)
nCol = ceil(sqrt(n));
nRow = ceil(n / nCol);
end

%% ========================================================================
function add_event_lines(cfg)
for i = 1:numel(cfg.eventLines)
    if i <= numel(cfg.eventLineLabels)
        xline(cfg.eventLines(i), '--k', cfg.eventLineLabels{i}, 'HandleVisibility','off');
    else
        xline(cfg.eventLines(i), '--k', 'HandleVisibility','off');
    end
end
end

%% ========================================================================
function add_event_lines_tg(cfg)
for i = 1:numel(cfg.eventLines)
    xline(cfg.eventLines(i), '--k', 'HandleVisibility','off');
    yline(cfg.eventLines(i), '--k', 'HandleVisibility','off');
end
end

%% ========================================================================
function plot_sig_bar(ax, times, sigMask)
if ~any(sigMask), return; end

yl = ylim(ax);
yBar = yl(1) + 0.04 * range(yl);
idx = find(sigMask(:)');

breaks = [0 find(diff(idx) > 1) numel(idx)];
for bi = 1:numel(breaks)-1
    seg = idx(breaks(bi)+1:breaks(bi+1));
    plot(ax, times(seg), yBar * ones(size(seg)), 'k-', ...
        'LineWidth', 4, 'HandleVisibility','off');
end
end

%% ========================================================================
function pairs = filter_pairs(pairList, validModels)
keep = false(size(pairList,1),1);
for i = 1:size(pairList,1)
    keep(i) = ismember(pairList{i,1}, validModels) && ismember(pairList{i,2}, validModels);
end
pairs = pairList(keep,:);
end

%% ========================================================================
function [T1, T2] = align_by_files(T1, files1, T2, files2)
name1 = basename_list(files1);
name2 = basename_list(files2);
[~, ia, ib] = intersect(name1, name2, 'stable');
T1 = T1(ia,:);
T2 = T2(ib,:);
end

%% ========================================================================
function [M1, M2] = align_by_files_3d(M1, files1, M2, files2)
name1 = basename_list(files1);
name2 = basename_list(files2);
[~, ia, ib] = intersect(name1, name2, 'stable');
M1 = M1(ia,:,:);
M2 = M2(ib,:,:);
end

%% ========================================================================
function names = basename_list(files)
names = cell(numel(files),1);
for i = 1:numel(files)
    [~, names{i}] = fileparts(files{i});
end
end

%% ========================================================================
function [DelayStats, PairDelayStats, delayDelta] = make_delay_delta_summary(D, validModels, cfg, delayIdx)

delayDelta = struct();
delayNames = struct();

rows = cell(0,8);

for mi = 1:numel(validModels)
    modelName = validModels{mi};

    [x, names] = get_delay_delta(D.(modelName), delayIdx);

    delayDelta.(modelName) = x;
    delayNames.(modelName) = names;

    xValid = x(~isnan(x));

    [~, p, ~, st] = ttest(xValid, 0, 'Tail', 'right');

    dz = mean(xValid, 'omitnan') ./ std(xValid, 0, 'omitnan');
    semX = std(xValid, 0, 'omitnan') ./ sqrt(numel(xValid));

    rows(end+1,:) = {modelName, numel(xValid), ...
        mean(xValid,'omitnan'), semX, p, st.tstat, st.df, dz}; %#ok<AGROW>
end

DelayStats = cell2table(rows, 'VariableNames', ...
    {'Model','N','MeanDelta','SEM','P_right','T','DF','CohenDz'});

DelayStats.P_right_corr = correct_pvalues( ...
    DelayStats.P_right, cfg.multCompMethod, cfg.alpha);

DelayStats.P_right_sig_corr = DelayStats.P_right_corr < cfg.alpha;


pairRows = cell(0,10);
pairs = filter_pairs(cfg.pairList, validModels);

for pi = 1:size(pairs,1)
    A = pairs{pi,1};
    B = pairs{pi,2};

    xa = delayDelta.(A);
    xb = delayDelta.(B);

    nameA = delayNames.(A);
    nameB = delayNames.(B);

    [~, ia, ib] = intersect(nameA, nameB, 'stable');

    xa = xa(ia);
    xb = xb(ib);

    keep = ~isnan(xa) & ~isnan(xb);
    xa = xa(keep);
    xb = xb(keep);

    [~, p, ~, st] = ttest(xa, xb, 'Tail', 'both');

    d = xa - xb;
    dz = mean(d, 'omitnan') ./ std(d, 0, 'omitnan');

    pairRows(end+1,:) = {A, B, numel(d), ...
        mean(xa,'omitnan'), mean(xb,'omitnan'), mean(d,'omitnan'), ...
        p, st.tstat, st.df, dz}; %#ok<AGROW>
end

PairDelayStats = cell2table(pairRows, 'VariableNames', ...
    {'ModelA','ModelB','N','MeanDeltaA','MeanDeltaB','MeanDiff','P_two','T','DF','CohenDz'});

PairDelayStats.P_two_corr = correct_pvalues( ...
    PairDelayStats.P_two, cfg.multCompMethod, cfg.alpha);

PairDelayStats.P_two_sig_corr = PairDelayStats.P_two_corr < cfg.alpha;

end


%% ========================================================================
function [x, names] = get_delay_delta(M, delayIdx)

name1 = basename_list(M.files);
name2 = basename_list(M.shuffleFiles);

[names, ia, ib] = intersect(name1, name2, 'stable');

deltaTS = M.diag(ia,:) - M.shuffle(ib,:);
x = mean(deltaTS(:,delayIdx), 2, 'omitnan');

end


%% ========================================================================
function add_pairwise_sig_bars(ax, PairDelayStats, validModels, cfg)

if isempty(PairDelayStats) || height(PairDelayStats) == 0
    return
end

if isfield(cfg, 'showNonSigPairBars')
    showNonSig = cfg.showNonSigPairBars;
else
    showNonSig = false;
end

if isfield(cfg, 'alpha')
    alpha = cfg.alpha;
else
    alpha = 0.05;
end

if ismember('P_two_corr', PairDelayStats.Properties.VariableNames)
    pName = 'P_two_corr';
else
    pName = 'P_two';
end

barInfo = [];

for i = 1:height(PairDelayStats)

    A = PairDelayStats.ModelA{i};
    B = PairDelayStats.ModelB{i};

    x1 = find(strcmp(validModels, A));
    x2 = find(strcmp(validModels, B));

    if isempty(x1) || isempty(x2)
        continue
    end

    p = PairDelayStats.(pName)(i);

    if isnan(p)
        continue
    end

    isSig = p < alpha;

    if ~isSig && ~showNonSig
        continue
    end

    if x1 > x2
        tmp = x1;
        x1 = x2;
        x2 = tmp;
    end

    barInfo(end+1,:) = [x1, x2, p, abs(x2-x1)]; %#ok<AGROW>
end

if isempty(barInfo)
    return
end

[~, ord] = sort(barInfo(:,4), 'ascend');
barInfo = barInfo(ord,:);

yl = ylim(ax);
yr = range(yl);

if yr == 0
    yr = 1;
end

yStart = yl(2) + 0.07 * yr;
barH   = 0.025 * yr;
stepH  = 0.10 * yr;

ylim(ax, [yl(1), yStart + stepH * size(barInfo,1) + 0.12 * yr]);

for bi = 1:size(barInfo,1)

    x1 = barInfo(bi,1);
    x2 = barInfo(bi,2);
    p  = barInfo(bi,3);

    y = yStart + (bi-1) * stepH;

    plot(ax, [x1 x1 x2 x2], [y y+barH y+barH y], ...
        'k-', 'LineWidth', 1.2, 'HandleVisibility','off');

    text(ax, mean([x1 x2]), y + barH + 0.008 * yr, p_to_stars(p), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 11, ...
        'FontWeight', 'bold');
end

end


%% ========================================================================
function s = p_to_stars(p)

if p < 0.001
    s = '***';
elseif p < 0.01
    s = '**';
elseif p < 0.05
    s = '*';
else
    s = 'n.s.';
end

end
%% ========================================================================
function [mats, times, usedFiles] = load_group_matrix(resultDir, varName, metricName, filePattern)
files = dir(fullfile(resultDir, filePattern));
mats = [];
times = [];
usedFiles = {};

for fi = 1:numel(files)
    fpath = fullfile(files(fi).folder, files(fi).name);
    S = load(fpath);

    if isfield(S, varName)
        R = S.(varName);
    else
        R = pick_result_struct(S, metricName);
    end

    if isempty(R) || ~isfield(R, metricName)
        continue
    end

    M = R.(metricName);

    if isvector(M)
        M2 = nan(numel(M), numel(M));
        M2(1:numel(M)+1:end) = M(:);
        M = M2;
    end

    if isempty(mats)
        mats = nan(numel(files), size(M,1), size(M,2));
        if isfield(R, 'times') && ~isempty(R.times)
            times = R.times(:)';
        else
            times = 1:size(M,1);
        end
    end

    mats(fi,:,:) = M; %#ok<AGROW>
    usedFiles{end+1,1} = fpath; %#ok<AGROW>
end

mats = mats(1:numel(usedFiles),:,:);
end

%% ========================================================================
function R = pick_result_struct(S, metricName)
R = [];
fn = fieldnames(S);
for i = 1:numel(fn)
    if isstruct(S.(fn{i})) && isfield(S.(fn{i}), metricName)
        R = S.(fn{i});
        return
    end
end
end

%% ========================================================================
function tf = has_offdiag_values(M)
Mmean = squeeze(mean(M, 1, 'omitnan'));
if size(Mmean,1) ~= size(Mmean,2)
    tf = true;
    return
end
offMask = ~eye(size(Mmean,1));
tf = any(~isnan(Mmean(offMask)));
end

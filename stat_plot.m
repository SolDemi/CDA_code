%% ===================== basic config =====================

clear; clc;
cfg = struct();

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);

sideFeatures = {'VoltageRawLR', 'AlphaRawLR', 'VoltageLminusR', 'AlphaLminusR', 'GlobalAlphaMean'};
loadFeatures = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
analysisNames = {'loadDecoding', 'sideDecoding', 'loadWithinSide', 'loadSideBalanced', 'loadCrossSide'};

decodefolder = 'LDA';   % 'SVM' or 'LDA'
maindir = fullfile(projectRoot, 'data1');

analysis = 3;           % 1=loadDecoding, 2=sideDecoding, 3=loadWithinSide

switch analysis
    case 1
        decodingDir = fullfile(maindir, ['decoding_' decodefolder]);
        cfg.modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};

        cfg.pairList = {
            'CDA',   'Alpha'
            'NoPCA', 'CDA'
            'NoPCA', 'Alpha'
            'PCA',   'NoPCA'
            'CDA', 'GlobalAlpha'
            % 'GlobalAlphaMean', 'CDA'
            % 'GlobalAlphaMean', 'GlobalAlpha'
            'PCA', 'GlobalAlpha'
            };
    case 2
        decodingDir = fullfile(maindir, ['decoding_' decodefolder '_spatialControl'],analysisNames{analysis});
        cfg.modelNames = sideFeatures;

        cfg.pairList = {
            'VoltageRawLR',   'AlphaRawLR'
            'VoltageLminusR', 'AlphaLminusR'
            'AlphaRawLR',     'GlobalAlphaMean'
            };

    otherwise
        decodingDir = fullfile(maindir, ['decoding_' decodefolder '_spatialControl'],analysisNames{analysis});
        cfg.modelNames = loadFeatures;
        cfg.pairList = {
            'CDA',   'Alpha'
            'NoPCA', 'CDA'
            'NoPCA', 'Alpha'
            'PCA',   'NoPCA'
            % 'PCA',   'CDA'
            'CDA', 'GlobalAlpha'
            % 'GlobalAlphaMean', 'CDA'
            % 'GlobalAlphaMean', 'GlobalAlpha'
            'PCA', 'GlobalAlpha'
            };
        
end
saveDir = fullfile(decodingDir, 'GroupStats');

if ~isfolder(saveDir), mkdir(saveDir); end

cfg.metric = 'AUC';                  % 'AUC' or 'predictAcc'
cfg.shuffleMetric = 'AUCShuffle';    % 'AUCShuffle' or 'predictAccShuffle'
cfg.filePattern = '*.mat';

[~, dataName] = fileparts(maindir);
switch lower(dataName)
    case 'data1'
        [cfg.includeSubjectIds, cfg.subjectInclusion] = data1_decoding_subjects(fullfile(maindir, 'data'), 75);
    case 'data2'
        [cfg.includeSubjectIds, cfg.subjectInclusion] = data2_decoding_subjects(fullfile(maindir, 'cda_alpha'), 160);
    otherwise
        cfg.includeSubjectIds = [];
        cfg.subjectInclusion = table();
end
cfg.excludeSubjectIds = [];
if ~isempty(cfg.includeSubjectIds)
    fprintf('Subject inclusion for %s: n = %d\n', dataName, numel(cfg.includeSubjectIds));
end

cfg.chance = 0.5;
cfg.delayWindow = [250 inf];         % maintenance summary: 250 ms to the end

if strcmpi(dataName, 'data1') 
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
cfg.doComponentComparisonTime = true;
cfg.doDelaySummary = true;
cfg.doPairwiseTime = true;
cfg.doTimeGeneralization = true;
cfg.doTGPairwise = true;
cfg.showAgainstShuffleInComponentPlot = true;
cfg.showPairwiseInComponentPlot = true;
cfg.useChanceIfNoShuffle = true;     % when shuffle metric is absent, test against cfg.chance
cfg.colorLimitBounds = [0 1];
cfg.doData1CDASetSizeTG = strcmpi(dataName, 'data1') && analysis == 3 && strcmpi(decodefolder, 'LDA');
cfg.runOnlyData1CDASetSizeTG = strcmpi(strtrim(char(getenv('STAT_PLOT_ONLY_DATA1_CDA_SET_SIZE'))), '1');



colors = lines(numel(cfg.modelNames));

%% ===================== data1 CDA-only ss1-vs3 / ss1-vs6 maintenance TG plots =====================
if cfg.doData1CDASetSizeTG
    cdaSetCfg = cfg;
    cdaSetCfg.comparisons = { ...
        'setsize1_vs3', fullfile(maindir, 'decoding_LDA_spatialControl', 'loadWithinSide_setsize1_vs3', 'CDA'), 'SS1 vs SS3'; ...
        'setsize1_vs6', fullfile(maindir, 'decoding_LDA_spatialControl', 'loadWithinSide', 'CDA'), 'SS1 vs SS6'};
    cdaSetCfg.modelName = 'CDA';
    cdaSetCfg.metric = 'AUC';
    cdaSetCfg.figureDir = fullfile(saveDir, 'figures');
    if ~isfolder(cdaSetCfg.figureDir), mkdir(cdaSetCfg.figureDir); end

    cdaSetData = struct();
    commonMaintTimes = [];
    maxDelta = 0;
    for ci = 1:size(cdaSetCfg.comparisons, 1)
        comparisonName = cdaSetCfg.comparisons{ci, 1};
        comparisonDir = cdaSetCfg.comparisons{ci, 2};
        comparisonLabel = cdaSetCfg.comparisons{ci, 3};

        [matData, tgTimes, matFiles] = load_group_matrix(comparisonDir, cdaSetCfg.modelName, ...
            cdaSetCfg.metric, cdaSetCfg.filePattern, cdaSetCfg.includeSubjectIds, cdaSetCfg.excludeSubjectIds);
        if isempty(matData)
            warning('No CDA %s matrices found in %s.', comparisonName, comparisonDir);
            continue;
        end

        maintIdx = tgTimes >= cdaSetCfg.delayWindow(1);
        if isfinite(cdaSetCfg.delayWindow(2))
            maintIdx = maintIdx & tgTimes <= cdaSetCfg.delayWindow(2);
        end
        if ~any(maintIdx)
            error('Maintenance window does not overlap time axis for %s.', comparisonName);
        end

        matMaint = matData(:, maintIdx, maintIdx);
        maintTimes = tgTimes(maintIdx);
        if isempty(commonMaintTimes)
            commonMaintTimes = maintTimes(:)';
        elseif numel(commonMaintTimes) ~= numel(maintTimes) || any(abs(commonMaintTimes(:) - maintTimes(:)) > 1e-9)
            error('Maintenance time axis differs between CDA set-size comparisons.');
        end

        statCfg = struct();
        statCfg.null = cdaSetCfg.chance;
        statCfg.nPerm = cdaSetCfg.nPerm2D;
        statCfg.tail = 'right';
        statCfg.clusterAlpha = cdaSetCfg.clusterAlpha;
        statCfg.alpha = cdaSetCfg.alpha;
        statCfg.randomSeed = cdaSetCfg.randomSeed;
        statCfg.verbose = false;
        stat = cluster_perm_2d_matrix(matMaint, maintTimes, maintTimes, statCfg);

        cdaSetData.(comparisonName).label = comparisonLabel;
        cdaSetData.(comparisonName).dir = comparisonDir;
        cdaSetData.(comparisonName).data = matMaint;
        cdaSetData.(comparisonName).times = maintTimes;
        cdaSetData.(comparisonName).files = matFiles;
        cdaSetData.(comparisonName).stat = stat;

        meanMat = squeeze(mean(matMaint, 1, 'omitnan'));
        maxDelta = max(maxDelta, max(abs(meanMat(:) - cdaSetCfg.chance), [], 'omitnan'));
        fprintf('Loaded data1 CDA %s maintenance TG: n=%d, matrix=%dx%d\n', ...
            comparisonLabel, size(matMaint, 1), size(matMaint, 2), size(matMaint, 3));
    end

    comparisonNames = fieldnames(cdaSetData);
    if ~isempty(comparisonNames)
        if isempty(maxDelta) || isnan(maxDelta) || maxDelta == 0
            maxDelta = 0.02;
        end
        climNow = [max(cdaSetCfg.colorLimitBounds(1), cdaSetCfg.chance - maxDelta), ...
            min(cdaSetCfg.colorLimitBounds(2), cdaSetCfg.chance + maxDelta)];
        if climNow(1) >= climNow(2)
            climNow = cdaSetCfg.chance + [-0.02 0.02];
        end

        nComparison = numel(comparisonNames);
        nTime = numel(commonMaintTimes);
        lowStart = min(commonMaintTimes);
        lowEnd = max(commonMaintTimes);
        lowSpan = lowEnd - lowStart;
        if lowSpan <= 0
            lowSpan = 1;
        end
        stackMat = nan(nComparison * nTime, nTime);
        yTicks = nan(1, nComparison);
        yLabels = cell(1, nComparison);

        for ci = 1:numel(comparisonNames)
            comparisonName = comparisonNames{ci};
            Dnow = cdaSetData.(comparisonName);
            meanMat = squeeze(mean(Dnow.data, 1, 'omitnan'));
            blockFromBottom = nComparison - ci + 1;
            rowIdx = (blockFromBottom - 1) * nTime + (1:nTime);
            stackMat(rowIdx,:) = meanMat';
            blockBase = (blockFromBottom - 1) * lowSpan;
            yTicks(blockFromBottom) = blockBase + lowSpan / 2;
            yLabels{blockFromBottom} = Dnow.label;
        end

        yTotal = nComparison * lowSpan;
        fig = figure('Color', 'w', 'Position', [70 60 1200 900]);
        ax = axes('Parent', fig);
        if isprop(ax, 'Toolbar')
            ax.Toolbar.Visible = 'off';
        end
        imagesc(ax, [lowStart lowEnd], [0 yTotal], stackMat);
        set(ax, 'YDir', 'normal');
        hold(ax, 'on');
        colormap(ax, parula);
        colorbar(ax);
        clim(ax, climNow);

        for ci = 1:numel(comparisonNames)
            comparisonName = comparisonNames{ci};
            Dnow = cdaSetData.(comparisonName);
            blockFromBottom = nComparison - ci + 1;
            blockBase = (blockFromBottom - 1) * lowSpan;
            yAxis = blockBase + Dnow.times(:)' - lowStart;
            plot(ax, Dnow.times, yAxis, '--k', 'LineWidth', 1);
            if any(Dnow.stat.significantMask(:))
                contour(ax, Dnow.times, yAxis, double(Dnow.stat.significantMask'), [1 1], ...
                    'Color', 'k', 'LineWidth', 1.2);
            end
            yline(ax, blockBase, '-k', 'HandleVisibility', 'off');
            yline(ax, blockBase + lowSpan, '-k', 'HandleVisibility', 'off');
        end

        xline(ax, cdaSetCfg.delayWindow(1), ':k', 'HandleVisibility', 'off');
        xlim(ax, [lowStart lowEnd]);
        ylim(ax, [0 yTotal]);
        daspect(ax, [1 1 1]);
        set(ax, 'YTick', yTicks, 'YTickLabel', yLabels, 'TickDir', 'out', 'FontSize', 11);
        xlabel(ax, 'Train time (ms)');
        ylabel(ax, 'Comparison / test time');
        title(ax, sprintf('data1 CDA maintenance time-generalization, %s', cdaSetCfg.metric), ...
            'FontWeight', 'bold');
        box(ax, 'off');

        savefig(fig, fullfile(cdaSetCfg.figureDir, sprintf('%s_data1_CDA_ss1vs3_ss1vs6_maint_TG_heatmaps.fig', cdaSetCfg.metric)));
        print(fig, fullfile(cdaSetCfg.figureDir, sprintf('%s_data1_CDA_ss1vs3_ss1vs6_maint_TG_heatmaps.png', cdaSetCfg.metric)), '-dpng', '-r300');
        close(fig);

        fig = figure('Color', 'w', 'Position', [100 100 780 480]);
        ax = axes('Parent', fig);
        if isprop(ax, 'Toolbar')
            ax.Toolbar.Visible = 'off';
        end
        hold(ax, 'on');
        plotColors = lines(numel(comparisonNames));
        diagSummary = table();
        diagMeanData = struct();

        for ci = 1:numel(comparisonNames)
            comparisonName = comparisonNames{ci};
            Dnow = cdaSetData.(comparisonName);
            nSub = size(Dnow.data, 1);
            diagBySubject = nan(nSub, numel(Dnow.times));
            for si = 1:nSub
                subjMat = squeeze(Dnow.data(si,:,:));
                diagBySubject(si,:) = diag(subjMat)';
            end

            y = mean(diagBySubject, 1, 'omitnan');
            e = std(diagBySubject, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(diagBySubject), 1));
            patch(ax, [Dnow.times(:)' fliplr(Dnow.times(:)')], [y+e fliplr(y-e)], plotColors(ci,:), ...
                'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(ax, Dnow.times, y, 'LineWidth', 2.0, 'Color', plotColors(ci,:), ...
                'DisplayName', Dnow.label);

            Tdiag = table();
            Tdiag.Comparison = repmat({comparisonName}, numel(Dnow.times), 1);
            Tdiag.Label = repmat({Dnow.label}, numel(Dnow.times), 1);
            Tdiag.TimeMs = Dnow.times(:);
            Tdiag.NSubject = sum(~isnan(diagBySubject), 1)';
            Tdiag.MeanDiagAUC = y(:);
            Tdiag.SEMDiagAUC = e(:);
            diagSummary = [diagSummary; Tdiag]; %#ok<AGROW>

            diagMeanData.(comparisonName).label = Dnow.label;
            diagMeanData.(comparisonName).files = Dnow.files;
            diagMeanData.(comparisonName).names = basename_list(Dnow.files);
            diagMeanData.(comparisonName).values = mean(diagBySubject, 2, 'omitnan');
        end

        yline(ax, cdaSetCfg.chance, ':k', 'Chance', 'HandleVisibility', 'off');
        xline(ax, cdaSetCfg.delayWindow(1), ':k', 'HandleVisibility', 'off');
        xlabel(ax, 'Maintenance train=test time (ms)');
        ylabel(ax, sprintf('Diagonal %s', cdaSetCfg.metric));
        title(ax, 'data1 CDA maintenance diagonal decoding');
        legend(ax, 'Location', 'best');
        box(ax, 'off');
        grid(ax, 'on');

        savefig(fig, fullfile(cdaSetCfg.figureDir, sprintf('%s_data1_CDA_ss1vs3_ss1vs6_maint_diag.fig', cdaSetCfg.metric)));
        print(fig, fullfile(cdaSetCfg.figureDir, sprintf('%s_data1_CDA_ss1vs3_ss1vs6_maint_diag.png', cdaSetCfg.metric)), '-dpng', '-r300');
        close(fig);

        diagMeanSubjectTable = table();
        diagMeanStats = table();
        if numel(comparisonNames) == 2
            Dmean1 = diagMeanData.(comparisonNames{1});
            Dmean2 = diagMeanData.(comparisonNames{2});
            [commonNames, ia, ib] = intersect(Dmean1.names, Dmean2.names, 'stable');
            y1 = Dmean1.values(ia);
            y2 = Dmean2.values(ib);
            keep = ~isnan(y1) & ~isnan(y2);
            commonNames = commonNames(keep);
            y1 = y1(keep);
            y2 = y2(keep);
            nPair = numel(y1);

            if nPair > 1
                [~, pTwo, ciBoth, statsBoth] = ttest(y2, y1, 'Tail', 'both');
                [~, pRight] = ttest(y2, y1, 'Tail', 'right');
                diffVals = y2 - y1;
                meanVals = [mean(y1, 'omitnan'), mean(y2, 'omitnan')];
                semVals = [std(y1, 0, 'omitnan'), std(y2, 0, 'omitnan')] ./ sqrt(nPair);
                semDiff = std(diffVals, 0, 'omitnan') ./ sqrt(nPair);
                dz = mean(diffVals, 'omitnan') ./ std(diffVals, 0, 'omitnan');

                diagMeanSubjectTable = table(commonNames, y1, y2, diffVals, ...
                    'VariableNames', {'SubjectFile','SS1vsSS3','SS1vsSS6','Diff_SS1vsSS6_minus_SS1vsSS3'});
                diagMeanStats = table(nPair, meanVals(1), semVals(1), meanVals(2), semVals(2), ...
                    mean(diffVals, 'omitnan'), semDiff, pTwo, pRight, statsBoth.tstat, statsBoth.df, ...
                    ciBoth(1), ciBoth(2), dz, ...
                    'VariableNames', {'N','Mean_SS1vsSS3','SEM_SS1vsSS3','Mean_SS1vsSS6','SEM_SS1vsSS6', ...
                    'MeanDiff_SS1vsSS6_minus_SS1vsSS3','SEMDiff','P_two','P_right_SS1vsSS6_gt_SS1vsSS3', ...
                    'T','DF','CI95_low','CI95_high','CohenDz'});

                fprintf('data1 CDA mean maintenance diagonal paired t-test (%s - %s): N=%d, meanDiff=%.6f, t(%d)=%.3f, p_two=%.6f\n', ...
                    Dmean2.label, Dmean1.label, nPair, mean(diffVals, 'omitnan'), statsBoth.df, statsBoth.tstat, pTwo);

                fig = figure('Color', 'w', 'Position', [120 120 640 520]);
                ax = axes('Parent', fig);
                if isprop(ax, 'Toolbar')
                    ax.Toolbar.Visible = 'off';
                end
                hold(ax, 'on');
                xPair = 1:2;
                for si = 1:nPair
                    plot(ax, xPair, [y1(si) y2(si)], '-', 'Color', [0.75 0.75 0.75], ...
                        'LineWidth', 0.7, 'HandleVisibility', 'off');
                end
                b = bar(ax, xPair, meanVals, 0.55, 'FaceColor', 'flat', 'EdgeColor', 'none');
                b.CData = plotColors(1:2,:);
                errorbar(ax, xPair, meanVals, semVals, 'k', 'LineStyle', 'none', ...
                    'LineWidth', 1.3, 'CapSize', 12, 'HandleVisibility', 'off');
                scatter(ax, ones(nPair, 1), y1, 28, plotColors(1,:), 'filled', ...
                    'MarkerEdgeColor', 'w', 'LineWidth', 0.4, 'HandleVisibility', 'off');
                scatter(ax, 2 * ones(nPair, 1), y2, 28, plotColors(2,:), 'filled', ...
                    'MarkerEdgeColor', 'w', 'LineWidth', 0.4, 'HandleVisibility', 'off');
                yline(ax, cdaSetCfg.chance, ':k', 'Chance', 'HandleVisibility', 'off');

                yAll = [y1; y2; meanVals(:) + semVals(:); cdaSetCfg.chance];
                yMin = min(yAll, [], 'omitnan');
                yMax = max(yAll, [], 'omitnan');
                yRange = yMax - yMin;
                if yRange == 0
                    yRange = 0.02;
                end
                bracketY = yMax + 0.10 * yRange;
                bracketH = 0.04 * yRange;
                ylim(ax, [yMin - 0.12 * yRange, bracketY + 0.22 * yRange]);
                plot(ax, [1 1 2 2], [bracketY bracketY+bracketH bracketY+bracketH bracketY], ...
                    'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
                text(ax, 1.5, bracketY + bracketH + 0.02 * yRange, p_to_stars(pTwo), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', 12, 'FontWeight', 'bold');
                text(ax, 1.5, bracketY + bracketH + 0.10 * yRange, ...
                    sprintf('paired t(%d)=%.2f, p=%.4f', statsBoth.df, statsBoth.tstat, pTwo), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);

                set(ax, 'XTick', xPair, 'XTickLabel', {Dmean1.label, Dmean2.label}, ...
                    'TickDir', 'out', 'FontSize', 11);
                xlim(ax, [0.45 2.55]);
                ylabel(ax, sprintf('Mean maintenance diagonal %s', cdaSetCfg.metric));
                title(ax, 'data1 CDA maintenance-averaged diagonal decoding');
                box(ax, 'off');
                grid(ax, 'on');

                savefig(fig, fullfile(cdaSetCfg.figureDir, sprintf('%s_data1_CDA_ss1vs3_ss1vs6_maint_diag_mean_bar.fig', cdaSetCfg.metric)));
                print(fig, fullfile(cdaSetCfg.figureDir, sprintf('%s_data1_CDA_ss1vs3_ss1vs6_maint_diag_mean_bar.png', cdaSetCfg.metric)), '-dpng', '-r300');
                close(fig);
            end
        end

        save(fullfile(saveDir, sprintf('%s_data1_CDA_ss1vs3_ss1vs6_maint_TG_stats.mat', cdaSetCfg.metric)), ...
            'cdaSetData', 'diagSummary', 'diagMeanSubjectTable', 'diagMeanStats', 'cdaSetCfg', '-v7.3');
    end
end

if cfg.runOnlyData1CDASetSizeTG
    fprintf('STAT_PLOT_ONLY_DATA1_CDA_SET_SIZE=1, skipping the general stat_plot sections.\n');
    return
end

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
    loadCfg.includeSubjectIds = cfg.includeSubjectIds;
    loadCfg.excludeSubjectIds = cfg.excludeSubjectIds;

    [diagData, times, usedFiles] = extract_decoding_timeseries(resultDir, loadCfg);

    [diagShuffle, shuffleFiles, baselineInfo] = load_or_make_baseline_timeseries( ...
        resultDir, loadCfg, diagData, times, usedFiles, cfg);

    D.(modelName).diag = diagData;
    D.(modelName).shuffle = diagShuffle;
    D.(modelName).files = usedFiles;
    D.(modelName).shuffleFiles = shuffleFiles;
    D.(modelName).color = colors(mi,:);
    D.(modelName).baseline = baselineInfo;

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
statAgainstShuffle = struct();
if cfg.doAgainstShuffleTime || ...
        (cfg.doComponentComparisonTime && cfg.showAgainstShuffleInComponentPlot)
    statAgainstShuffle = make_against_shuffle_time_stats(D, validModels, times, cfg);
end

if cfg.doAgainstShuffleTime
    nModel = numel(validModels);
    [nRow, nCol] = subplot_grid(nModel);

    fig = figure('Color','w','Position',[100 100 1200 780]);

    for mi = 1:nModel
        modelName = validModels{mi};

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
        if isfield(statAgainstShuffle, modelName)
            plot_sig_bar(gca, times, statAgainstShuffle.(modelName).significantMask);
        end

        xlabel('Time (ms)');
        ylabel(cfg.metric);
        title(sprintf('%s vs %s', modelName, D.(modelName).baseline.label));
        box off; grid on;
    end

    savefig(fig, fullfile(saveDir, sprintf('%s_againstShuffle_time.fig', cfg.metric)));
    print(fig, fullfile(saveDir, sprintf('%s_againstShuffle_time.png', cfg.metric)), '-dpng', '-r300');

    save(fullfile(saveDir, sprintf('%s_againstShuffle_time_stats.mat', cfg.metric)), ...
        'statAgainstShuffle', 'cfg', 'times');
end

%% ===================== shared pairwise time statistics =====================
statPairTime = struct();
pairInfo = cell(0,3);

if cfg.doComponentComparisonTime || cfg.doPairwiseTime
    pairs = filter_pairs(cfg.pairList, validModels);
    [statPairTime, pairInfo] = make_pairwise_time_stats(D, pairs, times, cfg);
end

%% ===================== 1b) component comparison in one plot =====================
if cfg.doComponentComparisonTime
    nModel = numel(validModels);
    plotData = cell(1, nModel);
    plotColors = nan(nModel, 3);

    for mi = 1:nModel
        modelName = validModels{mi};
        plotData{mi} = D.(modelName).diag;
        plotColors(mi,:) = D.(modelName).color;
    end

    fig = figure('Color','w','Position',[90 90 1180 620]); hold on;
    sigCfg = struct();
    sigCfg.alpha = cfg.alpha;
    sigCfg.modelNames = validModels;
    sigCfg.againstShuffleStats = struct();
    sigCfg.pairwiseStats = struct();
    sigCfg.pairInfo = cell(0,3);

    if cfg.showAgainstShuffleInComponentPlot
        sigCfg.againstShuffleStats = statAgainstShuffle;
        sigCfg.baselineLabels = make_baseline_label_map(D, validModels);
    end

    if cfg.showPairwiseInComponentPlot
        sigCfg.pairwiseStats = statPairTime;
        sigCfg.pairInfo = pairInfo;
    end

    plot_shaded_errorbar_fourCurve(times, plotData, nModel, ...
        [min(times) max(times)], [], 'Time (ms)', cfg.metric, validModels, plotColors, sigCfg);

    yline(cfg.chance, ':k', 'Chance', 'HandleVisibility','off');
    add_event_lines(cfg);

    title(sprintf('Component comparison: %s', cfg.metric));
    grid on;

    savefig(fig, fullfile(saveDir, sprintf('%s_componentComparison_time.fig', cfg.metric)));
    print(fig, fullfile(saveDir, sprintf('%s_componentComparison_time.png', cfg.metric)), '-dpng', '-r300');

    ComponentPairClusters = make_pairwise_time_cluster_table(statPairTime, pairInfo);
    if ~isempty(ComponentPairClusters)
        writetable(ComponentPairClusters, ...
            fullfile(saveDir, sprintf('%s_componentComparison_time_clusters.csv', cfg.metric)));
    end

    save(fullfile(saveDir, sprintf('%s_componentComparison_time_stats.mat', cfg.metric)), ...
        'statPairTime', 'pairInfo', 'statAgainstShuffle', 'cfg', 'times');
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

    ylabel(sprintf('Mean %s - baseline', cfg.metric));
    title(sprintf('Maintenance-period decoding evidence: %s - baseline', cfg.metric));

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
    if isempty(pairInfo)
        fprintf('Skip pairwise time-resolved contrasts: no valid pairs.\n');
    else
        [nRow, nCol] = subplot_grid(size(pairInfo,1));

        fig = figure('Color','w','Position',[100 100 1250 760]);

        for pi = 1:size(pairInfo,1)
            A = pairInfo{pi,1};
            B = pairInfo{pi,2};
            statName = pairInfo{pi,3};
            dataDiff = statPairTime.(statName).data;

            subplot(nRow, nCol, pi); hold on;

            y = mean(dataDiff, 1, 'omitnan');
            e = sem_rows(dataDiff);

            patch([times fliplr(times)], [y+e fliplr(y-e)], [0.25 0.25 0.25], ...
                'FaceAlpha', 0.18, 'EdgeColor', 'none');
            plot(times, y, 'k', 'LineWidth', 2.1);

            yline(0, ':k');
            add_event_lines(cfg);
            plot_sig_bar(gca, times, statPairTime.(statName).significantMask);

            xlabel('Time (ms)');
            ylabel(sprintf('%s - %s', A, B));
            title(sprintf('%s minus %s', A, B));
            box off; grid on;
        end

        savefig(fig, fullfile(saveDir, sprintf('%s_pairwise_time.fig', cfg.metric)));
        print(fig, fullfile(saveDir, sprintf('%s_pairwise_time.png', cfg.metric)), '-dpng', '-r300');
    end

    save(fullfile(saveDir, sprintf('%s_pairwise_time_stats.mat', cfg.metric)), ...
        'statPairTime', 'pairInfo', 'cfg', 'times');
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

        [matData, tgTimes, matFiles] = load_group_matrix(resultDir, modelName, cfg.metric, cfg.filePattern, cfg.includeSubjectIds, cfg.excludeSubjectIds);
        [matShuffle, shuffleFiles, baselineInfo] = load_or_make_baseline_matrix( ...
            resultDir, modelName, matData, tgTimes, matFiles, cfg);

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
            title(sprintf('%s: %s - %s', modelName, cfg.metric, baselineInfo.label));

            add_event_lines_tg(cfg);
            box off;
        end

        sgtitle(sprintf('Time-generalization: %s - baseline', cfg.metric), ...
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

        [matA, tgTimes, filesA] = load_group_matrix(fullfile(decodingDir, A), A, cfg.metric, cfg.filePattern, cfg.includeSubjectIds, cfg.excludeSubjectIds);
        [matB, ~, filesB] = load_group_matrix(fullfile(decodingDir, B), B, cfg.metric, cfg.filePattern, cfg.includeSubjectIds, cfg.excludeSubjectIds);

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
function [baselineData, baselineFiles, baselineInfo] = load_or_make_baseline_timeseries( ...
    resultDir, loadCfg, diagData, times, usedFiles, cfg)

metricCandidates = shuffle_metric_candidates(cfg);

for ci = 1:numel(metricCandidates)
    thisCfg = loadCfg;
    thisCfg.metric = metricCandidates{ci};
    thisCfg.silentMissingMetric = true;

    try
        [baselineData, baselineTimes, baselineFiles] = extract_decoding_timeseries(resultDir, thisCfg);
        validate_time_axis(baselineTimes, times, sprintf('%s %s', loadCfg.resultVarName, metricCandidates{ci}));
        baselineInfo = make_baseline_info('shuffle', metricCandidates{ci}, cfg);
        if ci > 1
            fprintf('  %s: using shuffle metric %s.\n', loadCfg.resultVarName, metricCandidates{ci});
        end
        return
    catch ME
        if is_missing_metric_error(ME)
            continue
        end
        rethrow(ME)
    end
end

if isfield(cfg, 'useChanceIfNoShuffle') && cfg.useChanceIfNoShuffle
    baselineData = repmat(cfg.chance, size(diagData));
    baselineFiles = usedFiles;
    baselineInfo = make_baseline_info('chance', '', cfg);
    fprintf('  %s: no usable shuffle metric (%s); using theoretical %s.\n', ...
        loadCfg.resultVarName, strjoin(metricCandidates, ', '), baselineInfo.label);
else
    error('No usable shuffle metric found in %s. Tried: %s.', ...
        resultDir, strjoin(metricCandidates, ', '));
end
end

%% ========================================================================
function [baselineMat, baselineFiles, baselineInfo] = load_or_make_baseline_matrix( ...
    resultDir, modelName, matData, times, usedFiles, cfg)

metricCandidates = shuffle_metric_candidates(cfg);

for ci = 1:numel(metricCandidates)
    [baselineMat, baselineTimes, baselineFiles] = load_group_matrix( ...
        resultDir, modelName, metricCandidates{ci}, cfg.filePattern, ...
        cfg.includeSubjectIds, cfg.excludeSubjectIds);

    if isempty(baselineMat) || isempty(baselineFiles)
        continue
    end

    validate_time_axis(baselineTimes, times, sprintf('%s %s TG', modelName, metricCandidates{ci}));
    baselineInfo = make_baseline_info('shuffle', metricCandidates{ci}, cfg);
    if ci > 1
        fprintf('  %s TG: using shuffle metric %s.\n', modelName, metricCandidates{ci});
    end
    return
end

if isfield(cfg, 'useChanceIfNoShuffle') && cfg.useChanceIfNoShuffle
    baselineMat = repmat(cfg.chance, size(matData));
    baselineFiles = usedFiles;
    baselineInfo = make_baseline_info('chance', '', cfg);
    fprintf('  %s TG: no usable shuffle metric (%s); using theoretical %s.\n', ...
        modelName, strjoin(metricCandidates, ', '), baselineInfo.label);
else
    error('No usable shuffle metric found in %s. Tried: %s.', ...
        resultDir, strjoin(metricCandidates, ', '));
end
end

%% ========================================================================
function metricCandidates = shuffle_metric_candidates(cfg)

metricCandidates = {};
if isfield(cfg, 'shuffleMetric') && ~isempty(cfg.shuffleMetric)
    metricCandidates{end+1} = cfg.shuffleMetric;
end

if isfield(cfg, 'metric') && ~isempty(cfg.metric)
    switch lower(cfg.metric)
        case 'acc'
            metricCandidates = [metricCandidates, ...
                {'AccShuffle', 'Acc_shuffle', 'accShuffle', 'acc_shuffle'}];
        case 'auc'
            metricCandidates = [metricCandidates, ...
                {'AUCShuffle', 'AUC_shuffle', 'aucShuffle', 'auc_shuffle'}];
        case 'predictacc'
            metricCandidates = [metricCandidates, ...
                {'predictAccShuffle', 'predictAcc_shuffle', 'predictaccShuffle', 'predictacc_shuffle'}];
    end
end

metricCandidates = unique(metricCandidates, 'stable');
if isempty(metricCandidates)
    metricCandidates = {'AccShuffle', 'Acc_shuffle'};
end
end

%% ========================================================================
function tf = is_missing_metric_error(ME)

msg = ME.message;
tf = contains(msg, 'No usable result files found') || ...
     contains(msg, 'No files matched');
end

%% ========================================================================
function validate_time_axis(candidateTimes, referenceTimes, labelText)

if isempty(candidateTimes) || isempty(referenceTimes)
    return
end

candidateTimes = candidateTimes(:)';
referenceTimes = referenceTimes(:)';
if numel(candidateTimes) ~= numel(referenceTimes) || any(abs(candidateTimes - referenceTimes) > 1e-9)
    error('Time vector mismatch for %s.', labelText);
end
end

%% ========================================================================
function baselineInfo = make_baseline_info(source, metricName, cfg)

baselineInfo = struct();
baselineInfo.source = source;
baselineInfo.metric = metricName;

switch lower(source)
    case 'shuffle'
        baselineInfo.label = 'shuffle';
    case 'chance'
        baselineInfo.label = sprintf('chance=%.3g', cfg.chance);
    otherwise
        baselineInfo.label = source;
end
end

%% ========================================================================
function labelMap = make_baseline_label_map(D, validModels)

labelMap = struct();
for mi = 1:numel(validModels)
    modelName = validModels{mi};
    if isfield(D.(modelName), 'baseline') && isfield(D.(modelName).baseline, 'label')
        labelMap.(modelName) = D.(modelName).baseline.label;
    else
        labelMap.(modelName) = 'baseline';
    end
end
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
function statAgainstShuffle = make_against_shuffle_time_stats(D, validModels, times, cfg)

statAgainstShuffle = struct();

for mi = 1:numel(validModels)
    modelName = validModels{mi};
    [diagData, shuffleData] = align_by_files( ...
        D.(modelName).diag, D.(modelName).files, ...
        D.(modelName).shuffle, D.(modelName).shuffleFiles);

    if isempty(diagData) || isempty(shuffleData)
        fprintf('Skip %s vs shuffle stats: no overlapping subjects.\n', modelName);
        continue
    end

    dataDiff = diagData - shuffleData;

    statCfg = struct();
    statCfg.null = 0;
    statCfg.nPerm = cfg.nPerm1D;
    statCfg.tail = 'right';
    statCfg.clusterAlpha = cfg.clusterAlpha;
    statCfg.alpha = cfg.alpha;
    statCfg.randomSeed = cfg.randomSeed;
    statCfg.verbose = false;

    statAgainstShuffle.(modelName) = cluster_perm_1d_timeseries(dataDiff, times, statCfg);
end

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
function [statPairTime, pairInfo] = make_pairwise_time_stats(D, pairs, times, cfg)

statPairTime = struct();
pairInfo = cell(0,3);

if isempty(pairs)
    return
end

for pi = 1:size(pairs,1)
    A = pairs{pi,1};
    B = pairs{pi,2};

    [XA, XB] = align_by_files(D.(A).diag, D.(A).files, D.(B).diag, D.(B).files);

    if isempty(XA) || isempty(XB)
        fprintf('Skip pair %s - %s: no overlapping subjects.\n', A, B);
        continue
    end

    dataDiff = XA - XB;

    statCfg = struct();
    statCfg.null = 0;
    statCfg.nPerm = cfg.nPerm1D;
    statCfg.tail = 'two';
    statCfg.clusterAlpha = cfg.clusterAlpha;
    statCfg.alpha = cfg.alpha;
    statCfg.randomSeed = cfg.randomSeed;
    statCfg.verbose = false;

    statName = sprintf('%s_minus_%s', A, B);
    statPairTime.(statName) = cluster_perm_1d_timeseries(dataDiff, times, statCfg);
    pairInfo(end+1,:) = {A, B, statName}; %#ok<AGROW>
end

end

function T = make_pairwise_time_cluster_table(statPairTime, pairInfo)

rows = cell(0,8);

for pi = 1:size(pairInfo,1)
    A = pairInfo{pi,1};
    B = pairInfo{pi,2};
    statName = pairInfo{pi,3};

    if ~isfield(statPairTime, statName)
        continue
    end

    clusters = statPairTime.(statName).significantClusters;

    for ci = 1:numel(clusters)
        rows(end+1,:) = {A, B, clusters(ci).startTime, clusters(ci).endTime, ...
            clusters(ci).p, clusters(ci).clusterStat, clusters(ci).nSamples, ...
            p_to_stars(clusters(ci).p)}; %#ok<AGROW>
    end
end

if isempty(rows)
    T = table();
else
    T = cell2table(rows, 'VariableNames', ...
        {'ModelA','ModelB','StartTime','EndTime','P_cluster','ClusterStat','NSamples','Stars'});
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
function [mats, times, usedFiles] = load_group_matrix(resultDir, varName, metricName, filePattern, includeSubjectIds, excludeSubjectIds)
if nargin < 5, includeSubjectIds = []; end
if nargin < 6, excludeSubjectIds = []; end
files = dir(fullfile(resultDir, filePattern));
files = filter_subject_files(files, includeSubjectIds, excludeSubjectIds);
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

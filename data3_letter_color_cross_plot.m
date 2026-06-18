%% Plot data3 letter-color cross-family sequential LDA results
% Draws group-level diagonal AUC curves and time-generalization matrices
% from data3_letter_color_cross_decoding.m outputs.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
addpath(codeDir);

cfg = struct();
cfg.analysisMode = 'maintOnly';
cfg.diagMetric = 'AUC';
cfg.matrixMetrics = {'AUC'};
cfg.chance = 0.5;
cfg.nPerm1D = 1000;
cfg.nPerm2D = 1000;
cfg.nPermSummary = 10000;
cfg.tail = 'right';
cfg.pairTail = 'both';
cfg.clusterAlpha = 0.05;
cfg.alpha = 0.05;
cfg.clusterStat = 'mass';
cfg.minClusterSize1D = 1;
cfg.minClusterSize2D = 2;
cfg.clusterConnectivity = 'withinSegmentBlocks';
cfg.randomSeed = 20260612;
cfg.figureDpi = 300;
cfg.modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
cfg.directionNames = {'letter_to_color', 'color_to_letter'};
cfg.directionLabels = {'Letter -> Color', 'Color -> Letter'};
cfg.directionColors = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980];
cfg.comparisons = { ...
    sprintf('letter_color_cross_setsize1_vs6_%s', cfg.analysisMode), ...
    fullfile(dataDir, sprintf('decoding_LDA_letter_color_cross_setsize1_vs6_segments_%s', cfg.analysisMode)); ...
    sprintf('letter_color_cross_setsize3_vs6_%s', cfg.analysisMode), ...
    fullfile(dataDir, sprintf('decoding_LDA_letter_color_cross_setsize3_vs6_segments_%s', cfg.analysisMode))};

modelsOverride = strtrim(char(getenv('DATA3_LETTER_COLOR_PLOT_MODELS')));
if ~isempty(modelsOverride)
    cfg.modelNames = regexp(modelsOverride, '[,;\s]+', 'split');
    cfg.modelNames = cfg.modelNames(~cellfun('isempty', cfg.modelNames));
end

nPermOverride = str2double(strtrim(char(getenv('DATA3_LETTER_COLOR_PLOT_NPERM'))));
if ~isnan(nPermOverride) && nPermOverride > 0
    cfg.nPerm1D = nPermOverride;
    cfg.nPerm2D = nPermOverride;
end

DiagTimeSummary = table();
DiagAggregateSummary = table();
DiagClusterSummary = table();
MatrixClusterSummary = table();
DirectionPairSummary = table();

for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    comparisonDir = cfg.comparisons{ci, 2};
    plotDir = fullfile(comparisonDir, 'GroupPlots');
    diagDir = fullfile(plotDir, 'diagonal');
    heatDir = fullfile(plotDir, 'heatmaps');
    summaryDir = fullfile(plotDir, 'summary');
    statsDir = fullfile(plotDir, 'stats');
    if ~isfolder(plotDir), mkdir(plotDir); end
    if ~isfolder(diagDir), mkdir(diagDir); end
    if ~isfolder(heatDir), mkdir(heatDir); end
    if ~isfolder(summaryDir), mkdir(summaryDir); end
    if ~isfolder(statsDir), mkdir(statsDir); end

    if contains(comparisonName, 'setsize3_vs6')
        legacyPatterns = {'*Acc*.*', '*direction_difference*.*', ...
            sprintf('%s_*_lowseg*_%s_diagonal.*', comparisonName, cfg.diagMetric), ...
            sprintf('%s_lowseg*_mean_diagonal_%s_by_model.*', comparisonName, cfg.diagMetric)};
    else
        legacyPatterns = {'*Acc*.*', '*direction_difference*.*'};
    end
    for legacyPatternIdx = 1:numel(legacyPatterns)
        legacyFiles = dir(fullfile(plotDir, '**', legacyPatterns{legacyPatternIdx}));
        for legacyFileIdx = 1:numel(legacyFiles)
            [~, ~, legacyExt] = fileparts(legacyFiles(legacyFileIdx).name);
            if any(strcmpi(legacyExt, {'.png', '.fig'}))
                delete(fullfile(legacyFiles(legacyFileIdx).folder, legacyFiles(legacyFileIdx).name));
            end
        end
    end

    fprintf('\nPlotting %s\n', comparisonName);

    aggregateMean = nan(numel(cfg.modelNames), 0, numel(cfg.directionNames));
    aggregateSem = nan(numel(cfg.modelNames), 0, numel(cfg.directionNames));
    aggregateSubjectMean = cell(numel(cfg.modelNames), 0, numel(cfg.directionNames));
    aggregateSubjectIds = cell(numel(cfg.modelNames), 0, numel(cfg.directionNames));
    validModelForAggregate = false(numel(cfg.modelNames), 1);

    for mi = 1:numel(cfg.modelNames)
        modelName = cfg.modelNames{mi};
        modelDir = fullfile(comparisonDir, modelName);
        if ~isfolder(modelDir)
            fprintf('  Skip %s: no folder.\n', modelName);
            continue;
        end

        files = dir(fullfile(modelDir, 'sub*.mat'));
        files = data3_filter_subject_mat_files(files, data3_subject_filter());
        if isempty(files)
            fprintf('  Skip %s: no subject files.\n', modelName);
            continue;
        end

        firstData = [];
        for fi = 1:numel(files)
            S = load(fullfile(files(fi).folder, files(fi).name), modelName);
            if isfield(S, modelName)
                firstData = S.(modelName);
                break;
            end
        end
        if isempty(firstData)
            fprintf('  Skip %s: variable not found.\n', modelName);
            continue;
        end

        nLowSeg = numel(firstData.direction(1).lowSegmentInfo);
        nHighSeg = numel(firstData.direction(1).highSegmentInfo);
        diagRows = cell(nLowSeg, nHighSeg, numel(cfg.directionNames));
        diagTimesAbs = cell(nLowSeg, nHighSeg);
        diagTimesRel = cell(nLowSeg, nHighSeg);
        diagSubjects = cell(nLowSeg, nHighSeg, numel(cfg.directionNames));

        matrixStack = struct();
        matrixSubjects = struct();
        for di = 1:numel(cfg.directionNames)
            directionName = cfg.directionNames{di};
            matrixStack.(directionName) = struct();
            matrixSubjects.(directionName) = [];
            for metri = 1:numel(cfg.matrixMetrics)
                metricName = cfg.matrixMetrics{metri};
                matrixStack.(directionName).(metricName) = [];
            end
        end

        referenceTimesLow = [];
        referenceTimesHigh = [];
        referenceDesign = [];

        for fi = 1:numel(files)
            fpath = fullfile(files(fi).folder, files(fi).name);
            S = load(fpath, modelName);
            if ~isfield(S, modelName)
                continue;
            end
            R = S.(modelName);
            if isfield(R, 'subject') && ~isempty(R.subject)
                subject = R.subject;
            else
                tok = regexp(files(fi).name, '^sub(\d+)\.mat$', 'tokens', 'once');
                subject = str2double(tok{1});
            end

            for di = 1:numel(cfg.directionNames)
                directionName = cfg.directionNames{di};
                dirNamesNow = {R.direction.name};
                directionIdx = find(strcmp(dirNamesNow, directionName), 1);
                if isempty(directionIdx)
                    warning('Skipping %s %s: missing direction %s.', modelName, files(fi).name, directionName);
                    continue;
                end
                D = R.direction(directionIdx);

                if isempty(referenceTimesLow)
                    referenceTimesLow = D.timesLow(:);
                    referenceTimesHigh = D.timesHigh(:);
                    referenceDesign = D.temporalDesign;
                else
                    if numel(D.timesLow) ~= numel(referenceTimesLow) || ...
                            any(abs(D.timesLow(:) - referenceTimesLow) > 1e-9) || ...
                            numel(D.timesHigh) ~= numel(referenceTimesHigh) || ...
                            any(abs(D.timesHigh(:) - referenceTimesHigh) > 1e-9)
                        error('Time axis mismatch in %s.', fpath);
                    end
                end

                if isfield(D, cfg.diagMetric)
                    Mdiag = D.(cfg.diagMetric);
                    for li = 1:nLowSeg
                        lowWindow = D.lowSegmentInfo(li).lowWindowMs;
                        rowIdx = D.timesLow(:) >= lowWindow(1) & D.timesLow(:) <= lowWindow(2);
                        for hi = 1:numel(D.highSegmentInfo)
                            highWindow = D.highSegmentInfo(hi).highWindowMs;
                            colIdx = D.timesHigh(:) >= highWindow(1) & D.timesHigh(:) <= highWindow(2);
                            block = Mdiag(rowIdx, colIdx);
                            blockTime = D.timesHigh(colIdx);
                            nDiag = min(size(block, 1), size(block, 2));
                            diagThis = diag(block(1:nDiag, 1:nDiag));
                            timeThisAbs = blockTime(1:nDiag);
                            timeThisRel = timeThisAbs(:) - highWindow(1);

                            if isempty(diagRows{li, hi, di})
                                diagRows{li, hi, di} = diagThis(:)';
                                diagTimesAbs{li, hi} = timeThisAbs(:)';
                                diagTimesRel{li, hi} = timeThisRel(:)';
                                diagSubjects{li, hi, di} = subject;
                            else
                                if numel(diagThis) ~= size(diagRows{li, hi, di}, 2)
                                    error('Diagonal length mismatch in %s.', fpath);
                                end
                                diagRows{li, hi, di}(end+1, :) = diagThis(:)';
                                diagSubjects{li, hi, di}(end+1, 1) = subject;
                            end
                        end
                    end
                end

                for metri = 1:numel(cfg.matrixMetrics)
                    metricName = cfg.matrixMetrics{metri};
                    if ~isfield(D, metricName)
                        continue;
                    end
                    M = D.(metricName);
                    if isempty(matrixStack.(directionName).(metricName))
                        matrixStack.(directionName).(metricName) = nan(1, size(M, 1), size(M, 2));
                        matrixStack.(directionName).(metricName)(1,:,:) = M;
                    else
                        matrixStack.(directionName).(metricName)(end+1,:,:) = M;
                    end
                end
                matrixSubjects.(directionName)(end+1, 1) = subject;
            end
        end

        if isempty(referenceTimesLow)
            fprintf('  Skip %s: no plottable data.\n', modelName);
            continue;
        end

        if size(aggregateMean, 2) < nLowSeg
            oldNLowSeg = size(aggregateMean, 2);
            aggregateMean(:, end+1:nLowSeg, :) = NaN;
            aggregateSem(:, end+1:nLowSeg, :) = NaN;
            aggregateSubjectMean(:, oldNLowSeg+1:nLowSeg, :) = ...
                cell(numel(cfg.modelNames), nLowSeg - oldNLowSeg, numel(cfg.directionNames));
            aggregateSubjectIds(:, oldNLowSeg+1:nLowSeg, :) = ...
                cell(numel(cfg.modelNames), nLowSeg - oldNLowSeg, numel(cfg.directionNames));
        end
        validModelForAggregate(mi) = true;

        %% Diagonal AUC curves
        makeSetsize3Grid = contains(comparisonName, 'setsize3_vs6') && nLowSeg == 3 && nHighSeg == 6;
        diagStats = cell(nLowSeg, nHighSeg, numel(cfg.directionNames));
        meanBySegment = cell(nLowSeg, nHighSeg, numel(cfg.directionNames));
        semBySegment = cell(nLowSeg, nHighSeg, numel(cfg.directionNames));
        allDiagY = [];

        for li = 1:nLowSeg
            for hi = 1:nHighSeg
                for di = 1:numel(cfg.directionNames)
                    Y = diagRows{li, hi, di};
                    if isempty(Y)
                        continue;
                    end
                    xRel = diagTimesRel{li, hi};
                    xAbs = diagTimesAbs{li, hi};
                    highWindow = firstData.direction(1).highSegmentInfo(hi).highWindowMs;

                    statCfg1D = struct();
                    statCfg1D.null = cfg.chance;
                    statCfg1D.nPerm = cfg.nPerm1D;
                    statCfg1D.tail = cfg.tail;
                    statCfg1D.clusterAlpha = cfg.clusterAlpha;
                    statCfg1D.alpha = cfg.alpha;
                    statCfg1D.clusterStat = cfg.clusterStat;
                    statCfg1D.minClusterSize = cfg.minClusterSize1D;
                    statCfg1D.randomSeed = cfg.randomSeed + ci * 100000 + mi * 1000 + li * 100 + hi * 10 + di;
                    statCfg1D.verbose = false;
                    diagStats{li, hi, di} = cluster_perm_1d_timeseries(Y, xRel, statCfg1D);

                    yMean = mean(Y, 1, 'omitnan');
                    nByTime = sum(~isnan(Y), 1);
                    ySem = std(Y, 0, 1, 'omitnan') ./ sqrt(nByTime);
                    meanBySegment{li, hi, di} = yMean;
                    semBySegment{li, hi, di} = ySem;
                    allDiagY = [allDiagY yMean - ySem yMean + ySem]; %#ok<AGROW>

                    for ti = 1:numel(xRel)
                        row = table({comparisonName}, {modelName}, li, ...
                            {sprintf('%g-%g ms', firstData.direction(1).lowSegmentInfo(li).lowWindowMs)}, ...
                            hi, {sprintf('%g-%g ms', highWindow)}, ...
                            cfg.directionNames(di), cfg.directionLabels(di), ...
                            xAbs(ti), xRel(ti), yMean(ti), ySem(ti), nByTime(ti), cfg.chance, ...
                            'VariableNames', {'Comparison', 'Model', 'LowSegment', ...
                            'LowWindow', 'HighSegment', 'HighWindow', ...
                            'Direction', 'DirectionLabel', 'HighTimeMs', ...
                            'SegmentTimeMs', 'MeanAUC', 'SEMAUC', 'NSubject', 'Chance'});
                        DiagTimeSummary = [DiagTimeSummary; row]; %#ok<AGROW>
                    end

                    for clusterIdx = 1:numel(diagStats{li, hi, di}.clusters)
                        clusterNow = diagStats{li, hi, di}.clusters(clusterIdx);
                        row = table({comparisonName}, {modelName}, li, ...
                            {sprintf('%g-%g ms', firstData.direction(1).lowSegmentInfo(li).lowWindowMs)}, ...
                            hi, {sprintf('%g-%g ms', highWindow)}, ...
                            cfg.directionNames(di), cfg.directionLabels(di), clusterIdx, ...
                            highWindow(1) + clusterNow.startTime, highWindow(1) + clusterNow.endTime, ...
                            clusterNow.startTime, clusterNow.endTime, clusterNow.nSamples, ...
                            clusterNow.clusterStat, clusterNow.p, ...
                            clusterNow.p <= cfg.alpha, cfg.nPerm1D, cfg.chance, ...
                            'VariableNames', {'Comparison', 'Model', 'LowSegment', ...
                            'LowWindow', 'HighSegment', 'HighWindow', ...
                            'Direction', 'DirectionLabel', 'ClusterID', ...
                            'StartHighTimeMs', 'EndHighTimeMs', ...
                            'StartSegmentTimeMs', 'EndSegmentTimeMs', 'NSamples', ...
                            'ClusterStat', 'PValue', 'Significant', 'NPerm', 'Chance'});
                        DiagClusterSummary = [DiagClusterSummary; row]; %#ok<AGROW>
                    end
                end
            end

            for di = 1:numel(cfg.directionNames)
                Yall = [];
                subjectList = [];
                for hi = 1:nHighSeg
                    Y = diagRows{li, hi, di};
                    if isempty(Y)
                        continue;
                    end
                    Yall = [Yall Y]; %#ok<AGROW>
                    if isempty(subjectList)
                        subjectList = diagSubjects{li, hi, di};
                    end
                end
                if isempty(Yall)
                    continue;
                end
                subjectMean = mean(Yall, 2, 'omitnan');
                aggregateMean(mi, li, di) = mean(subjectMean, 'omitnan');
                aggregateSem(mi, li, di) = std(subjectMean, 0, 'omitnan') ./ sqrt(sum(~isnan(subjectMean)));
                aggregateSubjectMean{mi, li, di} = subjectMean;
                aggregateSubjectIds{mi, li, di} = subjectList;
                row = table({comparisonName}, {modelName}, li, ...
                    {sprintf('%g-%g ms', firstData.direction(1).lowSegmentInfo(li).lowWindowMs)}, ...
                    cfg.directionNames(di), cfg.directionLabels(di), ...
                    aggregateMean(mi, li, di), aggregateSem(mi, li, di), ...
                    sum(~isnan(subjectMean)), cfg.chance, ...
                    'VariableNames', {'Comparison', 'Model', 'LowSegment', ...
                    'LowWindow', 'Direction', 'DirectionLabel', ...
                    'MeanDiagonalAUC', 'SEMDiagonalAUC', 'NSubject', 'Chance'});
                DiagAggregateSummary = [DiagAggregateSummary; row]; %#ok<AGROW>
            end
        end

        if ~isempty(allDiagY)
            yMin = min(allDiagY, [], 'omitnan');
            yMax = max(allDiagY, [], 'omitnan');
            yPad = max(0.02, 0.08 * (yMax - yMin));
            yLimitsNow = [max(0, yMin - yPad), min(1, yMax + yPad)];
        else
            yLimitsNow = [0.45 0.65];
        end

        yRangeNow = yLimitsNow(2) - yLimitsNow(1);
        if makeSetsize3Grid
            fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1680 900]);
            tileLayout = tiledlayout(fig, nLowSeg, nHighSeg, 'TileSpacing', 'compact', 'Padding', 'compact');
            axesHandles = gobjects(nLowSeg, nHighSeg);
            lineHandles = gobjects(1, numel(cfg.directionNames));
            for li = 1:nLowSeg
                for hi = 1:nHighSeg
                    axesHandles(li, hi) = nexttile(tileLayout, (li - 1) * nHighSeg + hi);
                    ax = axesHandles(li, hi);
                    hold(ax, 'on');
                    for di = 1:numel(cfg.directionNames)
                        Y = diagRows{li, hi, di};
                        if isempty(Y)
                            continue;
                        end
                        xRel = diagTimesRel{li, hi};
                        yMean = meanBySegment{li, hi, di};
                        ySem = semBySegment{li, hi, di};
                        colorNow = cfg.directionColors(di,:);
                        fill(ax, [xRel fliplr(xRel)], [yMean - ySem fliplr(yMean + ySem)], colorNow, ...
                            'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                        if ~isgraphics(lineHandles(di))
                            lineHandles(di) = plot(ax, xRel, yMean, 'LineWidth', 1.8, 'Color', colorNow);
                        else
                            plot(ax, xRel, yMean, 'LineWidth', 1.8, 'Color', colorNow, 'HandleVisibility', 'off');
                        end

                        if ~isempty(diagStats{li, hi, di}) && ~isempty(diagStats{li, hi, di}.significantClusters)
                            ySig = yLimitsNow(1) + (0.06 + 0.035 * (di - 1)) * yRangeNow;
                            for clusterIdx = 1:numel(diagStats{li, hi, di}.significantClusters)
                                clusterNow = diagStats{li, hi, di}.significantClusters(clusterIdx);
                                plot(ax, [clusterNow.startTime clusterNow.endTime], [ySig ySig], ...
                                    'Color', cfg.directionColors(di,:), 'LineWidth', 3.2, ...
                                    'HandleVisibility', 'off');
                            end
                        end
                    end
                    yline(ax, cfg.chance, 'k--', 'HandleVisibility', 'off');
                    highWindow = firstData.direction(1).highSegmentInfo(hi).highWindowMs;
                    xlim(ax, [0, highWindow(2) - highWindow(1)]);
                    ylim(ax, yLimitsNow);
                    if li == 1
                        title(ax, sprintf('H%d', hi));
                    end
                    if li == nLowSeg
                        xlabel(ax, 'ms in segment');
                    else
                        set(ax, 'XTickLabel', []);
                    end
                    if hi == 1
                        ylabel(ax, sprintf('L%d\nDiagonal %s', li, cfg.diagMetric));
                    else
                        set(ax, 'YTickLabel', []);
                    end
                    box(ax, 'off');
                    hold(ax, 'off');
                end
            end
            sgtitle(fig, sprintf('%s | %s | diagonal %s', comparisonName, modelName, cfg.diagMetric), ...
                'Interpreter', 'none');
            validLegend = isgraphics(lineHandles);
            if any(validLegend)
                legend(axesHandles(1, 1), lineHandles(validLegend), cfg.directionLabels(validLegend), ...
                    'Location', 'southoutside', 'Orientation', 'horizontal');
            end

            outBase = fullfile(diagDir, sprintf('%s_%s_%s_diagonal_3x6', ...
                comparisonName, modelName, cfg.diagMetric));
            print(fig, [outBase '.png'], '-dpng', sprintf('-r%d', cfg.figureDpi));
            savefig(fig, [outBase '.fig']);
            close(fig);
        else
            for li = 1:nLowSeg
                fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1480 430]);
                tileLayout = tiledlayout(fig, 1, nHighSeg, 'TileSpacing', 'compact', 'Padding', 'compact');
                axesHandles = gobjects(1, nHighSeg);
                lineHandles = gobjects(1, numel(cfg.directionNames));
                for hi = 1:nHighSeg
                    axesHandles(hi) = nexttile(tileLayout, hi);
                    ax = axesHandles(hi);
                    hold(ax, 'on');
                    for di = 1:numel(cfg.directionNames)
                        Y = diagRows{li, hi, di};
                        if isempty(Y)
                            continue;
                        end
                        xRel = diagTimesRel{li, hi};
                        yMean = meanBySegment{li, hi, di};
                        ySem = semBySegment{li, hi, di};
                        colorNow = cfg.directionColors(di,:);
                        fill(ax, [xRel fliplr(xRel)], [yMean - ySem fliplr(yMean + ySem)], colorNow, ...
                            'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                        if ~isgraphics(lineHandles(di))
                            lineHandles(di) = plot(ax, xRel, yMean, 'LineWidth', 2, 'Color', colorNow);
                        else
                            plot(ax, xRel, yMean, 'LineWidth', 2, 'Color', colorNow, 'HandleVisibility', 'off');
                        end

                        if ~isempty(diagStats{li, hi, di}) && ~isempty(diagStats{li, hi, di}.significantClusters)
                            ySig = yLimitsNow(1) + (0.06 + 0.035 * (di - 1)) * yRangeNow;
                            for clusterIdx = 1:numel(diagStats{li, hi, di}.significantClusters)
                                clusterNow = diagStats{li, hi, di}.significantClusters(clusterIdx);
                                plot(ax, [clusterNow.startTime clusterNow.endTime], [ySig ySig], ...
                                    'Color', cfg.directionColors(di,:), 'LineWidth', 4, ...
                                    'HandleVisibility', 'off');
                            end
                        end
                    end
                    yline(ax, cfg.chance, 'k--', 'HandleVisibility', 'off');
                    highWindow = firstData.direction(1).highSegmentInfo(hi).highWindowMs;
                    xlim(ax, [0, highWindow(2) - highWindow(1)]);
                    ylim(ax, yLimitsNow);
                    title(ax, sprintf('H%d', hi));
                    xlabel(ax, 'ms in segment');
                    if hi == 1
                        ylabel(ax, sprintf('Diagonal %s', cfg.diagMetric));
                    else
                        set(ax, 'YTickLabel', []);
                    end
                    box(ax, 'off');
                    hold(ax, 'off');
                end
                sgtitle(fig, sprintf('%s | %s | low segment %d', comparisonName, modelName, li), ...
                    'Interpreter', 'none');
                validLegend = isgraphics(lineHandles);
                if any(validLegend)
                    legend(axesHandles(1), lineHandles(validLegend), cfg.directionLabels(validLegend), ...
                        'Location', 'southoutside', 'Orientation', 'horizontal');
                end

                outBase = fullfile(diagDir, sprintf('%s_%s_lowseg%d_%s_diagonal', ...
                    comparisonName, modelName, li, cfg.diagMetric));
                print(fig, [outBase '.png'], '-dpng', sprintf('-r%d', cfg.figureDpi));
                savefig(fig, [outBase '.fig']);
                close(fig);
            end
        end

        if makeSetsize3Grid
            summaryMean = nan(nLowSeg, nHighSeg, numel(cfg.directionNames));
            summarySem = nan(nLowSeg, nHighSeg, numel(cfg.directionNames));
            summaryPairP = nan(nLowSeg, nHighSeg);
            summaryAllY = [];
            for li = 1:nLowSeg
                lowWindow = firstData.direction(1).lowSegmentInfo(li).lowWindowMs;
                for hi = 1:nHighSeg
                    highWindow = firstData.direction(1).highSegmentInfo(hi).highWindowMs;
                    subjectMeans = cell(1, numel(cfg.directionNames));
                    subjectIds = cell(1, numel(cfg.directionNames));
                    for di = 1:numel(cfg.directionNames)
                        Y = diagRows{li, hi, di};
                        if isempty(Y)
                            continue;
                        end
                        subjectMeans{di} = mean(Y, 2, 'omitnan');
                        subjectIds{di} = diagSubjects{li, hi, di};
                        summaryMean(li, hi, di) = mean(subjectMeans{di}, 'omitnan');
                        summarySem(li, hi, di) = std(subjectMeans{di}, 0, 'omitnan') ./ sqrt(sum(~isnan(subjectMeans{di})));
                        summaryAllY = [summaryAllY summaryMean(li, hi, di) + summarySem(li, hi, di)]; %#ok<AGROW>
                    end

                    if numel(subjectMeans) >= 2 && ~isempty(subjectMeans{1}) && ~isempty(subjectMeans{2})
                        [commonSubjects, idxL2C, idxC2L] = intersect(subjectIds{1}(:), subjectIds{2}(:));
                        pairedL2C = subjectMeans{1}(idxL2C);
                        pairedC2L = subjectMeans{2}(idxC2L);
                        pairDiff = pairedC2L(:) - pairedL2C(:);
                        validPair = ~isnan(pairDiff);
                        pairDiff = pairDiff(validPair);
                        commonSubjects = commonSubjects(validPair);

                        if numel(pairDiff) >= 2
                            [~, pT, ~, statT] = ttest(pairDiff, 0, 'Tail', cfg.pairTail);
                            rng(cfg.randomSeed + ci * 100000 + mi * 1000 + li * 100 + hi * 10, 'twister');
                            signMat = (randi([0 1], numel(pairDiff), cfg.nPermSummary) * 2) - 1;
                            permMean = mean(pairDiff(:) .* signMat, 1, 'omitnan');
                            obsMean = mean(pairDiff, 'omitnan');
                            if strcmpi(cfg.pairTail, 'two') || strcmpi(cfg.pairTail, 'both')
                                pPerm = (1 + sum(abs(permMean) >= abs(obsMean))) / (cfg.nPermSummary + 1);
                            elseif strcmpi(cfg.pairTail, 'right')
                                pPerm = (1 + sum(permMean >= obsMean)) / (cfg.nPermSummary + 1);
                            else
                                pPerm = (1 + sum(permMean <= obsMean)) / (cfg.nPermSummary + 1);
                            end
                            tValue = statT.tstat;
                            dfValue = statT.df;
                        else
                            pT = NaN;
                            pPerm = NaN;
                            tValue = NaN;
                            dfValue = NaN;
                            obsMean = mean(pairDiff, 'omitnan');
                        end

                        summaryPairP(li, hi) = pPerm;
                        pairSem = std(pairDiff, 0, 'omitnan') ./ sqrt(sum(~isnan(pairDiff)));
                        row = table({comparisonName}, {modelName}, li, ...
                            {sprintf('%g-%g ms', lowWindow)}, hi, ...
                            {sprintf('%g-%g ms', highWindow)}, obsMean, pairSem, ...
                            numel(pairDiff), tValue, dfValue, pT, pPerm, cfg.nPermSummary, ...
                            'VariableNames', {'Comparison', 'Model', 'LowSegment', ...
                            'LowWindow', 'HighSegment', 'HighWindow', ...
                            'MeanDiff_C2L_minus_L2C', 'SEMDiff', ...
                            'NSubject', 'TValue', 'DF', 'PValueTTest', ...
                            'PValueSignFlip', 'NPerm'});
                        DirectionPairSummary = [DirectionPairSummary; row]; %#ok<AGROW>
                    end
                end
            end

            if isempty(summaryAllY) || all(isnan(summaryAllY))
                summaryYLim = [0.45 0.65];
            else
                summaryYLim = [0.45, min(1, max(0.6, max(summaryAllY, [], 'omitnan') + 0.05))];
            end

            fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1680 900]);
            tileLayout = tiledlayout(fig, nLowSeg, nHighSeg, 'TileSpacing', 'compact', 'Padding', 'compact');
            for li = 1:nLowSeg
                for hi = 1:nHighSeg
                    ax = nexttile(tileLayout, (li - 1) * nHighSeg + hi);
                    yBar = squeeze(summaryMean(li, hi, :))';
                    eBar = squeeze(summarySem(li, hi, :))';
                    bh = bar(ax, 1:numel(cfg.directionNames), yBar, 0.66);
                    bh.FaceColor = 'flat';
                    bh.CData = cfg.directionColors(1:numel(cfg.directionNames), :);
                    hold(ax, 'on');
                    errorbar(ax, 1:numel(cfg.directionNames), yBar, eBar, 'k.', 'LineWidth', 1, ...
                        'HandleVisibility', 'off');
                    if ~isnan(summaryPairP(li, hi))
                        yTop = max(yBar + eBar, [], 'omitnan') + 0.014;
                        plot(ax, [1 2], [yTop yTop], 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
                        text(ax, 1.5, yTop + 0.006, sprintf('p=%.3g', summaryPairP(li, hi)), ...
                            'HorizontalAlignment', 'center', 'FontSize', 7);
                    end
                    yline(ax, cfg.chance, 'k--', 'HandleVisibility', 'off');
                    ylim(ax, summaryYLim);
                    xlim(ax, [0.4, numel(cfg.directionNames) + 0.6]);
                    set(ax, 'XTick', 1:numel(cfg.directionNames), 'XTickLabel', {'L->C', 'C->L'});
                    if li < nLowSeg
                        set(ax, 'XTickLabel', []);
                    end
                    if li == 1
                        title(ax, sprintf('H%d', hi));
                    end
                    if hi == 1
                        ylabel(ax, sprintf('L%d\nMean AUC', li));
                    else
                        set(ax, 'YTickLabel', []);
                    end
                    box(ax, 'off');
                    hold(ax, 'off');
                end
            end
            sgtitle(fig, sprintf('%s | %s | mean diagonal %s', comparisonName, modelName, cfg.diagMetric), ...
                'Interpreter', 'none');
            outBase = fullfile(summaryDir, sprintf('%s_%s_mean_diagonal_%s_summary_3x6', ...
                comparisonName, modelName, cfg.diagMetric));
            print(fig, [outBase '.png'], '-dpng', sprintf('-r%d', cfg.figureDpi));
            savefig(fig, [outBase '.fig']);
            close(fig);
        end

        %% Time-generalization matrices, one model per metric
        for metri = 1:numel(cfg.matrixMetrics)
            metricName = cfg.matrixMetrics{metri};
            stack1 = matrixStack.(cfg.directionNames{1}).(metricName);
            stack2 = matrixStack.(cfg.directionNames{2}).(metricName);
            if isempty(stack1) || isempty(stack2)
                continue;
            end

            values = [stack1(:); stack2(:)];
            delta = max(abs(values - cfg.chance), [], 'omitnan');
            if isempty(delta) || isnan(delta) || delta == 0
                delta = 0.05;
            end
            delta = max(delta, 0.02);
            colorLim = [max(0, cfg.chance - delta), min(1, cfg.chance + delta)];
            if colorLim(1) >= colorLim(2)
                colorLim = cfg.chance + [-0.05 0.05];
            end

            fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 1250 520]);
            for di = 1:numel(cfg.directionNames)
                directionName = cfg.directionNames{di};
                stackNow = matrixStack.(directionName).(metricName);
                meanMat = squeeze(mean(stackNow, 1, 'omitnan'));
                statCfg2D = struct();
                statCfg2D.null = cfg.chance;
                statCfg2D.nPerm = cfg.nPerm2D;
                statCfg2D.tail = cfg.tail;
                statCfg2D.clusterAlpha = cfg.clusterAlpha;
                statCfg2D.alpha = cfg.alpha;
                statCfg2D.clusterStat = cfg.clusterStat;
                statCfg2D.minClusterSize = cfg.minClusterSize2D;
                statCfg2D.clusterConnectivity = cfg.clusterConnectivity;
                statCfg2D.randomSeed = cfg.randomSeed + ci * 100000 + mi * 1000 + metri * 100 + di;
                statCfg2D.verbose = false;
                stat2D = cluster_perm_2d_matrix(stackNow, referenceTimesLow, ...
                    referenceTimesHigh, referenceDesign, statCfg2D);

                save(fullfile(statsDir, sprintf('%s_%s_%s_%s_2d_cluster_stats.mat', ...
                    comparisonName, modelName, metricName, directionName)), ...
                    'stat2D', 'cfg', 'comparisonName', 'modelName', 'metricName', ...
                    'directionName', '-v7.3');

                for clusterIdx = 1:numel(stat2D.clusters)
                    clusterNow = stat2D.clusters(clusterIdx);
                    [rowIdx, colIdx] = ind2sub(size(stat2D.significantMask), clusterNow.idx);
                    row = table({comparisonName}, {modelName}, {metricName}, ...
                        {directionName}, cfg.directionLabels(di), clusterIdx, ...
                        min(referenceTimesLow(rowIdx)), max(referenceTimesLow(rowIdx)), ...
                        min(referenceTimesHigh(colIdx)), max(referenceTimesHigh(colIdx)), ...
                        clusterNow.nPixels, clusterNow.clusterStat, clusterNow.p, ...
                        clusterNow.p <= cfg.alpha, cfg.nPerm2D, cfg.chance, ...
                        'VariableNames', {'Comparison', 'Model', 'Metric', ...
                        'Direction', 'DirectionLabel', 'ClusterID', ...
                        'LowStartTimeMs', 'LowEndTimeMs', 'HighStartTimeMs', ...
                        'HighEndTimeMs', 'NPixels', 'ClusterStat', 'PValue', ...
                        'Significant', 'NPerm', 'Chance'});
                    MatrixClusterSummary = [MatrixClusterSummary; row]; %#ok<AGROW>
                end

                lowBlockIdx = cell(1, nLowSeg);
                highBlockIdx = cell(1, nHighSeg);
                lowBlockN = zeros(1, nLowSeg);
                highBlockN = zeros(1, nHighSeg);
                for liSeg = 1:nLowSeg
                    lowWindow = firstData.direction(1).lowSegmentInfo(liSeg).lowWindowMs;
                    lowBlockIdx{liSeg} = find(referenceTimesLow >= lowWindow(1) & referenceTimesLow <= lowWindow(2));
                    lowBlockN(liSeg) = numel(lowBlockIdx{liSeg});
                end
                for hiSeg = 1:nHighSeg
                    highWindow = firstData.direction(1).highSegmentInfo(hiSeg).highWindowMs;
                    highBlockIdx{hiSeg} = find(referenceTimesHigh >= highWindow(1) & referenceTimesHigh <= highWindow(2));
                    highBlockN(hiSeg) = numel(highBlockIdx{hiSeg});
                end

                gapN = 2;
                displayRowsN = sum(lowBlockN) + gapN * (nLowSeg - 1);
                displayColsN = sum(highBlockN) + gapN * (nHighSeg - 1);
                meanDisplay = nan(displayRowsN, displayColsN);
                sigDisplay = false(displayRowsN, displayColsN);
                lowCenters = nan(1, nLowSeg);
                highCenters = nan(1, nHighSeg);
                rowStart = 1;
                for liSeg = 1:nLowSeg
                    rowEnd = rowStart + lowBlockN(liSeg) - 1;
                    lowCenters(liSeg) = (rowStart + rowEnd) / 2;
                    colStart = 1;
                    for hiSeg = 1:nHighSeg
                        colEnd = colStart + highBlockN(hiSeg) - 1;
                        if liSeg == 1
                            highCenters(hiSeg) = (colStart + colEnd) / 2;
                        end
                        meanDisplay(rowStart:rowEnd, colStart:colEnd) = ...
                            meanMat(lowBlockIdx{liSeg}, highBlockIdx{hiSeg});
                        sigDisplay(rowStart:rowEnd, colStart:colEnd) = ...
                            stat2D.significantMask(lowBlockIdx{liSeg}, highBlockIdx{hiSeg});
                        colStart = colEnd + gapN + 1;
                    end
                    rowStart = rowEnd + gapN + 1;
                end

                ax = subplot(1, numel(cfg.directionNames), di, 'Parent', fig);
                imageHandle = imagesc(ax, meanDisplay);
                set(imageHandle, 'AlphaData', ~isnan(meanDisplay));
                axis(ax, 'xy');
                hold(ax, 'on');
                set(ax, 'Color', 'w');
                colormap(ax, parula);
                colorbar(ax);
                clim(ax, colorLim);
                if any(sigDisplay(:))
                    contour(ax, 1:displayColsN, 1:displayRowsN, double(sigDisplay), [1 1], ...
                        'Color', 'k', 'LineWidth', 1.4);
                end
                set(ax, 'XTick', highCenters, ...
                    'XTickLabel', arrayfun(@(x) sprintf('H%d', x), 1:nHighSeg, 'UniformOutput', false));
                set(ax, 'YTick', lowCenters, ...
                    'YTickLabel', arrayfun(@(x) sprintf('L%d', x), 1:nLowSeg, 'UniformOutput', false));
                xlabel(ax, 'Set-size-6 test segment');
                ylabel(ax, 'Low-set-size train segment');
                title(ax, sprintf('%s | %s | %s | %s', ...
                    comparisonName, modelName, metricName, cfg.directionLabels{di}), ...
                    'Interpreter', 'none');
                for hiSeg = 1:(nHighSeg - 1)
                    xline(ax, highCenters(hiSeg) + highBlockN(hiSeg) / 2 + gapN / 2, ...
                        '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 0.8);
                end
                for liSeg = 1:(nLowSeg - 1)
                    yline(ax, lowCenters(liSeg) + lowBlockN(liSeg) / 2 + gapN / 2, ...
                        '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 0.8);
                end
                hold(ax, 'off');
            end

            outBase = fullfile(heatDir, sprintf('%s_%s_%s_heatmaps', ...
                comparisonName, modelName, metricName));
            print(fig, [outBase '.png'], '-dpng', sprintf('-r%d', cfg.figureDpi));
            savefig(fig, [outBase '.fig']);
            close(fig);
        end

        fprintf('  %s plotted.\n', modelName);
    end

    %% Extra summary: mean diagonal AUC by model and direction
    if ~contains(comparisonName, 'setsize3_vs6')
        validModelIdx = find(validModelForAggregate);
        modelLabels = cfg.modelNames(validModelIdx);
        for li = 1:size(aggregateMean, 2)
            if all(isnan(aggregateMean(validModelIdx, li, :)), 'all')
                continue;
            end

            fig = figure('Color', 'w', 'Visible', 'off', 'Position', [100 100 980 460]);
            ax = axes('Parent', fig);
            Ybar = squeeze(aggregateMean(validModelIdx, li, :));
            Ebar = squeeze(aggregateSem(validModelIdx, li, :));
            if isvector(Ybar)
                Ybar = Ybar(:)';
                Ebar = Ebar(:)';
            end

            pairPermP = nan(numel(validModelIdx), 1);
            pairTTestP = nan(numel(validModelIdx), 1);
            for modelPos = 1:numel(validModelIdx)
                modelIdx = validModelIdx(modelPos);
                subjectsL2C = aggregateSubjectIds{modelIdx, li, 1};
                subjectsC2L = aggregateSubjectIds{modelIdx, li, 2};
                valuesL2C = aggregateSubjectMean{modelIdx, li, 1};
                valuesC2L = aggregateSubjectMean{modelIdx, li, 2};
                if isempty(subjectsL2C) || isempty(subjectsC2L)
                    continue;
                end

                [commonSubjects, idxL2C, idxC2L] = intersect(subjectsL2C(:), subjectsC2L(:));
                pairedL2C = valuesL2C(idxL2C);
                pairedC2L = valuesC2L(idxC2L);
                pairDiff = pairedC2L(:) - pairedL2C(:);
                validPair = ~isnan(pairDiff);
                pairDiff = pairDiff(validPair);
                commonSubjects = commonSubjects(validPair);

                if numel(pairDiff) >= 2
                    [~, pT, ~, statT] = ttest(pairDiff, 0, 'Tail', cfg.pairTail);
                    rng(cfg.randomSeed + ci * 100000 + modelIdx * 1000 + li * 10, 'twister');
                    signMat = (randi([0 1], numel(pairDiff), cfg.nPermSummary) * 2) - 1;
                    permMean = mean(pairDiff(:) .* signMat, 1, 'omitnan');
                    obsMean = mean(pairDiff, 'omitnan');
                    if strcmpi(cfg.pairTail, 'two') || strcmpi(cfg.pairTail, 'both')
                        pPerm = (1 + sum(abs(permMean) >= abs(obsMean))) / (cfg.nPermSummary + 1);
                    elseif strcmpi(cfg.pairTail, 'right')
                        pPerm = (1 + sum(permMean >= obsMean)) / (cfg.nPermSummary + 1);
                    else
                        pPerm = (1 + sum(permMean <= obsMean)) / (cfg.nPermSummary + 1);
                    end
                    tValue = statT.tstat;
                    dfValue = statT.df;
                else
                    pT = NaN;
                    pPerm = NaN;
                    tValue = NaN;
                    dfValue = NaN;
                    obsMean = mean(pairDiff, 'omitnan');
                end

                pairPermP(modelPos) = pPerm;
                pairTTestP(modelPos) = pT;
                pairSem = std(pairDiff, 0, 'omitnan') ./ sqrt(sum(~isnan(pairDiff)));
                row = table({comparisonName}, cfg.modelNames(modelIdx), li, ...
                    {sprintf('%g-%g ms', firstData.direction(1).lowSegmentInfo(li).lowWindowMs)}, ...
                    NaN, {'all high segments'}, obsMean, pairSem, numel(pairDiff), ...
                    tValue, dfValue, pT, pPerm, cfg.nPermSummary, ...
                    'VariableNames', {'Comparison', 'Model', 'LowSegment', ...
                    'LowWindow', 'HighSegment', 'HighWindow', ...
                    'MeanDiff_C2L_minus_L2C', 'SEMDiff', ...
                    'NSubject', 'TValue', 'DF', 'PValueTTest', ...
                    'PValueSignFlip', 'NPerm'});
                DirectionPairSummary = [DirectionPairSummary; row]; %#ok<AGROW>
            end

            bh = bar(ax, Ybar, 'grouped');
            hold(ax, 'on');
            barX = nan(numel(modelLabels), numel(cfg.directionNames));
            for di = 1:numel(cfg.directionNames)
                bh(di).FaceColor = cfg.directionColors(di,:);
                xEnd = bh(di).XEndPoints;
                barX(:, di) = xEnd(:);
                errorbar(ax, xEnd, Ybar(:,di), Ebar(:,di), 'k.', 'LineWidth', 1);
            end
            for modelPos = 1:numel(validModelIdx)
                if any(isnan(barX(modelPos,:))) || isnan(pairPermP(modelPos))
                    continue;
                end
                yTop = max(Ybar(modelPos,:) + Ebar(modelPos,:), [], 'omitnan') + 0.015;
                plot(ax, barX(modelPos,:), [yTop yTop], 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
                text(ax, mean(barX(modelPos,:)), yTop + 0.006, sprintf('p=%.3g', pairPermP(modelPos)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8, 'Rotation', 0);
            end
            yline(ax, cfg.chance, 'k--', 'Chance');
            set(ax, 'XTick', 1:numel(modelLabels), 'XTickLabel', modelLabels);
            ylabel(ax, 'Mean diagonal AUC');
            title(ax, sprintf('%s | low segment %d mean diagonal AUC', comparisonName, li), ...
                'Interpreter', 'none');
            legend(ax, cfg.directionLabels, 'Location', 'best');
            box(ax, 'off');
            ylim(ax, [0.45, min(1, max(0.6, max(Ybar(:) + Ebar(:), [], 'omitnan') + 0.03))]);
            hold(ax, 'off');

            outBase = fullfile(summaryDir, sprintf('%s_lowseg%d_mean_diagonal_%s_by_model', ...
                comparisonName, li, cfg.diagMetric));
            print(fig, [outBase '.png'], '-dpng', sprintf('-r%d', cfg.figureDpi));
            savefig(fig, [outBase '.fig']);
            close(fig);
        end
    end

    writetable(DiagTimeSummary(strcmp(DiagTimeSummary.Comparison, comparisonName), :), ...
        fullfile(plotDir, sprintf('%s_diagonal_time_summary.csv', comparisonName)));
    writetable(DiagAggregateSummary(strcmp(DiagAggregateSummary.Comparison, comparisonName), :), ...
        fullfile(plotDir, sprintf('%s_diagonal_aggregate_summary.csv', comparisonName)));
    if ismember('Comparison', DiagClusterSummary.Properties.VariableNames)
        writetable(DiagClusterSummary(strcmp(DiagClusterSummary.Comparison, comparisonName), :), ...
            fullfile(plotDir, sprintf('%s_diagonal_chance_cluster_summary.csv', comparisonName)));
    else
        writetable(table(), fullfile(plotDir, sprintf('%s_diagonal_chance_cluster_summary.csv', comparisonName)));
    end
    if ismember('Comparison', MatrixClusterSummary.Properties.VariableNames)
        writetable(MatrixClusterSummary(strcmp(MatrixClusterSummary.Comparison, comparisonName), :), ...
            fullfile(plotDir, sprintf('%s_matrix_AUC_chance_cluster_summary.csv', comparisonName)));
    else
        writetable(table(), fullfile(plotDir, sprintf('%s_matrix_AUC_chance_cluster_summary.csv', comparisonName)));
    end
    if ismember('Comparison', DirectionPairSummary.Properties.VariableNames)
        writetable(DirectionPairSummary(strcmp(DirectionPairSummary.Comparison, comparisonName), :), ...
            fullfile(plotDir, sprintf('%s_diagonal_direction_pair_summary.csv', comparisonName)));
    else
        writetable(table(), fullfile(plotDir, sprintf('%s_diagonal_direction_pair_summary.csv', comparisonName)));
    end
end

save(fullfile(dataDir, sprintf('data3_letter_color_cross_plots_%s.mat', cfg.diagMetric)), ...
    'DiagTimeSummary', 'DiagAggregateSummary', 'DiagClusterSummary', ...
    'MatrixClusterSummary', 'DirectionPairSummary', 'cfg', '-v7.3');

fprintf('\nLetter-color cross-family plots finished.\n');

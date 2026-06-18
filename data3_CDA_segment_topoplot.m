%% Plot data3 CDA segment LDA weights and high-low activity topographies
% This script reads the saved setsize1-vs-6 and setsize3-vs-6 CDA LDA
% segment results. It does not rerun decoding.
%
% Interpretation:
%   1) LDA weights show which lateralized electrode-pair features support
%      high-load evidence in the model.
%   2) CDA high-low activity maps are descriptive condition differences and
%      are the better check for whether posterior CDA activity itself grows
%      across high-set-size segments.
%   3) Both maps are canonical lateralized maps: right-side electrodes are
%      plotted as contralateral and left-side electrodes as ipsilateral,
%      matching the model feature contra-minus-ipsi convention.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
addpath(codeDir);

cfg = data3_default_cfg();
comparisonList = { ...
    'setsize1_vs6_maintOnly', fullfile(dataDir, 'decoding_LDA_setsize1_vs6_segments_maintOnly'); ...
    'setsize3_vs6_maintOnly', fullfile(dataDir, 'decoding_LDA_setsize3_vs6_segments_maintOnly')};
modelName = 'CDA';

outputDir = fullfile(dataDir, 'CDA_segment_topographies_maintOnly');
if ~isfolder(outputDir)
    mkdir(outputDir);
end

topoplotPath = which('topoplot');
if isempty(topoplotPath)
    error('EEGLAB topoplot was not found on the MATLAB path.');
end

eeglabRoot = fileparts(fileparts(fileparts(topoplotPath)));
standardLocCandidates = { ...
    fullfile(eeglabRoot, 'functions', 'supportfiles', 'channel_location_files', 'eeglab', 'Standard-10-20-Cap81.ced'), ...
    fullfile(eeglabRoot, 'functions', 'supportfiles', 'channel_location_files', 'eeglab', 'Standard-10-10-Cap47.ced'), ...
    fullfile(eeglabRoot, 'functions', 'supportfiles', 'channel_location_files', 'eeglab', 'Standard-10-10-Cap33.ced')};

standardLocFile = '';
for lfi = 1:numel(standardLocCandidates)
    if isfile(standardLocCandidates{lfi})
        standardLocFile = standardLocCandidates{lfi};
        break;
    end
end
if isempty(standardLocFile)
    error('Cannot find an EEGLAB standard channel-location file under %s.', eeglabRoot);
end

[standardChanlocs, standardLabels] = readlocs(standardLocFile);
leftTopoIdx = data3_chan_indices(standardLabels, cfg.leftPosteriorLabels);
rightTopoIdx = data3_chan_indices(standardLabels, cfg.rightPosteriorLabels);
if numel(leftTopoIdx) ~= numel(cfg.leftPosteriorLabels) || ...
        numel(rightTopoIdx) ~= numel(cfg.rightPosteriorLabels)
    error('The standard location file is missing one or more posterior CDA channels: %s.', standardLocFile);
end
plotChanlocs = standardChanlocs([leftTopoIdx rightTopoIdx]);

nColor = 256;
halfColor = nColor / 2;
bluePart = [linspace(0.05, 1, halfColor)', linspace(0.20, 1, halfColor)', linspace(0.80, 1, halfColor)'];
redPart = [linspace(1, 0.80, halfColor)', linspace(1, 0.10, halfColor)', linspace(1, 0.05, halfColor)'];
signedCmap = [bluePart; redPart];

summaryRows = {};
scoreRows = {};
trendRows = {};
aucTrendRows = {};
aucLinkRows = {};

for ci = 1:size(comparisonList, 1)
    comparisonName = comparisonList{ci, 1};
    resultDir = fullfile(comparisonList{ci, 2}, modelName);
    resultFiles = dir(fullfile(resultDir, 'sub*.mat'));
    if isempty(resultFiles)
        warning('No %s files found in %s.', modelName, resultDir);
        continue;
    end

    allWeights = [];
    allActivityDiff = [];
    allAucDiag = [];
    subjectList = [];
    temporalDesign = [];
    channelLabels = {};

    for sf = 1:numel(resultFiles)
        resultPath = fullfile(resultFiles(sf).folder, resultFiles(sf).name);
        S = load(resultPath, modelName);
        if ~isfield(S, modelName)
            warning('Skip %s because it does not contain %s.', resultPath, modelName);
            continue;
        end

        Decode = S.(modelName);
        if ~isfield(Decode, 'weightsByPair') || isempty(Decode.weightsByPair)
            warning('Skip %s because weightsByPair is missing.', resultPath);
            continue;
        end

        nPair = size(Decode.weightsByPair, 1);
        nBin = size(Decode.weightsByPair, 2);
        nLowSeg = size(Decode.weightsByPair, 3);
        nHighSeg = size(Decode.weightsByPair, 4);
        weightsNow = mean(Decode.weightsByPair, 2, 'omitnan');
        weightsNow = reshape(weightsNow, [nPair, nLowSeg, nHighSeg]);

        aucDiagNow = nan(nLowSeg, nHighSeg);
        if isfield(Decode, 'AUC') && ~isempty(Decode.AUC)
            for li = 1:nLowSeg
                rowIdx = (li - 1) * nBin + (1:nBin);
                for hi = 1:nHighSeg
                    colIdx = (hi - 1) * nBin + (1:nBin);
                    aucBlock = Decode.AUC(rowIdx, colIdx);
                    aucDiagNow(li,hi) = mean(diag(aucBlock), 'omitnan');
                end
            end
        end

        Scda = load(Decode.inputSource, 'cda');
        lowSetSize = Decode.temporalDesign.lowSetSize;
        highSetSize = Decode.temporalDesign.highSetSize;
        lowField = sprintf('setsize%d', lowSetSize);
        highField = sprintf('setsize%d', highSetSize);
        lowTime = Scda.cda.timeBySetSize.(lowField)(:)';
        highTime = Scda.cda.timeBySetSize.(highField)(:)';

        lowActivity = nan(nPair, nLowSeg);
        highActivity = nan(nPair, nHighSeg);
        sideNames = {'L', 'R'};

        for li = 1:nLowSeg
            lowWindow = Decode.temporalDesign.lowWindowsMs(li,:);
            lowTimeIdx = lowTime >= lowWindow(1) & lowTime <= lowWindow(2);
            lowSideMean = nan(numel(sideNames), nPair);

            for si = 1:numel(sideNames)
                sideName = sideNames{si};
                leftName = sprintf('left_%s_%d', sideName, lowSetSize);
                rightName = sprintf('right_%s_%d', sideName, lowSetSize);
                leftX = Scda.cda.trial.(leftName);
                rightX = Scda.cda.trial.(rightName);
                if strcmpi(sideName, 'L')
                    X = rightX - leftX;
                else
                    X = leftX - rightX;
                end

                Xwin = mean(X(:,:,lowTimeIdx), 3, 'omitnan');
                lowSideMean(si,:) = mean(Xwin, 1, 'omitnan');
            end

            lowActivity(:,li) = mean(lowSideMean, 1, 'omitnan')';
        end

        for hi = 1:nHighSeg
            highWindow = [Decode.temporalDesign.highSegmentStartsMs(hi), ...
                Decode.temporalDesign.highSegmentStartsMs(hi) + Decode.temporalDesign.highSegmentWidthMs];
            highTimeIdx = highTime >= highWindow(1) & highTime <= highWindow(2);
            highSideMean = nan(numel(sideNames), nPair);

            for si = 1:numel(sideNames)
                sideName = sideNames{si};
                leftName = sprintf('left_%s_%d', sideName, highSetSize);
                rightName = sprintf('right_%s_%d', sideName, highSetSize);
                leftX = Scda.cda.trial.(leftName);
                rightX = Scda.cda.trial.(rightName);
                if strcmpi(sideName, 'L')
                    X = rightX - leftX;
                else
                    X = leftX - rightX;
                end

                Xwin = mean(X(:,:,highTimeIdx), 3, 'omitnan');
                highSideMean(si,:) = mean(Xwin, 1, 'omitnan');
            end

            highActivity(:,hi) = mean(highSideMean, 1, 'omitnan')';
        end

        activityDiffNow = nan(nPair, nLowSeg, nHighSeg);
        for li = 1:nLowSeg
            for hi = 1:nHighSeg
                activityDiffNow(:,li,hi) = highActivity(:,hi) - lowActivity(:,li);
            end
        end

        allWeights(end+1,:,:,:) = weightsNow; %#ok<SAGROW>
        allActivityDiff(end+1,:,:,:) = activityDiffNow; %#ok<SAGROW>
        allAucDiag(end+1,:,:) = aucDiagNow; %#ok<SAGROW>
        subjectList(end+1,1) = Decode.subject; %#ok<SAGROW>
        temporalDesign = Decode.temporalDesign;
        channelLabels = Decode.channelLabels;
    end

    if isempty(allWeights)
        warning('No usable %s results found for %s.', modelName, comparisonName);
        continue;
    end

    nSubject = size(allWeights, 1);
    nPair = size(allWeights, 2);
    nLowSeg = size(allWeights, 3);
    nHighSeg = size(allWeights, 4);

    groupWeights = squeeze(mean(allWeights, 1, 'omitnan'));
    groupActivityDiff = squeeze(mean(allActivityDiff, 1, 'omitnan'));
    if nLowSeg == 1
        groupWeights = reshape(groupWeights, [nPair, nLowSeg, nHighSeg]);
        groupActivityDiff = reshape(groupActivityDiff, [nPair, nLowSeg, nHighSeg]);
    end

    measures = {'LDA_weight', 'CDA_high_minus_low'};
    dataForMeasure = {groupWeights, groupActivityDiff};
    unitsForMeasure = {'a.u.', 'uV'};
    measureRowLabels = {'LDA w.', 'CDA H-L'};
    maxAbsByMeasure = nan(1, numel(measures));
    for measureIdx = 1:numel(measures)
        maxAbsVal = max(abs(dataForMeasure{measureIdx}), [], 'all', 'omitnan');
        if isempty(maxAbsVal) || isnan(maxAbsVal) || maxAbsVal == 0
            maxAbsVal = 1;
        end
        maxAbsByMeasure(measureIdx) = maxAbsVal;
    end

    for measureIdx = 1:numel(measures)
        measureName = measures{measureIdx};
        plotData = dataForMeasure{measureIdx};
        maxAbsVal = maxAbsByMeasure(measureIdx);

        fig = figure('Color', 'w', 'Position', [100 100 240*nHighSeg 220*nLowSeg]);
        colormap(fig, signedCmap);

        for li = 1:nLowSeg
            for hi = 1:nHighSeg
                subplot(nLowSeg, nHighSeg, (li - 1) * nHighSeg + hi);
                pairValues = plotData(:,li,hi);
                topoValues = [-pairValues(:); pairValues(:)];
                topoplot(topoValues, plotChanlocs, ...
                    'maplimits', [-maxAbsVal maxAbsVal], ...
                    'electrodes', 'on', ...
                    'style', 'both');

                lowWindow = temporalDesign.lowWindowsMs(li,:);
                highWindow = [temporalDesign.highSegmentStartsMs(hi), ...
                    temporalDesign.highSegmentStartsMs(hi) + temporalDesign.highSegmentWidthMs];
                if li == 1
                    title(sprintf('H%d %d-%d', hi, round(highWindow(1)), round(highWindow(2))), ...
                        'FontSize', 8, 'FontWeight', 'normal');
                end
                if hi == 1
                    text(-0.82, 0, sprintf('L%d\n%d-%d', li, round(lowWindow(1)), round(lowWindow(2))), ...
                        'HorizontalAlignment', 'right', ...
                        'VerticalAlignment', 'middle', ...
                        'FontSize', 8, ...
                        'FontWeight', 'bold', ...
                        'Clipping', 'off');
                end
            end
        end

        cb = colorbar;
        cb.Position = [0.93 0.18 0.015 0.64];
        cb.Label.String = unitsForMeasure{measureIdx};

        outBase = fullfile(outputDir, sprintf('%s_%s_%s_topoplot', ...
            comparisonName, modelName, measureName));
        savefig(fig, [outBase '.fig']);
        exportgraphics(fig, [outBase '.png'], 'Resolution', 300);
        close(fig);

        for li = 1:nLowSeg
            for hi = 1:nHighSeg
                for pi = 1:nPair
                    if strcmp(measureName, 'LDA_weight')
                        subjValues = squeeze(allWeights(:,pi,li,hi));
                    else
                        subjValues = squeeze(allActivityDiff(:,pi,li,hi));
                    end

                    nValid = sum(~isnan(subjValues));
                    semVal = std(subjValues, 0, 'omitnan') ./ sqrt(max(nValid, 1));
                    lowWindow = temporalDesign.lowWindowsMs(li,:);
                    highWindow = [temporalDesign.highSegmentStartsMs(hi), ...
                        temporalDesign.highSegmentStartsMs(hi) + temporalDesign.highSegmentWidthMs];

                    summaryRows(end+1,:) = {comparisonName, measureName, channelLabels{pi}, ...
                        li, lowWindow(1), lowWindow(2), hi, highWindow(1), highWindow(2), ...
                        nValid, mean(subjValues, 'omitnan'), semVal}; %#ok<SAGROW>
                end
            end
        end
    end

    combinedRows = nLowSeg * numel(measures);
    fig = figure('Color', 'w', 'Position', [80 80 250*nHighSeg 190*combinedRows]);
    colormap(fig, signedCmap);
    colorbarAxes = gobjects(numel(measures), 1);

    for measureIdx = 1:numel(measures)
        plotData = dataForMeasure{measureIdx};
        maxAbsVal = maxAbsByMeasure(measureIdx);

        for li = 1:nLowSeg
            rowIdx = (measureIdx - 1) * nLowSeg + li;
            for hi = 1:nHighSeg
                subplot(combinedRows, nHighSeg, (rowIdx - 1) * nHighSeg + hi);
                pairValues = plotData(:,li,hi);
                topoValues = [-pairValues(:); pairValues(:)];
                topoplot(topoValues, plotChanlocs, ...
                    'maplimits', [-maxAbsVal maxAbsVal], ...
                    'electrodes', 'on', ...
                    'style', 'both');

                lowWindow = temporalDesign.lowWindowsMs(li,:);
                highWindow = [temporalDesign.highSegmentStartsMs(hi), ...
                    temporalDesign.highSegmentStartsMs(hi) + temporalDesign.highSegmentWidthMs];
                if rowIdx == 1
                    title(sprintf('H%d %d-%d', hi, round(highWindow(1)), round(highWindow(2))), ...
                        'FontSize', 8, 'FontWeight', 'normal');
                end
                if hi == 1
                    text(-0.82, 0, sprintf('%s\nL%d\n%d-%d', measureRowLabels{measureIdx}, ...
                        li, round(lowWindow(1)), round(lowWindow(2))), ...
                        'HorizontalAlignment', 'right', ...
                        'VerticalAlignment', 'middle', ...
                        'FontSize', 8, ...
                        'FontWeight', 'bold', ...
                        'Clipping', 'off');
                end
                if li == 1 && hi == nHighSeg
                    colorbarAxes(measureIdx) = gca;
                end
            end
        end
    end

    for measureIdx = 1:numel(measures)
        cb = colorbar(colorbarAxes(measureIdx));
        cb.Label.String = unitsForMeasure{measureIdx};
        if numel(measures) == 2
            cb.Position = [0.93, 0.56 - (measureIdx - 1) * 0.42, 0.015, 0.28];
        else
            cb.Position = [0.93, 0.18, 0.015, 0.64];
        end
    end

    outBase = fullfile(outputDir, sprintf('%s_%s_LDA_weight_and_CDA_activity_topoplot', ...
        comparisonName, modelName));
    savefig(fig, [outBase '.fig']);
    exportgraphics(fig, [outBase '.png'], 'Resolution', 300);
    close(fig);

    subjectDataForMeasure = {allWeights, allActivityDiff};
    slopeByMeasure = cell(1, numel(measures));
    xTrend = (1:nHighSeg)';

    for measureIdx = 1:numel(measures)
        measureName = measures{measureIdx};
        subjData = subjectDataForMeasure{measureIdx};
        scoreMat = nan(nSubject, nLowSeg, nHighSeg);
        slopeMat = nan(nSubject, nLowSeg);

        for subjIdx = 1:nSubject
            for li = 1:nLowSeg
                for hi = 1:nHighSeg
                    pairValues = squeeze(subjData(subjIdx,:,li,hi));
                    scoreMat(subjIdx,li,hi) = -mean(pairValues, 'omitnan');

                    lowWindow = temporalDesign.lowWindowsMs(li,:);
                    highWindow = [temporalDesign.highSegmentStartsMs(hi), ...
                        temporalDesign.highSegmentStartsMs(hi) + temporalDesign.highSegmentWidthMs];
                    scoreRows(end+1,:) = {comparisonName, measureName, subjectList(subjIdx), ...
                        li, lowWindow(1), lowWindow(2), hi, highWindow(1), highWindow(2), ...
                        scoreMat(subjIdx,li,hi)}; %#ok<SAGROW>
                end

                y = squeeze(scoreMat(subjIdx,li,:));
                valid = ~isnan(xTrend) & ~isnan(y);
                if sum(valid) >= 2
                    pfit = polyfit(xTrend(valid), y(valid), 1);
                    slopeMat(subjIdx,li) = pfit(1);
                end
            end
        end

        slopeByMeasure{measureIdx} = slopeMat;
        for li = 1:nLowSeg
            slopes = slopeMat(:,li);
            slopes = slopes(~isnan(slopes));
            nValid = numel(slopes);

            if nValid >= 2
                [~, pSlope, ~, stSlope] = ttest(slopes, 0, 'Tail', 'right');
                meanSlope = mean(slopes, 'omitnan');
                semSlope = std(slopes, 0, 'omitnan') ./ sqrt(nValid);
                dzSlope = meanSlope ./ std(slopes, 0, 'omitnan');
                tSlope = stSlope.tstat;
                dfSlope = stSlope.df;
            else
                pSlope = NaN;
                meanSlope = NaN;
                semSlope = NaN;
                dzSlope = NaN;
                tSlope = NaN;
                dfSlope = NaN;
            end

            scoreFirst = squeeze(scoreMat(:,li,1));
            scoreLast = squeeze(scoreMat(:,li,nHighSeg));
            lowWindow = temporalDesign.lowWindowsMs(li,:);
            trendRows(end+1,:) = {comparisonName, measureName, li, lowWindow(1), lowWindow(2), ...
                nValid, mean(scoreFirst, 'omitnan'), mean(scoreLast, 'omitnan'), ...
                meanSlope, semSlope, tSlope, dfSlope, pSlope, dzSlope}; %#ok<SAGROW>
        end
    end

    aucSlopeMat = nan(nSubject, nLowSeg);
    for subjIdx = 1:nSubject
        for li = 1:nLowSeg
            for hi = 1:nHighSeg
                lowWindow = temporalDesign.lowWindowsMs(li,:);
                highWindow = [temporalDesign.highSegmentStartsMs(hi), ...
                    temporalDesign.highSegmentStartsMs(hi) + temporalDesign.highSegmentWidthMs];
                scoreRows(end+1,:) = {comparisonName, 'AUC_diag', subjectList(subjIdx), ...
                    li, lowWindow(1), lowWindow(2), hi, highWindow(1), highWindow(2), ...
                    allAucDiag(subjIdx,li,hi)}; %#ok<SAGROW>
            end

            y = squeeze(allAucDiag(subjIdx,li,:));
            valid = ~isnan(xTrend) & ~isnan(y);
            if sum(valid) >= 2
                pfit = polyfit(xTrend(valid), y(valid), 1);
                aucSlopeMat(subjIdx,li) = pfit(1);
            end
        end
    end

    for li = 1:nLowSeg
        slopes = aucSlopeMat(:,li);
        slopes = slopes(~isnan(slopes));
        nValid = numel(slopes);
        if nValid >= 2
            [~, pSlope, ~, stSlope] = ttest(slopes, 0, 'Tail', 'right');
            meanSlope = mean(slopes, 'omitnan');
            semSlope = std(slopes, 0, 'omitnan') ./ sqrt(nValid);
            dzSlope = meanSlope ./ std(slopes, 0, 'omitnan');
            tSlope = stSlope.tstat;
            dfSlope = stSlope.df;
        else
            pSlope = NaN;
            meanSlope = NaN;
            semSlope = NaN;
            dzSlope = NaN;
            tSlope = NaN;
            dfSlope = NaN;
        end

        aucFirst = squeeze(allAucDiag(:,li,1));
        aucLast = squeeze(allAucDiag(:,li,nHighSeg));
        lowWindow = temporalDesign.lowWindowsMs(li,:);
        aucTrendRows(end+1,:) = {comparisonName, li, lowWindow(1), lowWindow(2), ...
            nValid, mean(aucFirst, 'omitnan'), mean(aucLast, 'omitnan'), ...
            meanSlope, semSlope, tSlope, dfSlope, pSlope, dzSlope}; %#ok<SAGROW>

        for measureIdx = 1:numel(measures)
            measureName = measures{measureIdx};
            topoSlope = slopeByMeasure{measureIdx}(:,li);
            aucSlope = aucSlopeMat(:,li);
            valid = ~isnan(topoSlope) & ~isnan(aucSlope);
            nCorr = sum(valid);
            if nCorr >= 3
                [rhoVal, pCorr] = corr(topoSlope(valid), aucSlope(valid), ...
                    'Type', 'Spearman', 'Rows', 'complete');
            else
                rhoVal = NaN;
                pCorr = NaN;
            end

            lowWindow = temporalDesign.lowWindowsMs(li,:);
            aucLinkRows(end+1,:) = {comparisonName, measureName, li, lowWindow(1), lowWindow(2), ...
                nCorr, rhoVal, pCorr}; %#ok<SAGROW>
        end
    end

    fprintf('%s: saved topographies for %d subjects: %s\n', ...
        comparisonName, nSubject, sprintf('%d ', subjectList));
end

if ~isempty(summaryRows)
    Summary = cell2table(summaryRows, 'VariableNames', { ...
        'Comparison', 'Measure', 'ChannelPair', ...
        'LowSegment', 'LowStartMs', 'LowEndMs', ...
        'HighSegment', 'HighStartMs', 'HighEndMs', ...
        'N', 'Mean', 'SEM'});
    outCsv = fullfile(outputDir, 'CDA_segment_topography_summary.csv');
    try
        writetable(Summary, outCsv);
    catch ME
        [outPath, outName, outExt] = fileparts(outCsv);
        outCsvAlt = fullfile(outPath, sprintf('%s_%s%s', outName, ...
            char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')), outExt));
        warning('Could not write %s (%s). Writing %s instead.', outCsv, ME.message, outCsvAlt);
        writetable(Summary, outCsvAlt);
    end
end

if ~isempty(scoreRows)
    Scores = cell2table(scoreRows, 'VariableNames', { ...
        'Comparison', 'Measure', 'Subject', ...
        'LowSegment', 'LowStartMs', 'LowEndMs', ...
        'HighSegment', 'HighStartMs', 'HighEndMs', ...
        'CanonicalHighStateStrength'});
    outCsv = fullfile(outputDir, 'CDA_segment_topography_subject_scores.csv');
    try
        writetable(Scores, outCsv);
    catch ME
        [outPath, outName, outExt] = fileparts(outCsv);
        outCsvAlt = fullfile(outPath, sprintf('%s_%s%s', outName, ...
            char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')), outExt));
        warning('Could not write %s (%s). Writing %s instead.', outCsv, ME.message, outCsvAlt);
        writetable(Scores, outCsvAlt);
    end
end

if ~isempty(trendRows)
    TrendStats = cell2table(trendRows, 'VariableNames', { ...
        'Comparison', 'Measure', 'LowSegment', 'LowStartMs', 'LowEndMs', ...
        'N', 'MeanH1', 'MeanHLast', ...
        'MeanSlopePerHighSegment', 'SEMSlope', 'tStat', 'df', 'pSlopeRight', 'dz'});

    TrendStats.pSlopeRightFDR = nan(height(TrendStats), 1);
    validP = ~isnan(TrendStats.pSlopeRight);
    if any(validP)
        validIdx = find(validP);
        [pSorted, sortIdx] = sort(TrendStats.pSlopeRight(validP));
        nP = numel(pSorted);
        qSorted = pSorted .* nP ./ (1:nP)';
        qSorted = flipud(cummin(flipud(qSorted)));
        qSorted(qSorted > 1) = 1;
        TrendStats.pSlopeRightFDR(validIdx(sortIdx)) = qSorted;
    end
    outCsv = fullfile(outputDir, 'CDA_segment_topography_trend_stats.csv');
    try
        writetable(TrendStats, outCsv);
    catch ME
        [outPath, outName, outExt] = fileparts(outCsv);
        outCsvAlt = fullfile(outPath, sprintf('%s_%s%s', outName, ...
            char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')), outExt));
        warning('Could not write %s (%s). Writing %s instead.', outCsv, ME.message, outCsvAlt);
        writetable(TrendStats, outCsvAlt);
    end
end

if ~isempty(aucTrendRows)
    AUCTrendStats = cell2table(aucTrendRows, 'VariableNames', { ...
        'Comparison', 'LowSegment', 'LowStartMs', 'LowEndMs', ...
        'N', 'MeanH1', 'MeanHLast', ...
        'MeanSlopePerHighSegment', 'SEMSlope', 'tStat', 'df', 'pSlopeRight', 'dz'});

    AUCTrendStats.pSlopeRightFDR = nan(height(AUCTrendStats), 1);
    validP = ~isnan(AUCTrendStats.pSlopeRight);
    if any(validP)
        validIdx = find(validP);
        [pSorted, sortIdx] = sort(AUCTrendStats.pSlopeRight(validP));
        nP = numel(pSorted);
        qSorted = pSorted .* nP ./ (1:nP)';
        qSorted = flipud(cummin(flipud(qSorted)));
        qSorted(qSorted > 1) = 1;
        AUCTrendStats.pSlopeRightFDR(validIdx(sortIdx)) = qSorted;
    end
    outCsv = fullfile(outputDir, 'CDA_segment_AUC_diag_trend_stats.csv');
    try
        writetable(AUCTrendStats, outCsv);
    catch ME
        [outPath, outName, outExt] = fileparts(outCsv);
        outCsvAlt = fullfile(outPath, sprintf('%s_%s%s', outName, ...
            char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')), outExt));
        warning('Could not write %s (%s). Writing %s instead.', outCsv, ME.message, outCsvAlt);
        writetable(AUCTrendStats, outCsvAlt);
    end
end

if ~isempty(aucLinkRows)
    AUCLinkStats = cell2table(aucLinkRows, 'VariableNames', { ...
        'Comparison', 'TopographyMeasure', 'LowSegment', 'LowStartMs', 'LowEndMs', ...
        'N', 'SpearmanRho', 'pSpearman'});

    AUCLinkStats.pSpearmanFDR = nan(height(AUCLinkStats), 1);
    validP = ~isnan(AUCLinkStats.pSpearman);
    if any(validP)
        validIdx = find(validP);
        [pSorted, sortIdx] = sort(AUCLinkStats.pSpearman(validP));
        nP = numel(pSorted);
        qSorted = pSorted .* nP ./ (1:nP)';
        qSorted = flipud(cummin(flipud(qSorted)));
        qSorted(qSorted > 1) = 1;
        AUCLinkStats.pSpearmanFDR(validIdx(sortIdx)) = qSorted;
    end
    outCsv = fullfile(outputDir, 'CDA_segment_topography_AUC_slope_link_stats.csv');
    try
        writetable(AUCLinkStats, outCsv);
    catch ME
        [outPath, outName, outExt] = fileparts(outCsv);
        outCsvAlt = fullfile(outPath, sprintf('%s_%s%s', outName, ...
            char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')), outExt));
        warning('Could not write %s (%s). Writing %s instead.', outCsv, ME.message, outCsvAlt);
        writetable(AUCLinkStats, outCsvAlt);
    end
end

fprintf('CDA segment topography outputs saved to:\n%s\n', outputDir);

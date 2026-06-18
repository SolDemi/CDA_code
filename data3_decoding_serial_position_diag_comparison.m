%% Serial-position diagonal decoding comparison for data3 sequential LDA
% This script reads saved data3 sequential LDA results and does not rerun decoding.
% The goal is to test whether high-set-size serial-position decoding strength
% changes from position 1 to 6, especially whether position 6 is stronger than
% position 5. For setsize3-vs-6, low-set-size serial positions are kept as
% separate conditions instead of being averaged together.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
addpath(codeDir);

cfg = struct();
cfg.analysisMode = 'maintOnly';
cfg.metric = 'AUC';
cfg.chance = 0.5;
cfg.models = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
cfg.comparisons = { ...
    sprintf('setsize1_vs6_%s', cfg.analysisMode), fullfile(dataDir, sprintf('decoding_LDA_setsize1_vs6_segments_%s', cfg.analysisMode)); ...
    sprintf('setsize3_vs6_%s', cfg.analysisMode), fullfile(dataDir, sprintf('decoding_LDA_setsize3_vs6_segments_%s', cfg.analysisMode))};
cfg.highPositions = 1:6;
cfg.adjacentPairs = [1 2; 2 3; 3 4; 4 5; 5 6];
cfg.subjectInclusion = 'original';
cfg.figureDpi = 300;
cfg.plotSignificanceAlpha = 0.05;
cfg.plotSignificanceP = 'PairedP_TwoTailed_Bonferroni5';
cfg.plotSignificanceLabel = 'Bonferroni two-tailed';

outputDir = fullfile(dataDir, 'serial_position_diagonal_comparison');
figureDir = fullfile(outputDir, 'figures');
if ~isfolder(outputDir), mkdir(outputDir); end
if ~isfolder(figureDir), mkdir(figureDir); end

staleFigureNames = {};
for ci = 1:size(cfg.comparisons, 1)
    staleFigureNames{end+1, 1} = sprintf('data3_%s_%s_serial_position_diag_mean.fig', cfg.comparisons{ci, 1}, cfg.metric); %#ok<SAGROW>
    staleFigureNames{end+1, 1} = sprintf('data3_%s_%s_serial_position_diag_mean.png', cfg.comparisons{ci, 1}, cfg.metric); %#ok<SAGROW>
    staleFigureNames{end+1, 1} = sprintf('data3_%s_%s_adjacent_diag_differences.fig', cfg.comparisons{ci, 1}, cfg.metric); %#ok<SAGROW>
    staleFigureNames{end+1, 1} = sprintf('data3_%s_%s_adjacent_diag_differences.png', cfg.comparisons{ci, 1}, cfg.metric); %#ok<SAGROW>
end
for staleIndex = 1:numel(staleFigureNames)
    stalePath = fullfile(figureDir, staleFigureNames{staleIndex});
    if isfile(stalePath)
        delete(stalePath);
    end
end

originalSubjects = str2double(data3_original_subjects());
modelColors = lines(numel(cfg.models));
comparisonLabels = {'SS1 vs SS6', 'SS3 vs SS6'};

% For this question, the diagonal within each low/high segment block is the
% same-time train-test decoding score for that high-set-size serial position.
% Off-diagonal cells describe temporal generalization and are not used here.
% In setsize3-vs-6, the three low-set-size segment blocks are not pooled
% because pooling changes the reference low serial position across the curve.

subjectTables = {};
inclusionTables = {};

%% Extract subject-level diagonal means for each high serial position
for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    comparisonDir = cfg.comparisons{ci, 2};

    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        modelDir = fullfile(comparisonDir, modelName);
        files = dir(fullfile(modelDir, 'sub*.mat'));

        if isempty(files)
            warning('No files found for %s %s in %s.', comparisonName, modelName, modelDir);
            continue;
        end

        for fi = 1:numel(files)
            fpath = fullfile(files(fi).folder, files(fi).name);
            Sload = load(fpath, modelName);
            if ~isfield(Sload, modelName)
                warning('%s does not contain %s.', fpath, modelName);
                continue;
            end
            Decode = Sload.(modelName);

            if isfield(Decode, 'subject') && ~isempty(Decode.subject)
                if isnumeric(Decode.subject)
                    subject = double(Decode.subject);
                else
                    subject = str2double(regexprep(char(Decode.subject), '^sub', '', 'ignorecase'));
                end
            else
                tok = regexp(files(fi).name, '^sub(\d+)\.mat$', 'tokens', 'once');
                if isempty(tok)
                    subject = NaN;
                else
                    subject = str2double(tok{1});
                end
            end

            includeSubject = true;
            includeReason = 'allDecoded';
            if strcmpi(cfg.subjectInclusion, 'original')
                includeSubject = any(originalSubjects == subject);
                includeReason = 'originalSubjects';
            end

            I = table();
            I.Subject = subject;
            I.Comparison = {comparisonName};
            I.Model = {modelName};
            I.File = {fpath};
            I.Included = includeSubject;
            I.InclusionReason = {includeReason};
            inclusionTables{end+1, 1} = I; %#ok<SAGROW>

            if ~includeSubject
                continue;
            end

            if ~isfield(Decode, 'temporalDesign')
                warning('%s is missing temporalDesign.', fpath);
                continue;
            end
            design = Decode.temporalDesign;
            nLowSeg = size(design.lowWindowsMs, 1);
            nHighSeg = numel(design.highSegmentStartsMs);

            if ~isfield(Decode, cfg.metric)
                warning('%s is missing metric %s.', fpath, cfg.metric);
                continue;
            end

            M = Decode.(cfg.metric);
            if mod(size(M, 1), nLowSeg) ~= 0 || mod(size(M, 2), nHighSeg) ~= 0
                warning('%s %s has matrix size %dx%d that does not match %d low x %d high segments.', ...
                    fpath, cfg.metric, size(M, 1), size(M, 2), nLowSeg, nHighSeg);
                continue;
            end

            lowBlockSize = size(M, 1) / nLowSeg;
            highBlockSize = size(M, 2) / nHighSeg;
            nDiagSamples = min(lowBlockSize, highBlockSize);

            highSegmentWidthMs = nan;
            if isfield(design, 'highSegmentWidthMs')
                highSegmentWidthMs = design.highSegmentWidthMs;
            end

            for li = 1:nLowSeg
                for hi = 1:nHighSeg
                    rowIdx = (li - 1) * lowBlockSize + (1:lowBlockSize);
                    colIdx = (hi - 1) * highBlockSize + (1:highBlockSize);
                    block = M(rowIdx, colIdx);

                    T = table();
                    T.Subject = subject;
                    T.Comparison = {comparisonName};
                    T.Model = {modelName};
                    T.Metric = {cfg.metric};
                    T.LowSerialPosition = li;
                    T.LowSegmentStartMs = design.lowWindowsMs(li, 1);
                    T.LowSegmentEndMs = design.lowWindowsMs(li, 2);
                    T.HighSerialPosition = hi;
                    T.HighSegmentStartMs = design.highSegmentStartsMs(hi);
                    T.HighSegmentEndMs = design.highSegmentStartsMs(hi) + highSegmentWidthMs;
                    T.MeanDiag = mean(diag(block(1:nDiagSamples, 1:nDiagSamples)), 'omitnan');
                    T.MeanDiagMinusChance = T.MeanDiag - cfg.chance;
                    T.NLowSegmentsAveraged = 1;
                    T.NLowSegmentsInComparison = nLowSeg;
                    T.NDiagSamplesPerLowSegment = nDiagSamples;
                    T.File = {fpath};

                    subjectTables{end+1, 1} = T; %#ok<SAGROW>
                end
            end
        end
    end
end

if isempty(subjectTables)
    error('No subject-level diagonal decoding rows were built.');
end

SubjectDiag = vertcat(subjectTables{:});
InclusionSummary = vertcat(inclusionTables{:});

writetable(SubjectDiag, fullfile(outputDir, 'data3_serial_position_diag_subject_summary.csv'));
writetable(InclusionSummary, fullfile(outputDir, 'data3_serial_position_diag_inclusion_summary.csv'));

%% Group mean table for each comparison x model x low segment x high position
positionTables = {};

for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(SubjectDiag.Comparison, comparisonName) & ...
            strcmp(SubjectDiag.Metric, cfg.metric) & ...
            strcmp(SubjectDiag.Model, modelName);
        TmAll = SubjectDiag(rows, :);
        lowPositions = unique(TmAll.LowSerialPosition)';

        for lowIndex = 1:numel(lowPositions)
            lowPos = lowPositions(lowIndex);
            Tm = TmAll(TmAll.LowSerialPosition == lowPos, :);
            subjects = unique(Tm.Subject);

            for pi = 1:numel(cfg.highPositions)
                pos = cfg.highPositions(pi);
                valueBySubject = nan(numel(subjects), 1);
                for si = 1:numel(subjects)
                    posRows = Tm.Subject == subjects(si) & Tm.HighSerialPosition == pos;
                    if any(posRows)
                        valueBySubject(si) = mean(Tm.MeanDiag(posRows), 'omitnan');
                    end
                end

                validRows = ~isnan(valueBySubject);
                nValid = sum(validRows);
                meanDiag = mean(valueBySubject(validRows), 'omitnan');
                semDiag = std(valueBySubject(validRows), 0, 'omitnan') ./ sqrt(nValid);

                P = table();
                P.Comparison = {comparisonName};
                P.Model = {modelName};
                P.Metric = {cfg.metric};
                P.LowSerialPosition = lowPos;
                P.LowSegmentStartMs = Tm.LowSegmentStartMs(find(Tm.LowSerialPosition == lowPos, 1));
                P.LowSegmentEndMs = Tm.LowSegmentEndMs(find(Tm.LowSerialPosition == lowPos, 1));
                P.HighSerialPosition = pos;
                P.NSubject = nValid;
                P.MeanDiag = meanDiag;
                P.SEMDiag = semDiag;
                P.MeanDiagMinusChance = meanDiag - cfg.chance;

                positionTables{end+1, 1} = P; %#ok<SAGROW>
            end
        end
    end
end

PositionSummary = vertcat(positionTables{:});
writetable(PositionSummary, fullfile(outputDir, 'data3_serial_position_diag_position_summary.csv'));

%% Paired adjacent serial-position tests within each comparison x model x low segment
adjacentTables = {};

for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(SubjectDiag.Comparison, comparisonName) & ...
            strcmp(SubjectDiag.Metric, cfg.metric) & ...
            strcmp(SubjectDiag.Model, modelName);
        TmAll = SubjectDiag(rows, :);
        lowPositions = unique(TmAll.LowSerialPosition)';

        for lowIndex = 1:numel(lowPositions)
            lowPos = lowPositions(lowIndex);
            Tm = TmAll(TmAll.LowSerialPosition == lowPos, :);
            subjects = unique(Tm.Subject);

            valueBySubject = nan(numel(subjects), numel(cfg.highPositions));
            for si = 1:numel(subjects)
                for pi = 1:numel(cfg.highPositions)
                    pos = cfg.highPositions(pi);
                    posRows = Tm.Subject == subjects(si) & Tm.HighSerialPosition == pos;
                    if any(posRows)
                        valueBySubject(si, pi) = mean(Tm.MeanDiag(posRows), 'omitnan');
                    end
                end
            end

            for pairIndex = 1:size(cfg.adjacentPairs, 1)
                posA = cfg.adjacentPairs(pairIndex, 1);
                posB = cfg.adjacentPairs(pairIndex, 2);
                x = valueBySubject(:, posA);
                y = valueBySubject(:, posB);
                validRows = ~isnan(x) & ~isnan(y);
                diffVal = y(validRows) - x(validRows);

                nValid = sum(validRows);
                meanA = mean(x(validRows), 'omitnan');
                meanB = mean(y(validRows), 'omitnan');
                meanDiff = mean(diffVal, 'omitnan');
                semDiff = std(diffVal, 0, 'omitnan') ./ sqrt(nValid);
                tStat = nan;
                pBoth = nan;
                pRight = nan;
                ciLow = nan;
                ciHigh = nan;
                cohenDz = nan;

                if nValid >= 2 && std(diffVal, 0, 'omitnan') > 0
                    [~, pBoth, ciBoth, statsBoth] = ttest(diffVal, 0, 'Tail', 'both');
                    [~, pRight] = ttest(diffVal, 0, 'Tail', 'right');
                    tStat = statsBoth.tstat;
                    ciLow = ciBoth(1);
                    ciHigh = ciBoth(2);
                    cohenDz = meanDiff ./ std(diffVal, 0, 'omitnan');
                end

                A = table();
                A.Comparison = {comparisonName};
                A.Model = {modelName};
                A.Metric = {cfg.metric};
                A.LowSerialPosition = lowPos;
                A.LowSegmentStartMs = Tm.LowSegmentStartMs(find(Tm.LowSerialPosition == lowPos, 1));
                A.LowSegmentEndMs = Tm.LowSegmentEndMs(find(Tm.LowSerialPosition == lowPos, 1));
                A.PositionA = posA;
                A.PositionB = posB;
                A.Contrast = {sprintf('pos%d_minus_pos%d', posB, posA)};
                A.NSubject = nValid;
                A.MeanPositionA = meanA;
                A.MeanPositionB = meanB;
                A.MeanDiff = meanDiff;
                A.SEMDiff = semDiff;
                A.T = tStat;
                A.PairedP_TwoTailed = pBoth;
                A.PairedP_RightTail_PosBGreater = pRight;
                A.PairedP_TwoTailed_Bonferroni5 = min(pBoth * size(cfg.adjacentPairs, 1), 1);
                A.PairedP_RightTail_Bonferroni5 = min(pRight * size(cfg.adjacentPairs, 1), 1);
                A.CILowDiff = ciLow;
                A.CIHighDiff = ciHigh;
                A.CohenDz = cohenDz;

                adjacentTables{end+1, 1} = A; %#ok<SAGROW>
            end
        end
    end
end

AdjacentSummary = vertcat(adjacentTables{:});
writetable(AdjacentSummary, fullfile(outputDir, 'data3_serial_position_diag_adjacent_comparison_summary.csv'));

%% Plot serial-position diagonal decoding curves
for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    lowPositions = unique(SubjectDiag.LowSerialPosition(strcmp(SubjectDiag.Comparison, comparisonName)))';
    nLowPlot = numel(lowPositions);

    figHeight = max(420, 290 * nLowPlot);
    fig = figure('Color', 'w', 'Position', [80 60 1500 figHeight]);
    for lowIndex = 1:nLowPlot
        lowPos = lowPositions(lowIndex);
        for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(SubjectDiag.Comparison, comparisonName) & ...
            strcmp(SubjectDiag.Metric, cfg.metric) & ...
            strcmp(SubjectDiag.Model, modelName) & ...
            SubjectDiag.LowSerialPosition == lowPos;
        Tm = SubjectDiag(rows, :);
        subjects = unique(Tm.Subject);
        valueBySubject = nan(numel(subjects), numel(cfg.highPositions));

        for si = 1:numel(subjects)
            for pi = 1:numel(cfg.highPositions)
                pos = cfg.highPositions(pi);
                posRows = Tm.Subject == subjects(si) & Tm.HighSerialPosition == pos;
                if any(posRows)
                    valueBySubject(si, pi) = mean(Tm.MeanDiag(posRows), 'omitnan');
                end
            end
        end

        meanByPosition = mean(valueBySubject, 1, 'omitnan');
        semByPosition = std(valueBySubject, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(valueBySubject), 1));

        subplot(nLowPlot, numel(cfg.models), (lowIndex - 1) * numel(cfg.models) + mi);
        hold on;
        plot(cfg.highPositions, valueBySubject', '-', ...
            'Color', [0.82 0.82 0.82], ...
            'LineWidth', 0.5, ...
            'HandleVisibility', 'off');
        errorbar(cfg.highPositions, meanByPosition, semByPosition, 'o-', ...
            'Color', modelColors(mi,:), ...
            'MarkerFaceColor', modelColors(mi,:), ...
            'LineWidth', 1.8, ...
            'MarkerSize', 5);
        yline(cfg.chance, ':k', 'Chance', 'HandleVisibility', 'off');

        yAll = [valueBySubject(:); meanByPosition(:) + semByPosition(:); meanByPosition(:) - semByPosition(:)];
        yAll = yAll(~isnan(yAll));
        if isempty(yAll)
            ylim([cfg.chance - 0.05, cfg.chance + 0.05]);
        else
            yMin = min([yAll; cfg.chance]);
            yMax = max([yAll; cfg.chance]);
            yPad = max((yMax - yMin) * 0.18, 0.02);
            ylim([yMin - yPad, yMax + yPad]);
        end

        yl = ylim;
        ySig = yl(2) - 0.08 * diff(yl);
        for pairIndex = 1:size(cfg.adjacentPairs, 1)
            posA = cfg.adjacentPairs(pairIndex, 1);
            posB = cfg.adjacentPairs(pairIndex, 2);
            statRows = strcmp(AdjacentSummary.Comparison, comparisonName) & ...
                strcmp(AdjacentSummary.Metric, cfg.metric) & ...
                strcmp(AdjacentSummary.Model, modelName) & ...
                AdjacentSummary.LowSerialPosition == lowPos & ...
                AdjacentSummary.PositionA == posA & ...
                AdjacentSummary.PositionB == posB;
            if ~any(statRows)
                continue;
            end
            statIndex = find(statRows, 1);
            pPlot = AdjacentSummary.(cfg.plotSignificanceP)(statIndex);
            if isnan(pPlot) || pPlot >= cfg.plotSignificanceAlpha
                continue;
            end
            if pPlot < 0.001
                starText = '***';
            elseif pPlot < 0.01
                starText = '**';
            else
                starText = '*';
            end
            plot([posA posB], [ySig ySig], 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off');
            text(mean([posA posB]), ySig + 0.02 * diff(yl), starText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontName', 'Arial', ...
                'FontSize', 10);
        end

        xlim([0.75 6.25]);
        set(gca, 'XTick', cfg.highPositions, ...
            'TickDir', 'out', ...
            'FontName', 'Arial', ...
            'FontSize', 9);
        if lowIndex == nLowPlot
            xlabel('Set-size-6 serial position');
        end
        if mi == 1
            ylabel(sprintf('Low pos %d\nMean diagonal %s', lowPos, cfg.metric));
        end
        if lowIndex == 1
            title(modelName, 'Interpreter', 'none');
        end
        box off;
        end
    end

    sgtitle(sprintf('%s %s serial-position diagonal decoding by low segment (%s stars)', ...
        comparisonLabels{ci}, cfg.metric, cfg.plotSignificanceLabel), ...
        'FontName', 'Arial', ...
        'FontWeight', 'bold', ...
        'Interpreter', 'none');

    fileBase = sprintf('data3_%s_%s_serial_position_diag_by_low_segment', comparisonName, cfg.metric);
    savefig(fig, fullfile(figureDir, [fileBase '.fig']));
    print(fig, fullfile(figureDir, [fileBase '.png']), '-dpng', sprintf('-r%d', cfg.figureDpi));
    close(fig);
end

%% Plot adjacent serial-position differences
for ci = 1:size(cfg.comparisons, 1)
    comparisonName = cfg.comparisons{ci, 1};
    lowPositions = unique(AdjacentSummary.LowSerialPosition(strcmp(AdjacentSummary.Comparison, comparisonName)))';
    nLowPlot = numel(lowPositions);

    figHeight = max(420, 290 * nLowPlot);
    fig = figure('Color', 'w', 'Position', [80 60 1500 figHeight]);
    for lowIndex = 1:nLowPlot
        lowPos = lowPositions(lowIndex);
        for mi = 1:numel(cfg.models)
        modelName = cfg.models{mi};
        rows = strcmp(AdjacentSummary.Comparison, comparisonName) & ...
            strcmp(AdjacentSummary.Metric, cfg.metric) & ...
            strcmp(AdjacentSummary.Model, modelName) & ...
            AdjacentSummary.LowSerialPosition == lowPos;
        Tm = AdjacentSummary(rows, :);

        subplot(nLowPlot, numel(cfg.models), (lowIndex - 1) * numel(cfg.models) + mi);
        hold on;
        x = 1:size(cfg.adjacentPairs, 1);
        bar(x, Tm.MeanDiff, ...
            'FaceColor', modelColors(mi,:), ...
            'EdgeColor', [0.15 0.15 0.15], ...
            'LineWidth', 0.5);
        errorbar(x, Tm.MeanDiff, Tm.SEMDiff, 'k.', 'LineWidth', 1.0);
        yline(0, ':k', 'HandleVisibility', 'off');

        yAll = [Tm.MeanDiff + Tm.SEMDiff; Tm.MeanDiff - Tm.SEMDiff; 0];
        yAll = yAll(~isnan(yAll));
        if isempty(yAll)
            ylim([-0.03 0.03]);
        else
            yMin = min(yAll);
            yMax = max(yAll);
            yPad = max((yMax - yMin) * 0.22, 0.01);
            ylim([yMin - yPad, yMax + yPad]);
        end

        yl = ylim;
        for pairIndex = 1:height(Tm)
            pPlot = Tm.(cfg.plotSignificanceP)(pairIndex);
            if isnan(pPlot) || pPlot >= cfg.plotSignificanceAlpha
                continue;
            end
            if pPlot < 0.001
                starText = '***';
            elseif pPlot < 0.01
                starText = '**';
            else
                starText = '*';
            end

            yText = Tm.MeanDiff(pairIndex) + Tm.SEMDiff(pairIndex) + 0.04 * diff(yl);
            if Tm.MeanDiff(pairIndex) < 0
                yText = Tm.MeanDiff(pairIndex) - Tm.SEMDiff(pairIndex) - 0.04 * diff(yl);
            end
            text(x(pairIndex), yText, starText, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontName', 'Arial', ...
                'FontSize', 10);
        end

        contrastLabels = cell(height(Tm), 1);
        for pairIndex = 1:height(Tm)
            contrastLabels{pairIndex} = sprintf('%d-%d', Tm.PositionB(pairIndex), Tm.PositionA(pairIndex));
        end

        xlim([0.4 size(cfg.adjacentPairs, 1) + 0.6]);
        set(gca, 'XTick', x, ...
            'XTickLabel', contrastLabels, ...
            'TickDir', 'out', ...
            'FontName', 'Arial', ...
            'FontSize', 9);
        if lowIndex == nLowPlot
            xlabel('Adjacent serial-position contrast');
        end
        if mi == 1
            ylabel(sprintf('Low pos %d\nLater - earlier %s', lowPos, cfg.metric));
        end
        if lowIndex == 1
            title(modelName, 'Interpreter', 'none');
        end
        box off;
        end
    end

    sgtitle(sprintf('%s %s adjacent serial-position differences by low segment (%s stars)', ...
        comparisonLabels{ci}, cfg.metric, cfg.plotSignificanceLabel), ...
        'FontName', 'Arial', ...
        'FontWeight', 'bold', ...
        'Interpreter', 'none');

    fileBase = sprintf('data3_%s_%s_adjacent_diag_differences_by_low_segment', comparisonName, cfg.metric);
    savefig(fig, fullfile(figureDir, [fileBase '.fig']));
    print(fig, fullfile(figureDir, [fileBase '.png']), '-dpng', sprintf('-r%d', cfg.figureDpi));
    close(fig);
end

fprintf('Saved serial-position diagonal decoding summaries to:\n%s\n', outputDir);
fprintf('Saved serial-position diagonal decoding figures to:\n%s\n', figureDir);

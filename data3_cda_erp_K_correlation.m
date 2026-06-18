%% data3 CDA ERP scatter plots against set-size-matched K
% SS3 uses the 3rd maintenance window; SS6 uses the 6th maintenance window.
% One subject contributes one mean CDA value and one behavior K value per set size.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
cdaDir = fullfile(dataDir, 'cda_alpha');
behaviorDir = fullfile(dataDir, 'Behavior_data_script', 'Beh_data');
resultDir = fullfile(dataDir, 'cda_erp_K_correlation');
figureDir = fullfile(resultDir, 'figures');

addpath(codeDir);
if ~isfolder(resultDir), mkdir(resultDir); end
if ~isfolder(figureDir), mkdir(figureDir); end

oldFiles = { ...
    'data3_cda_erp_K_behavior_summary.csv', ...
    'data3_cda_erp_K_correlation.mat', ...
    'data3_cda_erp_K_timepoint_correlations.csv', ...
    'data3_cda_erp_K_window_correlations.csv', ...
    'data3_cda_erp_K_window_subject_metrics.csv', ...
    'data3_cda_erp_K_scatter_subject_metrics.csv', ...
    'data3_cda_erp_K_scatter_correlation.csv'};
for fi = 1:numel(oldFiles)
    oldPath = fullfile(resultDir, oldFiles{fi});
    if isfile(oldPath)
        delete(oldPath);
    end
end

oldFigureFiles = { ...
    'data3_cda_erp_KCorrected_timepoint_correlation.fig', ...
    'data3_cda_erp_KCorrected_timepoint_correlation.png', ...
    'data3_cda_erp_KCorrected_window_scatter.fig', ...
    'data3_cda_erp_KCorrected_window_scatter.png'};
for fi = 1:numel(oldFigureFiles)
    oldPath = fullfile(figureDir, oldFigureFiles{fi});
    if isfile(oldPath)
        delete(oldPath);
    end
end

cfg = data3_default_cfg();
setSizes = [3 6];
behaviorBlocks = 1:4;
targetSerialPosition = [3 6];

cdaFiles = dir(fullfile(cdaDir, 'sub*.mat'));
cdaFiles = data3_filter_subject_mat_files(cdaFiles, data3_subject_filter());
if isempty(cdaFiles)
    error('No data3 cda_alpha/sub*.mat files found in %s.', cdaDir);
end

SubjectMetrics = table();

for fi = 1:numel(cdaFiles)
    cdaFile = fullfile(cdaFiles(fi).folder, cdaFiles(fi).name);
    C = load(cdaFile, 'cda');
    subject = C.cda.subject;
    fprintf('data3 CDA ERP-K scatter: sub%d\n', subject);

    AllBehavior = table();
    for bi = 1:numel(behaviorBlocks)
        blockNum = behaviorBlocks(bi);
        behaviorFile = fullfile(behaviorDir, sprintf('cda_cVl_serial_data%d_%d.mat', subject, blockNum));
        if ~isfile(behaviorFile)
            warning('Missing behavior file for subject %d block %d: %s', subject, blockNum, behaviorFile);
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
        BlockBehavior.Subject = repmat(subject, nBehTrial, 1);
        BlockBehavior.Block = repmat(blockNum, nBehTrial, 1);
        BlockBehavior.BlockTrial = (1:nBehTrial)';
        BlockBehavior.SetSize = behData.set_size(:);
        BlockBehavior.Change = behData.change(:);
        BlockBehavior.Response = behData.resp(:);
        BlockBehavior.Correct = behData.acc(:);
        AllBehavior = [AllBehavior; BlockBehavior]; %#ok<AGROW>
    end

    for seti = 1:numel(setSizes)
        setSize = setSizes(seti);
        serialPosition = targetSerialPosition(seti);
        setField = sprintf('setsize%d', setSize);
        loadStr = num2str(setSize);
        setIdx = find([1 3 6] == setSize, 1);
        windowsMs = cfg.cdaItemWindowsBySetSizeMs{setIdx};
        windowMs = windowsMs(serialPosition, :);
        timeMs = C.cda.timeBySetSize.(setField)(:);

        leftL = C.cda.trial.(sprintf('left_L_%s', loadStr));
        rightL = C.cda.trial.(sprintf('right_L_%s', loadStr));
        leftR = C.cda.trial.(sprintf('left_R_%s', loadStr));
        rightR = C.cda.trial.(sprintf('right_R_%s', loadStr));

        cdaLeftTrial = mean(rightL - leftL, 2, 'omitnan');
        cdaRightTrial = mean(leftR - rightR, 2, 'omitnan');
        cdaLeftTrial = reshape(cdaLeftTrial, size(cdaLeftTrial, 1), size(cdaLeftTrial, 3));
        cdaRightTrial = reshape(cdaRightTrial, size(cdaRightTrial, 1), size(cdaRightTrial, 3));
        cdaTrial = [cdaLeftTrial; cdaRightTrial];
        cdaMean = mean(cdaTrial, 1, 'omitnan');
        windowIdx = timeMs >= windowMs(1) & timeMs <= windowMs(2);
        cdaWindow = mean(cdaMean(windowIdx), 'omitnan');

        setRows = AllBehavior.SetSize == setSize & ...
            ~isnan(AllBehavior.Change) & ~isnan(AllBehavior.Response) & ~isnan(AllBehavior.Correct);
        Tset = AllBehavior(setRows, :);
        changeRows = Tset.Change == 1;
        noChangeRows = Tset.Change == 0;

        hitRate = nan;
        falseAlarmRate = nan;
        accuracy = nan;
        kValue = nan;
        if ~isempty(Tset)
            accuracy = mean(Tset.Correct == 1, 'omitnan');
        end
        if any(changeRows)
            hitRate = mean(Tset.Response(changeRows) == 1, 'omitnan');
        end
        if any(noChangeRows)
            falseAlarmRate = mean(Tset.Response(noChangeRows) == 1, 'omitnan');
        end
        if ~isnan(hitRate) && ~isnan(falseAlarmRate)
            kValue = setSize * (hitRate - falseAlarmRate);
        end

        S = table();
        S.Subject = subject;
        S.SetSize = setSize;
        S.SerialPosition = serialPosition;
        S.WindowStartMs = windowMs(1);
        S.WindowEndMs = windowMs(2);
        S.CDA_uV = cdaWindow;
        S.K = kValue;
        S.Accuracy = accuracy;
        S.HitRate = hitRate;
        S.FalseAlarmRate = falseAlarmRate;
        S.NEEGTrial = size(cdaTrial, 1);
        S.NBehaviorTrial = height(Tset);
        SubjectMetrics = [SubjectMetrics; S]; %#ok<AGROW>
    end
end

CorrelationSummary = table();
for seti = 1:numel(setSizes)
    setSize = setSizes(seti);
    rows = SubjectMetrics.SetSize == setSize;
    x = SubjectMetrics.CDA_uV(rows);
    y = SubjectMetrics.K(rows);
    keepRows = ~isnan(x) & ~isnan(y);

    pearsonR = nan;
    pearsonP = nan;
    spearmanRho = nan;
    spearmanP = nan;
    if sum(keepRows) >= 4 && std(x(keepRows), 0, 'omitnan') > 0 && std(y(keepRows), 0, 'omitnan') > 0
        [pearsonR, pearsonP] = corr(x(keepRows), y(keepRows), 'Type', 'Pearson');
        [spearmanRho, spearmanP] = corr(x(keepRows), y(keepRows), 'Type', 'Spearman');
    end

    R = table();
    R.SetSize = setSize;
    R.SerialPosition = targetSerialPosition(seti);
    R.WindowStartMs = SubjectMetrics.WindowStartMs(find(rows, 1));
    R.WindowEndMs = SubjectMetrics.WindowEndMs(find(rows, 1));
    R.N = sum(keepRows);
    R.PearsonR = pearsonR;
    R.PearsonP = pearsonP;
    R.SpearmanRho = spearmanRho;
    R.SpearmanP = spearmanP;
    CorrelationSummary = [CorrelationSummary; R]; %#ok<AGROW>
end

figure('Color', 'w', 'Position', [100 100 860 380]);
for seti = 1:numel(setSizes)
    setSize = setSizes(seti);
    rows = SubjectMetrics.SetSize == setSize;
    x = SubjectMetrics.CDA_uV(rows);
    y = SubjectMetrics.K(rows);
    keepRows = ~isnan(x) & ~isnan(y);

    subplot(1, numel(setSizes), seti);
    ax = gca;
    if isprop(ax, 'Toolbar')
        ax.Toolbar.Visible = 'off';
    end
    scatter(x, y, 46, 'filled');
    hold on;
    if sum(keepRows) >= 2 && std(x(keepRows), 0, 'omitnan') > 0
        pFit = polyfit(x(keepRows), y(keepRows), 1);
        xFit = linspace(min(x(keepRows)), max(x(keepRows)), 100);
        yFit = polyval(pFit, xFit);
        plot(xFit, yFit, 'k-', 'LineWidth', 1.2);
    end
    cRows = CorrelationSummary.SetSize == setSize;
    text(0.05, 0.95, sprintf('r = %.3f\np = %.3f\nN = %d', ...
        CorrelationSummary.PearsonR(cRows), CorrelationSummary.PearsonP(cRows), ...
        CorrelationSummary.N(cRows)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'Interpreter', 'none');
    xlabel('Mean CDA ERP (uV)');
    ylabel('K');
    title(sprintf('SS%d pos%d (%d-%d ms)', setSize, ...
        CorrelationSummary.SerialPosition(cRows), CorrelationSummary.WindowStartMs(cRows), ...
        CorrelationSummary.WindowEndMs(cRows)));
    box off;
end
saveas(gcf, fullfile(figureDir, 'data3_cda_erp_K_scatter.png'));
savefig(gcf, fullfile(figureDir, 'data3_cda_erp_K_scatter.fig'));
close(gcf);

disp('data3 CDA ERP-K scatter complete.');
disp(fullfile(figureDir, 'data3_cda_erp_K_scatter.png'));
disp(CorrelationSummary);

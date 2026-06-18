%% data1 SS6 CDA maintenance mean scatter plot against K
% One subject contributes one SS6 CDA maintenance mean and one mean color
% change-detection capacity value.
% K is computed from the extra color change detection task, not from the
% whole-report EEG task.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data1');
eegDir = fullfile(dataDir, 'data');
behaviorDir = fullfile(dataDir, 'Behavior Files');
resultDir = fullfile(dataDir, 'cda_erp_K_correlation');
figureDir = fullfile(resultDir, 'figures');

addpath(codeDir);

if ~isfolder(resultDir), mkdir(resultDir); end
if ~isfolder(figureDir), mkdir(figureDir); end

oldFigureFiles = { ...
    'data1_ss6_cda_K_scatter.fig', ...
    'data1_ss6_cda_K_scatter.png'};
for fi = 1:numel(oldFigureFiles)
    oldPath = fullfile(figureDir, oldFigureFiles{fi});
    if isfile(oldPath)
        delete(oldPath);
    end
end

cfg = struct();
cfg.maintenanceWindowMs = [250 inf];
cfg.minTrialsPerSetSize = 75;
cfg.kSetSizes = [3 6 8];
cfg.leftPosteriorLabels = {'O1','OL','P3','PO3','T5'};
cfg.rightPosteriorLabels = {'O2','OR','P4','PO4','T6'};
cfg.leftSS6Condition = 3;
cfg.rightSS6Condition = 6;

files = dir(fullfile(eegDir, '*_EEG_timeLockMem.mat'));
if isempty(files)
    error('No data1 EEG files found in %s.', eegDir);
end

SubjectMetrics = table();

for fi = 1:numel(files)
    fileName = files(fi).name;
    tok = regexp(fileName, '^(\d+)_EEG_timeLockMem\.mat$', 'tokens', 'once');
    if isempty(tok)
        continue;
    end
    subject = str2double(tok{1});

    behaviorFile = fullfile(behaviorDir, sprintf('%d_ColorK.mat', subject));
    if ~isfile(behaviorFile)
        fprintf('Skip subject %d: missing color change detection file %s\n', subject, behaviorFile);
        continue;
    end

    B = load(behaviorFile, 'prefs', 'stim');
    if ~isfield(B, 'prefs') || ~isfield(B, 'stim')
        warning('Skip subject %d: %s is missing prefs or stim.', subject, behaviorFile);
        continue;
    end

    kBySetSize = nan(1, numel(cfg.kSetSizes));
    hitRateBySetSize = nan(1, numel(cfg.kSetSizes));
    falseAlarmRateBySetSize = nan(1, numel(cfg.kSetSizes));
    accuracyBySetSize = nan(1, numel(cfg.kSetSizes));
    nBehaviorBySetSize = nan(1, numel(cfg.kSetSizes));

    setSizeData = B.stim.setSize(:);
    changeData = B.stim.change(:);
    responseData = B.stim.response(:);
    accuracyData = B.stim.accuracy(:);
    changeKey = B.prefs.changeKey;

    for ksi = 1:numel(cfg.kSetSizes)
        setSizeNow = cfg.kSetSizes(ksi);
        setRows = setSizeData == setSizeNow & ~isnan(changeData) & ~isnan(responseData);
        changeRows = setRows & changeData == 1;
        noChangeRows = setRows & changeData == 0;

        if any(changeRows)
            hitRateBySetSize(ksi) = mean(responseData(changeRows) == changeKey, 'omitnan');
        end
        if any(noChangeRows)
            falseAlarmRateBySetSize(ksi) = mean(responseData(noChangeRows) == changeKey, 'omitnan');
        end
        if any(setRows)
            accuracyBySetSize(ksi) = mean(accuracyData(setRows), 'omitnan');
            nBehaviorBySetSize(ksi) = sum(setRows);
        end
        if ~isnan(hitRateBySetSize(ksi)) && ~isnan(falseAlarmRateBySetSize(ksi))
            kBySetSize(ksi) = setSizeNow * (hitRateBySetSize(ksi) - falseAlarmRateBySetSize(ksi));
        end
    end

    if any(isnan(kBySetSize))
        fprintf('Skip subject %d: missing valid color change detection K for set sizes 3/6/8.\n', subject);
        continue;
    end
    meanK = mean(kBySetSize);

    fprintf('data1 SS6 CDA-meanK scatter: subject %d\n', subject);
    S = load(fullfile(files(fi).folder, fileName), 'eeg');
    eeg = S.eeg;

    [includeSubject, inclusionInfo] = data1_subject_inclusion(eeg, cfg.minTrialsPerSetSize);
    if ~includeSubject
        fprintf('Skip subject %d: original criterion failed, set-size trial counts = [%s]\n', ...
            subject, sprintf('%d ', inclusionInfo.trialCountsPerSetSize));
        continue;
    end

    time = eeg.time(:)';
    maintIdx = time >= cfg.maintenanceWindowMs(1);
    if isfinite(cfg.maintenanceWindowMs(2))
        maintIdx = maintIdx & time <= cfg.maintenanceWindowMs(2);
    end
    if ~any(maintIdx)
        error('Maintenance window does not overlap eeg.time for subject %d.', subject);
    end

    chanLabels = eeg.chanLabels(:)';
    [tfLeft, leftIdx] = ismember(cfg.leftPosteriorLabels, chanLabels);
    [tfRight, rightIdx] = ismember(cfg.rightPosteriorLabels, chanLabels);
    if any(~tfLeft) || any(~tfRight)
        error('Subject %d is missing posterior CDA channels.', subject);
    end

    eeg0 = eeg.baselined;
    artifactInd = logical(eeg.arf.artifactInd);
    nTrial = size(eeg0, 2);
    nTime = size(eeg0, 4);

    leftContra = reshape(eeg0(cfg.leftSS6Condition,:,rightIdx,:), [nTrial, numel(rightIdx), nTime]);
    leftIpsi = reshape(eeg0(cfg.leftSS6Condition,:,leftIdx,:), [nTrial, numel(leftIdx), nTime]);
    rightContra = reshape(eeg0(cfg.rightSS6Condition,:,leftIdx,:), [nTrial, numel(leftIdx), nTime]);
    rightIpsi = reshape(eeg0(cfg.rightSS6Condition,:,rightIdx,:), [nTrial, numel(rightIdx), nTime]);

    leftCDA = leftContra - leftIpsi;
    rightCDA = rightContra - rightIpsi;

    keepLeft = ~artifactInd(cfg.leftSS6Condition,:);
    keepRight = ~artifactInd(cfg.rightSS6Condition,:);
    keepLeft = keepLeft(:) & squeeze(all(all(isfinite(leftCDA), 2), 3));
    keepRight = keepRight(:) & squeeze(all(all(isfinite(rightCDA), 2), 3));

    cdaTrial = cat(1, leftCDA(keepLeft,:,:), rightCDA(keepRight,:,:));
    cdaMaint = cdaTrial(:,:,maintIdx);
    cdaMean = mean(cdaMaint(:), 'omitnan');

    M = table();
    M.Subject = subject;
    M.CDA_uV = cdaMean;
    M.MeanK = meanK;
    M.K3 = kBySetSize(1);
    M.K6 = kBySetSize(2);
    M.K8 = kBySetSize(3);
    M.Accuracy3 = accuracyBySetSize(1);
    M.Accuracy6 = accuracyBySetSize(2);
    M.Accuracy8 = accuracyBySetSize(3);
    M.NBehavior3 = nBehaviorBySetSize(1);
    M.NBehavior6 = nBehaviorBySetSize(2);
    M.NBehavior8 = nBehaviorBySetSize(3);
    M.NLeftSS6 = sum(keepLeft);
    M.NRightSS6 = sum(keepRight);
    M.NSS6 = size(cdaTrial, 1);
    SubjectMetrics = [SubjectMetrics; M]; %#ok<AGROW>
end

if isempty(SubjectMetrics)
    error('No subjects had both valid SS6 CDA and mean color change detection K.');
end

x = SubjectMetrics.CDA_uV;
y = SubjectMetrics.MeanK;
keepRows = ~isnan(x) & ~isnan(y);

pearsonR = nan;
pearsonP = nan;
spearmanRho = nan;
spearmanP = nan;
if sum(keepRows) >= 4 && std(x(keepRows), 0, 'omitnan') > 0 && std(y(keepRows), 0, 'omitnan') > 0
    [pearsonR, pearsonP] = corr(x(keepRows), y(keepRows), 'Type', 'Pearson');
    [spearmanRho, spearmanP] = corr(x(keepRows), y(keepRows), 'Type', 'Spearman');
end

fig = figure('Color', 'w', 'Position', [100 100 460 380]);
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
text(0.05, 0.95, sprintf('r = %.3f\np = %.3f\nN = %d', ...
    pearsonR, pearsonP, sum(keepRows)), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'Interpreter', 'none');
xlabel('Mean SS6 CDA maintenance ERP (uV)');
ylabel('Mean K across CD set sizes 3, 6, 8');
title(sprintf('data1 SS6 CDA vs mean color CD K (%d ms-end)', cfg.maintenanceWindowMs(1)));
box off;

saveas(fig, fullfile(figureDir, 'data1_ss6_cda_K_scatter.png'));
savefig(fig, fullfile(figureDir, 'data1_ss6_cda_K_scatter.fig'));
close(fig);

CorrelationSummary = table(sum(keepRows), pearsonR, pearsonP, spearmanRho, spearmanP, ...
    'VariableNames', {'N', 'PearsonR', 'PearsonP', 'SpearmanRho', 'SpearmanP'});

disp('data1 SS6 CDA-meanK scatter complete.');
disp(fullfile(figureDir, 'data1_ss6_cda_K_scatter.png'));
disp(CorrelationSummary);

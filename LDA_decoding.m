clear, clc
maindir = erase(pwd, 'code');
datadir = fullfile(maindir, 'cda_alpha');
outputdir = fullfile(maindir, 'decoding_LDA\');

%% config
cfg = struct();
cfg.cvType = 'holdout';        % 'holdout' reproduces 2/3 train, 1/3 test style
cfg.trainRatio = 2/3;
cfg.nFolds = 3;                % only used when cfg.cvType = 'kfold'
cfg.superTrial = 1;
cfg.nIter = 100;

cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';    % article-style 50-ms bins

cfg.doTimeGeneralization = false;
cfg.doPCA = false;
cfg.nPCs = 5;
cfg.discrimType = 'diagLinear';
cfg.standardize = false;

cfg.doShuffle = true;          % shuffled TRAINING labels empirical chance
cfg.balanceTrials = true;      % balance classes each iteration
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];

cfg.useParallel = true;
cfg.verbose = true;
cfg.randomSeed = [];

%% group stats config
statsCfg = struct();
statsCfg.doGroupStats = true;
statsCfg.makeFigure = true;
statsCfg.alpha = 0.05;
statsCfg.tail = 'right';
statsCfg.delayWindow = [400 950];

%% folders
folderNames = {'CDA', 'Alpha', 'NoPCA', 'PCA', 'GroupStats'};
for i = 1:numel(folderNames)
    outFolder = fullfile(outputdir, folderNames{i});
    if ~isfolder(outFolder), mkdir(outFolder); end
end

%% single-subject decoding
files = dir(fullfile(datadir, 'sub*'));
validCDAFiles = {};

for s = numel(files):-1:1
    file = files(s).name;
    fprintf('Now Processing: %s\n', file)
    load(fullfile(datadir, file))   % expects cda; optionally alpha

    if min(cda.trials_per_cond) < 160
        fprintf('  skipped: fewer than 160 trials in at least one condition.\n')
        continue
    end

    labels = [ones(size(cda.trial.diff_2,1),1); 2*ones(size(cda.trial.diff_6,1),1)];
    data1  = cat(1, cda.trial.diff_2(:,:,201:end), cda.trial.diff_6(:,:,201:end));
    data1  = permute(data1, [2,3,1]);  % channels x time x trials

    CDA = LDA_function_singleSubj(data1, labels, cda.time(201:end), cfg);

    saveName = fullfile(outputdir, 'CDA', [erase(file,'.mat') sprintf('_%dSuperTrials_withShuffle.mat', cfg.superTrial)]);
    save(saveName, 'CDA')
    validCDAFiles{end+1,1} = saveName; %#ok<SAGROW>

    % Optional alpha / combined analyses. Uncomment if needed.
    % data2 = cat(1, alpha.trial.diff_2(:,:,201:end), alpha.trial.diff_6(:,:,201:end));
    % data2 = permute(data2, [2,3,1]);
    % Alpha = LDA_function_singleSubj(data2, labels, alpha.time(201:end), cfg);
    % save(fullfile(outputdir, 'Alpha', [erase(file,'.mat') sprintf('_%dSuperTrials_withShuffle.mat', cfg.superTrial)]), 'Alpha')
    %
    % data3 = cat(1, data1, data2);
    % NoPCA = LDA_function_singleSubj(data3, labels, cda.time(201:end), cfg);
    % save(fullfile(outputdir, 'NoPCA', [erase(file,'.mat') sprintf('_%dSuperTrials_withShuffle.mat', cfg.superTrial)]), 'NoPCA')
    %
    % cfgPCA = cfg; cfgPCA.doPCA = true;
    % PCA = LDA_function_singleSubj(data3, labels, cda.time(201:end), cfgPCA);
    % save(fullfile(outputdir, 'PCA', [erase(file,'.mat') sprintf('_%dSuperTrials_withShuffle.mat', cfg.superTrial)]), 'PCA')
end

%% group-level empirical-chance analysis
if statsCfg.doGroupStats% && ~isempty(validCDAFiles)
    GroupStats_CDA = group_stats_against_shuffle([], 'CDA', statsCfg);
    save(fullfile(outputdir, 'GroupStats', 'GroupStats_CDA_againstShuffle.mat'), 'GroupStats_CDA', 'cfg', 'statsCfg')

    if statsCfg.makeFigure
        savePrefix = fullfile(outputdir, 'GroupStats', 'CDA_againstShuffle');
        plot_group_stats_against_shuffle(GroupStats_CDA, savePrefix);
    end
end

%% ========================================================================
function GroupStats = group_stats_against_shuffle(resultFiles, varName, statsCfg)
resultFiles = dir('D:\projects\CDA\decoding_LDA\CDA\sub*_withShuffle.mat');
nSub = numel(resultFiles);

for si = 1:nSub
    S = load(fullfile(resultFiles(si).folder,resultFiles(si).name), varName);
    R = S.(varName);

    acc = extract_time_resolved(R.predictAcc);
    shuf = extract_time_resolved(R.predictAccShuffle);

    if si == 1
        times = R.times(:)';
        allAcc = nan(nSub, numel(times));
        allShuffle = nan(nSub, numel(times));
    end

    allAcc(si,:) = acc(:)';
    allShuffle(si,:) = shuf(:)';
end

allDiff = allAcc - allShuffle;
nTime = numel(times);
alphaBonf = statsCfg.alpha / nTime;

p = nan(1,nTime);
t = nan(1,nTime);
sigBonf = false(1,nTime);

for ti = 1:nTime
    x = allDiff(:,ti);
    x = x(~isnan(x));
    if numel(x) >= 2
        [sigBonf(ti), p(ti), ~, stats] = ttest(x, 0, 'Alpha', alphaBonf, 'Tail', statsCfg.tail);
        t(ti) = stats.tstat;
    end
end

pBonf = min(p * nTime, 1);
delayIdx = times >= statsCfg.delayWindow(1) & times <= statsCfg.delayWindow(2);
delayDiff = mean(allDiff(:,delayIdx), 2, 'omitnan');
[delayH, delayP, delayCI, delayStats] = ttest(delayDiff, 0, 'Alpha', statsCfg.alpha, 'Tail', statsCfg.tail);

GroupStats = struct();
GroupStats.files = resultFiles(:);
GroupStats.times = times;
GroupStats.acc = allAcc;
GroupStats.shuffle = allShuffle;
GroupStats.diff = allDiff;
GroupStats.meanAcc = mean(allAcc, 1, 'omitnan');
GroupStats.meanShuffle = mean(allShuffle, 1, 'omitnan');
GroupStats.meanDiff = mean(allDiff, 1, 'omitnan');
GroupStats.semDiff = std(allDiff, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(allDiff),1));
GroupStats.pointwise.p = p;
GroupStats.pointwise.pBonf = pBonf;
GroupStats.pointwise.sigBonf = sigBonf;
GroupStats.pointwise.t = t;
GroupStats.delay.window = statsCfg.delayWindow;
GroupStats.delay.subjectMeanDiff = delayDiff;
GroupStats.delay.meanDiff = mean(delayDiff, 'omitnan');
GroupStats.delay.semDiff = std(delayDiff, 0, 'omitnan') ./ sqrt(sum(~isnan(delayDiff)));
GroupStats.delay.h = delayH;
GroupStats.delay.p = delayP;
GroupStats.delay.t = delayStats.tstat;
GroupStats.delay.df = delayStats.df;
GroupStats.delay.ci = delayCI;
GroupStats.statsCfg = statsCfg;
end

%% ========================================================================
function v = extract_time_resolved(M)
if isvector(M)
    v = M(:);
else
    v = diag(M);
end
end

%% ========================================================================
function plot_group_stats_against_shuffle(GroupStats, savePrefix)
times = GroupStats.times(:)';
fig = figure('Color', 'w', 'Position', [100 100 900 600]);

subplot(2,1,1)
hold on
plot(times, GroupStats.meanAcc, 'LineWidth', 2)
plot(times, GroupStats.meanShuffle, 'LineWidth', 2)
xlabel('Time')
ylabel('Classification accuracy')
legend({'Intact labels', 'Shuffled training labels'}, 'Location', 'best')
title('LDA decoding accuracy vs empirical chance')
box off

subplot(2,1,2)
hold on
plot(times, GroupStats.meanDiff, 'LineWidth', 2)
yline(0, '--')
sigIdx = GroupStats.pointwise.sigBonf;
if any(sigIdx)
    ySig = max(GroupStats.meanDiff + GroupStats.semDiff, [], 'omitnan');
    scatter(times(sigIdx), repmat(ySig, 1, sum(sigIdx)), 20, 'filled')
end
xlabel('Time')
ylabel('Accuracy - shuffled accuracy')
title(sprintf('Intact minus shuffle; delay [%g %g], p = %.4g', ...
    GroupStats.delay.window(1), GroupStats.delay.window(2), GroupStats.delay.p))
box off

savefig(fig, [savePrefix '.fig'])
print(fig, [savePrefix '.png'], '-dpng', '-r300')
end

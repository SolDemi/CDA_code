clear; clc; delete(gcp('nocreate'));

maindir   = [erase(pwd,'code') 'data0\'];
datadir   = [maindir 'data'];
outputdir = [maindir 'decoding_SVM\'];

% create folders
folderNames = {'CDA', 'Alpha', 'NoPCA', 'PCA'};
for fi = 1:numel(folderNames)
    if ~isfolder([outputdir folderNames{fi}])
        mkdir([outputdir folderNames{fi}])
    end
end

%% config: keep consistent with SVM_decoding.m
cfg = struct();
cfg.cvType = 'holdout';
cfg.trainRatio = 2/3;
cfg.nFolds = 3;

cfg.superTrial = 10;
cfg.nIter = 100;

cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';

cfg.doTimeGeneralization = 1;
cfg.doPCA = false;
cfg.nPCs = 5;
cfg.discrimType = 'Linear';
cfg.standardize = 1;

cfg.doShuffle = true;
cfg.balanceTrials = true;
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];
cfg.useAUC = 1;
cfg.useParallel = true;
cfg.verbose = 0;
cfg.randomSeed = [];

%% channel config
L_labels = {'O1','OL','P3','PO3','T5'};
R_labels = {'O2','OR','P4','PO4','T6'};

% ------------------------------------------------------------
% 原始对应关系：
% o = 1: R - L, condition 1 vs 3
% o = 2: L - R, condition 4 vs 6
%
% 现在不再分别跑 o = 1/2，而是把两侧 trial 统一成：
% contralateral - ipsilateral
% 然后左右 trial 合并后一起解码
% ------------------------------------------------------------
sideCfg(1).channel_care  = R_labels;
sideCfg(1).channel_minus = L_labels;
sideCfg(1).cond1 = 1;
sideCfg(1).cond2 = 3;

sideCfg(2).channel_care  = L_labels;
sideCfg(2).channel_minus = R_labels;
sideCfg(2).cond1 = 4;
sideCfg(2).cond2 = 6;

%% alpha config
baselinewindow = [-1500, -1100];
frep = [8, 12];

%% decoding
files = dir(fullfile(datadir, '*.mat'));

for s = 1:numel(files)

    file = files(s).name;
    fprintf('Now Processing: %s\n', file)

    tmp = load(fullfile(files(s).folder, file));
    eeg0 = tmp.eeg.baselined;
    time = tmp.eeg.time;

    % 把 artifact trial 设为 NaN
    eeg0(repmat(tmp.eeg.arf.artifactInd, ...
        [1, 1, size(eeg0,3), size(eeg0,4)])) = NaN;

    X_cond1_all = [];
    X_cond2_all = [];

    for sidei = 1:2

        channel_care  = sideCfg(sidei).channel_care;
        channel_minus = sideCfg(sidei).channel_minus;

        cond1 = sideCfg(sidei).cond1;
        cond2 = sideCfg(sidei).cond2;

        care = eeg0(:,:,ismember(tmp.eeg.chanLabels, channel_care),:);
        minus = eeg0(:,:,ismember(tmp.eeg.chanLabels, channel_minus),:);

        % contralateral - ipsilateral
        eeg_lat = care - minus;

        % condition 1
        a = squeeze(eeg_lat(cond1,:,:,:));              % trials x channels x time
        a(tmp.eeg.arf.artifactInd(cond1,:),:,:) = [];
        a = permute(a, [2, 3, 1]);                      % channels x time x trials

        nan_layers = squeeze(all(all(isnan(a), 1), 2));
        a = a(:,:,~nan_layers);

        % condition 2
        b = squeeze(eeg_lat(cond2,:,:,:));              % trials x channels x time
        b(tmp.eeg.arf.artifactInd(cond2,:),:,:) = [];
        b = permute(b, [2, 3, 1]);                      % channels x time x trials

        nan_layers = squeeze(all(all(isnan(b), 1), 2));
        b = b(:,:,~nan_layers);

        % 合并左右侧 trial
        X_cond1_all = cat(3, X_cond1_all, a);
        X_cond2_all = cat(3, X_cond2_all, b);
    end

    % CDA data: channels x time x trials
    data1 = cat(3, X_cond1_all, X_cond2_all);

    % labels: 和 SVM_decoding.m 一致，用 1/6 表示两个 load 类别
    labels = [
        ones(size(X_cond1_all,3), 1) * 1;
        ones(size(X_cond2_all,3), 1) * 6
    ];

    if size(data1,3) < 75
        fprintf('Skip %s: too few trials, nTrial = %d\n', file, size(data1,3));
        continue
    end

    if min([size(X_cond1_all,3), size(X_cond2_all,3)]) < cfg.superTrial
        fprintf('Skip %s: too few trials in one class\n', file);
        continue
    end

    % ============================================================
    % 1. Decode load based on CDA
    % ============================================================
    cfg.doPCA = false;

    CDA = SVM_function_singleSubj(data1, labels, time, cfg);
    CDA.nCond1 = size(X_cond1_all,3);
    CDA.nCond2 = size(X_cond2_all,3);
    CDA.labels = labels;

    save([outputdir 'CDA\' file], 'CDA', '-v7.3');

    % ============================================================
    % 2. Decode load based on alpha
    % ============================================================
    data2 = calculate_high_gamma_power( ...
        data1, tmp.eeg.settings.srate, time, baselinewindow, frep);

    cfg.doPCA = false;

    Alpha = SVM_function_singleSubj(data2, labels, time, cfg);
    Alpha.nCond1 = size(X_cond1_all,3);
    Alpha.nCond2 = size(X_cond2_all,3);
    Alpha.labels = labels;
    Alpha.baselinewindow = baselinewindow;
    Alpha.frep = frep;

    save([outputdir 'Alpha\' file], 'Alpha', '-v7.3');

    % ============================================================
    % 3. Decode load based on CDA + alpha, without PCA
    % ============================================================
    data3 = cat(1, data1, data2);   % feature/channel dimension concat

    cfg.doPCA = false;

    NoPCA = SVM_function_singleSubj(data3, labels, time, cfg);
    NoPCA.nCond1 = size(X_cond1_all,3);
    NoPCA.nCond2 = size(X_cond2_all,3);
    NoPCA.labels = labels;

    save([outputdir 'NoPCA\' file], 'NoPCA', '-v7.3');

    % ============================================================
    % 4. Decode load based on CDA + alpha, with PCA
    % ============================================================
    cfg.doPCA = true;

    PCA = SVM_function_singleSubj(data3, labels, time, cfg);
    PCA.nCond1 = size(X_cond1_all,3);
    PCA.nCond2 = size(X_cond2_all,3);
    PCA.labels = labels;

    save([outputdir 'PCA\' file], 'PCA', '-v7.3');

    fprintf('Finished %s: nCond1 = %d, nCond2 = %d\n\n', ...
        file, size(X_cond1_all,3), size(X_cond2_all,3));
end
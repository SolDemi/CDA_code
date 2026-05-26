clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);

maindir   = fullfile(projectRoot, 'data0');
datadir   = fullfile(maindir, 'data');
outputdir = fullfile(maindir, 'decoding_LDA');

%% create folders
folderNames = {'CDA', 'Alpha', 'GlobalAlpha', 'NoPCA', 'PCA'};
for fi = 1:numel(folderNames)
    tmpDir = fullfile(outputdir, folderNames{fi});
    if ~isfolder(tmpDir)
        mkdir(tmpDir);
    end
end

%% config: consistent with LDA_decoding.m
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
cfg.discrimType = 'diagLinear';
cfg.standardize = 1;

cfg.doShuffle = true;
cfg.balanceTrials = true;
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];
cfg.useAUC = 0;
cfg.useParallel = true;
cfg.verbose = 0;
cfg.randomSeed = [];

%% channel config
L_labels = {'O1','OL','P3','PO3','T5'};
R_labels = {'O2','OR','P4','PO4','T6'};
global_labels = [L_labels, R_labels];

% Original condition mapping:
% side 1 uses R-L for condition 1 vs 3.
% side 2 uses L-R for condition 4 vs 6.
% CDA and lateralized alpha are contra-minus-ipsi features.
% Global alpha uses all posterior left/right channels without subtraction.
sideCfg(1).channel_care  = R_labels;
sideCfg(1).channel_minus = L_labels;
sideCfg(1).cond1 = 1;
sideCfg(1).cond2 = 3;

sideCfg(2).channel_care  = L_labels;
sideCfg(2).channel_minus = R_labels;
sideCfg(2).cond1 = 4;
sideCfg(2).cond2 = 6;

%% alpha config
baselinewindow = [-1400, -1100];
frep = [8, 12];

%% decoding
files = dir(fullfile(datadir, '*.mat'));

for s = 1:numel(files)

    file = files(s).name;
    fprintf('Now Processing: %s\n', file);

    tmp = load(fullfile(files(s).folder, file));

    eeg0 = tmp.eeg.baselined;
    time = tmp.eeg.time;
    artifactInd = tmp.eeg.arf.artifactInd;
    chanLabels = tmp.eeg.chanLabels;

    % containers
    CDA_cond1_all = [];
    CDA_cond2_all = [];

    Alpha_cond1_all = [];
    Alpha_cond2_all = [];

    GlobalAlpha_cond1_all = [];
    GlobalAlpha_cond2_all = [];

    for sidei = 1:2

        channel_care  = sideCfg(sidei).channel_care;
        channel_minus = sideCfg(sidei).channel_minus;

        cond1 = sideCfg(sidei).cond1;
        cond2 = sideCfg(sidei).cond2;

        care_raw = eeg0(:,:,ismember(chanLabels, channel_care),:);
        minus_raw = eeg0(:,:,ismember(chanLabels, channel_minus),:);
        global_raw = eeg0(:,:,ismember(chanLabels, global_labels),:);

        %% condition 1
        [a_care, a_minus, a_global] = get_clean_lateral_trials( ...
            care_raw, minus_raw, global_raw, artifactInd, cond1);

        % CDA: contra ERP - ipsi ERP
        a_cda = a_care - a_minus;

        % lateralized alpha: alpha(contra) - alpha(ipsi)
        a_alpha_care = calculate_hilbert_band_power( ...
            a_care, tmp.eeg.settings.srate, time, baselinewindow, frep);

        a_alpha_minus = calculate_hilbert_band_power( ...
            a_minus, tmp.eeg.settings.srate, time, baselinewindow, frep);

        a_alpha = a_alpha_care - a_alpha_minus;

        % global alpha: alpha(left + right posterior channels)
        a_global_alpha = calculate_hilbert_band_power( ...
            a_global, tmp.eeg.settings.srate, time, baselinewindow, frep);

        %% condition 2
        [b_care, b_minus, b_global] = get_clean_lateral_trials( ...
            care_raw, minus_raw, global_raw, artifactInd, cond2);

        % CDA: contra ERP - ipsi ERP
        b_cda = b_care - b_minus;

        % lateralized alpha: alpha(contra) - alpha(ipsi)
        b_alpha_care = calculate_hilbert_band_power( ...
            b_care, tmp.eeg.settings.srate, time, baselinewindow, frep);

        b_alpha_minus = calculate_hilbert_band_power( ...
            b_minus, tmp.eeg.settings.srate, time, baselinewindow, frep);

        b_alpha = b_alpha_care - b_alpha_minus;

        % global alpha: alpha(left + right posterior channels)
        b_global_alpha = calculate_hilbert_band_power( ...
            b_global, tmp.eeg.settings.srate, time, baselinewindow, frep);

        %% merge left/right trials
        CDA_cond1_all = cat(3, CDA_cond1_all, a_cda);
        CDA_cond2_all = cat(3, CDA_cond2_all, b_cda);

        Alpha_cond1_all = cat(3, Alpha_cond1_all, a_alpha);
        Alpha_cond2_all = cat(3, Alpha_cond2_all, b_alpha);

        GlobalAlpha_cond1_all = cat(3, GlobalAlpha_cond1_all, a_global_alpha);
        GlobalAlpha_cond2_all = cat(3, GlobalAlpha_cond2_all, b_global_alpha);
    end

    %% final data: channels x time x trials
    data_CDA = cat(3, CDA_cond1_all, CDA_cond2_all);
    data_Alpha = cat(3, Alpha_cond1_all, Alpha_cond2_all);
    data_GlobalAlpha = cat(3, GlobalAlpha_cond1_all, GlobalAlpha_cond2_all);

    if any(~isfinite(data_CDA), 'all') || ...
            any(~isfinite(data_Alpha), 'all') || ...
            any(~isfinite(data_GlobalAlpha), 'all')
        error('Non-finite values remain in %s', file);
    end

    labels = [
        ones(size(CDA_cond1_all,3), 1) * 1;
        ones(size(CDA_cond2_all,3), 1) * 6
    ];

    if size(data_CDA,3) < 75
        fprintf('Skip %s: too few trials, nTrial = %d\n', file, size(data_CDA,3));
        continue
    end

    if min([size(CDA_cond1_all,3), size(CDA_cond2_all,3)]) < cfg.superTrial
        fprintf('Skip %s: too few trials in one class\n', file);
        continue
    end

    % % ============================================================
    % 1. CDA decoding
    % % =============================================================
    cfg.doPCA = false;

    CDA = LDA_function_singleSubj(data_CDA, labels, time, cfg);
    CDA.nCond1 = size(CDA_cond1_all,3);
    CDA.nCond2 = size(CDA_cond2_all,3);
    CDA.labels = labels;
    CDA.channelLabels = {'contraMinusIpsi posterior pairs'};

    save(fullfile(outputdir, 'CDA', file), 'CDA', '-v7.3');

    %% ============================================================
    % 2. lateralized alpha decoding
    % =============================================================
    cfg.doPCA = false;

    Alpha = LDA_function_singleSubj(data_Alpha, labels, time, cfg);
    Alpha.nCond1 = size(Alpha_cond1_all,3);
    Alpha.nCond2 = size(Alpha_cond2_all,3);
    Alpha.labels = labels;
    Alpha.baselinewindow = baselinewindow;
    Alpha.frep = frep;
    Alpha.channelLabels = {'contraMinusIpsi posterior pairs'};

    save(fullfile(outputdir, 'Alpha', file), 'Alpha', '-v7.3');

    %% ============================================================
    % 3. global alpha decoding
    % =============================================================
    cfg.doPCA = false;

    GlobalAlpha = LDA_function_singleSubj(data_GlobalAlpha, labels, time, cfg);
    GlobalAlpha.nCond1 = size(GlobalAlpha_cond1_all,3);
    GlobalAlpha.nCond2 = size(GlobalAlpha_cond2_all,3);
    GlobalAlpha.labels = labels;
    GlobalAlpha.baselinewindow = baselinewindow;
    GlobalAlpha.frep = frep;
    GlobalAlpha.channelLabels = global_labels;

    save(fullfile(outputdir, 'GlobalAlpha', file), 'GlobalAlpha', '-v7.3');

    %% ============================================================
    % 4. CDA + lateralized alpha, without PCA
    % =============================================================
    data_NoPCA = cat(1, data_CDA, data_Alpha);

    cfg.doPCA = false;

    NoPCA = LDA_function_singleSubj(data_NoPCA, labels, time, cfg);
    NoPCA.nCond1 = size(CDA_cond1_all,3);
    NoPCA.nCond2 = size(CDA_cond2_all,3);
    NoPCA.labels = labels;
    NoPCA.baselinewindow = baselinewindow;
    NoPCA.frep = frep;
    NoPCA.channelLabels = {'CDA posterior pairs', 'lateralized alpha posterior pairs'};

    save(fullfile(outputdir, 'NoPCA', file), 'NoPCA', '-v7.3');

    %% ============================================================
    % 5. CDA + lateralized alpha, with PCA
    % =============================================================
    cfg.doPCA = true;

    PCA = LDA_function_singleSubj(data_NoPCA, labels, time, cfg);
    PCA.nCond1 = size(CDA_cond1_all,3);
    PCA.nCond2 = size(CDA_cond2_all,3);
    PCA.labels = labels;
    PCA.baselinewindow = baselinewindow;
    PCA.frep = frep;
    PCA.channelLabels = {'CDA posterior pairs', 'lateralized alpha posterior pairs'};

    save(fullfile(outputdir, 'PCA', file), 'PCA', '-v7.3');

    fprintf('Finished %s: nCond1 = %d, nCond2 = %d\n\n', ...
        file, size(CDA_cond1_all,3), size(CDA_cond2_all,3));
end


%% ========================================================================
% local function
% ========================================================================
function [Xcare, Xminus, Xglobal] = get_clean_lateral_trials( ...
    care_raw, minus_raw, global_raw, artifactInd, condIdx)
% care_raw / minus_raw / global_raw:
%   condition x trial x channel x time
%
% artifactInd:
%   condition x trial
%
% outputs:
%   channel x time x clean_trial

    nTrial = size(care_raw, 2);

    care = reshape(care_raw(condIdx,:,:,:), ...
        [nTrial, size(care_raw,3), size(care_raw,4)]);

    minus = reshape(minus_raw(condIdx,:,:,:), ...
        [nTrial, size(minus_raw,3), size(minus_raw,4)]);

    globalDat = reshape(global_raw(condIdx,:,:,:), ...
        [nTrial, size(global_raw,3), size(global_raw,4)]);

    % Remove artifact-marked trials first.
    keepTrial = ~artifactInd(condIdx,:);

    % Then remove trials containing NaN or Inf in any feature set.
    badCare = squeeze(any(any(~isfinite(care), 2), 3))';
    badMinus = squeeze(any(any(~isfinite(minus), 2), 3))';
    badGlobal = squeeze(any(any(~isfinite(globalDat), 2), 3))';

    keepTrial = keepTrial & ~badCare & ~badMinus & ~badGlobal;

    if ~any(keepTrial)
        Xcare = zeros(size(care_raw,3), size(care_raw,4), 0);
        Xminus = zeros(size(minus_raw,3), size(minus_raw,4), 0);
        Xglobal = zeros(size(global_raw,3), size(global_raw,4), 0);
        return
    end

    care = care(keepTrial,:,:);
    minus = minus(keepTrial,:,:);
    globalDat = globalDat(keepTrial,:,:);

    Xcare = permute(care, [2, 3, 1]);
    Xminus = permute(minus, [2, 3, 1]);
    Xglobal = permute(globalDat, [2, 3, 1]);
end

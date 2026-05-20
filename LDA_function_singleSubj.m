function result = LDA_function_singleSubj(eegData, labels, times, cfg)
% LDA_function_singleSubj
% Lightweight time-resolved / time-generalization binary LDA decoding.
%
% Required external functions on MATLAB path:
%   balance_trials_by_label.m     % optional trial balancing
%   func_make_superTrials.m       % optional supertrial averaging
%
% Key cfg fields:
%   cfg.cvType               : 'holdout' or 'kfold' default = 'kfold'
%   cfg.trainRatio           : training proportion for holdout default = 2/3
%   cfg.nFolds               : K for kfold default = 10
%   cfg.nIter                : random iterations default = 1
%   cfg.doShuffle            : shuffled-training-label baseline default = false
%   cfg.useAUC               : compute AUC/perfcurve default = false
%   cfg.balanceTrials        : downsample classes each iter default = false
%   cfg.useParallel          : use parfor over cfg.nIter default = false
%   cfg.superTrial           : trials averaged into supertrial default = 1
%
% LDA-specific cfg fields:
%   cfg.discrimType          : 'diagLinear' or 'linear' default = 'diagLinear'
%   cfg.ldaEngine            : 'fitcdiscr' or 'classify' default = 'fitcdiscr'
%
% Outputs:
%   result.Acc, result.AccTrain, result.weights
%   result.AUC only if cfg.useAUC = true
%   result.AccShuffle, result.AccMinusShuffle if doShuffle = true
%   result.AUCShuffle, result.AUCMinusShuffle only if doShuffle && useAUC

if nargin < 4 || isempty(cfg), cfg = struct(); end
cfg = fill_default_cfg(cfg);

labels = labels(:);
times  = times(:)';

[~, nTime, nTrials] = size(eegData);
if numel(labels) ~= nTrials || numel(times) ~= nTime
    error('Mismatch among eegData, labels, and times.');
end

uLabels = unique(labels);
if numel(uLabels) ~= 2
    error('LDA_function_singleSubj currently supports binary classification only.');
end

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed, 'twister');
end

% Temporal binning/smoothing.
if cfg.smooth_window > 0 || ~isempty(cfg.smooth_step)
    [eegData, times] = apply_temporal_window( ...
        eegData, times, cfg.smooth_window, cfg.smooth_step, cfg.timeWindowMode);
end
[nCh, nTime, ~] = size(eegData);

% Relabel to 1/2 internally.
labels_internal = zeros(size(labels));
labels_internal(labels == uLabels(1)) = 1;
labels_internal(labels == uLabels(2)) = 2;

if strcmpi(cfg.cvType, 'holdout')
    nSplits = 1;
else
    nSplits = cfg.nFolds;
end

Acc_all = nan(nTime, nTime, nSplits, cfg.nIter);
trainAcc_all   = nan(nSplits, nTime, cfg.nIter);
weights_all    = nan(nCh, nTime, nSplits, cfg.nIter);

if cfg.useAUC
    AUC_all = nan(nTime, nTime, nSplits, cfg.nIter);
else
    AUC_all = [];
end

if cfg.doShuffle
    AccShuffle_all = nan(nTime, nTime, nSplits, cfg.nIter);
    if cfg.useAUC
        AUCShuffle_all = nan(nTime, nTime, nSplits, cfg.nIter);
    else
        AUCShuffle_all = [];
    end
else
    AccShuffle_all = [];
    AUCShuffle_all = [];
end

balanceInfo = cell(cfg.nIter, 1);

% Parallelization is intentionally placed at the iteration level.
if cfg.useParallel
    parfor sampi = 1:cfg.nIter
        [Acc_iter, AUC_iter, trainAcc_iter, weights_iter, ...
            AccShuffle_iter, AUCShuffle_iter, balanceInfo_iter] = run_one_iteration( ...
            eegData, labels_internal, sampi, nTime, nCh, nSplits, cfg);

        Acc_all(:,:,:,sampi) = Acc_iter;
        trainAcc_all(:,:,sampi) = trainAcc_iter;
        weights_all(:,:,:,sampi) = weights_iter;
        balanceInfo{sampi} = balanceInfo_iter;

        if cfg.useAUC
            AUC_all(:,:,:,sampi) = AUC_iter;
        end

        if cfg.doShuffle
            AccShuffle_all(:,:,:,sampi) = AccShuffle_iter;
            if cfg.useAUC
                AUCShuffle_all(:,:,:,sampi) = AUCShuffle_iter;
            end
        end

        if cfg.verbose
            fprintf(' sample %d/%d done\n', sampi, cfg.nIter);
        end
    end
else
    for sampi = 1:cfg.nIter
        [Acc_iter, AUC_iter, trainAcc_iter, weights_iter, ...
            AccShuffle_iter, AUCShuffle_iter, balanceInfo_iter] = run_one_iteration( ...
            eegData, labels_internal, sampi, nTime, nCh, nSplits, cfg);

        Acc_all(:,:,:,sampi) = Acc_iter;
        trainAcc_all(:,:,sampi) = trainAcc_iter;
        weights_all(:,:,:,sampi) = weights_iter;
        balanceInfo{sampi} = balanceInfo_iter;

        if cfg.useAUC
            AUC_all(:,:,:,sampi) = AUC_iter;
        end

        if cfg.doShuffle
            AccShuffle_all(:,:,:,sampi) = AccShuffle_iter;
            if cfg.useAUC
                AUCShuffle_all(:,:,:,sampi) = AUCShuffle_iter;
            end
        end

        if cfg.verbose
            fprintf(' sample %d/%d done\n', sampi, cfg.nIter);
        end
    end
end

Acc = squeeze(mean(mean(Acc_all, 3, 'omitnan'), 4, 'omitnan'));
weights    = squeeze(mean(mean(weights_all, 3, 'omitnan'), 4, 'omitnan'));
trainAcc   = squeeze(mean(mean(trainAcc_all, 1, 'omitnan'), 3, 'omitnan'));

if cfg.useAUC
    AUC = squeeze(mean(mean(AUC_all, 3, 'omitnan'), 4, 'omitnan'));
end

if ~cfg.doTimeGeneralization
    Acc = keep_diagonal_only(Acc);
    if cfg.useAUC
        AUC = keep_diagonal_only(AUC);
    end
end

result = struct();
result.Acc = Acc;
if cfg.useAUC
    result.AUC = AUC;
end
result.AccTrain = trainAcc(:);
result.weights = weights;
result.times = times(:);
result.cfg = cfg;
result.classLabelsOriginal = uLabels;
result.balanceInfo = balanceInfo;

if cfg.doShuffle
    AccShuffle = squeeze(mean(mean(AccShuffle_all, 3, 'omitnan'), 4, 'omitnan'));
    if ~cfg.doTimeGeneralization
        AccShuffle = keep_diagonal_only(AccShuffle);
    end

    result.AccShuffle = AccShuffle;
    result.AccMinusShuffle = Acc - AccShuffle;

    if cfg.useAUC
        AUCShuffle = squeeze(mean(mean(AUCShuffle_all, 3, 'omitnan'), 4, 'omitnan'));
        if ~cfg.doTimeGeneralization
            AUCShuffle = keep_diagonal_only(AUCShuffle);
        end

        result.AUCShuffle = AUCShuffle;
        result.AUCMinusShuffle = AUC - AUCShuffle;
    end
end

end

%% ========================================================================
function cfg = fill_default_cfg(cfg)

% Backward-compatible aliases.
if isfield(cfg,'avgNTrials') && ~isfield(cfg,'superTrial')
    cfg.superTrial = cfg.avgNTrials;
end
if isfield(cfg,'binSize') && ~isfield(cfg,'smooth_window')
    cfg.smooth_window = cfg.binSize;
end
if isfield(cfg,'seed') && ~isfield(cfg,'randomSeed')
    cfg.randomSeed = cfg.seed;
end
if isfield(cfg,'zscore') && ~isfield(cfg,'standardize')
    cfg.standardize = cfg.zscore;
end

if ~isfield(cfg,'cvType'), cfg.cvType = 'kfold'; end
if ~isfield(cfg,'trainRatio'), cfg.trainRatio = 2/3; end
if ~isfield(cfg,'nFolds'), cfg.nFolds = 10; end
if ~isfield(cfg,'superTrial'), cfg.superTrial = 1; end
if ~isfield(cfg,'nIter'), cfg.nIter = 1; end
if ~isfield(cfg,'doPCA'), cfg.doPCA = false; end
if ~isfield(cfg,'nPCs'), cfg.nPCs = 5; end
if ~isfield(cfg,'smooth_window'), cfg.smooth_window = 0; end
if ~isfield(cfg,'smooth_step'), cfg.smooth_step = []; end
if ~isfield(cfg,'timeWindowMode'), cfg.timeWindowMode = 'centered'; end
if ~isfield(cfg,'doTimeGeneralization'), cfg.doTimeGeneralization = true; end
if ~isfield(cfg,'discrimType'), cfg.discrimType = 'diagLinear'; end
if ~isfield(cfg,'ldaEngine'), cfg.ldaEngine = 'fitcdiscr'; end
if ~isfield(cfg,'standardize'), cfg.standardize = false; end
if ~isfield(cfg,'doShuffle'), cfg.doShuffle = false; end
if ~isfield(cfg,'useAUC'), cfg.useAUC = false; end
if ~isfield(cfg,'balanceTrials'), cfg.balanceTrials = false; end
if ~isfield(cfg,'balanceNPerCell'), cfg.balanceNPerCell = []; end
if ~isfield(cfg,'balanceFactors'), cfg.balanceFactors = []; end
if ~isfield(cfg,'verbose'), cfg.verbose = true; end
if ~isfield(cfg,'randomSeed'), cfg.randomSeed = []; end
if ~isfield(cfg,'useParallel'), cfg.useParallel = false; end

cfg.cvType = lower(char(cfg.cvType));
cfg.timeWindowMode = lower(char(cfg.timeWindowMode));
cfg.ldaEngine = lower(char(cfg.ldaEngine));

if ~ismember(cfg.ldaEngine, {'fitcdiscr','classify'})
    error('cfg.ldaEngine must be ''fitcdiscr'' or ''classify''.');
end

end

%% ========================================================================
function [Acc_iter, AUC_iter, trainAcc_iter, weights_iter, ...
    AccShuffle_iter, AUCShuffle_iter, balanceInfo_iter] = run_one_iteration( ...
    eegData, labels_internal, sampi, nTime, nCh, nSplits, cfg)

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed + sampi - 1, 'twister');
end

Acc_iter = nan(nTime, nTime, nSplits);
trainAcc_iter   = nan(nSplits, nTime);
weights_iter    = nan(nCh, nTime, nSplits);

if cfg.useAUC
    AUC_iter = nan(nTime, nTime, nSplits);
else
    AUC_iter = [];
end

if cfg.doShuffle
    AccShuffle_iter = nan(nTime, nTime, nSplits);
    if cfg.useAUC
        AUCShuffle_iter = nan(nTime, nTime, nSplits);
    else
        AUCShuffle_iter = [];
    end
else
    AccShuffle_iter = [];
    AUCShuffle_iter = [];
end

balanceInfo_iter = [];

% Optional trial balancing.
if cfg.balanceTrials
    balLabels = labels_internal(:);
    if ~isempty(cfg.balanceFactors)
        balLabels = [balLabels, cfg.balanceFactors];
    end

    seedNow = [];
    if ~isempty(cfg.randomSeed)
        seedNow = cfg.randomSeed + sampi - 1;
    end

    [dataIter, balLabelsOut, ~, balanceInfo_iter] = balance_trials_by_label( ...
        eegData, balLabels, ...
        'trialDim', 3, ...
        'nPerCell', cfg.balanceNPerCell, ...
        'seed', seedNow, ...
        'shuffleOutput', true);

    labelsIter = balLabelsOut(:,1);
else
    dataIter = eegData;
    labelsIter = labels_internal;
end

% Optional supertrials, separately within each class.
data1 = dataIter(:,:,labelsIter == 1);
data2 = dataIter(:,:,labelsIter == 2);

if cfg.superTrial > 1
    data1 = func_make_superTrials(data1, cfg.superTrial);
    data2 = func_make_superTrials(data2, cfg.superTrial);
end

allTrials = cat(3, data1, data2);
allLabels = [ones(size(data1,3),1); 2*ones(size(data2,3),1)];

[trainIdxList, testIdxList] = make_cv_splits(allLabels, cfg);

for spliti = 1:nSplits
    trainIdx = trainIdxList{spliti};
    testIdx  = testIdxList{spliti};

    trainY = allLabels(trainIdx);
    testY  = allLabels(testIdx);

    if cfg.doShuffle
        trainYShuffleByTime = cell(nTime, 1);
        for ti = 1:nTime
            trainYShuffleByTime{ti} = trainY(randperm(numel(trainY)));
        end
    else
        trainYShuffleByTime = cell(nTime, 1);
    end

    for trainTime = 1:nTime
        [accRow, aucRow, trainAccVal, weightVec, accShufRow, aucShufRow] = decode_one_time( ...
            allTrials, trainIdx, testIdx, trainY, testY, trainYShuffleByTime{trainTime}, ...
            trainTime, nTime, nCh, cfg);

        Acc_iter(trainTime,:,spliti) = accRow;
        trainAcc_iter(spliti,trainTime) = trainAccVal;
        weights_iter(:,trainTime,spliti) = weightVec;

        if cfg.useAUC
            AUC_iter(trainTime,:,spliti) = aucRow;
        end

        if cfg.doShuffle
            AccShuffle_iter(trainTime,:,spliti) = accShufRow;
            if cfg.useAUC
                AUCShuffle_iter(trainTime,:,spliti) = aucShufRow;
            end
        end
    end
end

end

%% ========================================================================
function [trainIdxList, testIdxList] = make_cv_splits(y, cfg)

if strcmpi(cfg.cvType, 'holdout')
    CVO = cvpartition(y, 'HoldOut', 1 - cfg.trainRatio);
    trainIdxList = {training(CVO)};
    testIdxList  = {test(CVO)};
else
    CVO = cvpartition(y, 'KFold', cfg.nFolds);
    trainIdxList = cell(cfg.nFolds, 1);
    testIdxList  = cell(cfg.nFolds, 1);

    for fi = 1:cfg.nFolds
        trainIdxList{fi} = training(CVO, fi);
        testIdxList{fi}  = test(CVO, fi);
    end
end

end

%% ========================================================================
function [accRow, aucRow, trainAccVal, weightVec, accShufRow, aucShufRow] = decode_one_time( ...
    allTrials, trainIdx, testIdx, trainY, testY, trainYShuffle, trainTime, nTime, nCh, cfg)

accRow     = nan(1, nTime);
accShufRow = nan(1, nTime);

if cfg.useAUC
    aucRow     = nan(1, nTime);
    aucShufRow = nan(1, nTime);
else
    aucRow     = [];
    aucShufRow = [];
end

weightVec = nan(nCh, 1);

Xtrain = squeeze(allTrials(:, trainTime, trainIdx))';
if isvector(Xtrain)
    Xtrain = reshape(Xtrain, sum(trainIdx), nCh);
end

mu_z = [];
sigma_z = [];
if cfg.standardize
    [Xtrain, mu_z, sigma_z] = zscore_train_only(Xtrain);
end

coeff = [];
mu_pca = [];
if cfg.doPCA
    [coeff, Xtrain, mu_pca] = fit_pca_train_only(Xtrain, cfg.nPCs);
end

useClassify = strcmpi(cfg.ldaEngine, 'classify');
classNames = unique(trainY);

% Fit or prepare classifier.
if useClassify
    if cfg.useAUC
        [labelTrain, ~, ~] = classify(Xtrain, Xtrain, trainY, cfg.discrimType);
    else
        labelTrain = classify(Xtrain, Xtrain, trainY, cfg.discrimType);
    end
else
    ldaModel = fitcdiscr(Xtrain, trainY, 'DiscrimType', cfg.discrimType);

    if cfg.doShuffle
        ldaModelShuffle = fitcdiscr(Xtrain, trainYShuffle, 'DiscrimType', cfg.discrimType);
    end

    labelTrain = predict(ldaModel, Xtrain);
end

trainAccVal = mean(labelTrain == trainY);

% Weight is computed from data directly, so it is consistent across engines.
w_use = lda_weight_from_data(Xtrain, trainY, cfg.discrimType);
if cfg.doPCA
    w_use = coeff * w_use;
end
if cfg.standardize
    w_use = w_use ./ sigma_z(:);
end
weightVec = w_use;

if cfg.doTimeGeneralization
    testTimes = 1:nTime;
else
    testTimes = trainTime;
end

for testTime = testTimes
    Xtest = squeeze(allTrials(:, testTime, testIdx))';
    if isvector(Xtest)
        Xtest = reshape(Xtest, sum(testIdx), nCh);
    end

    if cfg.standardize
        Xtest = (Xtest - mu_z) ./ sigma_z;
    end

    if cfg.doPCA
        Xtest = (Xtest - mu_pca) * coeff;
    end

    if useClassify
        if cfg.useAUC
            [labelTest, ~, posterior] = classify(Xtest, Xtrain, trainY, cfg.discrimType);
            aucRow(testTime) = binary_auc(testY, posterior, classNames, 2);
        else
            labelTest = classify(Xtest, Xtrain, trainY, cfg.discrimType);
        end
    else
        if cfg.useAUC
            [labelTest, score] = predict(ldaModel, Xtest);
            aucRow(testTime) = binary_auc(testY, score, ldaModel.ClassNames, 2);
        else
            labelTest = predict(ldaModel, Xtest);
        end
    end

    accRow(testTime) = mean(labelTest == testY);

    if cfg.doShuffle
        if useClassify
            if cfg.useAUC
                [labelShuf, ~, posteriorShuf] = classify(Xtest, Xtrain, trainYShuffle, cfg.discrimType);
                aucShufRow(testTime) = binary_auc(testY, posteriorShuf, classNames, 2);
            else
                labelShuf = classify(Xtest, Xtrain, trainYShuffle, cfg.discrimType);
            end
        else
            if cfg.useAUC
                [labelShuf, scoreShuf] = predict(ldaModelShuffle, Xtest);
                aucShufRow(testTime) = binary_auc(testY, scoreShuf, ldaModelShuffle.ClassNames, 2);
            else
                labelShuf = predict(ldaModelShuffle, Xtest);
            end
        end

        accShufRow(testTime) = mean(labelShuf == testY);
    end
end

end

%% ========================================================================
function [eegOut, timesOut] = apply_temporal_window(eegData, times, win, step, modeName)

if isempty(step)
    step = win;
end

if strcmpi(modeName, 'bin')
    binStarts = times(1):step:(times(end)-win);
    eegOut = zeros(size(eegData,1), numel(binStarts), size(eegData,3), 'like', eegData);
    timesOut = binStarts + win/2;

    for bi = 1:numel(binStarts)
        t1 = binStarts(bi);
        t2 = t1 + win;

        if bi < numel(binStarts)
            tidx = times >= t1 & times < t2;
        else
            tidx = times >= t1 & times <= t2;
        end

        eegOut(:,bi,:) = mean(eegData(:,tidx,:), 2, 'omitnan');
    end
else
    if win > 0
        halfWin = win/2;
        centerIdx = find(times >= times(1)+halfWin & times <= times(end)-halfWin);
    else
        centerIdx = 1:numel(times);
    end

    if ~isempty(step)
        targetTimes = times(centerIdx(1)):step:times(centerIdx(end));
        tmp = nan(size(targetTimes));

        for ii = 1:numel(targetTimes)
            [~,k] = min(abs(times(centerIdx)-targetTimes(ii)));
            tmp(ii) = centerIdx(k);
        end

        centerIdx = unique(tmp, 'stable');
    end

    eegOut = zeros(size(eegData,1), numel(centerIdx), size(eegData,3), 'like', eegData);
    timesOut = times(centerIdx);

    for ii = 1:numel(centerIdx)
        ct = times(centerIdx(ii));

        if win > 0
            tidx = times >= ct-win/2 & times <= ct+win/2;
        else
            tidx = centerIdx(ii);
        end

        eegOut(:,ii,:) = mean(eegData(:,tidx,:), 2, 'omitnan');
    end
end

end

%% ========================================================================
function [Xz, mu_z, sigma_z] = zscore_train_only(X)

mu_z = mean(X, 1, 'omitnan');
sigma_z = std(X, 0, 1, 'omitnan');
sigma_z(sigma_z == 0 | isnan(sigma_z)) = 1;
Xz = (X - mu_z) ./ sigma_z;

end

%% ========================================================================
function [coeff, score, mu] = fit_pca_train_only(X, nPCs)

maxPC = min([size(X,1)-1, size(X,2), nPCs]);
[coeff, score, ~, ~, ~, mu] = pca(X);
coeff = coeff(:,1:maxPC);
score = score(:,1:maxPC);

end

%% ========================================================================
function w = lda_weight_from_data(X, y, discrimType)

X1 = X(y == 1, :);
X2 = X(y == 2, :);

mu1 = mean(X1, 1, 'omitnan');
mu2 = mean(X2, 1, 'omitnan');

if strcmpi(discrimType, 'linear')
    S1 = cov(X1);
    S2 = cov(X2);

    n1 = size(X1,1);
    n2 = size(X2,1);

    Sp = ((n1-1)*S1 + (n2-1)*S2) / max(n1+n2-2, 1);
    w = pinv(Sp + eye(size(Sp))*eps) * (mu2 - mu1)';
else
    v1 = var(X1, 0, 1, 'omitnan');
    v2 = var(X2, 0, 1, 'omitnan');

    n1 = size(X1,1);
    n2 = size(X2,1);

    vp = ((n1-1)*v1 + (n2-1)*v2) / max(n1+n2-2, 1);
    vp(vp <= eps | isnan(vp)) = eps;

    w = ((mu2 - mu1) ./ vp)';
end

end

%% ========================================================================
function aucVal = binary_auc(testY, score, classNames, posClass)

aucVal = NaN;
posCol = find(classNames == posClass, 1);

if ~isempty(posCol) && numel(unique(testY)) == 2
    [~,~,~,aucVal] = perfcurve(testY, score(:,posCol), posClass);
end

end

%% ========================================================================
function M = keep_diagonal_only(M)

d = diag(M);
M(:) = NaN;
M(1:size(M,1)+1:end) = d;

end
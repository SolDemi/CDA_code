function result = SVM_function_singleSubj(eegData, labels, times, cfg)
% SVM_function_singleSubj
% Lightweight time-resolved / time-generalization binary SVM decoding.
%
% This version follows the interface and loop structure of
% LDA_function_singleSubj:
%   - cfg.cvType = 'holdout' or 'kfold'
%   - cfg.trainRatio for holdout
%   - cfg.nFolds for kfold
%   - cfg.nIter random iterations
%   - cfg.doShuffle shuffled-training-label baseline
%   - cfg.balanceTrials optional trial balancing
%   - cfg.useParallel parfor over cfg.nIter, not over time
%   - same output field names as LDA_function_singleSubj
%
% Required input:
%   eegData : nCh x nTime x nTrials
%   labels  : nTrials x 1, binary labels
%   times   : 1 x nTime or nTime x 1
%   cfg     : struct
%
% Key cfg fields shared with LDA_function_singleSubj:
%   cfg.cvType       : 'holdout' or 'kfold'              default = 'kfold'
%   cfg.trainRatio   : training proportion for holdout   default = 2/3
%   cfg.nFolds       : K for kfold                       default = 10
%   cfg.nIter        : random iterations                 default = 1
%   cfg.doShuffle    : shuffled-training-label baseline  default = false
%   cfg.balanceTrials: downsample classes each iter      default = false
%   cfg.useParallel  : use parfor over cfg.nIter         default = false
%   cfg.superTrial   : trials averaged into supertrial   default = 1
%
% SVM-specific cfg fields:
%   cfg.kernelFunction : SVM kernel                      default = 'linear'
%   cfg.kernelScale    : SVM KernelScale                 default = 'auto'
%   cfg.boxConstraint  : SVM BoxConstraint               default = 1
%   cfg.standardize    : z-score using training data     default = false
%
% Outputs:
%   result.predictAcc, result.AUC, result.predictAccTrain, result.weights
%   result.times, result.cfg, result.classLabelsOriginal, result.balanceInfo
%   result.predictAccShuffle, result.AUCShuffle,
%   result.predictAccMinusShuffle, result.AUCMinusShuffle if doShuffle = true
%
% Notes:
%   - Output matrices are nTimeOut x nTimeOut.
%   - If cfg.doTimeGeneralization = false, only the diagonal is filled.
%   - PCA and z-scoring are fitted on training data only.
%   - weights are returned only for linear SVM; nonlinear kernels return NaN.

if nargin < 4 || isempty(cfg), cfg = struct(); end
cfg = fill_default_cfg(cfg);

validateattributes(eegData, {'numeric'}, {'nonempty','3d'}, mfilename, 'eegData', 1);
validateattributes(labels,  {'numeric','logical'}, {'nonempty','vector'}, mfilename, 'labels', 2);
validateattributes(times,   {'numeric'}, {'nonempty','vector'}, mfilename, 'times', 3);
validateattributes(cfg,     {'struct'}, {'scalar'}, mfilename, 'cfg', 4);

labels = labels(:);
times  = times(:)';
[~, nTime, nTrials] = size(eegData);

if numel(labels) ~= nTrials || numel(times) ~= nTime
    error('Mismatch among eegData, labels, and times.');
end

uLabels = unique(labels);
if numel(uLabels) ~= 2
    error('SVM_function_singleSubj currently supports binary classification only.');
end

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed, 'twister');
end

% Temporal binning/smoothing, same convention as LDA_function_singleSubj.
if cfg.smooth_window > 0 || ~isempty(cfg.smooth_step)
    [eegData, times] = apply_temporal_window(eegData, times, cfg.smooth_window, cfg.smooth_step, cfg.timeWindowMode);
end
[nCh, nTime, ~] = size(eegData);

% Relabel to 1/2 internally, but keep original labels in result.
labels_internal = zeros(size(labels));
labels_internal(labels == uLabels(1)) = 1;
labels_internal(labels == uLabels(2)) = 2;

if strcmpi(cfg.cvType, 'holdout')
    nSplits = 1;
else
    nSplits = cfg.nFolds;
end

predictAcc_all = nan(nTime, nTime, nSplits, cfg.nIter);
AUC_all        = nan(nTime, nTime, nSplits, cfg.nIter);
trainAcc_all   = nan(nSplits, nTime, cfg.nIter);
weights_all    = nan(nCh, nTime, nSplits, cfg.nIter);

if cfg.doShuffle
    predictAccShuffle_all = nan(nTime, nTime, nSplits, cfg.nIter);
    AUCShuffle_all        = nan(nTime, nTime, nSplits, cfg.nIter);
else
    predictAccShuffle_all = [];
    AUCShuffle_all = [];
end

balanceInfo = cell(cfg.nIter, 1);

% Parallelization is at the iteration level, consistent with
% LDA_function_singleSubj. Each worker computes one complete iteration.
if cfg.useParallel
    parfor sampi = 1:cfg.nIter
        [predictAcc_iter, AUC_iter, trainAcc_iter, weights_iter, ...
            predictAccShuffle_iter, AUCShuffle_iter, balanceInfo_iter] = run_one_iteration( ...
            eegData, labels_internal, sampi, nTime, nCh, nSplits, cfg);

        predictAcc_all(:,:,:,sampi) = predictAcc_iter;
        AUC_all(:,:,:,sampi) = AUC_iter;
        trainAcc_all(:,:,sampi) = trainAcc_iter;
        weights_all(:,:,:,sampi) = weights_iter;
        balanceInfo{sampi} = balanceInfo_iter;

        if cfg.doShuffle
            predictAccShuffle_all(:,:,:,sampi) = predictAccShuffle_iter;
            AUCShuffle_all(:,:,:,sampi) = AUCShuffle_iter;
        end

        if cfg.verbose
            fprintf(' sample %d/%d done\n', sampi, cfg.nIter);
        end
    end
else
    for sampi = 1:cfg.nIter
        [predictAcc_iter, AUC_iter, trainAcc_iter, weights_iter, ...
            predictAccShuffle_iter, AUCShuffle_iter, balanceInfo_iter] = run_one_iteration( ...
            eegData, labels_internal, sampi, nTime, nCh, nSplits, cfg);

        predictAcc_all(:,:,:,sampi) = predictAcc_iter;
        AUC_all(:,:,:,sampi) = AUC_iter;
        trainAcc_all(:,:,sampi) = trainAcc_iter;
        weights_all(:,:,:,sampi) = weights_iter;
        balanceInfo{sampi} = balanceInfo_iter;

        if cfg.doShuffle
            predictAccShuffle_all(:,:,:,sampi) = predictAccShuffle_iter;
            AUCShuffle_all(:,:,:,sampi) = AUCShuffle_iter;
        end

        if cfg.verbose
            fprintf(' sample %d/%d done\n', sampi, cfg.nIter);
        end
    end
end

predictAcc = squeeze(mean(mean(predictAcc_all, 3, 'omitnan'), 4, 'omitnan'));
AUC        = squeeze(mean(mean(AUC_all,        3, 'omitnan'), 4, 'omitnan'));
weights    = squeeze(mean(mean(weights_all,    3, 'omitnan'), 4, 'omitnan'));
trainAcc   = squeeze(mean(mean(trainAcc_all,   1, 'omitnan'), 3, 'omitnan'));

if ~cfg.doTimeGeneralization
    predictAcc = keep_diagonal_only(predictAcc);
    AUC = keep_diagonal_only(AUC);
end

result = struct();
result.predictAcc = predictAcc;
result.AUC = AUC;
result.predictAccTrain = trainAcc(:);
result.weights = weights;
result.times = times(:);
result.cfg = cfg;
result.classLabelsOriginal = uLabels;
result.balanceInfo = balanceInfo;

if cfg.doShuffle
    predictAccShuffle = squeeze(mean(mean(predictAccShuffle_all, 3, 'omitnan'), 4, 'omitnan'));
    AUCShuffle        = squeeze(mean(mean(AUCShuffle_all,        3, 'omitnan'), 4, 'omitnan'));

    if ~cfg.doTimeGeneralization
        predictAccShuffle = keep_diagonal_only(predictAccShuffle);
        AUCShuffle = keep_diagonal_only(AUCShuffle);
    end

    result.predictAccShuffle = predictAccShuffle;
    result.AUCShuffle = AUCShuffle;
    result.predictAccMinusShuffle = predictAcc - predictAccShuffle;
    result.AUCMinusShuffle = AUC - AUCShuffle;
end

end

%% ========================================================================
function cfg = fill_default_cfg(cfg)
% Backward-compatible aliases used in earlier decoding scripts.
if isfield(cfg,'avgNTrials') && ~isfield(cfg,'superTrial'), cfg.superTrial = cfg.avgNTrials; end
if isfield(cfg,'binSize')    && ~isfield(cfg,'smooth_window'), cfg.smooth_window = cfg.binSize; end
if isfield(cfg,'seed')       && ~isfield(cfg,'randomSeed'), cfg.randomSeed = cfg.seed; end
if isfield(cfg,'zscore')     && ~isfield(cfg,'standardize'), cfg.standardize = cfg.zscore; end
if isfield(cfg,'svmKernel')  && ~isfield(cfg,'kernelFunction'), cfg.kernelFunction = cfg.svmKernel; end

% Shared cfg fields, matched to LDA_function_singleSubj.
if ~isfield(cfg,'cvType'),               cfg.cvType = 'kfold'; end
if ~isfield(cfg,'trainRatio'),           cfg.trainRatio = 2/3; end
if ~isfield(cfg,'nFolds'),               cfg.nFolds = 10; end
if ~isfield(cfg,'superTrial'),           cfg.superTrial = 1; end
if ~isfield(cfg,'nIter'),                cfg.nIter = 1; end
if ~isfield(cfg,'doPCA'),                cfg.doPCA = false; end
if ~isfield(cfg,'nPCs'),                 cfg.nPCs = 5; end
if ~isfield(cfg,'smooth_window'),        cfg.smooth_window = 0; end
if ~isfield(cfg,'smooth_step'),          cfg.smooth_step = []; end
if ~isfield(cfg,'timeWindowMode'),       cfg.timeWindowMode = 'centered'; end
if ~isfield(cfg,'doTimeGeneralization'), cfg.doTimeGeneralization = true; end
if ~isfield(cfg,'standardize'),          cfg.standardize = false; end
if ~isfield(cfg,'doShuffle'),            cfg.doShuffle = false; end
if ~isfield(cfg,'balanceTrials'),        cfg.balanceTrials = false; end
if ~isfield(cfg,'balanceNPerCell'),      cfg.balanceNPerCell = []; end
if ~isfield(cfg,'balanceFactors'),       cfg.balanceFactors = []; end
if ~isfield(cfg,'verbose'),              cfg.verbose = true; end
if ~isfield(cfg,'randomSeed'),           cfg.randomSeed = []; end
if ~isfield(cfg,'useParallel'),          cfg.useParallel = false; end

% SVM-specific cfg fields.
if ~isfield(cfg,'kernelFunction'),       cfg.kernelFunction = 'linear'; end
if ~isfield(cfg,'kernelScale'),          cfg.kernelScale = 'auto'; end
if ~isfield(cfg,'boxConstraint'),        cfg.boxConstraint = 1; end
if ~isfield(cfg,'solver'),               cfg.solver = []; end
if ~isfield(cfg,'scoreTransform'),       cfg.scoreTransform = []; end

cfg.cvType = lower(char(cfg.cvType));
cfg.timeWindowMode = lower(char(cfg.timeWindowMode));
cfg.kernelFunction = char(cfg.kernelFunction);

if ~ismember(cfg.cvType, {'holdout','kfold'})
    error('cfg.cvType must be either ''holdout'' or ''kfold''.');
end
if ~ismember(cfg.timeWindowMode, {'centered','bin'})
    error('cfg.timeWindowMode must be either ''centered'' or ''bin''.');
end

validateattributes(cfg.trainRatio, {'numeric'}, {'scalar','>',0,'<',1});
validateattributes(cfg.nFolds, {'numeric'}, {'scalar','integer','>=',2});
validateattributes(cfg.superTrial, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.nIter, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.doPCA, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.nPCs, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.smooth_window, {'numeric'}, {'scalar','>=',0});
if ~isempty(cfg.smooth_step)
    validateattributes(cfg.smooth_step, {'numeric'}, {'scalar','>',0});
end
validateattributes(cfg.doTimeGeneralization, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.standardize, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.doShuffle, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.balanceTrials, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.verbose, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.useParallel, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.boxConstraint, {'numeric'}, {'scalar','positive'});
end

%% ========================================================================
function [predictAcc_iter, AUC_iter, trainAcc_iter, weights_iter, ...
    predictAccShuffle_iter, AUCShuffle_iter, balanceInfo_iter] = run_one_iteration( ...
    eegData, labels_internal, sampi, nTime, nCh, nSplits, cfg)

% Give each iteration its own deterministic random stream when requested.
% This makes serial and parfor runs reproducible at the iteration level.
if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed + sampi - 1, 'twister');
end

predictAcc_iter = nan(nTime, nTime, nSplits);
AUC_iter        = nan(nTime, nTime, nSplits);
trainAcc_iter   = nan(nSplits, nTime);
weights_iter    = nan(nCh, nTime, nSplits);

if cfg.doShuffle
    predictAccShuffle_iter = nan(nTime, nTime, nSplits);
    AUCShuffle_iter        = nan(nTime, nTime, nSplits);
else
    predictAccShuffle_iter = [];
    AUCShuffle_iter = [];
end

balanceInfo_iter = [];

% Optional trial balancing. If cfg.balanceFactors is provided, balance
% jointly by [classLabel, balanceFactors]. This matches the LDA function's
% intended behavior, with a local fallback if balance_trials_by_label.m is
% not on the MATLAB path.
if cfg.balanceTrials
    balLabels = labels_internal(:);
    if ~isempty(cfg.balanceFactors)
        if size(cfg.balanceFactors, 1) ~= numel(labels_internal)
            error('cfg.balanceFactors must have the same number of rows as trials.');
        end
        balLabels = [balLabels, cfg.balanceFactors];
    end

    seedNow = [];
    if ~isempty(cfg.randomSeed), seedNow = cfg.randomSeed + sampi - 1; end

    if exist('balance_trials_by_label', 'file') == 2
        [dataIter, balLabelsOut, ~, balanceInfo_iter] = balance_trials_by_label( ...
            eegData, balLabels, 'trialDim', 3, 'nPerCell', cfg.balanceNPerCell, ...
            'seed', seedNow, 'shuffleOutput', true);
    else
        [dataIter, balLabelsOut, ~, balanceInfo_iter] = balance_trials_by_label( ...
            eegData, balLabels, cfg.balanceNPerCell, seedNow, true);
    end
    labelsIter = balLabelsOut(:,1);
else
    dataIter = eegData;
    labelsIter = labels_internal;
end

% Optional supertrials, separately within each class.
data1 = dataIter(:,:,labelsIter == 1);
data2 = dataIter(:,:,labelsIter == 2);

if cfg.superTrial > 1
    data1 = make_supertrials_consistent(data1, cfg.superTrial);
    data2 = make_supertrials_consistent(data2, cfg.superTrial);
end

allTrials = cat(3, data1, data2);
allLabels = [ones(size(data1,3),1); 2*ones(size(data2,3),1)];

if strcmpi(cfg.cvType, 'kfold') && min(histcounts(allLabels, 0.5:1:2.5)) < cfg.nFolds
    error('At least one class has fewer trials/supertrials than cfg.nFolds.');
end

[trainIdxList, testIdxList] = make_cv_splits(allLabels, cfg);

for spliti = 1:nSplits
    trainIdx = trainIdxList{spliti};
    testIdx  = testIdxList{spliti};
    trainY   = allLabels(trainIdx);
    testY    = allLabels(testIdx);

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

        predictAcc_iter(trainTime,:,spliti) = accRow;
        AUC_iter(trainTime,:,spliti) = aucRow;
        trainAcc_iter(spliti,trainTime) = trainAccVal;
        weights_iter(:,trainTime,spliti) = weightVec;

        if cfg.doShuffle
            predictAccShuffle_iter(trainTime,:,spliti) = accShufRow;
            AUCShuffle_iter(trainTime,:,spliti) = aucShufRow;
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

accRow = nan(1, nTime);
aucRow = nan(1, nTime);
accShufRow = nan(1, nTime);
aucShufRow = nan(1, nTime);
weightVec = nan(nCh, 1);

Xtrain = squeeze(allTrials(:, trainTime, trainIdx))';
if isvector(Xtrain), Xtrain = reshape(Xtrain, sum(trainIdx), nCh); end

% Manual train-only z-scoring is used instead of fitcsvm(...,'Standardize',true)
% so that PCA, test-time projection, and weight back-projection follow the
% same logic as LDA_function_singleSubj.
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

svmModel = fit_svm_model(Xtrain, trainY, cfg);
if cfg.doShuffle
    svmModelShuffle = fit_svm_model(Xtrain, trainYShuffle, cfg);
else
    svmModelShuffle = [];
end

% Linear SVM weights in original channel space. For nonlinear kernels,
% a single channel-space weight vector is not well-defined.
if strcmpi(cfg.kernelFunction, 'linear') && isprop(svmModel, 'Beta') && ~isempty(svmModel.Beta)
    w_use = svmModel.Beta;
    if cfg.doPCA, w_use = coeff * w_use; end
    if cfg.standardize, w_use = w_use ./ sigma_z(:); end
    weightVec = w_use;
end

labelTrain = predict(svmModel, Xtrain);
trainAccVal = mean(labelTrain == trainY);

if cfg.doTimeGeneralization
    testTimes = 1:nTime;
else
    testTimes = trainTime;
end

for testTime = testTimes
    Xtest = squeeze(allTrials(:, testTime, testIdx))';
    if isvector(Xtest), Xtest = reshape(Xtest, sum(testIdx), nCh); end

    if cfg.standardize, Xtest = (Xtest - mu_z) ./ sigma_z; end
    if cfg.doPCA, Xtest = (Xtest - mu_pca) * coeff; end

    [labelTest, score] = predict(svmModel, Xtest);
    accRow(testTime) = mean(labelTest == testY);
    aucRow(testTime) = binary_auc(testY, score, svmModel.ClassNames, 2);

    if cfg.doShuffle
        [labelShuf, scoreShuf] = predict(svmModelShuffle, Xtest);
        accShufRow(testTime) = mean(labelShuf == testY);
        aucShufRow(testTime) = binary_auc(testY, scoreShuf, svmModelShuffle.ClassNames, 2);
    end
end
end

%% ========================================================================
function svmModel = fit_svm_model(Xtrain, trainY, cfg)
args = {'KernelFunction', cfg.kernelFunction, ...
        'Standardize', false, ...
        'BoxConstraint', cfg.boxConstraint};

if ~isempty(cfg.kernelScale)
    args = [args, {'KernelScale', cfg.kernelScale}]; 
end
if ~isempty(cfg.solver)
    args = [args, {'Solver', cfg.solver}]; 
end
if ~isempty(cfg.scoreTransform)
    args = [args, {'ScoreTransform', cfg.scoreTransform}]; 
end

svmModel = fitcsvm(Xtrain, trainY, args{:});
end

%% ========================================================================
function [eegOut, timesOut] = apply_temporal_window(eegData, times, win, step, modeName)
if isempty(step), step = win; end

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

    if isempty(centerIdx)
        error('No valid time points remain after temporal windowing.');
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
if maxPC < 1
    error('Not enough observations/features for PCA.');
end
[coeff, score, ~, ~, ~, mu] = pca(X);
coeff = coeff(:,1:maxPC);
score = score(:,1:maxPC);
end

%% ========================================================================
function aucVal = binary_auc(testY, score, classNames, posClass)
aucVal = NaN;
posCol = find(classNames == posClass, 1);
if ~isempty(posCol) && size(score,2) >= posCol && numel(unique(testY)) == 2
    [~,~,~,aucVal] = perfcurve(testY, score(:,posCol), posClass);
end
end

%% ========================================================================
function dataSup = make_supertrials_consistent(data, superTrial)
if superTrial <= 1
    dataSup = data;
    return;
end

if exist('func_make_superTrials', 'file') == 2
    dataSup = func_make_superTrials(data, superTrial);
else
    dataSup = make_supertrials_local(data, superTrial);
end
end

%% ========================================================================
function dataSup = make_supertrials_local(data, superTrial)
[nCh, nTime, nTrials] = size(data);
nSuperTrials = floor(nTrials / superTrial);
if nSuperTrials < 1
    error('Not enough trials (%d) to create supertrials with superTrial = %d.', nTrials, superTrial);
end

randIdx = randperm(nTrials);
randIdx = randIdx(1:nSuperTrials * superTrial);
dataSup = zeros(nCh, nTime, nSuperTrials, 'like', data);

for si = 1:nSuperTrials
    useIdx = randIdx((si-1)*superTrial + 1 : si*superTrial);
    dataSup(:,:,si) = mean(data(:,:,useIdx), 3, 'omitnan');
end
end

%% ========================================================================
function M = keep_diagonal_only(M)
d = diag(M);
M(:) = NaN;
M(1:size(M,1)+1:end) = d;
end

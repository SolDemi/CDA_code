function result = LDA_function_singleSubj(eegData, labels, times, cfg)
% LDA_function_singleSubj
%
% Time-resolved / time-generalization binary LDA decoding for single-subject
% EEG/iEEG data. The interface and output format are intentionally matched
% to SVM_function_confusion_singleSubj.m.
%
% INPUT
%   eegData : nCh x nTime x nTrials
%   labels  : nTrials x 1, binary class labels. Original labels are stored
%             in result.classLabelsOriginal, but internal labels are 1/2.
%   times   : 1 x nTime or nTime x 1, same unit as smooth_window/smooth_step
%   cfg     : struct
%
% COMMON CFG FIELDS
%   cfg.nFolds               : number of CV folds, default = 10
%   cfg.superTrial           : number of trials averaged into one supertrial,
%                              default = 1
%   cfg.nIter                : number of random resampling runs, default = 1
%   cfg.doPCA                : true/false, default = false
%   cfg.nPCs                 : number of PCs if doPCA = true, default = 5
%   cfg.smooth_window        : temporal smoothing window size in same unit as
%                              times, default = 0. If 0, no averaging.
%   cfg.smooth_step          : step size between adjacent output time points,
%                              default = []. If [], every valid time point is
%                              used. If smooth_window = 0 and smooth_step > 0,
%                              only temporal downsampling is applied.
%   cfg.doTimeGeneralization : true/false, default = true
%   cfg.discrimType          : fitcdiscr DiscrimType, default = 'diagLinear'
%   cfg.standardize          : z-score features using training set only,
%                              default = false
%   cfg.verbose              : true/false, default = true
%   cfg.randomSeed           : numeric scalar or [], default = []
%   cfg.useParallel          : true/false, use parfor over trainTime,
%                              default = false
%
% BACKWARD-COMPATIBLE CFG ALIASES
%   cfg.avgNTrials -> cfg.superTrial
%   cfg.binSize    -> cfg.smooth_window
%   cfg.seed       -> cfg.randomSeed
%   cfg.zscore     -> cfg.standardize
%
% OUTPUT
%   result.predictAcc           : nTimeOut x nTimeOut
%   result.AUC                  : nTimeOut x nTimeOut
%   result.predictAccTrain      : nTimeOut x 1
%   result.weights              : nCh x nTimeOut
%   result.times                : processed time vector
%   result.cfg                  : full config used
%   result.classLabelsOriginal  : original class labels
%
% NOTES
%   - Output matrices are always nTimeOut x nTimeOut.
%   - If doTimeGeneralization = false, only the diagonal is filled; off-
%     diagonal entries are NaN.
%   - PCA and standardization are fitted on training data only within each
%     fold/time point, avoiding train-test leakage.
%   - weights are returned in the original channel/feature space when PCA
%     and/or standardization are used.

%% =========================
% 0. Input check
% ==========================
if nargin < 4
    cfg = struct();
end

validateattributes(eegData, {'numeric'}, {'nonempty','3d'}, mfilename, 'eegData', 1);
validateattributes(labels,  {'numeric','logical'}, {'nonempty','vector'}, mfilename, 'labels', 2);
validateattributes(times,   {'numeric'}, {'nonempty','vector'}, mfilename, 'times', 3);
validateattributes(cfg,     {'struct'}, {'scalar'}, mfilename, 'cfg', 4);

labels = labels(:);
times  = times(:)';
[nCh, nTime, nTrials] = size(eegData);

if numel(labels) ~= nTrials
    error('Number of labels (%d) must equal number of trials in eegData (%d).', numel(labels), nTrials);
end
if numel(times) ~= nTime
    error('Length of times (%d) must equal size(eegData,2) (%d).', numel(times), nTime);
end

uLabels = unique(labels);
if numel(uLabels) ~= 2
    error('This version currently supports binary classification only. Found %d unique labels.', numel(uLabels));
end

%% =========================
% 1. Default cfg
% ==========================
cfg = fill_default_cfg(cfg);

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed);
end

%% =========================
% 2. Optional temporal smoothing / stepping
% ==========================
if cfg.smooth_window > 0 || ~isempty(cfg.smooth_step)
    [eegData, times] = apply_temporal_smoothing(eegData, times, cfg.smooth_window, cfg.smooth_step);
end
[nCh, nTime, ~] = size(eegData);

%% =========================
% 3. Relabel to 1 / 2 internally
% ==========================
labels_internal = zeros(size(labels));
labels_internal(labels == uLabels(1)) = 1;
labels_internal(labels == uLabels(2)) = 2;

idx1 = labels_internal == 1;
idx2 = labels_internal == 2;

data1 = eegData(:,:,idx1);
data2 = eegData(:,:,idx2);

n1 = size(data1,3);
n2 = size(data2,3);
if min(n1,n2) < cfg.nFolds
    error('At least one class has fewer trials than nFolds. class1=%d, class2=%d, nFolds=%d', n1, n2, cfg.nFolds);
end

%% =========================
% 4. Preallocate
% ==========================
predictAcc_all = nan(nTime, nTime, cfg.nFolds, cfg.nIter);
AUC_all        = nan(nTime, nTime, cfg.nFolds, cfg.nIter);
trainAcc_all   = nan(cfg.nFolds, nTime, cfg.nIter);
weights_all    = nan(nCh, nTime, cfg.nFolds, cfg.nIter);

%% =========================
% 5. Main loop
% ==========================
for sampi = 1:cfg.nIter

    % ----- Supertrials, separately within each class -----
    if cfg.superTrial > 1
        dataSup1 = make_supertrials(data1, cfg.superTrial);
        dataSup2 = make_supertrials(data2, cfg.superTrial);
    else
        dataSup1 = data1;
        dataSup2 = data2;
    end

    allTrials = cat(3, dataSup1, dataSup2);
    allLabels = [ones(size(dataSup1,3),1); 2*ones(size(dataSup2,3),1)];

    if min([sum(allLabels==1), sum(allLabels==2)]) < cfg.nFolds
        error(['After supertrial averaging, at least one class has fewer samples ' ...
               'than nFolds. class1=%d, class2=%d, nFolds=%d'], ...
               sum(allLabels==1), sum(allLabels==2), cfg.nFolds);
    end

    % Stratified CV
    CVO = cvpartition(allLabels, 'KFold', cfg.nFolds);

    for foldi = 1:cfg.nFolds
        trainIdx = training(CVO, foldi);
        testIdx  = test(CVO, foldi);

        trainY = allLabels(trainIdx);
        testY  = allLabels(testIdx);

        if cfg.useParallel
            parfor trainTime = 1:nTime
                [accRow, aucRow, trainAccVal, weightVec] = decode_one_train_time_lda( ...
                    allTrials, trainIdx, testIdx, trainY, testY, ...
                    trainTime, nTime, nCh, cfg);

                predictAcc_all(trainTime,:,foldi,sampi) = accRow;
                AUC_all(trainTime,:,foldi,sampi)        = aucRow;
                trainAcc_all(foldi,trainTime,sampi)     = trainAccVal;
                weights_all(:,trainTime,foldi,sampi)    = weightVec;
            end
        else
            for trainTime = 1:nTime
                [accRow, aucRow, trainAccVal, weightVec] = decode_one_train_time_lda( ...
                    allTrials, trainIdx, testIdx, trainY, testY, ...
                    trainTime, nTime, nCh, cfg);

                predictAcc_all(trainTime,:,foldi,sampi) = accRow;
                AUC_all(trainTime,:,foldi,sampi)        = aucRow;
                trainAcc_all(foldi,trainTime,sampi)     = trainAccVal;
                weights_all(:,trainTime,foldi,sampi)    = weightVec;
            end
        end
    end

    if cfg.verbose
        fprintf(' sample %d/%d done\n', sampi, cfg.nIter);
    end
end

%% =========================
% 6. Average across folds and iterations
% ==========================
predictAcc = mean(predictAcc_all, [3 4], 'omitnan');
AUC        = mean(AUC_all,        [3 4], 'omitnan');
weights    = mean(weights_all,    [3 4], 'omitnan');
trainAcc   = squeeze(mean(trainAcc_all, [1 3], 'omitnan'))';

%% =========================
% 7. If no TG, keep only diagonal and off-diagonal NaN
% ==========================
if ~cfg.doTimeGeneralization
    predictAcc = keep_diagonal_only(predictAcc);
    AUC        = keep_diagonal_only(AUC);
end

%% =========================
% 8. Pack output
% ==========================
result = struct();
result.predictAcc          = predictAcc;
result.AUC                 = AUC;
result.predictAccTrain     = trainAcc(:);
result.weights             = weights;
result.times               = times(:);
result.cfg                 = cfg;
result.classLabelsOriginal = uLabels;

end

%% ========================================================================
function [accRow, aucRow, trainAccVal, weightVec] = decode_one_train_time_lda( ...
    allTrials, trainIdx, testIdx, trainY, testY, trainTime, nTime, nCh, cfg)
% Decode one training time point. The helper returns complete row vectors so
% that the trainTime loop can be safely parallelized with parfor.

accRow      = nan(1, nTime);
aucRow      = nan(1, nTime);
trainAccVal = NaN;
weightVec   = nan(nCh, 1);

% ----- Training data -----
Xtrain = squeeze(allTrials(:, trainTime, trainIdx))';  % nTrain x nCh
if isvector(Xtrain)
    Xtrain = reshape(Xtrain, sum(trainIdx), nCh);
end

% ----- Standardize on training data only -----
mu_z = [];
sigma_z = [];
if cfg.standardize
    [Xtrain, mu_z, sigma_z] = zscore_train_only(Xtrain);
end

% ----- PCA on training data only -----
coeff = [];
mu_pca = [];
if cfg.doPCA
    [coeff, Xtrain, mu_pca] = fit_pca_train_only(Xtrain, cfg.nPCs);
end

% ----- Train LDA -----
ldaModel = fitcdiscr(Xtrain, trainY, 'DiscrimType', cfg.discrimType);

% ----- Save weights in original feature space -----
w_use = estimate_lda_weights(Xtrain, trainY, ldaModel, cfg.discrimType);
if cfg.doPCA
    weightVec = coeff * w_use;
else
    weightVec = w_use;
end
if cfg.standardize
    weightVec = weightVec ./ sigma_z(:);
end

% ----- Training accuracy -----
labelTrain = predict(ldaModel, Xtrain);
trainAccVal = mean(labelTrain == trainY);

% ----- Define test times -----
if cfg.doTimeGeneralization
    testTimes = 1:nTime;
else
    testTimes = trainTime;
end

% ----- Test -----
for testTime = testTimes
    Xtest = squeeze(allTrials(:, testTime, testIdx))';  % nTest x nCh
    if isvector(Xtest)
        Xtest = reshape(Xtest, sum(testIdx), nCh);
    end

    if cfg.standardize
        Xtest = apply_zscore_to_test(Xtest, mu_z, sigma_z);
    end

    if cfg.doPCA
        Xtest = apply_pca_to_test(Xtest, coeff, mu_pca);
    end

    [labelTest, score] = predict(ldaModel, Xtest);
    accRow(testTime) = mean(labelTest == testY);

    % ----- AUC. Internal positive class is class 2. -----
    try
        posClass = 2;
        posCol = find(ldaModel.ClassNames == posClass, 1, 'first');
        if isempty(posCol)
            aucVal = NaN;
        elseif numel(unique(testY)) == 2
            [~,~,~,aucVal] = perfcurve(testY, score(:,posCol), posClass);
        else
            aucVal = NaN;
        end
    catch
        aucVal = NaN;
    end
    aucRow(testTime) = aucVal;
end

end

%% ========================================================================
function cfg = fill_default_cfg(cfg)
% Backward-compatible aliases from the old LDA_function_singleSubj.m.
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

if ~isfield(cfg,'nFolds'),               cfg.nFolds = 10; end
if ~isfield(cfg,'superTrial'),           cfg.superTrial = 1; end
if ~isfield(cfg,'nIter'),                cfg.nIter = 1; end
if ~isfield(cfg,'doPCA'),                cfg.doPCA = false; end
if ~isfield(cfg,'nPCs'),                 cfg.nPCs = 5; end
if ~isfield(cfg,'smooth_window'),        cfg.smooth_window = 0; end
if ~isfield(cfg,'smooth_step'),          cfg.smooth_step = []; end
if ~isfield(cfg,'doTimeGeneralization'), cfg.doTimeGeneralization = true; end
if ~isfield(cfg,'discrimType'),          cfg.discrimType = 'diagLinear'; end
if ~isfield(cfg,'standardize'),          cfg.standardize = false; end
if ~isfield(cfg,'verbose'),              cfg.verbose = true; end
if ~isfield(cfg,'randomSeed'),           cfg.randomSeed = []; end
if ~isfield(cfg,'useParallel'),          cfg.useParallel = false; end

validateattributes(cfg.nFolds, {'numeric'}, {'scalar','integer','>=',2});
validateattributes(cfg.superTrial, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.nIter, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.doPCA, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.nPCs, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.smooth_window, {'numeric'}, {'scalar','>=',0});
validateattributes(cfg.doTimeGeneralization, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.standardize, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.verbose, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.useParallel, {'numeric','logical'}, {'scalar'});

if ~isempty(cfg.smooth_step)
    validateattributes(cfg.smooth_step, {'numeric'}, {'scalar','>',0});
end
if ~ischar(cfg.discrimType) && ~isstring(cfg.discrimType)
    error('cfg.discrimType must be a char or string.');
end
end

%% ========================================================================
function [eegData_out, times_out] = apply_temporal_smoothing(eegData, times, smooth_window, smooth_step)
% Apply temporal smoothing with optional step size.
%
% INPUT
%   eegData       : nCh x nTime x nTrials
%   times         : 1 x nTime
%   smooth_window : window size, same unit as times. If 0, no averaging.
%   smooth_step   : step size between adjacent output time points. If [],
%                   every valid time point is used as output center.
%
% OUTPUT
%   eegData_out : nCh x nOutputTime x nTrials
%   times_out   : output time vector corresponding to window centers

times = times(:)';
if nargin < 4
    smooth_step = [];
end
validateattributes(smooth_window, {'numeric'}, {'scalar','>=',0});
if ~isempty(smooth_step)
    validateattributes(smooth_step, {'numeric'}, {'scalar','>',0});
end

if smooth_window == 0 && isempty(smooth_step)
    eegData_out = eegData;
    times_out = times;
    return;
end

if smooth_window > 0
    half_window = smooth_window / 2;
    center_min = times(1) + half_window;
    center_max = times(end) - half_window;
    if center_min > center_max
        error('No valid time points remain. smooth_window is too large for the current time range.');
    end
    validCenterIdx = find(times >= center_min & times <= center_max);
else
    validCenterIdx = 1:numel(times);
end
if isempty(validCenterIdx)
    error('No valid output time points were found.');
end

if isempty(smooth_step)
    centerIdx = validCenterIdx;
else
    targetTimes = times(validCenterIdx(1)) : smooth_step : times(validCenterIdx(end));
    if isempty(targetTimes)
        targetTimes = times(validCenterIdx(1));
    end
    centerIdx = nan(size(targetTimes));
    for ii = 1:numel(targetTimes)
        [~, nearestPos] = min(abs(times(validCenterIdx) - targetTimes(ii)));
        centerIdx(ii) = validCenterIdx(nearestPos);
    end
    centerIdx = unique(centerIdx, 'stable');
end

nCh = size(eegData, 1);
nTrials = size(eegData, 3);
nOut = numel(centerIdx);
eegData_out = zeros(nCh, nOut, nTrials, 'like', eegData);

if smooth_window > 0
    half_window = smooth_window / 2;
    for ii = 1:nOut
        tIdx = centerIdx(ii);
        centerTime = times(tIdx);
        winIdx = times >= (centerTime - half_window) & times <= (centerTime + half_window);
        eegData_out(:, ii, :) = mean(eegData(:, winIdx, :), 2);
    end
else
    eegData_out = eegData(:, centerIdx, :);
end

times_out = times(centerIdx);
end

%% ========================================================================
function dataSup = make_supertrials(data, superTrial)
% Randomly average trials within each class to create supertrials.
%
% INPUT
%   data       : nCh x nTime x nTrials
%   superTrial : number of raw trials averaged into one supertrial
%
% OUTPUT
%   dataSup : nCh x nTime x nSuperTrials

[nCh, nTime, nTrials] = size(data);
if superTrial <= 1
    dataSup = data;
    return;
end

nSuperTrials = floor(nTrials / superTrial);
if nSuperTrials < 1
    error('Not enough trials (%d) to create supertrials with superTrial = %d.', nTrials, superTrial);
end

randIdx = randperm(nTrials);
randIdx = randIdx(1:nSuperTrials * superTrial);

dataSup = zeros(nCh, nTime, nSuperTrials, 'like', data);
for si = 1:nSuperTrials
    useIdx = randIdx((si-1)*superTrial + 1 : si*superTrial);
    dataSup(:,:,si) = mean(data(:,:,useIdx), 3);
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
function Xz = apply_zscore_to_test(X, mu_z, sigma_z)
Xz = (X - mu_z) ./ sigma_z;
end

%% ========================================================================
function [coeff, Xtrain_pca, mu] = fit_pca_train_only(Xtrain, nPCs)
[nObs, nFeat] = size(Xtrain);
maxPC = min([nObs-1, nFeat, nPCs]);
if maxPC < 1
    error('Not enough training observations/features for PCA.');
end
[coeff, score, ~, ~, ~, mu] = pca(Xtrain);
coeff = coeff(:,1:maxPC);
Xtrain_pca = score(:,1:maxPC);
end

%% ========================================================================
function Xtest_pca = apply_pca_to_test(Xtest, coeff, mu)
Xtest_pca = (Xtest - mu) * coeff;
end

%% ========================================================================
function w = estimate_lda_weights(X, y, mdl, discrimType)
% Return a binary LDA discriminant vector in the current feature space.
% Positive weights favor class 2 relative to class 1.

% First try MATLAB's model coefficient. This works for standard two-class
% discriminant models and is consistent with the fitted model.
try
    if numel(mdl.ClassNames) == 2 && isfield(mdl.Coeffs(1,2), 'Linear')
        w = mdl.Coeffs(1,2).Linear;
        w = w(:);
        if numel(w) == size(X,2)
            return;
        end
    end
catch
end

% Fallback: manually estimate the discriminant vector. This is especially
% transparent for diagLinear.
c1 = mdl.ClassNames(1);
c2 = mdl.ClassNames(2);
X1 = X(y == c1, :);
X2 = X(y == c2, :);
mu1 = mean(X1, 1);
mu2 = mean(X2, 1);

if strcmpi(discrimType, 'linear')
    S1 = cov(X1);
    S2 = cov(X2);
    n1 = size(X1,1);
    n2 = size(X2,1);
    pooledCov = ((n1 - 1) * S1 + (n2 - 1) * S2) / max(n1 + n2 - 2, 1);
    pooledCov = pooledCov + eye(size(pooledCov)) * eps;
    w = pinv(pooledCov) * (mu2 - mu1)';
else
    v1 = var(X1, 0, 1);
    v2 = var(X2, 0, 1);
    n1 = size(X1,1);
    n2 = size(X2,1);
    pooledVar = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1);
    pooledVar(pooledVar <= eps | isnan(pooledVar)) = eps;
    w = ((mu2 - mu1) ./ pooledVar)';
end
end

%% ========================================================================
function M = keep_diagonal_only(M)
diagVals = diag(M);
M(:) = NaN;
M(1:size(M,1)+1:end) = diagVals;
end

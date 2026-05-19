function result = SVM_crossSide_singleSubj(trainData, trainLabels, testData, testLabels, times, cfg)
% SVM_crossSide_singleSubj
% Train a binary SVM on one side/condition subset and test on another.
% This is intended for cross-side load generalization:
%   train: attended-left low vs high
%   test : attended-right low vs high
%
% Inputs:
% trainData/testData: channels x time x trials
% trainLabels/testLabels: binary labels, e.g. 1=low, 2=high
% times: original time vector
% cfg: mostly follows SVM_function_singleSubj.m

if nargin < 6 || isempty(cfg), cfg = struct(); end
cfg = fill_default_cfg(cfg);
trainLabels = trainLabels(:);
testLabels = testLabels(:);
times = times(:)';

if ~isempty(cfg.randomSeed), rng(cfg.randomSeed, 'twister'); end

origTimes = times(:)';

[trainData, times] = apply_temporal_window(trainData, origTimes, ...
    cfg.smooth_window, cfg.smooth_step, cfg.timeWindowMode);

[testData,  ~] = apply_temporal_window(testData,  origTimes, ...
    cfg.smooth_window, cfg.smooth_step, cfg.timeWindowMode);

[nCh, nTime, ~] = size(trainData);

uLabels = unique([trainLabels; testLabels]);
trainY0 = relabel_binary(trainLabels, uLabels);
testY0  = relabel_binary(testLabels,  uLabels);

predictAcc_all = nan(nTime, nTime, cfg.nIter);
trainAcc_all = nan(nTime, cfg.nIter);
weights_all = nan(nCh, nTime, cfg.nIter);
if cfg.useAUC
    AUC_all = nan(nTime, nTime, cfg.nIter);
else
    AUC_all = [];
end

if cfg.doShuffle
    predictAccShuffle_all = nan(nTime, nTime, cfg.nIter);
    if cfg.useAUC
        AUCShuffle_all = nan(nTime, nTime, cfg.nIter);
    else
        AUCShuffle_all = [];
    end
else
    predictAccShuffle_all = [];
    AUCShuffle_all = [];
end

for iter = 1:cfg.nIter
    if ~isempty(cfg.randomSeed), rng(cfg.randomSeed + iter - 1, 'twister'); end

    [trDat, trY] = balance_and_supertrial(trainData, trainY0, cfg);
    [teDat, teY] = balance_and_supertrial(testData,  testY0,  cfg);

    if numel(unique(trY)) < 2 || numel(unique(teY)) < 2
        continue;
    end

    for trainTime = 1:nTime
        Xtrain = squeeze(trDat(:, trainTime, :))';
        if isvector(Xtrain), Xtrain = reshape(Xtrain, numel(trY), nCh); end

        [Xtrain, mu_z, sigma_z] = zscore_train_only(Xtrain, cfg.standardize);
        [coeff, Xtrain, mu_pca] = pca_train_only(Xtrain, cfg.doPCA, cfg.nPCs);

        svmModel = fit_svm_model(Xtrain, trY, cfg);
        trainPred = predict(svmModel, Xtrain);
        trainAcc_all(trainTime, iter) = mean(trainPred == trY);

        if strcmpi(cfg.kernelFunction, 'linear') && isprop(svmModel, 'Beta') && ~isempty(svmModel.Beta)
            w = svmModel.Beta;
            if cfg.doPCA, w = coeff * w; end
            if cfg.standardize, w = w ./ sigma_z(:); end
            weights_all(:, trainTime, iter) = w;
        end

        if cfg.doShuffle
            trYShuffle = trY(randperm(numel(trY)));
            svmModelShuffle = fit_svm_model(Xtrain, trYShuffle, cfg);
        end

        if cfg.doTimeGeneralization
            testTimes = 1:nTime;
        else
            testTimes = trainTime;
        end

        parfor testTime = testTimes
            Xtest = squeeze(teDat(:, testTime, :))';
            if isvector(Xtest), Xtest = reshape(Xtest, numel(teY), nCh); end
            if cfg.standardize, Xtest = (Xtest - mu_z) ./ sigma_z; end
            if cfg.doPCA, Xtest = (Xtest - mu_pca) * coeff; end

            if cfg.useAUC
                [labelTest, score] = predict(svmModel, Xtest);
                AUC_all(trainTime, testTime, iter) = binary_auc(teY, score, svmModel.ClassNames, 2);
            else
                labelTest = predict(svmModel, Xtest);
            end
            predictAcc_all(trainTime, testTime, iter) = mean(labelTest == teY);

            if cfg.doShuffle
                if cfg.useAUC
                    [labelShuf, scoreShuf] = predict(svmModelShuffle, Xtest);
                    AUCShuffle_all(trainTime, testTime, iter) = binary_auc(teY, scoreShuf, svmModelShuffle.ClassNames, 2);
                else
                    labelShuf = predict(svmModelShuffle, Xtest);
                end
                predictAccShuffle_all(trainTime, testTime, iter) = mean(labelShuf == teY);
            end
        end
    end
end

result = struct();
result.predictAcc = mean(predictAcc_all, 3, 'omitnan');
result.predictAccTrain = mean(trainAcc_all, 2, 'omitnan');
result.weights = mean(weights_all, 3, 'omitnan');
result.times = times(:);
result.cfg = cfg;
result.classLabelsOriginal = uLabels;

if cfg.useAUC
    result.AUC = mean(AUC_all, 3, 'omitnan');
end

if cfg.doShuffle
    result.predictAccShuffle = mean(predictAccShuffle_all, 3, 'omitnan');
    result.predictAccMinusShuffle = result.predictAcc - result.predictAccShuffle;
    if cfg.useAUC
        result.AUCShuffle = mean(AUCShuffle_all, 3, 'omitnan');
        result.AUCMinusShuffle = result.AUC - result.AUCShuffle;
    end
end
end

%% ========================================================================
function cfg = fill_default_cfg(cfg)
if ~isfield(cfg,'superTrial'), cfg.superTrial = 1; end
if ~isfield(cfg,'nIter'), cfg.nIter = 1; end
if ~isfield(cfg,'doPCA'), cfg.doPCA = false; end
if ~isfield(cfg,'nPCs'), cfg.nPCs = 5; end
if ~isfield(cfg,'smooth_window'), cfg.smooth_window = 0; end
if ~isfield(cfg,'smooth_step'), cfg.smooth_step = []; end
if ~isfield(cfg,'timeWindowMode'), cfg.timeWindowMode = 'centered'; end
if ~isfield(cfg,'doTimeGeneralization'), cfg.doTimeGeneralization = true; end
if ~isfield(cfg,'standardize'), cfg.standardize = false; end
if ~isfield(cfg,'doShuffle'), cfg.doShuffle = false; end
if ~isfield(cfg,'useAUC'), cfg.useAUC = false; end
if ~isfield(cfg,'balanceTrials'), cfg.balanceTrials = true; end
if ~isfield(cfg,'balanceNPerCell'), cfg.balanceNPerCell = []; end
if ~isfield(cfg,'randomSeed'), cfg.randomSeed = []; end
if ~isfield(cfg,'kernelFunction'), cfg.kernelFunction = 'linear'; end
if ~isfield(cfg,'kernelScale'), cfg.kernelScale = 'auto'; end
if ~isfield(cfg,'boxConstraint'), cfg.boxConstraint = 1; end
cfg.timeWindowMode = lower(char(cfg.timeWindowMode));
cfg.kernelFunction = char(cfg.kernelFunction);
end

function y = relabel_binary(labels, uLabels)
y = zeros(size(labels));
y(labels == uLabels(1)) = 1;
y(labels == uLabels(2)) = 2;
end

function [dataOut, labelsOut] = balance_and_supertrial(data, labels, cfg)
if cfg.balanceTrials
    [data, labels] = balance_by_label_simple(data, labels, cfg.balanceNPerCell);
end

data1 = data(:,:,labels == 1);
data2 = data(:,:,labels == 2);
if cfg.superTrial > 1
    data1 = func_make_superTrials(data1, cfg.superTrial);
    data2 = func_make_superTrials(data2, cfg.superTrial);
end

dataOut = cat(3, data1, data2);
labelsOut = [ones(size(data1,3),1); 2*ones(size(data2,3),1)];
ord = randperm(numel(labelsOut));
dataOut = dataOut(:,:,ord);
labelsOut = labelsOut(ord);
end

function [dataOut, labelsOut] = balance_by_label_simple(data, labels, nPerCell)
u = unique(labels);
if isempty(nPerCell)
    nPerCell = min(arrayfun(@(x) sum(labels == x), u));
end
idxKeep = [];
for i = 1:numel(u)
    idx = find(labels == u(i));
    idx = idx(randperm(numel(idx), nPerCell));
    idxKeep = [idxKeep; idx(:)]; %#ok<AGROW>
end
idxKeep = idxKeep(randperm(numel(idxKeep)));
dataOut = data(:,:,idxKeep);
labelsOut = labels(idxKeep);
end

function svmModel = fit_svm_model(Xtrain, trainY, cfg)
args = {'KernelFunction', cfg.kernelFunction, ...
    'Standardize', false, ...
    'BoxConstraint', cfg.boxConstraint};
if ~isempty(cfg.kernelScale)
    args = [args, {'KernelScale', cfg.kernelScale}]; 
end
svmModel = fitcsvm(Xtrain, trainY, args{:});
end

function [Xz, mu_z, sigma_z] = zscore_train_only(X, doStandardize)
mu_z = [];
sigma_z = [];
Xz = X;
if doStandardize
    mu_z = mean(X, 1, 'omitnan');
    sigma_z = std(X, 0, 1, 'omitnan');
    sigma_z(sigma_z == 0 | isnan(sigma_z)) = 1;
    Xz = (X - mu_z) ./ sigma_z;
end
end

function [coeff, score, mu] = pca_train_only(X, doPCA, nPCs)
coeff = [];
score = X;
mu = [];
if doPCA
    maxPC = min([size(X,1)-1, size(X,2), nPCs]);
    [coeff, score, ~, ~, ~, mu] = pca(X);
    coeff = coeff(:,1:maxPC);
    score = score(:,1:maxPC);
end
end

function aucVal = binary_auc(testY, score, classNames, posClass)
aucVal = NaN;
posCol = find(classNames == posClass, 1);
if ~isempty(posCol) && size(score,2) >= posCol && numel(unique(testY)) == 2
    [~,~,~,aucVal] = perfcurve(testY, score(:,posCol), posClass);
end
end

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
    if ~isempty(step) && step > 0
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

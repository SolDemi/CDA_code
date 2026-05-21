function result = LDA_crossSide_singleSubj(trainData, trainLabels, testData, testLabels, times, cfg)
% LDA_crossSide_singleSubj
% Train a binary LDA decoder on one side and test it on the other side.
%
% Data format:
%   trainData/testData: channels x time x trials
%   labels            : binary labels, e.g., 1 = low load, 2 = high load
%
% Main output:
%   result.Acc: train-time x test-time accuracy
%               If cfg.doTimeGeneralization = false, only the diagonal is filled.

if nargin < 6 || isempty(cfg), cfg = struct(); end
cfg = set_default_cfg(cfg);

trainLabels = trainLabels(:);
testLabels  = testLabels(:);
times       = times(:)';

% Core dimension checks
if ndims(trainData) ~= 3 || ndims(testData) ~= 3
    error('trainData and testData must be channels x time x trials.');
end
if size(trainData,1) ~= size(testData,1)
    error('trainData and testData must have the same number of channels.');
end
if size(trainData,2) ~= numel(times) || size(testData,2) ~= numel(times)
    error('The time dimension must match numel(times).');
end
if size(trainData,3) ~= numel(trainLabels) || size(testData,3) ~= numel(testLabels)
    error('The trial dimension must match the label vector.');
end

classLabels = unique([trainLabels; testLabels]);
if numel(classLabels) ~= 2
    error('This function supports binary classification only.');
end

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed, 'twister');
end

% Relabel to 1/2 internally, while keeping original labels in the output.
trainY0 = 1 + double(trainLabels == classLabels(2));
testY0  = 1 + double(testLabels  == classLabels(2));

origTimes = times;
[trainData, times] = apply_temporal_window(trainData, origTimes, cfg.smooth_window, ...
    cfg.smooth_step, cfg.timeWindowMode);
[testData,  ~] = apply_temporal_window(testData,  origTimes, cfg.smooth_window, ...
    cfg.smooth_step, cfg.timeWindowMode);

[nCh, nTime, ~] = size(trainData);

Acc_all      = nan(nTime, nTime, cfg.nIter);
AccTrain_all = nan(nTime, cfg.nIter);

if cfg.saveWeights
    weights_all = nan(nCh, nTime, cfg.nIter);
else
    weights_all = [];
end

if cfg.useAUC
    AUC_all = nan(nTime, nTime, cfg.nIter);
else
    AUC_all = [];
end

if cfg.doShuffle
    AccShuffle_all = nan(nTime, nTime, cfg.nIter);
    if cfg.useAUC
        AUCShuffle_all = nan(nTime, nTime, cfg.nIter);
    else
        AUCShuffle_all = [];
    end
else
    AccShuffle_all = [];
    AUCShuffle_all = [];
end

balanceInfo = cell(cfg.nIter, 1);

for iter = 1:cfg.nIter
    if ~isempty(cfg.randomSeed)
        rng(cfg.randomSeed + iter - 1, 'twister');
    end

    [trDat, trY, infoTrain] = prepare_trials(trainData, trainY0, cfg);
    [teDat, teY, infoTest]  = prepare_trials(testData,  testY0,  cfg);
    balanceInfo{iter} = struct('train', infoTrain, 'test', infoTest);

    for trainTime = 1:nTime
        Xtrain = get_time_data(trDat, trainTime);

        if cfg.standardize
            mu_z = mean(Xtrain, 1, 'omitnan');
            sd_z = std(Xtrain, 0, 1, 'omitnan');
            sd_z(sd_z == 0 | isnan(sd_z)) = 1;
            Xtrain = (Xtrain - mu_z) ./ sd_z;
        else
            mu_z = [];
            sd_z = [];
        end

        if cfg.doPCA
            [coeff, score, ~, ~, ~, mu_pca] = pca(Xtrain);
            nPC = min([cfg.nPCs, size(coeff,2), size(Xtrain,1)-1]);
            coeff = coeff(:, 1:nPC);
            Xtrain = score(:, 1:nPC);
        else
            coeff = [];
            mu_pca = [];
        end

        ldaModel = fitcdiscr(Xtrain, trY, ...
            'DiscrimType', cfg.discrimType, ...
            'Prior', cfg.prior);

        trainPred = predict(ldaModel, Xtrain);
        AccTrain_all(trainTime, iter) = mean(trainPred == trY);

        if cfg.saveWeights
            w = ldaModel.Coeffs(1,2).Linear;
            if cfg.doPCA
                w = coeff * w;
            end
            if cfg.standardize
                w = w ./ sd_z(:);
            end
            weights_all(:, trainTime, iter) = w;
        end

        if cfg.doShuffle
            shufY = trY(randperm(numel(trY)));
            ldaShuffle = fitcdiscr(Xtrain, shufY, ...
                'DiscrimType', cfg.discrimType, ...
                'Prior', cfg.prior);
        end

        if cfg.doTimeGeneralization
            testTimes = 1:nTime;
        else
            testTimes = trainTime;
        end

        for testTime = testTimes
            Xtest = get_time_data(teDat, testTime);

            if cfg.standardize
                Xtest = (Xtest - mu_z) ./ sd_z;
            end
            if cfg.doPCA
                Xtest = (Xtest - mu_pca) * coeff;
            end

            if cfg.useAUC
                [pred, score] = predict(ldaModel, Xtest);
                AUC_all(trainTime, testTime, iter) = get_binary_auc(teY, score, ldaModel.ClassNames, 2);
            else
                pred = predict(ldaModel, Xtest);
            end
            Acc_all(trainTime, testTime, iter) = mean(pred == teY);

            if cfg.doShuffle
                if cfg.useAUC
                    [predShuf, scoreShuf] = predict(ldaShuffle, Xtest);
                    AUCShuffle_all(trainTime, testTime, iter) = get_binary_auc(teY, scoreShuf, ldaShuffle.ClassNames, 2);
                else
                    predShuf = predict(ldaShuffle, Xtest);
                end
                AccShuffle_all(trainTime, testTime, iter) = mean(predShuf == teY);
            end
        end
    end

    if cfg.verbose
        fprintf('cross-side LDA iteration %d/%d done\n', iter, cfg.nIter);
    end
end

result = struct();
result.Acc = mean(Acc_all, 3, 'omitnan');
result.AccTrain = mean(AccTrain_all, 2, 'omitnan');
result.times = times(:);
result.cfg = cfg;
result.classLabelsOriginal = classLabels;
result.balanceInfo = balanceInfo;

if cfg.saveWeights
    result.weights = mean(weights_all, 3, 'omitnan');
end

if cfg.useAUC
    result.AUC = mean(AUC_all, 3, 'omitnan');
end

if cfg.doShuffle
    result.AccShuffle = mean(AccShuffle_all, 3, 'omitnan');
    result.AccMinusShuffle = result.Acc - result.AccShuffle;

    if cfg.useAUC
        result.AUCShuffle = mean(AUCShuffle_all, 3, 'omitnan');
        result.AUCMinusShuffle = result.AUC - result.AUCShuffle;
    end
end

end

%% ========================================================================
function cfg = set_default_cfg(cfg)

% Backward-compatible aliases
if isfield(cfg, 'avgNTrials') && ~isfield(cfg, 'superTrial')
    cfg.superTrial = cfg.avgNTrials;
end
if isfield(cfg, 'binSize') && ~isfield(cfg, 'smooth_window')
    cfg.smooth_window = cfg.binSize;
end
if isfield(cfg, 'seed') && ~isfield(cfg, 'randomSeed')
    cfg.randomSeed = cfg.seed;
end
if isfield(cfg, 'zscore') && ~isfield(cfg, 'standardize')
    cfg.standardize = cfg.zscore;
end

if ~isfield(cfg, 'superTrial'), cfg.superTrial = 1; end
if ~isfield(cfg, 'nIter'), cfg.nIter = 1; end
if ~isfield(cfg, 'balanceTrials'), cfg.balanceTrials = true; end
if ~isfield(cfg, 'balanceNPerCell'), cfg.balanceNPerCell = []; end

if ~isfield(cfg, 'doTimeGeneralization'), cfg.doTimeGeneralization = true; end
if ~isfield(cfg, 'discrimType'), cfg.discrimType = 'diaglinear'; end
if ~isfield(cfg, 'prior'), cfg.prior = 'uniform'; end

if ~isfield(cfg, 'standardize'), cfg.standardize = false; end
if ~isfield(cfg, 'doPCA'), cfg.doPCA = false; end
if ~isfield(cfg, 'nPCs'), cfg.nPCs = 5; end

if ~isfield(cfg, 'smooth_window'), cfg.smooth_window = 0; end
if ~isfield(cfg, 'smooth_step'), cfg.smooth_step = []; end
if ~isfield(cfg, 'timeWindowMode'), cfg.timeWindowMode = 'centered'; end

if ~isfield(cfg, 'doShuffle'), cfg.doShuffle = false; end
if ~isfield(cfg, 'useAUC'), cfg.useAUC = false; end
if ~isfield(cfg, 'saveWeights'), cfg.saveWeights = false; end

if ~isfield(cfg, 'randomSeed'), cfg.randomSeed = []; end
if ~isfield(cfg, 'verbose'), cfg.verbose = true; end

cfg.discrimType = lower(char(cfg.discrimType));
cfg.timeWindowMode = lower(char(cfg.timeWindowMode));

end

%% ========================================================================
function [dataOut, labelsOut, info] = prepare_trials(data, labels, cfg)

idx1 = find(labels == 1);
idx2 = find(labels == 2);

info = struct();
info.nOriginal = [numel(idx1), numel(idx2)];

if cfg.balanceTrials
    n = min(info.nOriginal);
    if ~isempty(cfg.balanceNPerCell)
        n = min(n, cfg.balanceNPerCell);
    end
    if cfg.superTrial > 1
        n = floor(n / cfg.superTrial) * cfg.superTrial;
    end

    idx1 = idx1(randperm(numel(idx1), n));
    idx2 = idx2(randperm(numel(idx2), n));

elseif cfg.superTrial > 1
    n1 = floor(numel(idx1) / cfg.superTrial) * cfg.superTrial;
    n2 = floor(numel(idx2) / cfg.superTrial) * cfg.superTrial;
    idx1 = idx1(randperm(numel(idx1), n1));
    idx2 = idx2(randperm(numel(idx2), n2));
end

info.idxKeep = [idx1(:); idx2(:)];
info.nAfterBalance = [numel(idx1), numel(idx2)];

data1 = data(:, :, idx1);
data2 = data(:, :, idx2);

if cfg.superTrial > 1
    data1 = func_make_superTrials(data1, cfg.superTrial);
    data2 = func_make_superTrials(data2, cfg.superTrial);
end

dataOut = cat(3, data1, data2);
labelsOut = [ones(size(data1,3),1); 2 * ones(size(data2,3),1)];

ord = randperm(numel(labelsOut));
dataOut = dataOut(:, :, ord);
labelsOut = labelsOut(ord);

info.nAfterSuperTrial = [sum(labelsOut == 1), sum(labelsOut == 2)];

end

%% ========================================================================
function X = get_time_data(data, t)

nCh = size(data, 1);
X = permute(data(:, t, :), [3 1 2]);
X = reshape(X, [], nCh);

end

%% ========================================================================
function aucVal = get_binary_auc(y, score, classNames, posClass)

aucVal = NaN;
posCol = find(classNames == posClass, 1);

if ~isempty(posCol) && numel(unique(y)) == 2
    [~, ~, ~, aucVal] = perfcurve(y, score(:, posCol), posClass);
end

end

%% ========================================================================
function [dataOut, timesOut] = apply_temporal_window(data, times, win, step, modeName)

if isempty(step)
    if win > 0
        step = win;
    else
        step = [];
    end
end

if win <= 0 && isempty(step)
    dataOut = data;
    timesOut = times;
    return;
end

if strcmpi(modeName, 'bin')
    binStarts = times(1):step:(times(end) - win);
    dataOut = zeros(size(data,1), numel(binStarts), size(data,3), 'like', data);
    timesOut = binStarts + win / 2;

    for i = 1:numel(binStarts)
        t1 = binStarts(i);
        t2 = t1 + win;

        if i < numel(binStarts)
            tidx = times >= t1 & times < t2;
        else
            tidx = times >= t1 & times <= t2;
        end

        dataOut(:, i, :) = mean(data(:, tidx, :), 2, 'omitnan');
    end
    return;
end

if win > 0
    centerIdx = find(times >= times(1) + win/2 & times <= times(end) - win/2);
else
    centerIdx = 1:numel(times);
end

if ~isempty(step) && step > 0
    targetTimes = times(centerIdx(1)):step:times(centerIdx(end));
    newIdx = nan(size(targetTimes));

    for i = 1:numel(targetTimes)
        [~, k] = min(abs(times(centerIdx) - targetTimes(i)));
        newIdx(i) = centerIdx(k);
    end

    centerIdx = unique(newIdx, 'stable');
end

dataOut = zeros(size(data,1), numel(centerIdx), size(data,3), 'like', data);
timesOut = times(centerIdx);

for i = 1:numel(centerIdx)
    ct = times(centerIdx(i));

    if win > 0
        tidx = times >= ct - win/2 & times <= ct + win/2;
    else
        tidx = centerIdx(i);
    end

    dataOut(:, i, :) = mean(data(:, tidx, :), 2, 'omitnan');
end

end

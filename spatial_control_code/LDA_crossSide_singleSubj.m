function result = LDA_crossSide_singleSubj(trainData, trainLabels, testData, testLabels, times, cfg)
% LDA_crossSide_singleSubj
% Train a binary LDA decoder on one side and test it on the other side.
%
% Data format:
%   trainData/testData: channels x time x trials
%   labels            : binary labels, e.g., 1 = low load, 2 = high load

if nargin < 6 || isempty(cfg), cfg = struct(); end

% Backward-compatible aliases.
if isfield(cfg, 'avgNTrials') && ~isfield(cfg, 'superTrial'), cfg.superTrial = cfg.avgNTrials; end
if isfield(cfg, 'binSize') && ~isfield(cfg, 'smooth_window'), cfg.smooth_window = cfg.binSize; end
if isfield(cfg, 'seed') && ~isfield(cfg, 'randomSeed'), cfg.randomSeed = cfg.seed; end
if isfield(cfg, 'zscore') && ~isfield(cfg, 'standardize'), cfg.standardize = cfg.zscore; end

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

trainLabels = trainLabels(:);
testLabels  = testLabels(:);
times       = times(:)';

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

trainY0 = 1 + double(trainLabels == classLabels(2));
testY0  = 1 + double(testLabels  == classLabels(2));

%% Optional temporal binning/smoothing
origTimes = times;
dataInput = {trainData, testData};
dataOutput = cell(1, 2);
for dataIdx = 1:2
    thisData = dataInput{dataIdx};
    step = cfg.smooth_step;
    if isempty(step)
        if cfg.smooth_window > 0
            step = cfg.smooth_window;
        else
            step = [];
        end
    end

    if cfg.smooth_window <= 0 && isempty(step)
        thisOut = thisData;
        thisTimes = origTimes;
    elseif strcmpi(cfg.timeWindowMode, 'bin')
        binStarts = origTimes(1):step:(origTimes(end) - cfg.smooth_window);
        thisOut = zeros(size(thisData,1), numel(binStarts), size(thisData,3), 'like', thisData);
        thisTimes = binStarts + cfg.smooth_window / 2;

        for bi = 1:numel(binStarts)
            t1 = binStarts(bi);
            t2 = t1 + cfg.smooth_window;

            if bi < numel(binStarts)
                tidx = origTimes >= t1 & origTimes < t2;
            else
                tidx = origTimes >= t1 & origTimes <= t2;
            end

            thisOut(:, bi, :) = mean(thisData(:, tidx, :), 2, 'omitnan');
        end
    else
        if cfg.smooth_window > 0
            centerIdx = find(origTimes >= origTimes(1) + cfg.smooth_window/2 & ...
                origTimes <= origTimes(end) - cfg.smooth_window/2);
        else
            centerIdx = 1:numel(origTimes);
        end

        if ~isempty(step) && step > 0
            targetTimes = origTimes(centerIdx(1)):step:origTimes(centerIdx(end));
            newIdx = nan(size(targetTimes));

            for ii = 1:numel(targetTimes)
                [~, k] = min(abs(origTimes(centerIdx) - targetTimes(ii)));
                newIdx(ii) = centerIdx(k);
            end

            centerIdx = unique(newIdx, 'stable');
        end

        thisOut = zeros(size(thisData,1), numel(centerIdx), size(thisData,3), 'like', thisData);
        thisTimes = origTimes(centerIdx);

        for ii = 1:numel(centerIdx)
            ct = origTimes(centerIdx(ii));

            if cfg.smooth_window > 0
                tidx = origTimes >= ct - cfg.smooth_window/2 & origTimes <= ct + cfg.smooth_window/2;
            else
                tidx = centerIdx(ii);
            end

            thisOut(:, ii, :) = mean(thisData(:, tidx, :), 2, 'omitnan');
        end
    end

    dataOutput{dataIdx} = thisOut;
    if dataIdx == 1
        times = thisTimes;
    end
end
trainData = dataOutput{1};
testData = dataOutput{2};

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

    iterDataInput = {trainData, testData};
    iterLabelsInput = {trainY0, testY0};
    iterDataOutput = cell(1, 2);
    iterLabelsOutput = cell(1, 2);
    iterInfo = cell(1, 2);

    for dataIdx = 1:2
        thisData = iterDataInput{dataIdx};
        thisLabels = iterLabelsInput{dataIdx};
        idx1 = find(thisLabels == 1);
        idx2 = find(thisLabels == 2);

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

        data1 = thisData(:, :, idx1);
        data2 = thisData(:, :, idx2);

        if cfg.superTrial > 1
            data1 = func_make_superTrials(data1, cfg.superTrial);
            data2 = func_make_superTrials(data2, cfg.superTrial);
        end

        thisData = cat(3, data1, data2);
        thisLabels = [ones(size(data1,3),1); 2 * ones(size(data2,3),1)];

        ord = randperm(numel(thisLabels));
        thisData = thisData(:, :, ord);
        thisLabels = thisLabels(ord);

        info.nAfterSuperTrial = [sum(thisLabels == 1), sum(thisLabels == 2)];
        iterDataOutput{dataIdx} = thisData;
        iterLabelsOutput{dataIdx} = thisLabels;
        iterInfo{dataIdx} = info;
    end

    trDat = iterDataOutput{1};
    trY = iterLabelsOutput{1};
    teDat = iterDataOutput{2};
    teY = iterLabelsOutput{2};
    balanceInfo{iter} = struct('train', iterInfo{1}, 'test', iterInfo{2});

    for trainTime = 1:nTime
        Xtrain = permute(trDat(:, trainTime, :), [3 1 2]);
        Xtrain = reshape(Xtrain, [], nCh);

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
            Xtest = permute(teDat(:, testTime, :), [3 1 2]);
            Xtest = reshape(Xtest, [], nCh);

            if cfg.standardize
                Xtest = (Xtest - mu_z) ./ sd_z;
            end
            if cfg.doPCA
                Xtest = (Xtest - mu_pca) * coeff;
            end

            if cfg.useAUC
                [pred, score] = predict(ldaModel, Xtest);
                aucVal = NaN;
                posCol = find(ldaModel.ClassNames == 2, 1);
                if ~isempty(posCol) && numel(unique(teY)) == 2
                    [~, ~, ~, aucVal] = perfcurve(teY, score(:, posCol), 2);
                end
                AUC_all(trainTime, testTime, iter) = aucVal;
            else
                pred = predict(ldaModel, Xtest);
            end
            Acc_all(trainTime, testTime, iter) = mean(pred == teY);

            if cfg.doShuffle
                if cfg.useAUC
                    [predShuf, scoreShuf] = predict(ldaShuffle, Xtest);
                    aucVal = NaN;
                    posCol = find(ldaShuffle.ClassNames == 2, 1);
                    if ~isempty(posCol) && numel(unique(teY)) == 2
                        [~, ~, ~, aucVal] = perfcurve(teY, scoreShuf(:, posCol), 2);
                    end
                    AUCShuffle_all(trainTime, testTime, iter) = aucVal;
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

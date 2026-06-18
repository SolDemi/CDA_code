function result = SVM_crossSide_singleSubj(trainData, trainLabels, testData, testLabels, times, cfg)
% SVM_crossSide_singleSubj
% Train a binary SVM on one side/condition subset and test on another.
% This is intended for cross-side load generalization:
%   train: attended-left low vs high
%   test : attended-right low vs high

if nargin < 6 || isempty(cfg), cfg = struct(); end
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

trainLabels = trainLabels(:);
testLabels = testLabels(:);
times = times(:)';

if ~isempty(cfg.randomSeed), rng(cfg.randomSeed, 'twister'); end

origTimes = times(:)';
dataInput = {trainData, testData};
dataOutput = cell(1, 2);
for dataIdx = 1:2
    thisData = dataInput{dataIdx};
    step = cfg.smooth_step;
    if isempty(step), step = cfg.smooth_window; end

    if strcmpi(cfg.timeWindowMode, 'bin')
        binStarts = origTimes(1):step:(origTimes(end)-cfg.smooth_window);
        thisOut = zeros(size(thisData,1), numel(binStarts), size(thisData,3), 'like', thisData);
        thisTimes = binStarts + cfg.smooth_window/2;

        for bi = 1:numel(binStarts)
            t1 = binStarts(bi);
            t2 = t1 + cfg.smooth_window;
            if bi < numel(binStarts)
                tidx = origTimes >= t1 & origTimes < t2;
            else
                tidx = origTimes >= t1 & origTimes <= t2;
            end
            thisOut(:,bi,:) = mean(thisData(:,tidx,:), 2, 'omitnan');
        end
    else
        if cfg.smooth_window > 0
            halfWin = cfg.smooth_window/2;
            centerIdx = find(origTimes >= origTimes(1)+halfWin & origTimes <= origTimes(end)-halfWin);
        else
            centerIdx = 1:numel(origTimes);
        end

        if ~isempty(step) && step > 0
            targetTimes = origTimes(centerIdx(1)):step:origTimes(centerIdx(end));
            tmpIdx = nan(size(targetTimes));
            for ii = 1:numel(targetTimes)
                [~, k] = min(abs(origTimes(centerIdx)-targetTimes(ii)));
                tmpIdx(ii) = centerIdx(k);
            end
            centerIdx = unique(tmpIdx, 'stable');
        end

        thisOut = zeros(size(thisData,1), numel(centerIdx), size(thisData,3), 'like', thisData);
        thisTimes = origTimes(centerIdx);
        for ii = 1:numel(centerIdx)
            ct = origTimes(centerIdx(ii));
            if cfg.smooth_window > 0
                tidx = origTimes >= ct-cfg.smooth_window/2 & origTimes <= ct+cfg.smooth_window/2;
            else
                tidx = centerIdx(ii);
            end
            thisOut(:,ii,:) = mean(thisData(:,tidx,:), 2, 'omitnan');
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

uLabels = unique([trainLabels; testLabels]);
trainY0 = zeros(size(trainLabels));
trainY0(trainLabels == uLabels(1)) = 1;
trainY0(trainLabels == uLabels(2)) = 2;
testY0 = zeros(size(testLabels));
testY0(testLabels == uLabels(1)) = 1;
testY0(testLabels == uLabels(2)) = 2;

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

    iterDataInput = {trainData, testData};
    iterLabelsInput = {trainY0, testY0};
    iterDataOutput = cell(1, 2);
    iterLabelsOutput = cell(1, 2);

    for dataIdx = 1:2
        thisData = iterDataInput{dataIdx};
        thisLabels = iterLabelsInput{dataIdx};

        if cfg.balanceTrials
            u = unique(thisLabels);
            if isempty(cfg.balanceNPerCell)
                nPerCell = min(arrayfun(@(x) sum(thisLabels == x), u));
            else
                nPerCell = cfg.balanceNPerCell;
            end
            idxKeep = [];
            for ui = 1:numel(u)
                idx = find(thisLabels == u(ui));
                idx = idx(randperm(numel(idx), nPerCell));
                idxKeep = [idxKeep; idx(:)]; %#ok<AGROW>
            end
            idxKeep = idxKeep(randperm(numel(idxKeep)));
            thisData = thisData(:,:,idxKeep);
            thisLabels = thisLabels(idxKeep);
        end

        data1 = thisData(:,:,thisLabels == 1);
        data2 = thisData(:,:,thisLabels == 2);
        if cfg.superTrial > 1
            data1 = func_make_superTrials(data1, cfg.superTrial);
            data2 = func_make_superTrials(data2, cfg.superTrial);
        end

        thisData = cat(3, data1, data2);
        thisLabels = [ones(size(data1,3),1); 2*ones(size(data2,3),1)];
        ord = randperm(numel(thisLabels));
        iterDataOutput{dataIdx} = thisData(:,:,ord);
        iterLabelsOutput{dataIdx} = thisLabels(ord);
    end

    trDat = iterDataOutput{1};
    trY = iterLabelsOutput{1};
    teDat = iterDataOutput{2};
    teY = iterLabelsOutput{2};

    if numel(unique(trY)) < 2 || numel(unique(teY)) < 2
        continue;
    end

    for trainTime = 1:nTime
        Xtrain = squeeze(trDat(:, trainTime, :))';
        if isvector(Xtrain), Xtrain = reshape(Xtrain, numel(trY), nCh); end

        mu_z = [];
        sigma_z = [];
        if cfg.standardize
            mu_z = mean(Xtrain, 1, 'omitnan');
            sigma_z = std(Xtrain, 0, 1, 'omitnan');
            sigma_z(sigma_z == 0 | isnan(sigma_z)) = 1;
            Xtrain = (Xtrain - mu_z) ./ sigma_z;
        end

        coeff = [];
        mu_pca = [];
        if cfg.doPCA
            maxPC = min([size(Xtrain,1)-1, size(Xtrain,2), cfg.nPCs]);
            [coeff, Xtrain, ~, ~, ~, mu_pca] = pca(Xtrain);
            coeff = coeff(:,1:maxPC);
            Xtrain = Xtrain(:,1:maxPC);
        end

        args = {'KernelFunction', cfg.kernelFunction, ...
            'Standardize', false, ...
            'BoxConstraint', cfg.boxConstraint};
        if ~isempty(cfg.kernelScale)
            args = [args, {'KernelScale', cfg.kernelScale}]; %#ok<AGROW>
        end
        svmModel = fitcsvm(Xtrain, trY, args{:});
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
            svmModelShuffle = fitcsvm(Xtrain, trYShuffle, args{:});
        else
            svmModelShuffle = [];
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
                aucVal = NaN;
                posCol = find(svmModel.ClassNames == 2, 1);
                if ~isempty(posCol) && size(score,2) >= posCol && numel(unique(teY)) == 2
                    [~,~,~,aucVal] = perfcurve(teY, score(:,posCol), 2);
                end
                AUC_all(trainTime, testTime, iter) = aucVal;
            else
                labelTest = predict(svmModel, Xtest);
            end
            predictAcc_all(trainTime, testTime, iter) = mean(labelTest == teY);

            if cfg.doShuffle
                if cfg.useAUC
                    [labelShuf, scoreShuf] = predict(svmModelShuffle, Xtest);
                    aucVal = NaN;
                    posCol = find(svmModelShuffle.ClassNames == 2, 1);
                    if ~isempty(posCol) && size(scoreShuf,2) >= posCol && numel(unique(teY)) == 2
                        [~,~,~,aucVal] = perfcurve(teY, scoreShuf(:,posCol), 2);
                    end
                    AUCShuffle_all(trainTime, testTime, iter) = aucVal;
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

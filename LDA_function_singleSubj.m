function result = LDA_function_singleSubj(eegData, labels, times, cfg)
% LDA_function_singleSubj
% Lightweight time-resolved / time-generalization binary LDA decoding.

if nargin < 4 || isempty(cfg), cfg = struct(); end

% Backward-compatible aliases and defaults.
if isfield(cfg,'avgNTrials') && ~isfield(cfg,'superTrial'), cfg.superTrial = cfg.avgNTrials; end
if isfield(cfg,'binSize') && ~isfield(cfg,'smooth_window'), cfg.smooth_window = cfg.binSize; end
if isfield(cfg,'seed') && ~isfield(cfg,'randomSeed'), cfg.randomSeed = cfg.seed; end
if isfield(cfg,'zscore') && ~isfield(cfg,'standardize'), cfg.standardize = cfg.zscore; end

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
if ~isfield(cfg,'parallelDimension'), cfg.parallelDimension = 'auto'; end
if ~isfield(cfg,'parallelMaxWorkers'), cfg.parallelMaxWorkers = Inf; end
if ~isfield(cfg,'returnDecisionValues'), cfg.returnDecisionValues = false; end

cfg.cvType = lower(char(cfg.cvType));
cfg.timeWindowMode = lower(char(cfg.timeWindowMode));
cfg.ldaEngine = lower(char(cfg.ldaEngine));
cfg.parallelDimension = lower(char(cfg.parallelDimension));

if ~ismember(cfg.cvType, {'holdout','kfold'})
    error('cfg.cvType must be ''holdout'' or ''kfold''.');
end
if ~ismember(cfg.ldaEngine, {'fitcdiscr','classify'})
    error('cfg.ldaEngine must be ''fitcdiscr'' or ''classify''.');
end
if ~ismember(cfg.parallelDimension, {'auto','iteration','iter','split','fold','kfold','none'})
    error('cfg.parallelDimension must be ''auto'', ''iteration'', ''split'', or ''none''.');
end

labels = labels(:);
times  = times(:)';

[~, nTime, nTrials] = size(eegData);
if numel(labels) ~= nTrials || numel(times) ~= nTime
    error('Mismatch among eegData, labels, and times.');
end
if cfg.returnDecisionValues && cfg.superTrial ~= 1
    error('Trial-level decision values require cfg.superTrial = 1.');
end

uLabels = unique(labels);
if numel(uLabels) ~= 2
    error('LDA_function_singleSubj currently supports binary classification only.');
end

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed, 'twister');
end

%% Temporal binning/smoothing
if cfg.smooth_window > 0 || ~isempty(cfg.smooth_step)
    win = cfg.smooth_window;
    step = cfg.smooth_step;
    if isempty(step), step = win; end

    if strcmpi(cfg.timeWindowMode, 'bin')
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
                [~, k] = min(abs(times(centerIdx)-targetTimes(ii)));
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

    eegData = eegOut;
    times = timesOut;
end
[nCh, nTime, ~] = size(eegData);

labels_internal = zeros(size(labels));
labels_internal(labels == uLabels(1)) = 1;
labels_internal(labels == uLabels(2)) = 2;

if strcmpi(cfg.cvType, 'holdout')
    nSplits = 1;
else
    nSplits = cfg.nFolds;
end

if ~cfg.useParallel
    cfg.parallelDimensionEffective = 'none';
else
    switch lower(cfg.parallelDimension)
        case 'auto'
            if cfg.nIter > 1
                cfg.parallelDimensionEffective = 'iteration';
            elseif nSplits > 1
                cfg.parallelDimensionEffective = 'split';
            else
                cfg.parallelDimensionEffective = 'none';
            end
        case {'iteration','iter'}
            if cfg.nIter > 1
                cfg.parallelDimensionEffective = 'iteration';
            elseif nSplits > 1
                cfg.parallelDimensionEffective = 'split';
            else
                cfg.parallelDimensionEffective = 'none';
            end
        case {'split','fold','kfold'}
            if nSplits > 1, cfg.parallelDimensionEffective = 'split'; else, cfg.parallelDimensionEffective = 'none'; end
        case 'none'
            cfg.parallelDimensionEffective = 'none';
    end
end
parallelWorkers = 0;
if strcmpi(cfg.parallelDimensionEffective, 'none')
    splitWorkers = 0;
else
    parallelWorkers = cfg.parallelMaxWorkers;
    if parallelWorkers <= 0
        cfg.parallelDimensionEffective = 'none';
        parallelWorkers = 0;
    elseif isempty(gcp('nocreate'))
        if isfinite(parallelWorkers)
            parpool(parallelWorkers);
        else
            parpool;
        end
    end
    if strcmpi(cfg.parallelDimensionEffective, 'split')
        splitWorkers = parallelWorkers;
    else
        splitWorkers = 0;
    end
end

Acc_all = nan(nTime, nTime, nSplits, cfg.nIter);
trainAcc_all = nan(nSplits, nTime, cfg.nIter);
weights_all = nan(nCh, nTime, nSplits, cfg.nIter);

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

if cfg.returnDecisionValues
    decisionScore_all = nan(nTrials, nTime, nTime, cfg.nIter);
    decisionCount_all = zeros(nTrials, nTime, nTime, cfg.nIter);
else
    decisionScore_all = [];
    decisionCount_all = [];
end

balanceInfo = cell(cfg.nIter, 1);
useClassify = strcmpi(cfg.ldaEngine, 'classify');

%% Trial balancing, supertrial construction, CV split, model training/testing
if strcmpi(cfg.parallelDimensionEffective, 'iteration')
    parfor (sampi = 1:cfg.nIter, parallelWorkers)
        [Acc_iter, trainAcc_iter, weights_iter, AUC_iter, AccShuffle_iter, AUCShuffle_iter, ...
            balanceInfo_iter, decisionScore_iter, decisionCount_iter] = ...
            lda_run_iteration(eegData, labels_internal, sampi, nSplits, nCh, nTime, cfg, useClassify, 0);

        Acc_all(:,:,:,sampi) = Acc_iter;
        trainAcc_all(:,:,sampi) = trainAcc_iter;
        weights_all(:,:,:,sampi) = weights_iter;
        if cfg.useAUC, AUC_all(:,:,:,sampi) = AUC_iter; end
        if cfg.doShuffle
            AccShuffle_all(:,:,:,sampi) = AccShuffle_iter;
            if cfg.useAUC, AUCShuffle_all(:,:,:,sampi) = AUCShuffle_iter; end
        end
        if cfg.returnDecisionValues
            decisionScore_all(:,:,:,sampi) = decisionScore_iter;
            decisionCount_all(:,:,:,sampi) = decisionCount_iter;
        end
        balanceInfo{sampi} = balanceInfo_iter;
        if cfg.verbose
            fprintf(' sample %d/%d done\n', sampi, cfg.nIter);
        end
    end
else
    for sampi = 1:cfg.nIter
        [Acc_iter, trainAcc_iter, weights_iter, AUC_iter, AccShuffle_iter, AUCShuffle_iter, ...
            balanceInfo_iter, decisionScore_iter, decisionCount_iter] = ...
            lda_run_iteration(eegData, labels_internal, sampi, nSplits, nCh, nTime, cfg, useClassify, splitWorkers);

        Acc_all(:,:,:,sampi) = Acc_iter;
        trainAcc_all(:,:,sampi) = trainAcc_iter;
        weights_all(:,:,:,sampi) = weights_iter;
        if cfg.useAUC, AUC_all(:,:,:,sampi) = AUC_iter; end
        if cfg.doShuffle
            AccShuffle_all(:,:,:,sampi) = AccShuffle_iter;
            if cfg.useAUC, AUCShuffle_all(:,:,:,sampi) = AUCShuffle_iter; end
        end
        if cfg.returnDecisionValues
            decisionScore_all(:,:,:,sampi) = decisionScore_iter;
            decisionCount_all(:,:,:,sampi) = decisionCount_iter;
        end
        balanceInfo{sampi} = balanceInfo_iter;
        if cfg.verbose
            fprintf(' sample %d/%d done\n', sampi, cfg.nIter);
        end
    end
end

Acc = squeeze(mean(mean(Acc_all, 3, 'omitnan'), 4, 'omitnan'));
weights = squeeze(mean(mean(weights_all, 3, 'omitnan'), 4, 'omitnan'));
trainAcc = squeeze(mean(mean(trainAcc_all, 1, 'omitnan'), 3, 'omitnan'));

if cfg.useAUC
    AUC = squeeze(mean(mean(AUC_all, 3, 'omitnan'), 4, 'omitnan'));
end

if ~cfg.doTimeGeneralization
    d = diag(Acc);
    Acc(:) = NaN;
    Acc(1:size(Acc,1)+1:end) = d;
    if cfg.useAUC
        d = diag(AUC);
        AUC(:) = NaN;
        AUC(1:size(AUC,1)+1:end) = d;
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

if cfg.returnDecisionValues
    decisionScoreSum = decisionScore_all;
    decisionScoreSum(isnan(decisionScoreSum)) = 0;
    decisionCount = sum(decisionCount_all, 4);
    decisionScore = sum(decisionScoreSum, 4) ./ decisionCount;
    decisionScore(decisionCount == 0) = NaN;

    result.decisionValues = struct();
    result.decisionValues.scoreClass2 = decisionScore;
    result.decisionValues.nTest = decisionCount;
    result.decisionValues.classLabelInternal = 2;
    result.decisionValues.classLabelOriginal = uLabels(2);
    result.decisionValues.scoreMeaning = 'Cross-validated score/posterior for class 2, averaged across iterations; dimensions are trial x trainTime x testTime.';
end

if cfg.doShuffle
    AccShuffle = squeeze(mean(mean(AccShuffle_all, 3, 'omitnan'), 4, 'omitnan'));
    if ~cfg.doTimeGeneralization
        d = diag(AccShuffle);
        AccShuffle(:) = NaN;
        AccShuffle(1:size(AccShuffle,1)+1:end) = d;
    end

    result.AccShuffle = AccShuffle;
    result.AccMinusShuffle = Acc - AccShuffle;

    if cfg.useAUC
        AUCShuffle = squeeze(mean(mean(AUCShuffle_all, 3, 'omitnan'), 4, 'omitnan'));
        if ~cfg.doTimeGeneralization
            d = diag(AUCShuffle);
            AUCShuffle(:) = NaN;
            AUCShuffle(1:size(AUCShuffle,1)+1:end) = d;
        end

        result.AUCShuffle = AUCShuffle;
        result.AUCMinusShuffle = AUC - AUCShuffle;
    end
end

end

function [Acc_iter, trainAcc_iter, weights_iter, AUC_iter, AccShuffle_iter, AUCShuffle_iter, ...
    balanceInfo_iter, decisionScore_iter, decisionCount_iter] = ...
    lda_run_iteration(eegData, labels_internal, sampi, nSplits, nCh, nTime, cfg, useClassify, splitWorkers)

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed + sampi - 1, 'twister');
end

if cfg.balanceTrials
    balLabels = labels_internal(:);
    if ~isempty(cfg.balanceFactors)
        balLabels = [balLabels, cfg.balanceFactors];
    end

    seedNow = [];
    if ~isempty(cfg.randomSeed), seedNow = cfg.randomSeed + sampi - 1; end

    [dataIter, balLabelsOut, ~, balanceInfo_iter] = balance_trials_by_label( ...
        eegData, balLabels, ...
        'trialDim', 3, ...
        'nPerCell', cfg.balanceNPerCell, ...
        'seed', seedNow, ...
        'shuffleOutput', true);

    labelsIter = balLabelsOut(:,1);
    originalTrialIter = balanceInfo_iter.idxBal(:);
else
    dataIter = eegData;
    labelsIter = labels_internal;
    originalTrialIter = (1:numel(labels_internal))';
    balanceInfo_iter = [];
end

data1 = dataIter(:,:,labelsIter == 1);
data2 = dataIter(:,:,labelsIter == 2);
originalTrial1 = originalTrialIter(labelsIter == 1);
originalTrial2 = originalTrialIter(labelsIter == 2);

if cfg.superTrial > 1
    data1 = func_make_superTrials(data1, cfg.superTrial);
    data2 = func_make_superTrials(data2, cfg.superTrial);
end

allTrials = cat(3, data1, data2);
allLabels = [ones(size(data1,3),1); 2*ones(size(data2,3),1)];
allOriginalTrials = [originalTrial1; originalTrial2];

if strcmpi(cfg.cvType, 'holdout')
    CVO = cvpartition(allLabels, 'HoldOut', 1 - cfg.trainRatio);
    trainIdxList = {training(CVO)};
    testIdxList  = {test(CVO)};
else
    CVO = cvpartition(allLabels, 'KFold', cfg.nFolds);
    trainIdxList = cell(cfg.nFolds, 1);
    testIdxList = cell(cfg.nFolds, 1);
    for fi = 1:cfg.nFolds
        trainIdxList{fi} = training(CVO, fi);
        testIdxList{fi} = test(CVO, fi);
    end
end

Acc_iter = nan(nTime, nTime, nSplits);
trainAcc_iter = nan(nSplits, nTime);
weights_iter = nan(nCh, nTime, nSplits);
if cfg.useAUC, AUC_iter = nan(nTime, nTime, nSplits); else, AUC_iter = []; end
if cfg.doShuffle
    AccShuffle_iter = nan(nTime, nTime, nSplits);
    if cfg.useAUC, AUCShuffle_iter = nan(nTime, nTime, nSplits); else, AUCShuffle_iter = []; end
else
    AccShuffle_iter = [];
    AUCShuffle_iter = [];
end
if cfg.returnDecisionValues
    decisionScoreBySplit = cell(nSplits, 1);
    decisionCountBySplit = cell(nSplits, 1);
else
    decisionScoreBySplit = {};
    decisionCountBySplit = {};
end

parfor (spliti = 1:nSplits, splitWorkers)
    trainIdx = trainIdxList{spliti};
    testIdx = testIdxList{spliti};
    trainY = allLabels(trainIdx);
    testY = allLabels(testIdx);
    testOriginalTrials = allOriginalTrials(testIdx);

    Acc_split = nan(nTime, nTime);
    trainAcc_split = nan(1, nTime);
    weights_split = nan(nCh, nTime);
    if cfg.useAUC, AUC_split = nan(nTime, nTime); else, AUC_split = []; end
    if cfg.returnDecisionValues
        decisionScore_split = nan(size(eegData, 3), nTime, nTime);
        decisionCount_split = zeros(size(eegData, 3), nTime, nTime);
    else
        decisionScore_split = [];
        decisionCount_split = [];
    end

    if cfg.doShuffle
        AccShuffle_split = nan(nTime, nTime);
        if cfg.useAUC, AUCShuffle_split = nan(nTime, nTime); else, AUCShuffle_split = []; end

        if ~isempty(cfg.randomSeed)
            rng(cfg.randomSeed + (sampi - 1) * 100000 + spliti, 'twister');
        end

        trainYShuffleByTime = cell(nTime, 1);
        for ti = 1:nTime
            trainYShuffleByTime{ti} = trainY(randperm(numel(trainY)));
        end
    else
        AccShuffle_split = [];
        AUCShuffle_split = [];
        trainYShuffleByTime = cell(nTime, 1);
    end

    for trainTime = 1:nTime
        accRow = nan(1, nTime);
        accShufRow = nan(1, nTime);
        if cfg.useAUC
            aucRow = nan(1, nTime);
            aucShufRow = nan(1, nTime);
        else
            aucRow = [];
            aucShufRow = [];
        end

        Xtrain = squeeze(allTrials(:, trainTime, trainIdx))';
        if isvector(Xtrain), Xtrain = reshape(Xtrain, sum(trainIdx), nCh); end

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
            [coeff, score, ~, ~, ~, mu_pca] = pca(Xtrain);
            coeff = coeff(:,1:maxPC);
            Xtrain = score(:,1:maxPC);
        end

        classNames = unique(trainY);
        ldaModel = [];
        ldaModelShuffle = [];
        if useClassify
            if cfg.useAUC
                [labelTrain, ~, ~] = classify(Xtrain, Xtrain, trainY, cfg.discrimType);
            else
                labelTrain = classify(Xtrain, Xtrain, trainY, cfg.discrimType);
            end
        else
            ldaModel = fitcdiscr(Xtrain, trainY, 'DiscrimType', cfg.discrimType);
            if cfg.doShuffle
                ldaModelShuffle = fitcdiscr(Xtrain, trainYShuffleByTime{trainTime}, 'DiscrimType', cfg.discrimType);
            end
            labelTrain = predict(ldaModel, Xtrain);
        end

        trainAccVal = mean(labelTrain == trainY);

        X1 = Xtrain(trainY == 1, :);
        X2 = Xtrain(trainY == 2, :);
        mu1 = mean(X1, 1, 'omitnan');
        mu2 = mean(X2, 1, 'omitnan');
        if strcmpi(cfg.discrimType, 'linear')
            S1 = cov(X1);
            S2 = cov(X2);
            n1 = size(X1,1);
            n2 = size(X2,1);
            Sp = ((n1-1)*S1 + (n2-1)*S2) / max(n1+n2-2, 1);
            w_use = pinv(Sp + eye(size(Sp))*eps) * (mu2 - mu1)';
        else
            v1 = var(X1, 0, 1, 'omitnan');
            v2 = var(X2, 0, 1, 'omitnan');
            n1 = size(X1,1);
            n2 = size(X2,1);
            vp = ((n1-1)*v1 + (n2-1)*v2) / max(n1+n2-2, 1);
            vp(vp <= eps | isnan(vp)) = eps;
            w_use = ((mu2 - mu1) ./ vp)';
        end
        if cfg.doPCA, w_use = coeff * w_use; end
        if cfg.standardize, w_use = w_use ./ sigma_z(:); end
        weightVec = w_use;

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

            if useClassify
                if cfg.useAUC || cfg.returnDecisionValues
                    [labelTest, ~, posterior] = classify(Xtest, Xtrain, trainY, cfg.discrimType);
                    aucVal = NaN;
                    posCol = find(classNames == 2, 1);
                    if ~isempty(posCol) && numel(unique(testY)) == 2
                        [~,~,~,aucVal] = perfcurve(testY, posterior(:,posCol), 2);
                    end
                    if cfg.useAUC
                        aucRow(testTime) = aucVal;
                    end
                    if cfg.returnDecisionValues && ~isempty(posCol)
                        decisionScore_split(testOriginalTrials, trainTime, testTime) = posterior(:,posCol);
                        decisionCount_split(testOriginalTrials, trainTime, testTime) = ~isnan(posterior(:,posCol));
                    end
                else
                    labelTest = classify(Xtest, Xtrain, trainY, cfg.discrimType);
                end
            else
                if cfg.useAUC || cfg.returnDecisionValues
                    [labelTest, score] = predict(ldaModel, Xtest);
                    aucVal = NaN;
                    posCol = find(ldaModel.ClassNames == 2, 1);
                    if ~isempty(posCol) && numel(unique(testY)) == 2
                        [~,~,~,aucVal] = perfcurve(testY, score(:,posCol), 2);
                    end
                    if cfg.useAUC
                        aucRow(testTime) = aucVal;
                    end
                    if cfg.returnDecisionValues && ~isempty(posCol)
                        decisionScore_split(testOriginalTrials, trainTime, testTime) = score(:,posCol);
                        decisionCount_split(testOriginalTrials, trainTime, testTime) = ~isnan(score(:,posCol));
                    end
                else
                    labelTest = predict(ldaModel, Xtest);
                end
            end

            accRow(testTime) = mean(labelTest == testY);

            if cfg.doShuffle
                if useClassify
                    if cfg.useAUC
                        [labelShuf, ~, posteriorShuf] = classify(Xtest, Xtrain, trainYShuffleByTime{trainTime}, cfg.discrimType);
                        aucVal = NaN;
                        posCol = find(classNames == 2, 1);
                        if ~isempty(posCol) && numel(unique(testY)) == 2
                            [~,~,~,aucVal] = perfcurve(testY, posteriorShuf(:,posCol), 2);
                        end
                        aucShufRow(testTime) = aucVal;
                    else
                        labelShuf = classify(Xtest, Xtrain, trainYShuffleByTime{trainTime}, cfg.discrimType);
                    end
                else
                    if cfg.useAUC
                        [labelShuf, scoreShuf] = predict(ldaModelShuffle, Xtest);
                        aucVal = NaN;
                        posCol = find(ldaModelShuffle.ClassNames == 2, 1);
                        if ~isempty(posCol) && numel(unique(testY)) == 2
                            [~,~,~,aucVal] = perfcurve(testY, scoreShuf(:,posCol), 2);
                        end
                        aucShufRow(testTime) = aucVal;
                    else
                        labelShuf = predict(ldaModelShuffle, Xtest);
                    end
                end

                accShufRow(testTime) = mean(labelShuf == testY);
            end
        end

        Acc_split(trainTime,:) = accRow;
        trainAcc_split(trainTime) = trainAccVal;
        weights_split(:,trainTime) = weightVec;
        if cfg.useAUC, AUC_split(trainTime,:) = aucRow; end
        if cfg.doShuffle
            AccShuffle_split(trainTime,:) = accShufRow;
            if cfg.useAUC, AUCShuffle_split(trainTime,:) = aucShufRow; end
        end
    end

    Acc_iter(:,:,spliti) = Acc_split;
    trainAcc_iter(spliti,:) = trainAcc_split;
    weights_iter(:,:,spliti) = weights_split;
    if cfg.useAUC, AUC_iter(:,:,spliti) = AUC_split; end
    if cfg.doShuffle
        AccShuffle_iter(:,:,spliti) = AccShuffle_split;
        if cfg.useAUC, AUCShuffle_iter(:,:,spliti) = AUCShuffle_split; end
    end
    if cfg.returnDecisionValues
        decisionScoreBySplit{spliti} = decisionScore_split;
        decisionCountBySplit{spliti} = decisionCount_split;
    end
end

if cfg.returnDecisionValues
    decisionScore_iter = nan(size(eegData, 3), nTime, nTime);
    decisionCount_iter = zeros(size(eegData, 3), nTime, nTime);
    for spliti = 1:nSplits
        scoreNow = decisionScoreBySplit{spliti};
        countNow = decisionCountBySplit{spliti};
        fillIdx = countNow > 0;
        decisionScore_iter(fillIdx) = scoreNow(fillIdx);
        decisionCount_iter = decisionCount_iter + countNow;
    end
else
    decisionScore_iter = [];
    decisionCount_iter = [];
end

end

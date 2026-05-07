function result = SVM_function_confusion_singleSubj(eegData, labels, times, cfg)
% SVM_function_confusion_singleSubj
%
% Decoding pipeline for single-subject EEG/iEEG data.
%
% INPUT
%   eegData : nCh x nTime x nTrials
%   labels  : nTrials x 1
%   times   : 1 x nTime or nTime x 1, same unit as smooth_window/smooth_step
%   cfg     : struct
%
% REQUIRED / COMMON CFG FIELDS
%   cfg.nFolds                : number of CV folds, default = 10
%   cfg.superTrial            : number of trials averaged into one supertrial, default = 1
%   cfg.nIter                 : number of random resampling runs, default = 1
%   cfg.doPCA                 : true/false, default = false
%   cfg.nPCs                  : number of PCs if doPCA = true, default = 5
%   cfg.smooth_window         : temporal smoothing window size in same unit as times, default = 0
%                               If 0, no temporal averaging is applied.
%   cfg.smooth_step           : step size between adjacent output time points, same unit as times, default = []
%                               If [], every valid time point is used as one output center.
%                               Example:
%                                   smooth_window = 500, smooth_step = 50
%                               means 500-ms sliding window with 50-ms step.
%                               If smooth_window = 0 and smooth_step > 0,
%                               the function only downsamples time points without averaging.
%   cfg.doTimeGeneralization  : true/false, default = true
%   cfg.kernelFunction        : SVM kernel, recommended = 'linear', default = 'linear'
%   cfg.standardize           : true/false, default = true
%   cfg.verbose               : true/false, default = true
%   cfg.randomSeed            : numeric scalar or [], default = []
%
% OUTPUT
%   result.predictAcc         : nTimeOut x nTimeOut
%   result.AUC                : nTimeOut x nTimeOut
%   result.predictAccTrain    : nTimeOut x 1
%   result.weights            : nCh x nTimeOut
%   result.times              : processed time vector
%   result.cfg                : full config used
%   result.classLabelsOriginal: original class labels
%
% NOTES
%   - Output matrices are always nTimeOut x nTimeOut.
%   - If doTimeGeneralization = false, only the diagonal is filled;
%     off-diagonal entries are NaN.
%   - PCA is fitted on training data only within each fold/time point.
%   - smooth_window and smooth_step must use the same unit as times.
%     For example, if times is in ms, use 500 and 50.
%     If times is in seconds, use 0.5 and 0.05.

%% =========================
% 0. Input check
% ==========================
if nargin < 4
    cfg = struct();
end

validateattributes(eegData, {'numeric'}, {'nonempty', '3d'}, mfilename, 'eegData', 1);
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
    [eegData, times] = apply_temporal_smoothing( ...
        eegData, times, cfg.smooth_window, cfg.smooth_step);
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

    % ----- Supertrials -----
    if cfg.superTrial > 1
        dataSup1 = make_supertrials(data1, cfg.superTrial);
        dataSup2 = make_supertrials(data2, cfg.superTrial);
    else
        dataSup1 = data1;
        dataSup2 = data2;
    end

    allTrials = cat(3, dataSup1, dataSup2);
    allLabels = [ones(size(dataSup1,3),1); 2*ones(size(dataSup2,3),1)];

    % Stratified CV
    CVO = cvpartition(allLabels, 'KFold', cfg.nFolds);

    for foldi = 1:cfg.nFolds
        trainIdx = training(CVO, foldi);
        testIdx  = test(CVO, foldi);

        trainY = allLabels(trainIdx);
        testY  = allLabels(testIdx);

        for trainTime = 1:nTime

            Xtrain = squeeze(allTrials(:,trainTime,trainIdx))';  % nTrain x nCh
            if isvector(Xtrain)
                Xtrain = reshape(Xtrain, sum(trainIdx), nCh);
            end

            % ----- PCA on training data only -----
            coeff = [];
            mu    = [];

            if cfg.doPCA
                [coeff, Xtrain, mu] = fit_pca_train_only(Xtrain, cfg.nPCs);
            end

            % ----- Train SVM -----
            svmModel = fitcsvm( ...
                Xtrain, trainY, ...
                'KernelFunction', cfg.kernelFunction, ...
                'Standardize',    cfg.standardize, ...
                'KernelScale',    'auto');

            % ----- Save weights if linear -----
            if strcmpi(cfg.kernelFunction, 'linear') && ...
                    isprop(svmModel, 'Beta') && ~isempty(svmModel.Beta)

                if cfg.doPCA
                    weights_all(:,trainTime,foldi,sampi) = coeff * svmModel.Beta;
                else
                    weights_all(:,trainTime,foldi,sampi) = svmModel.Beta;
                end
            end

            % ----- Training accuracy -----
            labelTrain = predict(svmModel, Xtrain);
            trainAcc_all(foldi,trainTime,sampi) = mean(labelTrain == trainY);

            % ----- Test -----
            if cfg.doTimeGeneralization
                testTimes = 1:nTime;
            else
                testTimes = trainTime;
            end

            for testTime = testTimes
                Xtest = squeeze(allTrials(:,testTime,testIdx))';  % nTest x nCh
                if isvector(Xtest)
                    Xtest = reshape(Xtest, sum(testIdx), nCh);
                end

                if cfg.doPCA
                    Xtest = apply_pca_to_test(Xtest, coeff, mu);
                end

                [labelTest, score] = predict(svmModel, Xtest);
                predictAcc_all(trainTime,testTime,foldi,sampi) = mean(labelTest == testY);

                % ----- AUC -----
                if size(score,2) >= 2
                    trueLabels = (testY == 2);
                    scorePos   = score(:,2);

                    if numel(unique(trueLabels)) == 2
                        [~,~,~,aucVal] = perfcurve(trueLabels, scorePos, 1);
                    else
                        aucVal = NaN;
                    end
                else
                    aucVal = NaN;
                end

                AUC_all(trainTime,testTime,foldi,sampi) = aucVal;
            end

            if cfg.verbose
                fprintf('.');
            end
        end
    end

    if cfg.verbose
        fprintf('  sample %d/%d done\n', sampi, cfg.nIter);
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
result.predictAcc           = predictAcc;
result.AUC                  = AUC;
result.predictAccTrain      = trainAcc(:);
result.weights              = weights;
result.times                = times(:);
result.cfg                  = cfg;
result.classLabelsOriginal  = uLabels;

end

%% ========================================================================
function cfg = fill_default_cfg(cfg)

if ~isfield(cfg,'nFolds'),               cfg.nFolds = 10; end
if ~isfield(cfg,'superTrial'),           cfg.superTrial = 1; end
if ~isfield(cfg,'nIter'),                cfg.nIter = 1; end
if ~isfield(cfg,'doPCA'),                cfg.doPCA = false; end
if ~isfield(cfg,'nPCs'),                 cfg.nPCs = 5; end
if ~isfield(cfg,'smooth_window'),        cfg.smooth_window = 0; end
if ~isfield(cfg,'smooth_step'),          cfg.smooth_step = []; end
if ~isfield(cfg,'doTimeGeneralization'), cfg.doTimeGeneralization = true; end
if ~isfield(cfg,'kernelFunction'),       cfg.kernelFunction = 'linear'; end
if ~isfield(cfg,'standardize'),          cfg.standardize = true; end
if ~isfield(cfg,'verbose'),              cfg.verbose = true; end
if ~isfield(cfg,'randomSeed'),           cfg.randomSeed = []; end

validateattributes(cfg.nFolds,      {'numeric'}, {'scalar','integer','>=',2});
validateattributes(cfg.superTrial,  {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.nIter,       {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.doPCA,       {'numeric','logical'}, {'scalar'});
validateattributes(cfg.nPCs,        {'numeric'}, {'scalar','integer','>=',1});
validateattributes(cfg.smooth_window, {'numeric'}, {'scalar','>=',0});
validateattributes(cfg.doTimeGeneralization, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.standardize, {'numeric','logical'}, {'scalar'});
validateattributes(cfg.verbose,     {'numeric','logical'}, {'scalar'});

if ~isempty(cfg.smooth_step)
    validateattributes(cfg.smooth_step, {'numeric'}, {'scalar','>',0});
end

if ~ischar(cfg.kernelFunction) && ~isstring(cfg.kernelFunction)
    error('cfg.kernelFunction must be a char or string.');
end

end

%% ========================================================================
function [eegData_out, times_out] = apply_temporal_smoothing(eegData, times, smooth_window, smooth_step)
% Apply temporal smoothing with optional step size.
%
% INPUT
%   eegData       : nCh x nTime x nTrials
%   times         : 1 x nTime
%   smooth_window : window size, same unit as times
%                   If 0, no averaging is applied.
%   smooth_step   : step size between adjacent output time points
%                   If [], every valid time point is used as output center.
%
% OUTPUT
%   eegData_out   : nCh x nOutputTime x nTrials
%   times_out     : output time vector corresponding to window centers

times = times(:)';

if nargin < 4
    smooth_step = [];
end

validateattributes(smooth_window, {'numeric'}, {'scalar','>=',0});

if ~isempty(smooth_step)
    validateattributes(smooth_step, {'numeric'}, {'scalar','>',0});
end

% If no smoothing and no step-size downsampling, return original data.
if smooth_window == 0 && isempty(smooth_step)
    eegData_out = eegData;
    times_out   = times;
    return;
end

% -------------------------------------------------------------------------
% 1. Define valid center time points
% -------------------------------------------------------------------------
if smooth_window > 0
    half_window = smooth_window / 2;

    center_min = times(1)   + half_window;
    center_max = times(end) - half_window;

    if center_min > center_max
        error('No valid time points remain. smooth_window is too large for the current time range.');
    end

    validCenterIdx = find(times >= center_min & times <= center_max);
else
    % No averaging, only temporal downsampling.
    validCenterIdx = 1:numel(times);
end

if isempty(validCenterIdx)
    error('No valid output time points were found.');
end

% -------------------------------------------------------------------------
% 2. Apply optional step size
% -------------------------------------------------------------------------
if isempty(smooth_step)
    % Old behavior: every valid time point is used as one output center.
    centerIdx = validCenterIdx;
else
    % New behavior: use centers separated approximately by smooth_step.
    targetTimes = times(validCenterIdx(1)) : smooth_step : times(validCenterIdx(end));

    if isempty(targetTimes)
        targetTimes = times(validCenterIdx(1));
    end

    centerIdx = nan(size(targetTimes));

    for ii = 1:numel(targetTimes)
        [~, nearestPos] = min(abs(times(validCenterIdx) - targetTimes(ii)));
        centerIdx(ii) = validCenterIdx(nearestPos);
    end

    % Remove duplicated indices caused by smooth_step smaller than sampling interval.
    centerIdx = unique(centerIdx, 'stable');
end

% -------------------------------------------------------------------------
% 3. Generate output data
% -------------------------------------------------------------------------
nCh     = size(eegData, 1);
nTrials = size(eegData, 3);
nOut    = numel(centerIdx);

eegData_out = zeros(nCh, nOut, nTrials, 'like', eegData);

if smooth_window > 0
    half_window = smooth_window / 2;

    for ii = 1:nOut
        tIdx = centerIdx(ii);
        centerTime = times(tIdx);

        winIdx = times >= (centerTime - half_window) & ...
                 times <= (centerTime + half_window);

        eegData_out(:, ii, :) = mean(eegData(:, winIdx, :), 2);
    end
else
    % No window averaging, only select stepped time points.
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
%   dataSup    : nCh x nTime x nSuperTrials

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
function [coeff, Xtrain_pca, mu] = fit_pca_train_only(Xtrain, nPCs)

[nObs, nFeat] = size(Xtrain);
maxPC = min([nObs-1, nFeat, nPCs]);

if maxPC < 1
    error('Not enough training observations/features for PCA.');
end

[coeff, score, ~, ~, ~, mu] = pca(Xtrain);
coeff     = coeff(:,1:maxPC);
Xtrain_pca = score(:,1:maxPC);

end

%% ========================================================================
function Xtest_pca = apply_pca_to_test(Xtest, coeff, mu)

Xtest_pca = (Xtest - mu) * coeff;

end

%% ========================================================================
function M = keep_diagonal_only(M)

diagVals = diag(M);
M(:) = NaN;
M(1:size(M,1)+1:end) = diagVals;

end

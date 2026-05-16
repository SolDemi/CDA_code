% classify_load_CDA_diagLinear_iterations.m
%
% Decode working-memory load using CDA-like contralateral-minus-ipsilateral
% voltage from paired posterior electrodes.
%
% This script is adapted from classify_load_linearDiag_iterations.m.
% Main changes:
%   1. Use data_raw instead of grand/CDA_data.
%   2. Compute CDA features trial-by-trial: contralateral - ipsilateral.
%   3. Use paired electrodes:
%        leftElecLabels  = {'O1','OL','P3','PO3','T5'};
%        rightElecLabels = {'O2','OR','P4','PO4','T6'};
%   4. Exclude subjects inside the loop if any of the four condition
%      pairs C2/C6/S2/S6 has fewer than minTrialsPerCond trials after
%      combining left and right trials.
%
% Expected per-subject data:
%   Each subject should have erp_singletrial.mat containing:
%     erp.allChans
%     erp.trial.L_C2, erp.trial.L_C6, erp.trial.L_S2, erp.trial.L_S6
%     erp.trial.R_C2, erp.trial.R_C6, erp.trial.R_S2, erp.trial.R_S6
%     cda.time or erp.time
%
% Trial data are assumed to be nTrials x nChannels x nTimes.

clear; clc;
dbstop if error;

%% paths
path_subj = [erase(pwd,'code') 'data_raw'];
outputdir = [erase(pwd,'code') 'CDA_classification\'];
if ~isfolder(outputdir)
    mkdir(outputdir);
end

%% config
binSize = 50;
nIter = 100;
minTrialsPerCond = 160;

leftElecLabels  = {'O1','OL','P3','PO3','T5'};
rightElecLabels = {'O2','OR','P4','PO4','T6'};

% Four logical conditions for trial-count exclusion.
% Each row: output condition name, left-trial field, right-trial field.
condPairs = {'C2','L_C2','R_C2'; ...
             'C6','L_C6','R_C6'; ...
             'S2','L_S2','R_S2'; ...
             'S6','L_S6','R_S6'};
condPairNames = condPairs(:,1)';
condLoads = [2, 6, 2, 6];

%% find subject files/folders
items = dir(path_subj);
items = items(~ismember({items.name}, {'.','..'}));

cls = struct();
cls.subjects = {};
cls.excluded_subjects = {};
cls.exclude_reason = {};
cls.nIter = nIter;
cls.binSize = binSize;
cls.minTrialsPerCond = minTrialsPerCond;
cls.leftElecLabels = leftElecLabels;
cls.rightElecLabels = rightElecLabels;
cls.condPairs = condPairs;
cls.condPairNames = condPairNames;
cls.condLoads = condLoads;
cls.featureType = 'CDA contra-minus-ipsi paired electrodes';

nValid = 0;

%% loop through subjects
for s = 1:numel(items)

    subjName = items(s).name;

    if items(s).isdir
        dataFile = fullfile(path_subj, subjName, 'erp_singletrial.mat');
    else
        [~,~,ext] = fileparts(subjName);
        if ~strcmpi(ext, '.mat')
            continue
        end
        dataFile = fullfile(path_subj, subjName);
    end

    if ~isfile(dataFile)
        cls.excluded_subjects{end+1,1} = subjName;
        cls.exclude_reason{end+1,1} = 'erp_singletrial.mat not found';
        fprintf('Skip %s: erp_singletrial.mat not found\n', subjName);
        continue
    end

    load(dataFile);  % should load erp and possibly cda

    if exist('cda','var') && isfield(cda,'time')
        tPts = cda.time;
    elseif isfield(erp,'time')
        tPts = erp.time;
    else
        error('No cda.time or erp.time found in %s', dataFile);
    end

    [tfLeft, leftIdx] = ismember(leftElecLabels, erp.allChans);
    [tfRight, rightIdx] = ismember(rightElecLabels, erp.allChans);
    if any(~tfLeft) || any(~tfRight)
        cls.excluded_subjects{end+1,1} = subjName;
        cls.exclude_reason{end+1,1} = 'missing CDA electrodes';
        fprintf('Skip %s: missing CDA electrodes\n', subjName);
        continue
    end

    % Trial-count exclusion is based on four condition pairs, not eight
    % left/right-specific fields. For example, C2 = L_C2 + R_C2.
    nTrialsPerCond = nan(1,size(condPairs,1));
    for ci = 1:size(condPairs,1)
        leftField  = condPairs{ci,2};
        rightField = condPairs{ci,3};
        nTrialsPerCond(ci) = size(erp.trial.(leftField), 1) + size(erp.trial.(rightField), 1);
    end

    if min(nTrialsPerCond) < minTrialsPerCond
        cls.excluded_subjects{end+1,1} = subjName;
        cls.exclude_reason{end+1,1} = sprintf('min trials per condition pair = %d', min(nTrialsPerCond));
        fprintf('Skip %s: min trials per condition pair = %d\n', subjName, min(nTrialsPerCond));
        continue
    end

    if nValid == 0
        tBins = min(tPts):binSize:max(tPts);
        nBins = numel(tBins) - 1;
        cls.time = tPts;
        cls.bins = tBins;
        cls.binCenters = tBins(1:end-1) + binSize/2;
        cls.acc_load_cda = [];
        cls.acc_load_cda_shuffle = [];
        cls.nTrials = [];
        cls.nTrialsPerCond = [];
    end

    %% construct CDA data: nTrials x nPairedElectrodes x nTimes
    tempDat = [];
    labels = [];

    for ci = 1:size(condPairs,1)
        leftField  = condPairs{ci,2};
        rightField = condPairs{ci,3};

        leftDat  = erp.trial.(leftField);
        rightDat = erp.trial.(rightField);

        % remember/attend left: right hemisphere is contralateral
        leftCDA = leftDat(:, rightIdx, :) - leftDat(:, leftIdx, :);

        % remember/attend right: left hemisphere is contralateral
        rightCDA = rightDat(:, leftIdx, :) - rightDat(:, rightIdx, :);

        thisCDA = cat(1, leftCDA, rightCDA);
        tempDat = cat(1, tempDat, thisCDA);
        labels = [labels; ones(size(thisCDA,1),1) * condLoads(ci)];
    end

    nValid = nValid + 1;
    cls.subjects{nValid,1} = subjName;
    cls.nTrials(nValid,1) = numel(labels);
    cls.nTrialsPerCond(nValid,:) = nTrialsPerCond;

    setSizes = unique(labels);
    nT_ss = nan(1,numel(setSizes));
    for ss = 1:numel(setSizes)
        nT_ss(ss) = sum(labels == setSizes(ss));
    end
    minT = min(nT_ss);

    cls_acc_load_cda = nan(nIter, nBins);
    cls_acc_load_cda_shuffle = nan(nIter, nBins);

    fprintf('Now processing %s (%d valid subjects)\n', subjName, nValid);
    tic

    for b = 1:nBins

        if b < nBins
            tMask = tPts >= tBins(b) & tPts < tBins(b+1);
        else
            tMask = tPts >= tBins(b) & tPts <= tBins(b+1);
        end

        % Average CDA voltage in current 50-ms bin.
        % Result: nTrials x nPairedElectrodes
        tDat = squeeze(nanmean(tempDat(:,:,tMask), 3));

        for it = 1:nIter

            % Randomly balance load 2 and load 6 trials.
            ss2Ind = find(labels == 2);
            ss6Ind = find(labels == 6);
            ss2Ind = ss2Ind(randperm(numel(ss2Ind)));
            ss6Ind = ss6Ind(randperm(numel(ss6Ind)));
            ss2Ind = ss2Ind(1:minT);
            ss6Ind = ss6Ind(1:minT);

            % Match original script: random 2/3 train, 1/3 test.
            cutoffs = round(linspace(1, minT, 4));

            trnIdx = [ss2Ind(1:cutoffs(3)); ss6Ind(1:cutoffs(3))];
            tstIdx = [ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)];

            trnDat = tDat(trnIdx,:);
            tstDat = tDat(tstIdx,:);
            trnLabels = labels(trnIdx);
            tstLabels = labels(tstIdx);

            % Intact-label classifier.
            classOutput = classify(tstDat, trnDat, trnLabels, 'diagLinear');
            cls_acc_load_cda(it,b) = mean(classOutput == tstLabels);

            % Shuffled-training-label empirical chance.
            trnLabelsShuffle = trnLabels(randperm(numel(trnLabels)));
            classOutputShuffle = classify(tstDat, trnDat, trnLabelsShuffle, 'diagLinear');
            cls_acc_load_cda_shuffle(it,b) = mean(classOutputShuffle == tstLabels);

        end
    end

    cls.acc_load_cda(nValid,:) = nanmean(cls_acc_load_cda, 1);
    cls.acc_load_cda_shuffle(nValid,:) = nanmean(cls_acc_load_cda_shuffle, 1);

    toc
    fprintf('Subject %s finished\n\n', subjName);
end

if nValid == 0
    error('No valid subjects were found. Check path_subj and minTrialsPerCond.');
end

save(fullfile(outputdir, 'load_classify_CDA_diagLinear_iterations.mat'), 'cls');
fprintf('Saved results to: %s\n', fullfile(outputdir, 'load_classify_CDA_diagLinear_iterations.mat'));


%% plot
xlim_plot = [-200 1000];
ylim_plot = [0.49 0.53];
xlabel_p  = 'Times';
ylabel_p  = 'ACC';
myColor1 = [92, 181, 152]./255;
myColor2 = [249, 107, 101]./255;
plot_shaded_errorbar_twoCurve(cls.binCenters,cls.acc_load_cda,cls.acc_load_cda_shuffle,xlim_plot,ylim_plot,xlabel_p,ylabel_p,'True ACC','Shuffle',myColor1,myColor2)
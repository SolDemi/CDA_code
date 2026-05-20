clear,clc
maindir = erase(pwd,'code');
datadir = [maindir 'cda_alpha\'];
outputdir = [maindir 'decoding_SVM\'];
% create folders
for i = 1:4
    switch i
        case 1, tmp_name = 'CDA';
        case 2, tmp_name = 'Alpha';
        case 3, tmp_name = 'NoPCA';
        case 4, tmp_name = 'PCA';
    end
    if ~isfolder( [outputdir tmp_name] )
        mkdir( [outputdir tmp_name] )
    end
end
 
%% config
cfg = struct();
cfg.cvType = 'holdout';        % 'holdout' reproduces 2/3 train, 1/3 test style
cfg.trainRatio = 2/3;
cfg.nFolds = 3;                % only used when cfg.cvType = 'kfold'
cfg.superTrial = 10;
cfg.nIter = 100;

cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';    % article-style 50-ms bins

cfg.doTimeGeneralization = 1;
cfg.doPCA = false;
cfg.nPCs = 5;
cfg.discrimType = 'Linear';
cfg.standardize = 1;

cfg.doShuffle = true;          % shuffled TRAINING labels empirical chance
cfg.balanceTrials = true;      % balance classes each iteration
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];
cfg.useAUC = 1;
cfg.useParallel = true;
cfg.verbose = 0;
cfg.randomSeed = [];
CDA = struct();
%%
files = dir([datadir 'sub*']);
for s = numel(files)-1:-1:1
    result = struct();
    file = files(s).name;
    fprintf("Now Processing: %s\n", file)
    load([datadir file]) % this file contains two structs: cda & alpha

    if min(cda.trials_per_cond) < 160
        continue
    end

    % perpare labels and data for SVM
    labels = [ones(size(cda.trial.diff_2,1),1)*2; ones(size(cda.trial.diff_6,1),1)*6];

    data1  = cat( 1, cda.trial.diff_2, cda.trial.diff_6 ); % trials x channels x time
    data1  = permute(data1, [2,3,1]);
    % decode load based on CDA
    CDA = SVM_function_singleSubj(data1,labels, cda.time,cfg);
    % CDA.ACC = subj_CDA.predictAcc;

    save([outputdir 'CDA\' file], "CDA")
    
    % % decode load based on alpha band
    data2 = cat( 1, alpha.trial.diff_2, alpha.trial.diff_6 ); % trials x channels x time
    data2  = permute(data2, [2,3,1]);
    Alpha = SVM_function_singleSubj(data2,labels, cda.time,cfg);

    save([outputdir 'Alpha\' file], "Alpha")
    % 
    %  % decode load based on CDA & alpha
    data3 = cat(1, data1, data2);
    NoPCA = SVM_function_singleSubj(data3,labels, cda.time,cfg);


     save([outputdir 'NoPCA\' file], "NoPCA")
    % 
    % % PCA before decoding 
    cfg.doPCA = 1;
    PCA = SVM_function_singleSubj(data3,labels, cda.time,cfg);

    save([outputdir 'PCA\' file], "PCA")

    % fprintf('Subject %s finished\n\n', subjName);
end

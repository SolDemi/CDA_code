clear,clc
maindir = erase(pwd,'code');
datadir = [maindir 'cda_alpha\'];
outputdir = [maindir 'decoding_LDA\'];
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


cfg.nFolds = 10;
cfg.avgNTrials = 10;
cfg.nIter = 1;
cfg.binSize = 50;
cfg.binUnit = 'ms';
cfg.doPCA = false;
cfg.zscore = false;
cfg.discrimType = 'diagLinear';
time = -200:4:996;

files = dir([datadir 'sub*']);
for s = numel(files):-1:1
    result = struct();
    file = files(s).name;
    fprintf("Now Processing: %s\n", file)
    load([datadir file]) % this file contains two structs: cda & alpha

    if min(cda.trials_per_cond) < 160
        continue
    end

    % perpare labels and data for SVM
    labels = [ones(size(cda.trial.diff_2,1),1); ones(size(cda.trial.diff_6,1),1)*2];

    data1  = cat( 1, cda.trial.diff_2(:,:,201:end), cda.trial.diff_6(:,:,201:end) ); % trials x channels x time
    data1  = permute(data1, [2,3,1]);
    % decode load based on CDA
    [Acc4TrainSet, predictAcc, weights, AUC] = LDA_function_singleSubj(data1, labels, time, cfg);
    CDA.trainACC = Acc4TrainSet;
    CDA.testACC = predictAcc;
    CDA.weights = weights;
    CDA.AUC = AUC;
    save([outputdir 'CDA\' file '_10SuperTrials.mat'], "CDA")

    % % decode load based on alpha band
    % data2 = cat( 1, alpha.trial.diff_2(:,:,201:end), alpha.trial.diff_6(:,:,201:end) ); % trials x channels x time
    % data2  = permute(data2, [2,3,1]);
    % [Acc4TrainSet, predictAcc, weights, AUC] = LDA_function_singleSubj(data2, labels, time, cfg);
    % 
    % Alpha.trainACC = Acc4TrainSet;
    % Alpha.testACC = predictAcc;
    % Alpha.weights = weights;
    % Alpha.AUC = AUC;
    % save([outputdir 'Alpha\' file], "Alpha")
    % 
    % % decode load based on CDA & alpha
    % data3 = cat(1, data1, data2);
    % [Acc4TrainSet, predictAcc, weights, AUC] = LDA_function_singleSubj(data3, labels, cfg);
    % 
    % NoPCA.trainACC = Acc4TrainSet;
    % NoPCA.testACC = predictAcc;
    % NoPCA.weights = weights;
    % NoPCA.AUC = AUC;
    % save([outputdir 'NoPCA\' file], "NoPCA")
    % 
    % % PCA before decoding
    % cfg.doPCA = true;
    % cfg.nPCs  = 5;
    % [Acc4TrainSet, predictAcc, weights, AUC] = LDA_function_singleSubj(data3, labels, cfg);
    % 
    % PCA.trainACC = Acc4TrainSet;
    % PCA.testACC = predictAcc;
    % PCA.weights = weights;
    % PCA.AUC = AUC;
    % save([outputdir 'PCA\' file], "PCA")


end

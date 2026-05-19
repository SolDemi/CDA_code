%% 1) LDA or SVM decoding: AUC above chance
loadCfg = struct();
loadCfg.metric = 'AUCMinusShuffle';
loadCfg.useDiagonal = true;      % use diag(AUC) from train-time x test-time matrix
% loadCfg.resultVarName = 'CDA'; % optional, if every .mat saves the variable as CDA
loadPath = fullfile( erase(pwd,'code'), 'data0', 'decoding_SVM/', 'Alpha/');
[groupAUC, times, files] = extract_decoding_timeseries(loadPath, loadCfg); 

statCfg = struct();
statCfg.null = 0;
statCfg.nPerm = 1000;
statCfg.tail = 'right';          % above chance
statCfg.clusterAlpha = 0.05;
statCfg.alpha = 0.05;
statCfg.randomSeed = 1;
statCfg.ylabel = 'AUC';
statCfg.title = 'Group-level decoding AUC';
statCfg.eventLines = [0 150];
statCfg.eventLineLabels = {'Stimulus onset', 'Stimulus offset'};

ldaStats = plot_group_timeseries_perm(groupAUC, times, statCfg);

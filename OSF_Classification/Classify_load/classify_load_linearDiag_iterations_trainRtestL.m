% KA : Add iterations to the single-trial analysis to be more consistent
% with the mini-block analysis!!! Also, be sure to balance # of trials per
% set size while we're at it!!

% Attempt clsifying left versus right over time.
dbstop if error;

load grand_cda_alltrials.mat

% only use subjects with a certain # of trials!
subjects = grand.subjects(grand.min_trials_per_cond>=160);

% Load example file
load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(1))),filesep,'erp_singletrial.mat'])

tPts = cda.time;
binSize = 50;
tBins = [min(tPts):binSize:max(tPts)];

cls.subjects = subjects;
cls.time = tPts;
cls.bins = tBins;
cls.binCenters = [min(tPts)+round(binSize./2):binSize:max(tPts)-binSize];
cls.nIter = 100; nIter = cls.nIter; % number of iterations for analysis
cls.chans = erp.allChans(1:20);

nSubs = length(subjects);
nBins = length(tBins) - 1;

cls.acc_load_allchans =  NaN(nSubs,nBins);
cls.acc_load_allchans_shuffle =  NaN(nSubs,nBins);
cls.nTrials = NaN(nSubs,1);

% Loop through subjects
for s = 1:nSubs
    %% Load subject
    load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(s))),filesep,'erp_singletrial.mat'])
    %%
    % Average for this timepoint, create a matrix and left/ right data
    tempDat_trn = [erp.trial.R_C2; erp.trial.R_C6; erp.trial.R_S2; erp.trial.R_S6; ];
    
    labels_trn = [ones(size(erp.trial.R_C2,1),1)*2; ones(size(erp.trial.R_C6,1),1)*6; ...
        ones(size(erp.trial.R_S2,1),1)*2; ones(size(erp.trial.R_S6,1),1)*6; ];
    
    tempDat_tst = [erp.trial.L_C2; erp.trial.L_C6; erp.trial.L_S2; erp.trial.L_S6];
    
    labels_tst = [ones(size(erp.trial.L_C2,1),1)*2; ones(size(erp.trial.L_C6,1),1)*6; ...
        ones(size(erp.trial.L_S2,1),1)*2; ones(size(erp.trial.L_S6,1),1)*6;];
    
    cls.nTrials_trn(s) = length(labels_trn);
    cls.nTrials_tst(s) = length(labels_tst);

    setSizes = unique(labels_trn); nSS = length(setSizes);
    
    % Set up a matrix / info for balancing trials in the training and test
    % sets.
    nT_ss_trn = NaN(1,nSS);
    nT_ss_tst = NaN(1,nSS);
    for ss = 1:nSS
        nT_ss_trn(ss) = sum(labels_trn==setSizes(ss));
        nT_ss_tst(ss) = sum(labels_tst==setSizes(ss));
    end
    minT_trn = min(nT_ss_trn);
    minT_tst = min(nT_ss_tst);

    cls_acc_load_allchans = NaN(nIter,nBins);
    cls_acc_load_allchans_shuffle = NaN(nIter,nBins);
    
    tic
    % Loop through timepoints
    for b = 1:nBins
        for it = 1:nIter
            %------------------------------------------------------------------
            % All electrodes
            %------------------------------------------------------------------
            % Data for this time point (exclude EOG data)
            
            % randomly trim excess trials from set sizes with more than the minimum.
            ss2Ind_trn = find(labels_trn==2); ss2Ind_trn = ShuffleKA(ss2Ind_trn); ss2Ind_trn = ss2Ind_trn(1:minT_trn);
            ss6Ind_trn = find(labels_trn==6); ss6Ind_trn = ShuffleKA(ss6Ind_trn); ss6Ind_trn = ss6Ind_trn(1:minT_trn);
            ss2Ind_tst = find(labels_tst==2); ss2Ind_tst = ShuffleKA(ss2Ind_tst); ss2Ind_tst = ss2Ind_tst(1:minT_tst);
            ss6Ind_tst = find(labels_tst==6); ss6Ind_tst = ShuffleKA(ss6Ind_tst); ss6Ind_tst = ss6Ind_tst(1:minT_tst);
            
            % choose balanced cutoffs for each
            cutoffs_trn = round(linspace(1,length(ss2Ind_trn),4));
            cutoffs_tst = round(linspace(1,length(ss2Ind_tst),4));

            % index the data and the labels by this random balanced index!
            tDat_trn = squeeze(nanmean(tempDat_trn(:,1:20,ismember(tPts,tBins(b):tBins(b+1))),3));
            tDat_tst = squeeze(nanmean(tempDat_tst(:,1:20,ismember(tPts,tBins(b):tBins(b+1))),3));

            % Assign the labels
            trnDat = tDat_trn([ss2Ind_trn(1:cutoffs_trn(3));ss6Ind_trn(1:cutoffs_trn(3));],:);
            trnLabels = labels_trn([ss2Ind_trn(1:cutoffs_trn(3));ss6Ind_trn(1:cutoffs_trn(3));]);
            
            tstDat = tDat_tst([ss2Ind_tst((cutoffs_tst(3)+1):end); ss6Ind_tst((cutoffs_tst(3)+1):end)],:);
            tstLabels = labels_tst([ss2Ind_tst((cutoffs_tst(3)+1):end); ss6Ind_tst((cutoffs_tst(3)+1):end)]);
            
            % Run the classifier
            [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
            
            % Calculate error rate!
            errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
            
            % Save accuracy!
            cls_acc_load_allchans(it,b) = 1 -  errRate;
            
            %------------------------------------------------------------------
            % All electrodes - SHUFFLE LABELS
            %------------------------------------------------------------------
            % Shuffle the training labels (keep training and test data all
            % the same)
            trnLabels = Shuffle(trnLabels);
            % Run the classifier
            [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
            
            % Calculate error rate!
            errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
            
            % Save accuracy!
            cls_acc_load_allchans_shuffle(it,b) = 1 - errRate;
            
            
        end % end iterations loop!
    end % end time bins loop!
    fprintf('.');
    % display time for this subject
    toc 
    
    cls.acc_load_allchans(s,:) = nanmean(cls_acc_load_allchans,1); % Average across iterations
    cls.acc_load_allchans_shuffle(s,:) = nanmean(cls_acc_load_allchans_shuffle,1);
    
    fprintf(sprintf('\n Subject %d out of %d finished \n',s,nSubs));
end % end subject loop

save([pwd,'/CDA_classification/load_classify_diagLinear_trainRtestL.mat'],'cls')



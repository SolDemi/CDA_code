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

nSubs = length(subjects);
nBins = length(tBins) - 1;

cls.acc_load_allchans =  NaN(nSubs,nBins,nBins);
cls.acc_load_allchans_shuffle =  NaN(nSubs,nBins,nBins);

% Loop through subjects
for s = 1:nSubs
    %% Load subject
    load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(s))),filesep,'erp_singletrial.mat'])
    %%
    % Average for this timepoint, create a matrix and left/ right data
    tempDat = [erp.trial.L_C2; erp.trial.L_C6; erp.trial.L_S2; erp.trial.L_S6; ...
        erp.trial.R_C2; erp.trial.R_C6; erp.trial.R_S2; erp.trial.R_S6; ];
    
    labels = [ones(size(erp.trial.L_C2,1),1)*2; ones(size(erp.trial.L_C6,1),1)*6; ...
        ones(size(erp.trial.L_S2,1),1)*2; ones(size(erp.trial.L_S6,1),1)*6; ...
        ones(size(erp.trial.R_C2,1),1)*2; ones(size(erp.trial.R_C6,1),1)*6; ...
        ones(size(erp.trial.R_S2,1),1)*2; ones(size(erp.trial.R_S6,1),1)*6; ];
    
    setSizes = unique(labels); nSS = length(setSizes);
    
    % Set up a matrix / info for balancing trials in the training and test
    % sets.
    nT_ss = NaN(1,nSS);
    for ss = 1:nSS
        nT_ss(ss) = sum(labels==setSizes(ss));
    end
    minT = min(nT_ss);
    
    cls_acc_load_allchans = NaN(nIter,nBins,nBins);
    cls_acc_load_allchans_shuffle = NaN(nIter,nBins,nBins);
    
    tic
    % Loop through timepoints
    for b1 = 1:nBins
        for b2 = 1:nBins
            for it = 1:nIter
                %------------------------------------------------------------------
                % All electrodes
                %------------------------------------------------------------------
                % Data for this time point (exclude EOG data)
                
                % randomly trim excess trials from set sizes with more than the minimum.
                ss2Ind = find(labels==2); ss2Ind = Shuffle(ss2Ind); ss2Ind = ss2Ind(1:minT);
                ss6Ind = find(labels==6); ss6Ind = Shuffle(ss6Ind); ss6Ind = ss6Ind(1:minT);
                
                % choose balanced cutoffs for each
                cutoffs = round(linspace(1,length(ss2Ind),4));
                
                % index the data and the labels by this random balanced index!
                trainingDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,tBins(b1):tBins(b1+1))),3));
                testingDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,tBins(b2):tBins(b2+1))),3));

                % Assign the labels
                trnDat = trainingDat([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));],:);
                trnLabels = labels([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));]);
                
                tstDat = testingDat([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)],:);
                tstLabels = labels([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)]);
                
                % Run the classifier
                [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
                
                % Calculate error rate!
                errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
                
                % Save accuracy!
                cls_acc_load_allchans(it,b1,b2) = 1 -  errRate;
                
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
                cls_acc_load_allchans_shuffle(it,b1,b2) = 1 - errRate;
                
            end % end iterations loop!
            fprintf('.');
        end % end training time bin loop!
    end % end testing time bin loop!
    toc % display time for this subject
    
    cls.acc_load_allchans(s,:,:) = nanmean(cls_acc_load_allchans,1); % Average across iterations
    cls.acc_load_allchans_shuffle(s,:,:) = nanmean(cls_acc_load_allchans_shuffle,1);
    
    fprintf(sprintf('\n Subject %d out of %d finished \n',s,nSubs));
end % end subject loop

save([pwd,'/CDA_classification/load_classify_diagLinear_iterations_crossTemporal.mat'],'cls')



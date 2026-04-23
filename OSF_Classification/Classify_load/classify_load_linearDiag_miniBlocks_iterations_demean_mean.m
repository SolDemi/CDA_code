% Attempt classifying load over time.
%
% This version: Instead of doing single trials, instead try breaking trials
% up into "blocks" of, say, 10 trials
%
% See if this is better than feeding the classifier single trial
% information!
%
% Is it just a global signal? Look at average accuracy for (1) all
% electrodes, but demeaned across electrodes, vs. (2)

dbstop if error;
clear all;
cls = struct();

addpath('CDA_data'); addpath('CDA_downsamp_analysis');
load grand_cda_alltrials.mat

% only use subjects with a certain # of trials!
subjects = grand.subjects(grand.min_trials_per_cond>=160);
cls.subjects = subjects;

% subjects = 1:8;

% Load example file
load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(1))),filesep,'erp_singletrial.mat'])

tPts = cda.time;
blockSizes = [10];

for bk = 1:length(blockSizes)
    
    binSize = 50; cls.binSize = binSize;
    
    tBins = [min(tPts):binSize:max(tPts)];
    
    cls.time = tPts;
    cls.bins = tBins;
    cls.binCenters = [min(tPts)+round(binSize./2):binSize:max(tPts)-round(binSize./2)];
    cls.blockSize = blockSizes(bk);
    cls.iterations = 100; nIterations = cls.iterations;
    
    nSubs = length(subjects);
    nBins = length(tBins) - 1;
    nChans = length(erp.allChans);
    cls.chanLabels = erp.allChans;
    
    cls.acc_load_allchans_demean =  NaN(nSubs,nBins);
    cls.acc_load_allchans_mean =  NaN(nSubs,nBins);
    cls.acc_load_allchans_shuffle =  NaN(nSubs,nBins);
    
    % Loop through subjects
    for s = 1:nSubs
        %% Load subject
        load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(s))),filesep,'erp_singletrial.mat'])
        %%
        % For each trial type, break up into "mini blocks"
        
        %%%% condition 1: Left Color 2
        blocks = 1:cls.blockSize:size(erp.trial.L_C2,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        L_C2_blocked = NaN(length(blocks),nIterations,size(erp.trial.L_C2,2),size(erp.trial.L_C2,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.L_C2,1));
            tDat = erp.trial.L_C2(shuffInd,:,:);
            for b = 1:length(blocks)
                L_C2_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        %%%% condition 2: Left Color 6
        blocks = 1:cls.blockSize:size(erp.trial.L_C6,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        L_C6_blocked = NaN(length(blocks),nIterations,size(erp.trial.L_C6,2),size(erp.trial.L_C6,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.L_C6,1));
            tDat = erp.trial.L_C6(shuffInd,:,:);
            for b = 1:length(blocks)
                L_C6_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        %%%% condition 3: Left Shape 2
        blocks = 1:cls.blockSize:size(erp.trial.L_S2,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        L_S2_blocked = NaN(length(blocks),nIterations,size(erp.trial.L_S2,2),size(erp.trial.L_S2,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.L_S2,1));
            tDat = erp.trial.L_S2(shuffInd,:,:);
            for b = 1:length(blocks)
                L_S2_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        %%%% condition 4: Left Shape 6
        blocks = 1:cls.blockSize:size(erp.trial.L_S6,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        L_S6_blocked = NaN(length(blocks),nIterations,size(erp.trial.L_S6,2),size(erp.trial.L_S6,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.L_S6,1));
            tDat = erp.trial.L_S6(shuffInd,:,:);
            for b = 1:length(blocks)
                L_S6_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        %%%% condition 5: Right Color 2
        blocks = 1:cls.blockSize:size(erp.trial.R_C2,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        R_C2_blocked = NaN(length(blocks),nIterations,size(erp.trial.R_C2,2),size(erp.trial.R_C2,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.R_C2,1));
            tDat = erp.trial.R_C2(shuffInd,:,:);
            for b = 1:length(blocks)
                R_C2_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        %%%% condition 6: Right Color 6
        blocks = 1:cls.blockSize:size(erp.trial.R_C6,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        R_C6_blocked = NaN(length(blocks),nIterations,size(erp.trial.R_C6,2),size(erp.trial.R_C6,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.R_C6,1));
            tDat = erp.trial.R_C6(shuffInd,:,:);
            for b = 1:length(blocks)
                R_C6_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        %%%% condition 7: Right Shape 2
        blocks = 1:cls.blockSize:size(erp.trial.R_S2,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        R_S2_blocked = NaN(length(blocks),nIterations,size(erp.trial.R_S2,2),size(erp.trial.R_S2,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.R_S2,1));
            tDat = erp.trial.R_S2(shuffInd,:,:);
            for b = 1:length(blocks)
                R_S2_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        %%%% condition 8: Right Shape 6
        blocks = 1:cls.blockSize:size(erp.trial.R_S6,1); blocks = blocks(1:end-1); % make sure we only use full blocks!
        R_S6_blocked = NaN(length(blocks),nIterations,size(erp.trial.R_S6,2),size(erp.trial.R_S6,3));
        for it = 1:nIterations
            shuffInd = ShuffleKA(1:size(erp.trial.R_S6,1));
            tDat = erp.trial.R_S6(shuffInd,:,:);
            for b = 1:length(blocks)
                R_S6_blocked(b,it,:,:) = squeeze(nanmean(tDat(blocks(b):blocks(b)+(cls.blockSize-1),:,:),1));
            end
        end
        
        % Average for this timepoint, create a matrix and left/ right data
        tempDat = [L_C2_blocked; L_C6_blocked; L_S2_blocked; L_S6_blocked; ...
            R_C2_blocked; R_C6_blocked; R_S2_blocked; R_S6_blocked; ];
        
        labels = [ones(size(L_C2_blocked,1),1)*2; ones(size(L_C6_blocked,1),1)*6; ...
            ones(size(L_S2_blocked,1),1)*2; ones(size(L_S6_blocked,1),1)*6; ...
            ones(size(R_C2_blocked,1),1)*2; ones(size(R_C6_blocked,1),1)*6; ...
            ones(size(R_S2_blocked,1),1)*2; ones(size(R_S6_blocked,1),1)*6; ];
        
        cls_acc_load_allchans_demean = NaN(nIterations,nBins);
        cls_acc_load_allchans_mean = NaN(nIterations,nBins);
        cls_acc_load_allchans_shuffle = NaN(nIterations,nBins);
        
        tic
        % Loop through timepoints
        parfor b = 1:nBins
            
            for it = 1:nIterations
                %------------------------------------------------------------------
                % All electrodes --- demeaned data
                %------------------------------------------------------------------
                % Data for this time point (exclude EOG data)
                tDat1 = squeeze(nanmean(tempDat(:,it,1:20,ismember(tPts,tBins(b):tBins(b+1))),4));
                
                % Demean the data 
                baseDat = squeeze(nanmean(tDat1,2)); 
                baseDat = repmat(baseDat,1,size(tDat1,2)); 
                tDat = tDat1 - baseDat; 
                
                % Run linear classifier
                whichTrials = 1:size(tDat,1); whichTrials = ShuffleKA(whichTrials); % Total number of trials we have
                
                % Make 2/3 of data training and 1/3 of data test
                cutoffs = round(linspace(1,length(whichTrials),4));
                
                trnDat = tDat(whichTrials(1:cutoffs(3)),:);
                trnLabels = labels(whichTrials(1:cutoffs(3)));
                
                tstDat = tDat(whichTrials((cutoffs(3)+1):end),:);
                tstLabels = labels(whichTrials((cutoffs(3)+1):end));
                
                [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
                
                % Calculate error rate!
                errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
                
                % Save accuracy!
                cls_acc_load_allchans_demean(it,b) = 1 -  errRate;
                
                fprintf('.');
                %------------------------------------------------------------------
                % All electrodes --- ONLY mean across all electrrodes
                %------------------------------------------------------------------
                % Data for this time point (exclude EOG data)
                tDat = squeeze(nanmean(tDat1,2)); 
                % Run linear classifier
                whichTrials = 1:size(tDat,1); whichTrials = ShuffleKA(whichTrials); % Total number of trials we have
                
                % Make 2/3 of data training and 1/3 of data test
                cutoffs = round(linspace(1,length(whichTrials),4));
                
                trnDat = tDat(whichTrials(1:cutoffs(3)),:);
                trnLabels = labels(whichTrials(1:cutoffs(3)));
                
                tstDat = tDat(whichTrials((cutoffs(3)+1):end),:);
                tstLabels = labels(whichTrials((cutoffs(3)+1):end));
                
                [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
                
                % Calculate error rate!
                errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
                
                % Save accuracy!
                cls_acc_load_allchans_mean(it,b) = 1 -  errRate;
                
                fprintf('.');
                %------------------------------------------------------------------
                % All electrodes - SHUFFLE LABELS
                %------------------------------------------------------------------
                % Run linear classifier
                whichTrials = 1:size(tDat,1); whichTrials = ShuffleKA(whichTrials); % Total number of trials we have
                
                % Make 2/3 of data training and 1/3 of data test
                cutoffs = round(linspace(1,length(whichTrials),4));
                
                trnDat = tDat(whichTrials(1:cutoffs(3)),:);
                trnLabels = labels(whichTrials(1:cutoffs(3))); trnLabels = ShuffleKA(trnLabels);
                
                tstDat = tDat(whichTrials((cutoffs(3)+1):end),:);
                tstLabels = labels(whichTrials((cutoffs(3)+1):end)); tstLabels = ShuffleKA(tstLabels);
                
                [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
                
                % Calculate error rate!
                errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
                
                % Save accuracy!
                cls_acc_load_allchans_shuffle(it,b) = 1 - errRate;
                
                fprintf('.');
                
            end % End iterations loop
        end % end time bins loop!
        toc % display time for this subject
        
        cls.acc_load_allchans_demean(s,:) = mean(cls_acc_load_allchans_demean);
        cls.acc_load_allchans_mean(s,:) = mean(cls_acc_load_allchans_mean);
        cls.acc_load_allchans_shuffle(s,:) = mean(cls_acc_load_allchans_shuffle);
        
        fprintf(sprintf('\n Subject %d out of %d finished \n',s,nSubs));
    end % end subject loop
    
    fName = sprintf('/CDA_classification/load_classify_diagLinear_demean_%dit_block%d.mat',cls.iterations,cls.blockSize);
    save([pwd,fName],'cls')
    
end



% Attempt clsifying left versus right over time.
dbstop if error;

% %%% all subjects 
subjects = [1:8,10:18,20:26,28:34,36:47,49:52,54:63,65:68,70,73:74,...
    76,78:102,104:112,115:121,123:133,135:138,140:147,150,152:156,...
    158:160,162:172,174:175,178:181,184:191,194,196:198,203:204,206:219];

% Load example file 
load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(1))),filesep,'erp_singletrial.mat'])

tPts = cda.time;
binSize = 50;
tBins = [min(tPts):binSize:max(tPts)]; 

cls.time = tPts;
cls.bins = tBins; 
cls.binCenters = [min(tPts)+round(binSize./2):binSize:max(tPts)-binSize];

nSubs = length(subjects); 
nBins = length(tBins) - 1; 

pchans = ismember(erp.allChans,{'O1','O2','OL','OR','PO3','PO4','T5','T6'}); 
cchans = ismember(erp.allChans,{'P3','Pz','P4','C3','Cz','C4','T3','T4'}); 

fchans = ismember(erp.allChans,{'F3','F4','Fz'}); 

cls.acc_task_allchans =  NaN(nSubs,nBins); 
cls.acc_task_allchans_shuffle =  NaN(nSubs,nBins); 
 

% Loop through subjects
for s = 1:nSubs
    %% Load subject
    load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(s))),filesep,'erp_singletrial.mat'])
    %%
    % Average for this timepoint, create a matrix and left/ right data
    tempDat = [erp.trial.L_C2; erp.trial.L_C6; erp.trial.L_S2; erp.trial.L_S6; ...
        erp.trial.R_C2; erp.trial.R_C6; erp.trial.R_S2; erp.trial.R_S6; ];
    
    labels = [ones(size(erp.trial.L_C2,1),1)*1; ones(size(erp.trial.L_C6,1),1)*1; ...
        ones(size(erp.trial.L_S2,1),1)*2; ones(size(erp.trial.L_S6,1),1)*2; ...
        ones(size(erp.trial.R_C2,1),1)*1; ones(size(erp.trial.R_C6,1),1)*1; ...
        ones(size(erp.trial.R_S2,1),1)*2; ones(size(erp.trial.R_S6,1),1)*2; ];
    
    cls_acc_task_allchans = NaN(1,nBins);
    cls_acc_task_allchans_shuffle = NaN(1,nBins);
    
    % Loop through timepoints
    for b = 1:nBins
        %------------------------------------------------------------------
        % All electrodes
        %------------------------------------------------------------------
        % Data for this time point (exclude EOG data)
        tDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,tBins(b):tBins(b+1))),3));
        
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
        cls_acc_task_allchans(b) = 1 -  errRate;
        
        fprintf('.');
       
        %------------------------------------------------------------------
        % All electrodes - SHUFFLE
        %------------------------------------------------------------------
        
        % Data for this time point (exclude EOG data)
        %         tDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,tBins(b):tBins(b+1))),3));
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
        cls_acc_task_allchans_shuffle(b) = 1 - errRate;
            
        fprintf('.');
    end % end time bins loop!
    
    cls.acc_task_allchans(s,:) = cls_acc_task_allchans;
    cls.acc_task_allchans_shuffle(s,:) = cls_acc_task_allchans_shuffle;
    
    fprintf(sprintf('\n Subject %d out of %d finished \n',s,nSubs));
end % end subject loop

save([pwd,'/CDA_classification/task_classify_diagLinear.mat'],'cls','subjects')




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

cls.acc_lr_allchans =  NaN(nSubs,nBins); 
cls.acc_lr_allchans_shuffle =  NaN(nSubs,nBins); 
cls.acc_lr_pchans =  NaN(nSubs,nBins); 
cls.acc_lr_fchans =  NaN(nSubs,nBins); 
cls.acc_lr_cchans =  NaN(nSubs,nBins); 

cls.weights_lr_allchans =  NaN(nSubs,nBins,20); 

% Loop through subjects
for s = 1:nSubs
    %% Load subject
    load([pwd,filesep,'CDA_data',filesep,char(num2str(subjects(s))),filesep,'erp_singletrial.mat'])
    %%
    % Average for this timepoint, create a matrix and left/ right data
    tempDat = [erp.trial.L_C2; erp.trial.L_C6; erp.trial.L_S2; erp.trial.L_S6; ...
        erp.trial.R_C2; erp.trial.R_C6; erp.trial.R_S2; erp.trial.R_S6; ];
    
    labels = [ones(size(erp.trial.L_C2,1),1)*1; ones(size(erp.trial.L_C6,1),1)*1; ...
        ones(size(erp.trial.L_S2,1),1)*1; ones(size(erp.trial.L_S6,1),1)*1; ...
        ones(size(erp.trial.R_C2,1),1)*2; ones(size(erp.trial.R_C6,1),1)*2; ...
        ones(size(erp.trial.R_S2,1),1)*2; ones(size(erp.trial.R_S6,1),1)*2; ];
    
    % Loop through timepoints
    for b = 1:nBins
        %------------------------------------------------------------------
        % All electrodes
        %------------------------------------------------------------------
        % Data for this time point (exclude EOG data)
        tDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,tBins(b):tBins(b+1))),3));
        
        % Run support vector machine.
        svm = fitcsvm(tDat,labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        % Crossvalidate
        cval = crossval(svm);
        % Figure out accuracy
        classLoss = kfoldLoss(cval);
        % Save accuracy!
        cls.acc_lr_allchans(s,b) = 1 - classLoss;
        cls.weights_lr_allchans(s,b,:) = svm.Mu; % weights? 
        
        fprintf('.');
        %------------------------------------------------------------------
        % All electrodes - SHUFFLE
        %------------------------------------------------------------------
        % Data for this time point (exclude EOG data)
        tDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,tBins(b):tBins(b+1))),3));
        
        % Run support vector machine.
        svm = fitcsvm(tDat,Shuffle(labels),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        % Crossvalidate
        cval = crossval(svm);
        % Figure out accuracy
        classLoss = kfoldLoss(cval);
        % Save accuracy!
        cls.acc_lr_allchans_shuffle(s,b) = 1 - classLoss;
%         cls.weights_lr_allchans(s,b,:) = svm.Mu; % weights?
        
        fprintf('.');
        %------------------------------------------------------------------
        % Posterior electrodes
        %------------------------------------------------------------------
        % Data for this time point (exclude EOG data)
        tDat = squeeze(nanmean(tempDat(:,pchans,ismember(tPts,tBins(b):tBins(b+1))),3));
        
        % Run support vector machine.
        svm = fitcsvm(tDat,labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        % Crossvalidate
        cval = crossval(svm);
        % Figure out accuracy
        classLoss = kfoldLoss(cval);
        % Save accuracy!
        cls.acc_lr_pchans(s,b) = 1 - classLoss;
%         cls.weights_lr_pchans(s,b,:) = svm.Mu; % weights? 
        
        fprintf('.');
        
        %------------------------------------------------------------------
        % Frontal electrodes
        %------------------------------------------------------------------
        % Data for this time point (exclude EOG data)
        tDat = squeeze(nanmean(tempDat(:,fchans,ismember(tPts,tBins(b):tBins(b+1))),3));
        
        % Run support vector machine.
        svm = fitcsvm(tDat,labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        % Crossvalidate
        cval = crossval(svm);
        % Figure out accuracy
        classLoss = kfoldLoss(cval);
        % Save accuracy!
        cls.acc_lr_fchans(s,b) = 1 - classLoss;
%         cls.weights_lr_fchans(s,b,:) = svm.Mu; % weights? 
        
        fprintf('.');
        
        %------------------------------------------------------------------
        % Central electrodes
        %------------------------------------------------------------------
        % Data for this time point (exclude EOG data)
        tDat = squeeze(nanmean(tempDat(:,cchans,ismember(tPts,tBins(b):tBins(b+1))),3));
        
        % Run support vector machine.
        svm = fitcsvm(tDat,labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        % Crossvalidate
        cval = crossval(svm);
        % Figure out accuracy
        classLoss = kfoldLoss(cval);
        % Save accuracy!
        cls.acc_lr_cchans(s,b) = 1 - classLoss;
%         cls.weights_lr_fchans(s,b,:) = svm.Mu; % weights? 
        
        fprintf('.');
        
        
    end % end time bins loop!
    
    fprintf(sprintf('\n Subject %d out of %d finished \n',s,nSubs));
end % end subject loop

save([pwd,'/CDA_classification/left_right_classify.mat'],'cls','subjects')



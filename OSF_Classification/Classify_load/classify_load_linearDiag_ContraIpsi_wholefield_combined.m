
% Attempt clsifying left versus right over time.
dbstop if error;

%%%% all subjects
subjects = [1:8,10:18,20:26,28:34,36:47,49:52,54:63,65:68,70,73:74,...
    76,78:102,104:112,115:121,123:133,135:138,140:147,150,152:156,...
    158:160,162:172,174:175,178:181,184:191,194,196:198,203:204,206:219];

% Load example file 
maindir = erase( pwd,'\code');
outputdir = [maindir,'\decoding\OSF_CDA\'];
if ~isfolder(outputdir)
    mkdir(outputdir)
end

load([maindir,filesep,'data_raw',filesep,char(num2str(subjects(1))),filesep,'erp_singletrial.mat'])

tPts = cda.time;
binSize = 50;
tBins = [min(tPts):binSize:max(tPts)];

cls.subjects = subjects;
cls.time = tPts;
cls.bins = tBins;
cls.binCenters = [min(tPts)+round(binSize./2):binSize:max(tPts)-binSize];
cls.nIter = 100; nIter = cls.nIter;

nSubs = length(subjects);
nBins = length(tBins) - 1;

pchans = ismember(erp.allChans,{'O1','O2','OL','OR','PO3','PO4','T5','T6'});
fchans = ismember(erp.allChans,{'F3','F4','Fz'});
cchans = ismember(erp.allChans,{'C3','Cz','C4','P3','Pz','P4','T3','T4'});
cls.allChans = erp.allChans;
cls.pairsL = {'O1','OL','T5','PO3','P3','C3','T3','F3'}; pL = cls.pairsL;
cls.pairsR = {'O2','OR','T6','PO4','P4','C4','T4','F4'}; pR = cls.pairsR;
cls.allPairs = {'O1/O2','OL/OR','T5/T6','PO3/PO4','P3/P4','C3/C4','T3/T4','F3/F4'}; allPairs = cls.allPairs;
cls.cdaPairs = {'O1/O2','OL/OR','T5/T6','PO3/PO4'}; cdaPairs = cls.cdaPairs;
nPairs = length(pL);

cls.acc_load_allchans =  NaN(nSubs,nBins);
cls.acc_load_allpairs =  NaN(nSubs,nBins);
cls.acc_load_allchans_allpairs =  NaN(nSubs,nBins);
cls.acc_load_allchans_shuffle =  NaN(nSubs,nBins);

% Loop through subjects
for s = 1:nSubs
    %% Load subject
    load([maindir,filesep,'data_raw',filesep,char(num2str(subjects(s))),filesep,'erp_singletrial.mat'])

    if min( cda.trials_per_cond ) < 160
        bad_sub = [bad_sub s];
        continue
    end

    %%
    % Average for this timepoint, create a matrix and left/ right data
    tempDat_chans = [erp.trial.L_C2; erp.trial.L_C6; erp.trial.L_S2; erp.trial.L_S6; ...
        erp.trial.R_C2; erp.trial.R_C6; erp.trial.R_S2; erp.trial.R_S6; ];
    
    % Average for this timepoint, create a matrix and left/ right data
    tempDatL = [erp.trial.L_C2; erp.trial.L_C6; erp.trial.L_S2; erp.trial.L_S6; ];
    % Rehape left data into contra and ipsi!
    pairDatL_contra = squeeze(tempDatL(:,ismember(cls.allChans,pR),:));
    pairDatL_ipsi = squeeze(tempDatL(:,ismember(cls.allChans,pL),:));
    pairDatL = NaN(size(pairDatL_contra,1),nPairs*2,size(pairDatL_contra,3));
    pairDatL(:,1:nPairs,:) = pairDatL_contra;
    pairDatL(:,nPairs+1:end,:) = pairDatL_ipsi;
    
    % Right data
    tempDatR = [erp.trial.R_C2; erp.trial.R_C6; erp.trial.R_S2; erp.trial.R_S6;];
    pairDatR_contra = squeeze(tempDatR(:,ismember(cls.allChans,pL),:));
    pairDatR_ipsi = squeeze(tempDatR(:,ismember(cls.allChans,pR),:));
    pairDatR = NaN(size(pairDatR_contra,1),nPairs*2,size(pairDatR_contra,3));
    pairDatR(:,1:nPairs,:) = pairDatR_contra;
    pairDatR(:,nPairs+1:end,:) = pairDatR_ipsi;
    % combine back into one matrix
    tempDat_pairs = [pairDatL;pairDatR];
    
    tempDat = NaN(size(tempDat_pairs,1),(size(tempDat_pairs,2)+size(tempDat_chans,2)-2),size(tempDat_pairs,3));
    tempDat(:,1:20,:) = tempDat_chans(:,1:20,:);
    tempDat(:,21:36,:) = tempDat_pairs;
    
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
    
    
    cls_acc_load_allchans = NaN(nIter,nBins);
    cls_acc_load_allchans_shuffle = NaN(nIter,nBins);
    cls_acc_load_allpairs = NaN(nIter,nBins);
    cls_acc_load_allchans_allpairs = NaN(nIter,nBins);
    
    tic
    % Loop through timepoints
    for b = 1:nBins
        for it = 1:nIter
            %------------------------------------------------------------------
            % All electrodes, only chans
            %------------------------------------------------------------------
            % Data for this time point (exclude EOG data)
            
            % randomly trim excess trials from set sizes with more than the minimum.
            ss2Ind = find(labels==2); ss2Ind = ss2Ind( randperm( length( ss2Ind ) ) ); ss2Ind = ss2Ind(1:minT);
            ss6Ind = find(labels==6); ss6Ind = ss6Ind( randperm( length( ss6Ind ) ) ); ss6Ind = ss6Ind(1:minT);
            
            % choose balanced cutoffs for each
            cutoffs = round(linspace(1,length(ss2Ind),4));
            
            % Data for this time point (exclude EOG data)
            tDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,tBins(b):tBins(b+1))),3));
            
            % Assign the labels
            trnDat = tDat([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));],:);
            trnLabels = labels([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));]);
            
            tstDat = tDat([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)],:);
            tstLabels = labels([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)]);
            
            [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
            
            % Calculate error rate!
            errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
            
            % Save accuracy!
            cls_acc_load_allchans(it,b) = 1 -  errRate;
            
            %------------------------------------------------------------------
            % All electrodes, only pairs
            %------------------------------------------------------------------
            % Data for this time point (exclude EOG data)
            tDat = squeeze(nanmean(tempDat(:,21:36,ismember(tPts,tBins(b):tBins(b+1))),3));
            
            % Assign the labels
            trnDat = tDat([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));],:);
            trnLabels = labels([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));]);
            
            tstDat = tDat([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)],:);
            tstLabels = labels([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)]);
            
            [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
            
            % Calculate error rate!
            errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
            
            % Save accuracy!
            cls_acc_load_allpairs(it,b) = 1 -  errRate;
            
            %------------------------------------------------------------------
            % All electrodes, chans AND pairs
            %------------------------------------------------------------------
            % Data for this time point (exclude EOG data)
            tDat = squeeze(nanmean(tempDat(:,1:36,ismember(tPts,tBins(b):tBins(b+1))),3));
            
            % Assign the labels
            trnDat = tDat([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));],:);
            trnLabels = labels([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));]);
            
            tstDat = tDat([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)],:);
            tstLabels = labels([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)]);
            
            [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
            
            % Calculate error rate!
            errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
            
            % Save accuracy!
            cls_acc_load_allchans_allpairs(it,b) = 1 -  errRate;
            
            %------------------------------------------------------------------
            % All electrodes - SHUFFLE LABELS
            %------------------------------------------------------------------
            
            % keep same data from step above.
            
            % Assign the labels
            trnDat = tDat([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));],:);
            trnLabels = labels([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));]);
            
            tstDat = tDat([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)],:);
            tstLabels = labels([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)]);
            
            % Shuffle the training labels! 
            trnLabels = trnLabels( randperm( length( trnLabels ) ) );
            
            [classOutput] = classify(tstDat,trnDat,trnLabels,'diagLinear');
            
            % Calculate error rate!
            errRate = sum(classOutput ~= tstLabels) ./ length(tstLabels);
            
            % Save accuracy!
            cls_acc_load_allchans_shuffle(it,b) = 1 - errRate;
            
        end
        fprintf('.');
    end % end time bins loop!
    toc % display time for this subject
    
    cls.acc_load_allchans(s,:) = mean(cls_acc_load_allchans,1);
    cls.acc_load_allchans_shuffle(s,:) = mean(cls_acc_load_allchans_shuffle,1);
    cls.acc_load_allpairs(s,:) = mean(cls_acc_load_allpairs,1);
    cls.acc_load_allchans_allpairs(s,:) = mean(cls_acc_load_allchans_allpairs,1);
    
    fprintf(sprintf('\n Subject %d out of %d finished \n',s,nSubs));
end % end subject loop


save([maindir,'\decoding\CDA_LDA.mat'],'cls')



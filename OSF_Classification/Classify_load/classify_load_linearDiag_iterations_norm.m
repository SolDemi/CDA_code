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
nChans = length(cls.chans);

cls.skew =  NaN(nSubs,nIter,nChans);
cls.kurt =  NaN(nSubs,nIter,nChans);
cls.pvals =  NaN(nSubs,nIter,nChans);

cls.skew_allchans =  NaN(nSubs,nIter);
cls.kurt_allchans =  NaN(nSubs,nIter);
cls.pvals_allchans =  NaN(nSubs,nIter);

cls.nTrials = NaN(nSubs,1);

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
    
    cls.nTrials(s) = length(labels);
    
    setSizes = unique(labels); nSS = length(setSizes);
    
    % Set up a matrix / info for balancing trials in the training and test
    % sets.
    nT_ss = NaN(1,nSS);
    for ss = 1:nSS
        nT_ss(ss) = sum(labels==setSizes(ss));
    end
    minT = min(nT_ss);
    
    cls_skew = NaN(nIter,nChans);
    cls_kurt = NaN(nIter,nChans);
    cls_pvals = NaN(nIter,nChans);
    
    cls_skew_allchans = NaN(nIter,1);
    cls_kurt_allchans = NaN(nIter,1);
    cls_pvals_allchans = NaN(nIter,1);
    
    
    tic
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
        tDat = squeeze(nanmean(tempDat(:,1:20,ismember(tPts,400:1050)),3));
        
        % Assign the labels
        trnDat = tDat([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));],:);
        trnLabels = labels([ss2Ind(1:cutoffs(3));ss6Ind(1:cutoffs(3));]);
        
        tstDat = tDat([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)],:);
        tstLabels = labels([ss2Ind((cutoffs(3)+1):end); ss6Ind((cutoffs(3)+1):end)]);
        
        % Calculate skewness and kurtosis for each channel!
        tmpDat = [trnDat;tstDat];
        cls_skew(it,:) = skewness(tmpDat);
        cls_kurt(it,:) = kurtosis(tmpDat);
        
        pvals = NaN(1,nChans);
        for ch = 1:nChans
            % Kolmogorov-Smirnov test for normality
            chDat = tmpDat(:,ch);
            chDat = [(chDat-mean(chDat)) ./ std(chDat)];
            [h,p] = kstest(chDat);
            pvals(ch) = p;
        end
        
        cls_pvals(it,:) = pvals;
        
        % Calculate combined across ALL channels
        tmpDat = tmpDat(:);
        cls_skew_allchans(it) = skewness(tmpDat);
        cls_kurt_allchans(it) = kurtosis(tmpDat);
        [h,p] = kstest((tmpDat-mean(tmpDat))./std(tmpDat));
        cls_pvals_allchans(it) = p;
        
        
    end % end iterations loop!
    fprintf('.');
    % display time for this subject
    toc
    
    cls.skew(s,:,:) = cls_skew;
    cls.kurt(s,:,:) = cls_kurt;
    cls.pvals(s,:,:) = cls_pvals;
    
    cls.skew_allchans(s,:) = cls_skew_allchans;
    cls.kurt_allchans(s,:) = cls_kurt_allchans;
    cls.pvals_allchans(s,:) = cls_pvals_allchans;
    fprintf(sprintf('\n Subject %d out of %d finished \n',s,nSubs));
end % end subject loop

save([pwd,'/CDA_classification/load_classify_diagLinear_iterations_norm.mat'],'cls','-v7.3')



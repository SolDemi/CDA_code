%% Loop through subjects to extract single-subject CDA values and save them within each subject's folder
%  After this, we can go through and do our downsmpling analyses and also
%  plot our grand CDA for the 4 conditions that we have. 
dbstop if error
clear all

%%%% all subjects 
subjects = [1:8,10:18,20:26,28:34,36:47,49:52,54:63,65:68,70,73:74,...
    76,78:102,104:112,115:121,123:133,135:138,140:147,150,152:156,...
    158:160,162:172,174:175,178:181,184:191,194,196:198,203:204,206:219];

datadir = pwd;
maindir = erase(datadir,'\code');
homedir = [maindir,'\erp'];

    cd(homedir)

output_dir = [maindir '\cda\'];
if ~isfolder(output_dir)
    mkdir(output_dir)
end

for s = 48:length(subjects)
    
    clear erp cda
    
    subj = subjects(s);

    
    
    

load( sprintf('sub%d', subj) ) 

erp.trial = {};
conditions_LR = {'L_C2';'L_C6';'L_S2';'L_S6';'R_C2';'R_C6';'R_S2';'R_S6';};
conditions = {'C2';'C6';'S2';'S6';};
channels = {'PO3';'PO4';'F3';'F4';'C3';'C4';'P3';'P4';'O1';'O2';'OL';'OR';'T3';'T4';'T5';'T6';'POz';'Cz';'Fz';'Pz'};

% Kei's "left" and "right" channels (erp . arf.chansLH / erp . arf.chansRH) are
% TOTALLY WRONG!!! REDO THIS!! ARGH!! 
erp.allChans = erp.arf.chanLabels;
erp.rightChans = ismember(erp.allChans,{'O2','OR','P4','PO4','T6'});
erp.leftChans = ismember(erp.allChans,{'O1','OL','P3','PO3','T5'});

cue_window = 1:50;%instep
baseline_window = 1:50;%instep
memory_window = 51:300;

erp.baseline = 201:250;
erp.pre_timepoint = 250;
erp.post_timepoint = 249;


erp.trialCodes = zeros( 1,length(erp.eventCodes) );

% Note, if you care about aligning single trials with behavior, you would
% first want to ditch ALL response codes in case they happen to appear in
% between the combination of trial codes that define experimental conditoins 
% (e.g.,  trial code C, trial Code C+1 below). 
% For our purposes, this doesn't really matter since we're throwing away a
% bunch of trials with artifacts anyway 
% 
% erp.eventTimes = erp.eventTimes(erp.eventCodes<212);
% erp.eventCodes = erp.eventCodes(erp.eventCodes<212);
% 

%Epoch1!!
%GOTTA SORT OUT THE CONDITION ORDER!!!!!!!!
%IMPLEMENT TRIALCODES based on eventcoes!!!!!!!!!
for ec = 1:1:length(erp.eventCodes)-1
        if erp.eventCodes(ec) == 7 
            if erp.eventCodes(ec+1) == 12 
                erp.trialCodes(ec+1) = 1;
            elseif erp.eventCodes(ec+1) == 16 
                erp.trialCodes(ec+1) = 2;
            elseif erp.eventCodes(ec+1) == 22 
                erp.trialCodes(ec+1) = 3;
            elseif erp.eventCodes(ec+1) == 26 
                erp.trialCodes(ec+1) = 4;
            end;
        elseif erp.eventCodes(ec) == 9 
            if erp.eventCodes(ec+1) == 12 
                erp.trialCodes(ec+1) = 5;
            elseif erp.eventCodes(ec+1) == 16 
                erp.trialCodes(ec+1) = 6;
            elseif erp.eventCodes(ec+1) == 22 
                erp.trialCodes(ec+1) = 7;
            elseif erp.eventCodes(ec+1) == 26 
                erp.trialCodes(ec+1) = 8;
            end;
        end;

end;

%IMPLEMENT TRIALCODES based on eventcoes!!!!!!!!! 
temp_arf_counter = 1; temp_arf_ind = [];

counters = ones(1,length(conditions_LR));
for ec  = 1:1:length(erp.eventCodes)

    if erp.trialCodes(ec) > 0 %found a trial segment!
        fieldname = (char(conditions_LR(erp.trialCodes(ec),:)));
        artifact = 0;
        %adjust the timewindow based on condition
            pre_timepoint = erp.pre_timepoint;
            post_timepoint = erp.post_timepoint;
        
            
        %check the time range if it's artifact free
        if sum(erp.arf.blink(erp.eventTimes(ec)-pre_timepoint:erp.eventTimes(ec)+post_timepoint))>1
            artifact = 1;
        elseif sum(erp.arf.eMove(erp.eventTimes(ec)-pre_timepoint:erp.eventTimes(ec)+post_timepoint))>1
            artifact = 1;
        elseif sum(sum(erp.arf.blocking(:,erp.eventTimes(ec)-pre_timepoint:erp.eventTimes(ec)+post_timepoint)))>1
            artifact = 1;
        end;
        
        temp_arf_ind(temp_arf_counter) = artifact;
        temp_arf_counter =temp_arf_counter+1;
      
        if artifact == 0
            erp.trial.(fieldname)(counters(erp.trialCodes(ec)),:,:)= erp.data(:,erp.eventTimes(ec)-pre_timepoint:erp.eventTimes(ec)+post_timepoint);

            counters(erp.trialCodes(ec))= counters(erp.trialCodes(ec))+1;
        end;
    end;


end;

%now baseline it!
for condition = 1:1:length(conditions_LR)

    fieldname = (char(conditions_LR(condition,:)));

    baseline = repmat(squeeze(mean(erp.trial.(fieldname)(:,:,erp.baseline),3)),[1,1,size(erp.trial.(fieldname),3)]);     
    erp.trial.(fieldname)= erp.trial.(fieldname)-baseline;

end;

%%%% make contra-ipsi plots


cda.trial.contra_C2 = [squeeze(nanmean(erp.trial.L_C2(:,erp.rightChans,:),2)) ; squeeze(nanmean(erp.trial.R_C2(:,erp.leftChans,:),2))];
cda.trial.ipsi_C2 = [squeeze(nanmean(erp.trial.L_C2(:,erp.leftChans,:),2)) ; squeeze(nanmean(erp.trial.R_C2(:,erp.rightChans,:),2))];

cda.trial.contra_C6 = [squeeze(nanmean(erp.trial.L_C6(:,erp.rightChans,:),2)) ; squeeze(nanmean(erp.trial.R_C6(:,erp.leftChans,:),2))];
cda.trial.ipsi_C6 = [squeeze(nanmean(erp.trial.L_C6(:,erp.leftChans,:),2)) ; squeeze(nanmean(erp.trial.R_C6(:,erp.rightChans,:),2))];

cda.trial.contra_S2 = [squeeze(nanmean(erp.trial.L_S2(:,erp.rightChans,:),2)) ; squeeze(nanmean(erp.trial.R_S2(:,erp.leftChans,:),2))];
cda.trial.ipsi_S2 = [squeeze(nanmean(erp.trial.L_S2(:,erp.leftChans,:),2)) ; squeeze(nanmean(erp.trial.R_S2(:,erp.rightChans,:),2))];

cda.trial.contra_S6 = [squeeze(nanmean(erp.trial.L_S6(:,erp.rightChans,:),2)) ; squeeze(nanmean(erp.trial.R_S6(:,erp.leftChans,:),2))];
cda.trial.ipsi_S6 = [squeeze(nanmean(erp.trial.L_S6(:,erp.leftChans,:),2)) ; squeeze(nanmean(erp.trial.R_S6(:,erp.rightChans,:),2))];

cda.diff_C2 = cda.trial.contra_C2 - cda.trial.ipsi_C2;
cda.diff_C6 = cda.trial.contra_C6 - cda.trial.ipsi_C6;
cda.diff_S2 = cda.trial.contra_S2 - cda.trial.ipsi_S2;
cda.diff_S6 = cda.trial.contra_S6 - cda.trial.ipsi_S6;

cda.diff_2 = [cda.diff_C2;cda.diff_S2];
cda.diff_6 = [cda.diff_C6;cda.diff_S6];

cda.trials_per_cond = [size(cda.diff_C2,1),size(cda.diff_C6,1),size(cda.diff_S2,1),size(cda.diff_S6,1)];
cda.trials_per_ss = [size(cda.diff_2,1),size(cda.diff_6,1)];
cda.min_trials_per_cond = min(cda.trials_per_cond);
cda.min_trials_per_ss = min(cda.trials_per_ss);

cda.preTime = erp.pre_timepoint.*4; 
cda.postTime = erp.post_timepoint.*4;
cda.time = -cda.preTime:4:cda.postTime;

save([output_dir sprintf('sub%d', subj)],'erp','cda');
% save('cda_diff_singletrial','cda')


fprintf(sprintf('Subject %d Complete! \n',s))

end

 
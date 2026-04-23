%attempts to bin trials

output_dir2 = [maindir '\bi_erp\'];
if ~isfolder(output_dir2)
    mkdir(output_dir2)
end

erp.trial = struct();
conditions_LR = {'L_C2';'L_C6';'L_S2';'L_S6';'R_C2';'R_C6';'R_S2';'R_S6';};
conditions = {'C2';'C6';'S2';'S6';};
channels = {'PO3';'PO4';'F3';'F4';'C3';'C4';'P3';'P4';'O1';'O2';'OL';'OR';'T3';'T4';'T5';'T6';'POz';'Cz';'Fz';'Pz'};
%channels_r = {'F4';'F3';'C4';'C3';'P4';'P3';'PO4';'PO3';'O2';'O1';'OR';'OL';'T4';'T3';'T6';'T5';'Fz';'Cz';'Pz'};

%cue_code = {7 9};
%mem_codes = {11 12 13 14 16 18};

cue_window = [1:50];%instep
baseline_window = [1:50];%instep
memory_window = [51:300];

erp.baseline = [201:250];
erp.pre_timepoint = 250;
erp.post_timepoint = 249;



erp.trialCodes = zeros(1,length(erp.eventCodes));

%Epoch1!!
%GOTTA SORT OUT THE CONDITION ORDER!!!!!!!!
%IMPLEMENT TRIALCODES based on eventcoes!!!!!!!!!
for ec = 1:1:length(erp.eventCodes)
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

        if artifact == 0
            erp.trial.(fieldname)(counters(erp.trialCodes(ec)),:,:)= erp.filtered_data(:,erp.eventTimes(ec)-pre_timepoint:erp.eventTimes(ec)+post_timepoint);

            counters(erp.trialCodes(ec))= counters(erp.trialCodes(ec))+1;
        end;
    end;


end;

%now baseline it!
% 
% for condition = 1:1:length(conditions_LR)
% 
%     fieldname = (char(conditions_LR(condition,:)));
% 
%     baseline = repmat(squeeze(mean(erp.trial.(fieldname)(:,:,erp.baseline),3)),[1,1,size(erp.trial.(fieldname),3)]);     
%     erp.trial.(fieldname)= erp.trial.(fieldname)-baseline;
% 
% end;

%Now create the erp&ave files
bi_erp = struct();
bi_erp_ave = struct();

channels = {'PO3';'PO4';'F3';'F4';'C3';'C4';'P3';'P4';'O1';'O2';'OL';'OR';'T3';'T4';'T5';'T6';'POz';'Cz';'Fz';'Pz'};
channels_r = {'PO4';'PO3';'F4';'F3';'C4';'C3';'P4';'P3';'O2';'O1';'OR';'OL';'T4';'T3';'T6';'T5';'POz';'Cz';'Fz';'Pz'};
channels_r_index = [2 1 4 3 6 5 8 7 10 9 12 11 14 13 16 15 17 18 19 20];
for condition = 1:1:length(conditions);
    condition_name = (char(conditions(condition,:)));
    condition_name_l = ['L_',char(conditions(condition,:))];
    condition_name_r = ['R_',char(conditions(condition,:))];
    for channel = 1:1:length(channels)
        channel_name = char(channels(channel,:));
        fieldname = [condition_name,'_',channel_name];
        filendname_l = [condition_name_l,'_',channel_name];


            bi_erp.(fieldname) = [squeeze(erp.trial.(condition_name_l)(:,channel,:));squeeze(erp.trial.(condition_name_r)(:,channels_r_index(channel),:))];
            bi_erp_ave.(fieldname) = mean(bi_erp.(fieldname)(:,:),1); 


    end;
    
    
    
end;


%save('erp.mat','erp');
save([output_dir2 sprintf('sub%d', i) 'bi_erp.mat'],'bi_erp')
save([output_dir2 sprintf('sub%d', i) 'bi_erp_ave.mat'],'bi_erp_ave');
% 

 
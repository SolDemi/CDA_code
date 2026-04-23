subs = {'00';};
%subs = ['00';'01';'02';'03';'04';];
%subs = [713,714,715,716];
numsubs = length(subs);


%Prepare the outout array
numBlocks = 9;%10
numTrials = 96;%96
numallTrials = numBlocks*numTrials;
Time = -196:4:4996;

%Define the time of interest
Onset_time_of_interest = 0;
Offset_time_of_interest = 1496;


Step_Onset_time_of_interest = Onset_time_of_interest/4+1;
Step_Offset_time_of_interest = Offset_time_of_interest/4+1;


%Create plots
%channels = {'PO3';'PO4';'F3';'F4';'C3';'C4';'P3';'P4';'O1';'O2';'OL';'OR';'T3';'T4';'T5';'T6';'POz';'Cz';'Fz';'Pz'};
channels = {'PO3';'F4';'F3';'C4';'C3';'P4';'P3';'O2';'O1';'OR';'OL';'T4';'T3';'T6';'T5';'PO4';'Cz';'Fz';'Pz';'POz'};
conditions = {'C2';'C6';'S2';'S6';};

subplot_xy = [5,5];
subplot_indecies = [17,19,2,4,7,9,12,14,22,24,21,25,6,10,16,20,18,8,3,13];  

%Spectrogram specs

Fs = 250; %sampling frequency
T = 1/Fs; %sample time
L = Step_Offset_time_of_interest-Step_Onset_time_of_interest+1; %length of signal
L_s = 250; %length of signal


t = (0:L-1)*T; %Time Vector




NFFT = 2^nextpow2(L_s); %Next power of 2 from length of 
%NFFT_c = 2^nextpow2(L_c); %Next power of 2 from length of 
%NFFT_m = 2^nextpow2(L_m); %Next power of 2 from length of 
%NFFT_mcs = 2^nextpow2(L_mcs); %Next power of 2 from length of 
f = Fs/2*linspace(0,1,NFFT/2+1);
%f_c = Fs/2*linspace(0,1,NFFT_c/2+1);
%f_m = Fs/2*linspace(0,1,NFFT_m/2+1);
%f_mcs = Fs/2*linspace(0,1,NFFT_mcs/2+1);

Window = 100; %25 frames = 100ms
NOverlaps = 95; %Number of overlapping window frames


    





%Whole


    %change directory
    %subject_directory = char(subs(sub));
    %cd (subject_directory);

%load three files
    %cd (subject_directory);
    %Load files
    load bi_erp;
    
    bi_spec = struct();
    
    %Create fields and counter
    
    condition = 1;
    while condition < length(conditions)+1
        channel = 1;
        while channel < length(channels)+1
           
           condition_name = [char(conditions(condition,:))];
           temp_name = [condition_name,'_PO3'];
           
           numTrials = size(bi_erp.(temp_name),1);
            
           fieldname = [char(conditions(condition,:)),'_',char(channels(channel,:))];
           
           bi_spec.(fieldname) = zeros(numTrials,35,56);
               
           
           channel = channel+1;
        end;
        
        
        condition = condition+1;
    end;
    
    %start data crackin'
    
    bi_spec.SpecFs = f;
    bi_spec.SpecTs = Window*T*1000/2-500:20:(2600-Window*T*1000/2)-400;


    condition = 1;
    while condition < length(conditions)+1
        condition_name = [char(conditions(condition,:))];
        temp_name = [condition_name,'_PO3'];
        
        
           
        numTrials = size(bi_erp.(temp_name),1);

        
        
        
        channel = 1;
        while channel < length(channels)+1
            
            %input fieldname;
            fieldname = [condition_name,'_',char(channels(channel,:))];
           
           
            
            trial = 1;
            while trial < numTrials+1
                
                clear S;
                clear F;
                clear T;
                clear p;
            
                %Do spectrogram
                [S,F,T,P]= spectrogram(bi_erp.(fieldname)(trial,Step_Onset_time_of_interest:Step_Offset_time_of_interest),Window,NOverlaps,NFFT,Fs);
                bi_spec.(fieldname)(trial,:,:) = P(1:35,:);

                trial = trial+1;
            end;
            
 
            
            
            channel = channel+1;
        end;
        condition = condition+1;
        disp(condition)
    end;
    

        
%Make WholeSpecAVE 
bi_spec_ave = struct();
bi_spec_ave.SpecFs = f;
bi_spec_ave.SpecTs =  Window*T*1000/2-500:20:(2600-Window*T*1000/2)-400;

        
    condition = 1;
    while condition < length(conditions)+1
        condition_name = [char(conditions(condition,:))];
        temp_name = [condition_name,'_PO3'];
           
        
        channel = 1;
        while channel < length(channels)+1
            
            %input fieldname;
            fieldname = [condition_name,'_',char(channels(channel,:))];

            
            
           

            %make average

            if size(bi_spec.(fieldname)(:,:,:),1) > 1
                bi_spec_ave.(fieldname)(:,:) = squeeze(mean(bi_spec.(fieldname)(:,:,:)));

                
                
            else
                bi_spec_ave.(fieldname)(:,:) = squeeze(bi_spec.(fieldname)(:,:,:));

            end;
           

            
            channel = channel+1;
        end;
        condition = condition+1;
    end;
        

   disp(1000);
    
    
    
    %save('bi_spec.mat', 'bi_spec','-v7.3');
    save('bi_spec_ave.mat', 'bi_spec_ave','-v7.3');





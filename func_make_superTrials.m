function superTrialData = func_make_superTrials(data,avgNum)
% average a certain number of trials to get less noisy data
% data: eegata, chn*time*trials
% avgNum:average how many trials into one trial?

    newTrialNum = floor(size(data,3)/avgNum);
    temp = 1:newTrialNum;
    temp2 = repmat(temp, [avgNum 1]);
    avgIdx = temp2(:);
    data = data(:,:,randperm(size(data,3))); % shuffle the trials of the original data.
    data = data(:,:,1:length(avgIdx));
    superTrialData = zeros(size(data,1),size(data,2),length(unique(avgIdx)));
    for ii = unique(avgIdx)'
        temp = mean(data(:,:,avgIdx==ii),3);
        superTrialData(:,:,ii) = temp;
    end
    
end
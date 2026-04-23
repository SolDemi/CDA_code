function blocking = blocking(erp,i)
% function detects blinks in the VEM channel

%thresh = erp.arf.thresh(chanc);
winSize =  100; % size of test window, ms
winSize = round(winSize/erp.rateAcq); % size of test window, samples
winStep = 50; % size of step between windows, ms
winStep = round(winStep/erp.rateAcq); % size of step, samples
tolerance = 1;%the noise fluctuation from the celing and floor
blockdur = 50; % size of the blocking window, ms


rawTS = erp.data(i,:);
wInd = 1; blocking = zeros(1,size(erp.data,2));
while 1
    % determine portion of rawTS to test
    wEnd = wInd + winSize; 
    window = wInd:wEnd;

    minPeak = min(rawTS(window)); % minimum peak amplitude
    maxPeak = max(rawTS(window)); % maximum peak amplitude

    if sum(rawTS(window)>= maxPeak- tolerance) >= round(blockdur/4)
        blocking(window) = 1;
    elseif  sum(rawTS(window)<= minPeak+ tolerance) >= round(blockdur/4)
        blocking(window) = 1;
    end;


    wInd = wInd + winStep;

    if wInd + winSize > length(rawTS)
        break
    end
end

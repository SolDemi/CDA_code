function blink = ppa_vogel(erp,i)
% function detects blinks in the VEM channel

thresh = erp.arf.thresh(21);
winSize = 100; % size of test window, ms
winSize = round(winSize/erp.rateAcq); % size of test window, samples
winStep = 50; % size of step between windows, ms
winStep = round(winStep/erp.rateAcq); % size of step, samples

rawTS = erp.data(i,:);
wInd = 1; blink = zeros(1,size(erp.data,2));
while 1
    % determine portion of rawTS to test
    wEnd = wInd + winSize; 
    window = wInd:wEnd;
    
    minPeak = min(rawTS(window)); % minimum peak amplitude
    maxPeak = max(rawTS(window)); % maximum peak amplitude
    
    p2p = abs(maxPeak-minPeak); % peak to peak amplitude
    
    if p2p > thresh
        blink(window) = 1;
    end
    
    wInd = wInd + winStep;
    
    if wInd + winSize > length(rawTS)
        break
    end
end
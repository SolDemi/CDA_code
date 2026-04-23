function eMove = step_vogel(erp,i)

% function detects eye movements

thresh = erp.arf.thresh(22);
winSize = 100; % size of test window, ms
winSize = round(winSize/erp.rateAcq); % size of test window, samples
winStep = 50; % size of step between windows, ms
winStep = round(winStep/erp.rateAcq); % size of step, samples

rawTS = erp.data(i,:);
wInd = 1; eMove = zeros(1,size(erp.data,2));
while 1
    % determine portion of rawTS to test
    wEnd = wInd + winSize; 
    p = round(mean([wInd wEnd]));
    window = wInd:wEnd;
    
    prePeak = mean(rawTS(wInd:p)); % mean amplitude prior to pointer index
    postPeak = mean(rawTS(p:wEnd)); % mean amplitude after pointer index
    
    stepAmp = postPeak-prePeak; % peak to peak amplitude
    
    if abs(stepAmp) > thresh
        eMove(window) = 1;
    end
    
    wInd = wInd + winStep;
    
    if wInd + winSize > length(rawTS)
        break
    end
end
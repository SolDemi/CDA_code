function bandPowerNorm = calculate_hilbert_band_power(data, fs, times, baselineWindow, freq_band)
% Band-limited Hilbert power extraction with baseline correction.
%
% Inputs:
%   data              - [electrodes x time x trials]
%   fs                - sampling rate (Hz)
%   baseline_window   - [start, end] in ms
%   freq_band         - [low, high] in Hz, e.g. [8 12]
%
% Outputs:
%   bandPowerNorm     - [electrodes x time x trials] log-power minus
%                       baseline mean log-power
    

    baseIdx = times >= baselineWindow(1) & times <= baselineWindow(2);


    bandPower = nan(size(data));
    for c = 1:size(data,1)

        tmp = squeeze(data(c,:,:));
        [row,col] = size(tmp);       
        tmp = tmp(:)';
        tmp = abs( hilbert( eegfilt( tmp,fs,freq_band(1),freq_band(2) ) ) ).^2;
        % Convert power to decibel scale; add a small constant to avoid log(0)
        tmp = 10 * log10(tmp + 1e-20 );
        
        bandPower(c,:,:) = reshape(tmp,[row,col]);

    end


    
    bandPowerBaseline = mean(bandPower(:,baseIdx,:),2);
    bandPowerBaseline = repmat(bandPowerBaseline, [1, size( bandPower,2 ),1]);
    bandPowerNorm = bandPower - bandPowerBaseline;
    % bandPowerNorm = 100*(bandPower - bandPowerBaseline)./bandPowerBaseline;

    % bandPowerBaselineSD = std(bandPower(:,baseIdx,:),[],2);
    % 
    % % transform bandPower to z-score
    % bandPowerCorrected = ( bandPower - repmat( bandPowerBaseline, [1, size(bandPower, 2), 1] ) ) ./ repmat(bandPowerBaselineSD, [1, size(bandPower, 2), 1]); 



end

function hgp_norm = calculate_high_gamma_power(data, fs, times, baselineWindow, freq_band)
% High gamma power extraction (no log or smoothing), with vectorized filtering and Hilbert transform.
%
% Inputs:
%   data              - [electrodes x time x trials]

%   fs                - sampling rate (Hz)
%   baseline_window   - [start, end] in ms
%   freq_band         - [low, high] in Hz, e.g. [70 150]
%
% Outputs:
%   hgp               - [electrodes x time x trials] Z-scored high gamma
%   hgp_baseline      - [electrodes x baseline_len x trials] baseline segments
    

    baseIdx = times >= baselineWindow(1) & times <= baselineWindow(2);


    %  filter the data
    hgp = nan(size(data));
    for c = 1:size(data,1)

        tmp = squeeze(data(c,:,:));
        [row,col] = size(tmp);       
        tmp = tmp(:)';
        tmp = abs( hilbert( eegfilt( tmp,fs,freq_band(1),freq_band(2) ) ) ).^2;
        % Convert power to decibel scale; add a small constant to avoid log(0)
        tmp = 10 * log10(tmp + 1e-20 );
        
        hgp(c,:,:) = reshape(tmp,[row,col]);

    end


    
    hgpBaseline = mean(hgp(:,baseIdx,:),2);
    hgpBaseline = repmat(hgpBaseline, [1, size( hgp,2 ),1]);
    hgp_norm = hgp - hgpBaseline;
    % hgp_norm = 100*(hgp - hgpBaseline)./hgpBaseline;

    % hgpBaselineSD = std(hgp(:,baseIdx,:),[],2);
    % 
    % % transform hgp to z-score
    % hgpCorrected = ( hgp - repmat( hgpBaseline, [1, size(hgp, 2), 1] ) ) ./ repmat(hgpBaselineSD, [1, size(hgp, 2), 1]); 



end

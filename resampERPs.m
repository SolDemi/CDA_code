function [erp, command] = resampERPs( erp, freq)

% finding the best ratio
[p,q] = rat(freq/erp.srate, 0.0001); % not used right now 

% set variable
% ------------
erp.data = reshape(erp.data, erp.nChans, erp.pnts);
oldpnts   = erp.pnts;

% resample for multiple channels
% -------------------------
if isfield(erp, 'event') & isfield(erp.event, 'type') & isstr(erp.event(1).type)
    bounds = strmatch('boundary', { erp.event.type });
    if ~isempty(bounds),
        disp('Data break detected and taken into account for resampling');
        bounds = [ erp.event(bounds).latency ];
        if bounds(1) < 0, bounds(1) = []; end; % remove initial boundary if any
    end;
    bounds = [1 round(bounds-0.5)+1 size(erp.data,2)+1];
    bounds(find(bounds(2:end)-bounds(1:end-1)==0))=[]; % remove doublet boundary if any
else 
    bounds = [1 size(erp.data,2)+1]; % [1:size(erp.data,2):size(erp.data,2)*size(erp.data,3)+1];
end;
if exist('resample') == 2
     usesigproc = 1;
else usesigproc = 0;
    disp('Signal Processing Toolbox absent: using custom interpolation instead of resample() function.');
    disp('This method uses cubic spline interpolation after anti-aliasing (see >> help spline)');    
end;

fprintf('resampling data %3.4f Hz\n', erp.srate*p/q);
for index1 = 1:size(erp.data,1)
%     fprintf('%d ', index1);	
    sigtmp = squeeze(erp.data(index1,:));
    
    if index1 == 1
        tmpres = [];
        indices = [1];
        for ind = 1:length(bounds)-1
            tmpres  = [ tmpres; myresample( double( sigtmp(bounds(ind):bounds(ind+1)-1)), p, q, usesigproc ) ];
            indices = [ indices size(tmpres,1)+1 ];
        end;
        if size(tmpres,1) == 1, erp.pnts  = size(tmpres,2);
        else                    erp.pnts  = size(tmpres,1);
        end;
        tmpeeglab = zeros(erp.nChans, erp.pnts);
    else
        for ind = 1:length(bounds)-1
            tmpres(indices(ind):indices(ind+1)-1,:) = myresample( double( sigtmp(bounds(ind):bounds(ind+1)-1) ), p, q, usesigproc );
        end;
    end; 
    tmpeeglab(index1,:, :) = tmpres;
end;
% fprintf('\n');	
erp.srate   = erp.srate*p/q;
erp.data = tmpeeglab;

% recompute all event latencies
% -----------------------------
if erp.calib == 0
    if isfield(erp.event, 'latency')
        fprintf('resampling event latencies...\n');
        for index1 = 1:length(erp.event)
            erp.event(index1).latency = erp.event(index1).latency * erp.pnts /oldpnts;
        end;
        if isfield(erp, 'urevent') & isfield(erp.urevent, 'latency')
            try,
                for index1 = 1:length(erp.event)
                    erp.urevent(index1).latency = erp.urevent(index1).latency * erp.pnts /oldpnts;
                end;
            catch,
                disp('pop_resample warning: ''urevent'' problem, reinitializing urevents');
                erp = rmfield(erp, 'urevent');
            end;
        end;
    end;
end

% store dataset
fprintf('resampling finished\n');

erp.pnts    = size(erp.data,2);

command = sprintf('erp = pop_resample( %s, %d);', inputname(1), freq);
return;

% resample if resample is not present
% -----------------------------------
function tmpeeglab = myresample(data, pnts, new_pnts, usesigproc);
    
    if usesigproc
        tmpeeglab = resample(data, pnts, new_pnts);
        return;
    end;
    
    % anti-alias filter
    % -----------------
    data         = eegfiltfft(data', 256, 0, 128*pnts/new_pnts);
    
    % spline interpolation
    % --------------------
    X            = [1:length(data)];
    nbnewpoints  = length(data)*pnts/new_pnts;
    nbnewpoints2 = ceil(nbnewpoints);
    lastpointval = length(data)/nbnewpoints*nbnewpoints2;        
    XX = linspace( 1, lastpointval, nbnewpoints2);
    
    cs = spline( X, data);
    tmpeeglab = ppval(cs, XX)';

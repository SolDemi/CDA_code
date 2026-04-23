function erp = merge_erp_blocks(erp1, erp2)
% Merge two ERP structs from the same subject/session split across raw files.

    if erp1.srate ~= erp2.srate
        error('Sampling rates do not match.');
    end

    if erp1.nChans ~= erp2.nChans
        error('Channel counts do not match.');
    end

    if size(erp1.data,1) ~= size(erp2.data,1)
        error('Channel dimensions do not match.');
    end

    erp = erp1;

    n1 = size(erp1.data, 2);

    % Concatenate continuous signals
    erp.data = [erp1.data, erp2.data];
    erp.filtered_data = [erp1.filtered_data, erp2.filtered_data];

    % Concatenate artifact masks
    erp.arf.blink = [erp1.arf.blink, erp2.arf.blink];
    erp.arf.eMove = [erp1.arf.eMove, erp2.arf.eMove];
    erp.arf.blocking = [erp1.arf.blocking, erp2.arf.blocking];

    % Shift latencies of the second block
    event2 = erp2.event;
    for k = 1:numel(event2)
        event2(k).latency = event2(k).latency + n1;
    end

    % Insert a boundary event to prevent false cross-file trial pairing
    boundary_event = struct('type', -999, 'latency', n1);

    erp.event = [erp1.event, boundary_event, event2];

    erp.eventCodes = cell2mat({erp.event.type});
    erp.eventTimes = round(cell2mat({erp.event.latency}));
    erp.pnts = size(erp.data, 2);
end
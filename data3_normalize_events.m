function events = data3_normalize_events(eegEvents)

events = struct('type', {}, 'latency', {});
for i = 1:numel(eegEvents)
    if isnumeric(eegEvents(i).type)
        txt = sprintf('S %d', eegEvents(i).type);
    else
        txt = char(eegEvents(i).type);
    end
    events(i).type = regexprep(strtrim(txt), '\s+', ' ');
    events(i).latency = double(eegEvents(i).latency);
end
end

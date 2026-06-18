function trials = data3_setsize_trials(events, loadVal)
% Return valid data3 trial descriptors for set size 1, 3, or 6.

switch loadVal
    case 1
        finalMarker = 'S 41';
        itemMarkers = {};
    case 3
        finalMarker = 'S 42';
        itemMarkers = {'S 75', 'S 76'};
    case 6
        finalMarker = 'S 43';
        itemMarkers = {'S 75', 'S 76', 'S 77', 'S 78', 'S 79'};
    otherwise
        error('Unsupported data3 set size: %g.', loadVal);
end

trials = struct('side', {}, 'family', {}, 'firstLatency', {}, 'finalLatency', {}, ...
    'itemLatencies', {});

for evIdx = 1:numel(events)
    if ~strcmp(events(evIdx).type, finalMarker)
        continue;
    end

    prevBoundary = find(strcmp({events(1:evIdx).type}, 'S 20'), 1, 'last');
    if isempty(prevBoundary)
        prevBoundary = max(1, evIdx - 12);
    end

    cueIdx = [];
    for i = evIdx-1:-1:prevBoundary
        if strcmp(events(i).type, 'S 31') || strcmp(events(i).type, 'S 32')
            cueIdx = i;
            break;
        end
    end
    if isempty(cueIdx)
        continue;
    end

    itemLatencies = nan(1, loadVal);
    for ii = 1:numel(itemMarkers)
        itemIdx = find(strcmp({events(cueIdx+1:evIdx-1).type}, itemMarkers{ii}), 1);
        if ~isempty(itemIdx)
            itemLatencies(ii) = events(cueIdx + itemIdx).latency;
        end
    end
    itemLatencies(loadVal) = events(evIdx).latency;
    if any(isnan(itemLatencies))
        continue;
    end

    familyIdx = [];
    for i = evIdx+1:min(numel(events), evIdx+4)
        if strcmp(events(i).type, 'S 51') || strcmp(events(i).type, 'S 52')
            familyIdx = i;
            break;
        end
    end
    if isempty(familyIdx)
        continue;
    end

    t = numel(trials) + 1;
    if strcmp(events(cueIdx).type, 'S 31')
        trials(t).side = 'L';
    else
        trials(t).side = 'R';
    end
    if strcmp(events(familyIdx).type, 'S 51')
        trials(t).family = 'C';
    else
        trials(t).family = 'L';
    end
    trials(t).firstLatency = itemLatencies(1);
    trials(t).finalLatency = itemLatencies(loadVal);
    trials(t).itemLatencies = itemLatencies;
end
end

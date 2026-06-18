function idx = data3_chan_indices(allLabels, targetLabels)

idx = zeros(1, numel(targetLabels));
for i = 1:numel(targetLabels)
    idx(i) = find(strcmpi(allLabels, targetLabels{i}), 1);
end
idx = idx(idx > 0);
end

function labels = data3_chan_labels(EEG)

labels = cell(1, numel(EEG.chanlocs));
for i = 1:numel(EEG.chanlocs)
    labels{i} = strtrim(EEG.chanlocs(i).labels);
end
end

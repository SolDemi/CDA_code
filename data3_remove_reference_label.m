function labelsRef = data3_remove_reference_label(labels, refLabel)

refIdx = find(strcmpi(labels, refLabel), 1);
keepIdx = setdiff(1:numel(labels), refIdx, 'stable');
labelsRef = labels(keepIdx);
end

function subjectFilter = data3_subject_filter()
% Use DATA3_SUBJECT_FILTER when set; otherwise match the original paper.

filterText = strtrim(char(getenv('DATA3_SUBJECT_FILTER')));
subjectFilter = {};
if ~isempty(filterText)
    parts = regexp(filterText, '[,;\s]+', 'split');
    parts = parts(~cellfun('isempty', parts));
    subjectFilter = regexprep(parts, '^sub', '', 'ignorecase');
end

if isempty(subjectFilter)
    subjectFilter = data3_original_subjects();
end
end

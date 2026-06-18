function files = data3_filter_subject_mat_files(files, subjectFilter)

if isempty(subjectFilter)
    return;
end

keep = false(size(files));
for i = 1:numel(files)
    tok = regexp(files(i).name, '^sub(\d+)\.mat$', 'tokens', 'once');
    keep(i) = ~isempty(tok) && any(strcmp(tok{1}, subjectFilter));
end
files = files(keep);
end

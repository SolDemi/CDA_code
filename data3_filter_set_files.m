function setFiles = data3_filter_set_files(setFiles, subjectFilter)

if isempty(subjectFilter)
    return;
end

keep = false(size(setFiles));
for i = 1:numel(setFiles)
    [~, baseName] = fileparts(setFiles(i).name);
    tok = regexp(baseName, '^sub(\d+)_all$', 'tokens', 'once');
    keep(i) = ~isempty(tok) && any(strcmp(tok{1}, subjectFilter));
end
setFiles = setFiles(keep);
end

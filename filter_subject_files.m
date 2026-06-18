function [files, subjectIds, keep] = filter_subject_files(files, includeSubjectIds, excludeSubjectIds)
%FILTER_SUBJECT_FILES Filter dir() output by parsed numeric subject ids.

if nargin < 2 || isempty(includeSubjectIds)
    includeSubjectIds = [];
end
if nargin < 3 || isempty(excludeSubjectIds)
    excludeSubjectIds = [];
end

includeSubjectIds = includeSubjectIds(:);
excludeSubjectIds = excludeSubjectIds(:);

subjectIds = nan(numel(files), 1);
for i = 1:numel(files)
    subjectIds(i) = parse_subject_id_from_filename(files(i).name);
end

keep = true(numel(files), 1);
if ~isempty(includeSubjectIds)
    keep = keep & ismember(subjectIds, includeSubjectIds);
end
if ~isempty(excludeSubjectIds)
    keep = keep & ~ismember(subjectIds, excludeSubjectIds);
end

files = files(keep);
subjectIds = subjectIds(keep);
end

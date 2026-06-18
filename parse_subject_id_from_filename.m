function subjectId = parse_subject_id_from_filename(fileName)
%PARSE_SUBJECT_ID_FROM_FILENAME Extract numeric subject id from common files.
%
% Supported examples:
%   sub123.mat
%   123_EEG_timeLockMem.mat

[~, baseName] = fileparts(fileName);

tok = regexp(baseName, '^sub(\d+)$', 'tokens', 'once');
if isempty(tok)
    tok = regexp(baseName, '^(\d+)(?:_|$)', 'tokens', 'once');
end

if isempty(tok)
    subjectId = NaN;
else
    subjectId = str2double(tok{1});
end
end

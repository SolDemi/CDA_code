function [subjectIds, Inclusion] = data2_decoding_subjects(datadir, minTrialsPerCond)
%data2_DECODING_SUBJECTS Return data2 subject ids passing original criterion.

if nargin < 2 || isempty(minTrialsPerCond)
    minTrialsPerCond = 160;
end

files = dir(fullfile(datadir, 'sub*.mat'));
subject = nan(numel(files), 1);
included = false(numel(files), 1);
minTrialCount = nan(numel(files), 1);
trialCounts = nan(numel(files), 4);
file = strings(numel(files), 1);

for i = 1:numel(files)
    file(i) = string(fullfile(files(i).folder, files(i).name));
    subject(i) = parse_subject_id_from_filename(files(i).name);

    [included(i), info] = data2_subject_inclusion(fullfile(files(i).folder, files(i).name), minTrialsPerCond);
    minTrialCount(i) = info.minTrialCount;
    trialCounts(i,1:numel(info.trialsPerCond)) = info.trialsPerCond;
end

subjectIds = subject(included);
subjectIds = sort(subjectIds(:));

Inclusion = table(subject, included, minTrialCount, ...
    trialCounts(:,1), trialCounts(:,2), trialCounts(:,3), trialCounts(:,4), file, ...
    'VariableNames', {'subject', 'included', 'minTrialCount', ...
    'nC2', 'nC6', 'nS2', 'nS6', 'file'});
end

function [subjectIds, Inclusion] = data1_decoding_subjects(datadir, minTrialsPerSetSize)
%data1_DECODING_SUBJECTS Return data1 subject ids passing original criterion.

if nargin < 2 || isempty(minTrialsPerSetSize)
    minTrialsPerSetSize = 75;
end

files = dir(fullfile(datadir, '*_EEG_timeLockMem.mat'));
subject = nan(numel(files), 1);
included = false(numel(files), 1);
minTrialCount = nan(numel(files), 1);
trialCounts = nan(numel(files), 3);
file = strings(numel(files), 1);

for i = 1:numel(files)
    file(i) = string(fullfile(files(i).folder, files(i).name));
    subject(i) = parse_subject_id_from_filename(files(i).name);

    S = load(fullfile(files(i).folder, files(i).name), 'eeg');
    [included(i), info] = data1_subject_inclusion(S.eeg, minTrialsPerSetSize);
    minTrialCount(i) = info.minTrialCount;
    trialCounts(i,:) = info.trialCountsPerSetSize;
end

subjectIds = subject(included);
subjectIds = sort(subjectIds(:));

Inclusion = table(subject, included, minTrialCount, ...
    trialCounts(:,1), trialCounts(:,2), trialCounts(:,3), file, ...
    'VariableNames', {'subject', 'included', 'minTrialCount', ...
    'nSetSize1', 'nSetSize3', 'nSetSize6', 'file'});
end

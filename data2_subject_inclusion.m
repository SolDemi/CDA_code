function [include, info] = data1_subject_inclusion(cdaOrFile, minTrialsPerCond)
%DATA1_SUBJECT_INCLUSION Apply original load-decoding trial-count rule.
%
% Original data1 load-decoding scripts used subjects with
% grand.min_trials_per_cond >= 160, equivalent here to
% min(cda.trials_per_cond) >= 160 across C2/C6/S2/S6 condition pairs.

if nargin < 2 || isempty(minTrialsPerCond)
    minTrialsPerCond = 160;
end

if ischar(cdaOrFile) || isstring(cdaOrFile)
    S = load(char(cdaOrFile), 'cda');
    cda = S.cda;
else
    cda = cdaOrFile;
end

if isfield(cda, 'trials_per_cond') && ~isempty(cda.trials_per_cond)
    trialsPerCond = double(cda.trials_per_cond(:)');
elseif isfield(cda, 'min_trials_per_cond')
    trialsPerCond = double(cda.min_trials_per_cond);
else
    error('cda struct is missing trials_per_cond/min_trials_per_cond.');
end

include = min(trialsPerCond) >= minTrialsPerCond;

info = struct();
info.dataset = 'data1';
info.source = 'original OSF load-decoding scripts';
info.criterion = 'min(cda.trials_per_cond) >= 160';
info.minTrialsPerCond = minTrialsPerCond;
info.trialsPerCond = trialsPerCond;
info.minTrialCount = min(trialsPerCond);
info.include = include;
end

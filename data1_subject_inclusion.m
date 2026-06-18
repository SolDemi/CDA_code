function [include, info] = data0_subject_inclusion(eegOrArtifactInd, minTrialsPerSetSize)
%DATA0_SUBJECT_INCLUSION Apply Adam et al. Experiment 1 exclusion rule.
%
% Participants are included only if they have at least 75 artifact-free
% trials in every set-size condition after artifact rejection. In the saved
% data, set sizes 1/3/6 correspond to condition pairs 1+4, 2+5, and 3+6.

if nargin < 2 || isempty(minTrialsPerSetSize)
    minTrialsPerSetSize = 75;
end

if isstruct(eegOrArtifactInd)
    artifactInd = eegOrArtifactInd.arf.artifactInd;
else
    artifactInd = eegOrArtifactInd;
end

artifactInd = logical(artifactInd);
if size(artifactInd, 1) ~= 6 && size(artifactInd, 2) == 6
    artifactInd = artifactInd';
end
if size(artifactInd, 1) < 6
    error('data0 artifactInd must contain 6 conditions.');
end

setSizeConditionPairs = [1 4; 2 5; 3 6];
setSizes = [1 3 6];
trialCounts = zeros(1, numel(setSizes));

for si = 1:numel(setSizes)
    conds = setSizeConditionPairs(si,:);
    trialCounts(si) = sum(~artifactInd(conds(1),:)) + sum(~artifactInd(conds(2),:));
end

include = all(trialCounts >= minTrialsPerSetSize);

info = struct();
info.dataset = 'data0';
info.source = 'Adam et al. Experiment 1';
info.criterion = '>=75 artifact-free trials in every set-size condition';
info.minTrialsPerSetSize = minTrialsPerSetSize;
info.setSizes = setSizes;
info.setSizeConditionPairs = setSizeConditionPairs;
info.trialCountsPerSetSize = trialCounts;
info.minTrialCount = min(trialCounts);
info.include = include;
end

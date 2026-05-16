function [dataBal, labelsBal, idxBal, info] = balance_trials_by_label(data, labels, varargin)
% balance_trials_by_label
% Lightweight balancing by label vector or nTrials x nFactors joint labels.

trialDim = [];
nPerCell = [];
seed = [];
shuffleOutput = true;

for ii = 1:2:numel(varargin)
    switch lower(varargin{ii})
        case 'trialdim',      trialDim = varargin{ii+1};
        case 'npercell',      nPerCell = varargin{ii+1};
        case 'seed',          seed = varargin{ii+1};
        case 'shuffleoutput', shuffleOutput = varargin{ii+1};
    end
end

if ~isempty(seed), rng(seed, 'twister'); end
if isvector(labels), labels = labels(:); end
if isempty(trialDim)
    if isempty(data), trialDim = 1; else, trialDim = ndims(data); end
end

[grpKey, ~, grp] = unique(labels, 'rows');
counts = accumarray(grp, 1);
if isempty(nPerCell), nPerCell = min(counts); end

idxBal = nan(size(grpKey,1) * nPerCell, 1);
for gi = 1:size(grpKey,1)
    idx = find(grp == gi);
    idx = idx(randperm(numel(idx)));
    idxBal((gi-1)*nPerCell + (1:nPerCell)) = idx(1:nPerCell);
end

if shuffleOutput, idxBal = idxBal(randperm(numel(idxBal))); end

labelsBal = labels(idxBal,:);
if isempty(data)
    dataBal = [];
else
    subs = repmat({':'}, 1, ndims(data));
    subs{trialDim} = idxBal;
    dataBal = data(subs{:});
end

info = struct('idxBal', idxBal, 'groupKey', grpKey, 'nPerCell', nPerCell, ...
    'countsOriginal', counts, 'countsDropped', counts - nPerCell, 'trialDim', trialDim);
end

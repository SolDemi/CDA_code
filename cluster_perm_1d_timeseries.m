function stat = cluster_perm_1d_timeseries(data, times, cfg)
% cluster_perm_1d_timeseries
%
% Generic 1-D cluster-based permutation test for group-level time-series data.
%
% This function is intentionally independent of LDA/SVM/RSA implementation.
% It only requires a subject-by-time matrix. Therefore it can be used for:
%   - LDA/SVM decoding accuracy or AUC time courses, e.g. diag(AUC)
%   - RSA rho time courses
%   - power/ERP time courses
%   - paired model contrasts, after passing the subject-wise difference matrix
%
% USAGE
%   cfg = struct();
%   cfg.null = 0.5;          % chance level for AUC/accuracy; use 0 for RSA rho
%   cfg.nPerm = 5000;
%   cfg.tail = 'right';      % 'right', 'left', or 'two'
%   cfg.clusterAlpha = 0.05; % cluster-forming threshold
%   cfg.alpha = 0.05;        % cluster-corrected threshold
%   stat = cluster_perm_1d_timeseries(data, times, cfg);
%
% INPUT
%   data  : nSubject x nTime matrix. Rows are exchangeable observations.
%   times : 1 x nTime vector. Use [] to default to 1:nTime.
%   cfg   : configuration structure.
%
% IMPORTANT CFG FIELDS
%   cfg.null           : scalar or 1 x nTime null value, default = 0
%                        For decoding AUC/accuracy, set to 0.5.
%                        For RSA Spearman/Pearson rho, set to 0.
%                        For paired contrasts, pass conditionA-conditionB and set 0.
%   cfg.nPerm          : number of random sign-flip permutations, default = 5000
%   cfg.tail           : 'right', 'left', or 'two', default = 'right'
%   cfg.clusterAlpha   : sample-level cluster-forming alpha, default = 0.05
%   cfg.alpha          : cluster-level corrected alpha, default = 0.05
%   cfg.clusterStat    : 'mass' or 'size', default = 'mass'
%                        'mass' sums t-values within a cluster.
%                        'size' uses cluster length.
%   cfg.minClusterSize : minimum consecutive samples for a cluster, default = 1
%   cfg.randomSeed     : [], numeric seed, or 'shuffle', default = []
%   cfg.verbose        : true/false, default = true
%
% OUTPUT
%   stat.data                  : original data
%   stat.diff                  : data - null
%   stat.times                 : time vector
%   stat.mean                  : group mean of original data
%   stat.sem                   : group SEM of original data
%   stat.meanDiff              : group mean of data-null
%   stat.tObs                  : observed one-sample t-values across subjects
%   stat.pObs                  : uncorrected p-values
%   stat.clusterFormingMask    : uncorrected cluster-forming mask
%   stat.clusters              : all observed clusters with corrected p-values
%   stat.significantClusters   : clusters with p <= cfg.alpha
%   stat.significantMask       : 1 x nTime corrected significant mask
%   stat.maxClusterStatNull    : max cluster statistic from each permutation
%   stat.cfg                   : cfg after defaults
%
% STATISTICAL MODEL
%   One-sample sign-flipping test on data - cfg.null. This is appropriate when
%   subject-level effects are exchangeable around zero under H0.
%
% NOTES
%   - This is a 1-D time-series test. It does not perform 2-D time-frequency
%     or train-time x test-time clustering.
%   - For 2-D clustering, the connected-component step should be generalized
%     to image/grid connectivity.

if nargin < 2 || isempty(times)
    times = [];
end
if nargin < 3 || isempty(cfg)
    cfg = struct();
end

validateattributes(data, {'numeric'}, {'2d', 'nonempty'}, mfilename, 'data', 1);
[nSubj, nTime] = size(data);

if isempty(times)
    times = 1:nTime;
end
times = times(:)';
if numel(times) ~= nTime
    error('times must have the same number of elements as size(data, 2).');
end

cfg = fill_default_cfg(cfg, nTime);

if ~isempty(cfg.randomSeed)
    if ischar(cfg.randomSeed) || isstring(cfg.randomSeed)
        rng(char(cfg.randomSeed));
    else
        rng(cfg.randomSeed);
    end
end

nullValue = cfg.null;
if isscalar(nullValue)
    nullValue = repmat(nullValue, 1, nTime);
else
    nullValue = nullValue(:)';
    if numel(nullValue) ~= nTime
        error('cfg.null must be scalar or a 1 x nTime vector.');
    end
end

D = bsxfun(@minus, data, nullValue);

% Observed statistics.
[tObs, pObs, dfObs] = one_sample_t(D, cfg.tail);
clusterMask = cluster_forming_mask(tObs, pObs, cfg.clusterAlpha, cfg.tail);
obsClusters = find_clusters_1d(clusterMask, cfg.minClusterSize);
obsClusterStats = compute_cluster_stats(tObs, obsClusters, cfg.tail, cfg.clusterStat);

% Random sign-flipping permutations.
maxClusterStatNull = zeros(cfg.nPerm, 1);
for pi = 1:cfg.nPerm
    signs = (randi([0 1], nSubj, 1) * 2) - 1;
    Dp = bsxfun(@times, D, signs);

    [tPerm, pPerm] = one_sample_t(Dp, cfg.tail);
    permMask = cluster_forming_mask(tPerm, pPerm, cfg.clusterAlpha, cfg.tail);
    permClusters = find_clusters_1d(permMask, cfg.minClusterSize);
    permClusterStats = compute_cluster_stats(tPerm, permClusters, cfg.tail, cfg.clusterStat);

    if isempty(permClusterStats)
        maxClusterStatNull(pi) = 0;
    else
        maxClusterStatNull(pi) = max(permClusterStats);
    end
end

% Correct observed clusters against max-cluster null distribution.
clusters = struct('idx', {}, 'startIdx', {}, 'endIdx', {}, 'startTime', {}, ...
    'endTime', {}, 'clusterStat', {}, 'p', {}, 'nSamples', {});
significantClusters = clusters;
significantMask = false(1, nTime);
clusterP = nan(1, numel(obsClusters));

for ci = 1:numel(obsClusters)
    idx = obsClusters{ci};
    clusterP(ci) = (1 + sum(maxClusterStatNull >= obsClusterStats(ci))) / (cfg.nPerm + 1);

    clusters(ci).idx = idx; %#ok<AGROW>
    clusters(ci).startIdx = idx(1); %#ok<AGROW>
    clusters(ci).endIdx = idx(end); %#ok<AGROW>
    clusters(ci).startTime = times(idx(1)); %#ok<AGROW>
    clusters(ci).endTime = times(idx(end)); %#ok<AGROW>
    clusters(ci).clusterStat = obsClusterStats(ci); %#ok<AGROW>
    clusters(ci).p = clusterP(ci); %#ok<AGROW>
    clusters(ci).nSamples = numel(idx); %#ok<AGROW>

    if clusterP(ci) <= cfg.alpha
        significantMask(idx) = true;
        significantClusters(end+1) = clusters(ci); %#ok<AGROW>
    end
end

stat = struct();
stat.data = data;
stat.diff = D;
stat.null = nullValue;
stat.times = times;
stat.nSubj = nSubj;
stat.nTime = nTime;
stat.mean = mean(data, 1, 'omitnan');
stat.sem = std(data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data), 1));
stat.meanDiff = mean(D, 1, 'omitnan');
stat.tObs = tObs;
stat.pObs = pObs;
stat.dfObs = dfObs;
stat.clusterFormingMask = clusterMask;
stat.clusters = clusters;
stat.clusterP = clusterP;
stat.significantClusters = significantClusters;
stat.significantMask = significantMask;
stat.maxClusterStatNull = maxClusterStatNull;
stat.cfg = cfg;

if cfg.verbose
    fprintf('cluster_perm_1d_timeseries: n=%d, nTime=%d, nPerm=%d, significant clusters=%d\n', ...
        nSubj, nTime, cfg.nPerm, numel(significantClusters));
    for ci = 1:numel(significantClusters)
        fprintf('  Cluster %d: %.3g to %.3g, p=%.4f, %s=%.4f\n', ...
            ci, significantClusters(ci).startTime, significantClusters(ci).endTime, ...
            significantClusters(ci).p, cfg.clusterStat, significantClusters(ci).clusterStat);
    end
end

end

%% ========================================================================
function cfg = fill_default_cfg(cfg, nTime)
if ~isfield(cfg, 'null'),           cfg.null = 0; end
if ~isfield(cfg, 'nPerm'),          cfg.nPerm = 5000; end
if ~isfield(cfg, 'tail'),           cfg.tail = 'right'; end
if ~isfield(cfg, 'clusterAlpha'),   cfg.clusterAlpha = 0.05; end
if ~isfield(cfg, 'alpha'),          cfg.alpha = 0.05; end
if ~isfield(cfg, 'clusterStat'),    cfg.clusterStat = 'mass'; end
if ~isfield(cfg, 'minClusterSize'), cfg.minClusterSize = 1; end
if ~isfield(cfg, 'randomSeed'),     cfg.randomSeed = []; end
if ~isfield(cfg, 'verbose'),        cfg.verbose = true; end

if ~isscalar(cfg.nPerm) || cfg.nPerm < 1 || cfg.nPerm ~= round(cfg.nPerm)
    error('cfg.nPerm must be a positive integer.');
end
if ~ismember(lower(cfg.tail), {'right','left','two'})
    error('cfg.tail must be ''right'', ''left'', or ''two''.');
end
if ~ismember(lower(cfg.clusterStat), {'mass','size'})
    error('cfg.clusterStat must be ''mass'' or ''size''.');
end
if ~isscalar(cfg.clusterAlpha) || cfg.clusterAlpha <= 0 || cfg.clusterAlpha >= 1
    error('cfg.clusterAlpha must be between 0 and 1.');
end
if ~isscalar(cfg.alpha) || cfg.alpha <= 0 || cfg.alpha >= 1
    error('cfg.alpha must be between 0 and 1.');
end
if ~isscalar(cfg.minClusterSize) || cfg.minClusterSize < 1 || cfg.minClusterSize ~= round(cfg.minClusterSize)
    error('cfg.minClusterSize must be a positive integer.');
end

if ~isscalar(cfg.null) && numel(cfg.null) ~= nTime
    error('cfg.null must be scalar or length nTime.');
end
end

%% ========================================================================
function [tVals, pVals, df] = one_sample_t(D, tail)
% One-sample t-test across rows, with NaN handling.
n = sum(~isnan(D), 1);
m = mean(D, 1, 'omitnan');
s = std(D, 0, 1, 'omitnan');
se = s ./ sqrt(n);

tVals = m ./ se;
tVals(se == 0 | n < 2 | isnan(tVals)) = 0;

df = n - 1;
pVals = ones(size(tVals));
valid = df > 0;

switch lower(tail)
    case 'right'
        pVals(valid) = 1 - tcdf(tVals(valid), df(valid));
    case 'left'
        pVals(valid) = tcdf(tVals(valid), df(valid));
    case 'two'
        pVals(valid) = 2 * (1 - tcdf(abs(tVals(valid)), df(valid)));
end

pVals(~valid | isnan(pVals)) = 1;
end

%% ========================================================================
function mask = cluster_forming_mask(tVals, pVals, clusterAlpha, tail)
switch lower(tail)
    case 'right'
        mask = pVals < clusterAlpha & tVals > 0;
    case 'left'
        mask = pVals < clusterAlpha & tVals < 0;
    case 'two'
        mask = pVals < clusterAlpha;
end
mask(isnan(mask)) = false;
end

%% ========================================================================
function clusters = find_clusters_1d(mask, minClusterSize)
idx = find(mask(:)');
clusters = {};
if isempty(idx)
    return;
end

breaks = [0, find(diff(idx) > 1), numel(idx)];
for bi = 1:(numel(breaks)-1)
    thisIdx = idx((breaks(bi)+1):breaks(bi+1));
    if numel(thisIdx) >= minClusterSize
        clusters{end+1} = thisIdx; %#ok<AGROW>
    end
end
end

%% ========================================================================
function clusterStats = compute_cluster_stats(tVals, clusters, tail, clusterStat)
clusterStats = nan(1, numel(clusters));
for ci = 1:numel(clusters)
    idx = clusters{ci};
    switch lower(clusterStat)
        case 'mass'
            switch lower(tail)
                case 'right'
                    clusterStats(ci) = sum(tVals(idx));
                case 'left'
                    clusterStats(ci) = sum(-tVals(idx));
                case 'two'
                    clusterStats(ci) = sum(abs(tVals(idx)));
            end
        case 'size'
            clusterStats(ci) = numel(idx);
    end
end
end

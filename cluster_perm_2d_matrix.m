function stat = cluster_perm_2d_matrix(data, xTimes, yTimes, cfg)
% data: nSub x nX x nY
% xTimes: train-time vector
% yTimes: test-time vector

if nargin < 4 || isempty(cfg), cfg = struct(); end

if ~isfield(cfg, 'null'), cfg.null = 0; end
if ~isfield(cfg, 'nPerm'), cfg.nPerm = 2000; end
if ~isfield(cfg, 'tail'), cfg.tail = 'right'; end
if ~isfield(cfg, 'clusterAlpha'), cfg.clusterAlpha = 0.05; end
if ~isfield(cfg, 'alpha'), cfg.alpha = 0.05; end
if ~isfield(cfg, 'clusterStat'), cfg.clusterStat = 'mass'; end
if ~isfield(cfg, 'minClusterSize'), cfg.minClusterSize = 1; end
if ~isfield(cfg, 'randomSeed'), cfg.randomSeed = []; end
if ~isfield(cfg, 'verbose'), cfg.verbose = true; end

[nSub, nX, nY] = size(data);

if isempty(xTimes), xTimes = 1:nX; end
if isempty(yTimes), yTimes = 1:nY; end
xTimes = xTimes(:)';
yTimes = yTimes(:)';

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed, 'twister');
end

if isscalar(cfg.null)
    D = data - cfg.null;
else
    D = data - reshape(cfg.null, 1, nX, nY);
end

[tObs, pObs] = one_sample_t_2d(D, cfg.tail);
obsMask = cluster_forming_mask_2d(tObs, pObs, cfg.clusterAlpha, cfg.tail);
obsClusters = find_clusters_2d(obsMask, cfg.minClusterSize);
obsStats = cluster_stats_2d(tObs, obsClusters, cfg.tail, cfg.clusterStat);

maxStatNull = zeros(cfg.nPerm,1);

for pi = 1:cfg.nPerm
    signs = (randi([0 1], nSub, 1) * 2) - 1;
    Dp = D .* reshape(signs, nSub, 1, 1);

    [tPerm, pPerm] = one_sample_t_2d(Dp, cfg.tail);
    permMask = cluster_forming_mask_2d(tPerm, pPerm, cfg.clusterAlpha, cfg.tail);
    permClusters = find_clusters_2d(permMask, cfg.minClusterSize);
    permStats = cluster_stats_2d(tPerm, permClusters, cfg.tail, cfg.clusterStat);

    if isempty(permStats)
        maxStatNull(pi) = 0;
    else
        maxStatNull(pi) = max(permStats);
    end
end

clusters = struct('idx', {}, 'clusterStat', {}, 'p', {}, 'nPixels', {});
significantClusters = clusters;
significantMask = false(nX, nY);
clusterP = nan(1, numel(obsClusters));

for ci = 1:numel(obsClusters)
    idx = obsClusters{ci};
    clusterP(ci) = (1 + sum(maxStatNull >= obsStats(ci))) / (cfg.nPerm + 1);

    clusters(ci).idx = idx; %#ok<AGROW>
    clusters(ci).clusterStat = obsStats(ci);
    clusters(ci).p = clusterP(ci);
    clusters(ci).nPixels = numel(idx);

    if clusterP(ci) <= cfg.alpha
        significantMask(idx) = true;
        significantClusters(end+1) = clusters(ci); %#ok<AGROW>
    end
end

stat = struct();
stat.data = data;
stat.diff = D;
stat.mean = squeeze(mean(data, 1, 'omitnan'));
stat.meanDiff = squeeze(mean(D, 1, 'omitnan'));
stat.tObs = tObs;
stat.pObs = pObs;
stat.clusterFormingMask = obsMask;
stat.clusters = clusters;
stat.clusterP = clusterP;
stat.significantClusters = significantClusters;
stat.significantMask = significantMask;
stat.maxClusterStatNull = maxStatNull;
stat.xTimes = xTimes;
stat.yTimes = yTimes;
stat.cfg = cfg;

if cfg.verbose
    fprintf('cluster_perm_2d_matrix: n=%d, size=%d x %d, sig clusters=%d\n', ...
        nSub, nX, nY, numel(significantClusters));
end
end

%% ========================================================================
function [tVals, pVals] = one_sample_t_2d(D, tail)
n = squeeze(sum(~isnan(D), 1));
m = squeeze(mean(D, 1, 'omitnan'));
s = squeeze(std(D, 0, 1, 'omitnan'));
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
function mask = cluster_forming_mask_2d(tVals, pVals, clusterAlpha, tail)
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
function clusters = find_clusters_2d(mask, minClusterSize)
CC = bwconncomp(mask, 4);
clusters = {};

for ci = 1:CC.NumObjects
    idx = CC.PixelIdxList{ci};
    if numel(idx) >= minClusterSize
        clusters{end+1} = idx; %#ok<AGROW>
    end
end
end

%% ========================================================================
function stats = cluster_stats_2d(tVals, clusters, tail, clusterStat)
stats = nan(1, numel(clusters));

for ci = 1:numel(clusters)
    idx = clusters{ci};

    switch lower(clusterStat)
        case 'mass'
            switch lower(tail)
                case 'right'
                    stats(ci) = sum(tVals(idx));
                case 'left'
                    stats(ci) = sum(-tVals(idx));
                case 'two'
                    stats(ci) = sum(abs(tVals(idx)));
            end

        case 'size'
            stats(ci) = numel(idx);
    end
end
end
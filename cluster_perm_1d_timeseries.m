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

if ~isfield(cfg, 'null'),           cfg.null = 0; end
if ~isfield(cfg, 'nPerm'),          cfg.nPerm = 1000; end
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

%% Observed statistics
n = sum(~isnan(D), 1);
m = mean(D, 1, 'omitnan');
s = std(D, 0, 1, 'omitnan');
se = s ./ sqrt(n);

tObs = m ./ se;
tObs(se == 0 | n < 2 | isnan(tObs)) = 0;

dfObs = n - 1;
pObs = ones(size(tObs));
valid = dfObs > 0;

switch lower(cfg.tail)
    case 'right'
        pObs(valid) = 1 - tcdf(tObs(valid), dfObs(valid));
    case 'left'
        pObs(valid) = tcdf(tObs(valid), dfObs(valid));
    case 'two'
        pObs(valid) = 2 * (1 - tcdf(abs(tObs(valid)), dfObs(valid)));
end
pObs(~valid | isnan(pObs)) = 1;

switch lower(cfg.tail)
    case 'right'
        clusterMask = pObs < cfg.clusterAlpha & tObs > 0;
    case 'left'
        clusterMask = pObs < cfg.clusterAlpha & tObs < 0;
    case 'two'
        clusterMask = pObs < cfg.clusterAlpha;
end
clusterMask(isnan(clusterMask)) = false;

idx = find(clusterMask(:)');
obsClusters = {};
if ~isempty(idx)
    breaks = [0, find(diff(idx) > 1), numel(idx)];
    for bi = 1:(numel(breaks)-1)
        thisIdx = idx((breaks(bi)+1):breaks(bi+1));
        if numel(thisIdx) >= cfg.minClusterSize
            obsClusters{end+1} = thisIdx; %#ok<AGROW>
        end
    end
end

obsClusterStats = nan(1, numel(obsClusters));
for ci = 1:numel(obsClusters)
    idx = obsClusters{ci};
    switch lower(cfg.clusterStat)
        case 'mass'
            switch lower(cfg.tail)
                case 'right'
                    obsClusterStats(ci) = sum(tObs(idx));
                case 'left'
                    obsClusterStats(ci) = sum(-tObs(idx));
                case 'two'
                    obsClusterStats(ci) = sum(abs(tObs(idx)));
            end
        case 'size'
            obsClusterStats(ci) = numel(idx);
    end
end

%% Random sign-flipping permutations
maxClusterStatNull = zeros(cfg.nPerm, 1);
for pi = 1:cfg.nPerm
    signs = (randi([0 1], nSubj, 1) * 2) - 1;
    Dp = bsxfun(@times, D, signs);

    n = sum(~isnan(Dp), 1);
    m = mean(Dp, 1, 'omitnan');
    s = std(Dp, 0, 1, 'omitnan');
    se = s ./ sqrt(n);

    tPerm = m ./ se;
    tPerm(se == 0 | n < 2 | isnan(tPerm)) = 0;

    dfPerm = n - 1;
    pPerm = ones(size(tPerm));
    valid = dfPerm > 0;

    switch lower(cfg.tail)
        case 'right'
            pPerm(valid) = 1 - tcdf(tPerm(valid), dfPerm(valid));
        case 'left'
            pPerm(valid) = tcdf(tPerm(valid), dfPerm(valid));
        case 'two'
            pPerm(valid) = 2 * (1 - tcdf(abs(tPerm(valid)), dfPerm(valid)));
    end
    pPerm(~valid | isnan(pPerm)) = 1;

    switch lower(cfg.tail)
        case 'right'
            permMask = pPerm < cfg.clusterAlpha & tPerm > 0;
        case 'left'
            permMask = pPerm < cfg.clusterAlpha & tPerm < 0;
        case 'two'
            permMask = pPerm < cfg.clusterAlpha;
    end
    permMask(isnan(permMask)) = false;

    idx = find(permMask(:)');
    permClusters = {};
    if ~isempty(idx)
        breaks = [0, find(diff(idx) > 1), numel(idx)];
        for bi = 1:(numel(breaks)-1)
            thisIdx = idx((breaks(bi)+1):breaks(bi+1));
            if numel(thisIdx) >= cfg.minClusterSize
                permClusters{end+1} = thisIdx; %#ok<AGROW>
            end
        end
    end

    permClusterStats = nan(1, numel(permClusters));
    for ci = 1:numel(permClusters)
        idx = permClusters{ci};
        switch lower(cfg.clusterStat)
            case 'mass'
                switch lower(cfg.tail)
                    case 'right'
                        permClusterStats(ci) = sum(tPerm(idx));
                    case 'left'
                        permClusterStats(ci) = sum(-tPerm(idx));
                    case 'two'
                        permClusterStats(ci) = sum(abs(tPerm(idx)));
                end
            case 'size'
                permClusterStats(ci) = numel(idx);
        end
    end

    if isempty(permClusterStats)
        maxClusterStatNull(pi) = 0;
    else
        maxClusterStatNull(pi) = max(permClusterStats);
    end
end

%% Cluster-level correction and saving results
clusters = struct('idx', {}, 'startIdx', {}, 'endIdx', {}, 'startTime', {}, ...
    'endTime', {}, 'clusterStat', {}, 'p', {}, 'nSamples', {});
significantClusters = clusters;
significantMask = false(1, nTime);
clusterP = nan(1, numel(obsClusters));

for ci = 1:numel(obsClusters)
    idx = obsClusters{ci};
    clusterP(ci) = (1 + sum(maxClusterStatNull >= obsClusterStats(ci))) / (cfg.nPerm + 1);

    clusters(ci).idx = idx;
    clusters(ci).startIdx = idx(1);
    clusters(ci).endIdx = idx(end);
    clusters(ci).startTime = times(idx(1));
    clusters(ci).endTime = times(idx(end));
    clusters(ci).clusterStat = obsClusterStats(ci);
    clusters(ci).p = clusterP(ci);
    clusters(ci).nSamples = numel(idx);

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

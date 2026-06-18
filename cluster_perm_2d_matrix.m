function stat = cluster_perm_2d_matrix(data, xTimes, yTimes, design, cfg)
% data: nSub x nX x nY
% xTimes: train-time vector
% yTimes: test-time vector
% design: optional temporal design for segment-block cluster connectivity

if nargin < 4 || isempty(design)
    design = [];
    cfg = struct();
elseif nargin < 5
    cfg = design;
    design = [];
elseif isempty(cfg)
    cfg = struct();
end

if ~isfield(cfg, 'null'), cfg.null = 0; end
if ~isfield(cfg, 'nPerm'), cfg.nPerm = 2000; end
if ~isfield(cfg, 'tail'), cfg.tail = 'right'; end
if ~isfield(cfg, 'clusterAlpha'), cfg.clusterAlpha = 0.05; end
if ~isfield(cfg, 'alpha'), cfg.alpha = 0.05; end
if ~isfield(cfg, 'clusterStat'), cfg.clusterStat = 'mass'; end
if ~isfield(cfg, 'minClusterSize'), cfg.minClusterSize = 1; end
if ~isfield(cfg, 'clusterConnectivity')
    if isempty(design)
        cfg.clusterConnectivity = 'continuous';
    else
        cfg.clusterConnectivity = 'withinSegmentBlocks';
    end
end
if ~isfield(cfg, 'randomSeed'), cfg.randomSeed = []; end
if ~isfield(cfg, 'verbose'), cfg.verbose = true; end

[nSub, nX, nY] = size(data);

if isempty(xTimes), xTimes = 1:nX; end
if isempty(yTimes), yTimes = 1:nY; end
xTimes = xTimes(:)';
yTimes = yTimes(:)';

switch lower(cfg.clusterConnectivity)
    case 'continuous'
        clusterBlockId = ones(nX, nY);
    case 'withinsegmentblocks'
        if isempty(design) || ~isfield(design, 'lowWindowsMs') || ...
                ~isfield(design, 'highSegmentStartsMs') || ~isfield(design, 'highSegmentWidthMs')
            error('cluster_perm_2d_matrix:MissingDesign', ...
                'Segment-block cluster connectivity requires temporal design fields.');
        end

        xSeg = nan(numel(xTimes), 1);
        for wi = 1:size(design.lowWindowsMs, 1)
            inWin = xTimes(:) >= design.lowWindowsMs(wi,1) & xTimes(:) <= design.lowWindowsMs(wi,2);
            xSeg(inWin) = wi;
        end

        highWindows = [design.highSegmentStartsMs(:), ...
            design.highSegmentStartsMs(:) + design.highSegmentWidthMs];
        ySeg = nan(numel(yTimes), 1);
        for wi = 1:size(highWindows, 1)
            inWin = yTimes(:) >= highWindows(wi,1) & yTimes(:) <= highWindows(wi,2);
            ySeg(inWin) = wi;
        end

        if any(isnan(xSeg)) || any(isnan(ySeg))
            error('cluster_perm_2d_matrix:TimeSegmentMismatch', ...
                'Some time points could not be assigned to segment windows.');
        end

        clusterBlockId = nan(nX, nY);
        for xi = 1:nX
            for yi = 1:nY
                clusterBlockId(xi,yi) = xSeg(xi) + (ySeg(yi) - 1) * size(design.lowWindowsMs, 1);
            end
        end
    otherwise
        error('cluster_perm_2d_matrix:UnknownConnectivity', ...
            'Unknown clusterConnectivity: %s.', cfg.clusterConnectivity);
end

if ~isempty(cfg.randomSeed)
    rng(cfg.randomSeed, 'twister');
end

if isscalar(cfg.null)
    D = data - cfg.null;
else
    D = data - reshape(cfg.null, 1, nX, nY);
end

%% Observed statistics
n = squeeze(sum(~isnan(D), 1));
m = squeeze(mean(D, 1, 'omitnan'));
s = squeeze(std(D, 0, 1, 'omitnan'));
se = s ./ sqrt(n);

tObs = m ./ se;
tObs(se == 0 | n < 2 | isnan(tObs)) = 0;

df = n - 1;
pObs = ones(size(tObs));
valid = df > 0;

switch lower(cfg.tail)
    case 'right'
        pObs(valid) = 1 - tcdf(tObs(valid), df(valid));
    case 'left'
        pObs(valid) = tcdf(tObs(valid), df(valid));
    case 'two'
        pObs(valid) = 2 * (1 - tcdf(abs(tObs(valid)), df(valid)));
end
pObs(~valid | isnan(pObs)) = 1;

switch lower(cfg.tail)
    case 'right'
        obsMask = pObs < cfg.clusterAlpha & tObs > 0;
    case 'left'
        obsMask = pObs < cfg.clusterAlpha & tObs < 0;
    case 'two'
        obsMask = pObs < cfg.clusterAlpha;
end
obsMask(isnan(obsMask)) = false;

obsClusters = {};
blockVals = unique(clusterBlockId(~isnan(clusterBlockId)));
for bi = 1:numel(blockVals)
    CC = bwconncomp(obsMask & (clusterBlockId == blockVals(bi)), 4);
    for ccIdx = 1:CC.NumObjects
        idx = CC.PixelIdxList{ccIdx};
        if numel(idx) >= cfg.minClusterSize
            obsClusters{end+1} = idx; %#ok<AGROW>
        end
    end
end

obsStats = nan(1, numel(obsClusters));
for ci = 1:numel(obsClusters)
    idx = obsClusters{ci};
    switch lower(cfg.clusterStat)
        case 'mass'
            switch lower(cfg.tail)
                case 'right'
                    obsStats(ci) = sum(tObs(idx));
                case 'left'
                    obsStats(ci) = sum(-tObs(idx));
                case 'two'
                    obsStats(ci) = sum(abs(tObs(idx)));
            end
        case 'size'
            obsStats(ci) = numel(idx);
    end
end

%% Random sign-flipping permutations
maxStatNull = zeros(cfg.nPerm,1);

for pi = 1:cfg.nPerm
    signs = (randi([0 1], nSub, 1) * 2) - 1;
    Dp = D .* reshape(signs, nSub, 1, 1);

    n = squeeze(sum(~isnan(Dp), 1));
    m = squeeze(mean(Dp, 1, 'omitnan'));
    s = squeeze(std(Dp, 0, 1, 'omitnan'));
    se = s ./ sqrt(n);

    tPerm = m ./ se;
    tPerm(se == 0 | n < 2 | isnan(tPerm)) = 0;

    df = n - 1;
    pPerm = ones(size(tPerm));
    valid = df > 0;

    switch lower(cfg.tail)
        case 'right'
            pPerm(valid) = 1 - tcdf(tPerm(valid), df(valid));
        case 'left'
            pPerm(valid) = tcdf(tPerm(valid), df(valid));
        case 'two'
            pPerm(valid) = 2 * (1 - tcdf(abs(tPerm(valid)), df(valid)));
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

    permClusters = {};
    for bi = 1:numel(blockVals)
        CC = bwconncomp(permMask & (clusterBlockId == blockVals(bi)), 4);
        for ccIdx = 1:CC.NumObjects
            idx = CC.PixelIdxList{ccIdx};
            if numel(idx) >= cfg.minClusterSize
                permClusters{end+1} = idx; %#ok<AGROW>
            end
        end
    end

    permStats = nan(1, numel(permClusters));
    for ci = 1:numel(permClusters)
        idx = permClusters{ci};
        switch lower(cfg.clusterStat)
            case 'mass'
                switch lower(cfg.tail)
                    case 'right'
                        permStats(ci) = sum(tPerm(idx));
                    case 'left'
                        permStats(ci) = sum(-tPerm(idx));
                    case 'two'
                        permStats(ci) = sum(abs(tPerm(idx)));
                end
            case 'size'
                permStats(ci) = numel(idx);
        end
    end

    if isempty(permStats)
        maxStatNull(pi) = 0;
    else
        maxStatNull(pi) = max(permStats);
    end
end

%% Cluster-level correction and saving results
clusters = struct('idx', {}, 'clusterStat', {}, 'p', {}, 'nPixels', {});
significantClusters = clusters;
significantMask = false(nX, nY);
clusterP = nan(1, numel(obsClusters));

for ci = 1:numel(obsClusters)
    idx = obsClusters{ci};
    clusterP(ci) = (1 + sum(maxStatNull >= obsStats(ci))) / (cfg.nPerm + 1);

    clusters(ci).idx = idx;
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
stat.design = design;
stat.clusterBlockId = clusterBlockId;
stat.cfg = cfg;

if cfg.verbose
    fprintf('cluster_perm_2d_matrix: n=%d, size=%d x %d, sig clusters=%d\n', ...
        nSub, nX, nY, numel(significantClusters));
end
end

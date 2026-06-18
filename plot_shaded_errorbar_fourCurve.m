function h = plot_shaded_errorbar_fourCurve(xData, varargin)
% plot_shaded_errorbar_fourCurve
%
% Backward-compatible shaded-error plotter. The original 4-curve call still
% works, and the newer call accepts any number of curves:
%
%   h = plot_shaded_errorbar_fourCurve(xData, yDataCell, nCurve, ...
%       xLimits, yLimits, xLabel, yLabel, legendLabels, colors, sigCfg)
%
% where yDataCell is a 1 x nCurve cell array of subject x time matrices.
%
% Optional sigCfg fields:
%   modelNames           : 1 x nCurve labels matching statistic field names
%   againstShuffleStats  : struct/cell of cluster_perm_1d_timeseries outputs;
%                          drawn below the original y-axis lower limit
%   pairwiseStats        : struct of pairwise cluster_perm_1d_timeseries outputs
%   pairInfo             : nPair x 3 cell array, {modelA, modelB, statName};
%                          drawn above the original y-axis upper limit
%   baselineLabels       : struct/cell of labels for againstShuffleStats rows
%   alpha                : cluster-level alpha used for star labels

if nargin >= 3 && iscell(varargin{1}) && isnumeric(varargin{2})
    yDataCell = varargin{1};
    nCurve = varargin{2};
    argOffset = 2;
else
    if nargin < 11
        error('Use either the legacy 4-curve signature or the cell-array multi-curve signature.');
    end
    yDataCell = varargin(1:4);
    nCurve = 4;
    argOffset = 4;
end

if numel(yDataCell) < nCurve
    error('yDataCell must contain at least nCurve matrices.');
end

xLimits = get_optional_arg(varargin, argOffset + 1, []);
yLimits = get_optional_arg(varargin, argOffset + 2, []);
myXlabel = get_optional_arg(varargin, argOffset + 3, 'Time (ms)');
myYlabel = get_optional_arg(varargin, argOffset + 4, 'Value');
myLegend = get_optional_arg(varargin, argOffset + 5, {});
myColor = get_optional_arg(varargin, argOffset + 6, []);
sigCfg = get_optional_arg(varargin, argOffset + 7, struct());

if isempty(myColor)
    myColor = lines(nCurve);
end
if size(myColor, 1) < nCurve || size(myColor, 2) ~= 3
    error('myColor must be an nCurve x 3 RGB matrix.');
end

xData = xData(:)';
mainLines = gobjects(nCurve, 1);
patches = gobjects(nCurve, 1);
edges = cell(nCurve, 1);
shaded = cell(nCurve, 1);

hold on;

for ci = 1:nCurve
    yData = yDataCell{ci};
    if size(yData, 2) ~= numel(xData)
        error('Curve %d has %d time points, but xData has %d.', ...
            ci, size(yData, 2), numel(xData));
    end

    nValid = sum(~isnan(yData), 1);
    yMean = mean(yData, 1, 'omitnan');
    ySE = std(yData, 0, 1, 'omitnan') ./ sqrt(nValid);

    shaded{ci} = shadedErrorBar(xData, yMean, [ySE; ySE]);
    shaded{ci}.mainLine.LineWidth = 1.5;
    shaded{ci}.mainLine.Color = myColor(ci,:);
    shaded{ci}.patch.FaceColor = myColor(ci,:);
    shaded{ci}.patch.FaceAlpha = 0.15;
    set(shaded{ci}.edge, 'LineStyle', 'none');

    mainLines(ci) = shaded{ci}.mainLine;
    patches(ci) = shaded{ci}.patch;
    edges{ci} = shaded{ci}.edge;
end

xlabel(myXlabel, 'FontSize', 14);
ylabel(myYlabel, 'FontSize', 14);
set(gca, 'FontSize', 14);
box off;
axis tight;

if isempty(xLimits)
    xLimits = get(gca, 'Xlim');
end
if isempty(yLimits)
    yLimits = get(gca, 'Ylim');
end
set(gca, 'Xlim', xLimits, 'Ylim', yLimits);

sigHandles = add_time_sig_annotations(gca, xData, sigCfg, myLegend, myColor);

line(get(gca, 'Xlim'), [0 0], 'Color', 'k', 'Linestyle', '--', ...
    'HandleVisibility', 'off');
line([0 0], get(gca, 'Ylim'), 'Color', 'k', 'Linestyle', '--', ...
    'HandleVisibility', 'off');

if ~isempty(myLegend)
    if numel(myLegend) < nCurve
        error('myLegend must contain at least nCurve labels.');
    end
    lgd = legend(mainLines, myLegend(1:nCurve), ...
        'AutoUpdate', 'off', 'Location', 'Best');
    set(lgd, 'FontSize', 14);
    legend boxoff;
end

h = struct();
h.shaded = shaded;
h.mainLine = mainLines;
h.patch = patches;
h.edge = edges;
h.sig = sigHandles;

end

%% ========================================================================
function value = get_optional_arg(args, idx, defaultValue)
if numel(args) >= idx && ~isempty(args{idx})
    value = args{idx};
else
    value = defaultValue;
end
end

%% ========================================================================
function sigHandles = add_time_sig_annotations(ax, times, sigCfg, modelLabels, modelColors)

sigHandles = struct();
sigHandles.againstShuffle = gobjects(0);
sigHandles.pairwise = gobjects(0);
sigHandles.labels = gobjects(0);
sigHandles.stars = gobjects(0);

if isempty(sigCfg) || ~isstruct(sigCfg)
    return
end

nCurve = size(modelColors, 1);
if isfield(sigCfg, 'modelNames') && ~isempty(sigCfg.modelNames)
    modelLabels = sigCfg.modelNames;
end
modelLabels = normalize_labels(modelLabels, nCurve);

alpha = get_struct_field(sigCfg, 'alpha', 0.05);
againstStats = get_struct_field(sigCfg, 'againstShuffleStats', struct());
baselineLabels = get_struct_field(sigCfg, 'baselineLabels', struct());
pairwiseStats = get_struct_field(sigCfg, 'pairwiseStats', struct());
pairInfo = get_struct_field(sigCfg, 'pairInfo', cell(0,3));

againstRows = struct('label', {}, 'stat', {}, 'color', {});
for ci = 1:nCurve
    stat = get_stat_by_name(againstStats, modelLabels{ci}, ci);
    if stat_has_sig(stat, alpha)
        baselineLabel = get_baseline_label(baselineLabels, modelLabels{ci}, ci);
        againstRows(end+1).label = sprintf('%s vs %s', modelLabels{ci}, baselineLabel); %#ok<AGROW>
        againstRows(end).stat = stat;
        againstRows(end).color = modelColors(ci,:);
    end
end

pairRows = struct('label', {}, 'stat', {}, 'color', {});
if ~isempty(pairInfo)
    for pi = 1:size(pairInfo, 1)
        if size(pairInfo, 2) < 3
            continue
        end

        A = pairInfo{pi,1};
        B = pairInfo{pi,2};
        statName = pairInfo{pi,3};
        stat = get_stat_by_name(pairwiseStats, statName, pi);

        if stat_has_sig(stat, alpha)
            pairRows(end+1).label = sprintf('%s-%s', A, B); %#ok<AGROW>
            pairRows(end).stat = stat;
            pairRows(end).color = pair_sig_color(A, B, modelLabels, modelColors);
        end
    end
end

if isempty(againstRows) && isempty(pairRows)
    return
end

yl = ylim(ax);
xl = xlim(ax);
yr = range(yl);
xr = range(xl);

if yr == 0 || isnan(yr)
    yr = 1;
end
if xr == 0 || isnan(xr)
    xr = 1;
end

baseBottom = yl(1);
baseTop = yl(2);

rowStep = 0.055 * yr;
bottomPad = 0.045 * yr;
topPad = 0.050 * yr;
tailPad = 0.055 * yr;

newBottom = yl(1);
newTop = yl(2);
if ~isempty(againstRows)
    newBottom = baseBottom - bottomPad - rowStep * numel(againstRows) - tailPad;
end
if ~isempty(pairRows)
    newTop = baseTop + topPad + rowStep * numel(pairRows) + tailPad;
end
ylim(ax, [newBottom newTop]);

labelX = xl(2) - 0.01 * xr;

for ri = 1:numel(againstRows)
    yBar = baseBottom - bottomPad - rowStep * (ri - 0.5);
    [lineHandles, starHandles] = draw_stat_clusters(ax, times, againstRows(ri).stat, ...
        yBar, againstRows(ri).color, alpha, yr);

    labelHandle = text(ax, labelX, yBar, againstRows(ri).label, ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','middle', ...
        'FontSize', 8, ...
        'Color', againstRows(ri).color, ...
        'BackgroundColor','w', ...
        'Margin', 1, ...
        'HandleVisibility','off', ...
        'Interpreter','none');

    sigHandles.againstShuffle = [sigHandles.againstShuffle; lineHandles(:)];
    sigHandles.stars = [sigHandles.stars; starHandles(:)];
    sigHandles.labels = [sigHandles.labels; labelHandle];
end

for ri = 1:numel(pairRows)
    yBar = baseTop + topPad + rowStep * (ri - 0.5);
    [lineHandles, starHandles] = draw_stat_clusters(ax, times, pairRows(ri).stat, ...
        yBar, pairRows(ri).color, alpha, yr);

    labelHandle = text(ax, labelX, yBar, pairRows(ri).label, ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','middle', ...
        'FontSize', 8, ...
        'Color', pairRows(ri).color, ...
        'BackgroundColor','w', ...
        'Margin', 1, ...
        'HandleVisibility','off', ...
        'Interpreter','none');

    sigHandles.pairwise = [sigHandles.pairwise; lineHandles(:)];
    sigHandles.stars = [sigHandles.stars; starHandles(:)];
    sigHandles.labels = [sigHandles.labels; labelHandle];
end

end

%% ========================================================================
function [lineHandles, starHandles] = draw_stat_clusters(ax, times, stat, yBar, colorValue, alpha, yRange)

lineHandles = gobjects(0);
starHandles = gobjects(0);

if ~isstruct(stat) || ~isfield(stat, 'significantClusters')
    return
end

clusters = stat.significantClusters;
lineWidth = 4;

for ci = 1:numel(clusters)
    cluster = clusters(ci);

    if isfield(cluster, 'p') && ~isnan(cluster.p) && cluster.p > alpha
        continue
    end
    if ~isfield(cluster, 'idx') || isempty(cluster.idx)
        continue
    end

    idx = cluster.idx(:)';
    idx = idx(idx >= 1 & idx <= numel(times));
    if isempty(idx)
        continue
    end

    lineHandle = plot(ax, times(idx), yBar * ones(size(idx)), '-', ...
        'Color', colorValue, ...
        'LineWidth', lineWidth, ...
        'HandleVisibility','off');

    if isfield(cluster, 'p') && ~isnan(cluster.p)
        starText = p_to_stars(cluster.p);
    else
        starText = '*';
    end

    xMid = mean(times([idx(1), idx(end)]));
    starHandle = text(ax, xMid, yBar + 0.012 * yRange, starText, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'FontSize', 9, ...
        'FontWeight','bold', ...
        'Color', colorValue, ...
        'HandleVisibility','off');

    lineHandles = [lineHandles; lineHandle]; %#ok<AGROW>
    starHandles = [starHandles; starHandle]; %#ok<AGROW>
end

end

%% ========================================================================
function tf = stat_has_sig(stat, alpha)

tf = false;
if ~isstruct(stat)
    return
end

if isfield(stat, 'significantClusters') && ~isempty(stat.significantClusters)
    clusters = stat.significantClusters;
    if isfield(clusters, 'p')
        pVals = [clusters.p];
        tf = any(~isnan(pVals) & pVals <= alpha);
    else
        tf = true;
    end
end

if ~tf && isfield(stat, 'significantMask')
    tf = any(stat.significantMask(:));
end

end

%% ========================================================================
function stat = get_stat_by_name(stats, label, idx)

stat = [];
if isempty(stats)
    return
end

if iscell(stats)
    if numel(stats) >= idx
        stat = stats{idx};
    end
    return
end

if ~isstruct(stats)
    return
end

if isfield(stats, 'significantMask') && isfield(stats, 'significantClusters')
    stat = stats;
    return
end

if numel(stats) > 1 && isfield(stats, 'significantMask')
    if numel(stats) >= idx
        stat = stats(idx);
    end
    return
end

fieldCandidates = unique({char(label), make_valid_field_name(label)}, 'stable');
for fi = 1:numel(fieldCandidates)
    fieldName = fieldCandidates{fi};
    if isfield(stats, fieldName)
        stat = stats.(fieldName);
        return
    end
end

end

%% ========================================================================
function labels = normalize_labels(labels, nLabel)

if isempty(labels)
    labels = arrayfun(@(idx) sprintf('Curve%d', idx), 1:nLabel, 'UniformOutput', false);
    return
end

if isstring(labels)
    labels = cellstr(labels);
elseif ischar(labels)
    labels = {labels};
end

labels = labels(:)';

if numel(labels) < nLabel
    for idx = (numel(labels) + 1):nLabel
        labels{idx} = sprintf('Curve%d', idx);
    end
end

for idx = 1:numel(labels)
    if isstring(labels{idx})
        labels{idx} = char(labels{idx});
    end
end

end

%% ========================================================================
function value = get_struct_field(s, name, defaultValue)

if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = defaultValue;
end

end

%% ========================================================================
function label = get_baseline_label(labels, modelLabel, idx)

label = 'shuffle';
if isempty(labels)
    return
end

if iscell(labels)
    if numel(labels) >= idx && ~isempty(labels{idx})
        label = labels{idx};
    end
    return
end

if isstring(labels)
    labels = cellstr(labels);
    if numel(labels) >= idx && ~isempty(labels{idx})
        label = labels{idx};
    end
    return
end

if ischar(labels)
    label = labels;
    return
end

if isstruct(labels)
    fieldCandidates = unique({char(modelLabel), make_valid_field_name(modelLabel)}, 'stable');
    for fi = 1:numel(fieldCandidates)
        fieldName = fieldCandidates{fi};
        if isfield(labels, fieldName) && ~isempty(labels.(fieldName))
            label = labels.(fieldName);
            return
        end
    end
end
end

%% ========================================================================
function fieldName = make_valid_field_name(label)

try
    fieldName = matlab.lang.makeValidName(char(label));
catch
    fieldName = regexprep(char(label), '[^A-Za-z0-9_]', '_');
    if isempty(regexp(fieldName, '^[A-Za-z]', 'once'))
        fieldName = ['x', fieldName];
    end
end

end

%% ========================================================================
function c = pair_sig_color(A, B, modelLabels, modelColors)

ia = find(strcmp(modelLabels, A), 1);
ib = find(strcmp(modelLabels, B), 1);

if isempty(ia) || isempty(ib)
    c = [0.1 0.1 0.1];
else
    c = 0.75 * mean(modelColors([ia ib],:), 1);
end

end

%% ========================================================================
function s = p_to_stars(p)

if p < 0.001
    s = '***';
elseif p < 0.01
    s = '**';
elseif p < 0.05
    s = '*';
else
    s = 'n.s.';
end

end

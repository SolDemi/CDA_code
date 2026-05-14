function [stat, h] = plot_group_timeseries_perm(data, times, cfg)
% plot_group_timeseries_perm
%
% Generic plotting wrapper for cluster_perm_1d_timeseries.
% It plots group mean +/- SEM and marks cluster-corrected significant periods.
%
% USAGE
%   cfg = struct();
%   cfg.null = 0.5;
%   cfg.ylabel = 'AUC';
%   cfg.title = 'LDA decoding';
%   [stat, h] = plot_group_timeseries_perm(groupAUC, times, cfg);
%
% INPUT
%   data  : nSubject x nTime matrix
%   times : 1 x nTime vector
%   cfg   : same statistical cfg as cluster_perm_1d_timeseries, plus plot fields
%
% EXTRA PLOT CFG FIELDS
%   cfg.doPlot          : true/false, default = true
%   cfg.color           : 1 x 3 RGB, default = [0 0.4470 0.7410]
%   cfg.xlim            : optional x-axis limit
%   cfg.ylim            : optional y-axis limit
%   cfg.xlabel          : default = 'Time (ms)'
%   cfg.ylabel          : default = 'Value'
%   cfg.title           : default = ''
%   cfg.eventLines      : numeric vector, default = 0
%   cfg.eventLineLabels : cellstr, default = {}
%   cfg.sigBarPosition  : 'bottom' or numeric y value, default = 'bottom'
%   cfg.showLegend      : true/false, default = false
%   cfg.label           : legend label, default = 'Mean'

if nargin < 3 || isempty(cfg)
    cfg = struct();
end
cfg = fill_default_plot_cfg(cfg);

stat = cluster_perm_1d_timeseries(data, times, cfg);

h = struct();
if ~cfg.doPlot
    return;
end

figure('Color', 'w', 'Position', [100 100 1000 520]);
h.ax = axes();
hold(h.ax, 'on');

x = stat.times;
y = stat.mean;
e = stat.sem;
c = cfg.color;

h.sem = patch([x, fliplr(x)], [y+e, fliplr(y-e)], c, ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
h.mean = plot(x, y, 'LineWidth', 2.2, 'Color', c, 'DisplayName', cfg.label);

% Null/chance line.
if isscalar(cfg.null)
    h.null = yline(cfg.null, ':k', 'Null', 'HandleVisibility', 'off');
else
    h.null = plot(x, cfg.null(:)', ':k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% Event lines.
h.eventLines = gobjects(numel(cfg.eventLines), 1);
for li = 1:numel(cfg.eventLines)
    if li <= numel(cfg.eventLineLabels) && ~isempty(cfg.eventLineLabels{li})
        h.eventLines(li) = xline(cfg.eventLines(li), '--k', cfg.eventLineLabels{li}, 'HandleVisibility', 'off');
    else
        h.eventLines(li) = xline(cfg.eventLines(li), '--k', 'HandleVisibility', 'off');
    end
end

xlabel(cfg.xlabel, 'FontSize', 14);
ylabel(cfg.ylabel, 'FontSize', 14);
if ~isempty(cfg.title)
    title(cfg.title, 'FontSize', 14);
end
box off;
grid on;
set(gca, 'FontSize', 14);

if ~isempty(cfg.xlim)
    xlim(cfg.xlim);
end
if ~isempty(cfg.ylim)
    ylim(cfg.ylim);
end

% Significant cluster bars.
yl = ylim;
yRange = yl(2) - yl(1);
if isnumeric(cfg.sigBarPosition)
    yBar = cfg.sigBarPosition;
else
    yBar = yl(1) + 0.04 * yRange;
end

h.sigBars = gobjects(numel(stat.significantClusters), 1);
for ci = 1:numel(stat.significantClusters)
    idx = stat.significantClusters(ci).idx;
    h.sigBars(ci) = plot(x(idx), yBar * ones(size(idx)), '-', ...
        'Color', c, 'LineWidth', 5, 'HandleVisibility', 'off');
end

if cfg.showLegend
    legend(h.mean, 'Location', 'best');
    legend boxoff;
end

end

%% ========================================================================
function cfg = fill_default_plot_cfg(cfg)
% Statistical defaults are filled again inside cluster_perm_1d_timeseries.
if ~isfield(cfg, 'doPlot'),          cfg.doPlot = true; end
if ~isfield(cfg, 'color'),           cfg.color = [0 0.4470 0.7410]; end
if ~isfield(cfg, 'xlim'),            cfg.xlim = []; end
if ~isfield(cfg, 'ylim'),            cfg.ylim = []; end
if ~isfield(cfg, 'xlabel'),          cfg.xlabel = 'Time (ms)'; end
if ~isfield(cfg, 'ylabel'),          cfg.ylabel = 'Value'; end
if ~isfield(cfg, 'title'),           cfg.title = ''; end
if ~isfield(cfg, 'eventLines'),      cfg.eventLines = 0; end
if ~isfield(cfg, 'eventLineLabels'), cfg.eventLineLabels = {}; end
if ~isfield(cfg, 'sigBarPosition'),  cfg.sigBarPosition = 'bottom'; end
if ~isfield(cfg, 'showLegend'),      cfg.showLegend = false; end
if ~isfield(cfg, 'label'),           cfg.label = 'Mean'; end

if numel(cfg.color) ~= 3
    error('cfg.color must be a 1 x 3 RGB vector.');
end
end

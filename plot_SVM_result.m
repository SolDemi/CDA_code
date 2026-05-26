%% Plot group-level SVM decoding results

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);

decodingDir = fullfile(projectRoot, 'data1', 'decoding_SVM_spatialControl', 'loadWithinSide');
saveDir = fullfile(decodingDir, 'GroupStats');
if ~isfolder(saveDir)
    mkdir(saveDir);
end

modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};
colors = lines(numel(modelNames));

loadCfg = struct();
loadCfg.metric = 'AUC';
loadCfg.useDiagonal = true;
loadCfg.filePattern = '*.mat';

stats = struct();
validModels = {};

for m = 1:numel(modelNames)
    modelName = modelNames{m};
    resultDir = fullfile(decodingDir, modelName);
    if ~isfolder(resultDir) || isempty(dir(fullfile(resultDir, '*.mat')))
        fprintf('Skip %s: no files found.\n', modelName);
        continue;
    end

    loadCfg.resultVarName = modelName;
    [diagAUC, time, usedFiles] = extract_decoding_timeseries(resultDir, loadCfg);

    stats.(modelName).diagAUC = diagAUC;
    stats.(modelName).mean = mean(diagAUC, 1, 'omitnan');
    stats.(modelName).sem = std(diagAUC, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(diagAUC), 1));
    stats.(modelName).n = size(diagAUC, 1);
    stats.(modelName).files = usedFiles;
    stats.(modelName).color = colors(m,:);

    validModels{end+1} = modelName; %#ok<SAGROW>
    fprintf('Loaded %s: nSub = %d, nTime = %d\n', modelName, size(diagAUC,1), size(diagAUC,2));
end

if isempty(validModels)
    error('No SVM result files found in %s.', decodingDir);
end

%% Time-course plot
fig = figure('Color', 'w', 'Position', [100 100 1100 620]); hold on;

for m = 1:numel(validModels)
    modelName = validModels{m};
    y = stats.(modelName).mean;
    e = stats.(modelName).sem;
    c = stats.(modelName).color;

    patch([time, fliplr(time)], [y+e, fliplr(y-e)], c, ...
        'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(time, y, 'LineWidth', 2, 'Color', c, ...
        'DisplayName', sprintf('%s (n=%d)', modelName, stats.(modelName).n));
end

xline(0, '--k', 'Memory array onset', 'HandleVisibility', 'off');
xline(150, '--k', 'Memory array offset', 'HandleVisibility', 'off');
yline(0.5, ':k', 'HandleVisibility', 'off');
xlim([-200 1000]);
ylim([0.4 0.9]);
xlabel('Time (ms)');
ylabel('AUC');
title('Group-level SVM decoding');
legend('Location', 'best');
box off; grid on;

savefig(fig, fullfile(saveDir, 'SVM_AUC_timeseries.fig'));
print(fig, fullfile(saveDir, 'SVM_AUC_timeseries.png'), '-dpng', '-r300');

%% Maintenance-window summary
summaryWin = [400 950];
timeIdx = time >= summaryWin(1) & time <= summaryWin(2);
rows = cell(numel(validModels), 6);

for m = 1:numel(validModels)
    modelName = validModels{m};
    subjMean = mean(stats.(modelName).diagAUC(:, timeIdx), 2, 'omitnan');
    subjMean = subjMean(~isnan(subjMean));
    [~, p, ~, st] = ttest(subjMean, 0.5, 'Tail', 'right');
    dz = (mean(subjMean, 'omitnan') - 0.5) ./ std(subjMean, 0, 'omitnan');

    rows(m,:) = {modelName, numel(subjMean), mean(subjMean, 'omitnan'), ...
        std(subjMean, 0, 'omitnan') ./ sqrt(numel(subjMean)), p, dz};

    fprintf('%s %d-%d ms: mean AUC = %.4f, t(%d) = %.3f, p = %.4g, dz = %.3f\n', ...
        modelName, summaryWin(1), summaryWin(2), rows{m,3}, st.df, st.tstat, p, dz);
end

Summary = cell2table(rows, 'VariableNames', ...
    {'Model', 'N', 'MeanAUC', 'SEM', 'P_right_vs_0p5', 'CohenDz'});
writetable(Summary, fullfile(saveDir, 'SVM_AUC_maintenance_summary.csv'));
save(fullfile(saveDir, 'SVM_AUC_group_stats.mat'), 'stats', 'Summary', 'time', 'loadCfg');

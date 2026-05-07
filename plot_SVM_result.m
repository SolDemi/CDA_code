clear, clc;
decodefolder = 'LDA';

maindir = erase(pwd,'code');
decodingDir = fullfile(maindir, ['decoding_' decodefolder] );

modelNames = {'CDA','Alpha','NoPCA','PCA'};
colors = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19; 0.49 0.18 0.56];
stats = struct();


time = -200:4:996;
idx  = dsearchn( [-200:4:996]', time'); %


for m = 1:numel(modelNames)
    modelName = modelNames{m};
    files = dir(fullfile(decodingDir, modelName, '*Super*'));
    disp(['Now Processing: ' modelName])
    diagAUC = [];
    for s = 1:numel(files)
        S = load(fullfile(files(s).folder, files(s).name));
        R = S.(modelName);
        if ~isvector(R.AUC) 
            diagAUC(s,:) = diag(R.AUC)'; %#ok<SAGROW>
        else
            diagAUC(s,:) = R.AUC;
        end

    end
    % diagAUC = diagAUC(:,idx);
    stats.(modelName).diagAUC = diagAUC;
    stats.(modelName).mean = mean(diagAUC, 1);
    stats.(modelName).sem  = std(diagAUC, 0, 1) ./ sqrt(size(diagAUC,1));
    stats.(modelName).n    = size(diagAUC,1);
end


% half_window = 50 / 2;
% validTimeMask = (time - half_window >= time(1)) & (time + half_window <= time(end));
% time = time(validTimeMask);

%% plotting
    xlim_plot = [-200 1000];
    ylim_plot = [0.49 0.53];
    xlabel_p  = 'Times';
    ylabel_p  = 'AUC';


plot_shaded_errorbar_fourCurve( time,stats.CDA.diagAUC,stats.Alpha.diagAUC,stats.NoPCA.diagAUC,stats.PCA.diagAUC,...
    xlim_plot,ylim_plot,xlabel_p,ylabel_p,modelNames,colors)
xline(0,'--r','Memory array onset')
yline(0.5,'--r')
xline(150,'--r','Memory array offset')



timeidx = dsearchn(time',[400 950]');
meanAUC = mean( stats.CDA.diagAUC(:,timeidx(1):timeidx(2) ) );
[h, p, ci, ~] = ttest(meanAUC, 0.5, 'Tail', 'right');
cohen_d = ( mean(meanAUC) - 0.5 ) / std(meanAUC)

%%
fig = figure('Color','w','Position',[100 100 1100 600]); hold on;
tmp_modelNames = {'CDA'};
for m = 1:numel(tmp_modelNames)
    modelName = tmp_modelNames{m};
    y = stats.(modelName).mean;
    e = stats.(modelName).sem;
    c = colors(m,:);

    patch([time, fliplr(time)], [y+e, fliplr(y-e)], c, ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none','HandleVisibility','off');

    plot(time, y, 'LineWidth', 2.2, 'Color', c, ...
        'DisplayName', sprintf('%s (n=%d)', modelName, stats.(modelName).n));
end

xline(0, '--k','HandleVisibility','off');
yline(0.5, ':k','HandleVisibility','off');
xlabel('Time (ms)');
ylabel('AUC (diag)');
title('Group-level AUC: CDA / Alpha / NoPCA / PCA');
legend('Location','best');
grid on; box off;
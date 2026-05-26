%% Compare rebuilt CDA with the author-provided CDA

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);

dataRoot = fullfile(projectRoot, 'data1');
manualDir = fullfile(dataRoot, 'cda_alpha');
rawDir = fullfile(dataRoot, 'data_raw');
outputDir = fullfile(dataRoot, 'erp');
if ~isfolder(outputDir)
    mkdir(outputDir);
end

rawFiles = dir(rawDir);
isSubject = [rawFiles.isdir] & ~ismember({rawFiles.name}, {'.', '..'});
subjects = rawFiles(isSubject);

cdaManual2 = [];
cdaManual6 = [];
cdaSource2 = [];
cdaSource6 = [];
plotTime = [];

for s = 1:numel(subjects)
    subj = subjects(s).name;
    fprintf('Now Processing: Subj%s\n', subj);

    sourceFile = fullfile(rawDir, subj, 'erp_singletrial.mat');
    manualFile = fullfile(manualDir, sprintf('sub%s.mat', subj));
    if ~isfile(sourceFile) || ~isfile(manualFile)
        fprintf('Skip subject %s: missing source or rebuilt file.\n', subj);
        continue;
    end

    source = load(sourceFile, 'cda');
    rebuilt = load(manualFile, 'cda');
    cda0 = source.cda;
    cda = rebuilt.cda;

    timeIdx = cda.time >= cda0.time(1) & cda.time <= cda0.time(end);
    if isempty(plotTime)
        plotTime = cda.time(timeIdx);
    end

    % Load 2: attended-left uses right-left; attended-right uses left-right.
    manualLoad2 = cat(1, ...
        cda.trial.right_L_2(:,:,timeIdx) - cda.trial.left_L_2(:,:,timeIdx), ...
        cda.trial.left_R_2(:,:,timeIdx)  - cda.trial.right_R_2(:,:,timeIdx));
    manualLoad2 = squeeze(mean(mean(manualLoad2, 1, 'omitnan'), 2, 'omitnan'))';

    % Load 6: same contra-minus-ipsi construction.
    manualLoad6 = cat(1, ...
        cda.trial.right_L_6(:,:,timeIdx) - cda.trial.left_L_6(:,:,timeIdx), ...
        cda.trial.left_R_6(:,:,timeIdx)  - cda.trial.right_R_6(:,:,timeIdx));
    manualLoad6 = squeeze(mean(mean(manualLoad6, 1, 'omitnan'), 2, 'omitnan'))';

    sourceLoad2 = mean(cda0.diff_2, 1, 'omitnan');
    sourceLoad6 = mean(cda0.diff_6, 1, 'omitnan');

    cdaManual2 = cat(1, cdaManual2, manualLoad2);
    cdaManual6 = cat(1, cdaManual6, manualLoad6);
    cdaSource2 = cat(1, cdaSource2, sourceLoad2);
    cdaSource6 = cat(1, cdaSource6, sourceLoad6);
end

fprintf('Finished CDA comparison: n = %d subjects.\n', size(cdaManual2, 1));

%% Plot
xlim_plot = [-200 1000];
ylim_plot = [-1.5 1.5];
xlabel_p  = 'Time (ms)';
ylabel_p  = 'Baselined Potential (uV)';
legend1   = 'Rebuilt CDA';
legend2   = 'Source CDA';
myColor1 = [38, 121, 178] ./ 255;
myColor2 = [235, 111, 41] ./ 255;

fig2 = figure('Color', 'w');
plot_shaded_errorbar_twoCurve(plotTime, cdaManual2, cdaSource2, ...
    xlim_plot, ylim_plot, xlabel_p, ylabel_p, legend1, legend2, myColor1, myColor2);
title('Load 2');
savefig(fig2, fullfile(outputDir, 'compare_CDA_load2.fig'));
print(fig2, fullfile(outputDir, 'compare_CDA_load2.png'), '-dpng', '-r300');

fig6 = figure('Color', 'w');
plot_shaded_errorbar_twoCurve(plotTime, cdaManual6, cdaSource6, ...
    xlim_plot, ylim_plot, xlabel_p, ylabel_p, legend1, legend2, myColor1, myColor2);
title('Load 6');
savefig(fig6, fullfile(outputDir, 'compare_CDA_load6.fig'));
print(fig6, fullfile(outputDir, 'compare_CDA_load6.png'), '-dpng', '-r300');

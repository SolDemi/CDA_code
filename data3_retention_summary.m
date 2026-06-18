%% data3 trial-retention summary using the author-script criterion
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
data3CodeDir = fullfile(projectRoot, 'data3_code', 'EEG_analysis_script');
outputDir = fullfile(dataDir, 'retention_summary');

addpath(codeDir);
addpath(data3CodeDir);
if ~isfolder(outputDir), mkdir(outputDir); end

cfg = data3_default_cfg();
setFiles = dir(fullfile(dataDir, 'sub*_all.set'));
setFiles = data3_filter_set_files(setFiles, data3_subject_filter());

rows = [];
leftTrial = [];
arTrial = [];
rawTrial = [];
subjects = [];
cellNames = {};

for sf = 1:numel(setFiles)
    [~, baseName] = fileparts(setFiles(sf).name);
    tok = regexp(baseName, '^sub(\d+)_all$', 'tokens', 'once');
    sn = str2double(tok{1});

    fprintf('data3 retention summary: sub%d\n', sn);
    EEG = data3_load_eeglab_set_with_fdt(fullfile(setFiles(sf).folder, setFiles(sf).name));
    retention = computeRetentionSummary(EEG, cfg);

    subjects(end+1,1) = sn; %#ok<SAGROW>
    leftTrial(end+1,:) = retention.leftTrial(:)'; %#ok<SAGROW>
    arTrial(end+1,:) = retention.arTrial(:)'; %#ok<SAGROW>
    rawTrial(end+1,:) = retention.rawTrial(:)'; %#ok<SAGROW>
    rows(end+1,:) = [sn, retention.meanTrialP, retention.minCellP, retention.keepBy75Percent]; %#ok<SAGROW>
    cellNames = retention.cellNames;
end

RetentionSummary = array2table(rows, 'VariableNames', ...
    {'subject', 'meanTrialP', 'minCellP', 'keepBy75Percent'});
originalSubjects = str2double(data3_original_subjects());
RetentionSummary.inOriginalFinalSample = ismember(RetentionSummary.subject, originalSubjects);
RetentionCounts = struct();
RetentionCounts.subjects = subjects;
RetentionCounts.cellNames = cellNames;
RetentionCounts.rawTrial = rawTrial;
RetentionCounts.arTrial = arTrial;
RetentionCounts.leftTrial = leftTrial;
RetentionCounts.authorCriterion = 'mean(left_trial across 12 cells) / 96';

save(fullfile(outputDir, 'data3_retention_summary.mat'), 'RetentionSummary', 'RetentionCounts');
writetable(RetentionSummary, fullfile(outputDir, 'data3_retention_summary.csv'));

function retention = computeRetentionSummary(EEG, cfg)
% Match the author-script trial-retention count across 12 condition cells.

chanLabelsOrig = data3_chan_labels(EEG);
events = data3_normalize_events(EEG.event);
dataUv = EEG.data * cfg.dataScaleToUv;

setSizes = [1 3 6];
cellNames = {'111','112','121','122','211','212','221','222','311','312','321','322'};
leftTrial = nan(numel(cellNames), 1);
rawTrial = nan(numel(cellNames), 1);
arTrial = nan(numel(cellNames), 1);

cellIdx = 0;
for si = 1:numel(setSizes)
    loadVal = setSizes(si);
    trials = data3_setsize_trials(events, loadVal);
    erpOffsets = data3_ms_offsets(cfg.erpEpochWindowBySetSizeMs(si,:), EEG.srate);
    erpTime = erpOffsets / EEG.srate * 1000;
    erpBaseIdx = erpTime >= cfg.erpBaselineWindowMs(1) & erpTime <= cfg.erpBaselineWindowMs(2);

    for family = {'L', 'C'}
        for side = {'L', 'R'}
            cellIdx = cellIdx + 1;
            keepTrial = strcmp({trials.side}, side{1}) & strcmp({trials.family}, family{1});
            theseTrials = trials(keepTrial);
            epochs = zeros(30, numel(erpOffsets), 0);

            for ti = 1:numel(theseTrials)
                idx = round(theseTrials(ti).firstLatency) + erpOffsets;
                if idx(1) < 1 || idx(end) > size(dataUv, 2)
                    continue;
                end

                epoch = dataUv(:, idx);
                epoch = epoch - mean(epoch(:, erpBaseIdx), 2);
                epoch = data3_reref_tp9_remove(epoch, chanLabelsOrig, cfg.referenceLabel);
                epochs(:,:,end+1) = epoch; %#ok<AGROW>
            end

            rawTrial(cellIdx) = size(epochs, 3);
            if isempty(epochs)
                leftTrial(cellIdx) = 0;
                arTrial(cellIdx) = 0;
                continue;
            end

            switch loadVal
                case 1
                    [INEEG_new, ~, ~] = AR_41_new(epochs);
                    [new_EEG, ~, ~] = HEOG_41_new2(INEEG_new);
                case 3
                    [INEEG_new, ~, ~] = AR_42_new(epochs);
                    [new_EEG, ~, ~] = HEOG_42_new2(INEEG_new);
                case 6
                    [INEEG_new, ~, ~] = AR_43_new(epochs);
                    [new_EEG, ~, ~] = HEOG_43_new2(INEEG_new);
            end

            arTrial(cellIdx) = size(INEEG_new, 3);
            leftTrial(cellIdx) = size(new_EEG, 3);
        end
    end
end

retention = struct();
retention.cellNames = cellNames;
retention.rawTrial = rawTrial;
retention.arTrial = arTrial;
retention.leftTrial = leftTrial;
retention.meanTrialP = mean(leftTrial, 'omitnan') / 96;
retention.minCellP = min(leftTrial) / 96;
retention.authorCriterion = 'mean(left_trial across 12 cells) / 96, following data3_code E1_preanalysis_Nofilt_AR.m';
retention.keepBy75Percent = retention.meanTrialP >= 0.75;
end

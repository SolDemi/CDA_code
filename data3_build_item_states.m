%% Build set-size-6 item-state features for data3
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
data3CodeDir = fullfile(projectRoot, 'data3_code', 'EEG_analysis_script');
outputDir = fullfile(dataDir, 'item_states');

addpath(codeDir);
addpath(data3CodeDir);

if ~isfolder(outputDir)
    mkdir(outputDir);
end

cfg = data3_default_cfg();
setFiles = dir(fullfile(dataDir, 'sub*_all.set'));
setFiles = data3_filter_set_files(setFiles, data3_subject_filter());

for sf = 1:numel(setFiles)
    setFile = fullfile(setFiles(sf).folder, setFiles(sf).name);
    [~, baseName] = fileparts(setFiles(sf).name);
    tok = regexp(baseName, '^sub(\d+)_all$', 'tokens', 'once');
    sn = tok{1};

    fprintf('Building data3 item states: sub%s\n', sn);
    EEG = data3_load_eeglab_set_with_fdt(setFile);
    state = makeItemStateFeatures(EEG, cfg);
    state.subject = str2double(sn);

    save(fullfile(outputDir, sprintf('sub%s.mat', sn)), 'state', '-v7.3');
    fprintf('sub%s setsize6-only item states: raw=%d, kept=%d\n', ...
        sn, state.rejection.nRawSetSize6, state.rejection.nFinalKeep);
end

fprintf('data3 item-state files saved to:\n%s\n', outputDir);

function state = makeItemStateFeatures(EEG, cfg)
% Build set-size-6 load1-load6 item-state features for one data3 subject.

chanLabelsOrig = data3_chan_labels(EEG);
chanLabels = data3_remove_reference_label(chanLabelsOrig, cfg.referenceLabel);

leftIdx  = data3_chan_indices(chanLabels, cfg.leftPosteriorLabels);
rightIdx = data3_chan_indices(chanLabels, cfg.rightPosteriorLabels);
postIdx = [leftIdx rightIdx];
nonPostIdx = data3_chan_indices(chanLabels, cfg.nonPosteriorLabels);
heogIdx = data3_chan_indices(chanLabels, cfg.heogLabels);

events = data3_normalize_events(EEG.event);
trials = data3_setsize_trials(events, 6);

erpOffsets = data3_ms_offsets(cfg.erpEpochWindowMs, EEG.srate);
alphaOffsets = data3_ms_offsets(cfg.alphaEpochWindowMs, EEG.srate);
erpTime = erpOffsets / EEG.srate * 1000;
alphaTime = alphaOffsets / EEG.srate * 1000;
erpBaseIdx = erpTime >= cfg.erpBaselineWindowMs(1) & erpTime <= cfg.erpBaselineWindowMs(2);

voltageEpochs = zeros(numel(chanLabels), numel(erpTime), 0);
alphaEpochs = zeros(numel(chanLabels), numel(alphaTime), 0);
sideLabel = zeros(0, 1);
familyLabel = zeros(0, 1);
itemLatencies = zeros(0, 6);
skipOutOfBounds = 0;

dataUv = EEG.data * cfg.dataScaleToUv;
for ti = 1:numel(trials)
    erpIdx = round(trials(ti).firstLatency) + erpOffsets;
    alphaIdx = round(trials(ti).firstLatency) + alphaOffsets;

    if erpIdx(1) < 1 || erpIdx(end) > size(dataUv, 2) || ...
            alphaIdx(1) < 1 || alphaIdx(end) > size(dataUv, 2)
        skipOutOfBounds = skipOutOfBounds + 1;
        continue;
    end

    voltageEpoch = dataUv(:, erpIdx);
    voltageEpoch = voltageEpoch - mean(voltageEpoch(:, erpBaseIdx), 2);
    voltageEpoch = data3_reref_tp9_remove(voltageEpoch, chanLabelsOrig, cfg.referenceLabel);

    alphaEpoch = dataUv(:, alphaIdx);
    alphaEpoch = data3_reref_tp9_remove(alphaEpoch, chanLabelsOrig, cfg.referenceLabel);

    voltageEpochs(:,:,end+1) = voltageEpoch; %#ok<AGROW>
    alphaEpochs(:,:,end+1) = alphaEpoch; %#ok<AGROW>

    sideLabel(end+1,1) = 1 + strcmp(trials(ti).side, 'R'); %#ok<AGROW>
    familyLabel(end+1,1) = 1 + strcmp(trials(ti).family, 'C'); %#ok<AGROW>
    itemLatencies(end+1,:) = trials(ti).itemLatencies; %#ok<AGROW>
end

[voltageAR, ~, Ikeep] = AR_43_new(voltageEpochs);
[~, ~, HEOG_in, heogInfo] = HEOG_43_new2(voltageAR);
keepTrial = Ikeep(HEOG_in);

voltageClean = voltageEpochs(:,:,keepTrial);
alphaClean = alphaEpochs(:,:,keepTrial);
alphaPower = calculate_hilbert_band_power(alphaClean, EEG.srate, alphaTime, ...
    cfg.alphaBaselineWindowMs, cfg.alphaFreqBand);

sideLabel = sideLabel(keepTrial);
familyLabel = familyLabel(keepTrial);
itemLatencies = itemLatencies(keepTrial,:);

nTrial = numel(keepTrial);
nLoad = size(cfg.cdaItemWindowsMs, 1);
nObs = nTrial * nLoad;

features = struct();
features.CDA = zeros(nObs, numel(leftIdx));
features.Alpha = zeros(nObs, numel(leftIdx));
features.GlobalAlpha = zeros(nObs, numel(postIdx));
features.GlobalAlphaMean = zeros(nObs, 1);
features.PosteriorAlphaLR = zeros(nObs, numel(postIdx));
features.HEOG = zeros(nObs, numel(heogIdx));
features.NonPosterior = zeros(nObs, numel(nonPostIdx));

loadLabel = zeros(nObs, 1);
trialGroup = zeros(nObs, 1);
obsSide = zeros(nObs, 1);
obsFamily = zeros(nObs, 1);

for ti = 1:nTrial
    for li = 1:nLoad
        row = (ti - 1) * nLoad + li;
        cdaWin = erpTime >= cfg.cdaItemWindowsMs(li,1) & erpTime <= cfg.cdaItemWindowsMs(li,2);
        alphaWin = alphaTime >= cfg.alphaItemWindowsMs(li,1) & alphaTime <= cfg.alphaItemWindowsMs(li,2);

        leftVolt = squeeze(mean(voltageClean(leftIdx, cdaWin, ti), 2))';
        rightVolt = squeeze(mean(voltageClean(rightIdx, cdaWin, ti), 2))';
        leftAlpha = squeeze(mean(alphaPower(leftIdx, alphaWin, ti), 2))';
        rightAlpha = squeeze(mean(alphaPower(rightIdx, alphaWin, ti), 2))';

        if sideLabel(ti) == 1
            features.CDA(row,:) = rightVolt - leftVolt;
            features.Alpha(row,:) = rightAlpha - leftAlpha;
        else
            features.CDA(row,:) = leftVolt - rightVolt;
            features.Alpha(row,:) = leftAlpha - rightAlpha;
        end

        posteriorAlpha = squeeze(mean(alphaPower(postIdx, alphaWin, ti), 2))';
        features.GlobalAlpha(row,:) = posteriorAlpha;
        features.GlobalAlphaMean(row) = mean(posteriorAlpha, 'omitnan');
        features.PosteriorAlphaLR(row,:) = posteriorAlpha;
        features.HEOG(row,:) = squeeze(mean(voltageClean(heogIdx, cdaWin, ti), 2))';
        features.NonPosterior(row,:) = squeeze(mean(voltageClean(nonPostIdx, cdaWin, ti), 2))';

        loadLabel(row) = li;
        trialGroup(row) = ti;
        obsSide(row) = sideLabel(ti);
        obsFamily(row) = familyLabel(ti);
    end
end

state = struct();
state.features = features;
state.univariate.CDA = mean(features.CDA, 2, 'omitnan');
state.univariate.Alpha = mean(features.Alpha, 2, 'omitnan');
state.univariate.GlobalAlpha = features.GlobalAlphaMean;
state.load = loadLabel;
state.trialGroup = trialGroup;
state.side = obsSide;
state.family = obsFamily;
state.sideNames = {'attendLeft', 'attendRight'};
state.familyNames = {'letter', 'color'};
state.chanLabels = chanLabels;
state.leftPosteriorLabels = cfg.leftPosteriorLabels;
state.rightPosteriorLabels = cfg.rightPosteriorLabels;
state.relPairLabels = cfg.relPairLabels;
state.nonPosteriorLabels = cfg.nonPosteriorLabels;
state.heogLabels = cfg.heogLabels;
state.itemLatencies = itemLatencies;
state.time = struct('erp', erpTime, 'alpha', alphaTime);
state.windows = struct('cdaItemWindowsMs', cfg.cdaItemWindowsMs, ...
    'alphaItemWindowsMs', cfg.alphaItemWindowsMs, ...
    'erpBaselineWindowMs', cfg.erpBaselineWindowMs, ...
    'alphaBaselineWindowMs', cfg.alphaBaselineWindowMs);
state.rejection = struct('nRawSetSize6', numel(trials), ...
    'nExtracted', size(voltageEpochs,3), ...
    'nARKeep', numel(Ikeep), ...
    'nHEOGKeep', numel(HEOG_in), ...
    'nFinalKeep', nTrial, ...
    'setSize6RetentionP', nTrial / max(numel(trials), 1), ...
    'participantExclusionNote', 'This is set-size-6-only retention; original participant exclusion uses mean retained trials across 12 cells / 96.', ...
    'skipOutOfBounds', skipOutOfBounds, ...
    'Ikeep', Ikeep, ...
    'HEOG_in', HEOG_in, ...
    'heogInfo', heogInfo);
end

%% Build data3 cda/alpha files for first-memory-locked decoding
% Output:
%   data3/cda_alpha/subXX.mat contains cda and alpha structs.
%
% Both structs keep set sizes 1, 3, and 6, including pre-zero samples.
% Fields are first-memory-locked and ready for sequential set-size decoding:
%   cda.trial.left_L_1, cda.trial.right_L_1, ..., cda.trial.left_R_6
%   alpha.trial.left_L_1, alpha.trial.right_L_1, ..., alpha.trial.left_R_6

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
data3CodeDir = fullfile(projectRoot, 'data3_code', 'EEG_analysis_script');
outputDir = fullfile(dataDir, 'cda_alpha');
if ~isfolder(outputDir)
    mkdir(outputDir);
end

addpath(codeDir);
addpath(data3CodeDir);

cfg = data3_default_cfg();
setSizes = [1 3 6];

setFiles = dir(fullfile(dataDir, 'sub*_all.set'));
setFiles = data3_filter_set_files(setFiles, data3_subject_filter());
if isempty(setFiles)
    error('No sub*_all.set files found in %s.', dataDir);
end

for sf = 1:numel(setFiles)
    setFile = fullfile(setFiles(sf).folder, setFiles(sf).name);
    [~, baseName] = fileparts(setFiles(sf).name);
    tok = regexp(baseName, '^sub(\d+)_all$', 'tokens', 'once');
    if isempty(tok)
        warning('Skip unexpected file name: %s', setFiles(sf).name);
        continue;
    end
    sn = tok{1};

    fprintf('Now building data3 cda/alpha: sub%s\n', sn);

    EEG = data3_load_eeglab_set_with_fdt(setFile);
    if EEG.srate ~= cfg.srateExpected
        warning('Subject %s has srate=%g, expected %g.', sn, EEG.srate, cfg.srateExpected);
    end

    chanLabelsOrig = data3_chan_labels(EEG);
    chanLabels = data3_remove_reference_label(chanLabelsOrig, cfg.referenceLabel);
    leftIdx  = data3_chan_indices(chanLabels, cfg.leftPosteriorLabels);
    rightIdx = data3_chan_indices(chanLabels, cfg.rightPosteriorLabels);

    events = data3_normalize_events(EEG.event);
    dataUv = EEG.data * cfg.dataScaleToUv;
    subject = str2double(sn);

    cda = struct();
    cda.featureName = 'CDA';
    cda.subject = subject;
    cda.srate = EEG.srate;
    cda.setSizes = setSizes;
    cda.leftElecLabels = cfg.leftPosteriorLabels;
    cda.rightElecLabels = cfg.rightPosteriorLabels;
    cda.relPairLabels = cfg.relPairLabels;
    cda.baselinewindow_ms = cfg.erpBaselineWindowMs;
    cda.alphaBaselineWindow_ms = cfg.alphaBaselineWindowMs;
    cda.alphaFreqBand = cfg.alphaFreqBand;
    cda.dataScaleToUv = cfg.dataScaleToUv;
    cda.referenceLabel = cfg.referenceLabel;
    cda.timeBySetSize = struct();
    cda.trial = struct();
    cda.extraction = struct();
    cda.temporalAlignment = struct( ...
        'savedTimeZero', 'first memory sample marker', ...
        'savedWindowDescription', 'First-memory-locked epochs by set size; includes pre-zero samples.', ...
        'erpBaselineTimeZero', 'first memory sample marker', ...
        'erpBaselineWindowMs', cfg.erpBaselineWindowMs, ...
        'alphaBaselineTimeZero', 'first memory sample marker', ...
        'alphaBaselineWindowMs', cfg.alphaBaselineWindowMs);
    cda.fieldConvention = ['left_L_1/right_L_1 = posterior left/right hemisphere channels in attended-left setsize-1 trials; ' ...
                           'left_R_1/right_R_1 = posterior left/right hemisphere channels in attended-right setsize-1 trials; ' ...
                           'same *_3 and *_6 fields for setsize-3 and setsize-6 trials; all fields are first-memory-locked.'];

    alpha = cda;
    alpha.featureName = 'Alpha';

    for si = 1:numel(setSizes)
        loadVal = setSizes(si);
        setField = sprintf('setsize%d', loadVal);

        setIdx = find(setSizes == loadVal, 1);
        if isempty(setIdx)
            error('Unsupported data3 set size: %g.', loadVal);
        end

        trials = data3_setsize_trials(events, loadVal);
        erpWindowMs = cfg.erpEpochWindowBySetSizeMs(setIdx,:);
        erpOffsets = data3_ms_offsets(erpWindowMs, EEG.srate);
        erpTime = erpOffsets / EEG.srate * 1000;
        erpBaseIdx = erpTime >= cfg.erpBaselineWindowMs(1) & erpTime <= cfg.erpBaselineWindowMs(2);

        alphaWindowMs = [cfg.alphaBaselineWindowMs(1), erpWindowMs(2)];
        alphaOffsets = data3_ms_offsets(alphaWindowMs, EEG.srate);
        alphaTime = alphaOffsets / EEG.srate * 1000;

        nChanNoRef = numel(chanLabelsOrig) - sum(strcmpi(chanLabelsOrig, cfg.referenceLabel));
        voltageEpochs = zeros(nChanNoRef, numel(erpTime), 0);
        alphaEpochs = zeros(nChanNoRef, numel(alphaTime), 0);
        sideLabel = zeros(0, 1);
        familyLabel = zeros(0, 1);
        extractedSetSizeTrialIndex = zeros(0, 1);
        skipOutOfBounds = 0;

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

            voltageEpochs(:,:,end+1) = voltageEpoch; 
            alphaEpochs(:,:,end+1) = alphaEpoch; 
            sideLabel(end+1,1) = 1 + strcmp(trials(ti).side, 'R'); 
            familyLabel(end+1,1) = 1 + strcmp(trials(ti).family, 'C'); 
            extractedSetSizeTrialIndex(end+1,1) = ti; 
        end

        if size(voltageEpochs, 3) == 0
            Ikeep = [];
            HEOG_in = [];
            heogInfo = [];
            heogFunction = '';
            keepTrial = [];
            voltageClean = zeros(nChanNoRef, numel(erpTime), 0);
            alphaPower = zeros(nChanNoRef, numel(alphaTime), 0);
        else
            switch loadVal
                case 1
                    [voltageAR, ~, Ikeep] = AR_41_new(voltageEpochs);
                    heogFunction = 'HEOG_41_new2';
                case 3
                    [voltageAR, ~, Ikeep] = AR_42_new(voltageEpochs);
                    heogFunction = 'HEOG_42_new2';
                case 6
                    [voltageAR, ~, Ikeep] = AR_43_new(voltageEpochs);
                    heogFunction = 'HEOG_43_new2';
                otherwise
                    error('Unsupported data3 set size for artifact rejection: %g.', loadVal);
            end

            if isempty(Ikeep)
                HEOG_in = [];
                heogInfo = [];
            else
                switch loadVal
                    case 1
                        [~, ~, HEOG_in, heogInfo] = HEOG_41_new2(voltageAR);
                    case 3
                        [~, ~, HEOG_in, heogInfo] = HEOG_42_new2(voltageAR);
                    case 6
                        [~, ~, HEOG_in, heogInfo] = HEOG_43_new2(voltageAR);
                    otherwise
                        error('Unsupported data3 set size for HEOG rejection: %g.', loadVal);
                end
            end

            keepTrial = Ikeep(HEOG_in);
            voltageClean = voltageEpochs(:,:,keepTrial);

            if isempty(keepTrial)
                alphaPower = zeros(nChanNoRef, numel(alphaTime), 0);
            else
                alphaClean = alphaEpochs(:,:,keepTrial);
                alphaPower = calculate_hilbert_band_power(alphaClean, EEG.srate, alphaTime, ...
                    cfg.alphaBaselineWindowMs, cfg.alphaFreqBand);
            end
        end

        setData = struct();
        setData.loadVal = loadVal;
        setData.erpTime = erpTime;
        setData.alphaTime = alphaTime;
        setData.voltageClean = voltageClean;
        setData.alphaPower = alphaPower;
        setData.sideLabel = sideLabel(keepTrial);
        setData.familyLabel = familyLabel(keepTrial);
        setData.nRawEventTrials = numel(trials);
        setData.nExtracted = size(voltageEpochs, 3);
        setData.nARKeep = numel(Ikeep);
        setData.nHEOGKeep = numel(HEOG_in);
        setData.nFinalKeep = numel(keepTrial);
        setData.extractedSetSizeTrialIndex = extractedSetSizeTrialIndex;
        setData.cleanSetSizeTrialIndex = extractedSetSizeTrialIndex(keepTrial);
        setData.skipOutOfBounds = skipOutOfBounds;
        setData.Ikeep = Ikeep;
        setData.HEOG_in = HEOG_in;
        setData.heogFunction = heogFunction;
        setData.heogInfo = heogInfo;
        setData.windowMs = struct('erp', erpWindowMs, 'alpha', alphaWindowMs);

        cda.timeBySetSize.(setField) = setData.erpTime;
        alpha.timeBySetSize.(setField) = setData.alphaTime;

        loadStr = num2str(loadVal);
        for sideSaveIdx = 1:2
            if sideSaveIdx == 1
                sideName = 'L';
            else
                sideName = 'R';
            end

            trialIdx = setData.sideLabel == sideSaveIdx;
            nTimeVoltage = size(setData.voltageClean, 2);
            if ~any(trialIdx)
                cda.trial.(sprintf('left_%s_%s', sideName, loadStr)) = zeros(0, numel(leftIdx), nTimeVoltage);
                cda.trial.(sprintf('right_%s_%s', sideName, loadStr)) = zeros(0, numel(rightIdx), nTimeVoltage);
            else
                cda.trial.(sprintf('left_%s_%s', sideName, loadStr)) = ...
                    permute(setData.voltageClean(leftIdx,:,trialIdx), [3 1 2]);
                cda.trial.(sprintf('right_%s_%s', sideName, loadStr)) = ...
                    permute(setData.voltageClean(rightIdx,:,trialIdx), [3 1 2]);
            end

            nTimeAlpha = size(setData.alphaPower, 2);
            if ~any(trialIdx)
                alpha.trial.(sprintf('left_%s_%s', sideName, loadStr)) = zeros(0, numel(leftIdx), nTimeAlpha);
                alpha.trial.(sprintf('right_%s_%s', sideName, loadStr)) = zeros(0, numel(rightIdx), nTimeAlpha);
            else
                alpha.trial.(sprintf('left_%s_%s', sideName, loadStr)) = ...
                    permute(setData.alphaPower(leftIdx,:,trialIdx), [3 1 2]);
                alpha.trial.(sprintf('right_%s_%s', sideName, loadStr)) = ...
                    permute(setData.alphaPower(rightIdx,:,trialIdx), [3 1 2]);
            end
        end

        metadata = rmfield(setData, {'voltageClean', 'alphaPower', 'sideLabel', 'familyLabel'});
        cda.extraction.(setField) = metadata;
        alpha.extraction.(setField) = metadata;
    end

    cda.trials_per_side_load = zeros(2, numel(setSizes));
    alpha.trials_per_side_load = zeros(2, numel(setSizes));
    for li = 1:numel(setSizes)
        loadStr = num2str(setSizes(li));
        cda.trials_per_side_load(1,li) = size(cda.trial.(sprintf('left_L_%s', loadStr)), 1);
        cda.trials_per_side_load(2,li) = size(cda.trial.(sprintf('left_R_%s', loadStr)), 1);
        alpha.trials_per_side_load(1,li) = size(alpha.trial.(sprintf('left_L_%s', loadStr)), 1);
        alpha.trials_per_side_load(2,li) = size(alpha.trial.(sprintf('left_R_%s', loadStr)), 1);
    end

    save(fullfile(outputDir, sprintf('sub%s.mat', sn)), 'cda', 'alpha', '-v7.3');
    fprintf('Subject %s complete: setsize final keeps = [%s]\n', ...
        sn, sprintf('%d ', arrayfun(@(x) cda.extraction.(sprintf('setsize%d', x)).nFinalKeep, setSizes)));
end

fprintf('data3 cda/alpha files saved to:\n%s\n', outputDir);

%% Build data3 cda/alpha files for the existing decoding pipeline
% Output format matches cda_alpha.m:
%   data3/cda_alpha/subXX.mat contains cda and alpha structs.
%
% For compatibility with run_load_within_side_models.m, data3 setsize 1 is
% stored in the *_2 fields as the low-load class, and setsize 6 is stored in
% the *_6 fields as the high-load class. The actual mapping is saved in
% cda.decodingLoadMap and alpha.decodingLoadMap.
%
% Trial rejection follows data3_code: condition epochs are cut from the
% first-sample baseline through the post-final-sample delay, then AR_41_new
% or AR_43_new is called to obtain Ikeep. For decoding, the saved CDA/alpha
% window is then realigned to the final memory sample marker so load 1 and
% load 6 share the same temporal meaning. The inferred HEOG_* functions
% from HEOG_inferred_functions are also called when available; otherwise
% this is recorded in data3Rejection.missingHEOGFunctions.

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

setFiles = dir(fullfile(dataDir, 'sub*_all.set'));
subjectFilter = parse_subject_filter(getenv('DATA3_SUBJECT_FILTER'));
if ~isempty(subjectFilter)
    setFiles = filter_set_files(setFiles, subjectFilter);
end
if isempty(setFiles)
    error('No sub*_all.set files found in %s.', dataDir);
end

%% Settings from data3_code
srateExpected = 250;
saveWindowMs = [0 996];           % final-sample-locked common decoding window
erpBaselineWindowMs = [-200 -4];  % first-sample baseline; data3_code uses the first 50 samples
alphaBaselineWindowMs = [-1400 -1000]; % first-sample baseline from data3_code
alphaFreqBand = [8 12];
% data3_code applies AR_41_new/AR_43_new directly to EEG.data with no
% additional unit conversion. Keep scale at 1 to preserve that behavior.
dataScaleToUv = 1;

% Original condition coding:
%   S31/S32 = attended left/right
%   S51/S52 = L/C condition family
%   S41/S42/S43 = setsize 1/3/6 final sample markers
% Only setsize 1 and 6 are used here so the result is binary low/high.
condPairs = { ...
    'L1', 'L_L1', 'R_L1'; ...
    'L6', 'L_L6', 'R_L6'; ...
    'C1', 'L_C1', 'R_C1'; ...
    'C6', 'L_C6', 'R_C6'};
condKeys = unique(condPairs(:,2:3));

leftPosteriorLabels  = {'P7', 'P3', 'PO3', 'PO7', 'O1'};
rightPosteriorLabels = {'P8', 'P4', 'PO4', 'PO8', 'O2'};
relPairLabels = {'P7/P8', 'P3/P4', 'PO3/PO4', 'PO7/PO8', 'O1/O2'};

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

    EEG = load_eeglab_set_with_fdt(setFile);
    if EEG.srate ~= srateExpected
        warning('Subject %s has srate=%g, expected %g.', sn, EEG.srate, srateExpected);
    end

    chanLabelsOrig = get_chan_labels(EEG);
    chanLabels = remove_reference_label(chanLabelsOrig, 'TP9');
    leftIdx  = get_channel_indices(chanLabels, leftPosteriorLabels);
    rightIdx = get_channel_indices(chanLabels, rightPosteriorLabels);

    saveOffsets = ms_to_sample_offsets(saveWindowMs, EEG.srate);
    saveTimes = saveOffsets / EEG.srate * 1000;

    events = normalize_events(EEG.event);
    [voltageEpochs, voltageSaveIdxByKey, alphaEpochs, alphaTimesByKey, alphaSaveIdxByKey, extractCounts] = ...
        extract_data3_code_epochs(EEG.data * dataScaleToUv, chanLabelsOrig, events, ...
        EEG.srate, condKeys, erpBaselineWindowMs, alphaBaselineWindowMs, saveWindowMs);

    [cdaCond, alphaCond, rejectionInfo] = apply_data3_code_rejection_and_build_conditions( ...
        voltageEpochs, voltageSaveIdxByKey, alphaEpochs, alphaTimesByKey, alphaSaveIdxByKey, condKeys, EEG.srate, ...
        alphaBaselineWindowMs, alphaFreqBand, numel(chanLabels), numel(saveTimes));

    cda = init_data3_decoding_struct(EEG.srate, saveTimes, relPairLabels, ...
        leftPosteriorLabels, rightPosteriorLabels, erpBaselineWindowMs, ...
        alphaBaselineWindowMs, alphaFreqBand, dataScaleToUv);
    cda.trials_per_cond = zeros(1, size(condPairs,1));
    cda.trials_per_side_cond = zeros(size(condPairs,1), 2);

    alpha = cda;
    alpha.baselinewindow_ms = alphaBaselineWindowMs;
    alpha.frep = alphaFreqBand;
    alpha.globalAlphaElecLabels = [leftPosteriorLabels, rightPosteriorLabels];
    alpha.featureConstruction = ['Only absolute posterior left/right alpha fields are stored. ' ...
        'Lateralized alpha, GlobalAlpha, and GlobalAlphaMean are constructed on demand during decoding.'];
    alpha.trial = struct();
    alpha.trials_per_cond = zeros(1, size(condPairs,1));
    alpha.trials_per_side_cond = zeros(size(condPairs,1), 2);

    for p = 1:size(condPairs,1)
        outName = condPairs{p,1};
        leftName = condPairs{p,2};
        rightName = condPairs{p,3};

        XL = cdaCond.(leftName);
        XR = cdaCond.(rightName);
        cda.trial = add_absolute_condition_fields(cda.trial, XL, XR, outName, leftIdx, rightIdx, numel(saveTimes));
        cda.trials_per_cond(p) = size(XL,1) + size(XR,1);
        cda.trials_per_side_cond(p,:) = [size(XL,1), size(XR,1)];

        XL = alphaCond.(leftName);
        XR = alphaCond.(rightName);
        alpha.trial = add_absolute_condition_fields(alpha.trial, XL, XR, outName, leftIdx, rightIdx, numel(saveTimes));
        alpha.trials_per_cond(p) = size(XL,1) + size(XR,1);
        alpha.trials_per_side_cond(p,:) = [size(XL,1), size(XR,1)];
    end

    cda.trial = add_data3_load_fields(cda.trial);
    cda.trial = remove_condition_level_absolute_fields(cda.trial, condPairs(:,1));
    cda = add_minimal_count_fields(cda);
    cda.data3Extraction = extractCounts;
    cda.data3Rejection = rejectionInfo;

    alpha.trial = add_data3_load_fields(alpha.trial);
    alpha.trial = remove_condition_level_absolute_fields(alpha.trial, condPairs(:,1));
    alpha = add_minimal_count_fields(alpha);
    alpha.data3Extraction = extractCounts;
    alpha.data3Rejection = rejectionInfo;

    save(fullfile(outputDir, sprintf('sub%s.mat', sn)), 'cda', 'alpha', '-v7.3');
    fprintf('Subject %s complete: raw=%d, ARkeep=%d, HEOGkeep=%d, badEvent=%d, outOfBounds=%d\n', ...
        sn, extractCounts.extracted, rejectionInfo.totalARKeep, rejectionInfo.totalHEOGKeep, ...
        extractCounts.skippedBadEvent, extractCounts.skippedOutOfBounds);
end

fprintf('data3 cda/alpha files saved to:\n%s\n', outputDir);

%% ========================= Helper functions =========================
function EEG = load_eeglab_set_with_fdt(setFile)
    S = load(setFile, '-mat');
    if ~isfield(S, 'EEG')
        error('No EEG variable found in %s.', setFile);
    end

    EEG = S.EEG;
    if isnumeric(EEG.data)
        return;
    end

    fdtFile = fullfile(fileparts(setFile), EEG.data);
    fid = fopen(fdtFile, 'rb');
    if fid < 0
        error('Cannot open FDT file: %s', fdtFile);
    end
    cleaner = onCleanup(@() fclose(fid));

    raw = fread(fid, [double(EEG.nbchan), double(EEG.pnts) * double(EEG.trials)], 'float32=>double');
    if numel(raw) ~= double(EEG.nbchan) * double(EEG.pnts) * double(EEG.trials)
        error('Unexpected FDT size for %s.', fdtFile);
    end
    EEG.data = reshape(raw, [double(EEG.nbchan), double(EEG.pnts), double(EEG.trials)]);
end

function subjectFilter = parse_subject_filter(filterText)
    subjectFilter = {};
    filterText = strtrim(char(filterText));
    if isempty(filterText)
        return;
    end

    parts = regexp(filterText, '[,;\s]+', 'split');
    parts = parts(~cellfun('isempty', parts));
    subjectFilter = regexprep(parts, '^sub', '', 'ignorecase');
end

function setFiles = filter_set_files(setFiles, subjectFilter)
    keep = false(size(setFiles));
    for i = 1:numel(setFiles)
        [~, baseName] = fileparts(setFiles(i).name);
        tok = regexp(baseName, '^sub(\d+)_all$', 'tokens', 'once');
        keep(i) = ~isempty(tok) && any(strcmp(tok{1}, subjectFilter));
    end
    setFiles = setFiles(keep);
end

function labels = get_chan_labels(EEG)
    labels = cell(1, numel(EEG.chanlocs));
    for i = 1:numel(EEG.chanlocs)
        labels{i} = strtrim(EEG.chanlocs(i).labels);
    end
end

function idx = get_channel_indices(allLabels, targetLabels)
    idx = zeros(1, numel(targetLabels));
    for i = 1:numel(targetLabels)
        thisIdx = find(strcmpi(allLabels, targetLabels{i}), 1);
        if isempty(thisIdx)
            error('Channel %s not found.', targetLabels{i});
        end
        idx(i) = thisIdx;
    end
end

function offsets = ms_to_sample_offsets(windowMs, srate)
    offsets = round(windowMs(1) / 1000 * srate) : round(windowMs(2) / 1000 * srate);
end

function labelsRef = remove_reference_label(labels, refLabel)
    refIdx = find(strcmpi(labels, refLabel), 1);
    if isempty(refIdx)
        error('Reference channel %s not found.', refLabel);
    end
    keepIdx = setdiff(1:numel(labels), refIdx, 'stable');
    labelsRef = labels(keepIdx);
end

function epochRef = rereference_epoch_to_tp9_and_remove(epoch, refIdx)
    epochRef = epoch - epoch(refIdx,:,:) ./ 2;
    keepIdx = setdiff(1:size(epoch,1), refIdx, 'stable');
    epochRef = epochRef(keepIdx,:,:);
end

function events = normalize_events(eegEvents)
    events = struct('type', {}, 'latency', {});
    for i = 1:numel(eegEvents)
        events(i).type = normalize_event_type(eegEvents(i).type); 
        events(i).latency = double(eegEvents(i).latency); 
    end
end

function txt = normalize_event_type(x)
    if isnumeric(x)
        txt = sprintf('S %d', x);
    else
        txt = char(string(x));
    end
    txt = regexprep(strtrim(txt), '\s+', ' ');
end

function loadVal = event_load_value(eventType)
    switch eventType
        case 'S 41'
            loadVal = 1;
        case 'S 42'
            loadVal = 3;
        case 'S 43'
            loadVal = 6;
        otherwise
            loadVal = NaN;
    end
end

function trialInfo = classify_data3_trial(events, loadEventIdx, loadVal)
    trialInfo = struct('isValid', false, 'side', '', 'family', '', 'anchorLatency', NaN);

    prevBoundary = find(strcmp({events(1:loadEventIdx).type}, 'S 20'), 1, 'last');
    if isempty(prevBoundary)
        prevBoundary = max(1, loadEventIdx - 12);
    end

    cueIdx = [];
    for i = loadEventIdx-1:-1:prevBoundary
        if strcmp(events(i).type, 'S 31') || strcmp(events(i).type, 'S 32')
            cueIdx = i;
            break;
        end
    end
    if isempty(cueIdx)
        return;
    end

    if strcmp(events(cueIdx).type, 'S 31')
        trialInfo.side = 'L';
    else
        trialInfo.side = 'R';
    end

    familyIdx = [];
    for i = loadEventIdx+1:min(numel(events), loadEventIdx+4)
        if strcmp(events(i).type, 'S 51') || strcmp(events(i).type, 'S 52')
            familyIdx = i;
            break;
        end
    end
    if isempty(familyIdx)
        return;
    end

    if strcmp(events(familyIdx).type, 'S 51')
        trialInfo.family = 'L';
    else
        trialInfo.family = 'C';
    end

    if loadVal == 1
        trialInfo.anchorLatency = events(loadEventIdx).latency;
    else
        anchorIdx = [];
        for i = cueIdx+1:loadEventIdx-1
            if strcmp(events(i).type, 'S 75')
                anchorIdx = i;
                break;
            end
        end
        if isempty(anchorIdx)
            return;
        end
        trialInfo.anchorLatency = events(anchorIdx).latency;
    end

    trialInfo.isValid = true;
end

function [voltageEpochs, voltageSaveIdxByKey, alphaEpochs, alphaTimesByKey, alphaSaveIdxByKey, counts] = extract_data3_code_epochs( ...
    dataUv, chanLabels, events, srate, condKeys, erpBaselineWindowMs, alphaBaselineWindowMs, saveWindowMs)

    voltageEpochs = init_empty_channel_time_conditions(condKeys);
    voltageSaveIdxByKey = struct();
    alphaEpochs = init_empty_channel_time_conditions(condKeys);
    alphaTimesByKey = struct();
    alphaSaveIdxByKey = struct();

    nSaveTime = numel(ms_to_sample_offsets(saveWindowMs, srate));
    for i = 1:numel(condKeys)
        key = condKeys{i};
        loadVal = load_value_from_condition_key(key);
        sequenceDurationMs = data3_sequence_duration_ms(loadVal, srate, erpBaselineWindowMs);

        voltageOffsets = data3_code_voltage_offsets(loadVal, srate);
        voltageSaveIdxByKey.(key) = voltageOffsets >= saveWindowMs(1) & voltageOffsets <= saveWindowMs(2);
        assert_saved_sample_count(voltageSaveIdxByKey.(key), nSaveTime, key, 'voltage');

        alphaOffsets = data3_alpha_offsets_from_first_sample( ...
            loadVal, srate, erpBaselineWindowMs, alphaBaselineWindowMs, saveWindowMs);
        alphaTimesByKey.(key) = alphaOffsets / srate * 1000;
        alphaSaveIdxByKey.(key) = alphaTimesByKey.(key) >= sequenceDurationMs + saveWindowMs(1) & ...
                                  alphaTimesByKey.(key) <= sequenceDurationMs + saveWindowMs(2);
        assert_saved_sample_count(alphaSaveIdxByKey.(key), nSaveTime, key, 'alpha');
    end

    counts = init_data3_extraction_counts(condKeys);
    counts.temporalAlignment = struct( ...
        'savedTimeZero', 'final memory sample marker (S41/S43)', ...
        'savedWindowMs', saveWindowMs, ...
        'savedWindowDescription', 'Post-final-sample decoding window; negative pre-final samples are omitted because set-size 6 is still in the sequential presentation period.', ...
        'erpBaselineTimeZero', 'first memory sample onset', ...
        'erpBaselineWindowMs', erpBaselineWindowMs, ...
        'alphaBaselineTimeZero', 'first memory sample onset', ...
        'alphaBaselineWindowMs', alphaBaselineWindowMs);
    for i = 1:numel(condKeys)
        key = condKeys{i};
        loadVal = load_value_from_condition_key(key);
        counts.byCondition.(key).sequenceDurationMs = data3_sequence_duration_ms(loadVal, srate, erpBaselineWindowMs);
        counts.byCondition.(key).voltageEpochSamples = numel(data3_code_voltage_offsets(loadVal, srate));
        counts.byCondition.(key).alphaEpochSamples = numel(alphaTimesByKey.(key));
        counts.byCondition.(key).savedSamples = nSaveTime;
    end

    refIdx = find(strcmpi(chanLabels, 'TP9'), 1);
    if isempty(refIdx)
        error('Reference channel TP9 not found.');
    end

    for evIdx = 1:numel(events)
        loadVal = event_load_value(events(evIdx).type);
        if ~(loadVal == 1 || loadVal == 6)
            continue;
        end

        trialInfo = classify_data3_trial(events, evIdx, loadVal);
        if ~trialInfo.isValid
            counts.skippedBadEvent = counts.skippedBadEvent + 1;
            continue;
        end

        key = sprintf('%s_%s%d', trialInfo.side, trialInfo.family, loadVal);
        voltageOffsets = data3_code_voltage_offsets(loadVal, srate);
        alphaOffsets = data3_alpha_offsets_from_first_sample( ...
            loadVal, srate, erpBaselineWindowMs, alphaBaselineWindowMs, saveWindowMs);
        voltageIdx = round(events(evIdx).latency) + voltageOffsets;
        alphaIdx = round(trialInfo.anchorLatency) + alphaOffsets;

        if voltageIdx(1) < 1 || voltageIdx(end) > size(dataUv, 2) || ...
                alphaIdx(1) < 1 || alphaIdx(end) > size(dataUv, 2)
            counts.skippedOutOfBounds = counts.skippedOutOfBounds + 1;
            continue;
        end

        voltageEpoch = dataUv(:, voltageIdx);
        voltageEpoch = voltageEpoch - mean(voltageEpoch(:,1:50), 2);
        voltageEpoch = rereference_epoch_to_tp9_and_remove(voltageEpoch, refIdx);

        alphaEpoch = dataUv(:, alphaIdx);
        alphaEpoch = rereference_epoch_to_tp9_and_remove(alphaEpoch, refIdx);

        voltageEpochs = append_channel_time_epoch(voltageEpochs, key, voltageEpoch);
        alphaEpochs = append_channel_time_epoch(alphaEpochs, key, alphaEpoch);

        counts.extracted = counts.extracted + 1;
        counts.byCondition.(key).raw = counts.byCondition.(key).raw + 1;
    end
end

function offsets = data3_code_voltage_offsets(loadVal, srate)
    switch loadVal
        case 1
            % E1_preanalysis_Nofilt.m extracts points 251:550 from
            % S41 epochs: -200 to +996 ms around S41.
            offsets = ms_to_sample_offsets([-200 996], srate);
        case 6
            % It extracts points 264:1125 from S43 epochs, i.e. -2448 to
            % +996 ms around S43. The first 50 samples are the baseline.
            offsets = ms_to_sample_offsets([-2448 996], srate);
        otherwise
            error('Unsupported data3 load value: %g.', loadVal);
    end
end

function sequenceDurationMs = data3_sequence_duration_ms(loadVal, srate, erpBaselineWindowMs)
    % The voltage epoch starts at the first-sample ERP baseline onset but is
    % indexed relative to the final set-size marker S41/S42/S43.
    voltageOffsets = data3_code_voltage_offsets(loadVal, srate);
    sequenceDurationMs = erpBaselineWindowMs(1) - voltageOffsets(1);
    if sequenceDurationMs < 0
        error('Unexpected negative data3 sequence duration for load %g.', loadVal);
    end
end

function offsets = data3_alpha_offsets_from_first_sample(loadVal, srate, erpBaselineWindowMs, alphaBaselineWindowMs, saveWindowMs)
    sequenceDurationMs = data3_sequence_duration_ms(loadVal, srate, erpBaselineWindowMs);
    offsets = ms_to_sample_offsets([alphaBaselineWindowMs(1) sequenceDurationMs + saveWindowMs(2)], srate);
end

function assert_saved_sample_count(saveIdx, nSaveTime, key, featureName)
    if sum(saveIdx) ~= nSaveTime
        error('data3 %s save window for %s has %d samples, expected %d.', ...
            featureName, key, sum(saveIdx), nSaveTime);
    end
end

function S = init_empty_channel_time_conditions(condKeys)
    S = struct();
    for i = 1:numel(condKeys)
        S.(condKeys{i}) = [];
    end
end

function counts = init_data3_extraction_counts(condKeys)
    counts = struct();
    counts.extracted = 0;
    counts.skippedBadEvent = 0;
    counts.skippedOutOfBounds = 0;
    counts.byCondition = struct();
    for i = 1:numel(condKeys)
        counts.byCondition.(condKeys{i}) = struct('raw', 0);
    end
end

function S = append_channel_time_epoch(S, key, epoch)
    if isempty(S.(key))
        S.(key) = zeros(size(epoch,1), size(epoch,2), 0);
    end

    n = size(S.(key), 3) + 1;
    S.(key)(:,:,n) = epoch;
end

function [cdaCond, alphaCond, rejectionInfo] = apply_data3_code_rejection_and_build_conditions( ...
    voltageEpochs, voltageSaveIdxByKey, alphaEpochs, alphaTimesByKey, alphaSaveIdxByKey, condKeys, srate, ...
    alphaBaselineWindowMs, alphaFreqBand, nChan, nSaveTime)

    cdaCond = struct();
    alphaCond = struct();
    rejectionInfo = init_data3_rejection_info(condKeys);

    for i = 1:numel(condKeys)
        key = condKeys{i};
        loadVal = load_value_from_condition_key(key);
        EEG_new = voltageEpochs.(key);
        alphaRaw = alphaEpochs.(key);
        nRaw = size(EEG_new, 3);
        rejectionInfo.byCondition.(key).raw = nRaw;

        if nRaw == 0
            cdaCond.(key) = zeros(0, nChan, nSaveTime);
            alphaCond.(key) = zeros(0, nChan, nSaveTime);
            continue;
        end

        [Ikeep, HEOG_in, heogApplied, heogFunction, heogRejInfo] = data3_code_keep_indices(EEG_new, loadVal);
        rejectionInfo.byCondition.(key).Ikeep = Ikeep;
        rejectionInfo.byCondition.(key).HEOG_in = HEOG_in;
        rejectionInfo.byCondition.(key).heogApplied = heogApplied;
        rejectionInfo.byCondition.(key).heogFunction = heogFunction;
        rejectionInfo.byCondition.(key).heogRejInfo = heogRejInfo;
        rejectionInfo.byCondition.(key).arKeep = numel(Ikeep);
        rejectionInfo.byCondition.(key).heogKeep = numel(HEOG_in);

        rejectionInfo.totalRaw = rejectionInfo.totalRaw + nRaw;
        rejectionInfo.totalARKeep = rejectionInfo.totalARKeep + numel(Ikeep);
        rejectionInfo.totalHEOGKeep = rejectionInfo.totalHEOGKeep + numel(HEOG_in);
        if ~heogApplied
            rejectionInfo.missingHEOGFunctions{end+1} = heogFunction; 
        end

        if isempty(Ikeep) || isempty(HEOG_in)
            cdaCond.(key) = zeros(0, nChan, nSaveTime);
            alphaCond.(key) = zeros(0, nChan, nSaveTime);
            continue;
        end

        voltageClean = EEG_new(:,:,Ikeep);
        voltageClean = voltageClean(:,:,HEOG_in);
        voltageSaveIdx = voltageSaveIdxByKey.(key);
        assert_saved_sample_count(voltageSaveIdx, nSaveTime, key, 'voltage');
        cdaCond.(key) = permute(voltageClean(:,voltageSaveIdx,:), [3 1 2]);

        alphaClean = alphaRaw(:,:,Ikeep);
        alphaClean = alphaClean(:,:,HEOG_in);
        alphaTimes = alphaTimesByKey.(key);
        alphaPower = calculate_hilbert_band_power(alphaClean, srate, alphaTimes, ...
            alphaBaselineWindowMs, alphaFreqBand);
        alphaSaveIdx = alphaSaveIdxByKey.(key);
        assert_saved_sample_count(alphaSaveIdx, nSaveTime, key, 'alpha');
        alphaCond.(key) = permute(alphaPower(:,alphaSaveIdx,:), [3 1 2]);
    end

    rejectionInfo.missingHEOGFunctions = unique(rejectionInfo.missingHEOGFunctions);
end

function rejectionInfo = init_data3_rejection_info(condKeys)
    rejectionInfo = struct();
    rejectionInfo.source = 'data3_code AR_41_new/AR_43_new plus inferred HEOG_*_new2 when available';
    rejectionInfo.totalRaw = 0;
    rejectionInfo.totalARKeep = 0;
    rejectionInfo.totalHEOGKeep = 0;
    rejectionInfo.missingHEOGFunctions = {};
    rejectionInfo.byCondition = struct();

    for i = 1:numel(condKeys)
        rejectionInfo.byCondition.(condKeys{i}) = struct( ...
            'raw', 0, 'arKeep', 0, 'heogKeep', 0, ...
            'Ikeep', [], 'HEOG_in', [], ...
            'heogApplied', false, 'heogFunction', '', 'heogRejInfo', []);
    end
end

function loadVal = load_value_from_condition_key(key)
    tok = regexp(key, '(\d+)$', 'tokens', 'once');
    if isempty(tok)
        error('Cannot infer load value from condition key %s.', key);
    end
    loadVal = str2double(tok{1});
end

function [Ikeep, HEOG_in, heogApplied, heogFunction, heogRejInfo] = data3_code_keep_indices(EEG_new, loadVal)
    heogRejInfo = [];

    switch loadVal
        case 1
            [INEEG_new, ~, Ikeep] = AR_41_new(EEG_new);
            heogFunction = 'HEOG_41_new2';
        case 6
            [INEEG_new, ~, Ikeep] = AR_43_new(EEG_new);
            heogFunction = 'HEOG_43_new2';
        otherwise
            error('Unsupported data3 load value for decoding: %g.', loadVal);
    end

    if isempty(Ikeep)
        HEOG_in = [];
        heogApplied = false;
        return;
    end

    if exist(heogFunction, 'file') == 2
        [~, ~, HEOG_in, heogRejInfo] = feval(heogFunction, INEEG_new);
        heogApplied = true;
    else
        HEOG_in = 1:size(INEEG_new, 3);
        heogApplied = false;
    end
end

function S = init_data3_decoding_struct(srate, times, relPairLabels, leftLabels, rightLabels, ...
    erpBaselineWindowMs, alphaBaselineWindowMs, alphaFreqBand, dataScaleToUv)

    S = struct();
    S.srate = srate;
    S.time = times;
    S.relPairLabels = relPairLabels;
    S.leftElecLabels = leftLabels;
    S.rightElecLabels = rightLabels;
    S.baselinewindow_ms = erpBaselineWindowMs;
    S.alphaBaselineWindow_ms = alphaBaselineWindowMs;
    S.alphaFreqBand = alphaFreqBand;
    S.dataScaleToUv = dataScaleToUv;
    S.decodingLoadMap = struct('field_2_actual_setsize', 1, 'field_6_actual_setsize', 6);
    S.data3TemporalAlignment = struct( ...
        'savedTimeZero', 'final memory sample marker (S41/S43)', ...
        'savedTimeMs', times, ...
        'savedWindowMs', [times(1) times(end)], ...
        'savedWindowDescription', 'Post-final-sample decoding window; negative pre-final samples are omitted because set-size 6 is still in the sequential presentation period.', ...
        'erpBaselineTimeZero', 'first memory sample onset', ...
        'erpBaselineWindowMs', erpBaselineWindowMs, ...
        'alphaBaselineTimeZero', 'first memory sample onset', ...
        'alphaBaselineWindowMs', alphaBaselineWindowMs);
    S.conditionLabels = {'L1', 'L6', 'C1', 'C6'};
    S.fieldConvention = ['left_L_2/right_L_2 = posterior left/right hemisphere channels in attended-left setsize-1 trials; ' ...
                         'left_R_2/right_R_2 = posterior left/right hemisphere channels in attended-right setsize-1 trials; ' ...
                         'same *_6 fields for setsize-6 trials; all saved time points are locked to the final memory sample marker.'];
    S.trial = struct();
end

function trial = add_absolute_condition_fields(trial, XL, XR, outName, leftIdx, rightIdx, nTime)
    trial.(['left_L_'  outName]) = select_channels(XL, leftIdx, nTime);
    trial.(['right_L_' outName]) = select_channels(XL, rightIdx, nTime);
    trial.(['left_R_'  outName]) = select_channels(XR, leftIdx, nTime);
    trial.(['right_R_' outName]) = select_channels(XR, rightIdx, nTime);
end

function Xsel = select_channels(X, idx, nTime)
    if isempty(X)
        Xsel = zeros(0, numel(idx), nTime);
    else
        Xsel = X(:, idx, :);
    end
end

function trial = add_data3_load_fields(trial)
    hemiFields = {'left_L', 'right_L', 'left_R', 'right_R'};
    loadDefs = { ...
        '2', {'L1', 'C1'}; ...
        '6', {'L6', 'C6'}};

    for li = 1:size(loadDefs,1)
        loadName = loadDefs{li,1};
        conds = loadDefs{li,2};

        for hi = 1:numel(hemiFields)
            h = hemiFields{hi};
            trial.(sprintf('%s_%s', h, loadName)) = cat(1, ...
                trial.(sprintf('%s_%s', h, conds{1})), ...
                trial.(sprintf('%s_%s', h, conds{2})));
        end
    end
end

function trial = remove_condition_level_absolute_fields(trial, condNames)
    hemiFields = {'left_L', 'right_L', 'left_R', 'right_R'};
    removeList = {};

    for hi = 1:numel(hemiFields)
        for ci = 1:numel(condNames)
            removeList{end+1,1} = sprintf('%s_%s', hemiFields{hi}, condNames{ci}); %#ok<AGROW>
        end
    end

    fieldsToRemove = intersect(removeList, fieldnames(trial));
    if ~isempty(fieldsToRemove)
        trial = rmfield(trial, fieldsToRemove);
    end
end

function S = add_minimal_count_fields(S)
    S.trials_per_side_load = [size(S.trial.left_L_2,1), size(S.trial.left_L_6,1); ...
                              size(S.trial.left_R_2,1), size(S.trial.left_R_6,1)];
    S.trials_per_ss = [size(S.trial.left_L_2,1) + size(S.trial.left_R_2,1), ...
                       size(S.trial.left_L_6,1) + size(S.trial.left_R_6,1)];
    S.min_trials_per_cond = min(S.trials_per_cond);
    S.min_trials_per_ss = min(S.trials_per_ss);
    S.min_trials_per_side_load = min(S.trials_per_side_load, [], 'all');
end

function Xout = run_power_function_keep_trial_chan_time(Xtrial, srate, timeMs, baselineWindowMs, freqBand)
    if isempty(Xtrial)
        Xout = [];
        return;
    end

    nTr = size(Xtrial,1);
    nCh = size(Xtrial,2);
    nTm = size(Xtrial,3);

    Xin = permute(Xtrial, [2 3 1]);
    Xpow = calculate_hilbert_band_power(Xin, srate, timeMs, baselineWindowMs, freqBand);
    Xout = coerce_to_trial_chan_time(Xpow, nTr, nCh, nTm);
end

function X = coerce_to_trial_chan_time(Xin, nTr, nCh, nTm)
    sz = size(Xin);

    if isequal(sz, [nTr, nCh, nTm])
        X = Xin;
    elseif isequal(sz, [nCh, nTm, nTr])
        X = permute(Xin, [3 1 2]);
    elseif isequal(sz, [nTr, nTm, nCh])
        X = permute(Xin, [1 3 2]);
    elseif isequal(sz, [nTm, nCh, nTr])
        X = permute(Xin, [3 2 1]);
    else
        error(['Unexpected output size from calculate_hilbert_band_power: [' ...
               num2str(sz) ']. Please check its output dimension order.']);
    end
end

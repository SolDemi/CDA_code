%% data3 univariate CDA / alpha serial-position replication
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
dataDir = fullfile(projectRoot, 'data3');
data3CodeDir = fullfile(projectRoot, 'data3_code', 'EEG_analysis_script');
stateDir = fullfile(projectRoot, 'data3', 'item_states');
resultDir = fullfile(projectRoot, 'data3', 'results');
figDir = fullfile(resultDir, 'figures');

addpath(codeDir);
addpath(data3CodeDir);
if ~isfolder(resultDir), mkdir(resultDir); end
if ~isfolder(figDir), mkdir(figDir); end

cfg = data3_default_cfg();
setFiles = dir(fullfile(dataDir, 'sub*_all.set'));
setFiles = data3_filter_set_files(setFiles, data3_subject_filter());

features = {'CDA', 'Alpha', 'GlobalAlpha'};
families = {'color', 'letter'};
setFields = {'ss1', 'ss3', 'ss6'};
setSizes = [1 3 6];

subjects = nan(numel(setFiles), 1);
replication = struct();
for feati = 1:numel(features)
    for fami = 1:numel(families)
        for seti = 1:numel(setFields)
            replication.(features{feati}).(families{fami}).(setFields{seti}) = ...
                nan(numel(setFiles), setSizes(seti));
        end
    end
end

for si = 1:numel(setFiles)
    [~, baseName] = fileparts(setFiles(si).name);
    tok = regexp(baseName, '^sub(\d+)_all$', 'tokens', 'once');
    subjects(si) = str2double(tok{1});

    fprintf('data3 univariate author-style replication: sub%d\n', subjects(si));
    EEG = data3_load_eeglab_set_with_fdt(fullfile(setFiles(si).folder, setFiles(si).name));

    familyCodes = {'C', 'L'};
    familyNames = {'color', 'letter'};
    localSetSizes = [1 3 6];

    U = struct();
    U.familyNames = familyNames;
    U.setSizes = localSetSizes;

    chanLabelsOrig = data3_chan_labels(EEG);
    chanLabels = data3_remove_reference_label(chanLabelsOrig, cfg.referenceLabel);
    leftIdx  = data3_chan_indices(chanLabels, cfg.leftPosteriorLabels);
    rightIdx = data3_chan_indices(chanLabels, cfg.rightPosteriorLabels);
    postIdx = [leftIdx rightIdx];

    events = data3_normalize_events(EEG.event);
    dataUv = EEG.data * cfg.dataScaleToUv;

    for famCodeIdx = 1:numel(familyCodes)
        familyCode = familyCodes{famCodeIdx};
        familyName = familyNames{famCodeIdx};

        for setLoopIdx = 1:numel(localSetSizes)
            loadVal = localSetSizes(setLoopIdx);
            condBySide = cell(1, 2);

            for sideLoopIdx = 1:2
                if sideLoopIdx == 1
                    sideName = 'L';
                else
                    sideName = 'R';
                end

                trials = data3_setsize_trials(events, loadVal);
                keepTrial = strcmp({trials.side}, sideName) & strcmp({trials.family}, familyCode);
                trials = trials(keepTrial);

                setIdx = find([1 3 6] == loadVal, 1);
                erpOffsets = data3_ms_offsets(cfg.erpEpochWindowBySetSizeMs(setIdx,:), EEG.srate);
                alphaOffsets = data3_ms_offsets(cfg.alphaEpochWindowBySetSizeMs(setIdx,:), EEG.srate);
                erpTime = erpOffsets / EEG.srate * 1000;
                alphaTime = alphaOffsets / EEG.srate * 1000;
                erpBaseIdx = erpTime >= cfg.erpBaselineWindowMs(1) & erpTime <= cfg.erpBaselineWindowMs(2);

                voltageEpochs = zeros(numel(chanLabels), numel(erpTime), 0);
                alphaEpochs = zeros(numel(chanLabels), numel(alphaTime), 0);

                for ti = 1:numel(trials)
                    erpIdx = round(trials(ti).firstLatency) + erpOffsets;
                    alphaIdx = round(trials(ti).firstLatency) + alphaOffsets;

                    if erpIdx(1) < 1 || erpIdx(end) > size(dataUv, 2) || ...
                            alphaIdx(1) < 1 || alphaIdx(end) > size(dataUv, 2)
                        continue;
                    end

                    voltageEpoch = dataUv(:, erpIdx);
                    voltageEpoch = voltageEpoch - mean(voltageEpoch(:, erpBaseIdx), 2);
                    voltageEpoch = data3_reref_tp9_remove(voltageEpoch, chanLabelsOrig, cfg.referenceLabel);
                    voltageEpochs(:,:,end+1) = voltageEpoch; %#ok<AGROW>

                    alphaEpoch = dataUv(:, alphaIdx);
                    alphaEpoch = data3_reref_tp9_remove(alphaEpoch, chanLabelsOrig, cfg.referenceLabel);
                    alphaEpochs(:,:,end+1) = alphaEpoch; %#ok<AGROW>
                end

                if isempty(voltageEpochs)
                    cond = [];
                else
                    switch loadVal
                        case 1
                            [voltageAR, ~, Ikeep] = AR_41_new(voltageEpochs);
                            [~, ~, HEOG_in] = HEOG_41_new2(voltageAR);
                        case 3
                            [voltageAR, ~, Ikeep] = AR_42_new(voltageEpochs);
                            [~, ~, HEOG_in] = HEOG_42_new2(voltageAR);
                        case 6
                            [voltageAR, ~, Ikeep] = AR_43_new(voltageEpochs);
                            [~, ~, HEOG_in] = HEOG_43_new2(voltageAR);
                    end

                    keepIdx = Ikeep(HEOG_in);
                    voltageClean = voltageEpochs(:,:,keepIdx);
                    alphaClean = alphaEpochs(:,:,keepIdx);
                    alphaPower = calculate_hilbert_band_power(alphaClean, EEG.srate, alphaTime, ...
                        cfg.alphaBaselineWindowMs, cfg.alphaFreqBand);

                    if strcmp(sideName, 'L')
                        cda = voltageClean(rightIdx,:,:) - voltageClean(leftIdx,:,:);
                        alphaLat = alphaPower(rightIdx,:,:) - alphaPower(leftIdx,:,:);
                    else
                        cda = voltageClean(leftIdx,:,:) - voltageClean(rightIdx,:,:);
                        alphaLat = alphaPower(leftIdx,:,:) - alphaPower(rightIdx,:,:);
                    end

                    cond = struct();
                    cond.CDA = cda;
                    cond.Alpha = alphaLat;
                    cond.GlobalAlpha = alphaPower(postIdx,:,:);
                    cond.erpTime = erpTime;
                    cond.alphaTime = alphaTime;
                    cond.nRaw = numel(trials);
                    cond.nKeep = numel(keepIdx);
                end

                condBySide{sideLoopIdx} = cond;
            end

            condL = condBySide{1};
            condR = condBySide{2};

            cda = cat(3, condL.CDA, condR.CDA);
            alpha = cat(3, condL.Alpha, condR.Alpha);
            globalAlpha = cat(3, condL.GlobalAlpha, condR.GlobalAlpha);

            cdaMean = mean(cda, 3, 'omitnan');
            alphaMean = mean(alpha, 3, 'omitnan');
            globalMean = mean(globalAlpha, 3, 'omitnan');

            windowsMs = cfg.cdaItemWindowsBySetSizeMs{setLoopIdx};
            cdaCurve = nan(1, size(windowsMs, 1));
            cdaInput = mean(cdaMean, 1, 'omitnan');
            for wi = 1:size(windowsMs, 1)
                idx = condL.erpTime >= windowsMs(wi,1) & condL.erpTime <= windowsMs(wi,2);
                cdaCurve(wi) = mean(cdaInput(idx), 'omitnan');
            end

            windowsMs = cfg.alphaItemWindowsBySetSizeMs{setLoopIdx};
            alphaCurve = nan(1, size(windowsMs, 1));
            alphaInput = mean(alphaMean, 1, 'omitnan');
            for wi = 1:size(windowsMs, 1)
                idx = condL.alphaTime >= windowsMs(wi,1) & condL.alphaTime <= windowsMs(wi,2);
                alphaCurve(wi) = mean(alphaInput(idx), 'omitnan');
            end

            globalCurve = nan(1, size(windowsMs, 1));
            globalInput = mean(globalMean, 1, 'omitnan');
            for wi = 1:size(windowsMs, 1)
                idx = condL.alphaTime >= windowsMs(wi,1) & condL.alphaTime <= windowsMs(wi,2);
                globalCurve(wi) = mean(globalInput(idx), 'omitnan');
            end

            setField = sprintf('ss%d', loadVal);
            U.(familyName).(setField).CDA = cdaCurve;
            U.(familyName).(setField).Alpha = alphaCurve;
            U.(familyName).(setField).GlobalAlpha = globalCurve;
            U.(familyName).(setField).nRaw = condL.nRaw + condR.nRaw;
            U.(familyName).(setField).nKeep = condL.nKeep + condR.nKeep;
        end
    end

    for feati = 1:numel(features)
        featName = features{feati};
        for fami = 1:numel(families)
            famName = families{fami};
            for seti = 1:numel(setFields)
                setName = setFields{seti};
                replication.(featName).(famName).(setName)(si,:) = U.(famName).(setName).(featName);
            end
        end
    end
end

summary = struct();
summary.subjects = subjects;
summary.features = features;
summary.families = families;
summary.setFields = setFields;
for feati = 1:numel(features)
    featName = features{feati};
    for fami = 1:numel(families)
        famName = families{fami};
        for seti = 1:numel(setFields)
            setName = setFields{seti};
            X = replication.(featName).(famName).(setName);
            summary.(featName).(famName).(setName).mean = mean(X, 1, 'omitnan');
            summary.(featName).(famName).(setName).sem = std(X, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(X), 1));
        end
    end
end

save(fullfile(resultDir, 'data3_univariate_replication.mat'), 'replication', 'summary');

figure('Color', 'w', 'Position', [100 100 980 760]);
plotRows = {'CDA', 'Alpha', 'GlobalAlpha'};
plotSets = {'ss3', 'ss6'};
for ri = 1:numel(plotRows)
    featName = plotRows{ri};
    for ci = 1:numel(plotSets)
        setName = plotSets{ci};
        subplot(numel(plotRows), numel(plotSets), (ri-1)*numel(plotSets)+ci);
        hold on;
        for fami = 1:numel(families)
            famName = families{fami};
            y = summary.(featName).(famName).(setName).mean;
            e = summary.(featName).(famName).(setName).sem;
            if strcmp(famName, 'color')
                errorbar(1:numel(y), y, e, '-o', 'LineWidth', 1.3);
            else
                errorbar(1:numel(y), y, e, '--s', 'LineWidth', 1.3);
            end
        end
        xlim([0.75 numel(y)+0.25]);
        xlabel('Serial position');
        ylabel(featName);
        title(sprintf('%s %s', featName, upper(setName)));
        set(gca, 'YDir', 'reverse');
        if ri == 1 && ci == 1
            legend({'Color', 'Letter'}, 'Location', 'best');
        end
        box off;
        hold off;
    end
end
saveas(gcf, fullfile(figDir, 'data3_univariate_replication.png'));
close(gcf);

stateFiles = dir(fullfile(stateDir, 'sub*.mat'));
stateFiles = data3_filter_subject_mat_files(stateFiles, data3_subject_filter());
if ~isempty(stateFiles)
    setsize6Only = struct();
    for feati = 1:numel(features)
        setsize6Only.(features{feati}) = nan(numel(stateFiles), 6);
    end

    for si = 1:numel(stateFiles)
        load(fullfile(stateFiles(si).folder, stateFiles(si).name), 'state');
        for feati = 1:numel(features)
            featName = features{feati};
            for li = 1:6
                setsize6Only.(featName)(si, li) = mean(state.univariate.(featName)(state.load == li), 'omitnan');
            end
        end
    end

    figure('Color', 'w', 'Position', [100 100 900 320]);
    for feati = 1:numel(features)
        featName = features{feati};
        X = setsize6Only.(featName);
        y = mean(X, 1, 'omitnan');
        e = std(X, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(X), 1));
        subplot(1, numel(features), feati);
        errorbar(1:6, y, e, '-o', 'LineWidth', 1.5);
        set(gca, 'YDir', 'reverse');
        xlim([0.75 6.25]);
        xlabel('Serial position / load');
        ylabel(featName);
        title(sprintf('%s setsize6-only', featName));
        box off;
    end
    saveas(gcf, fullfile(figDir, 'data3_univariate_setsize6_only.png'));
    close(gcf);
end

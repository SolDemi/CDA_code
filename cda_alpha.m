%% Rebuild long-window single-trial ERP/CDA/alpha from author-provided erp_singletrial.mat
% Output keeps single-trial, single-channel data.
% No averaging across trials or channels.
% The saved cda/alpha structs keep both:
%   1) collapsed load fields: diff_2, diff_6
%   2) side-specific load fields: diff_L_2, diff_R_2, diff_L_6, diff_R_6
% The side-specific fields are required for within-side load decoding.

dbstop if error
clear; clc;

%%%% all subjects
subjects = [1:8,10:18,20:26,28:34,36:47,49:52,54:63,65:68,70,73:74,...
    76,78:102,104:112,115:121,123:133,135:138,140:147,150,152:156,...
    158:160,162:172,174:175,178:181,184:191,194,196:198,203:204,206:219];

datadir = pwd;
maindir = erase(datadir,'\code');
homedir = [maindir,'\data_raw\'];
rawfiles = dir(homedir);
output_dir = [maindir '\cda_alpha\'];
if ~isfolder(output_dir)
    mkdir(output_dir)
end

conditions_LR = {'L_C2';'L_C6';'L_S2';'L_S6';'R_C2';'R_C6';'R_S2';'R_S6'};
condPairs = {'C2','L_C2','R_C2'; ...
             'C6','L_C6','R_C6'; ...
             'S2','L_S2','R_S2'; ...
             'S6','L_S6','R_S6'};

%% Long window settings
pre_timepoint  = 250;   % -1000 ms at 250 Hz
post_timepoint = 249;   % +996 ms
erp_baseline_window_ms   = [-200 0];      % for ERP/CDA voltage baseline
alpha_baseline_window_ms = [-400 -100];   % for alpha power baseline
frep = [8 12];

% Absolute left/right posterior CDA channels
leftElecLabels  = {'O1','OL','P3','PO3','T5'};
rightElecLabels = {'O2','OR','P4','PO4','T6'};

bad_subs = cellfun(@(x) any(isletter(x)), {rawfiles.name});
bad_subs(1:2) = 1;
subjects = rawfiles(~bad_subs);

for s = 1:length(subjects)

    clear erp cda alpha S

    sn = subjects(s).name;
    subjDir = fullfile(homedir, sn);
    inFile = fullfile(subjDir, 'erp_singletrial.mat');

    % Load author-provided ERP structure with continuous data + ARF
    S = load(inFile, 'erp');
    erp = S.erp;

    %% Basic settings
    erp.pre_timepoint  = pre_timepoint;
    erp.post_timepoint = post_timepoint;
    erp.times_ms = (-erp.pre_timepoint : erp.post_timepoint) / erp.srate * 1000;

    % ERP voltage baseline indices
    erp.baselinewindow_ms = erp_baseline_window_ms;
    erp.baseline = dsearchn(erp.times_ms', erp_baseline_window_ms');
    erp.baseline = erp.baseline(1):erp.baseline(2);

    % Exact channel order from loaded file
    erp.allChans = erp.allChans(:)';

    leftIdx  = get_channel_indices(erp.allChans, leftElecLabels);
    rightIdx = get_channel_indices(erp.allChans, rightElecLabels);

    % Relative pair labels after left/right remapping
    relPairLabels = {'O1/O2','OL/OR','P3/P4','PO3/PO4','T5/T6'};

    %% Rebuild trial codes from event stream
    erp.trialCodes = zeros(1, length(erp.eventCodes));

    for ec = 1:length(erp.eventCodes)-1
        if erp.eventCodes(ec) == 7
            switch erp.eventCodes(ec+1)
                case 12
                    erp.trialCodes(ec+1) = 1; % L_C2
                case 16
                    erp.trialCodes(ec+1) = 2; % L_C6
                case 22
                    erp.trialCodes(ec+1) = 3; % L_S2
                case 26
                    erp.trialCodes(ec+1) = 4; % L_S6
            end
        elseif erp.eventCodes(ec) == 9
            switch erp.eventCodes(ec+1)
                case 12
                    erp.trialCodes(ec+1) = 5; % R_C2
                case 16
                    erp.trialCodes(ec+1) = 6; % R_C6
                case 22
                    erp.trialCodes(ec+1) = 7; % R_S2
                case 26
                    erp.trialCodes(ec+1) = 8; % R_S6
            end
        end
    end

    %% Initialize containers
    erp.trial_raw = struct();   % raw voltage epochs, no ERP baseline correction
    erp.trial     = struct();   % baselined voltage epochs for ERP/CDA
    erp.keep_event_idx  = struct();
    erp.keep_event_time = struct();
    erp.nTrials = struct();

    for c = 1:numel(conditions_LR)
        fn = conditions_LR{c};
        erp.trial_raw.(fn) = [];
        erp.trial.(fn) = [];
        erp.keep_event_idx.(fn) = [];
        erp.keep_event_time.(fn) = [];
        erp.nTrials.(fn) = 0;
    end

    %% Extract long-window clean trials
    nPnts = size(erp.data, 2);

    for ec = 1:length(erp.eventCodes)

        tc = erp.trialCodes(ec);
        if tc == 0
            continue;
        end

        win = (erp.eventTimes(ec) - 50) : (erp.eventTimes(ec) + erp.post_timepoint); % same as the authors' code, 50 steps mean 200ms

        % Skip epochs exceeding data bounds
        if win(1) < 1 || win(end) > nPnts
            continue;
        end

        % Artifact rejection using author-provided arf
        isBlink = sum(erp.arf.blink(win)) > 1;
        isMove  = sum(erp.arf.eMove(win)) > 1;
        isBlock = sum(sum(erp.arf.blocking(:, win))) > 1;

        if isBlink || isMove || isBlock
            continue;
        end

        fn = conditions_LR{tc};
        tr = size(erp.trial_raw.(fn), 1) + 1;

        win = (erp.eventTimes(ec) - erp.pre_timepoint) : (erp.eventTimes(ec) + erp.post_timepoint);
        epochRaw = erp.data(:, win);               % channels x time
        erp.trial_raw.(fn)(tr,:,:) = epochRaw;     % trials x channels x time
        erp.keep_event_idx.(fn)(tr,1)  = ec;
        erp.keep_event_time.(fn)(tr,1) = erp.eventTimes(ec);
    end

    %% ERP baseline correction for CDA/ERP only
    for c = 1:numel(conditions_LR)
        fn = conditions_LR{c};
        X = erp.trial_raw.(fn);   % trials x channels x time

        if isempty(X)
            erp.nTrials.(fn) = 0;
            erp.trial.(fn) = [];
            continue;
        end

        baseMean = mean(X(:,:,erp.baseline), 3);                  % trials x channels
        erp.trial.(fn) = X - repmat(baseMean, [1 1 size(X,3)]);   % trials x channels x time
        erp.nTrials.(fn) = size(X,1);
    end

    %% Build single-trial CDA data without channel averaging
    cda = struct();
    cda.srate = erp.srate;
    cda.time = erp.times_ms;
    cda.relPairLabels = relPairLabels;
    cda.leftElecLabels = leftElecLabels;
    cda.rightElecLabels = rightElecLabels;
    cda.trial = struct();
    cda.trials_per_cond = zeros(1, size(condPairs,1));
    cda.trials_per_side_cond = zeros(size(condPairs,1), 2); % columns: L, R

    for p = 1:size(condPairs,1)

        outName   = condPairs{p,1};   % C2/C6/S2/S6
        leftName  = condPairs{p,2};   % L_*
        rightName = condPairs{p,3};   % R_*

        XL = erp.trial.(leftName);    % trials x channels x time, baselined
        XR = erp.trial.(rightName);

        cda.trial = add_relative_condition_fields(cda.trial, XL, XR, outName, leftIdx, rightIdx);

        cda.trials_per_cond(p) = size(cda.trial.(['diff_' outName]), 1);
        cda.trials_per_side_cond(p,:) = [size(cda.trial.(['diff_L_' outName]), 1), ...
                                         size(cda.trial.(['diff_R_' outName]), 1)];
    end

    cda.trial = add_load_level_fields(cda.trial);
    cda.trial = remove_condition_level_relative_fields(cda.trial);

    cda.trials_per_ss = [size(cda.trial.diff_2,1), size(cda.trial.diff_6,1)];
    cda.trials_per_side_load = [size(cda.trial.diff_L_2,1), size(cda.trial.diff_L_6,1); ...
                                size(cda.trial.diff_R_2,1), size(cda.trial.diff_R_6,1)]; % rows: L/R; columns: load 2/6
    cda.min_trials_per_cond = min(cda.trials_per_cond);
    cda.min_trials_per_ss   = min(cda.trials_per_ss);
    cda.min_trials_per_side_load = min(cda.trials_per_side_load, [], 'all');

    %% Alpha extraction from raw voltage epochs
    alpha = struct();
    alpha.srate = erp.srate;
    alpha.time = erp.times_ms;
    alpha.baselinewindow_ms = alpha_baseline_window_ms;
    alpha.frep = frep;
    alpha.relPairLabels = relPairLabels;
    alpha.leftElecLabels = leftElecLabels;
    alpha.rightElecLabels = rightElecLabels;
    alpha.trial = struct();
    alpha.trials_per_cond = zeros(1, size(condPairs,1));
    alpha.trials_per_side_cond = zeros(size(condPairs,1), 2); % columns: L, R

    for c = 1:numel(conditions_LR)
        fn = conditions_LR{c};
        Xraw = erp.trial_raw.(fn);   % trials x channels x time

        if isempty(Xraw)
            alpha.trial.(fn) = [];
            continue;
        end

        alpha.trial.(fn) = run_power_function_keep_trial_chan_time( ...
            Xraw, erp.srate, erp.times_ms, alpha_baseline_window_ms, frep);
        % output: trials x channels x time
    end

    % Build contra/ipsi alpha, preserving side-specific load fields.
    for p = 1:size(condPairs,1)

        outName   = condPairs{p,1};
        leftName  = condPairs{p,2};
        rightName = condPairs{p,3};

        XL = alpha.trial.(leftName);  % trials x channels x time
        XR = alpha.trial.(rightName);

        alpha.trial = add_relative_condition_fields(alpha.trial, XL, XR, outName, leftIdx, rightIdx);

        alpha.trials_per_cond(p) = size(alpha.trial.(['diff_' outName]), 1);
        alpha.trials_per_side_cond(p,:) = [size(alpha.trial.(['diff_L_' outName]), 1), ...
                                           size(alpha.trial.(['diff_R_' outName]), 1)];
    end

    alpha.trial = add_load_level_fields(alpha.trial);
    alpha.trial = rmfield(alpha.trial, intersect(conditions_LR, fieldnames(alpha.trial)));
    alpha.trial = remove_condition_level_relative_fields(alpha.trial);

    alpha.trials_per_ss = [size(alpha.trial.diff_2,1), size(alpha.trial.diff_6,1)];
    alpha.trials_per_side_load = [size(alpha.trial.diff_L_2,1), size(alpha.trial.diff_L_6,1); ...
                                  size(alpha.trial.diff_R_2,1), size(alpha.trial.diff_R_6,1)]; % rows: L/R; columns: load 2/6
    alpha.min_trials_per_cond = min(alpha.trials_per_cond);
    alpha.min_trials_per_ss   = min(alpha.trials_per_ss);
    alpha.min_trials_per_side_load = min(alpha.trials_per_side_load, [], 'all');

    %% Save
    save([output_dir sprintf('sub%s.mat',sn)], 'cda', 'alpha', '-v7.3');

    fprintf('Subject %s complete!\n', sn);
end

%% ========================= Helper functions =========================
function idx = get_channel_indices(allChans, targetLabels)
% Return channel indices in the exact order of targetLabels.
    idx = zeros(1, numel(targetLabels));
    for i = 1:numel(targetLabels)
        thisIdx = find(strcmp(allChans, targetLabels{i}), 1);
        if isempty(thisIdx)
            error('Channel %s not found.', targetLabels{i});
        end
        idx(i) = thisIdx;
    end
end

function trial = add_relative_condition_fields(trial, XL, XR, outName, leftIdx, rightIdx)
% Add contra/ipsi/diff fields for one condition, both collapsed across side
% and separated by attended side.
%
% For left-cued trials:  contra = right hemisphere, ipsi = left hemisphere.
% For right-cued trials: contra = left hemisphere,  ipsi = right hemisphere.

    contraL = XL(:, rightIdx, :);
    ipsiL   = XL(:, leftIdx,  :);
    diffL   = contraL - ipsiL;

    contraR = XR(:, leftIdx,  :);
    ipsiR   = XR(:, rightIdx, :);
    diffR   = contraR - ipsiR;

    trial.(['contra_L_' outName]) = contraL;
    trial.(['ipsi_L_'   outName]) = ipsiL;
    trial.(['diff_L_'   outName]) = diffL;

    trial.(['contra_R_' outName]) = contraR;
    trial.(['ipsi_R_'   outName]) = ipsiR;
    trial.(['diff_R_'   outName]) = diffR;

    trial.(['contra_' outName]) = cat(1, contraL, contraR);
    trial.(['ipsi_'   outName]) = cat(1, ipsiL,   ipsiR);
    trial.(['diff_'   outName]) = cat(1, diffL,   diffR);
end

function trial = add_load_level_fields(trial)
% Collapse C/S conditions within each load, while keeping side information.

    metrics = {'contra', 'ipsi', 'diff'};
    sides = {'L', 'R'};
    loadDefs = {
        '2', {'C2', 'S2'}
        '6', {'C6', 'S6'}
        };

    for li = 1:size(loadDefs,1)
        loadName = loadDefs{li,1};
        conds = loadDefs{li,2};

        for mi = 1:numel(metrics)
            metric = metrics{mi};

            % Collapsed across attended side, for backward compatibility.
            trial.(sprintf('%s_%s', metric, loadName)) = cat(1, ...
                trial.(sprintf('%s_%s', metric, conds{1})), ...
                trial.(sprintf('%s_%s', metric, conds{2})));

            % Side-specific load fields, required for within-side decoding.
            for si = 1:numel(sides)
                side = sides{si};
                trial.(sprintf('%s_%s_%s', metric, side, loadName)) = cat(1, ...
                    trial.(sprintf('%s_%s_%s', metric, side, conds{1})), ...
                    trial.(sprintf('%s_%s_%s', metric, side, conds{2})));
            end
        end
    end
end

function trial = remove_condition_level_relative_fields(trial)
% Remove intermediate C2/C6/S2/S6 relative fields after final load-level
% fields are constructed. This keeps file size manageable while preserving
% all fields needed for ordinary and within-side decoding.

    metrics = {'contra', 'ipsi', 'diff'};
    sides = {'', 'L', 'R'};
    conds = {'C2', 'C6', 'S2', 'S6'};
    removeList = {};

    for mi = 1:numel(metrics)
        for si = 1:numel(sides)
            for ci = 1:numel(conds)
                if isempty(sides{si})
                    removeList{end+1,1} = sprintf('%s_%s', metrics{mi}, conds{ci}); %#ok<AGROW>
                else
                    removeList{end+1,1} = sprintf('%s_%s_%s', metrics{mi}, sides{si}, conds{ci}); %#ok<AGROW>
                end
            end
        end
    end

    fieldsToRemove = intersect(removeList, fieldnames(trial));
    if ~isempty(fieldsToRemove)
        trial = rmfield(trial, fieldsToRemove);
    end
end

function Xout = run_power_function_keep_trial_chan_time(Xtrial, srate, time_ms, baselinewindow_ms, frep)
% Input:
%   Xtrial: trials x channels x time
% Output:
%   Xout:   trials x channels x time

    if isempty(Xtrial)
        Xout = [];
        return;
    end

    nTr = size(Xtrial,1);
    nCh = size(Xtrial,2);
    nTm = size(Xtrial,3);

    % Convert to channels x time x trials for external function
    Xin = permute(Xtrial, [2 3 1]);

    Xpow = calculate_high_gamma_power(Xin, srate, time_ms, baselinewindow_ms, frep);

    % Try to coerce output back to trials x channels x time
    Xout = coerce_to_trial_chan_time(Xpow, nTr, nCh, nTm);
end

function X = coerce_to_trial_chan_time(Xin, nTr, nCh, nTm)
% Try common dimension layouts.
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
        error(['Unexpected output size from calculate_high_gamma_power: [' ...
               num2str(sz) ']. Please check its output dimension order.']);
    end
end

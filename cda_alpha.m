%% Rebuild long-window single-trial posterior voltage/alpha from author-provided erp_singletrial.mat
% Minimal-storage version.
%
% The saved cda/alpha structs store only absolute posterior left/right
% hemisphere data by attended side and load. Derived features such as CDA
% contra-minus-ipsi, lateralized alpha, GlobalAlpha, and GlobalAlphaMean are
% computed on demand in the decoding scripts.
%
% Saved field convention, for both cda.trial and alpha.trial:
%   left_L_2   : attended-left,  load 2, left-hemisphere posterior channels
%   right_L_2  : attended-left,  load 2, right-hemisphere posterior channels
%   left_R_2   : attended-right, load 2, left-hemisphere posterior channels
%   right_R_2  : attended-right, load 2, right-hemisphere posterior channels
%   same fields for load 6.
%
% This avoids storing multiple duplicated derived arrays such as diff,
% contra/ipsi, global, and globalMean.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);

maindir = fullfile(projectRoot, 'data1');
homedir = fullfile(maindir, 'data_raw');
rawfiles = dir(homedir);
output_dir = fullfile(maindir, 'cda_alpha');
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

% Absolute left/right posterior channels
leftElecLabels  = {'O1','OL','P3','PO3','T5'};
rightElecLabels = {'O2','OR','P4','PO4','T6'};

bad_subs = cellfun(@(x) any(isletter(x)), {rawfiles.name});
bad_subs(1:2) = 1;
subjects = rawfiles(~bad_subs);

for s = 1:length(subjects)

    clear erp cda alpha S alphaCond

    sn = subjects(s).name;
    subjDir = fullfile(homedir, sn);
    inFile = fullfile(subjDir, 'erp_singletrial.mat');

    S = load(inFile, 'erp');
    erp = S.erp;

    %% Basic settings
    erp.pre_timepoint  = pre_timepoint;
    erp.post_timepoint = post_timepoint;
    erp.times_ms = (-erp.pre_timepoint : erp.post_timepoint) / erp.srate * 1000;

    erp.baselinewindow_ms = erp_baseline_window_ms;
    erp.baseline = dsearchn(erp.times_ms', erp_baseline_window_ms');
    erp.baseline = erp.baseline(1):erp.baseline(2);

    erp.allChans = erp.allChans(:)';
    leftIdx  = get_channel_indices(erp.allChans, leftElecLabels);
    rightIdx = get_channel_indices(erp.allChans, rightElecLabels);

    relPairLabels = {'O1/O2','OL/OR','P3/P4','PO3/PO4','T5/T6'};

    %% Rebuild trial codes from event stream
    erp.trialCodes = zeros(1, length(erp.eventCodes));

    for ec = 1:length(erp.eventCodes)-1
        if erp.eventCodes(ec) == 7
            switch erp.eventCodes(ec+1)
                case 12, erp.trialCodes(ec+1) = 1; % L_C2
                case 16, erp.trialCodes(ec+1) = 2; % L_C6
                case 22, erp.trialCodes(ec+1) = 3; % L_S2
                case 26, erp.trialCodes(ec+1) = 4; % L_S6
            end
        elseif erp.eventCodes(ec) == 9
            switch erp.eventCodes(ec+1)
                case 12, erp.trialCodes(ec+1) = 5; % R_C2
                case 16, erp.trialCodes(ec+1) = 6; % R_C6
                case 22, erp.trialCodes(ec+1) = 7; % R_S2
                case 26, erp.trialCodes(ec+1) = 8; % R_S6
            end
        end
    end

    %% Initialize containers
    erp.trial_raw = struct();
    erp.trial     = struct();
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

        winCheck = (erp.eventTimes(ec) - 50) : (erp.eventTimes(ec) + erp.post_timepoint);
        if winCheck(1) < 1 || winCheck(end) > nPnts
            continue;
        end

        isBlink = sum(erp.arf.blink(winCheck)) > 1;
        isMove  = sum(erp.arf.eMove(winCheck)) > 1;
        isBlock = sum(sum(erp.arf.blocking(:, winCheck))) > 1;
        if isBlink || isMove || isBlock
            continue;
        end

        fn = conditions_LR{tc};
        tr = size(erp.trial_raw.(fn), 1) + 1;

        win = (erp.eventTimes(ec) - erp.pre_timepoint) : (erp.eventTimes(ec) + erp.post_timepoint);
        erp.trial_raw.(fn)(tr,:,:) = erp.data(:, win);  % trials x channels x time
        erp.keep_event_idx.(fn)(tr,1)  = ec;
        erp.keep_event_time.(fn)(tr,1) = erp.eventTimes(ec);
    end

    %% ERP baseline correction for CDA/voltage only
    for c = 1:numel(conditions_LR)
        fn = conditions_LR{c};
        X = erp.trial_raw.(fn);

        if isempty(X)
            erp.nTrials.(fn) = 0;
            erp.trial.(fn) = [];
            continue;
        end

        baseMean = mean(X(:,:,erp.baseline), 3);
        erp.trial.(fn) = X - repmat(baseMean, [1 1 size(X,3)]);
        erp.nTrials.(fn) = size(X,1);
    end

    %% CDA/voltage: save only absolute posterior left/right channels
    cda = init_minimal_struct(erp.srate, erp.times_ms, relPairLabels, leftElecLabels, rightElecLabels);
    cda.trials_per_cond = zeros(1, size(condPairs,1));
    cda.trials_per_side_cond = zeros(size(condPairs,1), 2); % columns: L, R

    for p = 1:size(condPairs,1)
        outName   = condPairs{p,1};
        leftName  = condPairs{p,2};
        rightName = condPairs{p,3};

        XL = erp.trial.(leftName);
        XR = erp.trial.(rightName);

        cda.trial = add_absolute_condition_fields(cda.trial, XL, XR, outName, leftIdx, rightIdx);
        cda.trials_per_cond(p) = size(XL,1) + size(XR,1);
        cda.trials_per_side_cond(p,:) = [size(XL,1), size(XR,1)];
    end

    cda.trial = add_absolute_load_fields(cda.trial);
    cda.trial = remove_condition_level_absolute_fields(cda.trial);
    cda = add_minimal_count_fields(cda);

    %% Alpha: compute power, then save only absolute posterior left/right channels
    alphaCond = struct();
    for c = 1:numel(conditions_LR)
        fn = conditions_LR{c};
        Xraw = erp.trial_raw.(fn);

        if isempty(Xraw)
            alphaCond.(fn) = [];
            continue;
        end

        alphaCond.(fn) = run_power_function_keep_trial_chan_time( ...
            Xraw, erp.srate, erp.times_ms, alpha_baseline_window_ms, frep);
    end

    alpha = init_minimal_struct(erp.srate, erp.times_ms, relPairLabels, leftElecLabels, rightElecLabels);
    alpha.baselinewindow_ms = alpha_baseline_window_ms;
    alpha.frep = frep;
    alpha.globalAlphaElecLabels = [leftElecLabels, rightElecLabels];
    alpha.featureConstruction = ['Only absolute posterior left/right alpha fields are stored. ' ...
        'Lateralized alpha, GlobalAlpha, and GlobalAlphaMean are constructed on demand during decoding.'];
    alpha.trials_per_cond = zeros(1, size(condPairs,1));
    alpha.trials_per_side_cond = zeros(size(condPairs,1), 2);

    for p = 1:size(condPairs,1)
        outName   = condPairs{p,1};
        leftName  = condPairs{p,2};
        rightName = condPairs{p,3};

        XL = alphaCond.(leftName);
        XR = alphaCond.(rightName);

        alpha.trial = add_absolute_condition_fields(alpha.trial, XL, XR, outName, leftIdx, rightIdx);
        alpha.trials_per_cond(p) = size(XL,1) + size(XR,1);
        alpha.trials_per_side_cond(p,:) = [size(XL,1), size(XR,1)];
    end

    alpha.trial = add_absolute_load_fields(alpha.trial);
    alpha.trial = remove_condition_level_absolute_fields(alpha.trial);
    alpha = add_minimal_count_fields(alpha);

    %% Save
    save(fullfile(output_dir, sprintf('sub%s.mat', sn)), 'cda', 'alpha', '-v7.3');
    fprintf('Subject %s complete!\n', sn);
end

%% ========================= Helper functions =========================
function S = init_minimal_struct(srate, times, relPairLabels, leftElecLabels, rightElecLabels)
    S = struct();
    S.srate = srate;
    S.time = times;
    S.relPairLabels = relPairLabels;
    S.leftElecLabels = leftElecLabels;
    S.rightElecLabels = rightElecLabels;
    S.fieldConvention = ['left_L_2/right_L_2 = posterior left/right hemisphere channels in attended-left load-2 trials; ' ...
                         'left_R_2/right_R_2 = posterior left/right hemisphere channels in attended-right load-2 trials; same for load 6.'];
    S.trial = struct();
end

function idx = get_channel_indices(allChans, targetLabels)
    idx = zeros(1, numel(targetLabels));
    for i = 1:numel(targetLabels)
        thisIdx = find(strcmp(allChans, targetLabels{i}), 1);
        if isempty(thisIdx)
            error('Channel %s not found.', targetLabels{i});
        end
        idx(i) = thisIdx;
    end
end

function trial = add_absolute_condition_fields(trial, XL, XR, outName, leftIdx, rightIdx)
% XL = attended-left trials, XR = attended-right trials.
    trial.(['left_L_'  outName]) = XL(:, leftIdx,  :);
    trial.(['right_L_' outName]) = XL(:, rightIdx, :);
    trial.(['left_R_'  outName]) = XR(:, leftIdx,  :);
    trial.(['right_R_' outName]) = XR(:, rightIdx, :);
end

function trial = add_absolute_load_fields(trial)
    hemiFields = {'left_L', 'right_L', 'left_R', 'right_R'};
    loadDefs = {
        '2', {'C2', 'S2'}
        '6', {'C6', 'S6'}
        };

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

function trial = remove_condition_level_absolute_fields(trial)
    hemiFields = {'left_L', 'right_L', 'left_R', 'right_R'};
    conds = {'C2', 'C6', 'S2', 'S6'};
    removeList = {};

    for hi = 1:numel(hemiFields)
        for ci = 1:numel(conds)
            removeList{end+1,1} = sprintf('%s_%s', hemiFields{hi}, conds{ci}); %#ok<AGROW>
        end
    end

    fieldsToRemove = intersect(removeList, fieldnames(trial));
    if ~isempty(fieldsToRemove)
        trial = rmfield(trial, fieldsToRemove);
    end
end

function S = add_minimal_count_fields(S)
    S.trials_per_side_load = [size(S.trial.left_L_2,1), size(S.trial.left_L_6,1); ...
                              size(S.trial.left_R_2,1), size(S.trial.left_R_6,1)]; % rows: attended L/R; columns: load 2/6
    S.trials_per_ss = [size(S.trial.left_L_2,1) + size(S.trial.left_R_2,1), ...
                       size(S.trial.left_L_6,1) + size(S.trial.left_R_6,1)];
    S.min_trials_per_cond = min(S.trials_per_cond);
    S.min_trials_per_ss = min(S.trials_per_ss);
    S.min_trials_per_side_load = min(S.trials_per_side_load, [], 'all');
end

function Xout = run_power_function_keep_trial_chan_time(Xtrial, srate, time_ms, baselinewindow_ms, frep)
    if isempty(Xtrial)
        Xout = [];
        return;
    end

    nTr = size(Xtrial,1);
    nCh = size(Xtrial,2);
    nTm = size(Xtrial,3);

    Xin = permute(Xtrial, [2 3 1]);
    Xpow = calculate_hilbert_band_power(Xin, srate, time_ms, baselinewindow_ms, frep);
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

clear; clc

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);
statRoot    = fullfile(projectRoot, 'data2');

datadir = fullfile(statRoot, 'cda_alpha');
if ~isfolder(datadir)
    datadir = fullfile(projectRoot, 'cda_alpha');
end

outputdir = fullfile(statRoot, 'decoding_LDA_spatialControl', 'loadWithinSide');

modelNames = {'CDA', 'Alpha', 'GlobalAlpha', 'GlobalAlphaMean', 'NoPCA', 'PCA'};
for i = 1:numel(modelNames)
    outFolder = fullfile(outputdir, modelNames{i});
    if ~isfolder(outFolder), mkdir(outFolder); end
end

%% config
cfg = struct();
cfg.cvType = 'kfold';
cfg.trainRatio = 2/3;
cfg.nFolds = 5;
cfg.superTrial = 1;
cfg.nIter = 50;

cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';

cfg.analysisWindow = [-200 inf];
cfg.doTimeGeneralization = true;
cfg.doPCA = false;
cfg.nPCs = 5;

cfg.discrimType = 'diagLinear';
cfg.ldaEngine = 'fitcdiscr';
cfg.standardize = true;

cfg.doShuffle = 0;
cfg.balanceTrials = true;
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];
cfg.useAUC = 1;
cfg.useParallel = true;
cfg.verbose = false;
cfg.randomSeed = [];

%% single-subject within-side load decoding
files = dir(fullfile(datadir, 'sub*.mat'));

for s = numel(files):-1:1
    file = files(s).name;
    fprintf('Now Processing: %s\n', file)

    load(fullfile(datadir, file), 'cda', 'alpha')

    [includeSubject, inclusionInfo] = data2_subject_inclusion(cda);
    if ~includeSubject
        fprintf('Skip %s: original data2 criterion failed, min trials per condition = %d\n', ...
            file, inclusionInfo.minTrialCount);
        continue
    end

    Results = run_load_within_side_models(cda, alpha, cfg, @LDA_function_singleSubj);

    for mi = 1:numel(modelNames)
        modelName = modelNames{mi};
        outFile = fullfile(outputdir, modelName, file);

        out = struct();
        Results.(modelName).subjectInclusion = inclusionInfo;
        out.(modelName) = Results.(modelName);
        save(outFile, '-struct', 'out', '-v7.3')
    end
end

fprintf('LDA within-side load decoding finished. Results saved to:\n%s\n', outputdir)

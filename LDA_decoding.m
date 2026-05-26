clear; clc

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);
addpath(codeDir);
statRoot    = fullfile(projectRoot, 'data1');

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
cfg.cvType = 'holdout';
cfg.trainRatio = 2/3;
cfg.nFolds = 3;
cfg.superTrial = 1;
cfg.nIter = 100;

cfg.smooth_window = 50;
cfg.smooth_step = 50;
cfg.timeWindowMode = 'bin';

cfg.analysisWindow = [-200 inf];
cfg.doTimeGeneralization = true;
cfg.doPCA = false;
cfg.nPCs = 5;

cfg.discrimType = 'diagLinear';
cfg.ldaEngine = 'fitcdiscr';
cfg.standardize = false;

cfg.doShuffle = true;
cfg.balanceTrials = true;
cfg.balanceNPerCell = [];
cfg.balanceFactors = [];
cfg.useAUC = true;
cfg.useParallel = true;
cfg.verbose = false;
cfg.randomSeed = [];

%% single-subject within-side load decoding
files = dir(fullfile(datadir, 'sub*.mat'));

for s = numel(files):-1:1
    file = files(s).name;
    fprintf('Now Processing: %s\n', file)

    load(fullfile(datadir, file), 'cda', 'alpha')

    Results = run_load_within_side_models(cda, alpha, cfg, @LDA_function_singleSubj);

    for mi = 1:numel(modelNames)
        modelName = modelNames{mi};
        outFile = fullfile(outputdir, modelName, file);

        out = struct();
        out.(modelName) = Results.(modelName);
        save(outFile, '-struct', 'out', '-v7.3')
    end
end

fprintf('LDA within-side load decoding finished. Results saved to:\n%s\n', outputdir)

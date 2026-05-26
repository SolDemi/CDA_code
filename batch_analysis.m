%% Batch analysis
% Main entry point for the CDA project analysis scripts.
% Edit the run flags below before starting a long batch job.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);

cd(codeDir);
addpath(codeDir);
addpath(fullfile(codeDir, 'spatial_control_code'));

cfg = struct();
cfg.runData0LoadDecoding = true;
cfg.runData0SpatialControl = true;
cfg.runData1Preprocess = true;
cfg.runData1CompareCDA = true;
cfg.runData1SVM = true;
cfg.runData1LDA = true;
cfg.runGroupStats = true;
cfg.runData3Preprocess = true;

fprintf('CDA batch analysis started.\nProject root: %s\n\n', projectRoot);

%% data0: load decoding
if cfg.runData0LoadDecoding
    fprintf('Running data0 load decoding...\n');
    run_script(fullfile(codeDir, 'process_data0.m'));
end

%% data0: spatial-control analyses
if cfg.runData0SpatialControl
    fprintf('Running data0 spatial-control decoding...\n');
    run_script(fullfile(codeDir, 'spatial_control_code', 'process_spatial_control_decoding.m'));
    run_script(fullfile(codeDir, 'spatial_control_code', 'stat_plot_spatial_control.m'));
end

%% data1: rebuild CDA/alpha and decode
if cfg.runData1Preprocess
    fprintf('Building data1 CDA/alpha files...\n');
    run_script(fullfile(codeDir, 'cda_alpha.m'));
end

if cfg.runData1CompareCDA
    fprintf('Comparing rebuilt CDA with source CDA...\n');
    run_script(fullfile(codeDir, 'compareCDA.m'));
end

if cfg.runData1SVM
    fprintf('Running data1 SVM decoding...\n');
    run_script(fullfile(codeDir, 'SVM_decoding.m'));
    run_script(fullfile(codeDir, 'plot_SVM_result.m'));
end

if cfg.runData1LDA
    fprintf('Running data1 LDA decoding...\n');
    run_script(fullfile(codeDir, 'LDA_decoding.m'));
end

if cfg.runGroupStats
    fprintf('Running group-level decoding statistics...\n');
    run_script(fullfile(codeDir, 'stat_plot.m'));
end

%% data3: build CDA/alpha files for the same decoding pipeline
if cfg.runData3Preprocess
    fprintf('Building data3 CDA/alpha files...\n');
    run_script(fullfile(codeDir, 'data3_cda_alpha.m'));
end

fprintf('\nCDA batch analysis finished.\n');

function run_script(scriptPath)
    run(scriptPath);
end

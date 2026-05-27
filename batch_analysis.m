%% Batch analysis
% Main entry point for the CDA project analysis scripts.
% Edit the run flags below before starting a long batch job.

clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);

cd(codeDir);
addpath(codeDir);
addpath(fullfile(codeDir, 'spatial_control_code'));

%% data0: 
% load decoding
process_data0

% data0: spatial-control analyses
process_spatial_control_decoding
stat_plot

%% data1: rebuild CDA/alpha and decode
cda_alpha

% Comparing rebuilt CDA with source CDA
compareCDA

SVM_decoding
plot_SVM_result


LDA_decoding

stat_plot

%% data3: build CDA/alpha files for the same decoding pipeline
data3_cda_alpha

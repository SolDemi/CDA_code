%% Batch analysis
% Main entry point for the CDA project analysis scripts.
% Edit the run flags below before starting a long batch job.
clear; clc;

codeDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(codeDir);

cd(codeDir);
addpath(codeDir);
addpath(fullfile(codeDir, 'spatial_control_code'));

%% data1: 

% data1: spatial-control analyses
process_spatial_control_decoding
stat_plot

%% data2: rebuild CDA/alpha and decode
cda_alpha

% Comparing rebuilt CDA with source CDA
compareCDA

SVM_decoding
plot_SVM_result


LDA_decoding

stat_plot

%% data3: build CDA/alpha files for the same decoding pipeline
data3_cda_alpha
data3_setsize1_vs6_LDA_decoding
data3_sequential_LDA_stats_plot

data3_decision_behavior_correlation
data3_plot_decision_behavior_correlation

% agjecent segment comparison
data3_decoding_serial_position_diag_comparison

% alpha control decoding
data3_spatial_matched_alpha_decoding
data3_rect_area_alpha_decoding


data3_letter_color_cross_decoding
data3_letter_color_cross_plot

data3_segment_state_RSA

% data3: sequential item-state analyses from Wang, Rajsic, & Woodman (2019)
data3_retention_summary
data3_build_item_states
data3_univariate_replication

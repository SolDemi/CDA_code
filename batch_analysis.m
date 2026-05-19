% batch_analysis
cd D:\projects\CDA\code

%% process data0
process_data0
stat_plot

% control analysis
addpath("spatial_control_code\")
process_spatial_control_decoding
%% process data1
% calculate cda、alpha（原始数据基线-200ms，无法提取alpha频段，所以重新修改基线至-1000ms）
cda_alpha

% compare CDA（看看原始数据CDA和新生成的CDA差异）
compareCDA

% SVM decode
SVM_decoding

% LDA decode
LDA_decoding

% plot SVM result
plot_SVM_result

stat_plot

% job


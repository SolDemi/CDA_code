% batch_analysis
cd D:\projects\CDA\code
% calculate cda、alpha（原始数据基线-200ms，无法提取alpha频段，所以重新修改基线至-1000ms）
cda_alpha

% compare CDA（看看原始数据CDA和新生成的CDA差异，由于基线变化，坏段会有所不同）
compareCDA

% SVM decode
SVM_decoding

% LDA decode
LDA_decoding

% plot SVM result
plot_SVM_result

stat_plot

% job
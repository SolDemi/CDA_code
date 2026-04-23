%--------------------------------------------------------------------------
% Notes about which data files and scripts we're actually plotting for the paper! 
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% 1. Overall classification of load over time (single trials).
%--------------------------------------------------------------------------
% Plotting script: P1_plot_classification_load_diagLinear_singleTrials.m
%
% Analysis script: classify_load_linearDiag_iterations.m
% 
% Data-file:load_classify_diagLinear_iterations.mat 
% 
% - This data file is 100 iterations of randomly shuffled single-trials.
% Importantly, this version also balances the number of trials per set size
% in the training and test sets!
% 
% - The older data file "load_classify_diagLinear.mat" is only a single
% itreation, and does not properly balance the trials.
% 
%--------------------------------------------------------------------------
% 1. Generalization of classifciation over timepoints.
%--------------------------------------------------------------------------
% Script: To generate file, currently running cross temporal generalization
% for single trials (with 100 iterations, and balancing training and
% testing categories), "classify_load_linearDiag_iterations_crossTemporal.m"
% 
% Plotting file:
% P2_plot_classification_load_diagLinear_singleTrials_crossTemp.m
% 
% Data-file: 
% load_classify_diagLinear_iterations_crossTemporal.mat
% 
%--------------------------------------------------------------------------
% 2. Single-trial classification with and without lateralized predictors.
%--------------------------------------------------------------------------
%
% Analysis script:
% Classify_load_linearDiag_ContraIpsi_wholefield_combined.m
%  -- Updated on 4/2020 to balance trials and do iterations!!! 
%
% Plotting script: P3_plot_classification_load_allchans_contraipsi.m()
% 
% File: load_classify_linearDiag_iter_contraipsi.mat
%
%--------------------------------------------------------------------------
% 2. Topo plot for lateralized versus whole-field signal
%--------------------------------------------------------------------------
%
% Plotting script: PlotTopo_lateralized_wholefield.m
% 
% File: grand_erp_alltrials.mat
%
%--------------------------------------------------------------------------
% 3. Mini-block analysis
%--------------------------------------------------------------------------
% Analysis Script: classify_load_linearDiag_miniBlocks_iterations
% - updated 4/28 to ensure training and testing blocks are balanced for set
% size!! 
% 
% Plotting Script: plot_classification_load_diagLinear_byBinSize_its.m
% 
% Data-files: 
% - load_classify_diagLinear_iterations.mat 
% - load_classify_diagLinear_100it_block10.mat (block5, block20, etc.)

%--------------------------------------------------------------------------
% S1. look at color alone versus shape alone! in main analysis we were using
% both color and shape trials...
%--------------------------------------------------------------------------

% Plotting file: plot_classification_load_diagLinear_singleTrials_colorshape.m
% 
% Data files: load_classify_diagLinear_iterations_color.mat; 
% load_classify_diagLinear_iterations_shape.mat
%--------------------------------------------------------------------------
% S2 SVM versus linear classifer
%--------------------------------------------------------------------------
% Analysis script: classify_load_svm.m
% Plotting script: plot_classification_load_svm.m
% Data file: load_classify_svm.mat; load_classify_diagLinear_iterations.mat 
% 
%--------------------------------------------------------------------------
% S3 Normal or non-normal distributed values for each electrode
%--------------------------------------------------------------------------
% Analysis script: classify_load_linearDiag_iterations_norm.m 
% Plotting script: plot_normality_test.m
% Data file: 
% 
%--------------------------------------------------------------------------
% S4: Train across sides (Train L Test R)
%--------------------------------------------------------------------------
% Analysis script:
% classify_load_linearDiag_iterations_trainLtestR.m
% classify_load_linearDiag_iterations_trainRtestL.m
% Plotting script: plot_classification_bySide.m
%
% Data files: load_classify_diagLinear_trainRtestR.mat,
% load_classify_diagLinear_trainLtestL.mat,
% load_classify_diagLinear_trainRtestL.mat,load_classify_diagLinear_trainLtestR.mat
% 
%--------------------------------------------------------------------------
% S5. Classification by groups of electrodes
%--------------------------------------------------------------------------
% Analysis script: classify_load_linearDiag_iterations_changroups.m
% 
% Plotting script:
% plot_classification_load_diagLinear_singleTrials_byElectrode.m
% 
% Files:  
% - load_classify_diagLinear_iterations_chanGroups.mat
% - load_classify_diagLinear_iterations_1electrode.mat
%--------------------------------------------------------------------------
% S6. Classification for meaned and demeaned data
%--------------------------------------------------------------------------
% Analysis script: classify_load_linearDiag_iterations_mean_demean.m
%
% Plotting script:
% plot_classification_load_diagLinear_singleTrials_meanDeman.m
%
% Data file: load_classify_diagLinear_iterations_mean_demean.mat
%



%--------------------------------------------------------------------------
% S5 Leave 1 electrode out.
%--------------------------------------------------------------------------
% Analysis script: 
% classify_load_linearDiag_miniBlocks_iterations_leave1electrode.m (mini
% blocks)
% classify_load_linearDiag_iterations_leave1electrode (Single trials!)
%
% File (single trials): load_classify_diagLinear_iterations_leave1electrode.mat
%
% Plotting: plot_classification_load_diagLinear_singleTrials_byLeaveElectrode.m


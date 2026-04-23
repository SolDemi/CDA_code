%% Compare CDA: 比较自己生成的CDA数据和数据集提供的数据集是否有差异


clear,clc
maindir = erase(pwd,'code');
datadir1 = [maindir 'cda_alpha\'];
datadir2 = [maindir 'data_raw\'];
outputdir = [maindir 'decoding\'];

rawfiles = dir(datadir2);

output_dir = [maindir '\erp\'];
if ~isfolder(output_dir)
    mkdir(output_dir)
end

bad_subs = cellfun(@(x) any(isletter(x)), {rawfiles.name});
bad_subs(1:2) = 1;
good_subs = rawfiles(~bad_subs);
[cda1total2,cda2total2,cda1total6,cda2total6] = deal([]);
for i = numel(good_subs):-1:1
    subj = good_subs(i).name;

    disp(['Now Processing: Subj' subj])

    datafile2 = fullfile(datadir2, good_subs(i).name, 'erp_singletrial.mat');
    load(datafile2)
    cda2 = cda;
    cda2plot2 = mean(cda2.diff_2); cda2plot6 = mean( cda2.diff_6 );



    load([datadir1 sprintf('sub%s',subj)])
    cda1plot2 = reshape(mean( mean( cda.trial.diff_2(:,:,201:end), 1 ), 2 ),1,[] );
    cda1plot6 = reshape(mean( mean( cda.trial.diff_6(:,:,201:end), 1 ), 2 ),1,[] );
    
    if ~isempty(cda2total2)
        cda2total2 = cat(1,cda2plot2,cda2total2); % subjects
        cda2total6 = cat(1,cda2plot6,cda2total6);
        cda1total2 = cat(1,cda1plot2,cda1total2);
        cda1total6 = cat(1,cda1plot6,cda1total6);
    else
        cda2total2 = cda2plot2;
        cda2total6 = cda2plot6;
        cda1total2 = cda1plot2;
        cda1total6 = cda1plot6;
    end
end
disp("Finished!")
    % figure
    % subplot 121
    % plot( cda2.time, cda1plot6, cda2.time, cda2plot6 ), hold on
    % xline( 0, '--r','HandleVisibility','off'), yline( 0, '--r','HandleVisibility','off' ), legend({'cda1','cda2'})
    % title( 'Load 6')
    % subplot 122
    % plot( cda2.time,cda1plot2 ,cda2.time,cda2plot2 ), hold on
    % xline( 0, '--r','HandleVisibility','off' ), yline( 0, '--r','HandleVisibility','off' ), legend({'cda1','cda2'})
    % title( 'Load 2')


    %% plotting
    xlim_plot = [-200 1000];
    ylim_plot = [-1.5 1.5];
    xlabel_p  = 'Times';
    ylabel_p  = 'Baselined Potential(μv)';
    legend1   = 'Manual create CDA';
    legend2   = 'Original CDA';
    myColor1 = [38, 121, 178]./255;
    myColor2 = [235,111,41]./255;
plot_shaded_errorbar_twoCurve(cda2.time, cda1total2, cda2total2, xlim_plot,ylim_plot,xlabel_p,ylabel_p,legend1,legend2,myColor1,myColor2)
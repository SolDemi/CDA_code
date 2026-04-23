
%% load data

load task_classify.mat

cls.subjects = subjects;
nSubs = length(subjects); 

%% Plot average classification over time
figure;

hold on;

errorbar(cls.binCenters, mean(cls.acc_task_allchans), std(cls.acc_task_allchans) ./ sqrt(nSubs), 'b', 'LineWidth',3)
errorbar(cls.binCenters, mean(cls.acc_task_allchans_shuffle), std(cls.acc_task_allchans_shuffle) ./ sqrt(nSubs), 'r', 'LineWidth',3)
plot(cls.binCenters,cls.acc_task_allchans','Color',[0 0 0 .3])
errorbar(cls.binCenters, mean(cls.acc_task_allchans), std(cls.acc_task_allchans) ./ sqrt(nSubs), 'b', 'LineWidth',3)
errorbar(cls.binCenters, mean(cls.acc_task_allchans_shuffle), std(cls.acc_task_allchans_shuffle) ./ sqrt(nSubs), 'r', 'LineWidth',3)
xlim([min(cls.binCenters),max(cls.binCenters)])

L = legend('SVM','Shuffled'); set(L,'box','off')
set(gca,'FontSize',20,'LineWidth',3,'TickDir','out','box','off')
xlabel('Time (ms)')
ylabel('Classification (1- crossval errors)')

title('Classify Task (regardless of side/load)')

%%  Compare all chans, pchans only and fchans only 
figure;

hold on;

errorbar(cls.binCenters, mean(cls.acc_task_allchans), std(cls.acc_task_allchans) ./ sqrt(nSubs), 'b', 'LineWidth',3)
errorbar(cls.binCenters, mean(cls.acc_task_pchans), std(cls.acc_task_pchans) ./ sqrt(nSubs), 'g', 'LineWidth',3)
errorbar(cls.binCenters, mean(cls.acc_task_fchans), std(cls.acc_task_fchans) ./ sqrt(nSubs), 'k', 'LineWidth',3)
errorbar(cls.binCenters, mean(cls.acc_task_allchans_shuffle), std(cls.acc_task_allchans_shuffle) ./ sqrt(nSubs), 'r', 'LineWidth',3)

xlim([min(cls.binCenters),max(cls.binCenters)])

L = legend('All electrodes','Posterior chans only','Frontal chans only','Shuffled'); set(L,'box','off')
set(gca,'FontSize',20,'LineWidth',3,'TickDir','out','box','off')
xlabel('Time (ms)')
ylabel('Classification (1- crossval errors)')

title('Classify Task (regardless of side/load)')

%% Plot topo map of SVM weights

tois = ismember(cls.binCenters,600:700);

w = mean(squeeze(abs(nanmean(cls.weights_task_allchans(:,tois,:),2))));

figure;
PlotTopo_Oregon(w)
axis off square
colorbar
%% Make a gif of classification over time 
% heat figure of alll pdfs used!!
filename = [pwd,'/CDA_classification/Figures/classify_task.gif'];


figure(1); clf;

for ii = 1:size(cls.weights_task_allchans,2)
    
    % Mean! 
        w = mean(squeeze(abs(cls.weights_task_allchans(:,ii,:))));
    
        % Variabiliyt! 
%         w = std(squeeze(abs(cls.weights_task_allchans(:,ii,:))));
    
    % single subject
%     w = (squeeze(abs(cls.weights_task_allchans(3,ii,:))));
    
    PlotTopo_Oregon(w,[0,2.5])
    axis off
    colorbar
    
    set(gcf,'Position',[1394 580 627 607])
    set(gca,'FontSize',20)
    title(sprintf('Time = %d',cls.binCenters(ii)));
    
    figure(1)
    drawnow
    
    
    % gif stuff!!
    frame = getframe(1); % should match figure number assigned
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if ii == 1
        imwrite(imind,cm,filename,'gif','Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end
%% Correlate average classification with average change detection performance! 

% load behavior
load K_cda_changedetection

allsubs = 1:219;
commonsubs = allsubs(ismember(allsubs,cd_cda.subNum) & ismember(allsubs,cls.subjects));
% align subjects
k = cd_cda.K_ave(ismember(cd_cda.subNum,commonsubs));


tois = ismember(cls.binCenters,400:1000);
classDat = nanmean(cls.acc_task_allchans(ismember(cls.subjects,commonsubs),tois),2);

figure;
do_correlation_plot(k,classDat)
xlabel('K (ERP task)')
ylabel('Side classification accuracy')

%% Correlate average classification with average change detection performance! 

% load behavior
load K_beh48_changedetection

allsubs = 1:219;
commonsubs = allsubs(ismember(allsubs,cd_48.subNum) & ismember(allsubs,cls.subjects));
% align subjects
k = cd_48.K_ave(ismember(cd_48.subNum,commonsubs));


tois = ismember(cls.binCenters,400:1000);
classDat = nanmean(cls.acc_task_allchans(ismember(cls.subjects,commonsubs),tois),2);

figure;
do_correlation_plot(k,classDat)
xlabel('K (separate task)')
ylabel('Side classification accuracy')


%% Plot subset of subjects

%% Plot average classification over time
figure;

nSubs = 20; 

hold on;

errorbar(cls.binCenters, mean(cls.acc_task_allchans(1:nSubs,:)), std(cls.acc_task_allchans(1:nSubs,:)) ./ sqrt(nSubs), 'b', 'LineWidth',3)
errorbar(cls.binCenters, mean(cls.acc_task_allchans_shuffle(1:nSubs,:)), std(cls.acc_task_allchans_shuffle(1:nSubs,:)) ./ sqrt(nSubs), 'r', 'LineWidth',3)
plot(cls.binCenters,cls.acc_task_allchans(1:nSubs,:)','Color',[0 0 0 .3])
errorbar(cls.binCenters, mean(cls.acc_task_allchans(1:nSubs,:)), std(cls.acc_task_allchans(1:nSubs,:)) ./ sqrt(nSubs), 'b', 'LineWidth',3)
errorbar(cls.binCenters, mean(cls.acc_task_allchans_shuffle(1:nSubs,:)), std(cls.acc_task_allchans_shuffle(1:nSubs,:)) ./ sqrt(nSubs), 'r', 'LineWidth',3)
xlim([min(cls.binCenters),max(cls.binCenters)])

L = legend('SVM','Shuffled'); set(L,'box','off')
set(gca,'FontSize',20,'LineWidth',3,'TickDir','out','box','off')
xlabel('Time (ms)')
ylabel('Classification (1- crossval errors)')

title('Classify Task (regardless of side/load)')

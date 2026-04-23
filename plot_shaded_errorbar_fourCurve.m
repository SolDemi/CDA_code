function plot_shaded_errorbar_fourCurve(xData1,yData1,yData2,yData3,yData4,xlim,ylim,myXlabel,myYlabel,myLegend, myColor)
    
%     myXlabel = 'Time (ms)';
%     myYlabel = 'ACC';
%     myColor1 = [38, 121, 178]./255;
%     myColor2 = [235,111,41]./255;
%     myColor1 = [92, 181, 152]./255;
%     myColor2 = [249, 107, 101]./255;

% myColor  = [38, 121, 178;
%             235,111,41;
%             92, 181, 152;
%             249, 107, 101]./255;
    
    
    
    yDataSE1 = std(yData1,[],1)./sqrt(size(yData1,1));
    yDataSE2 = std(yData2,[],1)./sqrt(size(yData2,1));
    yDataSE3 = std(yData3,[],1)./sqrt(size(yData3,1));
    yDataSE4 = std(yData4,[],1)./sqrt(size(yData4,1));
    
%     A = shadedErrorBar(xData1,mean(yData1),[yDataSE1;yDataSE1],'',1); 
    A = shadedErrorBar(xData1,mean(yData1),[yDataSE1;yDataSE1]); 
    hold on
    A.mainLine.LineWidth = 1.5;
    A.mainLine.Color = myColor(1,:);
    A.patch.FaceColor = myColor(1,:);
    A.patch.FaceAlpha = 0.15;
    set(A.edge,'LineStyle','none')
    
    B = shadedErrorBar(xData1,mean(yData2),[yDataSE2;yDataSE2]); 
    B.mainLine.LineWidth = 1.5;
    B.mainLine.Color = myColor(2,:);
    B.patch.FaceColor = myColor(2,:);
    B.patch.FaceAlpha = 0.15;
    set(B.edge,'LineStyle','none')
    

    C = shadedErrorBar(xData1,mean(yData3),[yDataSE3;yDataSE3]); 
    C.mainLine.LineWidth = 1.5;
    C.mainLine.Color = myColor(3,:);
    C.patch.FaceColor = myColor(3,:);
    C.patch.FaceAlpha = 0.15;
    set(C.edge,'LineStyle','none')


    D = shadedErrorBar(xData1,mean(yData4),[yDataSE4;yDataSE4]); 
    D.mainLine.LineWidth = 1.5;
    D.mainLine.Color = myColor(4,:);
    D.patch.FaceColor = myColor(4,:);
    D.patch.FaceAlpha = 0.15;
    set(D.edge,'LineStyle','none')

    
    xlabel(myXlabel,'FontSize',14)
    ylabel(myYlabel,'FontSize',14)
    set(gca,'FontSize',14)
    line(get(gca,'Xlim'), [0 0],'Color','k','Linestyle','--');

    box off 
    axis tight
    set(gca,'Ylim',ylim,'Xlim',xlim,'FontSize',14)
    line([0 0], get(gca,'Ylim'),'Color','k','Linestyle','--');
    
    if ~isempty(myLegend)
        [~, hobj, ~, ~] = legend([A.mainLine,B.mainLine,C.mainLine,D.mainLine],myLegend{1},myLegend{2},myLegend{3},myLegend{4},"AutoUpdate","off",'Location','Best');
        hl = findobj(hobj,'type','line');
        set(hl,'LineWidth',1.5);
        h2 = findobj(hobj,'type','text');
        set(h2,'FontSize',14);
        legend boxoff
    end
    

%     B.Annotation.LegendInformation.IconDisplayStyle = 'off';

    
end
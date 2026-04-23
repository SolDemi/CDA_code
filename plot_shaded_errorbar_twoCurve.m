function plot_shaded_errorbar_twoCurve(xData1,yData1,yData2,xlim,ylim,myXlabel,myYlabel,myLegend1,myLegend2, myColor1, myColor2)
    
%     myXlabel = 'Time (ms)';
%     myYlabel = 'ACC';
%     myColor1 = [38, 121, 178]./255;
%     myColor2 = [235,111,41]./255;
%     myColor1 = [92, 181, 152]./255;
%     myColor2 = [249, 107, 101]./255;
    
    
    
    yDataSE1 = std(yData1,[],1,"omitnan")./sqrt(size(yData1,1));
    yDataSE2 = std(yData2,[],1,"omitnan")./sqrt(size(yData2,1));
    
%     A = shadedErrorBar(xData1,mean(yData1),[yDataSE1;yDataSE1],'',1); 
    A = shadedErrorBar(xData1,mean(yData1),[yDataSE1;yDataSE1]); 
    hold on
    A.mainLine.LineWidth = 1.5;
    A.mainLine.Color = myColor1;
    A.patch.FaceColor = myColor1;
    A.patch.FaceAlpha = 0.5;
    set(A.edge,'LineStyle','none')
    
    B = shadedErrorBar(xData1,nanmean(yData2),[yDataSE2;yDataSE2]); hold on
    B.mainLine.LineWidth = 1.5;
    B.mainLine.Color = myColor2;
    B.patch.FaceColor = myColor2;
    B.patch.FaceAlpha = 0.5;
    set(B.edge,'LineStyle','none')
    
    
    xlabel(myXlabel,'FontSize',14)
    ylabel(myYlabel,'FontSize',14)
    set(gca,'FontSize',14)
    line(get(gca,'Xlim'), [0 0],'Color','k','Linestyle','--');

    box off 
    axis tight
    set(gca,'Ylim',ylim,'Xlim',xlim,'FontSize',14)
    line([0 0], get(gca,'Ylim'),'Color','k','Linestyle','--');
    
    if ~isempty(myLegend1)
        [~, hobj, ~, ~] =  legend([A.mainLine,B.mainLine],myLegend1,myLegend2,"AutoUpdate","off",'Location','Best');
        hl = findobj(hobj,'type','line');
        set(hl,'LineWidth',1.5);
        h2 = findobj(hobj,'type','text');
        set(h2,'FontSize',14);
        legend boxoff
    end
    

%     B.Annotation.LegendInformation.IconDisplayStyle = 'off';

    
end
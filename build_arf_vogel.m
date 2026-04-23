function arf = build_arf_vogel

% Define channels
arf.chans = 1:22;
arf.chanLabels = {'PO3','PO4','F3','F4','C3','C4','P3','P4','O1','O2','OL','OR','T3','T4','T5','T6','POz','Cz','Fz','Pz','VEM','HEM'};

arf.chansLH = 3:2:18; % Left hemisphere channels
arf.chansRH = 4:2:18; % Right hemisphere channels
arf.numChanPairs = 8;
arf.chanPairLabels = {'F3/F4','C3/C4','P3/P4','PO3/PO4','O1/O2','OL/OR','T3/T4','T5/T6'};

% Define artifact rejection criteria
rHEM = 15; % horizontal eye movement
rVEM = 75; % vertical eye movement
rPOT = 55; % parietal/occipital/temporal electrodes
rFrZ = 450; % frontal/medial electrodes

thresh = ones(1,length(arf.chans)).*rFrZ;
for c = 1:length(arf.chans)
    a = arf.chans(c);
    if (a > 5 && a < 18) 
        thresh(c) = rPOT;
        criter{c} = 'block';
    else
        if a == 21
            thresh(c) = rVEM;
            criter{c} = 'blink';
        elseif a == 22
            thresh(c) = rHEM;
            criter{c} = 'eyemove';
        else
            criter{c} = 'block';
        end
    end
end

arf.thresh = thresh;
arf.criter = criter;
        
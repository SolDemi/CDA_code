function erp = arf_vogel(erp)
% function for rejecting trials

for i = 1:size(erp.data,1) % go through each channel and apply correct arf criteria
    if i == 21 % VEM channel
        % Check for blinks
        erp.arf.blink = ppa_vogel(erp,i);
        
    elseif i == 22 % HEM channel
        % Check for eye movements
        erp.arf.eMove = step_vogel(erp,i);
    else
        erp.arf.blocking(i,:) = blocking(erp,i);
    end
end

function epochRef = data3_reref_tp9_remove(epoch, chanLabels, refLabel)
% Convert data3 from online right-mastoid reference to average-mastoid
% reference. The recorded TP9 channel is left mastoid minus right mastoid;
% subtracting TP9/2 implements rereferencing to (left+right mastoids)/2.
% TP9 is then removed, matching the original data3 author scripts.

refIdx = find(strcmpi(chanLabels, refLabel), 1);
epochRef = epoch - epoch(refIdx,:,:) ./ 2;
keepIdx = setdiff(1:size(epoch,1), refIdx, 'stable');
epochRef = epochRef(keepIdx,:,:);
end

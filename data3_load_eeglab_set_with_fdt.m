function EEG = data3_load_eeglab_set_with_fdt(setFile)

S = load(setFile, '-mat');
EEG = S.EEG;
if isnumeric(EEG.data)
    return;
end

fdtFile = fullfile(fileparts(setFile), EEG.data);
fid = fopen(fdtFile, 'rb');
cleaner = onCleanup(@() fclose(fid));

raw = fread(fid, [double(EEG.nbchan), double(EEG.pnts) * double(EEG.trials)], 'float32=>double');
EEG.data = reshape(raw, [double(EEG.nbchan), double(EEG.pnts), double(EEG.trials)]);
end

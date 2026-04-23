clear erp;
datadir = cd;
maindir = erase(datadir,'\code');
datadir = [maindir,'\data_raw'];
rawfiles = dir(datadir);

output_dir = [maindir '\erp\'];
if ~isfolder(output_dir)
    mkdir(output_dir)
end

bad_subs = cellfun(@(x) any(isletter(x)), {rawfiles.name});
bad_subs(1:2) = 1;
good_subs = rawfiles(~bad_subs);

for i = 1:numel(good_subs)
    subj = good_subs(i).name;

    disp(['Now Processing: Subj' subj])

    datafiles = dir(fullfile(datadir, good_subs(i).name, 'raw*'));
    datafiles = datafiles(~[datafiles.isdir]);

    if isempty(datafiles)
        warning('No raw files found for subject %s', subj);
        continue
    end

    erp = [];
    for nraw = 1:numel(datafiles)
        thisfile = fullfile(datafiles(nraw).folder, datafiles(nraw).name);

        tmp_erp = convertERPSS(thisfile);
        tmp_erp.data = tmp_erp.data / 12.5;

        tmp_erp.arf = build_arf_vogel;
        tmp_erp = arf_vogel(tmp_erp);

        tmp_erp.filtered_data = eegfilt(tmp_erp.data, 250, 0, 30);

        if isempty(erp)
            erp = tmp_erp;
        else
            erp = merge_erp_blocks(erp, tmp_erp);
        end
    end

    save(fullfile(output_dir, ['sub' subj '.mat']), 'erp');
end
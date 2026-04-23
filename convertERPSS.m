function erp = convertERPSS(filename)

fprintf('Importing Data\n');

% read ERPSS format
[erp.data,events,header] = read_erpss(filename);
erp.srate = header.srate;
erp.nChans = size(erp.data,1); % Number of Channels
erp.rateAcq = 4; % Rate of Data Acquisition
erp.event = struct( 'type', { events.event_code }, 'latency', {events.sample_offset});
erp.pnts = size(erp.data,2);
erp.calib = 0;
erp = resampERPs(erp, 250);

erp.eventCodes = cell2mat({erp.event.type}); % Event Codes
erp.eventTimes = round(cell2mat({erp.event.latency})); % Event Times
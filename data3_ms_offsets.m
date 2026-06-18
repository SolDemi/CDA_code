function offsets = data3_ms_offsets(windowMs, srate)

offsets = round(windowMs(1) / 1000 * srate) : round(windowMs(2) / 1000 * srate);
end

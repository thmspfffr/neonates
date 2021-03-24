%% load data

inps = 0;
output_folder = 'E:\PFC micro\results\modelledSTTC\'; % where to save
lags = [5, 10, 50, 100, 500, 1000] / 1000;
num_units = 400;

for inp = inps
    
    load(['E:\PFC micro\results\modelled_spikes\neonates_spiking_inp', num2str(inp), '_v2.mat'])
        
    %% compute xCorr and/or Tiling Coefficient
    PYRs = outp.spikesE;
    INs = outp.spikesI;
    length_rec = [0 size(PYRs, 2) / 1000];
    
    for ampa = 1 : size(PYRs, 3)
        for gaba = 1 : size(PYRs, 4)
            PYRspikes = PYRs(:, :, ampa, gaba, 1);
            INspikes = INs(:, :, ampa, gaba, 1);
            spike_matrix = cat(1, PYRspikes, INspikes);
            STTC = NaN(num_units * (num_units - 1) / 2, numel(lags));
            for lag_idx = 1 : numel(lags)
                pair_idx = 1;
                for unit1 = 1 : num_units
                    spikes1 = find(spike_matrix(unit1, :)) / 1000;
                    num_spikes1 = numel(spikes1);
                    for unit2 = unit1 + 1 : num_units
                        spikes2 = find(spike_matrix(unit2, :)) / 1000;
                        num_spikes2 = numel(spikes2);
                        STTC(pair_idx, lag_idx) =  getSTTC(num_spikes1, num_spikes2, ...
                            lags(lag_idx), length_rec, spikes1, spikes2);
                        pair_idx = pair_idx + 1;
                    end
                end
            end
            firing_rate = log10(sum(spike_matrix, 2) / length_rec(2));
            save([output_folder, 'STTC_v2_', num2str(ampa), '_', num2str(gaba), '_', num2str(inp + 1)], 'STTC')
            save([output_folder, 'fr_v2_', num2str(ampa), '_', num2str(gaba), '_', num2str(inp + 1)], 'firing_rate')
            disp(['done with parameters ', num2str(ampa), '-', num2str(gaba), '-', num2str(inp + 1)])
        end
    end
end

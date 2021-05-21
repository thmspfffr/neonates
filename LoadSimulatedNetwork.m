clear
root_dir = 'D:\model_v6\';
lags = [25, 50, 100, 1000] / 1000;
params.Fs = 200;
params.fpass = [1 100];
window = [1 0.5];
freqs4slope = [30 100];
to_plot = 0;
VrevE = 0;
VrevI = -0.08;
gLeak = 25;
idx = 1;
version = 6;

for iAmpa = 15
    for iGaba = 0 : 20
        disp(['loading_GABA_', num2str(iGaba), '_AMPA_', num2str(iAmpa)])
        load([root_dir, 'eLife_v' num2str(version), '_GABA_', num2str(iGaba), '_AMPA_', num2str(iAmpa)])
        FR_int(iAmpa + 1, iGaba + 1) = mean(sum(spikes_IN, 2)) / 10;
        FR_pyrs(iAmpa + 1, iGaba + 1) = mean(sum(spikes_PYR, 2)) / 10;
        spike_matrix = cat(1, spikes_PYR, spikes_IN);
        length_rec = [0 size(spike_matrix, 2) / 1000];
        num_units = size(spike_matrix, 1);
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
        STTC_avg(iAmpa + 1, iGaba + 1, :) = nanmedian(STTC);
        LFP = sum(cat(1, gI .* (voltage_PYR- VrevI) - gE .* (voltage_PYR - VrevE)));
        LFP = ZeroPhaseFilter(LFP, params.Fs, [0.1, 99]);
        [PSD, ~, freqs] = mtspecgramc(LFP, window, params);
        [slope(iAmpa + 1, iGaba + 1), ~] = getLFPslope(median(PSD), freqs, freqs4slope, to_plot);
        g_rel(iAmpa + 1, iGaba + 1) = g;
    end
end

%% 

figure; scatter(g_rel(:), slope(:), 50, 'filled'); alpha(0.5)
xlabel('gI/gE'); ylabel('1/f slope')
set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')

for col_idx = 1 : size(STTC_avg , 3)
    STTC2plot = STTC_avg(:, :, col_idx);
    pct90 = prctile(STTC2plot(:), 90);
    pct10 = prctile(STTC2plot(:), 10);
    figure; scatter(g_rel(:), STTC2plot(:), 50, 'filled'); alpha(0.5)
    xlabel('gI/gE'); ylabel('STTC')
    set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')
    figure; imagesc(STTC2plot, [pct10, pct90]); colormap(reds)
    ylabel('AMPA'); xlabel('GABA')
    set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')
end

figure; scatter(g_rel(:), FR_int(:), 50, 'filled'); alpha(0.5)
hold on; scatter(g_rel(:), FR_pyrs(:), 50, 'filled'); alpha(0.5)
xlabel('gI/gE'); ylabel('firing rate')
title('blue=IN - orange=PYRs')
set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')

%% put in a structure to save extracted results

summary.FR_IN = FR_int;
summary.FR_IN = FR_pyrs;
summary.gI_over_gE = g_rel;
summary.slope = slope;
summary.STTC25 = STTC_avg(:, :, 1);
summary.STTC50 = STTC_avg(:, :, 2);
summary.STTC100 = STTC_avg(:, :, 3);
summary.STTC1000 = STTC_avg(:, :, 4);
summary.columns = 'AMPA_levels';
summary.rows = 'GABA_levels';
summary.freqs4slope = freqs4slope;

save([root_dir, 'summary'], 'summary')

%% plot PSD for fixed AMPA level

YlGnBu = cbrewer('seq', 'YlGnBu', 100);
figure; hold on
arrayfun( @(i) plot(freqs, PiEsDi(i, :), ...
    'Color', YlGnBu(round(100 / 21 * i), :), 'LineWidth', 3 ), 1:5:21 );
set(gca, 'YScale', 'log', 'XScale', 'log'); set(gca, 'FontName', 'Arial')
set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14)
xlim([2 95]); ylim([0.0015 0.025])
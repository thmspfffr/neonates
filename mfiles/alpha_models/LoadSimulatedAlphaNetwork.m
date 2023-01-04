
clear
root_dir = 'E:\neonates\alpha_models\results\alpha_v';
params.Fs = 200;
params.fpass = [0.5 100];
window = [4 0.5];
freqs4slope0 = [1 10];
freqs4slope1 = [30 90];
to_plot = 0;
VrevE = 0;
VrevI = -0.08;
idx = 1;
version = 3;

for iAmpa = 0 : 9
    for iGaba = 0 : 9
        disp(['loading_GABA_', num2str(iGaba), '_AMPA_', num2str(iAmpa)])
        load([root_dir, num2str(version), '_GABA_', num2str(iGaba), '_AMPA_', num2str(iAmpa)])
        LFP = sum(cat(1, abs(gI .* (voltage_PYR- VrevI)) - gE .* (voltage_PYR - VrevE)));
        LFP = ZeroPhaseFilter(LFP, params.Fs, [0.1, 99]);
        [PSD, ~, freqs] = mtspecgramc(LFP, window, params);
        medPSD(idx, :) = smooth(nanmedian(PSD), 25);
        [slope0(iAmpa + 1, iGaba + 1), ~] = getLFPslope(smooth(median(PSD), 25), freqs, freqs4slope0, to_plot);
        [slope1(iAmpa + 1, iGaba + 1), ~] = getLFPslope(smooth(median(PSD), 25), freqs, freqs4slope1, to_plot);
        g_rel(iAmpa + 1, iGaba + 1) = g;
        power_alpha(iAmpa + 1, iGaba + 1) = sum(median(PSD(:, freqs > 8 & freqs < 12)));
        iGabas(iAmpa + 1, iGaba + 1) = iGaba;
        iAmpas(iAmpa + 1, iGaba + 1) = iAmpa;
        idx = idx + 1;
    end
end

%% 

figure; scatter(g_rel(:), -slope0(:), 50, 'filled')
hold on;
scatter(g_rel(:), -slope1(:), 50, 'filled')
plot(fitlm(g_rel(:), -slope0(:)))
plot(fitlm(g_rel(:), -slope1(:)))
set(gca, 'FontName', 'Arial'); set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14);
xlabel('Relative inhibition'); ylabel('1/f slope')


%% plot PSDs

norm_g = (g_rel(:) - nanmin(g_rel(:))) / (nanmax(g_rel(:)) - nanmin(g_rel(:)));
norm_g(norm_g == 0) = nanmin(norm_g(norm_g > 0)) / 2;

cmap = cbrewer('seq', 'YlGnBu', 100);
figure; hold on
arrayfun( @(i) plot(freqs, medPSD(i, :) ./ sum(medPSD(i, :)), ...
    'Color', cmap(round(100 * norm_g(i)), :), 'LineWidth', 3 ), 1:size(medPSD,1) );
set(gca, 'YScale', 'log', 'XScale', 'log'); set(gca, 'FontName', 'Arial')
set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14);
xlim([2 95]); xlabel('Frequency (Hz)'); ylabel('simulated LFP Power')

idx = 1;
for iAmpa = 1 : 5
    figure; hold on
    for iGaba = 1 : 5
        plot(freqs, medPSD(idx, :) ./ sum(medPSD(idx, :)), 'Color', ...
            cmap(round(100 / 5 * iGaba), :), 'LineWidth', 3 )
        idx = idx + 1;
    end
    set(gca, 'YScale', 'log', 'XScale', 'log'); set(gca, 'FontName', 'Arial')
    set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14);
    xlim([2 95]); xlabel('Frequency (Hz)'); ylabel('simulated LFP Power')
end
    

figure
loglog(freqs, median(medPSD), 'r', 'LineWidth', 3)
set(gca, 'YScale', 'log', 'XScale', 'log'); set(gca, 'FontName', 'Arial')
set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14);
xlim([2 95]); xlabel('Frequency (Hz)'); ylabel('simulated LFP Power')

close all
figure; scatter(g_rel(:), power_alpha(:), 50, iGabas(:), 'filled')
set(gca, 'FontName', 'Arial'); set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14);
xlabel('Relative inhibition'); ylabel('8-12 PSD power')
colormap(inferno()); colorbar()

figure; scatter(g_rel(:), power_alpha(:), 50, iAmpas(:), 'filled')
set(gca, 'FontName', 'Arial'); set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14);
xlabel('Relative inhibition'); ylabel('8-12 PSD power')
colormap(inferno()); colorbar()
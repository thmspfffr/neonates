
clear
root_dir = 'E:\neonates\alpha_models\results\alpha_v';
params.Fs = 200;
params.fpass = [0.5 100];
window = [4 0.5];
freqs4slope0 = [1 10];
freqs4slope1 = [30 80];
to_plot = 0;
VrevE = 0;
VrevI = -0.08;
idx = 1;
version = 2;

for iAmpa = 0 : 4
    for iGaba = 0 : 4
        disp(['loading_GABA_', num2str(iGaba), '_AMPA_', num2str(iAmpa)])
        load([root_dir, num2str(version), '_GABA_', num2str(iGaba), '_AMPA_', num2str(iAmpa)])
        LFP = sum(cat(1, gI .* (voltage_PYR- VrevI) - gE .* (voltage_PYR - VrevE)));
        LFP = ZeroPhaseFilter(LFP, params.Fs, [0.1, 99]);
        [PSD, ~, freqs] = mtspecgramc(LFP, window, params);
        medPSD(idx, :) = smooth(nanmean(PSD), 25);
%         figure;
%         loglog(freqs, smooth(nanmean(PSD), 25))
%         title(['GABA ', num2str(iGaba), ' AMPA ', num2str(iAmpa), ...
%             ' idx ' num2str(idx)])
%         xlim([0.5 80])
%         waitforbuttonpress
%         close
        [slope0(iAmpa + 1, iGaba + 1), ~] = getLFPslope(smooth(mean(PSD), 25), freqs, freqs4slope0, to_plot);
        [slope1(iAmpa + 1, iGaba + 1), ~] = getLFPslope(smooth(mean(PSD), 25), freqs, freqs4slope1, to_plot);
        g_rel(iAmpa + 1, iGaba + 1) = g;
        idx = idx + 1;
    end
end

% figure; loglog(freqs, medPSD', 'k'); hold on
% loglog(freqs, mean(medPSD), 'r', 'LineWidth', 3)
% xlim([2 80]); ylim([1.3 3.6]*10^(-6))

figure; scatter(g_rel(:), -slope0(:), 50, 'filled')
hold on; xlim([10 23])
scatter(g_rel(:), -slope1(:), 50, 'filled')



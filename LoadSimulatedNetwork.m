clear

addpath(genpath('~/Documents/MATLAB/chronux_2_12/'))

lags = [1000] / 1000;
params.Fs = 200;
params.fpass = [1 100];
window = [1 0.5];
freqs4slope = [30 100];
to_plot = 0;
VrevE = 0;
VrevI = -0.08;
gLeak = 25;
idx = 1;
version = 9;


root_dir = sprintf('~/neonates/proc/v%d/',version);

for iAmpa = 0:20
  
  fn = sprintf('LoadSimulatedNetwork_AMPA%d_v%d',iAmpa,version);
  if tp_parallel(fn,root_dir,1,0)
    continue
  end
  iAmpa
  if iAmpa == 14
    ntr = 1;
  else
    ntr = 1;
  end
  
  for itr = 0 : ntr-1
    
    
    for iGaba = 0 : 20
        disp(['loading_GABA_', num2str(iGaba), '_AMPA_', num2str(iAmpa), '_TR_', num2str(itr)])
        load([root_dir, 'elife_model_v' num2str(version), '_GABA', num2str(iGaba), '_AMPA', num2str(iAmpa), '_tr', num2str(itr)])
        out.FR_int(iGaba + 1,itr+1) = mean(sum(spikes_IN, 2)) / 50;
        out.FR_pyrs(iGaba + 1,itr+1) = mean(sum(spikes_PYR, 2)) / 50;
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
        out.STTC_avg(iGaba + 1, itr+1, :) = nanmedian(STTC);
        LFP = sum(cat(1, gI .* (voltage_PYR- VrevI) - gE .* (voltage_PYR - VrevE)));
        LFP = ZeroPhaseFilter(LFP, params.Fs, [0.1, 99]);
        [PSD, ~, freqs] = mtspecgramc(LFP, window, params);
        [out.slope(iGaba + 1, itr+1), ~] = getLFPslope(median(PSD), freqs, freqs4slope, to_plot);
        out.g_rel(iGaba + 1, itr+1) = g;
    end
  end
  save([root_dir fn '.mat'],'out')
end


exit



%% 

% figure; scatter(g_rel(:), slope(:), 50, 'filled'); alpha(0.5)
% xlabel('gI/gE'); ylabel('1/f slope')
% set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')
% 
% for col_idx = 1 : size(STTC_avg , 3)
%     STTC2plot = STTC_avg(:, :, col_idx);
%     pct90 = prctile(STTC2plot(:), 90);
%     pct10 = prctile(STTC2plot(:), 10);
%     figure; scatter(g_rel(:), STTC2plot(:), 50, 'filled'); alpha(0.5)
%     xlabel('gI/gE'); ylabel('STTC')
%     set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')
%     figure; imagesc(STTC2plot, [pct10, pct90]); colormap(reds)
%     ylabel('AMPA'); xlabel('GABA')
%     set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')
% end
% 
% figure; scatter(g_rel(:), FR_int(:), 50, 'filled'); alpha(0.5)
% hold on; scatter(g_rel(:), FR_pyrs(:), 50, 'filled'); alpha(0.5)
% xlabel('gI/gE'); ylabel('firing rate')
% title('blue=IN - orange=PYRs')
% set(gca, 'FontSize', 14); set(gca, 'FontName', 'Arial')

%% put in a structure to save extracted results
freqs4slope = [30 100];
clear summary
idx=0:20; %idx(idx==14)=[];
version = 9;
root_dir = sprintf('~/neonates/proc/v%d/',version);
for iAmpa = idx
  fn = sprintf('LoadSimulatedNetwork_AMPA%d_v%d',iAmpa,version);
  load([root_dir fn '.mat'])
  
  summary.FR_IN(:,iAmpa+1) = out.FR_int(:,1);
  summary.FR_PYR(:,iAmpa+1) = out.FR_pyrs(:,1);
  summary.gI_over_gE(:,iAmpa+1)  = out.g_rel(:,1);
  summary.slope(:,iAmpa+1)  = out.slope(:,1);
  % summary.STTC25 = STTC_avg(:, :, 1);
  % summary.STTC50 = STTC_avg(:, :, 2);
  % summary.STTC100 = STTC_avg(:, :, 3);
  summary.STTC1000(:,iAmpa+1)  = out.STTC_avg(:,1);
  summary.columns = 'AMPA_levels';
  summary.rows = 'GABA_levels';
  summary.freqs4slope = freqs4slope;
  
  if size(out.slope,2)>1
    iAmpa
    summary.slope_trials = out.slope;
    summary.STTC1000_trials = out.STTC_avg;
    summary.gIgE_trials = out.g_rel;
  end
    
end
save(sprintf([root_dir, 'summary_v%d'],version), 'summary')

%% plot PSD for fixed AMPA level

YlGnBu = cbrewer('seq', 'YlGnBu', 100);
figure; hold on
arrayfun( @(i) plot(freqs, PiEsDi(i, :), ...
    'Color', YlGnBu(round(100 / 21 * i), :), 'LineWidth', 3 ), 1:5:21 );
set(gca, 'YScale', 'log', 'XScale', 'log'); set(gca, 'FontName', 'Arial')
set(gca, 'TickDir', 'out'); set(gca, 'FontSize', 14)
xlim([2 95]); ylim([0.0015 0.025])
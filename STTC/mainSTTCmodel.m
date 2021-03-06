%% load data

inps = 0:2;
output_folder = '~/neonates/proc/v1/'; % where to save
lags = [5, 10, 50, 100, 500, 1000] / 1000;
num_units = 400;
v=1;

for inp = 1:2%inps
  
  
  load(sprintf('~/neonates/proc/neonates_spiking_inp%d_v%d.mat',inp,v))
  
  %% compute xCorr and/or Tiling Coefficient
  PYRs = outp.spikesE;
  INs = outp.spikesI;
  length_rec = [0 size(PYRs, 2) / 1000];
  
  for ampa = 1 : size(PYRs, 3)
    for gaba = 1 : size(PYRs, 4)
      
      
      fn = sprintf('mainSTTCmodel_ampa%d_gaba%d_inp%d_v%d',ampa,gaba,inp,v);
      if tp_parallel(fn,output_folder,1,0)
        continue
      end
      
      fprintf('Processing inp%d, ampa%d, gaba%d...\n',inp, ampa,gaba)

      
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


%%
clear sttc

v=1
for inp = 1:3
for iampa=1:41
  for igaba=1:42
    igaba
    load(sprintf('/home/tpfeffer/neonates/proc/v%d/STTC_v2_%d_%d_%d.mat',v,iampa,igaba,inp))
    sttc(iampa,igaba,inp,:)=squeeze(nanmedian(STTC));
    load(sprintf('/home/tpfeffer/neonates/proc/v%d/fr_v2_%d_%d_%d.mat',v,iampa,igaba,inp))
    fr(iampa,igaba,inp,:) = mean(firing_rate);
  end
end
end

%%
figure_w

inp = 2;
AMPA_mods   = linspace(2,6,41);
GABA_mods   = linspace(0.7,4.8,42);
%     #runtime     = 30000.0 * ms 

for i = 1 : 6
subplot(3,2,i)
imagesc(sttc(:,:,inp,i),[0 .2]); axis square
colormap(plasma)
set(gca,'xtick',1:10:length(GABA_mods),'xticklabel',GABA_mods(1:10:length(GABA_mods)),'fontsize',6,'ydir','normal'); 
set(gca,'ytick',1:10:length(AMPA_mods),'yticklabel',AMPA_mods(1:10:length(AMPA_mods)),'fontsize',6,'ydir','normal'); 
tp_editplots
xlabel('GABA')
ylabel('AMPA')
end

print(gcf,'-dpdf',sprintf('~/neonates/plots/neonates_sttc_inp%d_v%d.pdf',inp,v))

figure_w
subplot(3,2,1)
imagesc(fr(:,:,inp),[log10(0) log10(2)]); axis square
colormap(plasma)
set(gca,'xtick',1:10:length(GABA_mods),'xticklabel',GABA_mods(1:10:length(GABA_mods)),'fontsize',6,'ydir','normal'); 
set(gca,'ytick',1:10:length(AMPA_mods),'yticklabel',AMPA_mods(1:10:length(AMPA_mods)),'fontsize',6,'ydir','normal'); 
tp_editplots
xlabel('GABA')
ylabel('AMPA')


print(gcf,'-dpdf',sprintf('~/neonates/plots/neonates_firingrate_inp%d_v%d.pdf',inp,v))





v =1;

clear tmp1 tmp2 
bg = 0
inmda = 0;
%%
v=3
clear outp frE frI sttcE pxx slp

% inputs      = np.linspace(0.7,1.1,3)
% AMPA_mods   = np.linspace(2,6,41)
% NMDA_mods   = np.linspace(1,1,0/0.1+1)
% GABA_mods   = np.linspace(0.7,4.8,42)
    
% outp.spikestE = zeros(320,30000,21,22,'uint8');
% outp.spikesI = zeros(80,30000,21,22,'uint8');
% outp.r =  zeros(19,22,3);
% outp.fr =  zeros(19,22,3);
for iinp = 0
for iampa = 0:20
  iampa
  for inmda=0
    for igaba = 0:45
      for itr = 0
  
   
%        
        frE(iampa+1,igaba+1,iinp+1,inmda+1)  = h5read(sprintf('~/neonates/proc/v%d/neonates_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'/frE');
        
        frI(iampa+1,igaba+1,iinp+1,inmda+1)  = h5read(sprintf('~/neonates/proc/v%d/neonates_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'/frI');
%         outp.sttc(:,inmda+1,iampa+1,igaba+1,iinp+1)=nanmean(nanmean(sttc,2),3);
        sttcE(iampa+1,igaba+1,iinp+1,inmda+1)  = h5read(sprintf('~/neonates/proc/v%d/neonates_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'/sttcE');
        pxx  = h5read(sprintf('~/neonates/proc/v%d/neonates_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'/pxx');
                
        fxx  = h5read(sprintf('~/neonates/proc/v%d/neonates_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'/fxx');

               pxx= pxx(fxx>3 & fxx< 40);
               fxx = fxx((fxx>3 & fxx< 40));
               
               X = [ones(length(fxx),1) log10(fxx)];
            Y = log10(pxx);
            tmp = X\Y; 
          slp(iampa+1,igaba+1,iinp+1,inmda+1) = tmp(2);

%         spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_I');
%         outp.spikesI(:,:,iampa+1,igaba+1)=uint8(spikes_tmp(:,1:80)');
%         outp.fr(iampa+1,igaba+1,iinp+1,itr+1)=mean(sum(spikes_tmp)/30);
%         r = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_corr_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spt_E_corr');
% %         
%         outp.r(iampa+1,igaba+1,iinp+1,itr+1)=r;
        end
      end
    end
end
%   save(sprintf('~/neonates/proc/neonates_spiking_inp%d_v%d.mat',iinp,v),'outp','-v7.3')

end




%% PLOT FR for INT AND PYR

 inputs      = np.linspace(0.7,1.1,3)
AMPA_mods   = np.linspace(2,6,41)
NMDA_mods   = np.linspace(1,1,0/0.1+1)
GABA_mods   = np.linspace(0.7,4.8,42)

figure_w;

subplot(2,2,1)
imagesc(squeeze(frE),[0 2]); axis square; colorbar
set(gca,'xtick',log10(freqoi(1:4:end)),'xticklabel',freqoi(1:4:end))

colormap(plasma)

subplot(2,2,2)
imagesc(squeeze(frI),[0 5]); axis square; colorbar
colormap(plasma)

subplot(2,2,3)
imagesc(squeeze(frE<1),[0 2]); axis square; colorbar

subplot(2,2,4)
imagesc(squeeze(frI<2),[0 2]); axis square; colorbar


%%
v = 1;

% inputs      = np.linspace(0.7,1.1,3)
% AMPA_mods   = np.linspace(2,6,41)
% NMDA_mods   = np.linspace(1,1,0/0.1+1)
% GABA_mods   = np.linspace(0.7,4.8,42)
% runtime     = 30000.0 * ms 
    
clear outp
outp.spikesE = zeros(320,30000,21,22,'uint8');
outp.spikesI = zeros(80,30000,21,22,'uint8');
% outp.r =  zeros(19,22,3);
% outp.fr =  zeros(19,22,3);
for iinp = 2
for iampa = 0:40
  iampa
    for igaba = 0:41
      for itr = 0
  
%         spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_I');
%         outp.spikesI(:,:,iampa+1,igaba+1,iinp,itr+1)=single(spikes_tmp(:,1:80)');
%         
        spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_E');
        outp.spikesE(:,:,iampa+1,igaba+1)=uint8(spikes_tmp(:,1:320)');
        spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_I');
        outp.spikesI(:,:,iampa+1,igaba+1)=uint8(spikes_tmp(:,1:80)');
%         outp.fr(iampa+1,igaba+1,iinp+1,itr+1)=mean(sum(spikes_tmp)/30);
%         r = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_corr_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spt_E_corr');
% %         
%         outp.r(iampa+1,igaba+1,iinp+1,itr+1)=r;
%         
      end
    end
end
  save(sprintf('~/neonates/proc/neonates_spiking_inp%d_v%d.mat',iinp,v),'outp','-v7.3')

end

%%
v=3

clear outp
outp.spikesE = zeros(320,600000,1,16,'uint8');
outp.spikesI = zeros(80,600000,1,16,'uint8');
% outp.r =  zeros(19,22,3);
% outp.fr =  zeros(19,22,3);
for iinp = 0
for iampa = 0:4
  iampa
    for igaba = 0:20
      for itr = 0
  
%         spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_I');
%         outp.spikesI(:,:,iampa+1,igaba+1,iinp,itr+1)=single(spikes_tmp(:,1:80)');
%         
        spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_E');
        outp.spikesE(:,:,iampa+1,igaba+1)=uint8(spikes_tmp(:,1:320)');
        spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_I');
        outp.spikesI(:,:,iampa+1,igaba+1)=uint8(spikes_tmp(:,1:80)');
%         outp.fr(iampa+1,igaba+1,iinp+1,itr+1)=mean(sum(spikes_tmp)/30);
%         r = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_corr_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spt_E_corr');
% %         
%         outp.r(iampa+1,igaba+1,iinp+1,itr+1)=r;
%         
      end
    end
end
  save(sprintf('~/neonates/proc/neonates_spiking_inp%d_v%d.mat',iinp,v),'outp','-v7.3')

end



%% VOLTAGES

v = 1;

clear  slope

for iampa = 0:40
  iampa
  for iinp = 0:2
    for igaba = 0:41
      for itr = 0
        
        voltageE = hdf5read(sprintf('~/neonates/proc/v1/neonates_network_voltage_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',iampa,inmda, igaba, iinp,itr,v),'volt_E');
        voltageI = hdf5read(sprintf('~/neonates/proc/v1/neonates_network_voltage_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',iampa,inmda, igaba, iinp,itr,v),'volt_I');
        voltage = cat(2,voltageE,voltageI);
        [pxx,fxx]=pwelch(sum(voltage,2),hanning(400),200,2:0.25:40,100);
        
        tmp = tp_dfa(voltage,[1 20],100,0.5,15);
        outp.dfa(iampa+1,igaba+1,iinp+1,itr+1)=mean(tmp.exp);
        
        X = [ones(1,length(fxx))' log10(fxx)'];
        Y = log10(pxx)';
        tmp = X\Y;
        outp.slope(iampa+1,igaba+1,iinp+1,itr+1)= tmp(2);
        outp.pow(:,:,iampa+1,igaba+1,iinp+1,itr+1) = [fxx; pxx];
%         
        spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_E');
        outp.fr(iampa+1,igaba+1,iinp+1,itr+1)=mean(sum(spikes_tmp)/30);
%         
        r = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_corr_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spt_E_corr');
        outp.r(iampa+1,igaba+1,iinp+1,itr+1)=r;

        
      end
    end
  end
end


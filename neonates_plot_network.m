

v =1;

clear tmp1 tmp2 
bg = 0
inmda = 0;
%%
clear outp
outp.spikesE = zeros(320,30000,21,22,3,'single');
% outp.spikesI = zeros(80,30000,21,22,3,'single');
% outp.r =  zeros(19,22,3);
% outp.fr =  zeros(19,22,3);

for iampa = 0:20
  iampa
  for iinp = 0:2
    for igaba = 0:21
      for itr = 0
  
%         spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_I');
%         outp.spikesI(:,:,iampa+1,igaba+1,iinp,itr+1)=single(spikes_tmp(:,1:80)');
%         
        spikes_tmp = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spike_train_E');
        outp.spikesE(:,:,iampa+1,igaba+1,iinp+1,itr+1)=single(spikes_tmp(:,1:320)');
%         
%         outp.fr(iampa+1,igaba+1,iinp+1,itr+1)=mean(sum(spikes_tmp)/30);
%         r = hdf5read(sprintf('~/neonates/proc/v%d/neonates_network_corr_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5',v,iampa,inmda, igaba, iinp,itr,v),'spt_E_corr');
% %         
%         outp.r(iampa+1,igaba+1,iinp+1,itr+1)=r;
%         
      end
    end
  end
end

% save(sprintf('~/neonates/proc/neonates_spiking_v%d.mat',v),'outp')


%% VOLTAGES

v = 7;

clear tmp1 tmp2  slope
bg = 0
for i_inh = 1:10
  for itr = 1:5
    
    d1 = hdf5read(sprintf('~/pupmod/decision_network/proc/pupmod_decision_network_inh%d_inp0_bg0_tr%d_voltage_v%d.h5',i_inh,itr,v),'volt_D1');
%     d2 = hdf5read(sprintf('~/pupmod/decision_network/proc/pupmod_decision_network_inh%d_inp0_bg0_tr%d_voltage_v%d.h5',i_inh,itr,v),'volt_D2');
    
[pxx(:,i_inh,itr),fxx]=pwelch((mean(d1,2)+mean(d2,2))./2,hanning(4000),2000,2:0.25:40,1000);
  tmp = nanmean(pxx(:,i_inh,itr),3);


      X = [ones(1,length(fxx))' log10(fxx)'];
    Y = log10(tmp);
    tmp = X\Y;   
   slope(i_inh,itr)= tmp(2);

  end
  

end


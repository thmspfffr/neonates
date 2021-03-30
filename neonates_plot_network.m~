

v =1;

clear tmp1 tmp2 
bg = 0
inmda = 0;
%%
clear outp
outp.spikesE = zeros(320,30000,21,22,'uint8');
outp.spikesI = zeros(80,30000,21,22,'uint8');
% outp.r =  zeros(19,22,3);
% outp.fr =  zeros(19,22,3);
for iinp = 0:2
for iampa = 0:20
  iampa
    for igaba = 0:21
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


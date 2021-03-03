
% PARAMETERS FROM PYTHON SCRIPT (run_hierarchical_model.py)
% ---------------------
% VERSION 3: explore parameters
% ---------------------
% v = 3
% inh_mod = 1
% all_inp_mod = numpy.array([1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43])
% all_bg_mod = numpy.array([1.18, 1.19, 1.20, 1.21, 1.22])
% bin_size = 20
% ---------------------
% VERSION 7: FINAL VERSION - see plot_network.m
% ---------------------
% v = 7;
% all_inh_mod = numpy.array([0.995, 0.996, 0.997, 0.998, 0.999, 1, 1.001, 1.002, 1.003, 1.004, 1.005])
% all_inp_mod = numpy.array([1.39])
% all_bg_mod = numpy.array([1.20])
% bin_size = 100
% ntrls = 20;
% ---------------------


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

save(sprintf('~/neonates/proc/neonates_spiking_v%d.mat',v),'outp')

%% DOMINANCE
inp = 0;

clear dur count dom_dur
for i_inh = 0:10
  for itr = 0:19

    tmp_cnt1 = 0;
    tmp_cnt2 = 0;

    d1 = hdf5read(sprintf('~/pupmod/decision_network/proc/pupmod_decision_network_inh%d_inp%d_bg0_tr%d_v%d.h5',i_inh,inp,itr,v),'rate_D1');
    d1 = lowpass(d1,1,10);
    d2 = hdf5read(sprintf('~/pupmod/decision_network/proc/pupmod_decision_network_inh%d_inp%d_bg0_tr%d_v%d.h5',i_inh,inp,itr,v),'rate_D2');
    d2 = lowpass(d2,1,10);
    % compute durations
    % --------------
    k1=bwlabel((d1>d2));
    k2=bwlabel((d2>d1));
    for i=1:max(k1)
      tmp_cnt1(i)=sum(k1==i);
    end
    for i=1:max(k2)
      tmp_cnt2(i)=sum(k2==i);
    end
    
    dur{i_inh+1}{itr+1} = [tmp_cnt1 tmp_cnt2];
    % --------------
    
    dom_dur(i_inh+1,itr+1) = mean(dur{i_inh+1}{itr+1})./10;
    count(i_inh+1,itr+1) = max(k1)+max(k2);
    end
end

count = squeeze(count);
%%
%
m = mean(count,2);
s = std(count,[],2);
% eb(1,:) = mean(dom_dur,2)+s;
% eb(2,:) = mean(dom_dur,2)-s;
% figure; set(gcf,'color','w'); 
subplot(2,2,3);hold on
plot(m,'k.','markersize',10)

for i = 1 : 11
  line([i i],[m(i)-s(i) m(i)+s(i)],'color','k')
end
% shadedErrorBar([],mean(count,2),[s'; s']);
set(gca,'xtick',1:11,'xticklabel',[-0.5:0.1:0.5])
% set(gca,'ytick',-0.1:0.2:1.8,'yticklabel',num2cell(round((10.^(-0.1:0.2:1.8)*10))./10))
plot(6,mean(count(6,:),2),'o')
tp_editplots
xlabel('\Delta(Feedback inhibition) [in %]'); ylabel('Number of transitions')
axis([0 12 -20 350]); axis square

print(gcf,'-dpdf',sprintf('~/pupmod/decision_network/plots/pupmod_decisionnetwork_domdur_v%d.pdf',v))

%% PLOT OTHER BACKGROUND INPUT PARAMETERS

% PARAMETERS FROM PYTHON SCRIPT (run_hierarchical_model.py)
% ---------------------
% VERSION 3: explore parameters
% ---------------------
% v = 3
% inh_mod = 1
% all_inp_mod = numpy.array([1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43])
% all_bg_mod = numpy.array([1.18, 1.19, 1.20, 1.21, 1.22])
% bin_size = 20



v = 3;

clear tmp1 tmp2 
% bg = 0
for inp = 0:6
  for bg_input = 0:4
    
    d1 = hdf5read(sprintf('~/pupmod/decision_network/proc/pupmod_decision_network_inp%d_bg%d_v%d.h5',inp,bg_input,v),'rate_D1');
    d2 = hdf5read(sprintf('~/pupmod/decision_network/proc/pupmod_decision_network_inp%d_bg%d_v%d.h5',inp,bg_input,v),'rate_D2');
    d1 = lowpass(d1,1,10);
    d2 = lowpass(d2,1,10);
%     t = (1:3000)./10;
%     for iseg = 1 : (15000/5)
%       tmp1(iseg) = mean(d1((iseg-1)*5+1:(iseg)*5));
%       tmp2(iseg) = mean(d2((iseg-1)*5+1:(iseg)*5));
%     end
%     d1 = tmp1'; d2 = tmp2';
    
    figure; set(gcf,'color','w');
    subplot(2,2,1); hold on; title(sprintf('i%d,bg%d: SR = %.3f',i,bg,mean(abs(diff(d1>d2)))))
    a=[d1 d2];
    a=round(a.*10)./10;
    [N,X]=hist3(a,'ctrs',{0:0.5:35,0:0.5:35});
    lab=round(X{1});
    
    imagesc(X{1},X{2},N,[0 20]); axis square; set(gca,'ydir','normal','tickdir','out')
    axis([0 35 0 35]); colormap(plasma)
    line([0 150],[0 150],'color','w','linestyle',':')
    xlabel('Firing rate (D1) [Hz]'); ylabel('Firing rate (D2) [Hz]')
    set(gca,'xtick',[0:5:35],'xticklabel',[0:5:35])
    tp_editplots
    
    subplot(2,2,2); hold on; title('Population Time series')
    plot(0.2:0.2:3000,d1,'color',[0.5 0.5 0.5]);
    plot(0.2:0.2:3000,d2,'k'); axis square
    axis([250 350 0 35]); set(gca,'tickdir','out');
    xlabel('Time [s]'); ylabel('Firing rate [Hz]')
    box on
    tp_editplots
    % clear d1 d2
%     print(gcf,'-dpdf',sprintf('~/pupmod/decision_network/plots/decision_network_phaseplane_inp%d_bg%d_v%d.pdf',i,bg,v))
  end
end

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


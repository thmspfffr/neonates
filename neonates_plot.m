clear summary
set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'})

% load ~/Downloads/summary.mat
load /Users/tpfeffer/Dropbox/projects/phd/neonates/summary_v9.mat

% load ~/summary_v6.mat
% summary = outp;
% 
% summary.gI_over_gE = outp.g_rel;
% summary.STTC1000 = outp.STTC_avg;

figure;

% clear summary

cmap = cbrewer('seq', 'YlGnBu', 25,'pchip'); cmap=cmap(end:-1:1,:);
cmap = cmap(3:end,:);

subplot(3,3,1); hold on
iampa = 4;

idx = 1:21; idx(idx==iampa)=[];

for ii = 1 : 21
  plot(summary.gI_over_gE(ii,idx),summary.slope(ii,idx),'o','markersize',4,'markerfacecolor',[0.7 0.7 0.7],'markeredgecolor','w')
end
for ii = 1 : 21
  ME = mean(summary.slope_trials(ii,:),2);
  ME_gIgE = mean(summary.gIgE_trials(ii,:),2); 
  plot(ME_gIgE,ME,'o','markersize',8,'markerfacecolor',cmap(ii,:),'markeredgecolor','w')
%   SEM = std(summary.slope_trials(ii,:),[],2);
%   line([ME_gIgE ME_gIgE],[ME-SEM ME+SEM],'color',cmap(ii,:),'linewidth',1)
end


[r,p]=corr(summary.gI_over_gE(:,iampa),mean(summary.slope_trials,2));
X = [ones(length(summary.gI_over_gE(:,iampa)),1) (summary.gI_over_gE(:,iampa))];
Y = mean(summary.slope_trials,2);
tmp = X\Y;

fprintf('r = %.4f | p = %.4f\n',r,p)

plot(summary.gI_over_gE(:,iampa),tmp(2).*summary.gI_over_gE(:,iampa)+tmp(1))

% lsline
xlabel('gI/gE')
ylabel('Log-Power')
% set(gca,'xtick',summary.gI_over_gE(:,iampa),'xticklabels',summary.gI_over_gE(1:end,iampa))
text(25,-0.3,sprintf('AMPA: %d',iampa))
axis square

tp_editplots
% LFP

subplot(3,3,2); hold on

% plot(summary.gI_over_gE(:,iampa),summary.STTC25(:,iampa),'o','markersize',10,'markerfacecolor','k','markeredgecolor','w')
% plot(summary.gI_over_gE(:,iampa),summary.STTC50(:,iampa),'o','markersize',10,'markerfacecolor','k','markeredgecolor','w')
% plot(summary.gI_over_gE(:,iampa),summary.STTC100(:,iampa),'o','markersize',10,'markerfacecolor','k','markeredgecolor','w')

for ii = 1 : 21

plot(summary.gI_over_gE(ii,idx),summary.STTC1000(ii,idx),'o','markersize',4,'markerfacecolor',[0.7 0.7 0.7],'markeredgecolor','w')

end
for ii = 1 : 21
  ME = mean(summary.STTC1000_trials(ii,:),2);
  ME_gIgE = mean(summary.gIgE_trials(ii,:),2); 
  plot(ME_gIgE,ME,'o','markersize',8,'markerfacecolor',cmap(ii,:),'markeredgecolor','w')
%   SEM = std(summary.STTC1000_trials(ii,:),[],2);
%   line([ME_gIgE ME_gIgE],[ME-SEM ME+SEM],'color',cmap(ii,:),'linewidth',2)
end

[r,p]=corr(summary.gI_over_gE(:,iampa),mean(summary.STTC1000_trials,2));
X = [ones(length(summary.gI_over_gE(:,iampa)),1) (summary.gI_over_gE(:,iampa))];
Y = mean(summary.STTC1000_trials,2);
tmp = X\Y;

fprintf('r = %.4f | p = %.4f',r,p)
plot(summary.gI_over_gE(:,iampa),tmp(2).*summary.gI_over_gE(:,iampa)+tmp(1))


xlabel('gI/gE')
ylabel('STTC (1s)')
axis square
axis([0 max(summary.gI_over_gE(:)) 0.05 0.4])
text(25,-0.3,sprintf('AMPA: %d',iampa))
tp_editplots
% 

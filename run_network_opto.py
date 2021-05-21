'''
LEAKY INTEGRATE AND FIRE NETWORK W. STIMULATION
----------------------
Runs a simulation (from 0s to 9s), with a stimulation period from
3s to 6s. During this period, the current slowly ramps up, mimicking
the optogenetic stimulation (or silencing) implemented experimentally
----------------------
'''

from brian import *
import numpy as np
import numpy.matlib as ml
from numpy.random import rand as rand
from numpy.random import randn as randn
import h5py
import os.path
from subprocess import call
from sys import platform
import elephant
from neo.core import SpikeTrain
from elephant.conversion import BinnedSpikeTrain
import quantities as pq
from scipy import signal

root_dir = '/home/tpfeffer/neonates/'

from make_circuit import make_circuit


if __name__ == '__main__':

    #------------------------------------------------------------------------------ 
    # Simulation parameters 
    #------------------------------------------------------------------------------ 
    # Timing 
    runtime  = 9000.0 * ms 
    stim_on  = 3000.0 * ms
    stim_off = 6000.0 * ms
    #------------------------------------------------------------------------------ 
    # VERSION 1: Parameters from run_network.py
    #------------------------------------------------------------------------------ 
    v = 1
    # Inputs: stimululus, AMPA, NMDA, GABA
    inputs      = np.array([0.9])
    AMPA_mods   = np.array([5.1]) 
    NMDA_mods   = np.array([1])
    GABA_mods   = np.linspace(0.7,6.2,56)
    ntrls       = 1
    stim_curr   = 0.2 # max stimulation current
    stim_frac   = 0.2 # fraciton of stimulated /inhibited neurons 
    #------------------------------------------------------------------------------ 
    # preallocate
    #resp = np.zeros([len(AMPA_mods), len(NMDA_mods), len(GABA_mods), len(inputs)])
    #mean_corr = np.zeros([len(AMPA_mods), len(NMDA_mods), len(GABA_mods), len(inputs)])
    myclock = EventClock(dt=1.0*ms,t=0.0*ms,order=1,makedefaultclock=False)
    voltage_clock = EventClock(dt=5.0*ms,t=0.0*ms,order=2,makedefaultclock=False)
    #------------------------------------------------------------------------------ 
    # Run simulation
    #------------------------------------------------------------------------------ 

    if not(os.path.exists('/home/tpfeffer/neonates/proc/v%d' %v)):
      os.makedirs('/home/tpfeffer/neonates/proc/v%d' %v)

    for itr in range(ntrls):
      # Loop through exp parameters
      for igaba in range(0,GABA_mods.size): 
          for iinp in range(0,inputs.size):
              for iampa in range(0,AMPA_mods.size):
                  for inmda in range(0,NMDA_mods.size):
                      for stim_protocol in range(1,3): # 1 = inhibit IN / 2 = excite IN

                        fn = os.path.expanduser(root_dir + 'proc/opto/v%d/neonates_network_opto_iampa%d_inmda%d_gaba%d_inp%d_stim%d_tr%d_v%d_processing.txt') % (v,iampa, inmda, igaba, iinp,stim_protocol,itr,v)
                        if os.path.isfile(fn)==False:
                            call(['touch', fn])
                        else:
                            continue

                        #  initialize  
                        defaultclock.reinit(t=0.0*ms)
                        myclock.reinit(t=0.0*ms)

                        clear(True) 

                        print("Computing INPUT%d, GABA%d, AMPA%d, trial%d ...") % (iinp, igaba,iampa,itr)

                        inp = inputs[0]
                        AMPA_mod = AMPA_mods[iampa]
                        NMDA_mod = NMDA_mods[inmda]
                        GABA_mod = GABA_mods[igaba]

                        Dgroups, Dconnections, Dnetfunctions, subgroups = make_circuit(inp,GABA_mod,AMPA_mod,NMDA_mod)

                        # get populations from the integrations circuit
                        popE = Dgroups['DE']
                        popI = Dgroups['DI']

                        #------------------------------------------------------------------------------
                        # Set up external stimulation (optogenetics protocol)
                        #------------------------------------------------------------------------------

                        if stim_protocol == 1: # inhibit inhibitory cells
                          stim = np.zeros([len(popI),1000 * int(stim_off-stim_on)])
                          rand_idx = np.random.choice(np.linspace(0,len(popI)-1,len(popI)),np.int(stim_frac*len(popI)),replace=False)
                          f = np.vectorize(np.int); rand_idx = f(rand_idx)
                          stim[rand_idx,:] = np.matlib.repmat(np.linspace(0,-stim_curr,1000 * int(stim_off-stim_on)),np.int(stim_frac*len(popI)),1)
                        elif stim_protocol == 2: # excite inhibitory cells
                          stim = np.zeros([len(popI),1000 * int(stim_off-stim_on)])
                          rand_idx = np.random.choice(np.linspace(0,len(popI)-1,len(popI)),np.int(stim_frac*len(popI)),replace=False)
                          f = np.vectorize(np.int); rand_idx = f(rand_idx)
                          stim[rand_idx,:] = np.matlib.repmat(np.linspace(0,stim_curr,1000 * int(stim_off-stim_on)),np.int(stim_frac*len(popI)),1)

                        @network_operation(myclock,stim_protocol)

                        def update_input():
                          if myclock.t >= stim_on and myclock.t < stim_off:
                            if stim_protocol == 1: # inhibit inhibitory cells
                              popI.I = stim[: ,int( (myclock.t - stim_on) / (1 * ms))] * nA
                            else: # excite inhibitory cells
                              popI.I = stim[: ,int( (myclock.t - stim_on) / (1 * ms))] * nA
                          else:
                            popI.I = 0.0 * nA
                            popE.I = 0.0 * nA

                        #------------------------------------------------------------------------------
                        # ---- set initial conditions (random)
                        popE.gen = popE.gen * (1 + 0.2 * rand(popE.__len__()))
                        popI.gen = popI.gen * (1 + 0.2 * rand(popI.__len__()))
                        popE.V = popE.V + rand(popE.__len__()) * 2 * mV
                        popI.V = popI.V + rand(popI.__len__()) * 2 * mV
                        #popE.I = 0.0 * nA
                        #popI.I = 0.0 * nA

                        # record spikes of excitatory/inhibitory neurons
                        Sp_E = SpikeMonitor(popE, record=True)           
                        Sp_I = SpikeMonitor(popI, record=True)

                        # record instantaneous excitatory/inhibitory population activity
                        R_E = PopulationRateMonitor(popE, bin=5*ms)
                        R_I = PopulationRateMonitor(popI, bin=5*ms)

                        # record voltage
                        Vm_E = StateMonitor(popE, 'V', record=True, clock=voltage_clock)
                        Vm_I = StateMonitor(popI, 'V', record=True, clock=voltage_clock)

                        gE = StateMonitor(popE, 'gea', record=True, clock=voltage_clock)
                        gI = StateMonitor(popE, 'gi', record=True, clock=voltage_clock)
                        
                        # record current
                        I_E = StateMonitor(popE, 'I', record=True, clock=myclock)
                        I_I = StateMonitor(popI, 'I', record=True, clock=myclock)

                        #------------------------------------------------------------------------------
                        # Run the simulation
                        #------------------------------------------------------------------------------
                        print("Running simulation...")
                        net = Network(Dgroups.values(),  Dconnections.values(), Dnetfunctions, update_input, Sp_E, Sp_I, R_E, R_I, Vm_E, Vm_I, gE, gI, I_E, I_I)
                        net.prepare()
                        net.run(runtime) 

                        print("Computing power spectra ...")
                        LFP = np.abs(np.concatenate([gE.values,gI.values])).sum(axis=0)
                        
                        fxx, pxx = signal.welch(LFP,200,window='hann')

                        # convert to array and save output has .h5            
                        spt_E = []; spt_E_idx = []
                        spt_I = []; spt_I_idx = []

                        for ineuron in range(0,len(Sp_E.spiketimes)):
                            spt_E = np.append(spt_E,Sp_E.spiketimes.values()[ineuron], axis=None)
                            spt_E_idx = np.append(spt_E_idx,ml.repmat(ineuron,1,len(Sp_E.spiketimes.values()[ineuron])), axis=None)

                        for ineuron in range(0,len(Sp_I.spiketimes)):
                            spt_I = np.append(spt_I,Sp_I.spiketimes.values()[ineuron], axis=None)
                            spt_I_idx = np.append(spt_I_idx,ml.repmat(ineuron,1,len(Sp_I.spiketimes.values()[ineuron])), axis=None)

                        spt_E = np.vstack((spt_E,spt_E_idx))
                        spt_I = np.vstack((spt_I,spt_I_idx))

                        # -------------------------------------------
                        # COMPUTE SPIKE COUNT CORRELATIONS
                        # -------------------------------------------
                        print("Computing spike count correlations...")
                        spikes = dict(); frE = np.zeros([320,3]); frI = np.zeros([80,3])

                        first_spike = 0 # start analysis at t = 0
                        for ineuron in range(0,320):
                            spikes[ineuron] = SpikeTrain(spt_E[0][(spt_E[1]==ineuron) & ((spt_E[0]>0) & (spt_E[0]<runtime))]*pq.s, t_start = 0, t_stop = runtime)
                            frE[ineuron,0] = np.sum((spikes[ineuron]>0) & (spikes[ineuron]<3))
                            frE[ineuron,1] = np.sum((spikes[ineuron]>3) & (spikes[ineuron]<6))
                            frE[ineuron,2] = np.sum((spikes[ineuron]>6) & (spikes[ineuron]<9))
                        for ineuron in range(0,80):
                            spikes[ineuron+320] = SpikeTrain(spt_I[0][(spt_I[1]==ineuron) & ((spt_I[0]>0) & (spt_I[0]<runtime))]*pq.s, t_start = 0, t_stop = runtime)
                            frI[ineuron,0] = np.sum((spikes[ineuron+320]>0) & (spikes[ineuron+320]<3))
                            frI[ineuron,1] = np.sum((spikes[ineuron+320]>3) & (spikes[ineuron+320]<6))
                            frI[ineuron,2] = np.sum((spikes[ineuron+320]>6) & (spikes[ineuron+320]<9))

                        frE=np.mean(frE,axis=0)/int(runtime)
                        frI=np.mean(frI,axis=0)/int(runtime)

                        st_baseline = []; st_stim = []; st_post = []
                        subsamp = 1
                        matidx = np.triu_indices(len(range(0,len(spikes),subsamp)),1)
                        for isp in range(0,len(spikes),subsamp):
                            st_baseline.append(spikes[isp][(spikes[isp]>0) & (spikes[isp]<=3)])
                            st_stim.append(spikes[isp][(spikes[isp]>3) & (spikes[isp]<=6)])
                            st_post.append(spikes[isp][(spikes[isp]>6) & (spikes[isp]<=9)])

                        stc=np.zeros([3,1])
                        sts_corr_baseline=BinnedSpikeTrain(st_baseline, binsize=100*pq.ms)
                        tmp1=elephant.spike_train_correlation.corrcoef(sts_corr_baseline)
                        sts_corr_stim=BinnedSpikeTrain(st_stim, binsize=100*pq.ms)
                        tmp2=elephant.spike_train_correlation.corrcoef(sts_corr_stim)
                        sts_corr_post=BinnedSpikeTrain(st_post, binsize=100*pq.ms)
                        tmp3=elephant.spike_train_correlation.corrcoef(sts_corr_post)
                        
                        # average correlations across upper triangular part
                        stc[0] = np.triu(tmp1,1).sum()/((tmp1.shape[0]**2-tmp1.shape[0])/2)
                        stc[1] = np.triu(tmp2,1).sum()/((tmp2.shape[0]**2-tmp2.shape[0])/2)
                        stc[2] = np.triu(tmp3,1).sum()/((tmp3.shape[0]**2-tmp3.shape[0])/2)

                        spike_array = np.zeros([400,9000])
                        if itr==0:
                          sts_corr_baseline=BinnedSpikeTrain(st_baseline, binsize=1*pq.ms)
                          spike_array[:,0:3000]=sts_corr_baseline.to_array()[:,0:3000]     
                          sts_corr_stim=BinnedSpikeTrain(st_stim, binsize=1*pq.ms)
                          spike_array[:,3000:6000]=sts_corr_stim.to_array()[:,3000:6000]
                          sts_corr_post=BinnedSpikeTrain(st_post, binsize=1*pq.ms)
                          spike_array[:,6000:9000]=sts_corr_post.to_array()[:,6000:9000]

                          hf = h5py.File(os.path.expanduser(root_dir + 'proc/opto/v%d/neonates_opto_spikes_iampa%d_inmda%d_gaba%d_inp%d_prot%d_tr%d_v%d.h5') % (v,iampa, inmda, igaba, iinp,stim_protocol,itr,v), 'w')
                          hf.create_dataset('spike_array', data=spike_array)
                          hf.close() 
                          
                        # stE = []; stI = []
                        # subsamp = 1
                        # matidx = np.triu_indices(len(range(0,len(spikesE),subsamp)),1)
                        # for isp in range(0,len(spikesE),subsamp):
                        #     stE.append(spikesE[isp])
                        #     stI.append(spikesI[isp])

                        # stsE=BinnedSpikeTrain(stE, binsize=1*pq.ms)
                        # stsI=BinnedSpikeTrain(stI, binsize=1*pq.ms)

                        # stsE_corr=BinnedSpikeTrain(stE, binsize=10*pq.ms)
                        # corr=elephant.spike_train_correlation.corrcoef(stsE_corr)

                        # hf = h5py.File(os.path.expanduser(root_dir + 'proc/opto/v%d/neonates_network_opto_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_stim%d_tr%d_v%d.h5') % (v,iampa, inmda, igaba, iinp,stim_protocol,itr,v), 'w')
                        # hf.create_dataset('spike_train_E', data= stsE.to_array())
                        # hf.create_dataset('spike_train_I', data= stsI.to_array())
                        # hf.close() 

                        # mean_corr = corr[matidx].mean()
                        curr = I_I.values
                        hf = h5py.File(os.path.expanduser(root_dir + 'proc/opto/v%d/neonates_opto_iampa%d_inmda%d_gaba%d_inp%d_prot%d_tr%d_v%d.h5') % (v,iampa, inmda, igaba, iinp,stim_protocol,itr,v), 'w')
                        # hf.create_dataset('sttc_all', data=sttc_all)
                        hf.create_dataset('stc', data=stc)
                        # hf.create_dataset('sttcE', data=sttcE)
                        # hf.create_dataset('sttcI', data=sttcI)
                        hf.create_dataset('frE', data=frE)
                        hf.create_dataset('frI', data=frI)  
                        hf.create_dataset('pxx', data=pxx)    
                        hf.create_dataset('fxx', data=fxx) 
                        hf.create_dataset('LFP', data=LFP)   
                        hf.create_dataset('curr',data=curr)                
                        hf.close() 



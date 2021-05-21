'''
LEAKY INTEGRATE AND FIRE CIRCUIT
----------------------
Run simulation of recurrent spiking network across wide range of parameters.
The goal of this exhaustive search is to identify parameter combinations that yield
a low spontaenous firing rate (1-5 Hz) and low spike correlations (r < 0.10). 
----------------------
Based on code from Wimmer et al. (2014), initial parameters from Wang (2002).
Last change: 29th of May 2020, loop across trials; input not through naka-rushton
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

from make_circuit1 import make_circuit
# Version
ntrls = 1

if __name__ == '__main__':
    
    #------------------------------------------------------------------------------ 
    # VERSION 1: Find plausible parameter range (coarse)
    #------------------------------------------------------------------------------ 
    v = 11
    # Inputs: stimululus, AMPA, NMDA, GABA
    inputs      = np.array([1,1.2])
    AMPA_mods   = np.array([1,1.2])
    GABA_mods   = np.array([1,1.2])
    runtime     = 8000.0 * ms 
    #------------------------------------------------------------------------------ 
    # VERSION 2: Simulate within plausible parameter range
    #------------------------------------------------------------------------------ 
    # v = 2
    # Inputs: stimululus, AMPA, NMDA, GABA
    # inputs      = np.linspace(0.7,0.9,2)
    # AMPA_mods   = np.linspace(2,6,41)
    # NMDA_mods   = np.linspace(1,1.2,2)
    # GABA_mods   = np.linspace(0.7,6.2,56)
    # runtime     = 60000.0 * ms 
    #------------------------------------------------------------------------------ 
    # VERSION 3: Same as v2, but double twice the time (and only two inputs)
    #------------------------------------------------------------------------------ 
    #v = 3
    # Inputs: stimululus, AMPA, NMDA, GABA
    #inputs      = np.linspace(0.9,0.95,1)
    #AMPA_mods   = np.linspace(0.2,5,21)
    #NMDA_mods   = np.linspace(1,1,0/0.1+1)
    #GABA_mods   = np.linspace(2,10,51)
    #runtime     = 20000.0 * ms 
    #------------------------------------------------------------------------------ 
    # VERSION 4: Same as v3, but double twice the time (and only two inputs)
    #------------------------------------------------------------------------------ 
    #v = 4
    # Inputs: stimululus, AMPA, NMDA, GABA
    #inputs      = np.linspace(0.9,0.95,1)
    #AMPA_mods   = np.linspace(0.2,5,21)
    #NMDA_mods   = np.linspace(1,1,0/0.1+1)
    #GABA_mods   = np.linspace(2,6,26)
    #runtime     = 20000.0 * ms 
    #------------------------------------------------------------------------------ 
    # VERSION 5: Same as v3, but with STTC for inh
    #------------------------------------------------------------------------------ 
    #v = 5
    # Inputs: stimululus, AMPA, NMDA, GABA
    #inputs      = np.linspace(0.9,0.95,1)
    #AMPA_mods   = np.linspace(0.2,5,21)
    #NMDA_mods   = np.linspace(1,1,0/0.1+1)
    #GABA_mods   = np.linspace(2,6,26)
    #runtime     = 20000.0 * ms 
    #------------------------------------------------------------------------------ 

    if not(os.path.exists('/home/tpfeffer/neonates/proc/v%d' %v)):
        os.makedirs('/home/tpfeffer/neonates/proc/v%d' %v)

    defaultclock = Clock(dt=0.1*ms,t=0*ms,order=0,makedefaultclock=True) # use different clock to change sampling rate
    voltage_clock = EventClock(dt=10*ms,t=0*ms,order=1,makedefaultclock=False) # use different clock to change sampling rate

    #------------------------------------------------------------------------------ 
    # Run simulation
    #------------------------------------------------------------------------------ 
    for itr in range(ntrls):
      # Loop through exp parameters
      for igaba in range(0,GABA_mods.size): 
          for iinp in range(0,inputs.size):
              for iampa in range(0,AMPA_mods.size):

                fn = os.path.expanduser(root_dir + 'proc/v%d/neonates_iampa%d_gaba%d_inp%d_tr%d_v%d_processing.txt') % (v,iampa, igaba, iinp,itr,v)
                if os.path.isfile(fn)==False:
                    call(['touch', fn])
                else:
                    continue
                #  initialize 
                defaultclock.reinit()
                voltage_clock.reinit()
                clear(True) 

                print("Computing INPUT%d, GABA%d, AMPA%d, trial%d ...") % (iinp, igaba,iampa,itr)

                inp = inputs[iinp]
                AMPA_mod = AMPA_mods[iampa]
                GABA_mod = GABA_mods[igaba]
                
                print('%d' % inp)

                Dgroups, Dconnections = make_circuit(inp,GABA_mod,AMPA_mod)

                # get populations from the integrations circuit
                popE = Dgroups['DE']
                popI = Dgroups['DI']

                DXE = Dgroups['DXE']
                # ---- set initial conditions (random)
                popE.V = popE.V + rand(popE.__len__()) * 2 * mV
                popI.V = popI.V + rand(popI.__len__()) * 2 * mV
                

                # record spikes of excitatory neurons
                Sp_E = SpikeMonitor(popE, record=True)
                # record spikes of inhibitory neurons
                Sp_I = SpikeMonitor(popI, record=True)
                # record instantaneous excitatory populations activity
                R_E = PopulationRateMonitor(popE, bin=5*ms)
                R_SE = PopulationRateMonitor(DXE, bin=5*ms)
                # record instantaneous inhibitory populations activity
                R_I = PopulationRateMonitor(popI, bin=5*ms)
                # record voltage
                Vm_E = StateMonitor(popE, 'V', record=True, clock=voltage_clock)
                Vm_I = StateMonitor(popI, 'V', record=True, clock=voltage_clock)
                # record exc. & inh. currents at E
                gE = StateMonitor(popE, 'gea', record=True, clock=voltage_clock)
                gI = StateMonitor(popE, 'gi', record=True, clock=voltage_clock)

                #------------------------------------------------------------------------------
                # Run the simulation
                #------------------------------------------------------------------------------
                print("Running simulation...")
                net = Network(Dgroups.values(),  Dconnections.values(), Sp_E, Sp_I, R_E, R_I, Vm_E, Vm_I, gE, gI)
                net.prepare()
                net.run(runtime) 

                print("Computing power spectra ...")
                # LFP = np.abs(np.concatenate([gE.values,gI.values])).mean(axis=0)
                # fxx, pxx = signal.welch(LFP,100,window='hann')

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

                # COMPUTE SPIKE COUNT CORRELATIONS
                # -------------------------------------------
                spikes = dict(); spikesI = dict(); frE = np.zeros([320,1]); frI = np.zeros([80,1])

                first_spike = 0 # start analysis at t = 0
                for ineuron in range(0,320):
                    spikes[ineuron] = SpikeTrain(spt_E[0][(spt_E[1]==ineuron) & (spt_E[0]>first_spike)]*pq.s, t_start = 0, t_stop = runtime)
                    frE[ineuron] = spikes[ineuron].shape[0]
                for ineuron in range(0,80):
                    spikes[ineuron+320] = SpikeTrain(spt_I[0][(spt_I[1]==ineuron) & (spt_I[0]>first_spike)]*pq.s, t_start = 0, t_stop = runtime)
                    frI[ineuron] = spikes[ineuron+320].shape[0]

                frE=np.mean(frE)/int(runtime)
                frI=np.mean(frI)/int(runtime)

                print('VmE: %.3f - %.3f' % (np.max(Vm_E.values),np.min(Vm_E.values)))
                print('gE: %.3f - %.3f' % (np.max(gE.values),np.min(gE.values)))

                print('FR_E = %.3f - FR_I = %.3f' % (frE, frI))

                # OMIT RUNS WHERE FIRING RATES ARE TOO HIGH 
                # (not plausible and STTC computation too costly)
                
                st = [];
                subsamp = 1
                matidx = np.triu_indices(len(range(0,len(spikes),subsamp)),1)
                for isp in range(0,len(spikes),subsamp):
                    st.append(spikes[isp])

                sts_corr=BinnedSpikeTrain(st, binsize=50*pq.ms)
                stc=elephant.spike_train_correlation.corrcoef(sts_corr)
                
                # average correlations across upper triangular part
                stc = np.triu(stc,1).sum()/((stc.shape[0]**2-stc.shape[0])/2)

                lags = np.array([0.1])
                #if frE > 1000 or frI > 1000:
                #    sttcE = np.zeros([1, lags.shape[0]])*np.nan
                #    sttcI = np.zeros([1, lags.shape[0]])*np.nan
                #    sttc_all = np.zeros([1, lags.shape[0]])*np.nan
                #    
                #    hf = h5py.File(os.path.expanduser(root_dir + 'proc/v%d/neonates_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5') % (v,iampa, inmda, igaba, iinp,itr,v), 'w')
                #    hf.create_dataset('sttc_all', data=sttc_all)
                #    hf.create_dataset('stc', data=stc)
                #    hf.create_dataset('sttcE', data=sttcE)
                #    hf.create_dataset('sttcI', data=sttcI)
                #    hf.create_dataset('frE', data=frE)
                #    hf.create_dataset('frI', data=frI)
                #    hf.create_dataset('pxx', data=pxx)
                #    hf.create_dataset('fxx', data=fxx)
                #    hf.close() 
                #    continue
                        
                sttc = np.zeros([len(spikes),len(spikes),len(lags)])
                #for ilag in range(0,len(lags)):      
                #    for ineuron in range(0,len(popE)+len(popI)):
                #        print('Computing STTC for Lag %d ms, Neuron %d' % (lags[ilag],ineuron)) 
                #        for jneuron in range(0,len(popE)+len(popI)):
                #            sttc[ineuron,jneuron,ilag]=elephant.spike_train_correlation.sttc(spikes[ineuron],spikes[jneuron],lags[ilag]*pq.s)

                sttcE = sttc[0:320,0:320,:].mean(axis=1).mean(axis=0)
                sttcI = sttc[320:400,320:400,:].mean(axis=1).mean(axis=0)
                sttc_all = sttc.mean(axis=1).mean(axis=0)

                hf = h5py.File(os.path.expanduser(root_dir + 'proc/v%d/neonates_iampa%d_gaba%d_inp%d_tr%d_v%d.h5') % (v,iampa, igaba, iinp,itr,v), 'w')
                hf.create_dataset('sttc_all', data=sttc_all)
                hf.create_dataset('stc', data=stc)
                hf.create_dataset('sttcE', data=sttcE)
                hf.create_dataset('sttcI', data=sttcI)
                hf.create_dataset('frE', data=frE)
                hf.create_dataset('frI', data=frI)  
                hf.create_dataset('pxx', data=pxx)    
                hf.create_dataset('fxx', data=fxx) 
                hf.create_dataset('LFP', data=LFP)                     
                hf.close() 



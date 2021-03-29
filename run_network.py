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

root_dir = '/home/tpfeffer/neonates/'

from make_circuit import make_circuit

# Version
v = 1
ntrls = 1;

if not(os.path.exists('/home/tpfeffer/neonates/proc/v%d' %v)):
    os.makedirs('/home/tpfeffer/neonates/proc/v%d' %v)

if __name__ == '__main__':
    
    #------------------------------------------------------------------------------ 
    # VERSION 1: Find plausible parameter range (coarse)
    #------------------------------------------------------------------------------ 
    #v = 1
    # Inputs: stimululus, AMPA, NMDA, GABA
    #inputs      = np.linspace(0.7,1.1,3)
    #AMPA_mods   = np.linspace(2,6,41)
    #NMDA_mods   = np.linspace(1,1,0/0.1+1)
    #GABA_mods   = np.linspace(0.7,4.8,42)
    #runtime     = 30000.0 * ms 
    #------------------------------------------------------------------------------ 
    # VERSION 2: Simulate within plausible parameter range
    #------------------------------------------------------------------------------ 
    v = 2
    # Inputs: stimululus, AMPA, NMDA, GABA
    inputs      = np.array([0.7])
    AMPA_mods   = np.linspace(1.8,2.2,5)
    NMDA_mods   = np.array([1])
    GABA_mods   = np.linspace(1.2,3.4,21)
    runtime     = 600000.0 * ms 
    #------------------------------------------------------------------------------ 
    # VERSION 3: Simulate within plausible parameter range
    #------------------------------------------------------------------------------ 
    v = 3
    # Inputs: stimululus, AMPA, NMDA, GABA
    inputs      = np.linspace(0.7,1.1,3)
    AMPA_mods   = np.linspace(2,6,41)
    NMDA_mods   = np.linspace(1,1,0/0.1+1)
    GABA_mods   = np.linspace(0.7,4.8,42)
    runtime     = 60000.0 * ms 
    #------------------------------------------------------------------------------ 
    # preallocate
    resp = np.zeros([len(AMPA_mods), len(NMDA_mods), len(GABA_mods), len(inputs)])
    mean_corr = np.zeros([len(AMPA_mods), len(NMDA_mods), len(GABA_mods), len(inputs)])

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
                  for inmda in range(0,NMDA_mods.size):

                      fn = os.path.expanduser(root_dir + 'proc/v%d/neonates_network_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d_processing.txt') % (v,iampa, inmda, igaba, iinp,itr,v)
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
                      inh = 1
                      AMPA_mod = AMPA_mods[iampa]
                      NMDA_mod = NMDA_mods[inmda]
                      GABA_mod = GABA_mods[igaba]
                      
                      Dgroups, Dconnections, Dnetfunctions, subgroups = make_circuit(inp,GABA_mod,AMPA_mod,NMDA_mod)

                      # get populations from the integrations circuit
                      popE = Dgroups['DE']
                      popI = Dgroups['DI']

                      # ---- set initial conditions (random)
                      popE.gen = popE.gen * (1 + 0.2 * rand(popE.__len__()))
                      popI.gen = popI.gen * (1 + 0.2 * rand(popI.__len__()))
                      popE.V = popE.V + rand(popE.__len__()) * 2 * mV
                      popI.V = popI.V + rand(popI.__len__()) * 2 * mV

                      # record spikes of excitatory neurons
                      Sp_E = SpikeMonitor(popE, record=True)
                      # record spikes of inhibitory neurons
                      Sp_I = SpikeMonitor(popI, record=True)
                      # record instantaneous excitatory populations activity
                      R_E = PopulationRateMonitor(popE, bin=5*ms)
                      # record instantaneous inhibitory populations activity
                      R_I = PopulationRateMonitor(popI, bin=5*ms)
                      # record voltage
                      Vm_E = StateMonitor(popE, 'V', record=True, clock=voltage_clock)
                      Vm_I = StateMonitor(popI, 'V', record=True, clock=voltage_clock)

                      #------------------------------------------------------------------------------
                      # Run the simulation
                      #------------------------------------------------------------------------------
                      print("Running simulation...")
                      net = Network(Dgroups.values(),  Dconnections.values(), Dnetfunctions, Sp_E, Sp_I, R_E, R_I, Vm_E, Vm_I)
                      net.prepare()
                      net.run(runtime) 

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
                      print("Computing spike count correlations...")
                      spikesE = dict()
                      spikesI = dict()
                      
                      first_spike = 0 # start analysis at t = 0
                      for ineuron in range(0,320):
                          spikesE[ineuron] = SpikeTrain(spt_E[0][(spt_E[1]==ineuron) & (spt_E[0]>first_spike)]*pq.s, t_start = 0, t_stop = runtime)
                      for ineuron in range(0,320):
                          spikesI[ineuron] = SpikeTrain(spt_I[0][(spt_I[1]==ineuron) & (spt_I[0]>first_spike)]*pq.s, t_start = 0, t_stop = runtime)

                      stE = []; stI = []
                      subsamp = 1
                      matidx = np.triu_indices(len(range(0,len(spikesE),subsamp)),1)
                      for isp in range(0,len(spikesE),subsamp):
                          stE.append(spikesE[isp])
                          stI.append(spikesI[isp])

                      stsE=BinnedSpikeTrain(stE, binsize=1*pq.ms)
                      stsI=BinnedSpikeTrain(stI, binsize=1*pq.ms)

                      stsE_corr=BinnedSpikeTrain(stE, binsize=10*pq.ms)
                      corr=elephant.spike_train_correlation.corrcoef(stsE_corr)
  
                      hf = h5py.File(os.path.expanduser(root_dir + 'proc/v%d/neonates_network_spiketrain_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5') % (v,iampa, inmda, igaba, iinp,itr,v), 'w')
                      hf.create_dataset('spike_train_E', data= stsE.to_array())
                      hf.create_dataset('spike_train_I', data= stsI.to_array())
                      hf.close() 

                      mean_corr = corr[matidx].mean()

                      print("Saving output...")
                      hf = h5py.File(os.path.expanduser(root_dir + 'proc/v%d/neonates_network_corr_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5') % (v,iampa, inmda, igaba, iinp,itr,v), 'w')
                      hf.create_dataset('spt_E_corr', data=mean_corr)
                      hf.close() 

                      volt_E = Vm_E.values
                      volt_I = Vm_I.values

                      hf = h5py.File(os.path.expanduser(root_dir + 'proc/v%d/neonates_network_voltage_iampa%d_inmda%d_gaba%d_inp%d_tr%d_v%d.h5' % (v,iampa, inmda, igaba, iinp,itr,v) ), 'w')
                      hf.create_dataset('volt_E', data=volt_E)
                      hf.create_dataset('volt_I', data=volt_I)
                      hf.close()



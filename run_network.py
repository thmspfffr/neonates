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

from brian2 import *
import numpy as np
import numpy.matlib as ml
from numpy.random import rand as rand
import h5py
import os.path
from subprocess import call
import elephant
from neo.core import SpikeTrain
from elephant.conversion import BinnedSpikeTrain
import quantities as pq
from scipy import signal
from scipy.io import savemat


root_dir = 'D:/models_neonates/results/'

#%% constants

N = 100                                      # Total number of neurons
f_inh = 0.20                                 # Fraction of inhibitory neurons
NE = int(N * (1.0 - f_inh))                  # Number of excitatory neurons (320)
NI = int(N * f_inh)                          # Number of inhibitory neurons (80)

gII_GABA = 1.0 * nS                          # Weight of inhibitory to inhibitory synapses (GABA)
d = 0.5 * ms                                 # Transmission delay of recurrent excitatory and inhibitory connections

# Connectivity - external connections
gextE = 2.5 * nS                             # Weight of external input to excitatory neurons: Increased from previous value
gextI = 1.62 * nS                            # Weight of external input to inhibitory neurons
  
# Neuron model
CmE = 0.5 * nF                               # Membrane capacitance of excitatory neurons
CmI = 0.2 * nF                               # Membrane capacitance of inhibitory neurons
gLeakE = 25.0 * nS                           # Leak conductance of excitatory neurons
gLeakI = 20.0 * nS                           # Leak conductance of inhibitory neurons
Vl = -70.0 * mV                              # Resting potential
Vt = -50.0 * mV                              # Spiking threshold
Vr = -55.0 * mV                              # Reset potential
tau_refE = 2.0 * ms                          # Absolute refractory period of excitatory neurons
tau_refI = 1.0 * ms                          # Absolute refractory period of inhibitory neurons
  
# Synapse model
VrevE = 0 * mV                               # Reversal potential of excitatory synapses
VrevI = -70 * mV                             # Reversal potential of inhibitory synapses
tau_AMPA = 2.0 * ms                          # Decay constant of AMPA-type conductances
tau_GABA = 5.0 * ms                          # Decay constant of GABA-type conductances
tau_NMDA_decay = 100.0 * ms                  # Decay constant of NMDA-type conductances
tau_NMDA_rise = 2.0 * ms                     # Rise constant of NMDA-type conductances
alpha_NMDA = 0.5 * kHz                       # Saturation constant of NMDA-type conductances

# Connectivity - local recurrent connections
gEI_AMPA = 0.04 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
gEI_NMDA = 0.13 * nS                         # Weight of excitatory to inhibitory synapses (NMDA)

#%% get spike matrix function

def get_spike_matrix(spike_monitor, num_neurons, len_stim):
    # initialize
    spike_matrix = zeros((num_neurons, len_stim + 1), dtype=bool)
    # loop over neurons that fired at least once
    for neuron_idx in unique(spike_monitor.i):
        # extract spike_times (in seconds)
        spike_times = spike_monitor.t[spike_monitor.i == neuron_idx]
        # convert them to milliseconds
        spike_times = round_(asarray(spike_times) * 1000).astype(int)
        spike_matrix[neuron_idx, spike_times] = 1
    return spike_matrix

#%% run shit 

from make_circuit import make_circuit
# Version
ntrls = 1


if __name__ == '__main__':
    #------------------------------------------------------------------------------ 
    v = 3
    # Inputs: stimululus, AMPA, NMDA, GABA
    inputs      = np.linspace(0.7,0.9,2)
    AMPA_mods   = np.linspace(3,5,5)
    NMDA_mods   = np.linspace(1.1,1.2,2)
    GABA_mods   = np.linspace(1,3,6)
    runtime     = 60000.0 * ms 

    if not(os.path.exists(root_dir + str(v))):
        os.makedirs(root_dir + str(v))

    defaultclock = Clock(dt=0.1*ms) # use different clock to change sampling rate
    voltage_clock = Clock(dt=10*ms) # use different clock to change sampling rate

    #------------------------------------------------------------------------------ 
    # Run simulation
    #------------------------------------------------------------------------------ 
    for itr in range(ntrls):
      # Loop through exp parameters
      for igaba, GABA_mod in enumerate(GABA_mods): 
          for iinp, inp in enumerate(inputs):
              for iampa, AMPA_mod in enumerate(AMPA_mods):
                  for inmda, NMDA_mod in enumerate(NMDA_mods):
                      
                      # Connectivity - local recurrent connections
                      gEE_AMPA = 0.05 * AMPA_mod * nS		         # Weight of AMPA synapses between excitatory neurons
                      gEE_NMDA = 0.165 * NMDA_mod * nS             # Weight of NMDA synapses between excitatory neurons
                      gEI_AMPA = 0.04 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
                      gEI_NMDA = 0.13 * nS                         # Weight of excitatory to inhibitory synapses (NMDA)
                      gIE_GABA = 1.99 * GABA_mod * nS              # Weight of inhibitory to excitatory synapses (GABA)
                        
                      # Inputs
                      nu_ext_exc = inp * 2000 * Hz                       # Firing rate of external Poisson input to excitatory neurons
                      nu_ext_inh = inp * 2000 * Hz                       # Firing rate of external Poisson input to inhibitory neurons
                      stim_ext = 0 * Hz

                      print('Computing input ' + str(itr) + ' GABA ' + str(igaba) + 
                            ' AMPA ' + str(iampa) + ' NMDA ' + str(inmda)) 

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
                      net = Network(Dgroups.values(),  Dconnections.values(), Dnetfunctions, Sp_E, Sp_I, Vm_E, Vm_I, gE, gI)
                      net.run(runtime) 

                      spike_matrixPYR = get_spike_matrix(Sp_E, NE,
                                           int(asarray(runtime) * 1000))
                      spike_matrixIN = get_spike_matrix(Sp_I, NI,
                                           int(asarray(runtime) * 1000))
                      
                      # save stuff
                      dict2save = {}
                      dict2save['spikes_PYR'] = spike_matrixIN
                      dict2save['spikes_IN'] = spike_matrixPYR
                      dict2save['gE'] = gE.gea
                      dict2save['gI'] = gI.gi
                      dict2save['voltage_PYR'] = Vm_E.V
                      dict2save['voltage_IN'] = Vm_I.V
                      savemat(root_dir + str(itr) + ' GABA ' + str(igaba) + 
                            ' AMPA ' + str(iampa) + ' NMDA ' + str(inmda)
                            + '.mat', dict2save)    
                        
                      del Sp_E, Sp_I, Vm_E, Vm_I, gE, gI
                      


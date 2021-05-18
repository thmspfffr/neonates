
# -*- coding: utf-8 -*-
"""
modified version of LIF model
based on Tom's model, based on Wang's model
simplified, with more randomness and a different input 

v1 = 1st full attempt
v2 = changed ext input from 2.7 Hz to 5 Hz
     changed p of connection of ext input from 2 (?!) to 1
v3 = changed parameter set:
        - AMPA_mods from np.linspace(0.75,1.5,11) to np.linspace(0.75,3,21)
        - GABA_mods from np.linspace(0.5,1.2,11) to np.linspace(0.25,1.5,21)
     extended simulation time from 10 to 60 seconds
v4 = changed ext input to 1.5 Hz as in the eLife model
v5 = introduced different conductance levels for various AMPA connections (ext, and within network)
     started modulating also the AMPA conductance of the ext input
v6 = increase 5x both AMPA and GABA conductances
     increase gI/gE of the parameter space
v7 = made conductances values with units
"""

#%% import package and define functions

import numpy as np
from scipy.io import savemat
from brian2 import *

#%% define function(s)

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
    

#%% set variables for HH model and neural network

########### saving and plotting stuff ###########
root_dir = 'D:/models_neonates/results/'
v = 7 
to_plot = 0
to_save = 0
start_dataset = 0
num_datasets = 1

########### network parameters ###########
n_neurons = 100
PYRsProp = 0.8
nPYRS = int(n_neurons * PYRsProp)
nINs = int(n_neurons - nPYRS)
simulation_time = 10 * second
defaultclock.dt = 0.1 * ms
voltage_clock = Clock(dt=5*ms) # use different clock to change sampling rate
PYRs2keep = 80
INs2keep = 20

########### stable connectivity parameters ###########
gEE_AMPA = 0.178 * 5 * nS		                   # Weight of recurrent AMPA synapses between excitatory neurons
gEE_AMPA_ext = 0.234 * 5 * nS		               # Weight of external AMPA synapses between excitatory neurons
gEI_AMPA = 0.254 * 5 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
gIE_GABA = 2.01 * 5 * nS                          # Weight of inhibitory to excitatory synapses (GABA)
gII_GABA = 2.7 * 5 * nS                           # Weight of inhibitory to inhibitory synapses (GABA)
    
   
# Neuron model
CmE = 0.5 * nF                               # Membrane capacitance of excitatory neurons
CmI = 0.2 * nF                               # Membrane capacitance of inhibitory neurons
gLeakE = 25.0 * nS                           # Leak conductance of excitatory neurons
gLeakI = 20.0 * nS                           # Leak conductance of inhibitory neurons
Vl = -70.0 * mV                              # Leak membrane potential
Vthr = -52.0 * mV                            # Spiking threshold
Vrest = -59.0 * mV                           # Reset potential
refractoryE = 2 * ms                         # refractory period
refractoryI = 1 * ms                         # refractory period
TauE = 20 * ms
TauI = 10 * ms

# Synapse model
VrevE = 0 * mV                               # Reversal potential of excitatory synapses
VrevI = -80 * mV                             # Reversal potential of inhibitory synapses
tau_AMPA_E = 2.0 * ms                        # Decay constant of AMPA-type conductances
tau_AMPA_I = 1.0 * ms                        # Decay constant of AMPA-type conductances
tau_GABA = 8.0 * ms                          # Decay constant of GABA-type conductances

########### excitatory input parameters ###########
num_imputs = 100
epsilonPoisson = 1
input_factor = 1.5

# Neuron equations
eqsPYR = '''
dV/dt = (-gAMPA*(V-VrevE) - gGABA*(V-VrevI) - gLeak*(V-Vl)) / Cm + sigma/tau**.5*xi : volt
dggAMPA_E/dt = -gAMPA_E/(tau_AMPA_E) : siemens
dgGABA_E/dt = -gGABA_E/(tau_GABA) : siemens
gLeak : siemens
Cm : farad
sigma : volt
tau : second
'''
eqsIN = '''
dV/dt = (-gAMPA*(V-VrevE) - gGABA*(V-VrevI) - gLeak*(V-Vl)) / Cm + sigma/tau**.5*xi : volt
dgAMPA_I/dt = -gAMPA_I/(tau_AMPA_E) : siemens
dgGABA_I/dt = -gGABA_I/(tau_GABA) : siemens
gLeak : siemens
Cm : farad
sigma : volt
tau : second
'''

########### parameters to loop over ###########
AMPA_mods   = np.linspace(0.5,2.5,21)
GABA_mods   = np.linspace(0.5,2,21)

for iAMPA, AMPA_mod in enumerate(AMPA_mods):
    for iGABA, GABA_mod in enumerate(GABA_mods):
        print('working on AMPA ' + str(iAMPA) + ' GABA ' + str(iGABA))

        ########### define neuron groups ###########
        PYRs = NeuronGroup(nPYRS, method='euler',
                           model=eqsPYR,
                           threshold = "V > Vthr",  reset = "V = Vrest",
                           refractory=refractoryE)
        PYRs.Cm = CmE
        PYRs.gLeak = gLeakE
        PYRs.tau = CmE / gLeakE
        PYRs.sigma = 10 * mV
                            
        IN = NeuronGroup(nINs, method='euler',
                         model=eqsIN,
                         threshold = "V > Vthr",  reset = "V = Vrest",
                         refractory=refractoryI)
        IN.Cm = CmI
        IN.gLeak = gLeakI
        IN.tau = CmI / gLeakI
        IN.sigma = 10 * mV
                          
        # define AMPA and GABA synapses parameters
        Cee = Synapses(PYRs, PYRs, 'w: siemens', on_pre='gAMPA_E+=w')
        Cei = Synapses(PYRs, IN, 'w: siemens', on_pre='gAMPA_I+=w')
        Cie = Synapses(IN, PYRs, 'w: siemens', on_pre='gGABA_E+=w')
        Cii = Synapses(IN, IN, 'w: siemens', on_pre='gGABA_I+=w')

        Cee.connect(p=0.2); Cee.delay = 0 * ms
        Cee.w = gEE_AMPA * AMPA_mod 
        Cei.connect(p=0.2); Cei.delay = 0 * ms
        Cei.w = gEI_AMPA * AMPA_mod
        Cie.connect(p=0.2); Cie.delay = 0 * ms
        Cie.w = gIE_GABA * GABA_mod
        Cii.connect(p=0.2); Cii.delay = 0 * ms
        Cii.w = gII_GABA * GABA_mod

        # initialize voltage
        PYRs.V = Vrest + (rand(nPYRS) * 5 - 5) * mV
        IN.V = Vrest + (rand(nINs) * 5 - 5) * mV    

        # Generate a Poisson stimulus to inject into the network
        input_rate = input_factor * Hz
        extInput = PoissonGroup(num_imputs, rates=input_rate)
        # connect to PYRs
        SPYR = Synapses(extInput, PYRs, 'w: siemens', on_pre='gAMPA_E+=w')
        SPYR.connect(p=epsilonPoisson); SPYR.w = gEE_AMPA_ext * AMPA_mod; SPYR.delay = 0 * ms

        # record spikes of excitatory neurons
        Sp_E = SpikeMonitor(PYRs, record=True)
        # record spikes of inhibitory neurons
        Sp_I = SpikeMonitor(IN, record=True)
        # record voltage
        Vm_E = StateMonitor(PYRs, 'V', record=True, clock=voltage_clock)
        Vm_I = StateMonitor(IN, 'V', record=True, clock=voltage_clock)
        # record exc. & inh. currents at E
        gE = StateMonitor(PYRs, 'gAMPA_E', record=True, clock=voltage_clock)
        gI = StateMonitor(PYRs, 'gGABA_E', record=True, clock=voltage_clock)

        #------------------------------------------------------------------------------
        # Run the simulation
        #------------------------------------------------------------------------------
        print("Running simulation...")
        run(simulation_time) 

        spike_matrixPYR = get_spike_matrix(Sp_E, nPYRS,
                            int(asarray(simulation_time) * 1000))
        spike_matrixIN = get_spike_matrix(Sp_I, nINs,
                             int(asarray(simulation_time) * 1000))
          
        # save stuff
        dict2save = {}
        dict2save['spikes_PYR'] = spike_matrixPYR
        dict2save['spikes_IN'] = spike_matrixIN
        dict2save['gE'] = gE.gAMPA_E
        dict2save['gI'] = gI.gGABA_E
        dict2save['voltage_PYR'] = Vm_E.V
        dict2save['voltage_IN'] = Vm_I.V
        dict2save['g'] = (gIE_GABA * GABA_mod) / (gEE_AMPA * AMPA_mod)
        savemat(root_dir + 'eLife_v' + str(v) + '_GABA_' + str(iGABA) + 
              '_AMPA_' + str(iAMPA) + '.mat', dict2save)    
            
        del Sp_E, Sp_I, Vm_E, Vm_I, gE, gI
                    
                
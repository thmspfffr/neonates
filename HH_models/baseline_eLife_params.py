# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:05:37 2022

@author: mchini
"""

#%% import package and define functions

import numpy as np
from scipy.io import savemat
from brian2 import *

#%% define custom functions

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

########### set variables for g_e and g_i scaling ###########
ge_scaling = linspace(0.25, 2, 6)
gi_scaling = linspace(0.5, 4, 6)

########### saving and plotting stuff ###########
folder2save = 'E:/neonates/results_baseline_HH/cacca_'
to_save = 1
num_datasets = 5

########### network parameters ###########
n_neurons = 100
PYRsProp = 0.8
nPYRS = int(n_neurons * PYRsProp)
nINs = int(n_neurons - nPYRS)
num_reps = 3 # number of time to repeat simple stim paradigm
simulation_time = 60 * second
defaultclock.dt = 0.1 * ms
voltage_clock = Clock(dt=5*ms) # use different clock to change sampling rate

########### neuron parameters ###########
CmE = 0.5 * nF                               # Membrane capacitance of excitatory neurons
CmI = 0.2 * nF                               # Membrane capacitance of inhibitory neurons
gLeakE = 25.0 * nS                           # Leak conductance of excitatory neurons
gLeakI = 20.0 * nS                           # Leak conductance of inhibitory neurons
Vl = -70.0 * mV                              # Leak current reversal
gNa = 20.0 * uS                              # Na conductance
VNa = 50 * mV                                # Na current reversal
gKD = 16.0 * uS                              # K conductance
VK = -100 * mV                               # K current reversal
Vthr = -52.0 * mV                            # Spiking threshold
Vrest = -59.0 * mV                           # Reset potential
refractoryE = 2 * ms                         # refractory period
refractoryI = 1 * ms                         # refractory period

########### stable connectivity parameters ###########
gEE_AMPA = 0.178 * 100 * nS		                 # Weight of recurrent AMPA synapses between excitatory neurons
gEE_AMPA_ext = 0.234 * 100 * nS		             # Weight of external AMPA synapses between excitatory neurons
gEI_AMPA = 0.254 * 100 * nS                        # Weight of excitatory to inhibitory synapses (AMPA)
gIE_GABA = 2.01 * 100 * nS                         # Weight of inhibitory to excitatory synapses (GABA)
gII_GABA = 2.7 * 100 * nS                          # Weight of inhibitory to inhibitory synapses (GABA)



########### synapses parameters ###########
VrevE = 0 * mV                               # Reversal potential of excitatory synapses
VrevI = -80 * mV                             # Reversal potential of inhibitory synapses
tau_AMPA_E = 2.0 * ms                        # Decay constant of AMPA-type conductances
tau_AMPA_I = 1.0 * ms                        # Decay constant of AMPA-type conductances
tau_GABA = 5.0 * ms                          # Decay constant of GABA-type conductances
gEE = 0.178 * nS
gEI = 0.254 * nS 
gIE = 2.01 * nS
gII = 2.7 * nS
epsylon = 0.2

########### excitatory input parameters ###########
num_imputs = 100
epsilonPoisson = 1
input_factor = 1.5

# define HH model with specific parameters
eqsPYR = Equations('''
dv/dt = (gLeakE*(Vl-v) + gea*(VrevE-v) + gi*(VrevI-v) -
         gNa*(m*m*m)*h*(v-VNa) - gKD*(n*n*n*n)*(v-VK)
         ) / CmE : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dgea/dt = -gea*(1./tau_AMPA_E) : siemens
dgi/dt = -gi*(1./tau_AMPA_I) : siemens
alpha_m = (0.32 * (54 + v/mV)) / (1 - exp(-0.25 * (v/mV + 54))) / ms : Hz
beta_m = (0.28 * (27 + v/mV)) / (exp(0.2 * (v/mV + 27)) - 1) / ms : Hz
alpha_h = 0.07 * exp(-0.05/mV * (v + 58 * mV)) / ms : Hz
beta_h = 1 / (1 + exp(-0.1/mV * (v + 28 * mV))) / ms : Hz
alpha_n = (0.032 * (52 + v/mV)) / (1 - exp(-0.2 * (v/mV + 52))) / ms : Hz
beta_n = 0.5 * exp(-0.025 * (57 + v/mV)) / ms : Hz
sigma : volt
''')

eqsIN = Equations('''
dv/dt = (gLeakE*(Vl-v) + gea*(VrevE-v) + gi*(VrevI-v) -
         gNa*(m*m*m)*h*(v-VNa) - gKD*(n*n*n*n)*(v-VK)
         ) / CmI : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dgea/dt = -gea*(1./tau_AMPA_I) : siemens
dgi/dt = -gi*(1./tau_GABA) : siemens
alpha_m = (0.32 * (54 + v/mV)) / (1 - exp(-0.25 * (v/mV + 54))) / ms : Hz
beta_m = (0.28 * (27 + v/mV)) / (exp(0.2 * (v/mV + 27)) - 1) / ms : Hz
alpha_h = 0.07 * exp(-0.05/mV * (v + 58 * mV)) / ms : Hz
beta_h = 1 / (1 + exp(-0.1/mV * (v + 28 * mV))) / ms : Hz
alpha_n = (0.032 * (52 + v/mV)) / (1 - exp(-0.2 * (v/mV + 52))) / ms : Hz
beta_n = 0.5 * exp(-0.025 * (57 + v/mV)) / ms : Hz
sigma : volt
''')


#%% simulate network

AMPA_mods   = np.linspace(0.5,2,21)
GABA_mods   = np.linspace(0.5,2,21)


for iAMPA, AMPA_mod in enumerate(AMPA_mods):
    for iGABA, GABA_mod in enumerate(GABA_mods):
        
        print('working on AMPA ' + str(iAMPA) + ' GABA ' + str(iGABA))
        
        # define parameters of neural network
        PYRs = NeuronGroup(nPYRS, model=eqsPYR,
                           method='exponential_euler',
                           threshold = "v>Vthr",
                           reset = "v=Vrest",
                           refractory=refractoryE)
#        PYRs.sigma = 10 * mV
        
        IN = NeuronGroup(nINs, model=eqsIN,
                         method='exponential_euler',
                         threshold = "v>Vthr",
                         reset = "v=Vrest",
                         refractory=refractoryI)
#        IN.sigma = 10 * mV
        
        # define AMPA and GABA synapses parameters
        Cee = Synapses(PYRs, PYRs, 'w:siemens', on_pre='gea+=w')
        Cei = Synapses(PYRs, IN, 'w:siemens', on_pre='gea+=w')
        Cie = Synapses(IN, PYRs, 'w:siemens', on_pre='gi+=w')
        Cii = Synapses(IN, IN, 'w:siemens', on_pre='gi+=w')
        # now connect synapses
        Cee.connect(p=0.2); Cee.delay = 0 * ms
        Cee.w = gEE_AMPA * AMPA_mod / gLeakE * nS
        Cei.connect(p=0.2); Cei.delay = 0 * ms
        Cei.w = gEI_AMPA * AMPA_mod / gLeakI * nS
        Cie.connect(p=0.2); Cie.delay = 0 * ms
        Cie.w = gIE_GABA * GABA_mod / gLeakE * nS
        Cii.connect(p=0.2); Cii.delay = 0 * ms
        Cii.w = gII_GABA * GABA_mod / gLeakI * nS
    
        # initialize voltage and synaptic conductance
        PYRs.v = Vrest + (rand(nPYRS) * 5 - 5) * mV
        IN.v = Vrest + (rand(nINs) * 5 - 5) * mV 
            
        # Generate a Poisson stimulus to inject into the network of PYRs
        # set input rate as something that is log-normal
        input_rate = input_factor * Hz
        extInput = PoissonGroup(num_imputs, rates=input_rate)
        
        SPYR = Synapses(extInput, PYRs, 'w:siemens', on_pre='gea+=w')
        SPYR.connect(p=epsilonPoisson); SPYR.delay = 0 * ms
        SPYR.w = gEE_AMPA_ext * AMPA_mod / gLeakE * nS
            
        ###### define parameters to track ######
        # record spikes of excitatory and inhibitory neurons
        Sp_E = SpikeMonitor(PYRs, record=True)
        Sp_I = SpikeMonitor(IN, record=True)
        # record voltage
        Vm_E = StateMonitor(PYRs, 'v', record=True, clock=voltage_clock)
        Vm_I = StateMonitor(IN, 'v', record=True, clock=voltage_clock)
        # record exc. & inh. currents at E
        gE = StateMonitor(PYRs, 'gea', record=True, clock=voltage_clock)
        gI = StateMonitor(PYRs, 'gi', record=True, clock=voltage_clock)
        
        # running the simulation
        run(simulation_time)
        
        # extract average number of spikes
        num_spikesPYRs = len(Sp_E.t) / (nPYRS * simulation_time)
        num_spikesIN = len(Sp_I.t) / (nINs * simulation_time)
        
        # extract spike matrices
        spike_matrixPYR = get_spike_matrix(Sp_E, nPYRS,
                           int(asarray(simulation_time) * 1000))
        spike_matrixIN = get_spike_matrix(Sp_I, nINs,
                          int(asarray(simulation_time) * 1000))
        
        print('mean firing rate PYRs ' + str(num_spikesPYRs) + ' Hz')    
        print('mean firing rate INs ' + str(num_spikesIN) + ' Hz')
        
        if to_save == 1:
            # save stuff
            dict2save = {}
            dict2save['spikes_IN'] = spike_matrixIN
            dict2save['spikes_PYR'] = spike_matrixPYR
            dict2save['voltage_PYR'] = Vm_E.v
            dict2save['voltage_IN'] = Vm_I.v
            dict2save['gE'] = gE.gea
            dict2save['gI'] = gI.gi
            dict2save['g'] = (gIE_GABA * GABA_mod + gII_GABA * GABA_mod) / (gEE_AMPA * AMPA_mod + gEI_AMPA * AMPA_mod)
            savemat(folder2save + 'ge' + str(iAMPA) + '_gi' + str(iGABA) + '.mat', dict2save)
        
        del spike_matrixPYR, spike_matrixIN # state_monitorIN, state_monitorPYR, 
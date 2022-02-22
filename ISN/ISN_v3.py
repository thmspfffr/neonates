# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:04:32 2022

@author: mchini

modified version of LIF model
based on Tom's model, based on Wang's model
simplified, with more randomness and a different input 
v0 = search for params that lead to runaway excitation without INs (log-normal AMPA mods of the same values does the trick)
v1 = also add the interneurons to the network
v2 = try to inhibit INs and check what happens
v3 = introduce II
"""

# random change
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
root_dir = 'E:/neonates/ISN_baseline/'
v = 3

########### network parameters ###########
n_neurons = 400
PYRsProp = 0.8
nPYRS = int(n_neurons * PYRsProp)
nINs = int(n_neurons - nPYRS)
defaultclock.dt = 0.1 * ms
voltage_clock = Clock(dt=5*ms) # use different clock to change sampling rate
simulation_time = 10*second
PYRs2keep = 320
INs2keep = 80

########### stable connectivity parameters ###########
gEE_AMPA = 0.178 * nS		                   # Weight of recurrent AMPA synapses between excitatory neurons
gEE_AMPA_ext = 0.234 * nS		               # Weight of external AMPA synapses between excitatory neurons
gEI_AMPA = 0.254 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
gIE_GABA = 2.01 * nS                          # Weight of inhibitory to excitatory synapses (GABA)
# [ORIGINALLY THIS WAS 2.7!]
gII_GABA = 2.7 * nS                           # Weight of inhibitory to inhibitory synapses (GABA)
    
   
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
tau_GABA = 5.0 * ms                          # Decay constant of GABA-type conductances

########### excitatory input parameters ###########
num_imputs = 100
epsilonPoisson = 1
input_factor = 5

# Neuron equations
eqsPYR = '''
dV/dt = (-gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + sigma/tau**.5*xi : volt
dgea/dt = -gea/(tau_AMPA_E) : 1
dgi/dt = -gi/(tau_GABA) : 1
tau : second
Cm : farad
sigma : volt
'''
eqsIN = '''
dV/dt = (-gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + sigma/tau**.5*xi: volt
dgea/dt = -gea/(tau_AMPA_I) : 1
dgi/dt = -gi/(tau_GABA) : 1
tau : second
Cm : farad
sigma : volt
'''

########### parameters to loop over ###########
AMPA_mods   = np.linspace(0.5,1,9)
GABA_mods   = np.linspace(0.5,2,9)
connectivity = 1

#%% now actually run the model

for iAMPA, AMPA_mod in enumerate(AMPA_mods):
    for iGABA, GABA_mod in enumerate(GABA_mods):
        ########### define neuron groups ###########
        PYRs = NeuronGroup(nPYRS, method='euler',
                        model=eqsPYR,
                        threshold = "V>Vthr",  reset = "V=Vrest",
                        refractory=refractoryE)
        PYRs.Cm = CmE
        PYRs.tau = CmE / gLeakE
        PYRs.sigma = 10 * mV
                            
        IN = NeuronGroup(nINs, method='euler',
                        model=eqsIN,
                        threshold = "V>Vthr",  reset = "V=Vrest",
                        refractory=refractoryI)
        IN.Cm = CmI
        IN.tau = CmI / gLeakI      
        IN.sigma = 10 * mV
                        
        # define AMPA and GABA synapses parameters
        Cee = Synapses(PYRs, PYRs, 'w: 1', on_pre='gea+=w')
        Cei = Synapses(PYRs, IN, 'w: 1', on_pre='gea+=w')
        Cie = Synapses(IN, PYRs, 'w: 1', on_pre='gi+=w')
        Cii = Synapses(IN, IN, 'w: 1', on_pre='gi+=w')
        
        # compute average of actual conductances
        gEE = lognormal(gEE_AMPA * AMPA_mod / gLeakE, gEE_AMPA * AMPA_mod / gLeakE, nPYRS**2) * AMPA_mod
        gEI = lognormal(gEI_AMPA * AMPA_mod / gLeakI, gEI_AMPA * AMPA_mod / gLeakI, nPYRS*nINs) * AMPA_mod
        gIE = lognormal(gIE_GABA * GABA_mod / gLeakE, gIE_GABA * GABA_mod / gLeakE, nPYRS*nINs) * GABA_mod
        gII = lognormal(gII_GABA * GABA_mod / gLeakI, gII_GABA * GABA_mod / gLeakI, nINs**2) / 5 * GABA_mod

        Cee.connect(p=connectivity); Cee.delay = 0 * ms
        Cee.w = gEE
        Cei.connect(p=connectivity); Cei.delay = 0 * ms
        Cei.w = gEI
        Cie.connect(p=connectivity); Cie.delay = 0 * ms
        Cie.w = gIE
        Cii.connect(p=connectivity); Cii.delay = 0 * ms
        Cii.w = gII

        # initialize voltage
        PYRs.V = Vrest + (rand(nPYRS) * 5 - 5) * mV
        IN.V = Vrest + (rand(nINs) * 5 - 5) * mV    

        # Generate a Poisson stimulus to inject into the network
        # number of neurons that provide excitation
        # set input rate as something that is log-normal
        #input_rate = exp(np.random.normal(input_factor, 0.5, num_imputs)) * Hz
        input_rate = input_factor * Hz
        extInput = PoissonGroup(num_imputs, rates=input_rate)
        # connect to PYRs
        SPYR = Synapses(extInput, PYRs, 'w: 1', on_pre='gea+=w')
        SPYR.connect(p=epsilonPoisson); SPYR.w = gEE_AMPA_ext * AMPA_mod / gLeakE; SPYR.delay = 0 * ms
        SIN = Synapses(extInput, IN, 'w: 1', on_pre='gea+=w')
        SIN.connect(p=epsilonPoisson); SIN.w = gEE_AMPA_ext * AMPA_mod / gLeakI; SPYR.delay = 0 * ms

        # record spikes of excitatory neurons
        Sp_E = SpikeMonitor(PYRs, record=True)
        # record spikes of inhibitory neurons
        Sp_I = SpikeMonitor(IN, record=True)
        # record voltage
        Vm_E = StateMonitor(PYRs, 'V', record=True, clock=voltage_clock)
        Vm_I = StateMonitor(IN, 'V', record=True, clock=voltage_clock)
        # record exc. & inh. currents at E
        gE = StateMonitor(PYRs, 'gea', record=True, clock=voltage_clock)
        gI = StateMonitor(PYRs, 'gi', record=True, clock=voltage_clock)

        #------------------------------------------------------------------------------
        # Run the simulation
        #------------------------------------------------------------------------------
        run(simulation_time) 

        spike_matrixPYR = get_spike_matrix(Sp_E, nPYRS,
                            int(asarray(simulation_time) * 1000))
        spike_matrixIN = get_spike_matrix(Sp_I, nINs,
                            int(asarray(simulation_time) * 1000))
        
                
        dict2save = {}
        dict2save['spikes_PYR'] = spike_matrixPYR
        dict2save['spikes_IN'] = spike_matrixIN
        dict2save['gE'] = gE.gea
        dict2save['gI'] = gI.gi
        dict2save['connectivity'] = connectivity
        dict2save['voltage_PYR'] = Vm_E.V
        dict2save['voltage_IN'] = Vm_I.V
        dict2save['g'] = (mean(gIE) + mean(gII)) / (mean(gEE) + mean(gEI))  
        dict2save['g_med'] = (median(gIE) + median(gII)) / (median(gEE) + median(gEI))  
        savemat(root_dir + str(v) + '_GABA_' + str(iGABA) + '_AMPA_' + str(iAMPA) + '.mat', dict2save)
        
        del Sp_E, Sp_I#, gE, gI, Vm_E, Vm_I, 
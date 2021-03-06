
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

"""

#%% import package and define functions

import numpy as np
from scipy.io import savemat
from brian2 import *
import os.path
from subprocess import call
import sttc 
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

v = 1

########### saving and plotting stuff ###########
# if not(os.path.exists('/home/tpfeffer/neonates/proc/v%d' %v)):
        # os.makedirs('/home/tpfeffer/neonates/proc/v%d' %v)

root_dir = '/home/tpfeffer/neonates/proc/v%d/' %v
to_plot = 0
to_save = 0
start_dataset = 0
num_datasets = 1

########### network parameters ###########
n_neurons = 400
PYRsProp = 0.8
nPYRS = int(n_neurons * PYRsProp)
nINs = int(n_neurons - nPYRS)
simulation_time = 60 * second
defaultclock.dt = 0.1 * ms
voltage_clock = Clock(dt=5*ms) # use different clock to change sampling rate
PYRs2keep = 320
INs2keep = 80

########### stable connectivity parameters ###########
gEE_AMPA = 0.178 * nS		                  # Weight of AMPA synapses between excitatory neurons
gEI_AMPA = 0.233 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
gIE_GABA = 2.01 * nS                          # Weight of inhibitory to excitatory synapses (GABA)
gII_GABA = 2.7 * nS                           # Weight of inhibitory to inhibitory synapses (GABA)
    

# Connectivity - external connections
gextE = 0.234 * nS                            # Weight of external input to excitatory neurons: Increased from previous value
gextI = 0.317 * nS                            # Weight of external input to inhibitory neurons
    
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
tau_AMPA_E_d = 2.0 * ms                        # Decay constant of AMPA-type conductances
tau_AMPA_E_r = 0.4 * ms                        # Decay constant of AMPA-type conductances
tau_AMPA_I_d = 1.0 * ms                        # Decay constant of AMPA-type conductances
tau_AMPA_I_r = 0.2 * ms                        # Decay constant of AMPA-type conductances

tau_GABA_E_d = 5.0 * ms                        # Decay constant of GABA-type conductances
tau_GABA_E_r = 0.25 * ms                        # Rise constant of GABA-type conductances
tau_GABA_I_d = 5.0 * ms                        # Decay constant of GABA-type conductances
tau_GABA_I_r = 0.25 * ms                        # Rise constant of GABA-type conductances

alpha_ampa = 1.0 *  ms**-1
alpha_gaba = 1.0 *  ms**-1

########### excitatory input parameters ###########
num_imputs = 100
epsilonPoisson = 1
input_factor = 5

# Neuron equations
eqsPYR = '''
dV/dt = (-gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + sigma/tau**.5*xi : volt

dgea/dt = -gea/(tau_AMPA_E_d) + geax*alpha_ampa*(1 - gea): 1
dgeax/dt = -geax/(tau_AMPA_E_r) : 1

dgi/dt = -gi/(tau_GABA_E_d) + gix*alpha_gaba*(1 - gi): 1
dgix/dt = -gix/(tau_GABA_E_r) : 1
tau : second
Cm : farad
sigma : volt
'''
eqsIN = '''
dV/dt = (-gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + sigma/tau**.5*xi: volt

dgea/dt = -gea/(tau_AMPA_I_d) + geax*alpha_ampa*(1 - gea): 1
dgeax/dt = -geax/(tau_AMPA_I_r) : 1

dgi/dt = -gi/(tau_GABA_I_d) + gix*alpha_gaba*(1 - gi): 1
dgix/dt = -gix/(tau_GABA_I_r) :1

tau : second
Cm : farad
sigma : volt
'''

########### parameters to loop over ###########
AMPA_mods   = np.linspace(0.75,3,21)
GABA_mods   = np.linspace(0.25,1.5,21)

for iAMPA, AMPA_mod in enumerate(AMPA_mods):
    for iGABA, GABA_mod in enumerate(GABA_mods):

        fn = os.path.expanduser(root_dir + '/elife_model_iampa%d_gaba%d_v%d_processing.txt') % (iAMPA, iGABA,v)
        if os.path.isfile(fn)==False:
            call(['touch', fn])
        else:
            continue
        print('working on AMPA ' + str(iAMPA) + ' GABA ' + str(iGABA))

        ########### define neuron groups ###########
        PYRs = NeuronGroup(nPYRS,
                           model=eqsPYR,
                           threshold = "V > Vthr",  reset = "V = Vrest",
                           refractory=refractoryE)
        PYRs.Cm = CmE
        PYRs.tau = CmE / gLeakE
        PYRs.sigma = 10 * mV
                            
        IN = NeuronGroup(nINs,
                         model=eqsIN,
                         threshold = "V > Vthr",  reset = "V = Vrest",
                         refractory=refractoryI)
        IN.Cm = CmI
        IN.tau = CmI / gLeakI      
        IN.sigma = 10 * mV
                          
        # define AMPA and GABA synapses parameters
        Cee = Synapses(PYRs, PYRs, 'w: 1', on_pre='geax+=w')
        Cei = Synapses(PYRs, IN, 'w: 1', on_pre='geax+=w')
        Cie = Synapses(IN, PYRs, 'w: 1', on_pre='gix+=w')
        Cii = Synapses(IN, IN, 'w: 1', on_pre='gix+=w')

        Cee.connect(p=0.2); Cee.delay = 0 * ms
        Cee.w = gEE_AMPA * AMPA_mod / gLeakE 
        Cei.connect(p=0.2); Cei.delay = 0 * ms
        Cei.w = gEI_AMPA * AMPA_mod / gLeakI
        Cie.connect(p=0.2); Cie.delay = 0 * ms
        Cie.w = gIE_GABA * GABA_mod / gLeakE
        Cii.connect(p=0.2); Cii.delay = 0 * ms
        Cii.w = gII_GABA * GABA_mod / gLeakI

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
        SPYR = Synapses(extInput, PYRs, 'w: 1', on_pre='geax+=w')
        SPYR.connect(p=epsilonPoisson); SPYR.w = gEE_AMPA / gLeakE; SPYR.delay = 0 * ms
        # connect to INs
        #SIN = Synapses(extInput, IN, 'w: 1', on_pre='gea+=w*(rand() + 1)')
        #SIN.connect(p=epsilonPoisson); SIN.w = gEI_AMPA / gLeakI; SPYR.delay = 0 * ms

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
        print("Running simulation...")
        #net = Network(Sp_E, Sp_I, Vm_E, Vm_I, gE, gI, PYRs, IN, extInput)
        run(simulation_time) 

        spike_matrixPYR = get_spike_matrix(Sp_E, nPYRS,
                            int(asarray(simulation_time) * 1000))
        spike_matrixIN = get_spike_matrix(Sp_I, nINs,
                                int(asarray(simulation_time) * 1000))

        sttc_tmp=sttc.matrix_spike_time_tiling_coefficient(spike_matrixPYR)
        sttc_tmp=sttc_tmp.mean()

        LFP = abs(gI.gi * (Vm_E.V- VrevI)) + abs(gE.gea * (Vm_E.V - VrevE))
        # save stuff
        dict2save = {}
        #dict2save['spikes_PYR'] = spike_matrixPYR
        #dict2save['spikes_IN'] = spike_matrixIN
        # dict2save['gE'] = gE.gea
        # dict2save['gI'] = gI.gi
        # dict2save['voltage_PYR'] = Vm_E.V
        # dict2save['voltage_IN'] = Vm_I.V
        dict2save['sttc'] = sttc_tmp
        dict2save['LFP'] = LFP
        dict2save['g'] = (gIE_GABA * GABA_mod) / (gEE_AMPA * AMPA_mod)
        savemat(root_dir + 'elife_model_v' + str(v) + '_GABA' + str(iGABA) + 
              '_AMPA' + str(iAMPA) + '.mat', dict2save)    
            
        del Sp_E, Sp_I, Vm_E, Vm_I, gE, gI
                    
                
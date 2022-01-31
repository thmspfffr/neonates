# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:21:53 2022

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
folder2save = 'E:/neonates/results_baseline_HH/'
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
area = 20000 * umetre ** 2 # area of neuron 
Cm = (1 * ufarad * cm ** -2) * area # membrane capacitance
gl = (0.1 * msiemens * cm ** -2) * area # leak conductance
g_na = (100 * msiemens * cm ** -2) * area # Na conductance
g_kd = (80 * msiemens * cm ** -2) * area # K conductance
El = -57 * mV # Leak current reversal
EK = -100 * mV # K current reversal
ENa = 50 * mV # Na current reversal
Vrest = -65 * mV # resting membrane potential
APthr = -30 * mV # action potential threshold
refractory = 3 * ms # refractory period

########### synapses parameters ###########

Ee = 0 * mV # excitatory synapses reversal
Ei = -70 *mV # inhibitory synapses reversal
we = 6 * nS # excitatory synaptic weight
wi = we * 8 # 50 * nS  # inhibitory synaptic weight
taue = 5 * ms # excitatory synapses decay constant
taui = 10 * ms # inhibitory synapses decay constant
g_e = (randn() * 1.5 + 4) * 10. * nS
g_i = g_e * 4
epsylonIN = 0.4
epsylonEE = 0.1

########### excitatory input parameters ###########
num_imputs = 1000
epsilonPoisson = 0.1
input_weight = 1

########### ramp parameters ###########
stim_start = int(3 * 1000 / 0.1)
stim_end = int(6 * 1000 / 0.1)
t_end = int(9 * 1000 / 0.1)
unit_time = 0.1 * ms
amplitude_start = 0 * pamp
amplitude_end = 0.5 * namp # -2.5 for inhibition

# define HH model with specific parameters
eqsPYR = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK)
         ) / Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = (0.32 * (54 + v/mV)) / (1 - exp(-0.25 * (v/mV + 54))) / ms : Hz
beta_m = (0.28 * (27 + v/mV)) / (exp(0.2 * (v/mV + 27)) - 1) / ms : Hz
alpha_h = 0.07 * exp(-0.05/mV * (v + 58 * mV)) / ms : Hz
beta_h = 1 / (1 + exp(-0.1/mV * (v + 28 * mV))) / ms : Hz
alpha_n = (0.032 * (52 + v/mV)) / (1 - exp(-0.2 * (v/mV + 52))) / ms : Hz
beta_n = 0.5 * exp(-0.025 * (57 + v/mV)) / ms : Hz
''')

eqsIN = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK)
         ) / Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = (0.32 * (54 + v/mV)) / (1 - exp(-0.25 * (v/mV + 54))) / ms : Hz
beta_m = (0.28 * (27 + v/mV)) / (exp(0.2 * (v/mV + 27)) - 1) / ms : Hz
alpha_h = 0.07 * exp(-0.05/mV * (v + 58 * mV)) / ms : Hz
beta_h = 1 / (1 + exp(-0.1/mV * (v + 28 * mV))) / ms : Hz
alpha_n = (0.032 * (52 + v/mV)) / (1 - exp(-0.2 * (v/mV + 52))) / ms : Hz
beta_n = 0.5 * exp(-0.025 * (57 + v/mV)) / ms : Hz
''')


#%% simulate network

for ge_idx, ge_scal in enumerate(ge_scaling):
    for gi_idx, gi_scal in enumerate(gi_scaling):
        # define parameters of neural network
        PYRs = NeuronGroup(nPYRS,
                           model=eqsPYR, method='exponential_euler',
                           threshold = "v>APthr",  reset = "v=Vrest",
                           refractory=refractory)
        IN = NeuronGroup(nINs,
                         model=eqsIN, method='exponential_euler',
                         threshold = "v>APthr",  reset = "v=Vrest",
                         refractory=refractory)
        
        # define synapses parameters
        Cee = Synapses(PYRs, PYRs, on_pre='ge+=we')
        Cei = Synapses(PYRs, IN, on_pre='ge+=we')
        Cie = Synapses(IN, PYRs, on_pre='gi+=wi')
        Cee.connect(condition='i!=j', p=epsylonEE)
        Cee.delay = 2 * ms
        Cei.connect(p=epsylonEE)
        Cei.delay = 2 * ms
        Cie.connect(p=epsylonIN)
        Cie.delay = 2 * ms
    
        # initialize voltage and synaptic conductance
        PYRs.v = Vrest + (rand(nPYRS) * 5 - 5) * mV
        PYRs.ge = g_e * ge_scal
        PYRs.gi = g_i * gi_scal
        IN.v = Vrest + (rand(nINs) * 5 - 5) * mV
        IN.ge = g_e * ge_scal
        IN.gi = g_i * gi_scal
            
        # Generate a Poisson stimulus to inject into the network of PYRs
        # set input rate as something that is log-normal
        input_rate = exp(np.random.normal(0, 0.5, num_imputs)) * Hz
        P = PoissonGroup(num_imputs, rates=input_rate)
        
        SPYR = Synapses(P, PYRs, on_pre='v += input_weight * 5 * mV')
        SPYR.connect(p = epsilonPoisson)
        SPYR.delay = 2 * ms
            
        ###### define parameters to track ######
        # record spikes
        spike_monitorPYR = SpikeMonitor(PYRs)
        spike_monitorIN = SpikeMonitor(IN)        
        # record voltage
        Vm_E = StateMonitor(PYRs, 'v', record=True, clock=voltage_clock)
        Vm_I = StateMonitor(IN, 'v', record=True, clock=voltage_clock)
        # record exc. & inh. currents at E
        gE = StateMonitor(PYRs, 'ge', record=True, clock=voltage_clock)
        gI = StateMonitor(PYRs, 'gi', record=True, clock=voltage_clock)
        
        # running the simulation
        run(simulation_time)
        
        # extract average number of spikes
        num_spikesPYRs = len(spike_monitorPYR.t) / (nPYRS * simulation_time)
        num_spikesIN = len(spike_monitorIN.t) / (nINs * simulation_time)
        
        # extract spike matrices
        spike_matrixPYR = get_spike_matrix(spike_monitorPYR, nPYRS,
                                           int(asarray(simulation_time) * 1000))
        spike_matrixIN = get_spike_matrix(spike_monitorIN, nINs,
                                           int(asarray(simulation_time) * 1000))
        
        print('mean firing rate PYRs ' + str(num_spikesPYRs) + ' Hz')    
        print('mean firing rate INs ' + str(num_spikesIN) + ' Hz')
        
        if to_save == 1:
            # save stuff
            dict2save = {}
            dict2save['IN_spike_matrix'] = spike_matrixIN
            dict2save['PYR_spike_matrix'] = spike_matrixPYR
            dict2save['voltage_PYR'] = Vm_E.v
            dict2save['voltage_IN'] = Vm_I.v
            dict2save['gE'] = gE.ge
            dict2save['gI'] = gI.gi
            dict2save['g'] = (g_i * gi_scal) / (g_e * ge_scal)
            savemat(folder2save + 'ge' + str(ge_idx) + '_gi' + str(gi_idx) + '.mat', dict2save)
        
        del spike_monitorIN, spike_monitorPYR # state_monitorIN, state_monitorPYR, 
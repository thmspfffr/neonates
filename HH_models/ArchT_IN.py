# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:04:33 2022

@author: mchini
"""


#%% import package and define functions

import numpy as np
from scipy.io import savemat
from brian2 import *

#%% define custom functions

def get_ramp_current(stim_start, stim_end, t_end, unit_time, amplitude_start, amplitude_end, num_reps):
    """Creates a ramp current. If stim_start == stim_end, then ALL entries are 0.
    Args:
        stim_start (int): start of the ramp
        stim_end (int): end of the ramp
        t_end(int): end of the stimulus (can be after end of ramp)
        unit_time (Brian2 unit): unit of stim_start and stim_end. e.g. 0.1*brian2.ms
        amplitude_start (Quantity): amplitude of the ramp at stim_start. e.g. 3.5*brian2.uamp
        amplitude_end (Quantity): amplitude of the ramp at stim_end. e.g. 4.5*brian2.uamp
        num_reps(int): number of repetitions, in case you want to repeat the same stimulus
                       multiple times       
    Returns:
        TimedArray: Brian2.TimedArray
    """
    assert isinstance(stim_start, int), "stim_start_ms must be of type int"
    assert isinstance(stim_end, int), "stim_end must be of type int"
    assert units.fundamentalunits.have_same_dimensions(amplitude_start, amp), \
        "amplitude must have the dimension of current e.g. brian2.uamp"
    assert units.fundamentalunits.have_same_dimensions(amplitude_end, amp), \
        "amplitude must have the dimension of current e.g. brian2.uamp"
    tmp_size = 1 + t_end  # +1 for t=0
    tmp = np.zeros(tmp_size) * amp
    if stim_end > stim_start:  # if deltaT is zero, we return a zero current
        slope = (amplitude_end - amplitude_start) / float((stim_end - stim_start))
        ramp = [amplitude_start + t * slope for t in range(0, (stim_end - stim_start) + 1)]
        tmp[stim_start: stim_end + 1] = ramp
    tmp = tile(tmp, num_reps)
    curr = TimedArray(tmp, dt=1. * unit_time)
    return curr

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

########### set variables for GABA abd Cl ###########
GABA_scaling = r_[-20, -10, 0]
V_scaling = r_[-20, -10, 0]

########### saving and plotting stuff ###########
folder4matrices = 'E:/neonates/results_opto_HH/'
to_plot = 1
to_save = 0
num_datasets = 1

########### network parameters ###########
n_neurons = 100
PYRsProp = 0.8
nPYRS = int(n_neurons * PYRsProp)
nINs = int(n_neurons - nPYRS)
num_reps = 3 # number of time to repeat simple stim paradigm
simulation_time = 9 * second * num_reps
defaultclock.dt = 0.1 * ms

########### neuron parameters ###########
area = 20000 * umetre ** 2 # area of neuron 
Cm = (1 * ufarad * cm ** -2) * area # membrane capacitance
gl = (0.1 * msiemens * cm ** -2) * area # leak conductance
g_na = (100 * msiemens * cm ** -2) * area # Na conductance
g_kd = (80 * msiemens * cm ** -2) * area # K conductance
Els = (-67 - V_scaling) * mV # Leak current reversal
EK = -100 * mV # K current reversal
ENa = 50 * mV # Na current reversal
Vrests = (-75 - V_scaling) * mV # resting membrane potential
APthr = -30 * mV # action potential threshold
refractory = 3 * ms # refractory period

########### synapses parameters ###########

Ee = 0 * mV # excitatory synapses reversal
Eis = (-80  - GABA_scaling) *mV # inhibitory synapses reversal
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
amplitude_end = - 0.5 * namp # -2.5 for inhibition

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
#Istim = empty_ramp(t) : amp
''')

eqsIN = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK)
         + Istim
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
Istim = ramp(t) : amp
''')


#%% simulate network

num_spikesPYRs = zeros_like(GABA_scaling)
num_spikesIN = zeros_like(GABA_scaling)

for idx, (El, Vrest, Ei) in enumerate(zip(Els, Vrests, Eis)):

    # define ramp stimulation
    ramp = get_ramp_current(stim_start, stim_end, t_end, unit_time, amplitude_start,
                        amplitude_end, num_reps)
    
    for db_idx in range(num_datasets):
            
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
        PYRs.ge = g_e
        PYRs.gi = g_i
        IN.v = Vrest + (rand(nINs) * 5 - 5) * mV
        IN.ge = g_e
        IN.gi = g_i
            
        # Generate a Poisson stimulus to inject into the network of PYRs
        # set input rate as something that is log-normal
        input_rate = exp(np.random.normal(0, 0.5, num_imputs)) * Hz
        P = PoissonGroup(num_imputs, rates=input_rate)
        
        SPYR = Synapses(P, PYRs, on_pre='v += input_weight * 5 * mV')
        SPYR.connect(p = epsilonPoisson)
        SPYR.delay = 2 * ms
            
        # define parameters to track
        spike_monitorPYR = SpikeMonitor(PYRs)
        spike_monitorIN = SpikeMonitor(IN)
        if to_plot > 0:
            populationPYR = PopulationRateMonitor(PYRs)            
            populationIN = PopulationRateMonitor(IN)
        if to_plot > 1:
            state_monitorPYR = StateMonitor(PYRs, 'v', record=True)
            state_monitorIN = StateMonitor(IN, 'v', record=True)
        
        # running the simulation
        run(simulation_time)
        
        # extract average number of spikes
        num_spikesPYRs[idx] = len(spike_monitorPYR.t) / (nPYRS * simulation_time)
        num_spikesIN[idx] = len(spike_monitorIN.t) / (nINs * simulation_time)
        
        # extract spike matrices
        spike_matrixPYR = get_spike_matrix(spike_monitorPYR, nPYRS,
                                           int(asarray(simulation_time) * 1000))
        spike_matrixIN = get_spike_matrix(spike_monitorIN, nINs,
                                           int(asarray(simulation_time) * 1000))
        
        # plot stuff
        if to_plot > 0:
            # plot spikes for PYRs
            figure() 
            subplot(1,2,1)
            plot(spike_monitorPYR.t, spike_monitorPYR.i, 'k*')
            subplot(1,2,2)
            plot(convolve(populationPYR.rate, ones(1000), 'same') / 1000)
            suptitle('Firing rate PYRs')
            if to_plot > 1:
                # plot some random voltage traces
                figure()
                for ii in range(10):
                    subplot(10, 1, ii + 1)
                    plot(asarray(state_monitorPYR.v[randint(0, nPYRS)]))
                suptitle('Example voltage traces PYRs - ' + title_suffix[idx])
                    
            # plot spikes for INs
            figure() 
            subplot(1,2,1)
            plot(spike_monitorIN.t, spike_monitorIN.i, 'k*')
            subplot(1,2,2)
            plot(convolve(populationIN.rate, ones(1000), 'same')/1000)
            suptitle('Firing rate INs')
            if to_plot > 1:
                # plot some random voltage traces
                figure()
                for ii in range(10):
                    subplot(10, 1, ii + 1)
                    plot(asarray(state_monitorIN.v[randint(0, nINs)]))
                suptitle('Example voltage traces INs - ' + title_suffix[idx])
            
        print('mean firing rate PYRs ' + str(num_spikesPYRs[idx]) + ' Hz')    
        print('mean firing rate INs ' + str(num_spikesIN[idx]) + ' Hz')
        
        if to_save == 1:
            # save stuff
            dict2save = {}
            dict2save['IN_spike_matrix'] = spike_matrixIN
            dict2save['PYR_spike_matrix'] = spike_matrixPYR
            savemat(folder4matrices + title_suffix[idx] + ' ' + str(db_idx) + '.mat', dict2save)
        
        del spike_monitorIN, spike_monitorPYR # state_monitorIN, state_monitorPYR, 
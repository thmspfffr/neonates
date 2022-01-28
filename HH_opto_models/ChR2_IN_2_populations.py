# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:02:19 2022

@author: mchini
"""


#%% import package and define functions

import numpy as np
from scipy.io import savemat
from brian2 import *

#%%

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

########### scaling parameters ###########
area_scaling = r_[0.25, 0.35, 0.5, 1]
main_scaling = r_[1, 1.2, 1.4, 2]
Na_scaling = r_[0.1, 0.12, 0.14, 0.2]
V_scaling = r_[-10, -7, -5, 0]
GABA_scaling = r_[-25, -15, -5, 0]
input_scaling = r_[-1.27, -1.24, -1.2, 0]

########### saving and plotting stuff ###########
folder4matrices = 'Q:/Personal/Mattia/PFCmicro/results/modeled spike matrices stim/activate IN/random/'
to_plot = 0
to_save = 1
start_dataset = 0
num_datasets = 5
title_suffix = ['age P4-6', 'age P7-9', 'age P10-12', 'age P old']

########### network parameters ###########
n_neurons = 1000
PYRsProp = 0.8
nPYRS = int(n_neurons * PYRsProp)
nINs = int((n_neurons - nPYRS) / 2)
num_reps = 60 # number of time to repeat simple stim paradigm
simulation_time = 9 * second * num_reps
defaultclock.dt = 0.1 * ms

########### neuron parameters ###########
areas = 20000 * umetre ** 2 * area_scaling # area of neuron 
Cms = (1 * ufarad * cm ** -2) * areas # membrane capacitance
g_ls = (0.1 * msiemens * cm ** -2) * areas * r_[1, 1.2, 1.4, 2] # leak conductance
g_nas = (100 * msiemens * cm ** -2) * areas * r_[0.1, 0.12, 0.14, 0.2] # Na conductance
g_kds = (80 * msiemens * cm ** -2) * areas * r_[1, 1.2, 1.4, 2] # K conductance
#gls = (0.1 * msiemens * cm ** -2) * areas * main_scaling # leak conductance
#g_nas = (100 * msiemens * cm ** -2) * areas * Na_scaling # Na conductance
#g_kds = (80 * msiemens * cm ** -2) * areas * main_scaling # K conductance
Els = (-67 - V_scaling) * mV # Leak current reversal
EK = -100 * mV # K current reversal
ENa = 50 * mV # Na current reversal
Vrests = (-75 - V_scaling) * mV # resting membrane potential
APthrs = (-20 - V_scaling) * mV # action potential threshold
refractory = 3 * ms # refractory period

########### synapses parameters ###########

Ee = 0 * mV # excitatory synapses reversal
Eis = (-80  - GABA_scaling) *mV # inhibitory synapses reversal
we = 6 * nS # excitatory synaptic weight
wi = we * 8 # 50 * nS  # inhibitory synaptic weight
taue = 5 * ms # excitatory synapses decay constant
taui = 10 * ms # inhibitory synapses decay constant
ges = expand_dims((randn(nPYRS) * 1.5 + 4) * 10., 1) * transpose(expand_dims(r_[1, 1.2, 1.4, 2], 1)) * nS
gis = ges * 4
#ges = (randn() * 1.5 + 4) * 10. * nS * main_scaling
#gis = ges * 4
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
amplitude_ends = + 2.5 * main_scaling * namp

########### to randomize ###########
PYRconduc = np.random.randint(800, 1100, nPYRS) / 1000
INconduc = np.random.randint(800, 1200, nINs) / 1000


# define HH model with specific parameters
eqsPYR = Equations('''
dv/dt = (g_l*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
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
g_na : siemens 
g_l : siemens 
g_kd : siemens
''')

eqsIN = Equations('''
dv/dt = (g_l*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
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
g_na : siemens 
g_l : siemens 
g_kd : siemens
''')

eqsIN2 = Equations('''
dv/dt = (g_l*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
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
g_na : siemens 
g_l : siemens 
g_kd : siemens
''')


#%% simulate network

num_spikesPYRs = zeros_like(areas)
num_spikesIN = zeros_like(areas)
num_spikesIN2 = zeros_like(areas)
si = 0 # starting index
title_suffix = ['age P4-6', 'age P7-9', 'age P10-12', 'age P old'][si:]

for idx, (area, Cm, gl, gna, gkd, El, Vrest, APthr, Ei, input_factor, amplitude_end) in enumerate( # g_e, g_i, 
                  zip(areas[si:], Cms[si:], g_ls[si:], g_nas[si:], g_kds[si:], 
                      Els[si:], Vrests[si:], APthrs[si:], Eis[si:], input_scaling[si:], 
                      amplitude_ends[si:])): # ges[si:], gis[si:], 

    # define ramp stimulation
    ramp = get_ramp_current(stim_start, stim_end, t_end, unit_time, amplitude_start,
                        amplitude_end, num_reps)
    
    for kk in arange(start_dataset, num_datasets):
            
        # define parameters of neural network
        PYRs = NeuronGroup(nPYRS,
                           model=eqsPYR, method='exponential_euler',
                           threshold = "v > APthr",  reset = "v = Vrest",
                           refractory=refractory)
        PYRs.g_na = gna * PYRconduc
        PYRs.g_kd = gkd * PYRconduc
        PYRs.g_l = gl * PYRconduc
        IN = NeuronGroup(nINs,
                         model=eqsIN, method='exponential_euler',
                         threshold = "v > APthr",  reset = "v = Vrest",
                         refractory=refractory)
        IN.g_na = gna * INconduc
        IN.g_kd = gkd * INconduc
        IN.g_l = gl * INconduc
        IN2 = NeuronGroup(nINs,
                         model=eqsIN2, method='exponential_euler',
                         threshold = "v > APthr",  reset = "v = Vrest",
                         refractory=refractory)
        IN2.g_na = gna * INconduc
        IN2.g_kd = gkd * INconduc
        IN2.g_l = gl * INconduc
        
        # define synapses parameters
        Cee = Synapses(PYRs, PYRs, on_pre='ge+=we')
        Cei = Synapses(PYRs, IN, on_pre='ge+=we')
        Cei2 = Synapses(PYRs, IN2, on_pre='ge+=we')
        Cie = Synapses(IN, PYRs, on_pre='gi+=wi')
        Cii = Synapses(IN, IN, on_pre='gi+=wi')
        Ci2e = Synapses(IN2, PYRs, on_pre='gi+=wi')
        Cii2 = Synapses(IN2, IN2, on_pre='gi+=wi')
        Cii3 = Synapses(IN, IN2, on_pre='gi+=wi')
        Cii4 = Synapses(IN2, IN, on_pre='gi+=wi')
        Cee.connect(condition='i!=j', p=epsylonEE)
        Cee.delay = 2 * ms
        Cei.connect(p=epsylonEE)
        Cei.delay = 2 * ms
        Cei2.connect(p=epsylonEE)
        Cei2.delay = 2 * ms
        Cie.connect(p=epsylonIN)
        Cie.delay = 2 * ms
        Cii.connect(condition='i!=j', p=epsylonEE)
        Cii.delay = 2 * ms
        Ci2e.connect(p=epsylonIN)
        Ci2e.delay = 2 * ms
        Cii2.connect(condition='i!=j', p=epsylonEE)
        Cii2.delay = 2 * ms        
        Cii3.connect(p=epsylonEE)
        Cii3.delay = 2 * ms
        Cii4.connect(p=epsylonEE)
        Cii4.delay = 2 * ms
        
        # initialize voltage and synaptic conductance
        PYRs.v = Vrest + (rand(nPYRS) * 5 - 5) * mV
        PYRs.ge = ges[:, si + idx]
        PYRs.gi = gis[:, si + idx]
        #PYRs.ge = g_e
        #PYRs.gi = g_i #'(randn() * 12 + 20) * 20. * nS'
        IN_g_idxs = np.random.randint(0, nPYRS, nINs)
        IN.v = Vrest + (rand(nINs) * 5 - 5) * mV
        IN.ge = ges[IN_g_idxs, si + idx]
        IN.gi = gis[IN_g_idxs, si + idx]
        #IN.ge = g_e
        #IN.gi = g_i #'(randn() * 12 + 20) * 20. * nS'
        IN2.v = Vrest + (rand(nINs) * 5 - 5) * mV
        IN2.ge = ges[IN_g_idxs, si + idx]
        IN2.gi = gis[IN_g_idxs, si + idx]
            
        # Generate a Poisson stimulus to inject into the network of PYRs
        # set input rate as something that is log-normal
        input_rate = exp(np.random.normal(input_factor, 0.5, num_imputs)) * Hz
        P = PoissonGroup(num_imputs, rates=input_rate)
        
        SPYR = Synapses(P, PYRs, on_pre='v += input_weight * 5 * mV')
        SPYR.connect(p = epsilonPoisson)
        SPYR.delay = 2 * ms
            
        # define parameters to track
        spike_monitorPYR = SpikeMonitor(PYRs)
        spike_monitorIN = SpikeMonitor(IN)
        spike_monitorIN2 = SpikeMonitor(IN2)
        if to_plot > 0:
            populationPYR = PopulationRateMonitor(PYRs)
            populationIN = PopulationRateMonitor(IN)
            populationIN2 = PopulationRateMonitor(IN2)
        if to_plot > 1:
            state_monitorPYR = StateMonitor(PYRs, 'v', record=True)
            state_monitorIN = StateMonitor(IN, 'v', record=True)
            state_monitorIN2 = StateMonitor(IN2, 'v', record=True)
        
        # running the simulation
        run(simulation_time)
        
        # extract average number of spikes
        num_spikesPYRs[idx] = len(spike_monitorPYR.t) / (nPYRS * simulation_time)
        num_spikesIN[idx] = len(spike_monitorIN.t) / (nINs * simulation_time)
        num_spikesIN2[idx] = len(spike_monitorIN2.t) / (nINs * simulation_time)
        
        # extract spike matrices
        spike_matrixPYR = get_spike_matrix(spike_monitorPYR, nPYRS,
                                           int(asarray(simulation_time) * 1000))
        spike_matrixIN = get_spike_matrix(spike_monitorIN, nINs,
                                           int(asarray(simulation_time) * 1000))
        spike_matrixIN2 = get_spike_matrix(spike_monitorIN2, nINs,
                                           int(asarray(simulation_time) * 1000))
        
        # plot stuff
        if to_plot > 0:
            # plot spikes for PYRs
            figure() 
            subplot(1,2,1)
            plot(spike_monitorPYR.t, spike_monitorPYR.i, 'k*')
            subplot(1,2,2)
            plot(convolve(populationPYR.rate, ones(1000), 'same') / 1000)
            suptitle('Firing rate PYRs - ' + title_suffix[idx])
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
            suptitle('Firing rate INs - ' + title_suffix[idx])
            if to_plot > 1:
                # plot some random voltage traces
                figure()
                for ii in range(10):
                    subplot(10, 1, ii + 1)
                    plot(asarray(state_monitorIN.v[randint(0, nINs)]))
                suptitle('Example voltage traces INs - ' + title_suffix[idx])

            # plot spikes for IN2
            figure() 
            subplot(1,2,1)
            plot(spike_monitorIN2.t, spike_monitorIN2.i, 'k*')
            subplot(1,2,2)
            plot(convolve(populationIN2.rate, ones(1000), 'same')/1000)
            suptitle('Firing rate IN2s - ' + title_suffix[idx])
            if to_plot > 1:
                # plot some random voltage traces
                figure()
                for ii in range(10):
                    subplot(10, 1, ii + 1)
                    plot(asarray(state_monitorIN2.v[randint(0, nINs)]))
                suptitle('Example voltage traces IN2s - ' + title_suffix[idx])
            
        print(title_suffix[idx] + ' - mean firing rate PYRs ' + str(num_spikesPYRs[idx]) + ' Hz')    
        print(title_suffix[idx] + ' - mean firing rate INs ' + str(num_spikesIN[idx]) + ' Hz')
        print(title_suffix[idx] + ' - mean firing rate IN2s ' + str(num_spikesIN2[idx]) + ' Hz')
        
        if to_save == 1:
            # save stuff
            dict2save = {}
            dict2save['IN_spike_matrix'] = spike_matrixIN
            dict2save['IN_spike_matrix_2'] = spike_matrixIN2
            dict2save['PYR_spike_matrix'] = spike_matrixPYR
            savemat(folder4matrices + title_suffix[idx] + ' ' + str(kk) + '.mat', dict2save)
        
        del spike_monitorIN, spike_monitorPYR # state_monitorIN, state_monitorPYR, 
    
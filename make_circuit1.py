from brian import *
import numpy


def make_circuit(inp,GABA_mod,AMPA_mod):
    
    '''
    Creates the spiking network described in Wang 2002.
    
    returns:
        groups, connections, update_nmda, subgroups
        
    groups, connections, and update_nmda have to be added to the "Network" in order to run the simulation.
    subgroups is used for establishing connections between the sensory and integration circuit; do not add subgroups to the "Network"

    ''' 
    
    # -----------------------------------------------------------------------------------------------
    # Model parameters for the integration circuit
    # ----------------------------------------------------------------------------------------------- 
    # Populations
    N = 400                                     # Total number of neurons
    f_inh = 0.20                                  # Fraction of inhibitory neurons
    NE = int(N * (1.0 - f_inh))                  # Number of excitatory neurons (320)
    NI = int(N * f_inh)                          # Number of inhibitory neurons (80)
    
    # Connectivity - local recurrent connections
    gEE_AMPA = 0.178 * AMPA_mod * nS		         # Weight of AMPA synapses between excitatory neurons
    gEI_AMPA = 0.233 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
    gIE_GABA = 2.01 * GABA_mod * nS              # Weight of inhibitory to excitatory synapses (GABA)
    gII_GABA = 2.7 * nS                             # Weight of inhibitory to inhibitory synapses (GABA)
      
    d = 0 * ms                                   # Transmission delay of recurrent excitatory and inhibitory connections
                                
    # Connectivity - external connections
    gextE = 0.234 * nS                             # Weight of external input to excitatory neurons: Increased from previous value
    gextI = 0.317 * nS                            # Weight of external input to inhibitory neurons

    # Neuron model
    CmE = 0.5 * nF                               # Membrane capacitance of excitatory neurons
    CmI = 0.2 * nF                               # Membrane capacitance of inhibitory neurons
    gLeakE = 25.0 * nS                           # Leak conductance of excitatory neurons
    gLeakI = 20.0 * nS                           # Leak conductance of inhibitory neurons
    Vl = -70.0 * mV                              # Resting potential
    Vt = -52.0 * mV                              # Spiking threshold
    Vr = -59.0 * mV                              # Reset potential
    tau_refE = 2.0 * ms                          # Absolute refractory period of excitatory neurons
    tau_refI = 1.0 * ms                          # Absolute refractory period of inhibitory neurons

    # Synapse model
    VrevE = 0 * mV                               # Reversal potential of excitatory synapses
    VrevI = -80 * mV                             # Reversal potential of inhibitory synapses

    tau_AMPA_decay_E = 2.0 * ms                  # Decay constant of AMPA-type conductances
    tau_AMPA_decay_I = 1.0 * ms                  # Decay constant of AMPA-type conductances
    tau_AMPA_rise_E = 0.4 * ms                   # Rise constant of AMPA-type conductances
    tau_AMPA_rise_I = 0.2 * ms                   # Rise constant of AMPA-type conductances

    tau_GABA_decay_E = 5 * ms                          # Decay constant of GABA-type conductances
    tau_GABA_decay_I = 5 * ms                          # Decay constant of GABA-type conductances
    tau_GABA_rise_E = 0.25 * ms                          # Decay constant of GABA-type conductances
    tau_GABA_rise_I = 0.25 * ms                          # Decay constant of GABA-type conductances

    tau_AMPA = 2 * ms
    tau_GABA = 1 * ms
    # Inputs
    nu_ext_exc = inp * 5 * Hz                       # Firing rate of external Poisson input to excitatory neurons
    nu_ext_inh = inp * 5 * Hz                       # Firing rate of external Poisson input to inhibitory neurons

    # -----------------------------------------------------------------------------------------------
    # Set up the model
    # ----------------------------------------------------------------------------------------------- 
     
    # Neuron equations
    eqsE = '''
    dV/dt = (gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + I/Cm: volt
    dgea/dt = -gea/(tau_AMPA_decay_E) : 1
    dgi/dt = -gi/(tau_GABA_decay_E) : 1
    tau : second
    I : nA
    Cm : nF
    '''
    eqsI = '''
    dV/dt = (gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) / (tau) + I/Cm: volt
    dgea/dt = -gea/(tau_AMPA_decay_I) : 1
    dgi/dt = -gi/(tau_GABA_decay_I) : 1
    tau : second
    I : nA
    Cm : nF
    '''

    # Set up the integration circuit
    popE = NeuronGroup(NE, model=eqsE, threshold=Vt, reset=Vr, refractory=tau_refE)
    popI = NeuronGroup(NI, model=eqsI, threshold=Vt, reset=Vr, refractory=tau_refI)
    popE.tau = CmE / gLeakE
    popI.tau = CmI / gLeakI      
    popE.I = 0.8 * nA
    popI.I = 0.7 * nA
    popE.Cm = CmE
    popI.Cm = CmI
    
    # Connections involving AMPA synapses
    C_DE_DE_AMPA = Connection(popE, popE, 'gea', delay = d, weight = gEE_AMPA / gLeakE, sparseness=0.2)     
    C_DE_DI_AMPA = Connection(popE, popI, 'gea', delay = d, weight = gEI_AMPA / gLeakI, sparseness=0.2)
    
    # Connections involving GABA synapses
    C_DI_DE = Connection(popI, popE, 'gi', delay = d, weight = gIE_GABA / gLeakE, sparseness=0.2)
    C_DI_DI = Connection(popI, popI, 'gi', delay = d, weight = gII_GABA / gLeakI, sparseness=0.2)   

    # External inputs
    extinputE = PoissonGroup(NE, rates = nu_ext_exc) 
    extinputI = PoissonGroup(NI, rates = nu_ext_inh)
   
    # Connect external inputs
    extconnE = IdentityConnection(extinputE, popE, 'gea', weight = gextE / gLeakE)
    extconnI = IdentityConnection(extinputI, popI, 'gi', weight = gextI / gLeakI)

    # Stimulus input
    # Connect stimulus inputs
    
    # Return the integration circuit
    groups = {'DE': popE, 'DI': popI, 'DXE': extinputE, 'DXI': extinputI}
    connections = {'extconnE': extconnE, 'extconnI': extconnI, 'C_DE_DE_AMPA': C_DE_DE_AMPA, 'C_DE_DI_AMPA': C_DE_DI_AMPA, 'C_DI_DE': C_DI_DE, 'C_DI_DI': C_DI_DI}
                     
    return groups, connections

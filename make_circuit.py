from brian import *
import numpy


def make_circuit(inp,GABA_mod,AMPA_mod,NMDA_mod):
    
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
    gEE_AMPA = 0.05 * AMPA_mod * nS		         # Weight of AMPA synapses between excitatory neurons
    gEE_NMDA = 0.165 * NMDA_mod * nS             # Weight of NMDA synapses between excitatory neurons
    gEI_AMPA = 0.04 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
    gEI_NMDA = 0.13 * nS                         # Weight of excitatory to inhibitory synapses (NMDA)
    gIE_GABA = 1.99 * GABA_mod * nS              # Weight of inhibitory to excitatory synapses (GABA)
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

    # Inputs
    nu_ext_exc = inp * 2000 * Hz                       # Firing rate of external Poisson input to excitatory neurons
    nu_ext_inh = inp * 2000 * Hz                       # Firing rate of external Poisson input to inhibitory neurons
    stim_ext = 0 * Hz

    # -----------------------------------------------------------------------------------------------
    # Set up the model
    # ----------------------------------------------------------------------------------------------- 
    
    # Neuron equations
    eqsE = '''
    dV/dt = (-gea*(V-VrevE) - gen*(V-VrevE)/(1.0+exp(-V/mV*0.062)/3.57) - gi*(V-VrevI) - (V-Vl)) / (tau) + I/Cm: volt
    dgea/dt = -gea/(tau_AMPA) : 1
    dgi/dt = -gi/(tau_GABA) : 1
    dspre/dt = -spre/(tau_NMDA_decay)+alpha_NMDA*xpre*(1-spre) : 1
    dxpre/dt= -xpre/(tau_NMDA_rise) : 1
    gen : 1
    tau : second
    I : nA
    Cm : nF
    '''
    eqsI = '''
    dV/dt = (-gea*(V-VrevE) - gen*(V-VrevE)/(1.0+exp(-V/mV*0.062)/3.57) - gi*(V-VrevI) - (V-Vl)) / (tau) + I/Cm: volt
    dgea/dt = -gea/(tau_AMPA) : 1
    dgi/dt = -gi/(tau_GABA) : 1
    gen : 1
    tau : second
    I : nA
    Cm : nF
    '''

    # Set up the integration circuit
    popE = NeuronGroup(NE, model=eqsE, threshold=Vt, reset=Vr, refractory=tau_refE)
    popI = NeuronGroup(NI, model=eqsI, threshold=Vt, reset=Vr, refractory=tau_refI)
    popE.tau = CmE / gLeakE
    popI.tau = CmI / gLeakI      
    popE.I = 0.0 * nA
    popI.I = 0.0 * nA
    popE.Cm = CmE
    popI.Cm = CmI
    
    # Connections involving AMPA synapses
    C_DE_DE_AMPA = Connection(popE, popE, 'gea', delay = d)             
    C_DE_DE_AMPA.connect_full(popE, popE, weight = gEE_AMPA / gLeakE)
    C_DE_DI_AMPA = Connection(popE, popI, 'gea', weight = gEI_AMPA / gLeakI, delay = d)

    # Connections involving NMDA synapses    
    # Note that due to the all-to-all connectivity, the contribution of NMDA can be calculated efficiently
    selfnmda = IdentityConnection(popE, popE, 'xpre', weight=1.0, delay = d)    
    E_nmda = asarray(popE.spre)
    E_gen = asarray(popE.gen) 
    I_gen = asarray(popI.gen)

    # Calculate NMDA contributions in each time step
    @network_operation(when='start')
    def update_nmda():
        sE = sum(E_nmda)
        E_gen[:] = gEE_NMDA / gLeakE * sE
        I_gen[:] = gEI_NMDA / gLeakI * sE  
    
    # Connections involving GABA synapses
    C_DI_DE = Connection(popI, popE, 'gi', weight = gIE_GABA / gLeakE, delay = d)
    C_DI_DI = Connection(popI, popI, 'gi', weight = gII_GABA / gLeakI, delay = d)
    
    # External inputs
    extinputE = PoissonGroup(NE, rates = nu_ext_exc) 
    extinputI = PoissonGroup(NI, rates = nu_ext_inh)
   
    # Connect external inputs
    extconnE = IdentityConnection(extinputE, popE, 'gea', weight = gextE / gLeakE)
    extconnI = IdentityConnection(extinputI, popI, 'gea', weight = gextI / gLeakI)

    # Stimulus input
    stiminput = PoissonGroup(NE, rates = stim_ext) 
    # Connect stimulus inputs
    stimconn = IdentityConnection(stiminput, popE, 'gea', weight = gextE / gLeakE)
    
    # Return the integration circuit
    groups = {'DE': popE, 'DI': popI, 'DXE': extinputE, 'DXI': extinputI, 'DSE': stiminput}
    subgroups = {'None'}
    connections = {'selfnmda': selfnmda,
                   'extconnE': extconnE, 'extconnI': extconnI, 
                   'stimconn': stimconn, 
                   'C_DE_DE_AMPA': C_DE_DE_AMPA, 'C_DE_DI_AMPA': C_DE_DI_AMPA, 'C_DI_DE': C_DI_DE, 'C_DI_DI': C_DI_DI }
                     
    return groups, connections, update_nmda, subgroups

import pandas as pd
from textwrap import dedent

# brian 2
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network
from brian2 import mV, ms, Hz

# file handling
from pathlib import Path

# parallelization
from joblib import Parallel, delayed, parallel_backend
from time import time

default_params = {
    # trials
    't_run'     : 1000 * ms,              # duration of trial
    'n_run'     : 30,                     # number of runs

    # network constants
    # Kakaria and de Bivort 2017 https://doi.org/10.3389/fnbeh.2017.00008
    'v_0'       : -52 * mV,               # resting potential
    'v_rst'     : -52 * mV,               # reset potential after spike
    'v_th'      : -45 * mV,               # threshold for spiking
    't_mbr'     :  20 * ms,               # membrane time scale (capacitance * resistance = .002 * uF * 10. * Mohm)

    # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
    'tau'       : 5 * ms,                 # time constant 

    # Lazar et al https://doi.org/10.7554/eLife.62362
    't_rfc'     : 2.2 * ms,               # refractory period

    # Paul et al 2015 doi: 10.3389/fncel.2015.00029
    't_dly'     : 1.8*ms,                 # delay for changes in post-synaptic neuron

    # Free parameter 
    'w_syn'     : .275 * mV,              # weight per synapse (note: modulated by exponential decay)
    # Default activation rates 
    'r_poi'     : 150*Hz,                 # default rate of the Poisson inputs
    'r_poi2'    :   0*Hz,                 # default rate of a 2nd class of Poisson inputs
    'f_poi'     : 250,                    # scaling factor for Poisson synapse; 250 is sufficient to cause spiking

    # equations for neurons               # alpha synapse https://doi.org/10.1017/CBO9780511815706; See https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
    'eqs'       : dedent(''' 
                    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                    dg/dt = -g / tau               : volt (unless refractory) 
                    rfc                            : second
                    '''),
    # condition for spike
    'eq_th'     : 'v > v_th', 
    # rules for spike        
    'eq_rst'    : 'v = v_rst; w = 0; g = 0 * mV', 
}


#######################
# brian2 model setup
def poi(neu, exc, exc2, params):
    '''Create PoissonInput for neurons.

    For each neuron in 'names' a PoissonInput is generated and 
    the refractory period of that neuron is set to 0 in NeuronGroup.

    Parameters
    ----------
    neu : NeuronGroup
        Defined brian2.NeuronGroup object
    exc : list
        Indices of neurons for which to create Poisson input
    exc2 : list
        Indices of neurons for which to create Poisson input with `r_poi`
    params : dict
        Constants and equations that are used to construct the brian2 network model

    Returns
    -------
    pois : list
        PoissonInput objects for each neuron in 'exc'
    neu : NeuronGroup
        NeuronGroup with adjusted refractory periods
    '''

    pois = []
    for i in exc:
        p = PoissonInput(
            target=neu[i], 
            target_var='v', 
            N=1, 
            rate=params['r_poi'], 
            weight=params['w_syn']*params['f_poi']
            )
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        pois.append(p)

    for i in exc2:
        p = PoissonInput(
            target=neu[i], 
            target_var='v', 
            N=1, 
            rate=params['r_poi2'], 
            weight=params['w_syn']*params['f_poi']
            )
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        pois.append(p)

    return pois, neu

def silence(slnc, syn):
    '''Silence neuron by setting weights of all synapses from it to 0

    Parameters
    ----------
    slnc : list
        List of neuron indices to silence
    syn : brian2.Synapses
        Defined synapses object

    Returns
    -------
    syn : brian2.Synapses
        Synapses with modified weights
    '''

    for i in slnc:
        syn.w[' {} == i'.format(i)] = 0*mV
    
    return syn

def create_model(path_comp, path_con, params):
    '''Create default network model.

    Convert the "completeness materialization" and "connectivity" dataframes
    into a brian2 neural network model. Network constants and equations
    are defined at the beginning of this file.

    Parameters
    ----------
    path_comp : str
        path to "completeness materialization" dataframe
    path_con : str
        path to "connectivity" dataframe
    params : dict
        Constants and equations that are used to construct the brian2 network model


    Returns
    -------
    neu : NeuronGroup
        brian2.NeuronGroup object with neurons as in 'path_comp'
    syn : Synapses
        brian2.Synapses object with connections as in 'path_con'
    spk_mon : SpikeMonitor
        brian2.SpikeMonitor object, which records time of spike events
    '''

    # load neuron connectivity dataframes
    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)

    neu = NeuronGroup( # create neurons
        N=len(df_comp),
        model=params['eqs'],
        method='linear',
        threshold=params['eq_th'],
        reset=params['eq_rst'],
        refractory='rfc',
        name='default_neurons',
        namespace=params,
    )
    neu.v = params['v_0'] # initialize values
    neu.g = 0
    neu.rfc = params['t_rfc']

    # create synapses
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'], name='default_synapses')

    # connect synapses
    i_pre = df_con.loc[:, 'Presynaptic_Index'].values
    i_post = df_con.loc[:, 'Postsynaptic_Index'].values
    syn.connect(i=i_pre, j=i_post)

    # define connection weight
    syn.w = df_con.loc[:,'Excitatory x Connectivity'].values * params['w_syn']

    # object to record spikes
    spk_mon = SpikeMonitor(neu) 

    return neu, syn, spk_mon

#####################
# running simulations
def get_spk_trn(spk_mon):
    '''Extracts spike times from 'spk_mon'

    The spike times recorded in the SpikeMonitor object during 
    simulation are converted to a list of times for each neurons.
    Returns dict with "brian ID": "list of spike times".

    Parameters
    ----------
    spk_mon : SpikeMonitor
        Contains recorded spike times

    Returns
    -------
    spk_trn : dict
        Mapping between brian neuron IDs and spike times
    '''

    spk_trn = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    
    return spk_trn

def construct_dataframe(res, exp_name, i2flyid):
    '''Take spike time dict and collects spikes in pandas dataframe

    Parameters
    ----------
    res : list
        List with spike time dicts for each trial
    exp_name : str
        Name of the experiment
    i2flyid : dict
        Mapping between Brian IDs and flywire IDs

    Returns
    -------
    df : pandas.DataFrame
        Dataframe where each row is one spike
    '''
    
    ids, ts, nrun = [], [], []

    for n, i in enumerate(res):
        for j, k  in i.items():
            ids.extend([j for _ in k])
            nrun.extend([n for _ in k])
            ts.extend([float(l) for l in k])

    d = {
        't': ts,
        'trial': nrun,
        'flywire_id': [i2flyid[i] for i in ids],
        'exp_name': exp_name
    }
    df = pd.DataFrame(d)

    return df

def run_trial(exc, exc2, slnc, path_comp, path_con, params):
    '''Run single trial of coactivation/silencng experiment

    During the coactivation experiment, the neurons in 'exc' are
    Poisson inputs. The simulation runs for 't_run'.
    

    Parameters
    ----------
    exc: list
        contains indices of neurons for PoissonInput
    exc2: list
        contains indices of neurons for PoissonInput of different frequency `r_poi2`
    slnc: list
        contains indices of neurons to silence
    path_comp: Path 
        path to "completeness materialization" dataframe
    path_con: Path
        path to "connectivity" dataframe
    params : dict
        Constants and equations that are used to construct the brian2 network model

    Returns
    -------
    spk_trn : dict
        Mapping between brian neuron IDs and spike times
    '''

    # get default network
    neu, syn, spk_mon = create_model(path_comp, path_con, params)
    # define Poisson input for excitation
    poi_inp, neu = poi(neu, exc, exc2, params)
    # silence neurons
    syn = silence(slnc, syn)
    # collect in Network object
    net = Network(neu, syn, spk_mon, *poi_inp)

    # run simulation
    net.run(duration=params['t_run'])

    # spike times 
    spk_trn = get_spk_trn(spk_mon)

    return spk_trn


def run_exp(exp_name, neu_exc, path_res, path_comp, path_con,
            params=default_params, neu_slnc=[], neu_exc2=[], 
            n_proc=-1, force_overwrite=False):
    '''
    Run default network experiment 
    Neurons in `neu_exc` are Poisson external inputs
    Neurons in `neu_slnc` are silenced


    Parameters
    ----------
        exp_name: str
            name of the experiment
        neu_exc: list
            contains custom names or flywire IDs of neurons to be excited
        path_res: str
            path to the output folder where spike data is stored
        path_comp: str 
            path to "completeness materialization" dataframe
        path_con: str
            path to "connectivity" dataframe
        params : dict
            Constants and equations that are used to construct the brian2 network model
        neu_slnc : list (optional)
            contains custom names or flywire IDs of neurons to be silenced
        neu_exc2 : list (optional)
            contains custom names or flywire IDs of neurons to be excited with different frequency
            requires `r_poi2` to be defined in `params`
        n_proc: int (optional)
            number of cores to be used for parallel runs
            default: -1 (use all available cores)
            n_proc=1 is equivalent serial code
        force_overwrite : bool (optional)
            If True, overwrite output files, else skip simulation, by default False
    '''

    # convert to Path objects
    path_res, path_comp, path_con = [ Path(i) for i in [path_res, path_comp, path_con] ]

    # define output files
    path_save = path_res / '{}.parquet'.format(exp_name)
    if path_save.is_file() and not force_overwrite:
        print('>>> Skipping experiment {} because {} exists and force_overwrite = {}'.format(exp_name, path_save, force_overwrite))
        return 

    # load name/id mappings
    df_comp = pd.read_csv(path_comp, index_col=0) # load completeness dataframe
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}  # flywire id: biran ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID
    
    # print info
    print('>>> Experiment:     {}'.format(exp_name))
    print('    Output file:    {}'.format(path_save))
    print('    Excited neurons: {}'.format(len(neu_exc + neu_exc2)))
    if neu_slnc:
        print('    Silenced neurons: {}'.format(len(neu_slnc)))
    
    # start parallel calculation
    n_run = params['n_run']
    start = time() 
    with parallel_backend('loky', n_jobs=n_proc):
        exc = [ flyid2i[n] for n in neu_exc ]
        exc2 = [ flyid2i[n] for n in neu_exc2 ]
        slnc = [ flyid2i[n] for n in neu_slnc ]
        res = Parallel()(
            delayed(
                run_trial)(exc, exc2, slnc, path_comp, path_con, params) for _ in range(n_run))

        
    # print simulation time
    walltime = time() - start 
    print('    Elapsed time:   {} s'.format(int(walltime)))

    # dataframe with spike times
    df = construct_dataframe(res, exp_name, i2flyid)

    # store spike data
    df.to_parquet(path_save, compression='brotli')

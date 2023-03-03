import pandas as pd
import numpy as np

import pickle

def useful_mappings(name2flyid, path_comp):
    '''Generate other useful mappings between custom names, flywire IDs
    and canonical IDs (starting with 0, will be equivalent to brian neuron IDs)

    Parameters
    ----------
    name2flyid : dict
        Mapping between custom neuron names and flywire IDs
    path_comp : str
        Path to completeness dataframe

    Returns
    -------
    flyid2name : dict
        Inverted name2flyid dictionary
    flyid2i : dict
        Mapping between flywire IDs and canonical IDs
    i2flyid : dict
        Inverted flyid2i dictionary
    name2i : dict
        Mapping between custom neuron names and canonical IDs
    i2name : dict
        Inverted name2s dictionary
    name_flyid2i : dict
        Mapping of custom neuron names and flywire IDs to canonical IDs
    '''

    flyid2name = { j: i for i, j in name2flyid.items() } # flywire ID: custom name

    df_comp = pd.read_csv(path_comp, index_col=0) # load completeness dataframe

    flyid2i = {j: i for i, j in enumerate(df_comp.index)}  # flywire id: biran ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    name2i = {i: flyid2i[j] for i, j in name2flyid.items() } # custom name: brian ID
    i2name = {j: i for i, j in name2i.items() } # brian ID: custom name

    name_flyid2i = name2i | flyid2i # union of dicts

    return flyid2name, flyid2i, i2flyid, name2i, i2name, name_flyid2i

##########
# analysis
def load_exps(l_pqt):
    '''Load simulation results from disk

    Parameters
    ----------
    l_pkl : list
        List of parquet files with simulation results

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs = []
    for p in l_pqt:
        # load metadata from pickle
        with open(p, 'rb') as f:
            df = pd.read_parquet(p)
            df.loc[:, 't'] = df.loc[:, 't'].astype(float)
            dfs.append(df)

    df = pd.concat(dfs)

    return df

def get_rate(df, duration):
    '''Calculate rate and standard deviation for all experiments
    in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    duration : float
        Trial duration in seconds

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, flyid, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name', sort=False):
        for f, df_f in df_e.groupby('flywire_id'):

            r = []
            for _, df_t in df_f.groupby('trial'):
                r.append(len(df_t) / duration)
            r = np.array(r)

            rate.append(r.mean())
            std.append(r.std())
            flyid.append(f)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'flyid' : flyid,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='flyid', values='r')
    df_std = df.pivot_table(columns='exp_name', index='flyid', values='std')

    return df_rate, df_std

def rename_index(df, name2flyid):
    '''Rename flywire IDs to custom neuron names in index
    Also sort index and columns

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with flywire IDs as index
    name2flyid : dict
        Mapping between custom neuron names and flywire IDs

    Returns
    -------
    df : pd.DataFrame
        Renamed and sorted dataframe
    '''

    # replace flywire IDs with custom names
    flyid2name = {v: k for k, v in name2flyid.items()}
    df = df.rename(index=flyid2name)

    # sort: str first (i.e. custom names), then int (i.e. flywire IDs)
    df.index = df.index.astype(str)
    df = df.loc[
        sorted(sorted(df.index.astype(str)), key=lambda x: (x[0].isdigit(), x)), 
        sorted(df.columns.sort_values(), key=lambda x: len(x.split('+')))
        ]
    
    return df
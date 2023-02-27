# -*- coding: utf-8 -*-
"""Generate_name_dictionaries.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fZnyeHQB0I6-du16SlnpJUZMVv_upBKI
"""

#Create flywire dictionaries: flyid2name, flyid2i, i2flyid, name2i, i2name, name2flyid

import os   
#Connect to your google drive. Not necessary if you don't want to save your csv files.
from google.colab import drive
drive.mount('/content/drive/')

path = '/content/drive/MyDrive/Scott lab/Computational Modelling/Test/'
os.chdir(path)

import pandas as pd
import pickle

# files to read/write
neurotransmitter_information_file    = './2023_02_01_completeness_587_final.csv'
#Map between names you've given your neurons and Flywire IDs.
xls_map     = './flywire_ids_587.xlsx'
pickle_map  = './name_mappings_587.pickle'
neurotransmitter_dataframe = pd.read_csv(neurotransmitter_information_file, index_col = 0) # neuron ids and excitation type

# helper functions
def check_unique(df_sing, df_paired):
    'check if names or IDs occur only once in dataframes created from xls_map'
    # names
    ds = pd.concat( (df_sing.loc[:, 'Name'], df_pair.loc[:, 'Name']), ignore_index=True) # merge names from both dataframes
    dup = ds.loc[ ds.duplicated(keep=False )] # series with duplicate values
    if dup.empty:
        print('INFO: All Names are unique')
    else:
        print('WARNING: Found duplicate Names:')
        print(dup)

    # IDs
    ds = pd.concat( (df_sing.loc[:, 'ID'], df_pair.loc[:, 'ID_left'], df_pair.loc[:, 'ID_right']), ignore_index=True) # merge IDs from both dataframes
    ds = ds.dropna() # igrone nan for IDs
    dup = ds.loc[ ds.duplicated(keep=False )] # series with duplicate values
    if dup.empty:
        print('INFO: All IDs are unique')
    else:
        print('WARNING: Found duplicate IDs:')
        ds_n = pd.concat( (df_sing.loc[:, 'Name'], df_pair.loc[:, 'Name'], df_pair.loc[:, 'Name']), ignore_index=True) # series with names of same structure as ds
        print(pd.concat( (ds_n.loc[dup.index], dup), axis=1 ) ) # pring names and IDs

def assign(name2id, name, id):
    'Assign id as integer to name in name2id dict, print warning if unsuccessful'
    try:
        name2id[name] = int(id)
    except ValueError:
        print('WARNING: Could not assign ID {} to name {}'.format(id, name))

def check_ids(ids_dict, ids_all):
    'print warning if IDs in dictionary are not present in indices of dataframe'
    warn = False
    for k, v in ids_dict.items():
        if not v in ids_all:
            print('WARNING: ID {} for neuron {} not found'.format(str(v), k))
            warn = True
    if not warn:
        print('INFO: IDs in {} appear to match with {}'.format(xls_map, neurotransmitter_information_file))

# Load excel file with name -> flywire ID definitions
dfs_pair = pd.read_excel( # sheets with left/right pairs
    xls_map,
    sheet_name=[ 'SEZ'],
    dtype={'ID_left': str, 'ID_right': str}
    )
df_pair = pd.concat(dfs_pair, ignore_index=True).dropna(how='all')

dfs_sing = pd.read_excel( # sheets where hemisphere doesn't matter
    xls_map,
    sheet_name=['sugar', 'bitter', 'water', 'jo_e', 'jo_f', 'jo_cdm', 'Ir94e'],
    dtype={'ID': str}
    )
df_sing = pd.concat(dfs_sing, ignore_index=True).dropna(how='all')

print('Loaded sheets ...')
for i in [*dfs_pair.keys(), *dfs_sing.keys()]:
    print('... {}'.format(i))

check_unique(df_sing, df_pair)

# create dictionary with custom names for given flywire ID
name2flyid = dict()

for i in df_pair.index: # left/right pairs
    n, id_l, id_r = df_pair.loc[i, ['Name', 'ID_left', 'ID_right']]
    n_l, n_r = '{}_l'.format(n), '{}_r'.format(n)
    assign(name2flyid, n_l, id_l)
    assign(name2flyid, n_r, id_r)

for i in df_sing.index: # single neurons
    n, id = df_sing.loc[i, ['Name', 'ID']]
    assign(name2flyid, n, id)    

print( 'Declared {} names for neurons'.format(len(name2flyid)))

# check if IDs are correct: if everything is correct, nothing is printed
flyids_all = neurotransmitter_dataframe.index # all flywire ids
check_ids(name2flyid, flyids_all)

# now that IDs are correct, define useful id/name mappings
flyid2name = { j: i for i, j in name2flyid.items() } # flywire ID: custom name

flyid2i = {j: i for i, j in enumerate(neurotransmitter_dataframe.index)}  # flywire id: biran ID
i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

name2i = {i: flyid2i[j] for i, j in name2flyid.items() } # custom name: brian ID
i2name = {j: i for i, j in name2i.items() } # brian ID: custom name

# save mappings to disk
list_of_dicts = [flyid2i, flyid2name, i2flyid, i2name, name2flyid, name2i ]
with open(pickle_map, 'wb') as f:
    pickle.dump(list_of_dicts, f, protocol=4)
print('Written pickle file: {}'.format(pickle_map))


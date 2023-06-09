# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:07:22 2023

@author: ikerl
"""

import hypernetx as hnx
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xgi
import itertools
import scipy.io as sio
import networkx as nx
from networkx import NetworkXException
import hyperplot
import matplotlib as mpl

def rawdata2data(rawdata, min_ord=2, max_ord=4,min_var=1,max_var=4,):
    '''
    Input: rawdata (dict with '__header__', 'Otot', 'data' fields).
    Output: dict with Otot fields (sorted_red, index_red, bootsig_red, etc) and 'orders'.
    '''

    data = {'sorted_red': {}, 'index_red': {}, 'bootsig_red': {},
                'sorted_syn': {}, 'index_syn': {}, 'bootsig_syn': {}}

    for target in range(min_var,max_var):
     for order in range(min_ord, max_ord+1):
        for key in data.keys():
           
            tmp = rawdata['Otot'][target][order][key]
            
            if not hasattr(tmp, '__len__'):  # convert matlab singletons to array
                tmp = np.array([tmp])

            if key == 'index_var' :  # VERY IMPORTANT!! matlab to python indexing
                data[key][order] = tmp-1
            else:
                data[key][order] = tmp  # python indexing

    data['orders'] = list(range(min_ord, max_ord+1))
     
    return data


def load_dataset(fpath, min_ord, max_ord,min_var=1,max_var=4, n_dims=None, n_points=None):
    '''
    Load dataset (output from O-info analysis, i.e. Otot structure)
    Returns
    -------
    data : dict ('sorted_red', 'index_red', 'bootsig_red', 'sorted_syn', 'index_syn', 'bootsig_syn')
    '''
    rawdata = loadmat(fpath)
    print(rawdata)
    for order in range(min_ord, max_ord):
     rawdata['Otot'] = [todict(x) for x in rawdata]
    data = rawdata2data(rawdata, min_ord, max_ord)
   
    return data

# functions to load .mat
def loadmat(fname):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    fname = str(fname)
    data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def todict(matobj):
    """
    A recursive function which constructs from mat objects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict

def _check_keys(dict_in,min_ord=2, max_ord=4,min_var=2,max_var=4,):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict_in['Otot']:
     
      for order in range(min_ord, max_ord):
      
       if isinstance(key[order], sio.matlab.mio5_params.mat_struct):
            key[order] = todict(key[order])
      
    return dict_in



dataset='lagged'
DATASET_DIR = Path.cwd() / 'data'
SAVE_DIR = Path.cwd() / 'figs'
 # LOAD DATA
print(f'DATASET: {dataset.upper()}')
print(8 * '=')
if dataset=='lagged':
         fpath = DATASET_DIR / 'Otot_lagged.mat'
         data = load_dataset(fpath, min_ord=1, max_ord=4)
else:
         raise ValueError('Dataset not accepted.') 
   


   



#gradient['synappear']=np.zeros(len(data["index_var"][1]))

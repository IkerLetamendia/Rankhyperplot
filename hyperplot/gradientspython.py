
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import itertools
import scipy.io as sio

import hyperplot

def rawdata2data(rawdata, min_ord=1, max_ord=2):
    '''
    Input: rawdata (dict with '__header__', 'Otot', 'data' fields).
    Output: dict with Otot fields (sorted_red, index_red, bootsig_red, etc) and 'orders'.
    '''

    data = {'O_val': {}, 'index_var': {}}

    for order in range(min_ord, max_ord+1):
        for key in data.keys():
            tmp = rawdata['goi'][order - 1][key]
            
            if not hasattr(tmp, '__len__'):  # convert matlab singletons to array
                tmp = np.array([tmp])

            if key == 'index_var' :  # VERY IMPORTANT!! matlab to python indexing
                data[key][order] = tmp 
            else:
                data[key][order] = tmp  # python indexing

    data['orders'] = list(range(min_ord, max_ord+1))

    return data
def add_datainfo2data(data, datainfo):
    '''
    Adds 'data', 'n_dims' and 'n_points' fields to data.
    Valid for 'empathy', 'eating', 'PTSD' datasets only.
    '''
    print('Adding data info...')
    data_shape = datainfo.shape
    print(f"shape: {data_shape}")
    data['data'] = datainfo
    data['n_dims'] = data_shape[1]
    data['n_points'] = data_shape[0]

def load_dataset(fpath, min_ord, max_ord, n_dims=None, n_points=None):
    '''
    Load dataset (output from O-info analysis, i.e. Otot structure)
    Returns
    -------
    data : dict ('sorted_red', 'index_red', 'bootsig_red', 'sorted_syn', 'index_syn', 'bootsig_syn')
    '''
    rawdata = loadmat(fpath)
    rawdata['goi'] = [todict(x) for x in rawdata['goi']]
    data = rawdata2data(rawdata, min_ord, max_ord)
    if 'data' in rawdata.keys():
        add_datainfo2data(data, rawdata['data'])
   # else:
    #    data['n_points'] = n_points
     #   data['n_dims'] = n_dims
   
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

def _check_keys(dict_in):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict_in:
        if isinstance(dict_in[key], sio.matlab.mio5_params.mat_struct):
            dict_in[key] = todict(dict_in[key])
    return dict_in



#gradient['synappear']=np.zeros(len(data["index_var"][1]))
def rankgradient(info,dataset):
 
 DATASET_DIR = Path.cwd() / 'data'
 SAVE_DIR = Path.cwd() / 'figs'
 # LOAD DATA
 print(f'DATASET: {dataset.upper()}')
 print(8 * '=')
 if dataset=='eating':
         fpath = DATASET_DIR / 'EatingDisorders.mat'
         data = load_dataset(fpath, min_ord=1, max_ord=2)

 elif dataset=='empathy':
        fpath = DATASET_DIR / 'GradientsBriganti.mat'
        data = load_dataset(fpath, min_ord=1, max_ord=2)
 else:
         raise ValueError('Dataset not accepted.') 


 

 gradient = {'redcount':[0]* len(data["index_var"][1]) ,'syncount':[0]* len(data["index_var"][1]),'redappear':[0]* len(data["index_var"][1]),'synappear':[0]* len(data["index_var"][1])}   
 for order in range(1,len(data['orders'])+1):
             a=0
             print(order)
             lst = list(range(1,len(data["index_var"][1])+1))
             all_combos = list(itertools.combinations(lst,order ))
             pltedg=np.zeros((len(all_combos),order))            
             for z in range(0,len(all_combos)): 
                  #a=data['index_var'][1][z]
                  for t in range(0,order):
                   pltedg[z,t]=int(all_combos[z][t])
                   # pltedg =data['index_var_' + stat][order] 
                   
             pltvls = data['O_val'][order]
             t=0 
             #print(pltedg)
             for edge, val in zip(pltedg, pltvls): 
                # print(edge)
                 for z in range(0,len(data["index_var"][order])+1):         
                       [i,j]=pltedg.shape
                      # print(i,j)               
                       for a in range(0,j):          
                               if (z+1)==pltedg[t][a]:
                                  
                                  #print(histograms["redcount"][z])
                                  if 0>pltvls[t]:
                                      print(pltedg[t])
                                      print(pltvls[t])
                                      gradient['syncount'][z]= gradient['syncount'][z]+val
                                      gradient['synappear'][z]= gradient['synappear'][z]+1
                                  if 0<pltvls[t]:
                                      gradient['redcount'][z]= gradient['redcount'][z]+val
                                      gradient['redappear'][z]= gradient['redappear'][z]+1  
                                
                 t=t+1   
 return gradient    
